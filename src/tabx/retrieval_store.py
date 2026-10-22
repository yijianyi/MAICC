from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .episode_adapter import TabxEpisodeSource
from .manifest import TabxManifest
from .normalization import ObservationNormalizer


class RetrievalError(RuntimeError):
    pass


@dataclass
class RetrievalBucket:
    anchor: int
    index: object
    task_index: np.ndarray
    row: np.ndarray
    agent_id: np.ndarray
    record_hash: np.ndarray
    episode_return: np.ndarray
    individual_return: np.ndarray


def normalized_utility(values, bounds):
    """Original affine return normalization; fixed bounds, no clipping."""
    values = np.asarray(values, dtype=np.float32)
    lower, upper = map(float, bounds)
    if upper <= lower:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def offline_scores(similarity, team_return, bounds, weight=0.8):
    return (np.asarray(similarity) + 1.0) / 2.0 + weight * normalized_utility(team_return, bounds)


def online_scores(similarity, team_return, individual_return, n_agents,
                  bounds, weight=0.8, ind_weight=0.2):
    return (offline_scores(similarity, team_return, bounds, weight)
            + ind_weight * normalized_utility(np.asarray(individual_return) * n_agents, bounds))


def training_scores(similarity, candidate_return, query_return, max_similarity):
    similarity = np.asarray(similarity)
    bonus = (similarity < max_similarity) & (np.asarray(candidate_return) <= query_return)
    return (similarity + 1.0) / 2.0 + bonus.astype(np.float32)


class TabxRetrievalStore:
    """Exact-prefix retrieval and agent-local online memory.

    Offline indexes are loaded on demand. The LRU bounds memory even when all
    history lengths are indexed; it does not change the candidate set.
    """

    def __init__(self, index_manifest: Path, data_manifest: TabxManifest,
                 normalizer: ObservationNormalizer, k: int, search_k: int,
                 device: str, seed: int = 0, h5_lru_size: int = 16,
                 enable_online_memory: bool = False, test_nepisode: int = 100,
                 index_load_mode: str = "ram", faiss_threads: int = 4,
                 trace_search: bool = False, use_utility: bool = True,
                 weight: float = 0.8, ind_weight: float = 0.2,
                 max_similarity: float = 0.98, index_cache_size: int = 2,
                 allow_approximate_index: bool = False,
                 expected_encoder_sha256: Optional[str] = None,
                 normalization_path: Optional[Path] = None):
        import faiss
        import h5py
        from .provenance import file_sha256

        self.faiss, self.h5py = faiss, h5py
        self.index_manifest_path = Path(index_manifest).expanduser().resolve()
        payload = json.loads(self.index_manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "tabx_retrieval_index_v4":
            raise RetrievalError("Rebuild the index with schema v4; v3 retrieval semantics are incompatible")
        if payload.get("prefix_policy") != "exact_all_windows":
            raise RetrievalError("Index must contain exact prefixes from all eligible windows")
        if payload.get("index_factory") != "Flat" and not allow_approximate_index:
            raise RetrievalError("Approximate index requires explicit allow_approximate_index=true")
        if payload.get("data_manifest_sha256") != file_sha256(data_manifest.path):
            raise RetrievalError("Index data manifest hash mismatch; rebuild the index")
        if normalization_path is not None and payload.get("normalization_sha256") != file_sha256(normalization_path):
            raise RetrievalError("Index normalization hash mismatch; rebuild the index")
        if expected_encoder_sha256 is not None and payload.get("encoder_sha256") != expected_encoder_sha256:
            raise RetrievalError("Index encoder hash mismatch; rebuild the index")
        self.embedding_dim = int(payload["embedding_dim"])
        self.context_length = int(payload["sequence_length"])
        if self.embedding_dim <= 0 or self.context_length != data_manifest.sequence_length:
            raise RetrievalError("Invalid index dimensions or context length")
        self.k = int(k)
        if self.k < 1:
            raise RetrievalError("k must be positive")
        self.search_k = max(int(search_k), self.k)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.enable_online_memory = bool(enable_online_memory)
        self.test_nepisode = max(1, int(test_nepisode))
        self.weight, self.ind_weight = float(weight), float(ind_weight)
        self.max_similarity = float(max_similarity)
        self.use_utility = bool(use_utility)
        self.return_bounds = tuple(map(float, payload["utility_bounds"]["team"]))
        if len(self.return_bounds) != 2 or not np.isfinite(self.return_bounds).all() or self.return_bounds[0] > self.return_bounds[1]:
            raise RetrievalError("Invalid fixed training-return bounds")
        self.index_load_mode = str(index_load_mode)
        if self.index_load_mode not in {"ram", "mmap"}:
            raise RetrievalError("index_load_mode must be ram or mmap")
        self.index_cache_size = max(1, int(index_cache_size))
        self.trace_search = bool(trace_search)
        faiss.omp_set_num_threads(max(1, int(faiss_threads)))
        self.tasks = tuple(data_manifest.tasks)
        self.source = TabxEpisodeSource(data_manifest, split="train", normalizer=normalizer,
                                       seed=seed, h5_lru_size=h5_lru_size)
        self._bucket_specs = {int(item["anchor"]): item for item in payload["buckets"]}
        if set(self._bucket_specs) != set(range(self.context_length)):
            raise RetrievalError("Index must cover every prefix length; rebuild all buckets")
        if any(int(item["vectors"]) < self.k for item in self._bucket_specs.values()):
            raise RetrievalError("Every prefix must have at least k offline candidates")
        self.buckets = OrderedDict()
        self._row_return_specs = payload["row_returns"]
        self._row_returns = {}
        self._query_record_hashes = None
        self._query_episode_returns = None
        self.reset_online_memory(seed)

    def close(self):
        self.buckets.clear()
        self._row_returns.clear()
        self.source.close()

    def _bucket(self, timestep):
        anchor = int(timestep)
        if not 0 <= anchor < self.context_length:
            raise RetrievalError(f"Query prefix {anchor + 1} exceeds history length {self.context_length}")
        if anchor in self.buckets:
            self.buckets.move_to_end(anchor)
            return self.buckets[anchor]
        # Evict BEFORE loading, so peak resident indexes also obey the limit.
        while len(self.buckets) >= self.index_cache_size:
            self.buckets.popitem(last=False)
        item = self._bucket_specs[anchor]
        base = self.index_manifest_path.parent
        flags = 0 if self.index_load_mode == "ram" else (
            int(getattr(self.faiss, "IO_FLAG_MMAP", 0)) | int(getattr(self.faiss, "IO_FLAG_READ_ONLY", 0)))
        started = time.perf_counter()
        index = self.faiss.read_index(str(base / item["index_file"]), flags)
        with self.h5py.File(str(base / item["metadata_file"]), "r") as handle:
            columns = {name: np.asarray(handle[name][:]) for name in (
                "task_index", "row", "agent_id", "record_hash", "episode_return", "individual_return")}
        if index.d != self.embedding_dim or index.ntotal != int(item["vectors"]):
            raise RetrievalError(f"Invalid index dimensions for prefix {anchor + 1}")
        if any(value.shape != (index.ntotal,) for value in columns.values()):
            raise RetrievalError(f"Index/metadata mismatch for prefix {anchor + 1}")
        if not np.isfinite(columns["episode_return"]).all():
            raise RetrievalError("Non-finite episode returns")
        for task_index in np.unique(columns["task_index"]):
            if not 0 <= task_index < len(self.tasks):
                raise RetrievalError("Index contains an unknown task")
            task = self.tasks[int(task_index)]
            if task.split != "train":
                raise RetrievalError("Offline retrieval may only contain training tasks")
            selected = columns["task_index"] == task_index
            rows, agents = columns["row"][selected], columns["agent_id"][selected]
            if np.any((rows < 0) | (rows >= task.n_trajectories)) or np.any(
                (agents < 0) | (agents >= task.n_agents)
            ):
                raise RetrievalError("Index contains an invalid trajectory or agent ID")
        bucket = RetrievalBucket(anchor=anchor, index=index, **columns)
        self.buckets[anchor] = bucket
        if self.trace_search:
            print(f"[retrieval-load] prefix={anchor + 1} vectors={index.ntotal} seconds={time.perf_counter()-started:.3f}", flush=True)
        return bucket

    def begin_query_batch(self, task_id: str, rows: Sequence[int]):
        task = self.source.manifest.task(task_id)
        if task.split != "train":
            raise RetrievalError("Training queries must belong to the training split")
        if task_id not in self._row_returns:
            from .provenance import file_sha256
            spec = self._row_return_specs[task_id]
            path = self.index_manifest_path.parent / spec["file"]
            if file_sha256(path) != spec["sha256"]:
                raise RetrievalError("Query episode-return metadata hash mismatch")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (task.n_trajectories,) or not np.isfinite(values).all():
                raise RetrievalError("Invalid full-episode query returns")
            self._row_returns[task_id] = values
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or not rows.size or np.any((rows < 0) | (rows >= task.n_trajectories)):
            raise RetrievalError("Invalid training query rows")
        self._query_record_hashes = self.source.record_hashes(task, rows)
        self._query_episode_returns = np.asarray(self._row_returns[task_id][rows], dtype=np.float32)

    def reset_online_memory(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.completed_online_episodes = 0
        self._online_indexes = {}
        self._online_contexts = {}
        self._online_returns = {}
        self._query_record_hashes = None
        self._query_episode_returns = None

    @property
    def offline_probability(self):
        if self.completed_online_episodes < self.k:
            return 1.0
        progress = self.completed_online_episodes / self.test_nepisode
        return max(0.01, min(1.0, math.exp(math.log(0.01) * progress)))

    def _online_mips(self, query, t, agent_id):
        key = (int(t), int(agent_id))
        index = self._online_indexes.get(key)
        if self.completed_online_episodes < self.k or index is None:
            return None
        metadata = self._online_returns[key]
        if len({item[3] for item in metadata}) < self.k:
            return None
        # Each source episode contributes its most similar window. Original
        # online retrieval chose k different episodes of this SAME agent.
        count = min(index.ntotal, self.search_k)
        while True:
            distance, ids = index.search(query, count)
            candidates, seen = [], set()
            for similarity, candidate in zip(distance[0], ids[0]):
                if candidate < 0:
                    continue
                candidate = int(candidate)
                episode_id = metadata[candidate][3]
                if episode_id not in seen:
                    seen.add(episode_id)
                    candidates.append((candidate, float(similarity)))
                if len(candidates) >= self.search_k:
                    break
            if len(candidates) >= self.search_k or count == index.ntotal:
                break
            count = min(index.ntotal, count * 2)
        if len(candidates) < self.k:
            return None
        scores = []
        for candidate, similarity in candidates:
            team, individual, n_present, _episode = metadata[candidate]
            scores.append(online_scores(similarity, team, individual, n_present,
                self.return_bounds, self.weight if self.use_utility else 0.0,
                self.ind_weight if self.use_utility else 0.0))
        order = np.argsort(-np.asarray(scores), kind="stable")[:self.k]
        return [self._online_contexts[key][candidates[i][0]] for i in order]

    def add_completed_online_episode(self, agent, normalized_obs, actions, rewards,
                                     terminated, completion, present, normalized_rtg):
        import torch as th
        if not self.enable_online_memory:
            return
        obs = np.asarray(normalized_obs, dtype=np.float32)
        act = np.asarray(actions, dtype=np.int64)
        rew = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        term = np.asarray(terminated, dtype=np.float32).reshape(-1, 1)
        comp = np.asarray(completion, dtype=np.float32).reshape(-1, 1)
        rtg = np.asarray(normalized_rtg, dtype=np.float32).reshape(-1, 1)
        present = np.asarray(present, dtype=bool).reshape(-1)
        if obs.ndim != 3 or act.shape != obs.shape[:2] or present.size != obs.shape[1]:
            raise RetrievalError("Invalid completed episode shapes")
        length, n_agents = obs.shape[:2]
        if length < 1 or any(len(x) != length for x in (rew, term, comp, rtg)):
            raise RetrievalError("Invalid completed episode length")
        context_length = self.context_length
        starts = set(range(0, max(1, length-context_length+1), context_length))
        starts.add(max(0, length-context_length))
        device = next(agent.parameters()).device
        pending, covered_until = [], 0
        individual_return = np.zeros(n_agents, dtype=np.float64)
        with th.no_grad():
            for start in sorted(starts):
                end = min(length, start + context_length)
                valid_length = end - start
                pad_obs = np.zeros((1, context_length, n_agents, obs.shape[-1]), np.float32)
                pad_action = np.full((1, context_length, n_agents), 7, np.int64)
                pad_post = np.zeros((1, context_length, 3), np.float32)
                pad_rtg = np.zeros((1, context_length, 1), np.float32)
                pad_mask = np.zeros((context_length, 1), np.float32)
                pad_obs[0, :valid_length] = obs[start:end]
                pad_action[0, :valid_length] = act[start:end]
                pad_post[0, :valid_length] = np.concatenate([rew[start:end], term[start:end], comp[start:end]], -1)
                pad_rtg[0, :valid_length] = rtg[start:end]
                pad_mask[:valid_length] = 1
                tensors = [th.as_tensor(x, device=device) for x in (pad_rtg, pad_obs, pad_action, pad_post)]
                embeddings, predicted = encode_dem_prefixes(agent, *tensors, return_rewards=True)
                embeddings = embeddings.cpu().numpy()
                first = max(covered_until, start) - start
                individual_return += predicted[0, first:valid_length].cpu().numpy().sum(0)
                covered_until = max(covered_until, end)
                for agent_id in np.flatnonzero(present):
                    context = (pad_obs[0, :, agent_id], pad_post[0, :, :1],
                               pad_action[0, :, agent_id, None], pad_post[0, :, 1:2],
                               pad_mask, pad_rtg[0], pad_post[0, :, 2:3])
                    # Store one shared payload per window/agent, not a copy per prefix.
                    payload = tuple(th.as_tensor(x.copy()) for x in context)
                    for anchor in range(valid_length):
                        vector = np.ascontiguousarray(embeddings[0, anchor, agent_id:agent_id+1])
                        if not np.isfinite(vector).all():
                            raise RetrievalError("Non-finite online key")
                        self.faiss.normalize_L2(vector)
                        pending.append(((anchor, int(agent_id)), vector, payload))
        team_return = float(rew.sum(dtype=np.float64))
        if not np.isfinite(team_return) or not np.isfinite(individual_return).all():
            raise RetrievalError("Non-finite online utility")
        for key, vector, payload in pending:
            if key not in self._online_indexes:
                self._online_indexes[key] = self.faiss.IndexFlatIP(self.embedding_dim)
                self._online_contexts[key], self._online_returns[key] = [], []
            self._online_indexes[key].add(vector)
            self._online_contexts[key].append(payload)
            self._online_returns[key].append((team_return, float(individual_return[key[1]]),
                                             int(present.sum()), self.completed_online_episodes))
        self.completed_online_episodes += 1

    def _select_candidates(self, bucket, queries, query_return, test_mode=False,
                           query_record_hashes=None):
        count = min(bucket.index.ntotal, max(self.search_k, self.k))
        while True:
            distance, indices = bucket.index.search(queries, count)
            pools = []
            for q, row in enumerate(indices):
                candidates = [(int(i), float(d)) for i, d in zip(row, distance[q]) if i >= 0
                    and (query_record_hashes is None or bucket.record_hash[int(i)] != query_record_hashes[q])]
                pools.append(candidates[:self.search_k])
            if all(len(pool) >= self.search_k for pool in pools) or count == bucket.index.ntotal:
                break
            count = min(bucket.index.ntotal, count * 2)
        selected = []
        for q, pool in enumerate(pools):
            if len(pool) < self.k:
                raise RetrievalError("Fewer than k candidates remain after source-episode exclusion")
            ids = np.asarray([item[0] for item in pool], dtype=np.int64)
            similarity = np.asarray([item[1] for item in pool])
            if test_mode:
                scores = offline_scores(similarity, bucket.episode_return[ids], self.return_bounds,
                                        self.weight if self.use_utility else 0.0)
            else:
                if query_return is None:
                    raise RetrievalError("Training retrieval requires complete source-episode returns")
                scores = training_scores(similarity, bucket.episode_return[ids],
                                         query_return[q], self.max_similarity)
            chosen = ids[np.argsort(-scores, kind="stable")[:self.k]]
            if not test_mode:
                chosen = self.rng.permutation(chosen)
            selected.append(chosen.tolist())
        return selected

    def mips(self, querys, t, query_return=None, test_mode=False,
             agent_ids=None, query_indices=None):
        import torch as th
        queries = np.array(querys, dtype=np.float32, order="C", copy=True)
        if queries.ndim != 2 or queries.shape[1] != self.embedding_dim or not np.isfinite(queries).all():
            raise RetrievalError("Invalid retrieval query")
        if not 0 <= int(t) < self.context_length:
            raise RetrievalError("Pass the actual bounded query prefix length, not episode time")
        self.faiss.normalize_L2(queries)
        size = len(queries)
        if size == 0:
            raise RetrievalError("Empty retrieval query")
        if test_mode:
            if agent_ids is None or len(agent_ids) != size:
                raise RetrievalError("Online retrieval requires an explicit local agent ID per query")
            hashes = returns = None
        else:
            if self._query_episode_returns is None:
                raise RetrievalError("Call begin_query_batch before training retrieval")
            positions = np.arange(len(self._query_episode_returns)) if query_indices is None else np.asarray(query_indices, dtype=np.int64)
            if positions.shape != (size,) or np.any(
                (positions < 0) | (positions >= len(self._query_episode_returns))
            ):
                raise RetrievalError("Training query rows do not match the sampled batch")
            hashes = self._query_record_hashes[positions]
            returns = self._query_episode_returns[positions]
            if query_return is not None:
                supplied = query_return.detach().cpu().numpy() if hasattr(query_return, "detach") else query_return
                if not np.allclose(np.asarray(supplied).reshape(-1), returns):
                    raise RetrievalError("query_return must equal complete episode return, not window return")
        contexts = [None] * size
        if test_mode and self.enable_online_memory:
            for q, agent_id in enumerate(agent_ids):
                if self.rng.random() >= self.offline_probability:
                    contexts[q] = self._online_mips(queries[q:q+1], t, int(agent_id))
        offline_rows = [q for q, item in enumerate(contexts) if item is None]
        if offline_rows:
            bucket = self._bucket(t)
            chosen = self._select_candidates(bucket, queries[offline_rows],
                None if returns is None else returns[offline_rows], test_mode,
                None if hashes is None else hashes[offline_rows])
            for q, candidates in zip(offline_rows, chosen):
                contexts[q] = [self.source.load_agent_context(
                    self.tasks[int(bucket.task_index[i])], int(bucket.row[i]), int(bucket.agent_id[i]))
                    for i in candidates]
        flat = [context for row in contexts for context in row]
        dtypes = (th.float32, th.float32, th.long, th.float32, th.float32, th.float32, th.float32)
        return tuple(th.stack([th.as_tensor(value, dtype=dtype) for value in column]).to(self.device)
                     for column, dtype in zip(zip(*flat), dtypes))

def encode_dem_prefixes(
    agent,
    seq_rtg,
    seq_obs,
    seq_action,
    seq_post,
    return_rewards=False,
):
    """Encode every timestep without rerunning DEM for every prefix."""
    import torch as th

    batch_size, seq_len, n_agents = seq_obs.shape[:3]
    embeddings: List[th.Tensor] = []
    predicted_rewards = []
    for agent_id in range(n_agents):
        items = []
        if agent.args.use_rtg:
            pre = agent.EM_pre_embd(seq_rtg)
        obs = agent.EM_obs_embd(seq_obs[:, :, agent_id])
        action = agent.EM_action_embd(seq_action[:, :, agent_id])
        post = agent.EM_post_embd(seq_post)
        if agent.args.use_rtg:
            inputs = th.stack([pre, obs, action, post], dim=2).reshape(
                batch_size, seq_len * 4, agent.n_embd
            )
            outputs = agent.DEM(inputs_embeds=inputs)["last_hidden_state"]
            z_o = outputs[:, 1::4]
            z_a = outputs[:, 2::4]
            z_context = outputs[:, 0::4]
        else:
            inputs = th.stack([obs, action, post], dim=2).reshape(
                batch_size, seq_len * 3, agent.n_embd
            )
            outputs = agent.DEM(inputs_embeds=inputs)["last_hidden_state"]
            z_o = outputs[:, 0::3]
            z_a = outputs[:, 1::3]
            z_context = outputs[:, 2::3]
        result = z_o + z_context if agent.args.use_rtg else z_o.clone()
        if agent.args.train_CEM and seq_len > 1:
            if agent.args.use_rtg:
                result[:, 1:] = result[:, 1:] + z_a[:, :-1]
            else:
                result[:, 1:] = (
                    result[:, 1:] + z_context[:, :-1] + z_a[:, :-1]
                )
        embeddings.append(result)
        if return_rewards:
            predicted_rewards.append(agent.EM_pred_reward(z_a).squeeze(-1))
    keys = th.stack(embeddings, dim=2)
    return (keys, th.stack(predicted_rewards, dim=2)) if return_rewards else keys
