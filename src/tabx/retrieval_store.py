from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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


class TabxRetrievalStore:
    """Persistent FAISS retrieval with lazy HDF5 context loading."""

    def __init__(
        self,
        index_manifest: Path,
        data_manifest: TabxManifest,
        normalizer: ObservationNormalizer,
        k: int,
        search_k: int,
        device: str,
        online_search_k: int = 16,
        seed: int = 0,
        h5_lru_size: int = 16,
        enable_online_memory: bool = False,
        test_nepisode: int = 100,
        index_load_mode: str = "ram",
        faiss_threads: int = 4,
        trace_search: bool = False,
    ):
        try:
            import faiss
            import h5py
        except ImportError as exc:
            raise RetrievalError(
                "faiss and h5py are required for TABX stage-1 training"
            ) from exc
        self.faiss = faiss
        self.h5py = h5py
        self.index_manifest_path = index_manifest.expanduser().resolve()
        try:
            payload = json.loads(
                self.index_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalError(
                f"Cannot read retrieval manifest {self.index_manifest_path}: {exc}"
            ) from exc
        if payload.get("schema") != "tabx_retrieval_index_v2":
            raise RetrievalError(
                f"Unsupported retrieval schema: {payload.get('schema')!r}"
            )
        if int(payload["embedding_dim"]) <= 0:
            raise RetrievalError("Invalid retrieval embedding dimension")
        if int(payload["sequence_length"]) != data_manifest.sequence_length:
            raise RetrievalError(
                "Retrieval index and data manifest sequence lengths differ"
            )
        self.embedding_dim = int(payload["embedding_dim"])
        self.k = int(k)
        self.search_k = max(int(search_k), self.k)
        self.online_search_k = max(int(online_search_k), self.k)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.enable_online_memory = bool(enable_online_memory)
        self.test_nepisode = max(1, int(test_nepisode))
        self.completed_online_episodes = 0
        self.index_load_mode = str(index_load_mode).strip().lower()
        if self.index_load_mode not in {"ram", "mmap"}:
            raise RetrievalError(
                "index_load_mode must be either 'ram' or 'mmap', got "
                f"{index_load_mode!r}"
            )
        self.faiss_threads = max(1, int(faiss_threads))
        self.trace_search = bool(trace_search)
        self._offline_search_calls = 0
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(self.faiss_threads)
        self.tasks = tuple(data_manifest.tasks)
        self.source = TabxEpisodeSource(
            manifest=data_manifest,
            split="train",
            normalizer=normalizer,
            seed=seed,
            h5_lru_size=h5_lru_size,
        )
        self.buckets: Dict[int, RetrievalBucket] = {}
        base = self.index_manifest_path.parent
        for item in payload.get("buckets", []):
            anchor = int(item["anchor"])
            index_path = base / str(item["index_file"])
            metadata_path = base / str(item["metadata_file"])
            started = time.perf_counter()
            if self.index_load_mode == "ram":
                index = faiss.read_index(str(index_path))
            else:
                mmap_flags = int(getattr(faiss, "IO_FLAG_MMAP", 0)) | int(
                    getattr(faiss, "IO_FLAG_READ_ONLY", 0)
                )
                try:
                    index = faiss.read_index(str(index_path), mmap_flags)
                except TypeError:
                    index = faiss.read_index(str(index_path))
            elapsed = time.perf_counter() - started
            size_gib = index_path.stat().st_size / float(1024 ** 3)
            print(
                "[retrieval-index] "
                f"anchor={anchor} mode={self.index_load_mode} "
                f"size_gib={size_gib:.3f} vectors={int(index.ntotal)} "
                f"load_seconds={elapsed:.3f} threads={self.faiss_threads}",
                flush=True,
            )
            with h5py.File(str(metadata_path), "r") as handle:
                task_index = np.asarray(handle["task_index"][:], dtype=np.int32)
                row = np.asarray(handle["row"][:], dtype=np.int64)
                agent_id = np.asarray(handle["agent_id"][:], dtype=np.int16)
                record_hash = np.asarray(
                    handle["record_hash"][:], dtype=np.uint64
                )
                episode_return = np.asarray(
                    handle["episode_return"][:], dtype=np.float32
                )
            size = task_index.shape[0]
            if not (
                row.shape[0] == size
                and agent_id.shape[0] == size
                and record_hash.shape[0] == size
                and episode_return.shape[0] == size
                and index.ntotal == size
            ):
                raise RetrievalError(
                    f"Index/metadata size mismatch for anchor={anchor}"
                )
            if size:
                self.buckets[anchor] = RetrievalBucket(
                    anchor=anchor,
                    index=index,
                    task_index=task_index,
                    row=row,
                    agent_id=agent_id,
                    record_hash=record_hash,
                    episode_return=episode_return,
                )
        if not self.buckets:
            raise RetrievalError("Retrieval manifest contains no non-empty buckets")
        self._anchors = np.asarray(sorted(self.buckets), dtype=np.int64)
        self._query_task_index: Optional[int] = None
        self._query_rows: Optional[np.ndarray] = None
        self._query_record_hashes: Optional[np.ndarray] = None
        self._query_agent_cursor = 0
        self._online_indexes = {
            int(anchor): faiss.IndexFlatIP(self.embedding_dim)
            for anchor in self._anchors
        }
        self._online_contexts: Dict[int, list] = {
            int(anchor): [] for anchor in self._anchors
        }

    def close(self) -> None:
        self.source.close()

    def begin_query_batch(
        self, task_id: str, rows: Sequence[int]
    ) -> None:
        task = self.source.manifest.task(task_id)
        self._query_task_index = task.index
        self._query_rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        self._query_record_hashes = self.source.record_hashes(task, rows)
        self._query_agent_cursor = 0

    def _nearest_bucket(self, timestep: int) -> RetrievalBucket:
        anchor_index = int(np.argmin(np.abs(self._anchors - int(timestep))))
        return self.buckets[int(self._anchors[anchor_index])]

    def reset_online_memory(self, seed: Optional[int] = None) -> None:
        self.completed_online_episodes = 0
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self._online_indexes = {
            int(anchor): self.faiss.IndexFlatIP(self.embedding_dim)
            for anchor in self._anchors
        }
        self._online_contexts = {int(anchor): [] for anchor in self._anchors}

    @property
    def offline_probability(self) -> float:
        # Original MAICC meaning: epsilon is the probability of offline memory.
        progress = self.completed_online_episodes / self.test_nepisode
        return max(0.01, min(1.0, math.exp(math.log(0.01) * progress)))

    def _online_mips(self, querys, t: int):
        import torch as th

        anchor = self._nearest_bucket(t).anchor
        index = self._online_indexes[anchor]
        if index.ntotal < self.k:
            return None
        queries = np.ascontiguousarray(querys, dtype=np.float32)
        self.faiss.normalize_L2(queries)
        _distance, ids = index.search(queries, min(self.k, index.ntotal))
        contexts = self._online_contexts[anchor]
        selected = []
        for row in ids:
            valid = [int(value) for value in row if int(value) >= 0]
            while len(valid) < self.k:
                valid.append(valid[len(valid) % len(valid)])
            selected.extend(contexts[value] for value in valid[: self.k])
        columns = list(zip(*selected))
        dtypes = (
            th.float32, th.float32, th.long, th.float32,
            th.float32, th.float32, th.float32,
        )
        return tuple(
            th.stack([th.as_tensor(value, dtype=dtype) for value in column]).to(
                self.device
            )
            for column, dtype in zip(columns, dtypes)
        )

    def add_completed_online_episode(
        self,
        agent,
        normalized_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        completion: np.ndarray,
        present: np.ndarray,
        normalized_rtg: np.ndarray,
    ) -> None:
        """Encode one terminal episode without gradients and add its windows."""
        import torch as th

        if not self.enable_online_memory:
            return
        obs = np.asarray(normalized_obs, dtype=np.float32)
        act = np.asarray(actions, dtype=np.int64)
        rew = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        term = np.asarray(terminated, dtype=np.float32).reshape(-1, 1)
        comp = np.asarray(completion, dtype=np.float32).reshape(-1, 1)
        rtg = np.asarray(normalized_rtg, dtype=np.float32).reshape(-1, 1)
        agent_present = np.asarray(present, dtype=bool).reshape(-1)
        if obs.ndim != 3 or act.shape != obs.shape[:2]:
            raise RetrievalError("Invalid completed online episode shapes")
        length, n_agents = obs.shape[:2]
        if not (
            rew.shape[0] == term.shape[0] == comp.shape[0] == rtg.shape[0] == length
        ):
            raise RetrievalError("Online episode fields have different lengths")
        context_length = self.source.manifest.sequence_length
        starts = list(range(0, max(1, length - context_length + 1), context_length))
        tail_start = max(0, length - context_length)
        if tail_start not in starts:
            starts.append(tail_start)

        device = next(agent.parameters()).device
        with th.no_grad():
            for start in sorted(set(starts)):
                end = min(length, start + context_length)
                valid_length = end - start
                pad_obs = np.zeros(
                    (1, context_length, n_agents, obs.shape[-1]), np.float32
                )
                pad_action = np.zeros((1, context_length, n_agents), np.int64)
                pad_post = np.zeros((1, context_length, 3), np.float32)
                pad_reward = np.zeros((context_length, 1), np.float32)
                pad_rtg = np.zeros((1, context_length, 1), np.float32)
                pad_mask = np.zeros((1, context_length, 1), np.float32)
                pad_obs[0, :valid_length] = obs[start:end]
                pad_action[0, :valid_length] = act[start:end]
                pad_post[0, :valid_length] = np.concatenate(
                    [rew[start:end], term[start:end], comp[start:end]], axis=-1
                )
                pad_reward[:valid_length] = rew[start:end]
                pad_rtg[0, :valid_length] = rtg[start:end]
                pad_mask[0, :valid_length] = 1
                tensors = [
                    th.as_tensor(value, device=device)
                    for value in (pad_rtg, pad_obs, pad_action, pad_post)
                ]
                embeddings = encode_dem_prefixes(agent, *tensors).cpu().numpy()
                candidate_anchors = [self._nearest_bucket(valid_length - 1).anchor]
                if start == 0:
                    candidate_anchors.extend(
                        int(anchor) for anchor in self._anchors
                        if int(anchor) < valid_length
                    )
                for anchor in sorted(set(candidate_anchors)):
                    position = min(anchor, valid_length - 1)
                    vectors = []
                    payloads = []
                    for agent_id in range(n_agents):
                        if not agent_present[agent_id]:
                            continue
                        vectors.append(embeddings[0, position, agent_id])
                        payloads.append(
                            (
                                th.as_tensor(pad_obs[0, :, agent_id]),
                                th.as_tensor(pad_reward),
                                th.as_tensor(pad_action[0, :, agent_id, None]),
                                th.as_tensor(pad_post[0, :, 1:2]),
                                th.as_tensor(pad_mask[0]),
                                th.as_tensor(pad_rtg[0]),
                                th.as_tensor(pad_post[0, :, 2:3]),
                            )
                        )
                    if vectors:
                        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
                        self.faiss.normalize_L2(matrix)
                        self._online_indexes[anchor].add(matrix)
                        self._online_contexts[anchor].extend(payloads)
        self.completed_online_episodes += 1

    def _select_candidates(
        self,
        bucket: RetrievalBucket,
        query_embeddings: np.ndarray,
        query_return: Optional[np.ndarray],
    ) -> List[List[int]]:
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if (
            queries.ndim != 2
            or queries.shape[0] <= 0
            or queries.shape[1] != self.embedding_dim
        ):
            raise RetrievalError(
                f"Query shape {queries.shape}, expected [B, {self.embedding_dim}]"
            )
        finite = np.isfinite(queries)
        if not bool(finite.all()):
            invalid = int(queries.size - np.count_nonzero(finite))
            invalid_by_row = np.count_nonzero(~finite, axis=1)
            invalid_rows = np.flatnonzero(invalid_by_row).tolist()
            raise RetrievalError(
                "Non-finite retrieval query before FAISS normalization: "
                f"shape={queries.shape}, invalid_values={invalid}, "
                f"total_values={queries.size}, invalid_rows={invalid_rows}, "
                f"invalid_counts={invalid_by_row[invalid_by_row > 0].tolist()}"
            )
        raw_norms = np.linalg.norm(queries, axis=1)
        call_id = self._offline_search_calls
        self._offline_search_calls += 1
        if self.trace_search:
            print(
                "[retrieval-query] "
                f"call={call_id} anchor={bucket.anchor} shape={queries.shape} "
                f"norm_min={float(raw_norms.min()):.6g} "
                f"norm_max={float(raw_norms.max()):.6g}",
                flush=True,
            )
        self.faiss.normalize_L2(queries)
        # Offline training queries must search deeply enough to remove every
        # window from the same source record. Online queries have no source
        # record to exclude, so starting at 128 candidates only makes HNSW
        # substantially more expensive; diversity filtering can expand the
        # candidate set adaptively when necessary.
        if self._query_record_hashes is None:
            initial_candidates = self.online_search_k
        else:
            initial_candidates = max(self.search_k, 128)
        candidate_count = min(bucket.index.ntotal, initial_candidates)
        max_candidates = min(bucket.index.ntotal, 4096)
        while True:
            if self.trace_search:
                print(
                    "[retrieval-search-start] "
                    f"call={call_id} anchor={bucket.anchor} "
                    f"queries={queries.shape[0]} candidates={candidate_count}",
                    flush=True,
                )
            search_started = time.perf_counter()
            _distance, indices = bucket.index.search(queries, candidate_count)
            search_elapsed = time.perf_counter() - search_started
            selected: List[List[int]] = []
            for query_index, candidates in enumerate(indices):
                pool: List[int] = []
                for candidate in candidates:
                    candidate_id = int(candidate)
                    if candidate_id < 0:
                        continue
                    if (
                        self._query_record_hashes is not None
                        and query_index < self._query_record_hashes.size
                        and bucket.record_hash[candidate_id]
                        == self._query_record_hashes[query_index]
                    ):
                        # Exclude every overlapping window from the same source
                        # record, not only the exact HDF5 row.
                        continue
                    pool.append(candidate_id)

                # Prefer k different source records; relax diversity only if
                # the adaptive search ceiling still cannot provide enough.
                accepted: List[int] = []
                accepted_records = set()
                for candidate_id in pool:
                    source_id = int(bucket.record_hash[candidate_id])
                    if source_id in accepted_records:
                        continue
                    accepted.append(candidate_id)
                    accepted_records.add(source_id)
                    if len(accepted) == self.k:
                        break
                if len(accepted) < self.k and candidate_count >= max_candidates:
                    for candidate_id in pool:
                        if candidate_id not in accepted:
                            accepted.append(candidate_id)
                        if len(accepted) == self.k:
                            break
                selected.append(accepted)
            if self.trace_search or search_elapsed >= 1.0:
                diversity = [len(items) for items in selected]
                print(
                    "[retrieval-search-done] "
                    f"call={call_id} anchor={bucket.anchor} "
                    f"candidates={candidate_count} seconds={search_elapsed:.6f} "
                    f"accepted_min={min(diversity, default=0)} "
                    f"accepted_max={max(diversity, default=0)}",
                    flush=True,
                )
            if all(len(items) >= self.k for items in selected):
                break
            if candidate_count >= max_candidates:
                break
            candidate_count = min(max_candidates, candidate_count * 2)

        for query_index, accepted in enumerate(selected):
            if not accepted:
                raise RetrievalError(
                    "No retrieval candidate remains after source-record exclusion"
                )
            while len(accepted) < self.k:
                accepted.append(accepted[len(accepted) % len(accepted)])
        return selected

    def mips(
        self,
        querys,
        t: int,
        query_return=None,
        test_mode: bool = False,
    ):
        import torch as th

        if test_mode and self.enable_online_memory:
            online = self._online_mips(querys, t)
            if online is not None and self.rng.random() >= self.offline_probability:
                return online
        bucket = self._nearest_bucket(t)
        returns = (
            None
            if query_return is None
            else np.asarray(query_return.detach().cpu(), dtype=np.float32).reshape(-1)
        )
        selected = self._select_candidates(
            bucket,
            np.asarray(querys, dtype=np.float32),
            returns,
        )
        observations: List[th.Tensor] = []
        rewards: List[th.Tensor] = []
        actions: List[th.Tensor] = []
        terminated: List[th.Tensor] = []
        masks: List[th.Tensor] = []
        rtgs: List[th.Tensor] = []
        completions: List[th.Tensor] = []
        for candidates in selected:
            for candidate_id in candidates:
                task_index = int(bucket.task_index[candidate_id])
                if not 0 <= task_index < len(self.tasks):
                    raise RetrievalError(
                        f"Invalid task_index={task_index} in retrieval metadata"
                    )
                task = self.tasks[task_index]
                context = self.source.load_agent_context(
                    task=task,
                    row=int(bucket.row[candidate_id]),
                    agent_id=int(bucket.agent_id[candidate_id]),
                )
                obs, reward, action, term, filled, rtg, completion = context
                observations.append(th.as_tensor(obs, dtype=th.float32))
                rewards.append(th.as_tensor(reward, dtype=th.float32))
                actions.append(th.as_tensor(action, dtype=th.long))
                terminated.append(th.as_tensor(term, dtype=th.float32))
                masks.append(th.as_tensor(filled, dtype=th.float32))
                rtgs.append(th.as_tensor(rtg, dtype=th.float32))
                completions.append(
                    th.as_tensor(completion, dtype=th.float32)
                )
        self._query_agent_cursor += 1
        return (
            th.stack(observations).to(self.device),
            th.stack(rewards).to(self.device),
            th.stack(actions).to(self.device),
            th.stack(terminated).to(self.device),
            th.stack(masks).to(self.device),
            th.stack(rtgs).to(self.device),
            th.stack(completions).to(self.device),
        )


def encode_dem_prefixes(
    agent,
    seq_rtg,
    seq_obs,
    seq_action,
    seq_post,
):
    """Encode every timestep without rerunning DEM for every prefix."""
    import torch as th

    batch_size, seq_len, n_agents = seq_obs.shape[:3]
    embeddings: List[th.Tensor] = []
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
    return th.stack(embeddings, dim=2)
