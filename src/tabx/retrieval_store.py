from __future__ import annotations

import json
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
        seed: int = 0,
        h5_lru_size: int = 16,
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
        if payload.get("schema") != "tabx_retrieval_index_v1":
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
        self.device = device
        self.rng = np.random.default_rng(seed)
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
            mmap_flags = int(getattr(faiss, "IO_FLAG_MMAP", 0)) | int(
                getattr(faiss, "IO_FLAG_READ_ONLY", 0)
            )
            try:
                index = faiss.read_index(str(index_path), mmap_flags)
            except TypeError:
                index = faiss.read_index(str(index_path))
            with h5py.File(str(metadata_path), "r") as handle:
                task_index = np.asarray(handle["task_index"][:], dtype=np.int32)
                row = np.asarray(handle["row"][:], dtype=np.int64)
                agent_id = np.asarray(handle["agent_id"][:], dtype=np.int16)
                episode_return = np.asarray(
                    handle["episode_return"][:], dtype=np.float32
                )
            size = task_index.shape[0]
            if not (
                row.shape[0] == size
                and agent_id.shape[0] == size
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
                    episode_return=episode_return,
                )
        if not self.buckets:
            raise RetrievalError("Retrieval manifest contains no non-empty buckets")
        self._anchors = np.asarray(sorted(self.buckets), dtype=np.int64)
        self._query_task_index: Optional[int] = None
        self._query_rows: Optional[np.ndarray] = None
        self._query_agent_cursor = 0

    def close(self) -> None:
        self.source.close()

    def begin_query_batch(
        self, task_id: str, rows: Sequence[int]
    ) -> None:
        task = self.source.manifest.task(task_id)
        self._query_task_index = task.index
        self._query_rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        self._query_agent_cursor = 0

    def _nearest_bucket(self, timestep: int) -> RetrievalBucket:
        anchor_index = int(np.argmin(np.abs(self._anchors - int(timestep))))
        return self.buckets[int(self._anchors[anchor_index])]

    def _select_candidates(
        self,
        bucket: RetrievalBucket,
        query_embeddings: np.ndarray,
        query_return: Optional[np.ndarray],
    ) -> List[List[int]]:
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.embedding_dim:
            raise RetrievalError(
                f"Query shape {queries.shape}, expected [B, {self.embedding_dim}]"
            )
        self.faiss.normalize_L2(queries)
        candidate_count = min(
            bucket.index.ntotal,
            max(self.search_k, self.k + 16),
        )
        _distance, indices = bucket.index.search(queries, candidate_count)
        selected: List[List[int]] = []
        for query_index, candidates in enumerate(indices):
            accepted: List[int] = []
            for candidate in candidates:
                candidate_id = int(candidate)
                if candidate_id < 0:
                    continue
                if (
                    self._query_task_index is not None
                    and self._query_rows is not None
                    and query_index < self._query_rows.size
                    and int(bucket.task_index[candidate_id])
                    == self._query_task_index
                    and int(bucket.row[candidate_id])
                    == int(self._query_rows[query_index])
                ):
                    continue
                accepted.append(candidate_id)
                if len(accepted) == self.k:
                    break
            if not accepted:
                raise RetrievalError(
                    "No retrieval candidate remains after self-exclusion"
                )
            while len(accepted) < self.k:
                accepted.append(accepted[len(accepted) % len(accepted)])
            selected.append(accepted)
        return selected

    def mips(
        self,
        querys,
        t: int,
        query_return=None,
        test_mode: bool = False,
    ):
        import torch as th

        if test_mode:
            raise RetrievalError(
                "TabxRetrievalStore online test_mode is handled by eval_tabx.py"
            )
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
                obs, reward, action, term, filled = context
                observations.append(th.as_tensor(obs, dtype=th.float32))
                rewards.append(th.as_tensor(reward, dtype=th.float32))
                actions.append(th.as_tensor(action, dtype=th.long))
                terminated.append(th.as_tensor(term, dtype=th.float32))
                masks.append(th.as_tensor(filled, dtype=th.float32))
        self._query_agent_cursor += 1
        return (
            th.stack(observations).to(self.device),
            th.stack(rewards).to(self.device),
            th.stack(actions).to(self.device),
            th.stack(terminated).to(self.device),
            th.stack(masks).to(self.device),
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
