from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .manifest import TabxManifest, TabxTask
from .normalization import ObservationNormalizer


class TabxDataError(RuntimeError):
    pass


@dataclass
class SampledBatch:
    batch: object
    task: TabxTask
    rows: np.ndarray


class TabxEpisodeSource:
    REQUIRED_FIELDS = (
        "obs", "actions", "reward", "rtg", "completion", "terminated",
        "truncation", "present", "filled",
    )

    def __init__(
        self,
        manifest: TabxManifest,
        split: str = "train",
        normalizer: Optional[ObservationNormalizer] = None,
        seed: int = 0,
        h5_lru_size: int = 16,
    ):
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise TabxDataError(
                "h5py is required for TABX training. Install requirements.txt."
            ) from exc
        self.manifest = manifest
        self.tasks = tuple(
            task
            for task in manifest.tasks_for_split(split)
            if task.n_trajectories > 0
        )
        if not self.tasks:
            raise TabxDataError(f"No non-empty TABX tasks for split={split!r}")
        manifest.validate_files(self.tasks)
        self.normalizer = (
            ObservationNormalizer.identity(manifest.obs_dim)
            if normalizer is None
            else normalizer
        )
        self.rng = np.random.default_rng(seed)
        self.h5_lru_size = max(1, int(h5_lru_size))
        self._handles: "OrderedDict[Path, object]" = OrderedDict()
        self._record_hashes: Dict[str, np.ndarray] = {}
        self._segment_start_rows: Dict[str, np.ndarray] = {}
        self._task_probabilities = np.full(
            len(self.tasks), 1.0 / len(self.tasks), dtype=np.float64
        )

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()

    def __enter__(self) -> "TabxEpisodeSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _open(self, task: TabxTask):
        import h5py

        handle = self._handles.pop(task.h5_path, None)
        if handle is None:
            handle = h5py.File(str(task.h5_path), "r")
            missing = [
                key for key in self.REQUIRED_FIELDS if key not in handle
            ]
            if missing:
                handle.close()
                raise TabxDataError(
                    f"{task.h5_path} is missing datasets: {missing}"
                )
        self._handles[task.h5_path] = handle
        while len(self._handles) > self.h5_lru_size:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def choose_task(self) -> TabxTask:
        index = int(
            self.rng.choice(len(self.tasks), p=self._task_probabilities)
        )
        return self.tasks[index]

    def batch_size_for_task(
        self,
        task: TabxTask,
        max_batch_size: int,
        agent_time_budget: int,
    ) -> int:
        if agent_time_budget <= 0:
            return max(1, int(max_batch_size))
        per_episode = task.n_agents * task.sequence_length
        budget_size = max(1, int(agent_time_budget) // max(1, per_episode))
        return max(1, min(int(max_batch_size), budget_size))

    def sample(
        self,
        max_batch_size: int,
        agent_time_budget: int = 0,
        task: Optional[TabxTask] = None,
    ) -> SampledBatch:
        selected_task = self.choose_task() if task is None else task
        batch_size = self.batch_size_for_task(
            selected_task, max_batch_size, agent_time_budget
        )
        replace = selected_task.n_trajectories < batch_size
        rows = np.asarray(
            self.rng.choice(
                selected_task.n_trajectories,
                size=batch_size,
                replace=replace,
            ),
            dtype=np.int64,
        )
        return self.load_rows(selected_task, rows)

    def load_rows(
        self, task: TabxTask, rows: Sequence[int]
    ) -> SampledBatch:
        import torch as th
        from components.episode_buffer import EpisodeBatch

        row_array = np.asarray(rows, dtype=np.int64).reshape(-1)
        if row_array.size == 0:
            raise TabxDataError("Cannot load an empty TABX batch")
        if row_array.min() < 0 or row_array.max() >= task.n_trajectories:
            raise TabxDataError(
                f"Rows outside task {task.task_id} trajectory range: "
                f"{row_array.tolist()}"
            )

        handle = self._open(task)
        arrays: Dict[str, np.ndarray] = {}
        # h5py fancy indexing requires sorted unique indices. Row-wise reads
        # support sampling with replacement and keep the requested order.
        for key in self.REQUIRED_FIELDS:
            arrays[key] = np.stack(
                [np.asarray(handle[key][int(row)]) for row in row_array],
                axis=0,
            )

        expected_time = task.sequence_length + 1
        expected = {
            "obs": (
                row_array.size,
                expected_time,
                task.n_agents,
                task.obs_dim,
            ),
            "actions": (
                row_array.size,
                expected_time,
                task.n_agents,
                1,
            ),
            "reward": (row_array.size, expected_time, 1),
            "rtg": (row_array.size, expected_time, 1),
            "completion": (row_array.size, expected_time, 1),
            "terminated": (row_array.size, expected_time, 1),
            "truncation": (row_array.size, expected_time, 1),
            "present": (
                row_array.size, expected_time, task.n_agents, 1
            ),
            "filled": (row_array.size, expected_time, 1),
        }
        for key, shape in expected.items():
            if arrays[key].shape != shape:
                raise TabxDataError(
                    f"Unexpected {key} shape in {task.h5_path}: "
                    f"{arrays[key].shape}, expected {shape}"
                )
        if not np.isfinite(arrays["obs"]).all():
            raise TabxDataError(f"NaN/Inf observation in {task.h5_path}")
        if not np.isfinite(arrays["reward"]).all():
            raise TabxDataError(f"NaN/Inf reward in {task.h5_path}")
        if arrays["actions"].min() < 0 or arrays["actions"].max() >= task.action_dim:
            raise TabxDataError(f"Action outside schema in {task.h5_path}")

        arrays["obs"] = self.normalizer.apply(arrays["obs"])
        arrays["rtg"] = arrays["rtg"].astype(np.float32) / task.return_scale
        scheme = {
            "obs": {
                "vshape": task.obs_dim,
                "group": "agents",
                "dtype": th.float32,
            },
            "actions": {
                "vshape": (1,),
                "group": "agents",
                "dtype": th.long,
            },
            "reward": {"vshape": (1,), "dtype": th.float32},
            "rtg": {"vshape": (1,), "dtype": th.float32},
            "completion": {"vshape": (1,), "dtype": th.uint8},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "truncation": {"vshape": (1,), "dtype": th.uint8},
            "present": {
                "vshape": (1,), "group": "agents", "dtype": th.uint8
            },
        }
        batch = EpisodeBatch(
            scheme=scheme,
            groups={"agents": task.n_agents},
            batch_size=int(row_array.size),
            max_seq_length=expected_time,
            device="cpu",
        )
        batch.update(
            {
                "obs": th.as_tensor(arrays["obs"], dtype=th.float32),
                "actions": th.as_tensor(arrays["actions"], dtype=th.long),
                "reward": th.as_tensor(arrays["reward"], dtype=th.float32),
                "rtg": th.as_tensor(arrays["rtg"], dtype=th.float32),
                "completion": th.as_tensor(
                    arrays["completion"], dtype=th.uint8
                ),
                "terminated": th.as_tensor(
                    arrays["terminated"], dtype=th.uint8
                ),
                "truncation": th.as_tensor(
                    arrays["truncation"], dtype=th.uint8
                ),
                "present": th.as_tensor(arrays["present"], dtype=th.uint8),
                "filled": th.as_tensor(arrays["filled"], dtype=th.long),
            },
            mark_filled=False,
        )
        return SampledBatch(batch=batch, task=task, rows=row_array)

    def load_agent_context(
        self, task: TabxTask, row: int, agent_id: int
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]:
        if not 0 <= row < task.n_trajectories:
            raise TabxDataError(
                f"row={row} outside task {task.task_id} trajectory range"
            )
        if not 0 <= agent_id < task.n_agents:
            raise TabxDataError(
                f"agent_id={agent_id} outside task {task.task_id} agent range"
            )
        handle = self._open(task)
        length = task.sequence_length
        obs = self.normalizer.apply(
            np.asarray(handle["obs"][row, :length, agent_id], dtype=np.float32)
        )
        reward = np.asarray(
            handle["reward"][row, :length], dtype=np.float32
        )
        rtg = np.asarray(handle["rtg"][row, :length], dtype=np.float32)
        rtg = rtg / task.return_scale
        completion = np.asarray(
            handle["completion"][row, :length], dtype=np.float32
        )
        action = np.asarray(
            handle["actions"][row, :length, agent_id], dtype=np.int64
        )
        terminated = np.asarray(
            handle["terminated"][row, :length], dtype=np.float32
        )
        filled = np.asarray(
            handle["filled"][row, :length], dtype=np.float32
        ).copy()
        filled[1:] *= 1.0 - terminated[:-1]
        return obs, reward, action, terminated, filled, rtg, completion

    def record_hashes(self, task: TabxTask, rows: Sequence[int]) -> np.ndarray:
        """Return stable source-episode IDs, shared by all overlapping windows."""
        cached = self._record_hashes.get(task.task_id)
        if cached is None:
            if task.index_path is None or not task.index_path.is_file():
                raise TabxDataError(
                    f"Missing provenance index for task {task.task_id}"
                )
            cached = np.zeros(task.n_trajectories, dtype=np.uint64)
            seen = np.zeros(task.n_trajectories, dtype=bool)
            segment_starts = np.zeros(task.n_trajectories, dtype=bool)
            with task.index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    row = int(item["row"])
                    token = json.dumps([str(item["source_record"]),
                                        int(item["source_segment_start"]),
                                        int(item["source_segment_end"])],
                                       separators=(",", ":")).encode("utf-8")
                    cached[row] = int.from_bytes(
                        hashlib.blake2b(token, digest_size=8).digest(), "little"
                    )
                    segment_starts[row] = (
                        int(item["source_start"])
                        == int(item["source_segment_start"])
                    )
                    seen[row] = True
            if not seen.all():
                raise TabxDataError(
                    f"Provenance index does not cover every row for {task.task_id}"
                )
            self._record_hashes[task.task_id] = cached
            self._segment_start_rows[task.task_id] = segment_starts
        return cached[np.asarray(rows, dtype=np.int64)]

    def segment_start_rows(
        self, task: TabxTask, rows: Sequence[int]
    ) -> np.ndarray:
        # Populate both provenance caches through the single JSONL scan.
        self.record_hashes(task, rows)
        return self._segment_start_rows[task.task_id][
            np.asarray(rows, dtype=np.int64)
        ]


def process_episode_batch(batch, max_return: float = 0.0):
    """Build the sequence tensors used by ICLearner and retrieval indexing."""
    import torch as th

    obs = batch["obs"][:, :-1]
    reward = batch["reward"][:, :-1]
    action = batch["actions"][:, :-1].squeeze(-1)
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

    # RTG was computed over the complete source episode before windowing and
    # normalized with the fixed per-task scale by TabxEpisodeSource.
    rtg = batch["rtg"][:, :-1].float()
    returns = th.zeros_like(reward)
    returns[:, 0] = reward[:, 0]
    for t in range(1, reward.shape[1]):
        returns[:, t] = returns[:, t - 1] + reward[:, t]
    completion = batch["completion"][:, :-1].float()
    post = th.cat([reward, terminated, completion], dim=-1)
    return rtg, obs, action, post, returns, mask
