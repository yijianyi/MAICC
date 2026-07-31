from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


class BaselineManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineTask:
    index: int
    task_id: str
    h5_path: Path
    index_path: Path
    n_agents: int
    n_trajectories: int
    obs_dim: int
    action_dim: int
    sequence_length: int
    return_min: float
    return_max: float


class FlatIndexMapper:
    """Map the original MAICC flattened retrieval ids back to TABX rows."""

    def __init__(self, task_agents: Sequence[int], trajectories_per_task: int):
        agents = np.asarray(task_agents, dtype=np.int64)
        if agents.ndim != 1 or agents.size == 0 or np.any(agents <= 0):
            raise ValueError("task_agents must contain positive integers")
        if int(trajectories_per_task) <= 0:
            raise ValueError("trajectories_per_task must be positive")
        self.task_agents = agents
        self.trajectories_per_task = int(trajectories_per_task)
        spans = agents * self.trajectories_per_task
        self.offsets = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(spans)]
        )

    @property
    def size(self) -> int:
        return int(self.offsets[-1])

    def encode(self, task_ids, trajectory_ids, agent_ids) -> np.ndarray:
        task_values, trajectory_values, agent_values = np.broadcast_arrays(
            np.asarray(task_ids, dtype=np.int64),
            np.asarray(trajectory_ids, dtype=np.int64),
            np.asarray(agent_ids, dtype=np.int64),
        )
        if task_values.size and (
            task_values.min() < 0
            or task_values.max() >= self.task_agents.size
        ):
            raise IndexError(
                f"task id outside [0, {self.task_agents.size})"
            )
        agents = self.task_agents[task_values]
        if trajectory_values.size and (
            trajectory_values.min() < 0
            or trajectory_values.max() >= self.trajectories_per_task
        ):
            raise IndexError(
                "trajectory id outside "
                f"[0, {self.trajectories_per_task})"
            )
        if agent_values.size and (
            agent_values.min() < 0 or np.any(agent_values >= agents)
        ):
            raise IndexError("agent id outside its task's agent range")
        return (
            self.offsets[task_values]
            + trajectory_values * agents
            + agent_values
        ).astype(np.int64)

    def decode(self, indices) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(indices, dtype=np.int64)
        if values.size and (values.min() < 0 or values.max() >= self.size):
            raise IndexError(
                f"retrieval id outside [0, {self.size}): {values.tolist()}"
            )
        task_ids = np.searchsorted(
            self.offsets[1:], values, side="right"
        ).astype(np.int64)
        local = values - self.offsets[task_ids]
        task_agents = self.task_agents[task_ids]
        trajectory_ids = local // task_agents
        agent_ids = local % task_agents
        return task_ids, trajectory_ids, agent_ids


class BaselineManifest:
    SCHEMA = "original_maicc_tabx_baseline_v1"

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        try:
            raw_bytes = self.path.read_bytes()
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BaselineManifestError(
                f"Cannot read baseline manifest {self.path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema") != self.SCHEMA:
            raise BaselineManifestError(
                f"Expected schema={self.SCHEMA!r} in {self.path}"
            )
        if payload.get("split") != "train":
            raise BaselineManifestError(
                "The MAICC training adapter only accepts a train manifest"
            )
        contract = payload.get("baseline_contract", {})
        required_contract = {
            "non_overlapping": True,
            "normalization": "none",
            "equal_trajectories_per_task": True,
        }
        for key, expected in required_contract.items():
            if contract.get(key) != expected:
                raise BaselineManifestError(
                    f"Baseline contract {key!r} must be {expected!r}"
                )
        strict_full_length = contract.get("strict_full_length")
        if not isinstance(strict_full_length, bool):
            raise BaselineManifestError(
                "Baseline contract 'strict_full_length' must be boolean"
            )
        self.strict_full_length = strict_full_length
        if self.strict_full_length:
            self.padding_mode = "strict"
        else:
            padding_mode = contract.get("padding_mode")
            if padding_mode != "terminal_zero_fill":
                raise BaselineManifestError(
                    "A padded baseline manifest must declare "
                    "padding_mode='terminal_zero_fill'"
                )
            self.padding_mode = padding_mode
        self.allows_terminal_padding = not self.strict_full_length

        schema = payload.get("global_schema", {})
        try:
            self.obs_dim = int(schema["obs_dim"])
            self.action_dim = int(schema["action_dim"])
            self.max_agents = int(schema["max_agents"])
            self.sequence_length = int(payload["sequence_length"])
            self.trajs_per_task = int(payload["trajs_per_task"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BaselineManifestError(
                f"Invalid global schema in {self.path}: {exc}"
            ) from exc
        if min(
            self.obs_dim,
            self.action_dim,
            self.max_agents,
            self.sequence_length,
            self.trajs_per_task,
        ) <= 0:
            raise BaselineManifestError("Manifest dimensions must be positive")
        if self.trajs_per_task % 20 != 0:
            raise BaselineManifestError(
                "trajs_per_task must be divisible by the original "
                "MAICC embedding batch size (20)"
            )

        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise BaselineManifestError("Training manifest contains no tasks")
        tasks = []
        for index, item in enumerate(raw_tasks):
            try:
                h5_path = self._resolve(item["h5_file"])
                index_path = self._resolve(item["index_file"])
                task = BaselineTask(
                    index=index,
                    task_id=str(item["task_id"]),
                    h5_path=h5_path,
                    index_path=index_path,
                    n_agents=int(item["n_agents"]),
                    n_trajectories=int(item["n_trajectories"]),
                    obs_dim=int(item["obs_dim"]),
                    action_dim=int(item["action_dim"]),
                    sequence_length=int(item["sequence_length"]),
                    return_min=float(item["return_stats"]["min"]),
                    return_max=float(item["return_stats"]["max"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise BaselineManifestError(
                    f"Invalid task entry at index {index}: {exc}"
                ) from exc
            if task.n_trajectories != self.trajs_per_task:
                raise BaselineManifestError(
                    f"Task {task.task_id} has {task.n_trajectories} rows; "
                    f"expected {self.trajs_per_task}"
                )
            if (
                task.obs_dim != self.obs_dim
                or task.action_dim != self.action_dim
                or task.sequence_length != self.sequence_length
                or not 0 < task.n_agents <= self.max_agents
            ):
                raise BaselineManifestError(
                    f"Task {task.task_id} violates the global schema"
                )
            tasks.append(task)
        if len({task.task_id for task in tasks}) != len(tasks):
            raise BaselineManifestError("Duplicate task_id in training manifest")
        self.tasks = tuple(tasks)
        self.task_nums = len(self.tasks)
        self.return_min = float(payload["return_stats"]["min"])
        self.return_max = float(payload["return_stats"]["max"])
        if not self.return_min < self.return_max:
            raise BaselineManifestError(
                "Global return range must satisfy min < max"
            )
        self.payload = payload
        self.fingerprint = hashlib.sha256(raw_bytes).hexdigest()
        self.index_mapper = FlatIndexMapper(
            [task.n_agents for task in self.tasks],
            self.trajs_per_task,
        )
        if self.index_mapper.size < 50:
            raise BaselineManifestError(
                "The training set contains fewer than the original "
                "MAICC l=50 retrieval candidates"
            )

    def _resolve(self, value: str) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.path.parent / path
        return path.resolve()

    def validate_files(self) -> None:
        missing = []
        for task in self.tasks:
            if not task.h5_path.is_file():
                missing.append(task.h5_path)
            if not task.index_path.is_file():
                missing.append(task.index_path)
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise BaselineManifestError(
                f"{len(missing)} baseline data files are missing: {preview}"
            )

    def task_by_id(self) -> Dict[str, BaselineTask]:
        return {task.task_id: task for task in self.tasks}
