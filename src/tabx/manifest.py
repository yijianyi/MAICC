from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class TabxTask:
    index: int
    task_id: str
    h5_path: Path
    index_path: Optional[Path]
    n_agents: int
    n_trajectories: int
    obs_dim: int
    action_dim: int
    sequence_length: int
    split: str
    return_min: float
    return_max: float
    return_scale: float
    target_return: float


class TabxManifest:
    SUPPORTED_SCHEMAS = {
        "maicc_tabx_manifest_v1",
        "maicc_tabx_manifest_v2",
    }

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"Cannot read manifest {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ManifestError(f"Manifest must contain a JSON object: {self.path}")
        schema = payload.get("schema")
        if schema not in self.SUPPORTED_SCHEMAS:
            raise ManifestError(f"Unsupported TABX manifest schema: {schema!r}")
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ManifestError("TABX manifest contains no tasks")

        global_schema = payload.get("global_schema", {})
        inferred_obs = {int(item["obs_dim"]) for item in raw_tasks}
        inferred_action = {int(item["action_dim"]) for item in raw_tasks}
        if len(inferred_obs) != 1:
            raise ManifestError(
                f"TABX tasks have multiple obs dimensions: {sorted(inferred_obs)}"
            )
        if len(inferred_action) != 1:
            raise ManifestError(
                "TABX tasks have multiple action dimensions: "
                f"{sorted(inferred_action)}"
            )
        self.obs_dim = int(global_schema.get("obs_dim", next(iter(inferred_obs))))
        self.action_dim = int(
            global_schema.get("action_dim", next(iter(inferred_action)))
        )
        self.max_agents = int(
            global_schema.get(
                "max_agents",
                max(int(item["n_agents"]) for item in raw_tasks),
            )
        )
        self.sequence_length = int(payload["sequence_length"])
        self.payload = payload

        tasks: List[TabxTask] = []
        for index, item in enumerate(raw_tasks):
            task_obs = int(item["obs_dim"])
            task_action = int(item["action_dim"])
            task_agents = int(item["n_agents"])
            task_sequence = int(item.get("sequence_length", self.sequence_length))
            if task_obs != self.obs_dim or task_action != self.action_dim:
                raise ManifestError(
                    f"Task {item.get('task_id')} violates the global schema"
                )
            if task_agents <= 0 or task_agents > self.max_agents:
                raise ManifestError(
                    f"Task {item.get('task_id')} has invalid n_agents={task_agents}"
                )
            if task_sequence != self.sequence_length:
                raise ManifestError(
                    f"Task {item.get('task_id')} has sequence_length="
                    f"{task_sequence}, expected {self.sequence_length}"
                )
            return_stats = item.get("return_stats", {})
            h5_path = Path(str(item["h5_file"]))
            if not h5_path.is_absolute():
                h5_path = self.path.parent / h5_path
            index_path = None
            if item.get("index_file"):
                index_path = Path(str(item["index_file"]))
                if not index_path.is_absolute():
                    index_path = self.path.parent / index_path
            tasks.append(
                TabxTask(
                    index=index,
                    task_id=str(item["task_id"]),
                    h5_path=h5_path.resolve(),
                    index_path=(
                        None if index_path is None else index_path.resolve()
                    ),
                    n_agents=task_agents,
                    n_trajectories=int(item["n_trajectories"]),
                    obs_dim=task_obs,
                    action_dim=task_action,
                    sequence_length=task_sequence,
                    split=str(item.get("split", "train")),
                    return_min=float(return_stats.get("min", 0.0)),
                    return_max=float(return_stats.get("max", 0.0)),
                    return_scale=max(
                        1e-6, float(return_stats.get("abs_p95_scale", 1.0))
                    ),
                    target_return=float(
                        return_stats.get(
                            "winning_p95_target", return_stats.get("max", 0.0)
                        )
                    ),
                )
            )
        self.tasks = tuple(tasks)
        self._by_id = {task.task_id: task for task in self.tasks}
        if len(self._by_id) != len(self.tasks):
            raise ManifestError("TABX manifest contains duplicate task_id values")

    @property
    def max_return(self) -> float:
        return max(task.return_max for task in self.tasks)

    @property
    def min_return(self) -> float:
        return min(task.return_min for task in self.tasks)

    def tasks_for_split(self, split: str) -> Sequence[TabxTask]:
        if split == "all":
            return self.tasks
        selected = tuple(task for task in self.tasks if task.split == split)
        if selected:
            return selected
        # V1 manifests and the initial 16-scenario overfit dataset have no
        # explicit split. Treat all tasks as training data only.
        if split == "train" and all(task.split == "train" for task in self.tasks):
            return self.tasks
        raise ManifestError(f"Manifest contains no tasks for split={split!r}")

    def task(self, task_id: str) -> TabxTask:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise ManifestError(f"Unknown TABX task_id: {task_id}") from exc

    def validate_files(self, tasks: Optional[Iterable[TabxTask]] = None) -> None:
        missing = [
            task.h5_path
            for task in (self.tasks if tasks is None else tasks)
            if not task.h5_path.is_file()
        ]
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise ManifestError(
                f"{len(missing)} converted HDF5 files are missing: {preview}"
            )
