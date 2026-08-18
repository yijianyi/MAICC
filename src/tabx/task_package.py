from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .manifest import TabxManifest, TabxTask


class TaskPackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class TabxTaskDefinition:
    task_id: str
    task: Mapping[str, Any]
    package_manifest: Mapping[str, Any]
    path: Path
    record_meta: Mapping[str, Any]

    @property
    def physics_name(self) -> str:
        return str(
            self.task.get("physics")
            or self.package_manifest.get("physics")
            or self.record_meta.get("physics")
            or "default"
        )

    @property
    def enemy_heuristic_name(self) -> str:
        return str(
            self.task.get("enemy_heuristic")
            or self.package_manifest.get("heuristic")
            or self.record_meta.get("enemy_heuristic")
            or "expert"
        )


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskPackageError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TaskPackageError(f"Expected a JSON object in {path}")
    return value


def _record_meta_for_task(
    manifest: TabxManifest,
    task: TabxTask,
    raw_data_root: Optional[Path],
) -> tuple[Dict[str, Any], Optional[Path]]:
    if raw_data_root is None or task.index_path is None:
        return {}, None
    try:
        with task.index_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        if not first_line:
            raise TaskPackageError(f"Empty provenance index: {task.index_path}")
        provenance = json.loads(first_line)
        source = Path(str(provenance["source_record"]))
    except (OSError, IndexError, KeyError, json.JSONDecodeError) as exc:
        raise TaskPackageError(
            f"Cannot resolve a source record from {task.index_path}: {exc}"
        ) from exc
    record_dir = source if source.is_absolute() else raw_data_root / source
    meta_path = record_dir / "meta.json"
    if not meta_path.is_file():
        raise TaskPackageError(
            f"Source record metadata for {task.task_id} does not exist: {meta_path}"
        )
    return _read_json(meta_path), record_dir


def _load_provenance_source(task: TabxTask) -> Optional[Path]:
    if task.index_path is None:
        return None
    try:
        with task.index_path.open("r", encoding="utf-8") as handle:
            line = handle.readline()
        if not line:
            return None
        value = json.loads(line)
        return Path(str(value["source_record"]))
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def infer_raw_data_root(
    manifest: TabxManifest,
    tasks: Sequence[TabxTask],
    configured: Optional[Path],
) -> Optional[Path]:
    """Infer the raw root when a server uses a user-specific mount prefix."""
    if configured is not None:
        value = configured.expanduser().resolve()
        if value.is_dir():
            return value
    source = next(
        (item for task in tasks if (item := _load_provenance_source(task))),
        None,
    )
    if source is None or source.is_absolute():
        return configured
    parts = source.parts
    marker = next(
        (index for index, part in enumerate(parts) if part.startswith("task_")),
        None,
    )
    if marker is None:
        return configured
    suffix = Path(*parts[:marker]) if marker else Path()
    candidates = [manifest.path.parent]
    candidates.extend(manifest.path.parents)
    for parent in candidates:
        candidate = (parent / suffix).resolve()
        if (candidate / source.relative_to(suffix)).exists():
            return candidate
    return configured


def _candidate_from_record_meta(meta: Mapping[str, Any]) -> Optional[Path]:
    oracle = meta.get("oracle", {})
    if not isinstance(oracle, Mapping):
        return None
    config_value = oracle.get("config")
    if not config_value:
        return None
    # .../<task-package>/oracle/config.json -> .../<task-package>/task.json
    return Path(str(config_value)).expanduser().parent.parent / "task.json"


def _iter_task_json(root: Path) -> Iterable[Path]:
    """Find task packages without descending into thousands of record dirs."""
    root = root.expanduser().resolve()
    if root.is_file():
        if root.name != "task.json":
            raise TaskPackageError(f"Expected task.json, got {root}")
        yield root
        return
    if not root.is_dir():
        return
    ignored = {".git", "__pycache__", "oracle"}
    for directory, subdirs, files in os.walk(str(root)):
        subdirs[:] = [
            name for name in subdirs
            if name not in ignored and not name.startswith("record-")
        ]
        if "task.json" in files:
            yield Path(directory) / "task.json"


def _definition_from_file(
    path: Path,
    task_id: str,
    record_meta: Mapping[str, Any],
) -> Optional[TabxTaskDefinition]:
    payload = _read_json(path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise TaskPackageError(f"'tasks' is not a list in {path}")
    matches = [
        item for item in tasks
        if isinstance(item, dict) and str(item.get("task_id")) == task_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise TaskPackageError(f"Duplicate task_id={task_id} in {path}")
    item = matches[0]
    if not isinstance(item.get("scenario"), dict):
        raise TaskPackageError(f"Task {task_id} has no scenario object in {path}")
    if not isinstance(item.get("zone_scenario"), dict):
        raise TaskPackageError(
            f"Task {task_id} has no zone_scenario object in {path}"
        )
    package_manifest = payload.get("manifest", {})
    if not isinstance(package_manifest, dict):
        package_manifest = {}
    return TabxTaskDefinition(
        task_id=task_id,
        task=item,
        package_manifest=package_manifest,
        path=path.resolve(),
        record_meta=dict(record_meta),
    )


def load_task_definitions(
    manifest: TabxManifest,
    tasks: Sequence[TabxTask],
    raw_data_root: Optional[Path],
    task_package_roots: Sequence[Path] = (),
) -> Dict[str, TabxTaskDefinition]:
    """Resolve each manifest task to its original custom TABX task object."""
    raw_root = (
        None if raw_data_root is None
        else raw_data_root.expanduser().resolve()
    )
    definitions: Dict[str, TabxTaskDefinition] = {}
    unresolved: Dict[str, tuple[Mapping[str, Any], Optional[Path]]] = {}
    direct_candidates: list[Path] = []

    for task in tasks:
        meta, record_dir = _record_meta_for_task(manifest, task, raw_root)
        unresolved[task.task_id] = (meta, record_dir)
        candidate = _candidate_from_record_meta(meta)
        if candidate is not None and candidate.is_file():
            direct_candidates.append(candidate.resolve())
        if record_dir is not None:
            nearby_parents = [record_dir]
            nearby_parents.extend(
                parent
                for index, parent in enumerate(record_dir.parents)
                if index < 4
            )
            for parent in nearby_parents:
                local = parent / "task.json"
                if local.is_file():
                    direct_candidates.append(local.resolve())

    searched: set[Path] = set()
    candidate_files: list[Path] = []
    for path in direct_candidates:
        if path not in searched:
            searched.add(path)
            candidate_files.append(path)
    for root in task_package_roots:
        for path in _iter_task_json(root):
            resolved = path.resolve()
            if resolved not in searched:
                searched.add(resolved)
                candidate_files.append(resolved)

    for path in candidate_files:
        for task_id, (meta, _record_dir) in tuple(unresolved.items()):
            definition = _definition_from_file(path, task_id, meta)
            if definition is not None:
                definitions[task_id] = definition
                unresolved.pop(task_id)
        if not unresolved:
            break

    if unresolved:
        missing = ", ".join(sorted(unresolved))
        roots = ", ".join(str(path) for path in task_package_roots) or "<none>"
        raise TaskPackageError(
            "Could not locate original task.json definitions for: "
            f"{missing}. raw_data_root={raw_root}; task_package_roots={roots}. "
            "Pass --task-package-root pointing to the server task_packages "
            "directory. The exact custom scenarios cannot be reconstructed "
            "from obs_flat alone."
        )
    return definitions
