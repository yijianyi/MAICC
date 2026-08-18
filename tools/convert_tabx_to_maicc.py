#!/usr/bin/env python3
"""
Convert TABX offline records into MAICC EpisodeBatch-compatible HDF5 files.

The converter is intentionally standalone: it does not import or modify MAICC
source code. Each task is written to a separate HDF5 file whose root contains
only fields already understood by MAICC's EpisodeBatch:

    obs         float32 [N, L + 1, A, O]
    actions     int64   [N, L + 1, A, 1]
    reward      float32 [N, L + 1, 1]
    rtg         float32 [N, L + 1, 1]  (full-episode return-to-go)
    completion  uint8   [N, L + 1, 1]  (TABX is_win)
    terminated  uint8   [N, L + 1, 1]  (TABX done, including timeout loss)
    truncation  uint8   [N, L + 1, 1]  (diagnostic only)
    present     uint8   [N, L + 1, A, 1]
    filled      int64   [N, L + 1, 1]

TABX-specific provenance is kept in manifest.json and one JSONL index beside
each HDF5 file, so a MAICC loader can safely iterate over all HDF5 root keys.

Examples:

    python tools/convert_tabx_to_maicc.py \
        task59_records_v2_interface_20260724_105051 \
        --output-dir datasets/tabx

    python tools/convert_tabx_to_maicc.py /path/to/full_tabx_records \
        --output-dir datasets/tabx \
        --sequence-length 128 \
        --stride 128

By default, an automatic sequence length equal to the longest discovered
episode/fragment is used. Natural terminal episodes shorter than the chosen
length are retained. Non-terminal fragments are retained only when they can
form a full-length window; an unsafe short remainder is dropped and reported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the runtime
    raise SystemExit(
        "This converter requires numpy. Install the MAICC "
        "requirements first: pip install -r requirements.txt"
    ) from exc

try:
    import h5py
except ImportError:  # Allow --dry-run in lightweight inspection runtimes.
    h5py = None


CONVERTER_VERSION = "3.0.0"
REQUIRED_ARRAYS = (
    "obs_flat.npy",
    "actions_behavior.npy",
    "reward_team.npy",
    "done.npy",
    "reset.npy",
    "truncation.npy",
    "episode_id.npy",
    "is_win.npy",
)


@dataclass(frozen=True)
class TaskPackage:
    task_id: str
    expected_allies: Optional[int]
    package_dir: Path


@dataclass(frozen=True)
class RecordInfo:
    path: Path
    task_id: str
    timesteps: int
    ally_capacity: int
    obs_dim: int
    action_dim: int


@dataclass(frozen=True)
class Segment:
    record: RecordInfo
    start: int
    end: int
    episode_id: int
    active_slots: Tuple[int, ...]
    boundary: str  # natural_terminal, timeout, or partial

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Window:
    segment: Segment
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def reaches_segment_end(self) -> bool:
        return self.end == self.segment.end


@dataclass
class TaskSummary:
    task_id: str
    n_agents: int
    obs_dim: int
    action_dim: int
    windows: List[Window]
    source_records: int
    source_segments: int


class ConversionError(RuntimeError):
    """Raised when TABX data cannot be mapped without guessing."""


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"Expected a JSON object in {path}")
    return value


def as_flat_ints(value: Any) -> List[int]:
    array = np.asarray(value)
    if array.size == 0:
        return []
    return [int(item) for item in array.reshape(-1)]


def discover_task_packages(root: Path) -> Dict[str, TaskPackage]:
    packages: Dict[str, TaskPackage] = {}
    # Production TABX dumps use both task_packages/ and task_pkgs/task_N/.
    # Searching for the exact task.json filename also lets the converter run
    # directly against the common multi-job dataset root.
    for task_json in sorted(root.rglob("task.json")):
        payload = read_json(task_json)
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            raise ConversionError(f"'tasks' must be a list in {task_json}")

        for task in tasks:
            if not isinstance(task, dict) or not task.get("task_id"):
                continue
            task_id = str(task["task_id"])
            scenario = task.get("scenario", {})
            teams = as_flat_ints(scenario.get("teams", []))
            expected_allies = teams.count(0) if teams else None
            candidate = TaskPackage(
                task_id=task_id,
                expected_allies=expected_allies,
                package_dir=task_json.parent,
            )
            previous = packages.get(task_id)
            if (
                previous is not None
                and previous.expected_allies is not None
                and candidate.expected_allies is not None
                and previous.expected_allies != candidate.expected_allies
            ):
                raise ConversionError(
                    f"Conflicting ally counts for task {task_id}: "
                    f"{previous.expected_allies} and {candidate.expected_allies}"
                )
            packages[task_id] = candidate
    return packages


def discover_record_dirs(root: Path) -> List[Path]:
    # A complete server dump contains job_*/records/task_*/record-* rather
    # than one top-level records/ directory. Match record metadata instead of
    # assuming either layout.
    result = [
        path.parent
        for path in root.rglob("meta.json")
        if path.parent.name.startswith("record-")
    ]
    if not result:
        raise ConversionError(f"No record-* directories found below {root}")
    return sorted(result)


def load_array(path: Path, mmap: bool = True) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ConversionError(f"Cannot load NumPy array {path}: {exc}") from exc


def require_record_arrays(record_dir: Path) -> None:
    missing = [name for name in REQUIRED_ARRAYS if not (record_dir / name).is_file()]
    if missing:
        raise ConversionError(
            f"Record {record_dir} is missing required files: {', '.join(missing)}"
        )


def infer_active_slots(
    record_dir: Path,
    start: int,
    ally_capacity: int,
    expected_allies: Optional[int],
    record_meta: Mapping[str, Any],
) -> Tuple[int, ...]:
    slots: List[int] = []
    health_path = record_dir / "unit_max_health.npy"
    team_path = record_dir / "unit_team.npy"

    if health_path.is_file():
        health = load_array(health_path)
        if health.ndim != 2 or health.shape[0] <= start:
            raise ConversionError(
                f"Unexpected unit_max_health shape in {record_dir}: {health.shape}"
            )
        width = min(ally_capacity, health.shape[1])
        active = np.asarray(health[start, :width]) > 0

        if team_path.is_file():
            teams = load_array(team_path)
            if teams.shape != health.shape:
                raise ConversionError(
                    f"unit_team and unit_max_health shapes differ in {record_dir}: "
                    f"{teams.shape} vs {health.shape}"
                )
            active &= np.asarray(teams[start, :width]) == 0
        slots = np.flatnonzero(active).astype(int).tolist()

    if not slots and expected_allies is not None:
        slots = list(range(expected_allies))

    if not slots:
        actual_from_meta = record_meta.get("actual_n_ally")
        if actual_from_meta is not None:
            slots = list(range(int(actual_from_meta)))

    if not slots:
        raise ConversionError(
            f"Cannot infer active ally slots for {record_dir} at timestep {start}. "
            "Provide task_packages/task.json or unit_max_health.npy."
        )

    if expected_allies is not None and len(slots) != expected_allies:
        raise ConversionError(
            f"Active ally count mismatch for {record_dir} at timestep {start}: "
            f"record has {len(slots)}, task.json expects {expected_allies}"
        )
    if min(slots) < 0 or max(slots) >= ally_capacity:
        raise ConversionError(
            f"Active ally slots {slots} exceed obs ally capacity {ally_capacity} "
            f"in {record_dir}"
        )
    return tuple(slots)


def determine_task_id(record_dir: Path, meta: Mapping[str, Any]) -> str:
    task_id = meta.get("task_id")
    if task_id:
        return str(task_id)
    parent_name = record_dir.parent.name
    match = re.search(r"(task_\d+_[0-9a-fA-F]+)$", parent_name)
    if match:
        return match.group(1)
    raise ConversionError(f"Cannot determine task_id for {record_dir}")


def scan_record(
    record_dir: Path,
    packages: Mapping[str, TaskPackage],
) -> Tuple[RecordInfo, List[Segment]]:
    require_record_arrays(record_dir)
    meta = read_json(record_dir / "meta.json")

    obs = load_array(record_dir / "obs_flat.npy")
    actions = load_array(record_dir / "actions_behavior.npy")
    reward = load_array(record_dir / "reward_team.npy")
    done = load_array(record_dir / "done.npy")
    reset = load_array(record_dir / "reset.npy")
    truncation = load_array(record_dir / "truncation.npy")
    episode_id = load_array(record_dir / "episode_id.npy")

    if obs.ndim != 3:
        raise ConversionError(
            f"obs_flat must have shape [T, ally_slots, obs_dim] in {record_dir}; "
            f"got {obs.shape}"
        )
    timesteps, ally_capacity, obs_dim = map(int, obs.shape)
    expected_leading_shapes = {
        "actions_behavior.npy": actions.shape[0] if actions.ndim >= 1 else None,
        "reward_team.npy": reward.shape[0] if reward.ndim >= 1 else None,
        "done.npy": done.shape[0] if done.ndim >= 1 else None,
        "reset.npy": reset.shape[0] if reset.ndim >= 1 else None,
        "truncation.npy": truncation.shape[0] if truncation.ndim >= 1 else None,
        "episode_id.npy": episode_id.shape[0] if episode_id.ndim >= 1 else None,
    }
    bad = {
        name: size
        for name, size in expected_leading_shapes.items()
        if size != timesteps
    }
    if bad:
        raise ConversionError(
            f"Arrays in {record_dir} do not share T={timesteps}: {bad}"
        )
    if actions.ndim != 2 or actions.shape[1] != ally_capacity:
        raise ConversionError(
            f"actions_behavior must have shape [T, {ally_capacity}] in "
            f"{record_dir}; got {actions.shape}"
        )
    for name, array in (
        ("done", done),
        ("reset", reset),
        ("truncation", truncation),
        ("episode_id", episode_id),
    ):
        if array.ndim != 1:
            raise ConversionError(
                f"{name}.npy must have shape [T] in {record_dir}; got {array.shape}"
            )

    task_id = determine_task_id(record_dir, meta)
    package = packages.get(task_id)
    expected_allies = package.expected_allies if package else None
    action_dim = int(meta.get("action_dim", int(np.max(actions)) + 1))
    if action_dim <= 0:
        raise ConversionError(f"Invalid action_dim={action_dim} in {record_dir}")

    record = RecordInfo(
        path=record_dir,
        task_id=task_id,
        timesteps=timesteps,
        ally_capacity=ally_capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    done_b = np.asarray(done) != 0
    reset_b = np.asarray(reset) != 0
    truncation_b = np.asarray(truncation) != 0
    episode_ids = np.asarray(episode_id, dtype=np.int64)

    segments: List[Segment] = []

    def append_segment(start: int, end: int, boundary: str) -> None:
        if end <= start:
            return
        active_slots = infer_active_slots(
            record_dir=record_dir,
            start=start,
            ally_capacity=ally_capacity,
            expected_allies=expected_allies,
            record_meta=meta,
        )
        segments.append(
            Segment(
                record=record,
                start=start,
                end=end,
                episode_id=int(episode_ids[start]),
                active_slots=active_slots,
                boundary=boundary,
            )
        )

    start = 0
    for index in range(timesteps):
        new_episode = index > start and (
            bool(reset_b[index])
            or int(episode_ids[index]) != int(episode_ids[index - 1])
        )
        if new_episode:
            append_segment(start, index, "partial")
            start = index

        if bool(done_b[index]) or bool(truncation_b[index]):
            boundary = "timeout" if bool(truncation_b[index]) else "natural_terminal"
            append_segment(start, index + 1, boundary)
            start = index + 1

    if start < timesteps:
        append_segment(start, timesteps, "partial")

    return record, segments


def build_windows(
    segments: Sequence[Segment],
    sequence_length: int,
    stride: int,
    partial_policy: str,
) -> Tuple[List[Window], Counter]:
    windows: List[Window] = []
    dropped: Counter = Counter()

    for segment in segments:
        if segment.boundary == "partial":
            if partial_policy == "error":
                raise ConversionError(
                    f"Non-terminal fragment found in {segment.record.path}: "
                    f"[{segment.start}, {segment.end})"
                )
            if partial_policy == "drop":
                dropped["partial_segments"] += 1
                dropped["partial_transitions"] += segment.length
                continue

        if segment.length < sequence_length:
            if segment.boundary in {"natural_terminal", "timeout"}:
                windows.append(Window(segment, segment.start, segment.end))
            else:
                # A short non-terminal/timeout sequence cannot be represented
                # safely by MAICC's single `filled` mask: its final observation
                # slot would otherwise look like a valid action timestep.
                dropped[f"short_{segment.boundary}_segments"] += 1
                dropped[f"short_{segment.boundary}_transitions"] += segment.length
            continue

        full_starts = list(
            range(
                segment.start,
                segment.end - sequence_length + 1,
                stride,
            )
        )
        for window_start in full_starts:
            windows.append(
                Window(
                    segment=segment,
                    start=window_start,
                    end=window_start + sequence_length,
                )
            )

        if not full_starts:
            continue

        last_window_end = full_starts[-1] + sequence_length
        if (
            last_window_end < segment.end
            and segment.boundary in {"natural_terminal", "timeout"}
        ):
            # Preserve the terminal transition using one terminal-aligned
            # full window. It may overlap the preceding window.
            aligned_start = segment.end - sequence_length
            if aligned_start != full_starts[-1]:
                windows.append(
                    Window(
                        segment=segment,
                        start=aligned_start,
                        end=segment.end,
                    )
                )
        elif last_window_end < segment.end:
            dropped["partial_remainder_transitions"] += (
                segment.end - last_window_end
            )

    return windows, dropped


def validate_task_dimensions(task_id: str, windows: Sequence[Window]) -> Tuple[int, int, int]:
    if not windows:
        raise ConversionError(f"Task {task_id} has no convertible windows")
    n_agents = windows[0].segment.record.ally_capacity
    obs_dim = windows[0].segment.record.obs_dim
    action_dim = windows[0].segment.record.action_dim

    for window in windows[1:]:
        actual = (
            window.segment.record.ally_capacity,
            window.segment.record.obs_dim,
            window.segment.record.action_dim,
        )
        expected = (n_agents, obs_dim, action_dim)
        if actual != expected:
            raise ConversionError(
                f"Task {task_id} does not have fixed [agents, obs_dim, action_dim]: "
                f"expected {expected}, got {actual} in {window.segment.record.path}"
            )
    return n_agents, obs_dim, action_dim


def select_team_reward(
    reward: np.ndarray,
    start: int,
    end: int,
    active_slots: Sequence[int],
    record_path: Path,
) -> np.ndarray:
    values = np.asarray(reward[start:end])
    if values.ndim == 1:
        result = values
    elif values.ndim == 2:
        if values.shape[1] == 1:
            result = values[:, 0]
        else:
            # Public TABX reward is duplicated by agent. Validate every stored
            # column, including padded slots, and select exactly column 0.
            active_rewards = values
            result = values[:, 0]
            if not np.allclose(
                active_rewards,
                result[:, None],
                rtol=1e-5,
                atol=1e-6,
                equal_nan=False,
            ):
                raise ConversionError(
                    f"reward_team columns differ across agent slots in "
                    f"{record_path}; MAICC common_reward needs one team scalar"
                )
    elif values.ndim == 3 and values.shape[-1] == 1:
        squeezed = values[..., 0]
        return select_team_reward(
            squeezed, 0, squeezed.shape[0], active_slots, record_path
        )
    else:
        raise ConversionError(
            f"Unsupported reward_team shape {reward.shape} in {record_path}"
        )
    return np.asarray(result, dtype=np.float32)


def compression_args(name: str) -> Dict[str, Any]:
    if name == "none":
        return {}
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    if name == "lzf":
        return {"compression": "lzf", "shuffle": True}
    raise AssertionError(name)


def safe_task_filename(task_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._")
    if not value:
        raise ConversionError(f"Cannot create a filename for task_id={task_id!r}")
    return value


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def open_record_arrays(record: RecordInfo) -> Dict[str, np.ndarray]:
    names = {
        "obs": "obs_flat.npy",
        "actions": "actions_behavior.npy",
        "reward": "reward_team.npy",
        "done": "done.npy",
        "truncation": "truncation.npy",
        "is_win": "is_win.npy",
        "unit_max_health": "unit_max_health.npy",
    }
    arrays: Dict[str, np.ndarray] = {}
    for key, name in names.items():
        path = record.path / name
        if path.is_file():
            arrays[key] = load_array(path)
    return arrays


def create_task_hdf5(
    summary: TaskSummary,
    output_dir: Path,
    input_root: Path,
    sequence_length: int,
    compression: str,
    overwrite: bool,
) -> Dict[str, Any]:
    if h5py is None:
        raise ConversionError(
            "Writing HDF5 requires h5py. Install the existing MAICC "
            "requirements first: pip install -r requirements.txt"
        )

    safe_name = safe_task_filename(summary.task_id)
    h5_path = output_dir / f"{safe_name}.h5"
    index_path = output_dir / f"{safe_name}.index.jsonl"
    h5_tmp = output_dir / f".{safe_name}.h5.tmp"
    index_tmp = output_dir / f".{safe_name}.index.jsonl.tmp"

    for target in (h5_path, index_path):
        if target.exists() and not overwrite:
            raise ConversionError(
                f"Output already exists: {target}. Use --overwrite to replace it."
            )
    for temp in (h5_tmp, index_tmp):
        if temp.exists():
            if not overwrite:
                raise ConversionError(
                    f"Stale temporary output exists: {temp}. "
                    "Remove it or use --overwrite."
                )
            temp.unlink()

    count = len(summary.windows)
    time_size = sequence_length + 1
    comp = compression_args(compression)

    returns: List[float] = []
    winning_returns: List[float] = []
    seen_segments = set()
    natural_terminal_windows = 0
    timeout_windows = 0
    partial_windows = 0
    cache_path: Optional[Path] = None
    cache: Dict[str, np.ndarray] = {}

    try:
        with h5py.File(h5_tmp, "w") as handle, index_tmp.open(
            "w", encoding="utf-8", newline="\n"
        ) as index_handle:
            handle.attrs["schema"] = "maicc_episode_batch_v2"
            handle.attrs["source_schema"] = "tabx_env_centric_v1"
            handle.attrs["converter_version"] = CONVERTER_VERSION
            handle.attrs["task_id"] = summary.task_id
            handle.attrs["sequence_length"] = sequence_length
            handle.attrs["n_agents"] = summary.n_agents
            handle.attrs["obs_dim"] = summary.obs_dim
            handle.attrs["action_dim"] = summary.action_dim
            handle.attrs["reward_mode"] = "common_reward_team_column_0"
            handle.attrs["action_source"] = "actions_behavior"
            handle.attrs["completion_source"] = "is_win"
            handle.attrs["terminated_source"] = "done_including_timeout"

            obs_ds = handle.create_dataset(
                "obs", shape=(count, time_size, summary.n_agents, summary.obs_dim),
                dtype=np.float32,
                chunks=(1, min(time_size, 32), summary.n_agents, summary.obs_dim),
                **comp,
            )
            actions_ds = handle.create_dataset(
                "actions", shape=(count, time_size, summary.n_agents, 1),
                dtype=np.int64,
                chunks=(1, min(time_size, 256), summary.n_agents, 1), **comp,
            )
            reward_ds = handle.create_dataset(
                "reward", shape=(count, time_size, 1), dtype=np.float32,
                chunks=(1, min(time_size, 256), 1), **comp,
            )
            rtg_ds = handle.create_dataset(
                "rtg", shape=(count, time_size, 1), dtype=np.float32,
                chunks=(1, min(time_size, 256), 1), **comp,
            )
            completion_ds = handle.create_dataset(
                "completion", shape=(count, time_size, 1), dtype=np.uint8,
                chunks=(1, min(time_size, 256), 1), **comp,
            )
            terminated_ds = handle.create_dataset(
                "terminated", shape=(count, time_size, 1), dtype=np.uint8,
                chunks=(1, min(time_size, 256), 1), **comp,
            )
            truncation_ds = handle.create_dataset(
                "truncation", shape=(count, time_size, 1), dtype=np.uint8,
                chunks=(1, min(time_size, 256), 1), **comp,
            )
            present_ds = handle.create_dataset(
                "present", shape=(count, time_size, summary.n_agents, 1),
                dtype=np.uint8,
                chunks=(1, min(time_size, 32), summary.n_agents, 1), **comp,
            )
            filled_ds = handle.create_dataset(
                "filled", shape=(count, time_size, 1), dtype=np.int64,
                chunks=(1, min(time_size, 256), 1), **comp,
            )

            sorted_windows = sorted(
                enumerate(summary.windows),
                key=lambda item: (
                    str(item[1].segment.record.path), item[1].start, item[1].end
                ),
            )

            # HDF5 row order must remain stable regardless of the record cache
            # traversal order, so write to the original row index.
            for row, window in sorted_windows:
                record = window.segment.record
                if cache_path != record.path:
                    cache = open_record_arrays(record)
                    cache_path = record.path

                obs = cache["obs"]
                actions = cache["actions"]
                reward = cache["reward"]
                done = cache["done"]
                truncation = cache["truncation"]
                is_win = cache.get("is_win")

                # Preserve the configured ally capacity. Never-present slots are
                # excluded through `present`; real agents remain present after death.
                slots = list(range(summary.n_agents))
                active_slots = list(window.segment.active_slots)
                length = window.length
                obs_values = np.asarray(
                    obs[window.start : window.end, slots, :],
                    dtype=np.float32,
                )
                action_values = np.asarray(
                    actions[window.start : window.end, slots],
                    dtype=np.int64,
                )
                reward_values = select_team_reward(
                    reward=reward,
                    start=window.start,
                    end=window.end,
                    active_slots=active_slots,
                    record_path=record.path,
                )
                terminated_values = (
                    np.asarray(done[window.start : window.end]) != 0
                ).astype(np.uint8)
                truncation_values = (
                    np.asarray(truncation[window.start : window.end]) != 0
                ).astype(np.uint8)

                segment_reward = select_team_reward(
                    reward=reward,
                    start=window.segment.start,
                    end=window.segment.end,
                    active_slots=active_slots,
                    record_path=record.path,
                )
                segment_rtg = np.cumsum(segment_reward[::-1], dtype=np.float64)[::-1]
                relative_start = window.start - window.segment.start
                rtg_values = np.asarray(
                    segment_rtg[relative_start : relative_start + length],
                    dtype=np.float32,
                )
                win_value = int(np.asarray(is_win[window.segment.end - 1]).item())
                completion_values = (
                    np.asarray(is_win[window.start : window.end]) != 0
                ).astype(np.uint8)
                present_values = np.zeros((length, summary.n_agents, 1), dtype=np.uint8)
                present_values[:, active_slots, 0] = 1

                if not np.isfinite(obs_values).all():
                    raise ConversionError(
                        f"NaN/Inf found in obs_flat for {record.path}, "
                        f"window [{window.start}, {window.end})"
                    )
                if not np.isfinite(reward_values).all():
                    raise ConversionError(
                        f"NaN/Inf found in reward_team for {record.path}, "
                        f"window [{window.start}, {window.end})"
                    )
                if action_values.size and (
                    int(action_values.min()) < 0
                    or int(action_values.max()) >= summary.action_dim
                ):
                    raise ConversionError(
                        f"Action outside [0, {summary.action_dim}) in {record.path}, "
                        f"window [{window.start}, {window.end})"
                    )

                obs_ds[row, :length] = obs_values
                actions_ds[row, :length, :, 0] = action_values
                reward_ds[row, :length, 0] = reward_values
                rtg_ds[row, :length, 0] = rtg_values
                completion_ds[row, :length, 0] = completion_values
                terminated_ds[row, :length, 0] = terminated_values
                truncation_ds[row, :length, 0] = truncation_values
                present_ds[row, :length] = present_values

                # Store a real next observation when this window ends inside
                # the same source segment. Terminal/record-boundary slots stay
                # zero and are never used as action targets by MAICC.
                if window.end < window.segment.end:
                    obs_ds[row, length] = np.asarray(
                        obs[window.end, slots, :], dtype=np.float32
                    )

                if length == sequence_length:
                    filled_ds[row, : length + 1, 0] = 1
                elif (
                    window.segment.boundary in {"natural_terminal", "timeout"}
                    and window.reaches_segment_end
                ):
                    filled_ds[row, : length + 1, 0] = 1
                else:
                    raise ConversionError(
                        "Internal error: an unsafe short non-terminal window "
                        f"reached the writer for {record.path}"
                    )

                window_return = float(reward_values.sum(dtype=np.float64))
                segment_key = (
                    str(record.path), window.segment.start, window.segment.end
                )
                if segment_key not in seen_segments:
                    seen_segments.add(segment_key)
                    episode_return = float(segment_reward.sum(dtype=np.float64))
                    returns.append(episode_return)
                    if win_value:
                        winning_returns.append(episode_return)
                if (
                    window.reaches_segment_end
                    and window.segment.boundary == "natural_terminal"
                ):
                    natural_terminal_windows += 1
                elif (
                    window.reaches_segment_end
                    and window.segment.boundary == "timeout"
                ):
                    timeout_windows += 1
                else:
                    partial_windows += 1

                index_entry = {
                    "row": row,
                    "task_id": summary.task_id,
                    "source_record": relative_or_absolute(
                        record.path, input_root
                    ),
                    "source_episode_id": window.segment.episode_id,
                    "source_start": window.start,
                    "source_end": window.end,
                    "source_segment_start": window.segment.start,
                    "source_segment_end": window.segment.end,
                    "active_ally_slots": active_slots,
                    "boundary": (
                        window.segment.boundary
                        if window.reaches_segment_end
                        else "window_boundary"
                    ),
                    "window_return": window_return,
                    "is_win": win_value,
                }
                index_handle.write(
                    json.dumps(index_entry, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )

        os.replace(h5_tmp, h5_path)
        os.replace(index_tmp, index_path)
    except Exception:
        for temp in (h5_tmp, index_tmp):
            if temp.exists():
                temp.unlink()
        raise

    return_array = np.asarray(returns, dtype=np.float64)
    abs_scale = float(np.percentile(np.abs(return_array), 95))
    if not np.isfinite(abs_scale) or abs_scale < 1e-6:
        abs_scale = 1.0
    winning_array = np.asarray(winning_returns, dtype=np.float64)
    target_return = (
        float(np.percentile(winning_array, 95))
        if winning_array.size
        else float(np.percentile(return_array, 95))
    )
    return {
        "task_id": summary.task_id,
        "h5_file": h5_path.name,
        "index_file": index_path.name,
        "n_trajectories": count,
        "n_agents": summary.n_agents,
        "obs_dim": summary.obs_dim,
        "action_dim": summary.action_dim,
        "sequence_length": sequence_length,
        "source_records": summary.source_records,
        "source_segments": summary.source_segments,
        "window_boundaries": {
            "natural_terminal": natural_terminal_windows,
            "timeout": timeout_windows,
            "partial_or_intermediate": partial_windows,
        },
        "return_stats": {
            "min": float(return_array.min()),
            "max": float(return_array.max()),
            "mean": float(return_array.mean()),
            "abs_p95_scale": abs_scale,
            "winning_p95_target": target_return,
        },
        "datasets": {
            "obs": [count, time_size, summary.n_agents, summary.obs_dim],
            "actions": [count, time_size, summary.n_agents, 1],
            "reward": [count, time_size, 1],
            "rtg": [count, time_size, 1],
            "completion": [count, time_size, 1],
            "terminated": [count, time_size, 1],
            "truncation": [count, time_size, 1],
            "present": [count, time_size, summary.n_agents, 1],
            "filled": [count, time_size, 1],
        },
    }


def write_manifest(
    output_dir: Path,
    payload: Mapping[str, Any],
    overwrite: bool,
) -> Path:
    target = output_dir / "manifest.json"
    temp = output_dir / ".manifest.json.tmp"
    if target.exists() and not overwrite:
        raise ConversionError(
            f"Output already exists: {target}. Use --overwrite to replace it."
        )
    if temp.exists():
        if not overwrite:
            raise ConversionError(
                f"Stale temporary output exists: {temp}. "
                "Remove it or use --overwrite."
            )
        temp.unlink()
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, target)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    return target


def parse_sequence_length(value: str) -> Optional[int]:
    if value.lower() == "auto":
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--sequence-length must be a positive integer or 'auto'"
        ) from exc
    if result <= 0:
        raise argparse.ArgumentTypeError(
            "--sequence-length must be a positive integer or 'auto'"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TABX record directories to MAICC EpisodeBatch-compatible "
            "HDF5 files without modifying MAICC source code."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help=(
            "TABX root containing records/ or job_*/records/, plus optional "
            "task_packages/ or task_pkgs/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/tabx"),
        help="Output directory (default: datasets/tabx)",
    )
    parser.add_argument(
        "--sequence-length",
        type=parse_sequence_length,
        default=32,
        metavar="N|auto",
        help=(
            "Number of transitions per full MAICC window (default: 32)."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride. Default: the selected sequence length.",
    )
    parser.add_argument(
        "--partial-policy",
        choices=("keep", "drop", "error"),
        default="drop",
        help=(
            "How to handle source segments without done/truncation. 'keep' "
            "emits only safe full-length windows. Default: drop incomplete tails."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="HDF5 compression (default: gzip)",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Convert only this task_id. May be specified more than once.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace converter-owned output files if they exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan, validate, and report the conversion plan without writing.",
    )
    parser.add_argument(
        "--invalid-record-policy",
        choices=("error", "skip"),
        default="error",
        help=(
            "Fail on the first invalid/incomplete record (default), or skip it "
            "and include the failure in manifest.json."
        ),
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=10,
        help="Fixed ally slot capacity in the converted tensors (default: 10).",
    )
    parser.add_argument(
        "--expected-obs-dim",
        type=int,
        default=None,
        help="Optional global obs_dim contract. All converted tasks must match it.",
    )
    parser.add_argument(
        "--expected-action-dim",
        type=int,
        default=8,
        help="Global action_dim contract (default: 8). Use 0 to infer it.",
    )
    return parser


def run_conversion(args: argparse.Namespace) -> Dict[str, Any]:
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    if not input_root.is_dir():
        raise ConversionError(f"Input root does not exist: {input_root}")

    packages = discover_task_packages(input_root)
    record_dirs = discover_record_dirs(input_root)
    selected_task_ids = set(args.task_ids or [])

    records: List[RecordInfo] = []
    segments: List[Segment] = []
    invalid_records: List[Dict[str, str]] = []
    for record_dir in record_dirs:
        try:
            record, record_segments = scan_record(record_dir, packages)
        except ConversionError as exc:
            if args.invalid_record_policy == "error":
                raise
            invalid_records.append(
                {
                    "record": relative_or_absolute(record_dir, input_root),
                    "error": str(exc),
                }
            )
            continue
        if selected_task_ids and record.task_id not in selected_task_ids:
            continue
        records.append(record)
        segments.extend(record_segments)

    if selected_task_ids:
        found = {record.task_id for record in records}
        missing = sorted(selected_task_ids - found)
        if missing:
            raise ConversionError(
                "Requested task_id values were not found: " + ", ".join(missing)
            )
    if not records:
        raise ConversionError("No records remain after task filtering")
    if not segments:
        raise ConversionError("No episode/fragment segments were discovered")

    sequence_length = args.sequence_length
    if sequence_length is None:
        sequence_length = max(segment.length for segment in segments)
    stride = args.stride if args.stride is not None else sequence_length
    if stride <= 0 or stride > sequence_length:
        raise ConversionError(
            f"--stride must be in [1, {sequence_length}], got {stride}"
        )

    task_segments: Dict[str, List[Segment]] = defaultdict(list)
    for segment in segments:
        task_segments[segment.record.task_id].append(segment)

    summaries: List[TaskSummary] = []
    dropped_total: Counter = Counter()
    for task_id in sorted(task_segments):
        current_segments = task_segments[task_id]
        windows, dropped = build_windows(
            current_segments,
            sequence_length=sequence_length,
            stride=stride,
            partial_policy=args.partial_policy,
        )
        dropped_total.update(dropped)
        if not windows:
            continue
        n_agents, obs_dim, action_dim = validate_task_dimensions(task_id, windows)
        if n_agents != args.max_agents:
            raise ConversionError(
                f"Task {task_id} has ally capacity {n_agents}, but "
                f"--max-agents={args.max_agents}. Use 10 for current data or 20 "
                "for the formal 20v20 dataset."
            )
        summaries.append(
            TaskSummary(
                task_id=task_id,
                n_agents=n_agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                windows=windows,
                source_records=len(
                    {window.segment.record.path for window in windows}
                ),
                source_segments=len(
                    {
                        (
                            window.segment.record.path,
                            window.segment.start,
                            window.segment.end,
                        )
                        for window in windows
                    }
                ),
            )
        )

    if not summaries:
        raise ConversionError(
            "No safe MAICC windows were produced. Use an automatic/shorter "
            "sequence length, or inspect --partial-policy."
        )

    obs_dims = sorted({item.obs_dim for item in summaries})
    action_dims = sorted({item.action_dim for item in summaries})
    if len(obs_dims) != 1:
        raise ConversionError(
            "TABX tasks do not share one global obs_dim: "
            + ", ".join(str(value) for value in obs_dims)
        )
    if len(action_dims) != 1:
        raise ConversionError(
            "TABX tasks do not share one global action_dim: "
            + ", ".join(str(value) for value in action_dims)
        )
    global_obs_dim = obs_dims[0]
    global_action_dim = action_dims[0]
    if (
        args.expected_obs_dim is not None
        and global_obs_dim != args.expected_obs_dim
    ):
        raise ConversionError(
            f"Global obs_dim={global_obs_dim}, expected "
            f"{args.expected_obs_dim}"
        )
    if (
        args.expected_action_dim > 0
        and global_action_dim != args.expected_action_dim
    ):
        raise ConversionError(
            f"Global action_dim={global_action_dim}, expected "
            f"{args.expected_action_dim}"
        )

    plan = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "records": len(records),
        "segments": len(segments),
        "tasks": len(summaries),
        "sequence_length": sequence_length,
        "stride": stride,
        "partial_policy": args.partial_policy,
        "invalid_record_policy": args.invalid_record_policy,
        "invalid_records": invalid_records,
        "global_schema": {
            "obs_dim": global_obs_dim,
            "action_dim": global_action_dim,
            "max_agents": args.max_agents,
        },
        "dropped": dict(sorted(dropped_total.items())),
        "task_plans": [
            {
                "task_id": item.task_id,
                "n_trajectories": len(item.windows),
                "n_agents": item.n_agents,
                "obs_dim": item.obs_dim,
                "action_dim": item.action_dim,
            }
            for item in summaries
        ],
    }

    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return {"plan": plan, "manifest": None}

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        planned_targets = [output_dir / "manifest.json"]
        for summary in summaries:
            safe_name = safe_task_filename(summary.task_id)
            planned_targets.extend(
                (
                    output_dir / f"{safe_name}.h5",
                    output_dir / f"{safe_name}.index.jsonl",
                )
            )
        existing = [path for path in planned_targets if path.exists()]
        if existing:
            raise ConversionError(
                "Output already exists; no files were written: "
                + ", ".join(str(path) for path in existing)
                + ". Use --overwrite to replace converter-owned files."
            )

    task_outputs = [
        create_task_hdf5(
            summary=summary,
            output_dir=output_dir,
            input_root=input_root,
            sequence_length=sequence_length,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        for summary in summaries
    ]

    manifest_payload = {
        "schema": "maicc_tabx_manifest_v2",
        "converter_version": CONVERTER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(input_root),
        "sequence_length": sequence_length,
        "stride": stride,
        "partial_policy": args.partial_policy,
        "compression": args.compression,
        "global_schema": {
            "obs_dim": global_obs_dim,
            "action_dim": global_action_dim,
            "max_agents": args.max_agents,
        },
        "invalid_record_policy": args.invalid_record_policy,
        "invalid_records": invalid_records,
        "dropped": dict(sorted(dropped_total.items())),
        "maicc_required_fields": [
            "obs",
            "actions",
            "reward",
            "rtg",
            "completion",
            "terminated",
            "truncation",
            "present",
            "filled",
        ],
        "maicc_runtime_fields_not_present": [
            "state",
            "avail_actions",
        ],
        "tasks": task_outputs,
    }
    manifest_path = write_manifest(
        output_dir=output_dir,
        payload=manifest_payload,
        overwrite=args.overwrite,
    )
    print(f"Wrote MAICC TABX manifest: {manifest_path}")
    return {"plan": plan, "manifest": manifest_payload}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_conversion(args)
    except ConversionError as exc:
        print(f"conversion error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("conversion interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
