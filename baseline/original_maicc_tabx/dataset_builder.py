from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .padding import (
    TerminalPaddingError,
    valid_transition_length,
    validate_zero_padding,
)


class BaselineDatasetError(RuntimeError):
    pass


REQUIRED_FIELDS = ("obs", "actions", "reward", "terminated", "filled")


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def safe_filename(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "task"


def select_non_overlapping(entries: Sequence[Mapping]) -> List[dict]:
    grouped: MutableMapping[Tuple[str, int], List[Mapping]] = defaultdict(list)
    for entry in entries:
        key = (
            str(entry["source_record"]),
            int(entry["source_episode_id"]),
        )
        grouped[key].append(entry)
    selected: List[dict] = []
    for key in sorted(grouped):
        last_end = None
        ordered = sorted(
            grouped[key],
            key=lambda item: (
                int(item["source_start"]),
                int(item["source_end"]),
                int(item["row"]),
            ),
        )
        for entry in ordered:
            start = int(entry["source_start"])
            end = int(entry["source_end"])
            if last_end is None or start >= last_end:
                selected.append(dict(entry))
                last_end = end
    return selected


def split_episode_groups(
    entries: Sequence[Mapping],
    task_id: str,
    seed: int,
    fractions: Mapping[str, float],
) -> Dict[str, List[dict]]:
    grouped: MutableMapping[Tuple[str, int], List[dict]] = defaultdict(list)
    for entry in entries:
        key = (
            str(entry["source_record"]),
            int(entry["source_episode_id"]),
        )
        grouped[key].append(dict(entry))
    keys = sorted(grouped)
    if not keys:
        return {name: [] for name in fractions}
    rng = np.random.default_rng(stable_seed(seed, task_id))
    order = rng.permutation(len(keys))
    keys = [keys[int(index)] for index in order]

    names = list(fractions)
    raw_counts = [len(keys) * float(fractions[name]) for name in names]
    counts = [int(math.floor(value)) for value in raw_counts]
    remainder = len(keys) - sum(counts)
    ranking = sorted(
        range(len(names)),
        key=lambda index: (
            -(raw_counts[index] - counts[index]),
            index,
        ),
    )
    for index in ranking[:remainder]:
        counts[index] += 1
    if counts[names.index("train")] == 0:
        donor = max(range(len(counts)), key=lambda index: counts[index])
        if donor == names.index("train") or counts[donor] <= 1:
            raise BaselineDatasetError(
                f"Task {task_id} has too few source episodes for a train split"
            )
        counts[donor] -= 1
        counts[names.index("train")] += 1

    result = {name: [] for name in names}
    cursor = 0
    for name, count in zip(names, counts):
        for key in keys[cursor : cursor + count]:
            result[name].extend(grouped[key])
        cursor += count
    return result


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineDatasetError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineDatasetError(f"Expected JSON object in {path}")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_index(path: Path) -> List[dict]:
    result = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("entry is not an object")
                result.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise BaselineDatasetError(
            f"Cannot read index {path}: {exc}"
        ) from exc
    return result


def _candidate_rows(
    h5_path: Path,
    index_entries: Sequence[Mapping],
    sequence_length: int,
    include_short_terminal: bool,
) -> List[dict]:
    import h5py

    with h5py.File(str(h5_path), "r") as handle:
        missing = [field for field in REQUIRED_FIELDS if field not in handle]
        if missing:
            raise BaselineDatasetError(
                f"{h5_path} is missing datasets {missing}"
            )
        rows = int(handle["filled"].shape[0])
        candidates = []
        seen_rows = set()
        for entry in index_entries:
            row = int(entry["row"])
            start = int(entry["source_start"])
            end = int(entry["source_end"])
            if row in seen_rows:
                raise BaselineDatasetError(
                    f"Duplicate row={row} in index for {h5_path}"
                )
            seen_rows.add(row)
            if not 0 <= row < rows:
                raise BaselineDatasetError(
                    f"Index row={row} outside HDF5 row count {rows}"
                )
            source_length = end - start
            if source_length < 1 or source_length > sequence_length:
                continue
            filled = np.asarray(handle["filled"][row])
            terminated = np.asarray(handle["terminated"][row])
            try:
                valid_length = valid_transition_length(
                    filled,
                    terminated,
                    sequence_length,
                    allow_terminal_padding=include_short_terminal,
                )
            except TerminalPaddingError as exc:
                if source_length < sequence_length and not include_short_terminal:
                    continue
                raise BaselineDatasetError(
                    f"Invalid row={row} in {h5_path}: {exc}"
                ) from exc
            if valid_length != source_length:
                raise BaselineDatasetError(
                    f"Index/HDF5 length mismatch for row={row} in {h5_path}: "
                    f"index={source_length}, filled={valid_length}"
                )
            if valid_length < sequence_length and str(
                entry.get("boundary", "")
            ) != "natural_terminal":
                raise BaselineDatasetError(
                    f"Short row={row} in {h5_path} is not a natural terminal"
                )
            candidate = dict(entry)
            candidate["valid_length"] = valid_length
            candidates.append(candidate)
        return select_non_overlapping(candidates)


def _choose_rows(
    entries: Sequence[Mapping],
    count: int,
    seed: int,
    task_id: str,
) -> List[dict]:
    if len(entries) < count:
        raise BaselineDatasetError(
            f"Task {task_id} has {len(entries)} eligible rows; needs {count}"
        )
    rng = np.random.default_rng(stable_seed(seed, f"rows:{task_id}"))
    order = rng.permutation(len(entries))[:count]
    return [dict(entries[int(index)]) for index in order]


def _write_task(
    source_h5: Path,
    entries: Sequence[Mapping],
    output_h5: Path,
    output_index: Path,
    task: Mapping,
    split: str,
    sequence_length: int,
    include_short_terminal: bool,
    overwrite: bool,
) -> dict:
    import h5py

    for path in (output_h5, output_index):
        if path.exists() and not overwrite:
            raise BaselineDatasetError(
                f"Output exists: {path}; pass --overwrite to replace it"
            )
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    h5_temp = output_h5.with_suffix(output_h5.suffix + ".tmp")
    index_temp = output_index.with_suffix(output_index.suffix + ".tmp")
    for path in (h5_temp, index_temp):
        if path.exists():
            path.unlink()

    returns = []
    try:
        with h5py.File(str(source_h5), "r") as source, \
                h5py.File(str(h5_temp), "w") as target, \
                index_temp.open("w", encoding="utf-8") as index_handle:
            target.attrs["schema"] = "original_maicc_tabx_baseline_v1"
            target.attrs["split"] = split
            target.attrs["task_id"] = str(task["task_id"])
            target.attrs["strict_full_length"] = not include_short_terminal
            target.attrs["padding_mode"] = (
                "terminal_zero_fill"
                if include_short_terminal
                else "strict"
            )
            target.attrs["normalization"] = "none"
            target.attrs["sequence_length"] = int(sequence_length)
            datasets = {}
            for field in REQUIRED_FIELDS:
                source_dataset = source[field]
                shape = (len(entries), *source_dataset.shape[1:])
                chunks = (
                    1,
                    *[
                        min(int(size), 32 if index == 0 else int(size))
                        for index, size in enumerate(shape[1:])
                    ],
                )
                datasets[field] = target.create_dataset(
                    field,
                    shape=shape,
                    dtype=source_dataset.dtype,
                    chunks=chunks,
                    compression="gzip",
                )

            for output_row, entry in enumerate(entries):
                source_row = int(entry["row"])
                for field in REQUIRED_FIELDS:
                    datasets[field][output_row] = source[field][source_row]
                filled = np.asarray(datasets["filled"][output_row])
                terminated = np.asarray(
                    datasets["terminated"][output_row]
                )
                try:
                    valid_length = valid_transition_length(
                        filled,
                        terminated,
                        sequence_length,
                        allow_terminal_padding=include_short_terminal,
                    )
                except TerminalPaddingError as exc:
                    raise BaselineDatasetError(
                        f"Invalid output row for task {task['task_id']}: {exc}"
                    ) from exc
                obs = np.asarray(datasets["obs"][output_row])
                reward = np.asarray(datasets["reward"][output_row])
                actions = np.asarray(datasets["actions"][output_row])
                arrays = {
                    "obs": obs,
                    "actions": actions,
                    "reward": reward,
                    "terminated": terminated,
                }
                try:
                    validate_zero_padding(
                        arrays,
                        valid_length,
                        sequence_length,
                    )
                except TerminalPaddingError as exc:
                    raise BaselineDatasetError(
                        f"Invalid terminal padding for task "
                        f"{task['task_id']}: {exc}"
                    ) from exc
                if not np.isfinite(obs).all() or not np.isfinite(reward).all():
                    raise BaselineDatasetError(
                        f"NaN/Inf reached strict output for task "
                        f"{task['task_id']}"
                    )
                if actions.size and (
                    int(actions.min()) < 0
                    or int(actions.max()) >= int(task["action_dim"])
                ):
                    raise BaselineDatasetError(
                        f"Invalid action reached strict output for task "
                        f"{task['task_id']}"
                    )
                trajectory_return = float(
                    reward[:valid_length].sum(dtype=np.float64)
                )
                returns.append(trajectory_return)
                output_entry = dict(entry)
                output_entry["source_converted_row"] = source_row
                output_entry["row"] = output_row
                output_entry["split"] = split
                output_entry["window_return"] = trajectory_return
                output_entry["valid_length"] = valid_length
                index_handle.write(
                    json.dumps(
                        output_entry,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(h5_temp, output_h5)
        os.replace(index_temp, output_index)
    except Exception:
        for path in (h5_temp, index_temp):
            if path.exists():
                path.unlink()
        raise

    values = np.asarray(returns, dtype=np.float64)
    lengths = np.asarray(
        [int(entry.get("valid_length", sequence_length)) for entry in entries],
        dtype=np.int64,
    )
    return {
        "task_id": str(task["task_id"]),
        "h5_file": output_h5.name,
        "index_file": output_index.name,
        "n_trajectories": len(entries),
        "n_agents": int(task["n_agents"]),
        "obs_dim": int(task["obs_dim"]),
        "action_dim": int(task["action_dim"]),
        "sequence_length": int(sequence_length),
        "split": split,
        "return_stats": {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        },
        "length_stats": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "full_rows": int(np.count_nonzero(lengths == sequence_length)),
            "padded_rows": int(np.count_nonzero(lengths < sequence_length)),
        },
    }


def _write_manifest(
    output_root: Path,
    source_manifest: Path,
    split: str,
    sequence_length: int,
    global_schema: Mapping,
    tasks: Sequence[Mapping],
    trajectories_per_task,
    include_short_terminal: bool,
    overwrite: bool,
) -> Path:
    manifest_path = output_root / f"{split}_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise BaselineDatasetError(
            f"Output exists: {manifest_path}; pass --overwrite to replace it"
        )
    all_min = min(float(task["return_stats"]["min"]) for task in tasks)
    all_max = max(float(task["return_stats"]["max"]) for task in tasks)
    weighted_sum = sum(
        float(task["return_stats"]["mean"])
        * int(task["n_trajectories"])
        for task in tasks
    )
    total_rows = sum(int(task["n_trajectories"]) for task in tasks)
    payload = {
        "schema": "original_maicc_tabx_baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest),
        "split": split,
        "sequence_length": int(sequence_length),
        "global_schema": {
            "obs_dim": int(global_schema["obs_dim"]),
            "action_dim": int(global_schema["action_dim"]),
            "max_agents": int(global_schema["max_agents"]),
        },
        "task_nums": len(tasks),
        "trajs_per_task": trajectories_per_task,
        "return_stats": {
            "min": all_min,
            "max": all_max,
            "mean": weighted_sum / total_rows,
        },
        "baseline_contract": {
            "strict_full_length": not include_short_terminal,
            "padding_mode": (
                "terminal_zero_fill"
                if include_short_terminal
                else "strict"
            ),
            "non_overlapping": True,
            "normalization": "none",
            "equal_trajectories_per_task": split == "train",
            "retrieval_split": "train",
        },
        "tasks": tasks,
    }
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, manifest_path)
    return manifest_path


def build_baseline_dataset(
    input_manifest: Path,
    output_dir: Path,
    trajectories_per_task: int = 0,
    selection_seed: int = 7,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    source_split: str = "train",
    include_short_terminal: bool = False,
    overwrite: bool = False,
) -> Dict[str, Path]:
    input_manifest = Path(input_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    payload = _load_json(input_manifest)
    if payload.get("schema") not in {
        "maicc_tabx_manifest_v1",
        "maicc_tabx_manifest_v2",
    }:
        raise BaselineDatasetError(
            f"Unsupported converted manifest schema {payload.get('schema')!r}"
        )
    fractions = {
        "train": float(train_fraction),
        "validation": float(validation_fraction),
        "test": float(test_fraction),
    }
    if any(value < 0 for value in fractions.values()) or not math.isclose(
        sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise BaselineDatasetError("Split fractions must be nonnegative and sum to 1")
    if fractions["train"] <= 0:
        raise BaselineDatasetError("train_fraction must be positive")

    sequence_length = int(payload["sequence_length"])
    schema = payload.get("global_schema", {})
    raw_tasks = [
        task
        for task in payload.get("tasks", [])
        if str(task.get("split", "train")) == source_split
    ]
    if not raw_tasks:
        raise BaselineDatasetError(
            f"No tasks found for source split {source_split!r}"
        )

    task_plans = []
    for task in raw_tasks:
        h5_path = _resolve(input_manifest.parent, task["h5_file"])
        index_path = _resolve(input_manifest.parent, task["index_file"])
        entries = _candidate_rows(
            h5_path,
            _load_index(index_path),
            sequence_length,
            include_short_terminal=include_short_terminal,
        )
        split_entries = split_episode_groups(
            entries,
            str(task["task_id"]),
            selection_seed,
            fractions,
        )
        task_plans.append(
            {
                "task": task,
                "source_h5": h5_path,
                "entries": split_entries,
            }
        )

    available = min(
        len(plan["entries"]["train"]) for plan in task_plans
    )
    if trajectories_per_task <= 0:
        trajectories_per_task = (available // 20) * 20
    trajectories_per_task = int(trajectories_per_task)
    if trajectories_per_task <= 0 or trajectories_per_task % 20 != 0:
        raise BaselineDatasetError(
            "Training trajectories per task must be a positive multiple of 20"
        )
    if trajectories_per_task < 32:
        raise BaselineDatasetError(
            "Training trajectories per task must be at least the original "
            "MAICC batch size (32)"
        )
    for plan in task_plans:
        task_id = str(plan["task"]["task_id"])
        available_rows = len(plan["entries"]["train"])
        if available_rows < trajectories_per_task:
            raise BaselineDatasetError(
                f"Task {task_id} has {available_rows} eligible rows; "
                f"needs {trajectories_per_task}"
            )
        plan["selected_train"] = _choose_rows(
            plan["entries"]["train"],
            trajectories_per_task,
            selection_seed,
            task_id,
        )
    candidates_per_timestep = np.zeros(
        sequence_length,
        dtype=np.int64,
    )
    for plan in task_plans:
        n_agents = int(plan["task"]["n_agents"])
        for entry in plan["selected_train"]:
            valid_length = int(
                entry.get("valid_length", sequence_length)
            )
            candidates_per_timestep[:valid_length] += n_agents
    if int(candidates_per_timestep.min()) < 50:
        sparse_timestep = int(candidates_per_timestep.argmin())
        raise BaselineDatasetError(
            f"Only {int(candidates_per_timestep.min())} valid retrieval "
            f"candidates at timestep={sparse_timestep}; original l=50"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for split in fractions:
        output_tasks = []
        split_dir = output_dir / split
        for plan in task_plans:
            task = plan["task"]
            entries = plan["entries"][split]
            if split == "train":
                entries = plan["selected_train"]
            if not entries:
                continue
            filename = safe_filename(str(task["task_id"]))
            h5_path = split_dir / f"{filename}.h5"
            index_path = split_dir / f"{filename}.index.jsonl"
            output_task = _write_task(
                source_h5=plan["source_h5"],
                entries=entries,
                output_h5=h5_path,
                output_index=index_path,
                task=task,
                split=split,
                sequence_length=sequence_length,
                include_short_terminal=include_short_terminal,
                overwrite=overwrite,
            )
            output_task["h5_file"] = str(
                h5_path.relative_to(output_dir)
            ).replace("\\", "/")
            output_task["index_file"] = str(
                index_path.relative_to(output_dir)
            ).replace("\\", "/")
            output_tasks.append(output_task)
        if not output_tasks:
            continue
        manifest_path = _write_manifest(
            output_root=output_dir,
            source_manifest=input_manifest,
            split=split,
            sequence_length=sequence_length,
            global_schema={
                "obs_dim": int(schema.get("obs_dim", output_tasks[0]["obs_dim"])),
                "action_dim": int(
                    schema.get("action_dim", output_tasks[0]["action_dim"])
                ),
                # MAICC sizes its dense positional table from args.n_agents.
                # Use the largest real task, not the converter's capacity cap.
                "max_agents": int(
                    max(task["n_agents"] for task in output_tasks)
                ),
            },
            tasks=output_tasks,
            trajectories_per_task=(
                trajectories_per_task if split == "train" else None
            ),
            include_short_terminal=include_short_terminal,
            overwrite=overwrite,
        )
        written[split] = manifest_path
    if "train" not in written:
        raise BaselineDatasetError("No baseline training manifest was produced")
    return written
