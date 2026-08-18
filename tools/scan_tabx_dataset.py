#!/usr/bin/env python3
"""Preflight a raw multi-job TABX dataset without writing converted data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.convert_tabx_to_maicc import (  # noqa: E402
    ConversionError,
    discover_record_dirs,
    discover_task_packages,
    load_array,
    read_json,
    scan_record,
)


DEFAULT_SERVER_ROOT = Path(
    "/mnt/cfs/shaopt/data/MARL/"
    "0803_overfit_7task_train_7000record_balance_winrate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw TABX records and report the global schema."
    )
    parser.add_argument("input_root", type=Path, nargs="?", default=DEFAULT_SERVER_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-agents", type=int, default=10)
    parser.add_argument("--max-enemies", type=int, default=10)
    parser.add_argument("--max-zones", type=int, default=4)
    parser.add_argument("--expected-action-dim", type=int, default=8)
    parser.add_argument("--expected-obs-dim", type=int, default=None)
    parser.add_argument("--allow-invalid-records", action="store_true")
    return parser.parse_args()


def task_shapes(root: Path) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for path in sorted(root.rglob("task.json")):
        try:
            payload = read_json(path)
        except ConversionError:
            continue
        for task in payload.get("tasks", []):
            if not isinstance(task, dict) or not task.get("task_id"):
                continue
            scenario = task.get("scenario", {})
            flat_teams = [
                int(value)
                for row in scenario.get("teams", [])
                for value in (row if isinstance(row, list) else [row])
            ]
            zone = task.get("zone_scenario", {})
            result[str(task["task_id"])] = {
                "allies": flat_teams.count(0),
                "enemies": flat_teams.count(1),
                "zones": int(zone.get("n_zone", 0)),
            }
    return result


def optional_nonnegative_int(
    payload: Mapping[str, Any],
    *keys: str,
) -> Optional[int]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ConversionError(f"{key} must be an integer, got {value!r}") from exc
        if result < 0:
            raise ConversionError(f"{key} must be non-negative, got {result}")
        return result
    return None


def infer_record_shape_metadata(
    record_dir: Path,
    segment_starts: Sequence[int],
    obs_ally_capacity: int,
) -> Dict[str, Optional[int]]:
    """Infer task shape when task packages are absent.

    Record metadata describes padded capacities. ``unit_team`` and
    ``unit_max_health`` identify the actual instantiated units at the start of
    every episode segment.
    """

    meta = read_json(record_dir / "meta.json")
    ally_capacity = optional_nonnegative_int(meta, "n_ally", "max_n_ally")
    if ally_capacity is None:
        ally_capacity = obs_ally_capacity
    if ally_capacity != obs_ally_capacity:
        raise ConversionError(
            f"n_ally={ally_capacity} differs from obs ally slots="
            f"{obs_ally_capacity} in {record_dir}"
        )

    unit_capacity = optional_nonnegative_int(meta, "n_units", "max_n_units")
    if unit_capacity is None:
        unit_capacity = ally_capacity
    if unit_capacity < ally_capacity:
        raise ConversionError(
            f"n_units={unit_capacity} is smaller than n_ally={ally_capacity} "
            f"in {record_dir}"
        )
    enemy_capacity = unit_capacity - ally_capacity
    zone_capacity = optional_nonnegative_int(meta, "max_n_zone")
    actual_zones = optional_nonnegative_int(meta, "actual_n_zone", "n_zone")
    actual_enemies = optional_nonnegative_int(
        meta,
        "actual_n_enemy",
        "n_enemy",
    )

    health_path = record_dir / "unit_max_health.npy"
    team_path = record_dir / "unit_team.npy"
    if health_path.is_file() and team_path.is_file() and segment_starts:
        health = load_array(health_path)
        teams = load_array(team_path)
        if health.ndim != 2 or health.shape != teams.shape:
            raise ConversionError(
                f"unit_max_health and unit_team must have the same [T, units] "
                f"shape in {record_dir}; got {health.shape} and {teams.shape}"
            )
        if health.shape[1] != unit_capacity:
            raise ConversionError(
                f"n_units={unit_capacity} differs from unit array width="
                f"{health.shape[1]} in {record_dir}"
            )
        enemy_counts = {
            int(
                np.count_nonzero(
                    (np.asarray(health[start]) > 0)
                    & (np.asarray(teams[start]) == 1)
                )
            )
            for start in segment_starts
        }
        if len(enemy_counts) != 1:
            raise ConversionError(
                f"active enemy count changes across segments in {record_dir}: "
                f"{sorted(enemy_counts)}"
            )
        inferred_enemies = next(iter(enemy_counts))
        if actual_enemies is not None and actual_enemies != inferred_enemies:
            raise ConversionError(
                f"actual enemy count mismatch in {record_dir}: metadata has "
                f"{actual_enemies}, arrays have {inferred_enemies}"
            )
        actual_enemies = inferred_enemies

    if actual_enemies is not None and actual_enemies > enemy_capacity:
        raise ConversionError(
            f"actual enemies={actual_enemies} exceed enemy capacity="
            f"{enemy_capacity} in {record_dir}"
        )
    if (
        actual_zones is not None
        and zone_capacity is not None
        and actual_zones > zone_capacity
    ):
        raise ConversionError(
            f"actual zones={actual_zones} exceed zone capacity={zone_capacity} "
            f"in {record_dir}"
        )

    return {
        "ally_capacity": ally_capacity,
        "enemy_capacity": enemy_capacity,
        "unit_capacity": unit_capacity,
        "zone_capacity": zone_capacity,
        "actual_enemies": actual_enemies,
        "actual_zones": actual_zones,
    }


def main() -> int:
    cli = parse_args()
    root = cli.input_root.expanduser().resolve()
    if not root.is_dir():
        print(f"dataset root does not exist: {root}", file=sys.stderr)
        return 2
    packages = discover_task_packages(root)
    shapes = task_shapes(root)
    record_dirs = discover_record_dirs(root)
    invalid: List[Dict[str, str]] = []
    per_task: Dict[str, Dict[str, Any]] = {}
    boundary_counts: Counter = Counter()
    segment_lengths: List[int] = []

    for index, record_dir in enumerate(record_dirs, start=1):
        try:
            record, segments = scan_record(record_dir, packages)
        except ConversionError as exc:
            invalid.append(
                {
                    "record": str(record_dir.relative_to(root)),
                    "error": str(exc),
                }
            )
            continue
        active_counts = {len(segment.active_slots) for segment in segments}
        if len(active_counts) != 1:
            invalid.append(
                {
                    "record": str(record_dir.relative_to(root)),
                    "error": f"active ally count changes across segments: {active_counts}",
                }
            )
            continue
        active_agents = next(iter(active_counts))
        try:
            record_shape = infer_record_shape_metadata(
                record_dir,
                [segment.start for segment in segments],
                record.ally_capacity,
            )
        except ConversionError as exc:
            invalid.append(
                {
                    "record": str(record_dir.relative_to(root)),
                    "error": str(exc),
                }
            )
            continue
        package_shape = shapes.get(record.task_id, {})
        actual_enemies = package_shape.get(
            "enemies",
            record_shape["actual_enemies"],
        )
        actual_zones = package_shape.get(
            "zones",
            record_shape["actual_zones"],
        )
        info = per_task.setdefault(
            record.task_id,
            {
                "task_id": record.task_id,
                "records": 0,
                "segments": 0,
                "n_agents": active_agents,
                "obs_dim": record.obs_dim,
                "action_dim": record.action_dim,
                "source_timesteps": 0,
                "allies": active_agents,
                "enemies": actual_enemies,
                "zones": actual_zones,
                "ally_capacity": record_shape["ally_capacity"],
                "enemy_capacity": record_shape["enemy_capacity"],
                "unit_capacity": record_shape["unit_capacity"],
                "zone_capacity": record_shape["zone_capacity"],
                "actual_shape_source": (
                    "task_package"
                    if record.task_id in shapes
                    else "record_arrays_and_metadata"
                ),
            },
        )
        expected = (
            info["n_agents"],
            info["obs_dim"],
            info["action_dim"],
            info["enemies"],
            info["zones"],
            info["ally_capacity"],
            info["enemy_capacity"],
            info["unit_capacity"],
            info["zone_capacity"],
        )
        actual = (
            active_agents,
            record.obs_dim,
            record.action_dim,
            actual_enemies,
            actual_zones,
            record_shape["ally_capacity"],
            record_shape["enemy_capacity"],
            record_shape["unit_capacity"],
            record_shape["zone_capacity"],
        )
        if actual != expected:
            invalid.append(
                {
                    "record": str(record_dir.relative_to(root)),
                    "error": f"task shape changed: {expected} -> {actual}",
                }
            )
            continue
        info["records"] += 1
        info["segments"] += len(segments)
        info["source_timesteps"] += record.timesteps
        for segment in segments:
            boundary_counts[segment.boundary] += 1
            segment_lengths.append(segment.length)
        if index % 100 == 0:
            print(
                f"[scan] records={index}/{len(record_dirs)} "
                f"valid_tasks={len(per_task)} invalid={len(invalid)}",
                flush=True,
            )

    tasks = sorted(per_task.values(), key=lambda item: item["task_id"])
    obs_dims = sorted({item["obs_dim"] for item in tasks})
    action_dims = sorted({item["action_dim"] for item in tasks})
    errors: List[str] = []
    warnings: List[str] = []
    if len(obs_dims) != 1:
        errors.append(f"global obs_dim is not unique: {obs_dims}")
    if len(action_dims) != 1:
        errors.append(f"global action_dim is not unique: {action_dims}")
    if cli.expected_obs_dim is not None and obs_dims != [cli.expected_obs_dim]:
        errors.append(
            f"obs_dim={obs_dims}, expected [{cli.expected_obs_dim}]"
        )
    if cli.expected_action_dim > 0 and action_dims != [cli.expected_action_dim]:
        errors.append(
            f"action_dim={action_dims}, expected [{cli.expected_action_dim}]"
        )
    for item in tasks:
        if item["ally_capacity"] > cli.max_agents:
            errors.append(
                f"{item['task_id']} has ally capacity "
                f"{item['ally_capacity']} "
                f"(limit {cli.max_agents})"
            )
        if item["enemy_capacity"] > cli.max_enemies:
            errors.append(
                f"{item['task_id']} has enemy capacity "
                f"{item['enemy_capacity']} "
                f"(limit {cli.max_enemies})"
            )
        if item["zone_capacity"] is None:
            errors.append(
                f"{item['task_id']} has no zone capacity metadata; "
                "the zone limit cannot be verified"
            )
        elif item["zone_capacity"] > cli.max_zones:
            errors.append(
                f"{item['task_id']} has zone capacity "
                f"{item['zone_capacity']} "
                f"(limit {cli.max_zones})"
            )
    if any(item["zones"] is None for item in tasks):
        warnings.append(
            "Actual zone counts are unavailable without task packages; "
            "the configured limit was verified against max_n_zone capacity."
        )
    if invalid and not cli.allow_invalid_records:
        errors.append(
            f"{len(invalid)} invalid/incomplete records; rerun with "
            "--allow-invalid-records only after reviewing them"
        )

    report = {
        "schema": "tabx_preflight_v2",
        "input_root": str(root),
        "record_directories": len(record_dirs),
        "valid_records": sum(item["records"] for item in tasks),
        "invalid_records": invalid,
        "task_packages": len(packages),
        "tasks": len(tasks),
        "global_schema": {
            "obs_dim": obs_dims[0] if len(obs_dims) == 1 else obs_dims,
            "action_dim": action_dims[0] if len(action_dims) == 1 else action_dims,
            "max_active_allies": max(
                (item["n_agents"] for item in tasks), default=0
            ),
            "max_active_enemies": max(
                (
                    item["enemies"]
                    for item in tasks
                    if item["enemies"] is not None
                ),
                default=None,
            ),
            "max_actual_zones": max(
                (
                    item["zones"]
                    for item in tasks
                    if item["zones"] is not None
                ),
                default=None,
            ),
            "max_ally_capacity": max(
                (item["ally_capacity"] for item in tasks),
                default=None,
            ),
            "max_enemy_capacity": max(
                (item["enemy_capacity"] for item in tasks),
                default=None,
            ),
            "max_unit_capacity": max(
                (item["unit_capacity"] for item in tasks),
                default=None,
            ),
            "max_zone_capacity": max(
                (
                    item["zone_capacity"]
                    for item in tasks
                    if item["zone_capacity"] is not None
                ),
                default=None,
            ),
        },
        "segments": {
            "boundaries": dict(sorted(boundary_counts.items())),
            "count": len(segment_lengths),
            "min_length": min(segment_lengths) if segment_lengths else None,
            "max_length": max(segment_lengths) if segment_lengths else None,
        },
        "task_details": tasks,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if cli.output is not None:
        output = cli.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name("." + output.name + ".tmp")
        temp.write_text(text + "\n", encoding="utf-8")
        temp.replace(output)
        print(f"Wrote preflight report: {output}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
