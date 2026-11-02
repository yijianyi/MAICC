#!/usr/bin/env python3
"""Split TABX tasks by reference-distribution availability without copying HDF5."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tabx.config import ConfigError, load_yaml_config
from tabx.manifest import ManifestError, TabxManifest
from tools.convert_tabx_to_maicc import (
    ConversionError, determine_task_id, discover_record_dirs, read_json,
)

GROUPS = ("rl_label", "advance_label")
DISTRIBUTION_FILE = "actions_reference_distribution.npy"
REPORT_SCHEMA = "maicc_tabx_label_groups_v1"


def scan_tasks(raw_root: Path, task_ids: Optional[set] = None) -> Dict[str, dict]:
    """Inspect every record, refusing to split one task across label groups."""
    root = raw_root.expanduser().resolve()
    if not root.is_dir():
        raise ConversionError(f"Raw data root does not exist: {root}")
    tasks: Dict[str, dict] = {}
    for record in discover_record_dirs(root):
        task_id = determine_task_id(record, read_json(record / "meta.json"))
        if task_ids is not None and task_id not in task_ids:
            continue
        available = (record / DISTRIBUTION_FILE).is_file()
        item = tasks.setdefault(task_id, {
            "task_id": task_id, "records_with_distribution": 0,
            "records_without_distribution": 0, "examples": {},
        })
        key = "with_distribution" if available else "without_distribution"
        item[f"records_{key}"] += 1
        item["examples"].setdefault(key, str(record.relative_to(root)))
    if task_ids is not None:
        missing = sorted(task_ids - tasks.keys())
        if missing:
            raise ConversionError("No raw records for converted task_id(s): " + ", ".join(missing))
    if not tasks:
        raise ConversionError("No matching raw tasks found")
    for task_id, item in tasks.items():
        if item["records_with_distribution"] and item["records_without_distribution"]:
            raise ConversionError(
                f"Task {task_id} has mixed distribution availability: "
                f"{item['examples']}. Repair the raw data or select a consistent dataset root."
            )
        item["has_reference_distribution"] = bool(item["records_with_distribution"])
    return tasks


def group_manifest(manifest: TabxManifest, task_ids: Sequence[str], group: str,
                   raw_root: Path) -> dict:
    payload = copy.deepcopy(manifest.payload)
    payload.setdefault("global_schema", {}).update(
        obs_dim=manifest.obs_dim, action_dim=manifest.action_dim,
        max_agents=manifest.max_agents)
    selected = set(task_ids)
    payload["tasks"] = []
    for original in manifest.payload["tasks"]:
        task_id = str(original["task_id"])
        if task_id not in selected:
            continue
        item = copy.deepcopy(original)
        task = manifest.task(task_id)
        item["h5_file"] = str(task.h5_path)
        if task.index_path is not None:
            item["index_file"] = str(task.index_path)
        item["label_group"] = group
        payload["tasks"].append(item)
    payload["source_root"] = str(raw_root)
    payload["label_grouping"] = {
        "group": group, "criterion": f"presence_of_{DISTRIBUTION_FILE}",
        "action_supervision": "behavior",
        "source_manifest": str(manifest.path),
        "source_manifest_sha256": hashlib.sha256(manifest.path.read_bytes()).hexdigest(),
    }
    return payload


def training_config(template: Mapping[str, Any], manifest_path: Path,
                    normalization: Path, run_dir: Path, stage: int,
                    stage0_steps: int, seed: int, steps: Optional[int]) -> dict:
    config = copy.deepcopy(dict(template))
    config.pop("_config_path", None)
    config.setdefault("data", {}).update(
        manifest=str(manifest_path), normalization=str(normalization))
    training = config.setdefault("training", {})
    training.update(seed=seed, output_dir=str(run_dir / f"stage{stage}"))
    # An absolute metrics filename would otherwise bypass the isolated run dir.
    training["metrics_file"] = "training_metrics.csv"
    if steps is not None:
        training["total_steps"] = steps
    if stage == 1:
        config.setdefault("retrieval", {}).update(
            stage0_checkpoint=str(run_dir / "stage0" / f"step_{stage0_steps:09d}"),
            index_dir=str(run_dir / "retrieval"),
            index_manifest=str(run_dir / "retrieval" / "retrieval_manifest.json"),
        )
    return config


def prepare_groups(raw_root: Optional[Path], output_dir: Path,
                   manifest_path: Optional[Path] = None,
                   stage0_template: Optional[Mapping[str, Any]] = None,
                   stage1_template: Optional[Mapping[str, Any]] = None,
                   run_root: Optional[Path] = None, seed: int = 7,
                   steps: Optional[int] = None, distribution_group: str = "rl_label",
                   overwrite: bool = False) -> dict:
    if distribution_group not in GROUPS:
        raise ConversionError(f"Invalid distribution group: {distribution_group}")
    if steps is not None and steps < 1:
        raise ConversionError("steps must be positive")
    if stage1_template is not None and stage0_template is None:
        raise ConversionError("Stage-1 config requires a stage-0 config")
    if stage0_template is not None and manifest_path is None:
        raise ConversionError("Training configs require --manifest")
    manifest = None if manifest_path is None else TabxManifest(manifest_path)
    if raw_root is None:
        source = None if manifest is None else manifest.payload.get("source_root")
        if not source:
            raise ConversionError("Provide --raw-root (or a manifest with source_root)")
        raw_root = Path(str(source))
        if not raw_root.is_absolute():
            raw_root = manifest.path.parent / raw_root
    raw_root = raw_root.expanduser().resolve()
    task_ids = None if manifest is None else {task.task_id for task in manifest.tasks}
    tasks = scan_tasks(raw_root, task_ids)
    output_dir = output_dir.expanduser().resolve()
    run_root = (output_dir / "runs" if run_root is None else run_root.expanduser().resolve())
    other_group = next(group for group in GROUPS if group != distribution_group)
    report = {
        "schema": REPORT_SCHEMA, "raw_root": str(raw_root),
        "source_manifest": None if manifest is None else str(manifest.path),
        "distribution_file": DISTRIBUTION_FILE,
        "distribution_present_group": distribution_group,
        "distribution_absent_group": other_group,
        "action_supervision": "behavior", "groups": {},
    }
    files: Dict[Path, str] = {}
    for group in GROUPS:
        ids = sorted(task_id for task_id, item in tasks.items()
                     if (distribution_group if item["has_reference_distribution"] else other_group) == group)
        directory = output_dir / group
        entry = {
            "task_ids": ids, "n_tasks": len(ids),
            "n_records": sum(tasks[task_id]["records_with_distribution"] +
                             tasks[task_id]["records_without_distribution"] for task_id in ids),
            "task_ids_file": str(directory / "task_ids.txt"),
            "manifest": None, "stage0_config": None, "stage1_config": None,
            "run_dir": str(run_root / group),
            "tasks": [tasks[task_id] for task_id in ids],
        }
        files[directory / "task_ids.txt"] = "".join(task_id + "\n" for task_id in ids)
        report["groups"][group] = entry
        if manifest is None or not ids:
            continue
        group_path = directory / "manifest.json"
        payload = group_manifest(manifest, ids, group, raw_root)
        entry["manifest"] = str(group_path)
        entry["n_trajectories"] = sum(int(task["n_trajectories"]) for task in payload["tasks"])
        entry["n_train_tasks"] = sum(str(task.get("split", "train")) == "train"
                                     for task in payload["tasks"])
        files[group_path] = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if stage0_template is None:
            continue
        if entry["n_train_tasks"] == 0:
            raise ConversionError(f"Group {group} has no training tasks; cannot generate training configs")
        total = steps if steps is not None else int(stage0_template.get("training", {}).get("total_steps", 100000))
        if total < 1:
            raise ConversionError("Stage-0 total_steps must be positive")
        for stage, template in enumerate((stage0_template, stage1_template)):
            if template is None:
                continue
            config_path = directory / f"stage{stage}.yaml"
            config = training_config(template, group_path, directory / "obs_normalization.npz",
                                     run_root / group, stage, total, seed, steps)
            # JSON is a YAML subset; the existing YAML loader reads this directly.
            files[config_path] = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
            entry[f"stage{stage}_config"] = str(config_path)
    report_path = output_dir / "label_groups.json"
    files[report_path] = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    protected = {manifest.path} if manifest is not None else set()
    if manifest is not None:
        protected.update(task.h5_path for task in manifest.tasks)
        protected.update(task.index_path for task in manifest.tasks if task.index_path is not None)
    if protected.intersection(files):
        raise ConversionError("Output would overwrite a source manifest or dataset")
    for path in files:
        if path.exists() and not overwrite:
            raise ConversionError(f"Output exists: {path}; choose a new directory or --overwrite")
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, help="Raw env-centric records; defaults to manifest source_root")
    parser.add_argument("--manifest", type=Path, help="Existing converted manifest; HDF5 files are reused")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path)
    parser.add_argument("--stage1-config", type=Path)
    parser.add_argument("--run-dir", type=Path, help="Independent run root, with one subdirectory per group")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, help="Use this update budget for both stages and both groups")
    parser.add_argument("--distribution-group", choices=GROUPS, default="rl_label")
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args(argv)
    try:
        report = prepare_groups(
            cli.raw_root, cli.output_dir, cli.manifest,
            None if cli.stage0_config is None else load_yaml_config(cli.stage0_config),
            None if cli.stage1_config is None else load_yaml_config(cli.stage1_config),
            cli.run_dir, cli.seed, cli.steps, cli.distribution_group, cli.overwrite)
    except (ConversionError, ManifestError, ConfigError, OSError, ValueError) as exc:
        parser.exit(1, f"Label grouping failed: {exc}\n")
    for group, entry in report["groups"].items():
        print(f"{group}: tasks={entry['n_tasks']} records={entry['n_records']}")
    print(f"Report: {cli.output_dir.resolve() / 'label_groups.json'}")
    print("Action supervision remains actions_behavior; reference files only determine task groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
