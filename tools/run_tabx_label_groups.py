#!/usr/bin/env python3
"""Run the existing MAICC stages independently for prepared TABX label groups."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabx.config import ConfigError, load_yaml_config

GROUPS = ("rl_label", "advance_label")
STAGES = ("normalize", "check", "smoke", "stage0", "index", "stage1", "all", "compare")


def commands_for_group(entry: dict, stage: str, python: str) -> list:
    config_path = entry.get("stage0_config")
    if not config_path:
        raise ValueError("No training config; prepare groups with --manifest and --stage0-config")
    config = load_yaml_config(Path(config_path))
    data, model = config["data"], config.get("model", {})
    training = config.get("training", {})
    command = lambda name: [python, str(ROOT / "tools" / name)]
    normalize = command("compute_tabx_normalization.py") + [
        data["manifest"], "--output", data["normalization"],
        "--n-zones", str(model.get("n_zones", 4)),
        "--seed", str(training.get("seed", 7)),
    ]
    if model.get("max_enemies") is not None:
        normalize += ["--max-enemies", str(model["max_enemies"])]
    stages = {
        "normalize": normalize,
        "check": command("check_tabx_baseline.py") + ["--config", config_path],
        "smoke": command("train_tabx_stage0.py") + [
            "--config", config_path, "--steps", "2", "--max-batch-size", "1",
            "--output-dir", str(Path(entry["run_dir"]) / "smoke_stage0")],
        "stage0": command("train_tabx_stage0.py") + ["--config", config_path],
    }
    if stage in ("index", "stage1", "all"):
        stage1 = entry.get("stage1_config")
        if not stage1:
            raise ValueError("Index/stage-1 requires preparation with --stage1-config")
        load_yaml_config(Path(stage1))
        stages["index"] = command("build_tabx_index.py") + ["--config", stage1]
        stages["stage1"] = command("train_tabx_stage1.py") + ["--config", stage1]
    order = ("normalize", "check", "stage0", "index", "stage1") if stage == "all" else (stage,)
    return [stages[name] for name in order]


def summarize_metrics(path: Path, group: str, stage: str, window: int) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [name for name in (reader.fieldnames or ()) if name.endswith("_loss")]
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty metrics: {path}")
    steps = [int(row["step"]) for row in rows]
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError(f"Non-increasing or duplicate steps in {path}")
    result = []
    for name in columns:
        values = [float(row[name]) for row in rows[-window:] if row.get(name, "").strip()]
        if not values:
            continue
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {name} in {path}")
        result.append(dict(group=group, stage=stage, metric=name, last_step=steps[-1],
                           logged_rows=len(rows), tail_samples=len(values),
                           tail_mean=statistics.mean(values),
                           tail_std=statistics.stdev(values) if len(values) > 1 else 0.0))
    if not result:
        raise ValueError(f"No populated loss metrics in {path}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--group", choices=("both",) + GROUPS, default="both")
    parser.add_argument("--stage", choices=STAGES, default="check")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compare-stage", choices=("stage0", "stage1"), default="stage0")
    parser.add_argument("--tail-window", type=int, default=1000, help="Last N logged rows for comparison")
    cli = parser.parse_args(argv)
    try:
        report = json.loads((cli.groups_dir / "label_groups.json").read_text(encoding="utf-8"))
        if report.get("schema") != "maicc_tabx_label_groups_v1":
            raise ValueError("Unsupported label-group report schema")
        if cli.tail_window < 1:
            raise ValueError("--tail-window must be positive")
        selected = GROUPS if cli.group == "both" else (cli.group,)
        active = []
        for group in selected:
            entry = report["groups"][group]
            if not entry["n_tasks"]:
                if cli.group != "both":
                    raise ValueError(f"Group {group} is empty")
                print(f"Skipping empty group {group}", flush=True)
                continue
            active.append((group, entry))
        if not active:
            raise ValueError("No non-empty groups")
        if cli.stage == "compare":
            rows = []
            for group, entry in active:
                path = Path(entry["run_dir"]) / cli.compare_stage / "training_metrics.csv"
                rows.extend(summarize_metrics(path, group, cli.compare_stage, cli.tail_window))
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            if not cli.dry_run:
                output = cli.groups_dir / f"comparison_{cli.compare_stage}.csv"
                output.write_text(buffer.getvalue(), encoding="utf-8")
                print(f"Comparison: {output}", flush=True)
            print(buffer.getvalue(), end="")
            print("Groups contain different tasks; compare task difficulty, evaluation and update budgets separately.")
            return 0
        # Validate every group's configs before starting any expensive work.
        plans = [(group, commands_for_group(entry, cli.stage, cli.python)) for group, entry in active]
        for group, commands in plans:
            for command in commands:
                print(f"[{group}] {shlex.join(command)}", flush=True)
                if not cli.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)
    except (ConfigError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Grouped run failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
