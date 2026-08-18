#!/usr/bin/env python3
"""Train MAICC stage-0 directly from converted TABX HDF5 files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabx.config import load_yaml_config  # noqa: E402
from tabx.training import run_stage0  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run streaming MAICC stage-0 training on TABX data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src/config/tabx/tabx_stage0.yaml",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=None
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    cem = parser.add_mutually_exclusive_group()
    cem.add_argument("--train-cem", action="store_true")
    cem.add_argument("--no-train-cem", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    config = load_yaml_config(cli.config)
    training = config.setdefault("training", {})
    model = config.setdefault("model", {})
    if cli.steps is not None:
        training["total_steps"] = cli.steps
    if cli.output_dir is not None:
        training["output_dir"] = str(cli.output_dir)
    if cli.max_batch_size is not None:
        training["max_batch_size"] = cli.max_batch_size
    if cli.train_cem:
        model["train_cem"] = True
    if cli.no_train_cem:
        model["train_cem"] = False
    data = dict(config.get("data", {}))
    manifest = (
        cli.manifest
        if cli.manifest is not None
        else Path(str(data["manifest"]))
    )
    normalization_value = (
        cli.normalization
        if cli.normalization is not None
        else data.get("normalization")
    )
    normalization = (
        None
        if normalization_value in (None, "", "none")
        else Path(str(normalization_value))
    )
    device = (
        cli.device
        if cli.device is not None
        else str(config.get("training", {}).get("device", "auto"))
    )
    checkpoint = run_stage0(
        config=config,
        manifest_path=manifest,
        normalization_path=normalization,
        requested_device=device,
        resume=cli.resume,
    )
    print(f"STAGE-0 TRAINING COMPLETE: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

