#!/usr/bin/env python3
"""Train MAICC stage-1 DT with the persistent TABX retrieval store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabx.config import load_yaml_config  # noqa: E402
from tabx.training import run_stage1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src/config/tabx/tabx_stage1.yaml",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    config = load_yaml_config(cli.config)
    if cli.steps is not None:
        config.setdefault("training", {})["total_steps"] = cli.steps
    if cli.output_dir is not None:
        config.setdefault("training", {})["output_dir"] = str(cli.output_dir)
    data = dict(config["data"])
    retrieval = dict(config["retrieval"])
    training = dict(config.get("training", {}))
    checkpoint = run_stage1(
        config=config,
        manifest_path=Path(str(data["manifest"])),
        normalization_path=Path(str(data["normalization"])),
        index_manifest_path=Path(str(retrieval["index_manifest"])),
        stage0_checkpoint=Path(str(retrieval["stage0_checkpoint"])),
        requested_device=str(training.get("device", "auto")),
        resume=cli.resume,
    )
    print(f"STAGE-1 TRAINING COMPLETE: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
