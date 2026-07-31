#!/usr/bin/env python3
"""Build a fixed-length, non-overlapping TABX dataset for original MAICC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.original_maicc_tabx.dataset_builder import (
    BaselineDatasetError,
    build_baseline_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_manifest",
        type=Path,
        help="manifest.json produced by convert_tabx_to_maicc.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajs-per-task",
        type=int,
        default=0,
        help="0 selects the largest common multiple of 20.",
    )
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--include-short-terminal",
        action="store_true",
        help=(
            "Retain naturally terminated rows shorter than sequence_length, "
            "using their zero padding and filled mask. This selects the "
            "padding-compatible learner/retrieval path at training time."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = build_baseline_dataset(
            input_manifest=args.input_manifest,
            output_dir=args.output_dir,
            trajectories_per_task=args.trajs_per_task,
            selection_seed=args.selection_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            source_split=args.source_split,
            include_short_terminal=args.include_short_terminal,
            overwrite=args.overwrite,
        )
    except BaselineDatasetError as exc:
        print(f"ORIGINAL MAICC BASELINE DATA ERROR: {exc}", file=sys.stderr)
        return 2
    for split, path in outputs.items():
        print(f"{split}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
