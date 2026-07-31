#!/usr/bin/env python3
"""Compute zero-preserving observation scales from converted TABX data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabx.manifest import TabxManifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/tabx/obs_normalization.npz"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--chunk-rows", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required") from exc

    manifest = TabxManifest(cli.manifest)
    output = cli.output.expanduser().resolve()
    if output.exists() and not cli.overwrite:
        raise SystemExit(f"Output exists: {output}; use --overwrite")
    sum_sq = np.zeros(manifest.obs_dim, dtype=np.float64)
    count = 0
    for task in manifest.tasks_for_split(cli.split):
        with h5py.File(str(task.h5_path), "r") as handle:
            for start in range(0, task.n_trajectories, cli.chunk_rows):
                end = min(task.n_trajectories, start + cli.chunk_rows)
                obs = np.asarray(
                    handle["obs"][start:end, :-1], dtype=np.float64
                )
                filled = np.asarray(
                    handle["filled"][start:end, :-1, 0], dtype=np.float64
                )
                weights = filled[:, :, None, None]
                sum_sq += np.sum(obs * obs * weights, axis=(0, 1, 2))
                count += int(filled.sum()) * task.n_agents
        print(
            f"[normalization] task={task.task_id} count={count}", flush=True
        )
    if count <= 0:
        raise SystemExit("No filled observations were found")
    scale = np.sqrt(sum_sq / count).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    mean = np.zeros(manifest.obs_dim, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean=mean,
        scale=scale,
        mode=np.asarray("scale_only"),
        count=np.asarray(count, dtype=np.int64),
    )
    print(
        f"Wrote {output}; count={count} "
        f"scale_range=[{scale.min():.6g}, {scale.max():.6g}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
