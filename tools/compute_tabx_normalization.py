#!/usr/bin/env python3
"""Compute fixed, zero-preserving TABX observation P99 scales.

The implementation never mean-centres observations. Ratio, binary, team and
one-hot-like fields keep scale 1; other continuous fields use a streaming
estimate of P99(abs(x)) over filled, present agent observations.
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--sample-probability", type=float, default=0.0005)
    parser.add_argument("--max-samples", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=7)
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
    if not 0 < cli.sample_probability <= 1:
        raise SystemExit("--sample-probability must be in (0, 1]")
    if cli.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    rng = np.random.default_rng(cli.seed)
    samples = np.empty(
        (cli.max_samples, manifest.obs_dim), dtype=np.float32
    )
    sample_count = 0
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
                present = np.asarray(
                    handle["present"][start:end, :-1, :, 0], dtype=bool
                )
                valid = (filled[:, :, None] > 0) & present
                values = np.abs(obs[valid])
                if values.size:
                    chosen = values[
                        rng.random(values.shape[0]) < cli.sample_probability
                    ]
                    remaining = cli.max_samples - sample_count
                    if chosen.shape[0] > remaining:
                        raise SystemExit(
                            "Normalization sample reservoir is full. Rerun with "
                            "a smaller --sample-probability or larger --max-samples."
                        )
                    if chosen.size:
                        samples[sample_count : sample_count + chosen.shape[0]] = chosen
                        sample_count += chosen.shape[0]
                    count += int(values.shape[0])
        print(
            f"[normalization] task={task.task_id} count={count}", flush=True
        )
    if count <= 0:
        raise SystemExit("No filled observations were found")
    if sample_count <= 0:
        raise SystemExit(
            "No normalization samples selected; increase --sample-probability"
        )
    scale = np.ones(manifest.obs_dim, dtype=np.float64)
    sampled = samples[:sample_count]
    for feature in range(manifest.obs_dim):
        nonzero = sampled[:, feature]
        nonzero = nonzero[nonzero > 0]
        if nonzero.size:
            scale[feature] = np.percentile(nonzero, 99)

    # TABX schema: own[14], (all units except self)*other[16], zones[4]*6.
    # These dimensions are already ratios, binary flags or categorical codes.
    semantic_unit_scale_one = {1, 4, 11, 12}
    other_blocks = (manifest.obs_dim - 14 - 4 * 6) // 16
    for index in semantic_unit_scale_one:
        scale[index] = 1.0
    for block in range(max(0, other_blocks)):
        offset = 14 + block * 16
        for index in (1, 4, 8, 11, 12, 13, 14):
            scale[offset + index] = 1.0
    zone_start = 14 + max(0, other_blocks) * 16
    for zone in range(4):
        # zone_type and effect/type marker are categorical/indicator fields.
        scale[zone_start + zone * 6] = 1.0
        scale[zone_start + zone * 6 + 5] = 1.0

    scale = scale.astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    mean = np.zeros(manifest.obs_dim, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean=mean,
        scale=scale,
        mode=np.asarray("scale_only"),
        statistic=np.asarray("global_random_sample_abs_p99"),
        count=np.asarray(count, dtype=np.int64),
        sample_count=np.asarray(sample_count, dtype=np.int64),
        sample_probability=np.asarray(cli.sample_probability, dtype=np.float64),
        seed=np.asarray(cli.seed, dtype=np.int64),
        schema_version=np.asarray("tabx_obs_normalizer_v2"),
        manifest_sha256=np.asarray(
            hashlib.sha256(cli.manifest.read_bytes()).hexdigest()
        ),
        obs_dim=np.asarray(manifest.obs_dim, dtype=np.int64),
        max_agents=np.asarray(manifest.max_agents, dtype=np.int64),
        sequence_length=np.asarray(manifest.sequence_length, dtype=np.int64),
    )
    print(
        f"Wrote {output}; count={count} samples={sample_count} "
        f"scale_range=[{scale.min():.6g}, {scale.max():.6g}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
