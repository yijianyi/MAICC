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
from typing import Optional, Tuple

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
    parser.add_argument(
        "--max-enemies",
        type=int,
        default=None,
        help=(
            "Enemy slot capacity used to decode the flattened observation. "
            "Defaults to manifest.max_agents, matching the symmetric TABX "
            "10v10/20v20 schemas."
        ),
    )
    parser.add_argument(
        "--n-zones",
        type=int,
        default=None,
        help=(
            "Optional zone-slot contract. The value is inferred from obs_dim "
            "and unit capacities when omitted."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_observation_layout(
    obs_dim: int,
    max_allies: int,
    max_enemies: Optional[int] = None,
    expected_zones: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Return (other-unit blocks, zone slots, enemy capacity)."""
    own_feature_dim = 14
    other_unit_feature_dim = 16
    zone_feature_dim = 6
    enemy_capacity = max_allies if max_enemies is None else int(max_enemies)
    if max_allies <= 0 or enemy_capacity <= 0:
        raise ValueError("TABX ally/enemy capacities must be positive")
    other_blocks = int(max_allies) + enemy_capacity - 1
    zone_width = int(obs_dim) - (
        own_feature_dim + other_blocks * other_unit_feature_dim
    )
    if zone_width < 0 or zone_width % zone_feature_dim:
        raise ValueError(
            "Cannot decode TABX observation layout: "
            f"obs_dim={obs_dim}, max_allies={max_allies}, "
            f"max_enemies={enemy_capacity}, remaining_zone_width={zone_width}"
        )
    zone_slots = zone_width // zone_feature_dim
    if expected_zones is not None and zone_slots != int(expected_zones):
        raise ValueError(
            f"Observation implies {zone_slots} zone slots, expected "
            f"{int(expected_zones)}"
        )
    return other_blocks, zone_slots, enemy_capacity


def update_reservoir(
    reservoir: np.ndarray,
    sample_count: int,
    candidate_count: int,
    candidates: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Uniformly retain at most reservoir.shape[0] candidate rows."""
    capacity = int(reservoir.shape[0])
    for row in candidates:
        candidate_count += 1
        if sample_count < capacity:
            reservoir[sample_count] = row
            sample_count += 1
            continue
        replacement = int(rng.integers(0, candidate_count))
        if replacement < capacity:
            reservoir[replacement] = row
    return sample_count, candidate_count


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
    candidate_count = 0
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
                    if chosen.size:
                        sample_count, candidate_count = update_reservoir(
                            samples,
                            sample_count,
                            candidate_count,
                            chosen,
                            rng,
                        )
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

    try:
        other_blocks, n_zones, max_enemies = infer_observation_layout(
            manifest.obs_dim,
            manifest.max_agents,
            cli.max_enemies,
            cli.n_zones,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # TABX schema: own[14], (all units except self)*other[16], zones*N*6.
    # These dimensions are already ratios, binary flags or categorical codes.
    semantic_unit_scale_one = {1, 4, 11, 12}
    for index in semantic_unit_scale_one:
        scale[index] = 1.0
    for block in range(max(0, other_blocks)):
        offset = 14 + block * 16
        for index in (1, 4, 8, 11, 12, 13, 14):
            scale[offset + index] = 1.0
    zone_start = 14 + max(0, other_blocks) * 16
    for zone in range(n_zones):
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
        candidate_count=np.asarray(candidate_count, dtype=np.int64),
        sample_probability=np.asarray(cli.sample_probability, dtype=np.float64),
        seed=np.asarray(cli.seed, dtype=np.int64),
        schema_version=np.asarray("tabx_obs_normalizer_v2"),
        manifest_sha256=np.asarray(
            hashlib.sha256(cli.manifest.read_bytes()).hexdigest()
        ),
        obs_dim=np.asarray(manifest.obs_dim, dtype=np.int64),
        max_agents=np.asarray(manifest.max_agents, dtype=np.int64),
        max_enemies=np.asarray(max_enemies, dtype=np.int64),
        n_zones=np.asarray(n_zones, dtype=np.int64),
        other_unit_blocks=np.asarray(other_blocks, dtype=np.int64),
        sequence_length=np.asarray(manifest.sequence_length, dtype=np.int64),
    )
    print(
        f"Wrote {output}; count={count} candidates={candidate_count} "
        f"samples={sample_count} zones={n_zones} "
        f"scale_range=[{scale.min():.6g}, {scale.max():.6g}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
