#!/usr/bin/env python3
"""Run one real CEM/DEM optimiser step on a converted TABX row."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabx.episode_adapter import TabxEpisodeSource  # noqa: E402
from tabx.manifest import TabxManifest  # noqa: E402
from tabx.normalization import ObservationNormalizer  # noqa: E402
from tabx.training import create_model_and_learner, select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-cem", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    manifest = TabxManifest(cli.dataset_dir / "manifest.json")
    task = manifest.tasks[cli.task_index]
    normalizer = ObservationNormalizer.load(
        cli.normalization, manifest.obs_dim, clip=10.0, preserve_zeros=True
    )
    config = {
        "model": {
            "agent": "tabx_transformer",
            "max_agents": manifest.max_agents,
            "episode_limit": 512,
            "n_embd": 32,
            "n_layer": 2,
            "n_head": 4,
            "hidden_dim": 64,
            "dropout": 0.0,
            "encoder_use_rtg": False,
            "policy_use_rtg": True,
            "train_cem": cli.train_cem,
        },
        "training": {
            "learning_rate": 5e-4,
            "max_batch_size": 1,
            "use_amp": False,
            "grad_norm_clip": 10.0,
            "log_interval": 0,
        },
        "retrieval": {"k": 2, "search_k": 128},
    }
    device = select_device(cli.device)
    _args, _mac, learner, logger = create_model_and_learner(
        config, manifest, device
    )
    source = TabxEpisodeSource(manifest, normalizer=normalizer, seed=7)
    try:
        sampled = source.load_rows(task, [cli.row])
        if device == "cuda":
            sampled.batch.to(device)
        learner.train(sampled.batch, 1, 1, None, train_mode=0)
    finally:
        source.close()
    print("TABX stage-0 smoke step completed with finite losses")
    print(logger.latest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
