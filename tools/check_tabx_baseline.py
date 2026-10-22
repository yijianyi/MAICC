#!/usr/bin/env python3
"""Read-only preflight for a configured TABX baseline run."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabx.config import load_yaml_config
from tabx.manifest import TabxManifest
from tabx.normalization import ObservationNormalizer
from tabx.training import build_maicc_args
from tabx.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    cli = parser.parse_args()
    config = load_yaml_config(cli.config)
    data = config["data"]
    manifest = TabxManifest(Path(data["manifest"]))
    ObservationNormalizer.load(Path(data["normalization"]), manifest.obs_dim,
                               manifest_path=manifest.path)
    args = build_maicc_args(config, manifest, "cpu")
    if not cli.metadata_only:
        manifest.validate_files(manifest.tasks_for_split("train"))
    budget = int(config["training"].get("agent_time_budget", 0))
    cap = int(config["training"]["max_batch_size"])
    micro_batch = min(cap, max(1, budget // (manifest.max_agents * manifest.sequence_length))) if budget > 0 else cap
    prefix_vectors = sum(t.n_trajectories * t.n_agents * t.sequence_length
                         for t in manifest.tasks_for_split("train"))
    print(json.dumps(dict(tasks=len(manifest.tasks_for_split("train")), obs_dim=manifest.obs_dim,
        max_agents=args.n_agents, max_enemies=args.max_enemies, n_zones=args.n_zones,
        other_unit_blocks=args.obs_other_blocks, sequence_length=manifest.sequence_length,
        cem_architecture=args.cem_architecture, effective_batch=cap,
        micro_batch_at_capacity=micro_batch, accumulation_steps=(cap+micro_batch-1)//micro_batch,
        sampled_windows_per_stage=cap*int(config["training"]["total_steps"]),
        exact_index_bytes_upper_bound=prefix_vectors*(4*args.n_embd+30),
        data_manifest_sha256=file_sha256(manifest.path),
        normalization_sha256=file_sha256(data["normalization"])), indent=2))
    print("Preflight passed. Task/record test splits must be specified separately for paper evaluation.")


if __name__ == "__main__":
    main()
