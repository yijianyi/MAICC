#!/usr/bin/env python3
"""Exhaustively validate an original-MAICC TABX data contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.original_maicc_tabx.manifest import (
    BaselineManifest,
    BaselineManifestError,
)
from baseline.original_maicc_tabx.padding import (
    TerminalPaddingError,
    valid_transition_length,
    validate_zero_padding,
)


def validate(manifest_path: Path, chunk_rows: int = 128) -> dict:
    import h5py

    manifest = BaselineManifest(manifest_path)
    manifest.validate_files()
    required = ("obs", "actions", "reward", "terminated", "filled")
    data_bytes = 0
    embedding_bytes = 0
    task_reports = []
    candidates_per_timestep = np.zeros(
        manifest.sequence_length,
        dtype=np.int64,
    )
    for task in manifest.tasks:
        task_lengths = []
        with h5py.File(str(task.h5_path), "r") as handle:
            missing = [field for field in required if field not in handle]
            if missing:
                raise BaselineManifestError(
                    f"{task.h5_path} is missing {missing}"
                )
            expected_time = manifest.sequence_length + 1
            expected = {
                "obs": (
                    task.n_trajectories,
                    expected_time,
                    task.n_agents,
                    manifest.obs_dim,
                ),
                "actions": (
                    task.n_trajectories,
                    expected_time,
                    task.n_agents,
                    1,
                ),
                "reward": (
                    task.n_trajectories,
                    expected_time,
                    1,
                ),
                "terminated": (
                    task.n_trajectories,
                    expected_time,
                    1,
                ),
                "filled": (
                    task.n_trajectories,
                    expected_time,
                    1,
                ),
            }
            for field, shape in expected.items():
                dataset = handle[field]
                if tuple(dataset.shape) != shape:
                    raise BaselineManifestError(
                        f"{task.task_id}/{field} has shape {dataset.shape}; "
                        f"expected {shape}"
                    )
                data_bytes += int(dataset.size * dataset.dtype.itemsize)
            for start in range(0, task.n_trajectories, chunk_rows):
                end = min(start + chunk_rows, task.n_trajectories)
                obs = np.asarray(handle["obs"][start:end])
                reward = np.asarray(handle["reward"][start:end])
                actions = np.asarray(handle["actions"][start:end])
                filled = np.asarray(handle["filled"][start:end])
                terminated = np.asarray(handle["terminated"][start:end])
                if not np.isfinite(obs).all():
                    raise BaselineManifestError(
                        f"NaN/Inf observation in {task.task_id} rows "
                        f"[{start}, {end})"
                    )
                if not np.isfinite(reward).all():
                    raise BaselineManifestError(
                        f"NaN/Inf reward in {task.task_id} rows "
                        f"[{start}, {end})"
                    )
                if actions.size and (
                    int(actions.min()) < 0
                    or int(actions.max()) >= manifest.action_dim
                ):
                    raise BaselineManifestError(
                        f"Action outside [0, {manifest.action_dim}) in "
                        f"{task.task_id} rows [{start}, {end})"
                    )
                for offset in range(end - start):
                    arrays = {
                        "obs": obs[offset],
                        "actions": actions[offset],
                        "reward": reward[offset],
                        "terminated": terminated[offset],
                        "filled": filled[offset],
                    }
                    try:
                        length = valid_transition_length(
                            arrays["filled"],
                            arrays["terminated"],
                            manifest.sequence_length,
                            manifest.allows_terminal_padding,
                        )
                        validate_zero_padding(
                            arrays,
                            length,
                            manifest.sequence_length,
                        )
                    except TerminalPaddingError as exc:
                        raise BaselineManifestError(
                            f"Invalid row={start + offset} in "
                            f"{task.task_id}: {exc}"
                        ) from exc
                    task_lengths.append(length)
                    candidates_per_timestep[:length] += task.n_agents

        index_rows = []
        index_lengths = []
        with task.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    index_rows.append(int(entry["row"]))
                    index_lengths.append(
                        int(entry.get("valid_length", task.sequence_length))
                    )
        if index_rows != list(range(task.n_trajectories)):
            raise BaselineManifestError(
                f"Index rows are not contiguous for {task.task_id}"
            )
        if index_lengths != task_lengths:
            raise BaselineManifestError(
                f"Index valid lengths do not match HDF5 for {task.task_id}"
            )
        embedding_bytes += (
            task.n_trajectories
            * manifest.sequence_length
            * task.n_agents
            * 64
            * 4
        )
        task_reports.append(
            {
                "task_id": task.task_id,
                "n_agents": task.n_agents,
                "rows": task.n_trajectories,
                "min_length": int(min(task_lengths)),
                "max_length": int(max(task_lengths)),
                "padded_rows": int(
                    np.count_nonzero(
                        np.asarray(task_lengths) < manifest.sequence_length
                    )
                ),
            }
        )
    min_candidates = int(candidates_per_timestep.min())
    if min_candidates < 50:
        raise BaselineManifestError(
            f"Only {min_candidates} valid candidates at the sparsest "
            "timestep; original retrieval requires l=50"
        )
    return {
        "manifest": str(manifest.path),
        "tasks": task_reports,
        "data_gib_uncompressed": data_bytes / (1024**3),
        "stage1_embedding_gib": embedding_bytes / (1024**3),
        "retrieval_candidates": manifest.index_mapper.size,
        "min_valid_retrieval_candidates": min_candidates,
        "padding_mode": manifest.padding_mode,
        "contract": "valid",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=128)
    args = parser.parse_args()
    try:
        report = validate(args.manifest, args.chunk_rows)
    except (BaselineManifestError, OSError, ValueError) as exc:
        print(f"BASELINE DATA VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
