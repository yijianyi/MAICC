#!/usr/bin/env python3
"""Build persistent timestep-bucketed FAISS indexes for TABX stage-1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabx.config import load_yaml_config  # noqa: E402
from tabx.episode_adapter import (  # noqa: E402
    TabxEpisodeSource,
    process_episode_batch,
)
from tabx.manifest import TabxManifest  # noqa: E402
from tabx.normalization import ObservationNormalizer  # noqa: E402
from tabx.retrieval_store import encode_dem_prefixes  # noqa: E402
from tabx.training import (  # noqa: E402
    create_model_and_learner,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "src/config/tabx/tabx_stage1.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def create_metadata_file(path: Path):
    import h5py

    handle = h5py.File(str(path), "w")
    for key, dtype in (
        ("task_index", np.int32),
        ("row", np.int64),
        ("agent_id", np.int16),
        ("episode_return", np.float32),
    ):
        handle.create_dataset(
            key,
            shape=(0,),
            maxshape=(None,),
            chunks=(4096,),
            dtype=dtype,
        )
    return handle


def append_metadata(handle, values: Dict[str, np.ndarray]) -> None:
    size = next(iter(values.values())).shape[0]
    if size == 0:
        return
    old = handle["row"].shape[0]
    for key, value in values.items():
        dataset = handle[key]
        dataset.resize((old + size,))
        dataset[old:] = value


def main() -> int:
    cli = parse_args()
    try:
        import faiss
        import torch as th
    except ImportError as exc:
        raise SystemExit("faiss and torch are required") from exc

    config = load_yaml_config(cli.config)
    data = dict(config["data"])
    retrieval = dict(config.get("retrieval", {}))
    training = dict(config.get("training", {}))
    manifest = TabxManifest(Path(str(data["manifest"])))
    normalization = ObservationNormalizer.load(
        Path(str(data["normalization"])),
        manifest.obs_dim,
        clip=float(data.get("normalization_clip", 10.0)),
        preserve_zeros=bool(data.get("preserve_zero_padding", True)),
    )
    device = select_device(str(training.get("device", "auto")))
    _args, mac, _learner, _logger = create_model_and_learner(
        config, manifest, device
    )
    checkpoint = (
        cli.checkpoint
        if cli.checkpoint is not None
        else Path(str(retrieval["stage0_checkpoint"]))
    )
    mac.load_models(str(checkpoint.expanduser().resolve()))
    mac.agent.eval()
    output_dir = (
        cli.output_dir
        if cli.output_dir is not None
        else Path(str(retrieval["index_dir"]))
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "retrieval_manifest.json"
    if manifest_path.exists() and not cli.overwrite:
        raise SystemExit(f"Output exists: {manifest_path}; use --overwrite")

    anchor_stride = max(1, int(retrieval.get("anchor_stride", 8)))
    anchors = list(range(0, manifest.sequence_length, anchor_stride))
    indexes = {
        anchor: faiss.IndexFlatIP(mac.agent.n_embd) for anchor in anchors
    }
    metadata_handles = {}
    for anchor in anchors:
        metadata_path = output_dir / f"bucket_{anchor:04d}.h5"
        if metadata_path.exists():
            if not cli.overwrite:
                raise SystemExit(f"Output exists: {metadata_path}")
            metadata_path.unlink()
        metadata_handles[anchor] = create_metadata_file(metadata_path)

    rows_per_batch = int(retrieval.get("index_batch_rows", 8))
    max_rows_per_task = int(retrieval.get("max_index_rows_per_task", 0))
    source = TabxEpisodeSource(
        manifest=manifest,
        split="train",
        normalizer=normalization,
        seed=int(training.get("seed", 7)),
        h5_lru_size=int(data.get("h5_lru_size", 16)),
    )
    try:
        with th.no_grad():
            for task in source.tasks:
                if (
                    max_rows_per_task > 0
                    and task.n_trajectories > max_rows_per_task
                ):
                    selected_rows = np.unique(
                        np.linspace(
                            0,
                            task.n_trajectories - 1,
                            num=max_rows_per_task,
                            dtype=np.int64,
                        )
                    )
                else:
                    selected_rows = np.arange(
                        task.n_trajectories, dtype=np.int64
                    )
                for offset in range(0, selected_rows.size, rows_per_batch):
                    rows = selected_rows[offset : offset + rows_per_batch]
                    sampled = source.load_rows(task, rows)
                    batch = sampled.batch
                    if device == "cuda":
                        batch.to(device)
                    seq = process_episode_batch(
                        batch, max_return=_args.max_return
                    )
                    rtg, obs, action, post, returns, mask = seq
                    embedding = encode_dem_prefixes(
                        mac.agent, rtg, obs, action, post
                    )
                    final_returns = returns[:, -1, 0].detach().cpu().numpy()
                    valid = mask[:, :, 0].detach().cpu().numpy() != 0
                    embedding_np = embedding.detach().cpu().numpy()
                    for anchor in anchors:
                        row_ids, agent_ids = np.nonzero(
                            np.repeat(
                                valid[:, anchor : anchor + 1],
                                task.n_agents,
                                axis=1,
                            )
                        )
                        if row_ids.size == 0:
                            continue
                        vectors = np.ascontiguousarray(
                            embedding_np[row_ids, anchor, agent_ids],
                            dtype=np.float32,
                        )
                        faiss.normalize_L2(vectors)
                        indexes[anchor].add(vectors)
                        append_metadata(
                            metadata_handles[anchor],
                            {
                                "task_index": np.full(
                                    row_ids.size, task.index, dtype=np.int32
                                ),
                                "row": rows[row_ids].astype(np.int64),
                                "agent_id": agent_ids.astype(np.int16),
                                "episode_return": final_returns[row_ids].astype(
                                    np.float32
                                ),
                            },
                        )
                print(
                    f"[index] task={task.task_id} "
                    f"rows={selected_rows.size}/{task.n_trajectories}",
                    flush=True,
                )
    finally:
        source.close()
        for handle in metadata_handles.values():
            handle.close()

    buckets = []
    for anchor in anchors:
        index_path = output_dir / f"bucket_{anchor:04d}.faiss"
        faiss.write_index(indexes[anchor], str(index_path))
        buckets.append(
            {
                "anchor": anchor,
                "vectors": int(indexes[anchor].ntotal),
                "index_file": index_path.name,
                "metadata_file": f"bucket_{anchor:04d}.h5",
            }
        )
    payload = {
        "schema": "tabx_retrieval_index_v1",
        "data_manifest": str(manifest.path),
        "stage0_checkpoint": str(checkpoint),
        "embedding_dim": int(mac.agent.n_embd),
        "sequence_length": manifest.sequence_length,
        "anchor_stride": anchor_stride,
        "buckets": buckets,
    }
    temp = output_dir / ".retrieval_manifest.json.tmp"
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(manifest_path)
    print(f"Wrote retrieval index: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
