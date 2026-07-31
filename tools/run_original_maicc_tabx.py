#!/usr/bin/env python3
"""Launch untouched Sacred MAICC with the strict TABX data adapter."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.original_maicc_tabx.bootstrap import install
from baseline.original_maicc_tabx.fidelity import verify_upstream_core
from baseline.original_maicc_tabx.manifest import BaselineManifest


PROTECTED_KEYS = {
    "agent",
    "learner",
    "mac",
    "n_embd",
    "n_layer",
    "n_head",
    "train_CEM",
    "use_rtg",
    "hidden_dim",
    "dropout",
    "lr",
    "k",
    "l",
    "max_similarity",
    "weight",
    "ind_weight",
    "action_selector",
    "agent_output_type",
    "common_reward",
    "train_mode",
    "batch_size",
    "task_nums",
    "trajs_per_task",
    "max_return",
    "min_return",
    "env",
    "env_args.manifest",
    "env_args.time_limit",
    "baseline_padding_mode",
    "baseline_data_parallel",
    "baseline_parallel_devices",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run original MAICC (e1640a9 logic) on a strict TABX baseline "
            "manifest while leaving the upstream source files untouched."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--load-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help=(
            "0 uses the byte-for-byte upstream learner. A positive value "
            "uses compute-equivalent gradient accumulation while retaining "
            "the effective --batch-size."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable activation checkpointing without changing the model graph.",
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help=(
            "Split each global batch over every CUDA device visible through "
            "CUDA_VISIBLE_DEVICES, then perform one weighted global loss and "
            "one Adam update."
        ),
    )
    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="extra_updates",
        metavar="KEY=VALUE",
        help="Additional non-protected Sacred config update.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _update(key: str, value) -> str:
    if isinstance(value, str):
        encoded = json.dumps(value)
    elif isinstance(value, bool):
        encoded = "True" if value else "False"
    else:
        encoded = str(value)
    return f"{key}={encoded}"


def _validate_extra_updates(values: List[str]) -> None:
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--set must be KEY=VALUE, got {value!r}")
        key = value.split("=", 1)[0]
        if key in PROTECTED_KEYS or key.startswith("model."):
            raise SystemExit(
                f"--set cannot override protected baseline key {key!r}"
            )


def build_sacred_argv(cli: argparse.Namespace, manifest: BaselineManifest):
    data_parallel = bool(getattr(cli, "data_parallel", False))
    parallel_devices = int(getattr(cli, "parallel_devices", 1))
    if cli.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if cli.micro_batch_size < 0:
        raise SystemExit("--micro-batch-size cannot be negative")
    if cli.micro_batch_size > cli.batch_size:
        raise SystemExit("--micro-batch-size cannot exceed --batch-size")
    if data_parallel and cli.micro_batch_size > 0:
        raise SystemExit(
            "--data-parallel cannot be combined with --micro-batch-size; "
            "use the original global batch directly"
        )
    if data_parallel and cli.device == "cpu":
        raise SystemExit("--data-parallel requires --device cuda or auto")
    if data_parallel and parallel_devices < 2:
        raise SystemExit(
            "--data-parallel requires at least two visible CUDA devices"
        )
    if cli.stage == 1 and cli.checkpoint_path is None:
        raise SystemExit("--checkpoint-path is required for stage 1")
    _validate_extra_updates(cli.extra_updates)

    if manifest.allows_terminal_padding and data_parallel:
        execution = "padding_compatible_data_parallel"
    elif data_parallel:
        execution = "data_parallel_equivalent"
    elif manifest.allows_terminal_padding:
        execution = (
            "padding_compatible_microbatch"
            if cli.micro_batch_size > 0
            else "padding_compatible_original"
        )
    else:
        execution = (
            "byte_exact_learner"
            if cli.micro_batch_size == 0
            else "compute_equivalent_microbatch"
        )

    updates = list(cli.extra_updates)
    updates.extend(
        [
            _update("seed", cli.seed),
            _update("env_args.manifest", str(manifest.path)),
            _update("env_args.time_limit", manifest.sequence_length),
            _update(
                "env_args.key",
                f"tabx_original_maicc_l{manifest.sequence_length}",
            ),
            _update("task_nums", manifest.task_nums),
            _update("trajs_per_task", manifest.trajs_per_task),
            _update("max_return", manifest.return_max),
            _update("min_return", manifest.return_min),
            _update("batch_size", cli.batch_size),
            _update("train_mode", cli.stage),
            _update("agent", "transformer"),
            _update("learner", "ic_learner"),
            _update("mac", "ic_mac"),
            _update("n_embd", 64),
            _update("n_layer", 8),
            _update("n_head", 8),
            _update("hidden_dim", 64),
            _update("dropout", 0.1),
            _update("lr", 0.0005),
            _update("train_CEM", True),
            _update("use_rtg", False),
            _update("k", 2),
            _update("l", 50),
            _update("max_similarity", 0.98),
            _update("weight", 0),
            _update("ind_weight", 0),
            _update("action_selector", "soft_policies"),
            _update("agent_output_type", "pi_logits"),
            _update("common_reward", True),
            _update(
                "baseline_micro_batch_size",
                cli.micro_batch_size,
            ),
            _update(
                "baseline_gradient_checkpointing",
                cli.gradient_checkpointing,
            ),
            _update("baseline_upstream_commit", "e1640a9"),
            _update(
                "baseline_manifest_sha256",
                manifest.fingerprint,
            ),
            _update(
                "baseline_execution",
                execution,
            ),
            _update(
                "baseline_padding_mode",
                manifest.padding_mode,
            ),
            _update("baseline_data_parallel", data_parallel),
            _update("baseline_parallel_devices", parallel_devices),
        ]
    )
    if cli.embedding_cache is not None:
        updates.append(
            _update(
                "env_args.embedding_cache",
                str(cli.embedding_cache.expanduser().resolve()),
            )
        )
    if cli.results_dir is not None:
        updates.append(
            _update(
                "local_results_path",
                str(cli.results_dir.expanduser().resolve()),
            )
        )
    if cli.t_max is not None:
        updates.append(_update("t_max", cli.t_max))
    if cli.device == "cpu":
        updates.append(_update("use_cuda", False))
    elif cli.device == "cuda":
        updates.append(_update("use_cuda", True))
    if cli.stage == 1:
        updates.extend(
            [
                _update(
                    "checkpoint_path",
                    str(cli.checkpoint_path.expanduser().resolve()),
                ),
                _update("load_step", cli.load_step),
            ]
        )

    return [
        str(REPO_ROOT / "src" / "main.py"),
        "--config=MAICC",
        "--env-config=tabx_baseline",
        "with",
        *updates,
    ]


def main() -> int:
    cli = parse_args()
    verify_upstream_core(REPO_ROOT)
    manifest = BaselineManifest(cli.manifest)
    manifest.validate_files()
    parallel_devices = 1
    if cli.data_parallel:
        if cli.device == "cpu":
            raise SystemExit("--data-parallel requires CUDA")
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--data-parallel requested but CUDA is unavailable")
        parallel_devices = int(torch.cuda.device_count())
        if parallel_devices < 2:
            raise SystemExit(
                "--data-parallel requires at least two visible CUDA devices; "
                "set CUDA_VISIBLE_DEVICES=0,1 (or more)"
            )
    cli.parallel_devices = parallel_devices
    sacred_argv = build_sacred_argv(cli, manifest)
    print(
        "[original-maicc-tabx] "
        f"commit=e1640a9 stage={cli.stage} tasks={manifest.task_nums} "
        f"trajs_per_task={manifest.trajs_per_task} "
        f"obs_dim={manifest.obs_dim} actions={manifest.action_dim} "
        f"max_agents={manifest.max_agents} "
        f"sequence_length={manifest.sequence_length} "
        f"padding_mode={manifest.padding_mode} "
        f"effective_batch={cli.batch_size} "
        f"micro_batch={cli.micro_batch_size or cli.batch_size} "
        f"data_parallel_devices={parallel_devices}",
        flush=True,
    )
    print(
        "[original-maicc-tabx] "
        "agent=transformer CEM=CustomGPT2Model layers=8 heads=8 "
        "train_CEM=True retrieval=original-InContextReplayBuffer",
        flush=True,
    )
    if cli.dry_run:
        print(" ".join(sacred_argv))
        return 0

    install(
        repo_root=REPO_ROOT,
        micro_batch_size=cli.micro_batch_size,
        gradient_checkpointing=cli.gradient_checkpointing,
        padding_compatible=manifest.allows_terminal_padding,
        data_parallel=cli.data_parallel,
    )
    import faiss
    import torch
    import transformers

    print(
        "[original-maicc-tabx] runtime "
        f"torch={torch.__version__} "
        f"transformers={transformers.__version__} "
        f"faiss={getattr(faiss, '__version__', 'unknown')}",
        flush=True,
    )
    sys.argv = sacred_argv
    runpy.run_path(str(REPO_ROOT / "src" / "main.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
