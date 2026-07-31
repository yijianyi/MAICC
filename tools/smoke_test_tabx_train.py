#!/usr/bin/env python3
"""Run one real MAICC stage-0 optimiser step on converted TABX data.

This is deliberately isolated from InContextReplayBuffer: one converted HDF5
row is loaded into the project's EpisodeBatch and passed through the real
ICMAC/TransformerAgent/ICLearner training path.  It is a compatibility smoke
test, not an end-to-end experiment or a quality evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch as th


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from components.episode_buffer import EpisodeBatch  # noqa: E402
from controllers.ic_controller import ICMAC  # noqa: E402
from learners.ic_learner import ICLearner  # noqa: E402


class CaptureLogger:
    """Small logger implementing the interface used by ICLearner.train()."""

    def __init__(self) -> None:
        self.stats: dict[str, float] = {}

    def log_stat(self, key: str, value: Any, _t: int) -> None:
        numeric = float(value)
        self.stats[key] = numeric
        print(f"[loss] {key}={numeric:.8g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one MAICC stage-0 train step on converted TABX data."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/tabx_smoke"),
        help="Directory containing manifest.json and converted HDF5 files.",
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--transitions",
        type=int,
        default=4,
        help="Number of action transitions used by the smoke step.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--train-cem",
        action="store_true",
        help="Also train the larger centralized embedding model path.",
    )
    return parser.parse_args()


def load_manifest(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run tools/convert_tabx_to_maicc.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") not in {
        "maicc_tabx_manifest_v1",
        "maicc_tabx_manifest_v2",
    }:
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema')!r}")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest contains no converted tasks")
    return manifest, tasks


def build_batch(
    h5_path: Path,
    row: int,
    transitions: int,
    n_agents: int,
    obs_dim: int,
) -> EpisodeBatch:
    required = ("obs", "actions", "reward", "terminated", "filled")
    with h5py.File(h5_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"{h5_path} is missing datasets: {missing}")

        n_rows, time_size, file_agents, file_obs_dim = handle["obs"].shape
        if not 0 <= row < n_rows:
            raise IndexError(f"--row {row} is outside [0, {n_rows - 1}]")
        if (file_agents, file_obs_dim) != (n_agents, obs_dim):
            raise ValueError(
                "Manifest/HDF5 observation shape mismatch: "
                f"manifest={(n_agents, obs_dim)}, file={(file_agents, file_obs_dim)}"
            )
        if transitions < 2:
            raise ValueError("--transitions must be at least 2 for next-observation loss")
        if transitions + 1 > time_size:
            raise ValueError(
                f"Requested {transitions} transitions, but file only has "
                f"{time_size} time slots"
            )

        sl = np.s_[row : row + 1, : transitions + 1]
        arrays = {key: np.asarray(handle[key][sl]) for key in required}

    expected_shapes = {
        "obs": (1, transitions + 1, n_agents, obs_dim),
        "actions": (1, transitions + 1, n_agents, 1),
        "reward": (1, transitions + 1, 1),
        "terminated": (1, transitions + 1, 1),
        "filled": (1, transitions + 1, 1),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(
                f"Unexpected {key} shape {arrays[key].shape}; expected {expected}"
            )
    if not np.isfinite(arrays["obs"]).all() or not np.isfinite(arrays["reward"]).all():
        raise ValueError("obs/reward contains NaN or Inf")
    if not np.all(arrays["filled"][0, : transitions, 0] == 1):
        raise ValueError(
            "The selected prefix contains unfilled transitions; choose a smaller "
            "--transitions value or another --row"
        )

    scheme = {
        "obs": {"vshape": obs_dim, "group": "agents", "dtype": th.float32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "reward": {"vshape": (1,), "dtype": th.float32},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    batch = EpisodeBatch(
        scheme=scheme,
        groups={"agents": n_agents},
        batch_size=1,
        max_seq_length=transitions + 1,
        device="cpu",
    )
    batch.update(
        {
            "obs": th.as_tensor(arrays["obs"], dtype=th.float32),
            "actions": th.as_tensor(arrays["actions"], dtype=th.long),
            "reward": th.as_tensor(arrays["reward"], dtype=th.float32),
            "terminated": th.as_tensor(arrays["terminated"], dtype=th.uint8),
            "filled": th.as_tensor(arrays["filled"], dtype=th.long),
        },
        mark_filled=False,
    )
    return batch


def main() -> int:
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    th.manual_seed(cli.seed)

    if cli.device == "cuda":
        if not th.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")
        th.cuda.manual_seed_all(cli.seed)
    if cli.n_embd % cli.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")

    dataset_dir = cli.dataset_dir.resolve()
    _, tasks = load_manifest(dataset_dir)
    if not 0 <= cli.task_index < len(tasks):
        raise IndexError(
            f"--task-index {cli.task_index} is outside [0, {len(tasks) - 1}]"
        )
    task = tasks[cli.task_index]
    n_agents = int(task["n_agents"])
    n_actions = int(task["action_dim"])
    obs_dim = int(task["obs_dim"])
    h5_path = dataset_dir / task["h5_file"]
    if not h5_path.is_file():
        raise FileNotFoundError(f"Converted task file does not exist: {h5_path}")

    batch = build_batch(
        h5_path=h5_path,
        row=cli.row,
        transitions=cli.transitions,
        n_agents=n_agents,
        obs_dim=obs_dim,
    )
    actions = batch["actions"][:, : cli.transitions]
    action_min, action_max = int(actions.min()), int(actions.max())
    if action_min < 0 or action_max >= n_actions:
        raise ValueError(
            f"Action IDs [{action_min}, {action_max}] are outside [0, {n_actions - 1}]"
        )

    # env='tabx' is intentional.  env='sc2v2' would force ICMAC's input width
    # to 227, which is incompatible with this task's 342-dimensional obs.
    args = SimpleNamespace(
        env="tabx",
        env_args={"time_limit": cli.transitions},
        n_agents=n_agents,
        n_actions=n_actions,
        k=1,
        action_selector="soft_policies",
        agent_output_type="pi_logits",
        agent="transformer",
        obs_agent_id=False,
        n_embd=cli.n_embd,
        n_layer=cli.n_layer,
        n_head=cli.n_head,
        dropout=cli.dropout,
        hidden_dim=cli.hidden_dim,
        use_rtg=False,
        train_CEM=cli.train_cem,
        lr=cli.lr,
        use_cuda=(cli.device == "cuda"),
        learner_log_interval=0,
        max_return=max(1.0, float(batch["reward"][:, : cli.transitions].sum().item())),
    )
    env_info = {
        "n_agents": n_agents,
        "n_actions": n_actions,
        "obs_shape": obs_dim,
    }
    logger = CaptureLogger()
    mac = ICMAC(env_info=env_info, buffer=None, args=args)
    learner = ICLearner(mac=mac, scheme=batch.scheme, logger=logger, args=args)
    if cli.device == "cuda":
        learner.cuda()
        batch.to("cuda")

    before = {
        name: parameter.detach().clone()
        for name, parameter in mac.agent.named_parameters()
    }
    print(
        "[data] "
        f"task={task['task_id']!r} row={cli.row} transitions={cli.transitions} "
        f"agents={n_agents} actions={n_actions} obs_dim={obs_dim} "
        f"action_range=[{action_min}, {action_max}] device={cli.device}"
    )
    print(
        "[model] "
        f"n_embd={cli.n_embd} n_layer={cli.n_layer} n_head={cli.n_head} "
        f"train_CEM={cli.train_cem}"
    )

    # train_mode=0 is MAICC's embedding-model pretraining path.  It does not
    # invoke MIPS, online environments, checkpoints, or stage-1 DT training.
    learner.train(
        batch=batch,
        t_env=0,
        episode_num=0,
        buffer=None,
        train_mode=0,
    )

    if learner.training_steps != 1:
        raise RuntimeError(f"Expected one training step, got {learner.training_steps}")
    if not logger.stats:
        raise RuntimeError("Learner emitted no loss statistics")
    non_finite_losses = {
        key: value for key, value in logger.stats.items() if not math.isfinite(value)
    }
    if non_finite_losses:
        raise FloatingPointError(f"Non-finite losses: {non_finite_losses}")

    bad_gradients = []
    gradients_seen = 0
    changed_parameters = 0
    bad_parameters = []
    for name, parameter in mac.agent.named_parameters():
        if parameter.grad is not None:
            gradients_seen += 1
            if not th.isfinite(parameter.grad).all():
                bad_gradients.append(name)
        if not th.isfinite(parameter).all():
            bad_parameters.append(name)
        if not th.equal(before[name], parameter.detach()):
            changed_parameters += 1
    if gradients_seen == 0:
        raise RuntimeError("Backward produced no gradients")
    if bad_gradients:
        raise FloatingPointError(f"Non-finite gradients in: {bad_gradients}")
    if bad_parameters:
        raise FloatingPointError(f"Non-finite parameters in: {bad_parameters}")
    if changed_parameters == 0:
        raise RuntimeError("Optimizer step did not change any model parameter")

    print(
        f"[check] gradients_finite=yes parameters_finite=yes "
        f"changed_parameters={changed_parameters}"
    )
    print("SMOKE TEST PASSED: one real MAICC stage-0 optimiser step completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
