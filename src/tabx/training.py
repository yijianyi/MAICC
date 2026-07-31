from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .episode_adapter import TabxEpisodeSource
from .manifest import TabxManifest
from .normalization import ObservationNormalizer


class TabxTrainingError(RuntimeError):
    pass


class ConsoleStatLogger:
    def __init__(self) -> None:
        self.latest: Dict[str, float] = {}

    def log_stat(self, key: str, value: Any, step: int) -> None:
        numeric = float(value)
        self.latest[key] = numeric
        print(f"[step {step}] {key}={numeric:.8g}", flush=True)


def select_device(requested: str) -> str:
    import torch as th

    value = requested.lower()
    if value == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if value == "cuda" and not th.cuda.is_available():
        raise TabxTrainingError(
            "CUDA was requested, but torch.cuda.is_available() is False"
        )
    if value not in {"cpu", "cuda"}:
        raise TabxTrainingError(f"Unsupported device: {requested!r}")
    return value


def seed_everything(seed: int, device: str) -> None:
    import torch as th

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if device == "cuda":
        th.cuda.manual_seed_all(seed)


def build_maicc_args(
    config: Mapping[str, Any],
    manifest: TabxManifest,
    device: str,
) -> SimpleNamespace:
    model = dict(config.get("model", {}))
    training = dict(config.get("training", {}))
    retrieval = dict(config.get("retrieval", {}))
    max_agents = int(
        model.get("max_agents", manifest.max_agents)
    )
    if max_agents < manifest.max_agents:
        raise TabxTrainingError(
            f"model.max_agents={max_agents} is below manifest max_agents="
            f"{manifest.max_agents}"
        )
    max_return_value = training.get("max_return", "auto")
    max_return = (
        max(1.0, manifest.max_return)
        if str(max_return_value).lower() == "auto"
        else float(max_return_value)
    )
    min_return_value = training.get("min_return", "auto")
    min_return = (
        manifest.min_return
        if str(min_return_value).lower() == "auto"
        else float(min_return_value)
    )
    n_embd = int(model.get("n_embd", 64))
    n_head = int(model.get("n_head", 4))
    if n_embd % n_head != 0:
        raise TabxTrainingError(
            f"model.n_embd={n_embd} must be divisible by n_head={n_head}"
        )
    return SimpleNamespace(
        env="tabx",
        env_args={"time_limit": manifest.sequence_length},
        n_agents=max_agents,
        n_actions=manifest.action_dim,
        k=int(retrieval.get("k", 2)),
        l=int(retrieval.get("search_k", 32)),
        max_similarity=float(retrieval.get("max_similarity", 0.98)),
        action_selector="soft_policies",
        agent_output_type="pi_logits",
        agent=str(model.get("agent", "transformer")),
        obs_agent_id=False,
        n_embd=n_embd,
        n_layer=int(model.get("n_layer", 4)),
        n_head=n_head,
        dropout=float(model.get("dropout", 0.1)),
        hidden_dim=int(model.get("hidden_dim", 128)),
        use_rtg=bool(model.get("use_rtg", False)),
        train_CEM=bool(model.get("train_cem", False)),
        lr=float(training.get("learning_rate", 5e-4)),
        use_cuda=device == "cuda",
        use_amp=bool(training.get("use_amp", False)),
        amp_dtype=str(training.get("amp_dtype", "float16")),
        learner_log_interval=int(training.get("log_interval", 100)),
        grad_norm_clip=float(training.get("grad_norm_clip", 10.0)),
        max_return=max_return,
        min_return=min_return,
        batch_size=int(training.get("max_batch_size", 8)),
        device=device,
    )


def create_model_and_learner(
    config: Mapping[str, Any],
    manifest: TabxManifest,
    device: str,
    logger: Optional[ConsoleStatLogger] = None,
):
    from controllers.ic_controller import ICMAC
    from learners.ic_learner import ICLearner

    args = build_maicc_args(config, manifest, device)
    env_info = {
        "n_agents": args.n_agents,
        "n_actions": manifest.action_dim,
        "obs_shape": manifest.obs_dim,
    }
    actual_logger = ConsoleStatLogger() if logger is None else logger
    mac = ICMAC(env_info=env_info, buffer=None, args=args)
    learner = ICLearner(
        mac=mac,
        scheme={},
        logger=actual_logger,
        args=args,
    )
    if device == "cuda":
        learner.cuda()
    return args, mac, learner, actual_logger


def save_checkpoint(
    root: Path,
    step: int,
    learner,
    config: Mapping[str, Any],
    manifest: TabxManifest,
) -> Path:
    import torch as th

    checkpoint_dir = root / f"step_{step:09d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    learner.save_models(str(checkpoint_dir))
    th.save(
        {
            "step": int(step),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": th.get_rng_state(),
            "cuda_random_state": (
                th.cuda.get_rng_state_all() if th.cuda.is_available() else None
            ),
        },
        str(checkpoint_dir / "trainer_state.th"),
    )
    with (checkpoint_dir / "config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(dict(config), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    with (checkpoint_dir / "manifest_path.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(str(manifest.path) + "\n")
    latest = root / "latest.txt"
    temp = root / ".latest.txt.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(str(checkpoint_dir) + "\n")
    temp.replace(latest)
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: Path, learner, device: str) -> int:
    import torch as th

    checkpoint = checkpoint_dir.expanduser().resolve()
    if not (checkpoint / "agent.th").is_file():
        raise TabxTrainingError(f"Missing agent.th in {checkpoint}")
    learner.load_models(str(checkpoint))
    state_path = checkpoint / "trainer_state.th"
    if not state_path.is_file():
        return 0
    state = th.load(str(state_path), map_location="cpu")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    th.set_rng_state(state["torch_random_state"])
    if (
        device == "cuda"
        and state.get("cuda_random_state") is not None
    ):
        th.cuda.set_rng_state_all(state["cuda_random_state"])
    return int(state["step"])


def run_stage0(
    config: Mapping[str, Any],
    manifest_path: Path,
    normalization_path: Optional[Path],
    requested_device: str,
    resume: Optional[Path] = None,
) -> Path:
    import torch as th

    manifest = TabxManifest(manifest_path)
    device = select_device(requested_device)
    training = dict(config.get("training", {}))
    data = dict(config.get("data", {}))
    seed = int(training.get("seed", 7))
    seed_everything(seed, device)
    normalizer = ObservationNormalizer.load(
        normalization_path,
        manifest.obs_dim,
        clip=float(data.get("normalization_clip", 10.0)),
        preserve_zeros=bool(data.get("preserve_zero_padding", True)),
    )
    args, _mac, learner, _logger = create_model_and_learner(
        config, manifest, device
    )

    output_dir = Path(str(training.get("output_dir", "results/tabx/stage0")))
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(training.get("total_steps", 100000))
    save_interval = int(training.get("save_interval", 10000))
    max_batch_size = int(training.get("max_batch_size", 8))
    agent_time_budget = int(training.get("agent_time_budget", 8192))
    h5_lru_size = int(data.get("h5_lru_size", 16))

    start_step = 0
    if resume is not None:
        start_step = load_checkpoint(resume, learner, device)
        print(f"[resume] checkpoint={resume} step={start_step}", flush=True)

    print(
        "[stage0] "
        f"tasks={len(manifest.tasks_for_split('train'))} "
        f"obs_dim={manifest.obs_dim} actions={manifest.action_dim} "
        f"max_agents={manifest.max_agents} sequence_length="
        f"{manifest.sequence_length} device={device} "
        f"train_CEM={args.train_CEM}",
        flush=True,
    )

    last_checkpoint = output_dir
    source = TabxEpisodeSource(
        manifest=manifest,
        split="train",
        normalizer=normalizer,
        seed=seed,
        h5_lru_size=h5_lru_size,
    )
    try:
        for step in range(start_step + 1, total_steps + 1):
            sampled = source.sample(
                max_batch_size=max_batch_size,
                agent_time_budget=agent_time_budget,
            )
            batch = sampled.batch
            if device == "cuda":
                batch.to(device)
            learner.train(
                batch=batch,
                t_env=step,
                episode_num=step,
                buffer=None,
                train_mode=0,
            )
            if step == 1 or step % int(training.get("progress_interval", 100)) == 0:
                print(
                    "[progress] "
                    f"step={step}/{total_steps} task={sampled.task.task_id} "
                    f"agents={sampled.task.n_agents} batch={batch.batch_size}",
                    flush=True,
                )
            if save_interval > 0 and step % save_interval == 0:
                last_checkpoint = save_checkpoint(
                    output_dir, step, learner, config, manifest
                )
                print(f"[checkpoint] {last_checkpoint}", flush=True)
        if save_interval <= 0 or total_steps % save_interval != 0:
            last_checkpoint = save_checkpoint(
                output_dir, total_steps, learner, config, manifest
            )
    finally:
        source.close()
    return last_checkpoint


def run_stage1(
    config: Mapping[str, Any],
    manifest_path: Path,
    normalization_path: Path,
    index_manifest_path: Path,
    stage0_checkpoint: Path,
    requested_device: str,
    resume: Optional[Path] = None,
) -> Path:
    from .retrieval_store import TabxRetrievalStore

    manifest = TabxManifest(manifest_path)
    device = select_device(requested_device)
    training = dict(config.get("training", {}))
    data = dict(config.get("data", {}))
    retrieval = dict(config.get("retrieval", {}))
    seed = int(training.get("seed", 7))
    seed_everything(seed, device)
    normalizer = ObservationNormalizer.load(
        normalization_path,
        manifest.obs_dim,
        clip=float(data.get("normalization_clip", 10.0)),
        preserve_zeros=bool(data.get("preserve_zero_padding", True)),
    )
    args, mac, learner, _logger = create_model_and_learner(
        config, manifest, device
    )
    output_dir = Path(str(training.get("output_dir", "results/tabx/stage1")))
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if resume is not None:
        start_step = load_checkpoint(resume, learner, device)
        print(f"[resume] checkpoint={resume} step={start_step}", flush=True)
    else:
        checkpoint = stage0_checkpoint.expanduser().resolve()
        mac.load_models(str(checkpoint))
        print(f"[stage1] loaded stage-0 agent from {checkpoint}", flush=True)

    total_steps = int(training.get("total_steps", 100000))
    save_interval = int(training.get("save_interval", 10000))
    max_batch_size = int(training.get("max_batch_size", 4))
    agent_time_budget = int(training.get("agent_time_budget", 4096))
    source = TabxEpisodeSource(
        manifest=manifest,
        split="train",
        normalizer=normalizer,
        seed=seed,
        h5_lru_size=int(data.get("h5_lru_size", 16)),
    )
    store = TabxRetrievalStore(
        index_manifest=index_manifest_path,
        data_manifest=manifest,
        normalizer=normalizer,
        k=args.k,
        search_k=args.l,
        device=device,
        seed=seed,
        h5_lru_size=int(data.get("h5_lru_size", 16)),
    )
    print(
        "[stage1] "
        f"tasks={len(source.tasks)} k={args.k} search_k={args.l} "
        f"sequence_length={manifest.sequence_length} device={device}",
        flush=True,
    )
    last_checkpoint = output_dir
    try:
        for step in range(start_step + 1, total_steps + 1):
            sampled = source.sample(
                max_batch_size=max_batch_size,
                agent_time_budget=agent_time_budget,
            )
            store.begin_query_batch(sampled.task.task_id, sampled.rows)
            batch = sampled.batch
            if device == "cuda":
                batch.to(device)
            learner.train(
                batch=batch,
                t_env=step,
                episode_num=step,
                buffer=store,
                train_mode=1,
            )
            if step == 1 or step % int(training.get("progress_interval", 100)) == 0:
                print(
                    "[progress] "
                    f"step={step}/{total_steps} task={sampled.task.task_id} "
                    f"agents={sampled.task.n_agents} batch={batch.batch_size}",
                    flush=True,
                )
            if save_interval > 0 and step % save_interval == 0:
                last_checkpoint = save_checkpoint(
                    output_dir, step, learner, config, manifest
                )
                print(f"[checkpoint] {last_checkpoint}", flush=True)
        if save_interval <= 0 or total_steps % save_interval != 0:
            last_checkpoint = save_checkpoint(
                output_dir, total_steps, learner, config, manifest
            )
    finally:
        source.close()
        store.close()
    return last_checkpoint
