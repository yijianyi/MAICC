#!/usr/bin/env python3
"""Evaluate MAICC on the exact custom TABX tasks used by the dataset."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabx.teammate_modes import (
    load_teammate_modes, condition_metadata, add_baseline_deltas,
)
from tabx.config import load_yaml_config  # noqa: E402
from tabx.evaluation_protocol import resolve_return_target
from tabx.provenance import encoder_sha256, file_sha256, source_identity  # noqa: E402
from tabx.manifest import TabxManifest, TabxTask  # noqa: E402
from tabx.normalization import ObservationNormalizer  # noqa: E402
from tabx.online_wrapper import TabxOnlineWrapper  # noqa: E402
from tabx.retrieval_store import TabxRetrievalStore  # noqa: E402
from tabx.task_package import (  # noqa: E402
    TabxTaskDefinition,
    infer_raw_data_root,
    load_task_definitions,
)
from tabx.training import (  # noqa: E402
    create_model_and_learner,
    seed_everything,
    select_device,
)


DEFAULT_RAW_DATA = Path(
    "/mnt/cfs/shaopt/data/MARL/"
    "0803_overfit_7task_train_7000record_balance_winrate"
)


CONDITION_FIELDS = (
    "condition_id", "teammate_mode", "teammate_policies",
    "replaced_agent_ids", "replacement_count",
)
EPISODE_FIELDS = CONDITION_FIELDS + (
    "task_id", "episode", "seed", "win", "return", "length",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or all manifest tasks by restoring their exact "
            "custom scenario and zone_scenario from task.json."
        )
    )
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "src/config/tabx/tabx_stage1.yaml",
    )
    parser.add_argument("--teammate-config", type=Path,
                        help="YAML teammate_modes; omitted means all_model only")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--task-id", action="append", dest="task_ids",
        help="Evaluate only this manifest task. Repeatable; defaults to tasks in the configured split.",
    )
    parser.add_argument("--split", choices=("train", "validation", "val", "test", "all"), default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--tabx-root", type=Path, default=REPO_ROOT / "TABX")
    parser.add_argument("--raw-data-root", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument(
        "--task-package-root", type=Path, action="append", default=[],
        help=(
            "Directory containing original task.json packages. Repeatable. "
            "Usually inferred from record meta oracle.config."
        ),
    )
    parser.add_argument("--max-allies", type=int, default=None)
    parser.add_argument("--max-enemies", type=int, default=None)
    parser.add_argument("--max-zones", type=int, default=None)
    parser.add_argument("--episode-limit", type=int, default=512)
    parser.add_argument("--disable-online-memory", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Keep fully completed task/mode conditions and restart only "
            "the interrupted condition with empty online memory."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "/mnt/cfs/shaopt/data/MARL/maicc_results/0803_online_eval"
        ),
    )
    return parser.parse_args()


def _jax_object(cls, value: Mapping[str, Any]):
    import jax.numpy as jnp

    def convert(item):
        array = jnp.asarray(item)
        # TABX's JSON loader represents scalar parameters (notably n_zone) as
        # [1, 1], while ordinary list-valued fields retain their natural shape.
        return array.reshape(1, 1) if array.ndim == 0 else array

    return cls(**{key: convert(item) for key, item in value.items()})


def resolve_tabx_root(requested: Path) -> Path:
    """Return the directory that directly contains TABX's ``src`` package."""
    supplied = requested.expanduser().resolve()
    candidates = [
        supplied,
        supplied.parent if supplied.name.lower() == "src" else supplied,
        REPO_ROOT / "TABX",
        REPO_ROOT.parent / "TABX",
    ]
    # Also support a renamed TABX checkout directly below MAICC without
    # recursively walking the very large local dataset directories.
    try:
        candidates.extend(
            child for child in REPO_ROOT.iterdir()
            if child.is_dir() and (child / "src" / "tabx" / "__init__.py").is_file()
        )
    except OSError:
        pass
    checked = []
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if root in checked:
            continue
        checked.append(root)
        if (root / "src" / "tabx" / "__init__.py").is_file():
            return root
        # Accept --tabx-root=/path/to/TABX/src as well.
        if (root / "tabx" / "__init__.py").is_file() and root.name == "src":
            return root.parent
    locations = "\n  - ".join(str(path) for path in checked)
    raise RuntimeError(
        "Cannot locate the TABX source package. Expected "
        "<tabx-root>/src/tabx/__init__.py. Checked:\n  - " + locations
        + "\nPass the actual checkout path with --tabx-root."
    )


def build_environment(
    tabx_root: Path,
    definition: TabxTaskDefinition,
    seed: int,
    max_allies: int,
    max_enemies: int,
    max_zones: int,
    episode_limit: int,
) -> TabxOnlineWrapper:
    # TABX imports live under its repository's `src` package. Put its root
    # first to avoid collision with MAICC/src.
    resolved_root = resolve_tabx_root(tabx_root)
    root = str(resolved_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    # A failed earlier import may leave a partial namespace cached. Clear it
    # before retrying after adding the verified checkout root.
    importlib.invalidate_caches()
    try:
        from src.tabx import TABX
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            f"TABX exists at {resolved_root}, but importing src.tabx failed. "
            f"Python={sys.version.split()[0]}; error={exc}. Current TABX "
            "requires Python >=3.10 and its JAX dependencies."
        ) from exc
    from src.tabx.config import TABXConfig
    from src.tabx.heuristic_policy import build_batched_heuristic_params
    from src.tabx.physics import build_batched_physics_params
    from src.tabx.scenarios.scenario import VectorizedScenario, ZoneScenario
    from src.tabx.scenarios.utils import (
        generate_padded_unit_scenario,
        generate_padded_zone_scenario,
    )
    from src.tabx.wrappers.wrappers import TABXEnemyHeuristicWrapper

    raw_scenario = _jax_object(VectorizedScenario, definition.task["scenario"])
    raw_zones = _jax_object(ZoneScenario, definition.task["zone_scenario"])
    teams = np.asarray(definition.task["scenario"]["teams"]).reshape(-1)
    actual_allies = int(np.count_nonzero(teams == 0))
    actual_enemies = int(np.count_nonzero(teams == 1))
    actual_zones = int(
        np.asarray(definition.task["zone_scenario"]["n_zone"]).reshape(-1)[0]
    )
    if actual_allies > max_allies or actual_enemies > max_enemies:
        raise RuntimeError(
            f"Task {definition.task_id} has {actual_allies} allies and "
            f"{actual_enemies} enemies, exceeding configured capacities "
            f"{max_allies}/{max_enemies}"
        )
    if actual_zones > max_zones:
        raise RuntimeError(
            f"Task {definition.task_id} has {actual_zones} zones, exceeding "
            f"configured capacity {max_zones}"
        )
    scenario = generate_padded_unit_scenario(
        raw_scenario, max_allies, max_enemies
    )
    zone_scenario = generate_padded_zone_scenario(raw_zones, max_zones)
    env_params = {
        "scenario": scenario,
        "zone_scenario": zone_scenario,
        "physics_params": build_batched_physics_params(
            definition.physics_name, n_repeat=1,
            squeeze_when_single_physics=True,
        ),
        "heuristic_params": build_batched_heuristic_params(
            definition.enemy_heuristic_name, n_repeat=1,
            squeeze_when_single_heuristic=True,
        ),
    }
    config = TABXConfig(
        max_n_ally=max_allies,
        max_n_enemy=max_enemies,
        max_n_zone=max_zones,
    )
    env = TABXEnemyHeuristicWrapper(
        TABX(
            cfg=config,
            world_state_type="concat",
            max_episode_steps=episode_limit,
        )
    )
    return TabxOnlineWrapper(env, env_params, seed=seed)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_episode(
    path: Path,
    task_id: str,
    episode: int,
    seed: int,
    win: bool,
    episode_return: float,
    length: int,
    condition: Mapping[str, Any],
) -> None:
    fields = EPISODE_FIELDS
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                **condition,
                "task_id": task_id,
                "episode": episode,
                "seed": seed,
                "win": int(win),
                "return": f"{episode_return:.9g}",
                "length": length,
            }
        )
        handle.flush()


def assert_finite_observation(
    label: str, values: np.ndarray, present: np.ndarray
) -> None:
    array = np.asarray(values)
    invalid = ~np.isfinite(array)
    if not bool(invalid.any()):
        return
    invalid_by_agent = invalid.reshape(array.shape[0], -1).sum(axis=1)
    invalid_agents = np.flatnonzero(invalid_by_agent).tolist()
    invalid_features = np.flatnonzero(
        invalid.reshape(array.shape[0], -1).any(axis=0)
    )
    present_invalid = [
        int(agent_id) for agent_id in invalid_agents
        if bool(np.asarray(present).reshape(-1)[agent_id])
    ]
    raise RuntimeError(
        f"Non-finite {label}: shape={array.shape}, "
        f"invalid_values={int(invalid.sum())}, "
        f"invalid_agents={invalid_agents}, "
        f"present_invalid_agents={present_invalid}, "
        f"invalid_counts={invalid_by_agent[invalid_by_agent > 0].tolist()}, "
        f"invalid_feature_count={int(invalid_features.size)}, "
        f"invalid_features_first64={invalid_features[:64].tolist()}"
    )


def evaluate_task(
    task: TabxTask,
    definition: TabxTaskDefinition,
    env: TabxOnlineWrapper,
    mac,
    store: TabxRetrievalStore,
    normalizer: ObservationNormalizer,
    device: str,
    episodes: int,
    seed_start: int,
    episode_csv: Path,
    condition: Mapping[str, Any],
) -> Dict[str, Any]:
    import torch as th
    from components.episode_buffer import EpisodeBatch

    env_info = env.get_env_info()
    expected = (task.n_agents, task.obs_dim, task.action_dim)
    actual = (
        env_info["n_agents"], env_info["obs_shape"], env_info["n_actions"]
    )
    if actual != expected:
        raise RuntimeError(
            f"Environment/manifest mismatch for {task.task_id}: "
            f"env={actual}, manifest={expected}"
        )

    scheme = {
        "obs": {
            "vshape": task.obs_dim, "group": "agents", "dtype": th.float32
        },
        "actions": {
            "vshape": (1,), "group": "agents", "dtype": th.long
        },
        "avail_actions": {
            "vshape": (task.action_dim,), "group": "agents", "dtype": th.int
        },
        "present": {
            "vshape": (1,), "group": "agents", "dtype": th.uint8
        },
        "reward": {"vshape": (1,), "dtype": th.float32},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "completion": {"vshape": (1,), "dtype": th.uint8},
    }
    wins = []
    episode_returns = []
    lengths = []
    for episode in range(episodes):
        raw_obs = env.reset()
        print(
            f"[episode-start] task={task.task_id} mode={condition['teammate_mode']} "
            f"episode={episode + 1}/{episodes} "
            f"present_agents={int(np.count_nonzero(env.present))}/"
            f"{env.max_allies}",
            flush=True,
        )
        mac.init_hidden(1)
        batch = EpisodeBatch(
            scheme, {"agents": task.n_agents}, 1,
            env_info["episode_limit"] + 1, device=device,
        )
        obs_history = []
        action_history = []
        reward_history = []
        term_history = []
        completion_history = []
        done = False
        timestep = 0
        total_return = 0.0
        final_win = False
        while not done and timestep < env_info["episode_limit"]:
            assert_finite_observation(
                f"raw observation at timestep={timestep}",
                raw_obs,
                env.present,
            )
            normalized_obs = normalizer.apply(raw_obs)
            assert_finite_observation(
                f"normalized observation at timestep={timestep}",
                normalized_obs,
                env.present,
            )
            avail = env.get_avail_actions()
            batch.update(
                {
                    "obs": th.as_tensor(normalized_obs[None], device=device),
                    "avail_actions": th.as_tensor(avail[None], device=device),
                    "present": th.as_tensor(
                        env.present[None, :, None], device=device
                    ),
                },
                ts=timestep,
            )
            with th.no_grad():
                chosen = mac.select_actions(
                    batch,
                    t_ep=timestep,
                    t_env=episode / max(1, episodes),
                    test_mode=True,
                )[0]
            action_np = chosen.detach().cpu().numpy()
            reward, done, info = env.step(action_np)
            action_np = env.last_executed_actions.copy()
            executed = th.as_tensor(action_np, dtype=th.long, device=device)
            final_win = bool(info.get("is_win", False))
            completion = bool(done and final_win)
            batch.update(
                {
                    "actions": executed[None, :, None],
                    "reward": [(reward,)],
                    "terminated": [(done,)],
                    "completion": [(completion,)],
                },
                ts=timestep,
            )
            mac.update_context(batch, timestep)
            obs_history.append(normalized_obs)
            action_history.append(action_np)
            reward_history.append(reward)
            term_history.append(done)
            completion_history.append(completion)
            total_return += reward
            raw_obs = env.get_obs()
            timestep += 1

        rewards = np.asarray(reward_history, dtype=np.float32)
        normalized_rtg = np.cumsum(rewards[::-1], dtype=np.float64)[::-1]
        normalized_rtg = normalized_rtg.astype(np.float32) / mac.args.online_return_scale
        store.add_completed_online_episode(
            mac.agent,
            np.asarray(obs_history, dtype=np.float32),
            np.asarray(action_history, dtype=np.int64),
            rewards,
            np.asarray(term_history, dtype=np.float32),
            np.asarray(completion_history, dtype=np.float32),
            env.present,
            normalized_rtg,
        )
        wins.append(final_win)
        episode_returns.append(total_return)
        lengths.append(timestep)
        append_episode(
            episode_csv, task.task_id, episode + 1,
            seed_start + episode, final_win, total_return, timestep, condition,
        )
        print(
            f"[online] task={task.task_id} mode={condition['teammate_mode']} episode={episode + 1}/{episodes} "
            f"win={int(final_win)} running_win_rate={np.mean(wins):.4f} "
            f"return={total_return:.6f} length={timestep} "
            f"offline_p={store.offline_probability:.4f}",
            flush=True,
        )

    return {
        **condition,
        "task_id": task.task_id,
        "task_package": str(definition.path),
        "physics": definition.physics_name,
        "enemy_heuristic": definition.enemy_heuristic_name,
        "episodes": episodes,
        "seed_start": seed_start,
        "seed_end": seed_start + episodes - 1,
        "wins": int(np.sum(wins)),
        "losses": episodes - int(np.sum(wins)),
        "win_rate": float(np.mean(wins)),
        "mean_return": float(np.mean(episode_returns)),
        "mean_length": float(np.mean(lengths)),
    }


def write_summary_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    fields = CONDITION_FIELDS + (
        "delta_win_rate", "delta_mean_return", "losses",
        "task_id", "episodes", "wins", "win_rate", "mean_return",
        "mean_length", "seed_start", "seed_end", "physics",
        "enemy_heuristic", "task_package",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(path)


def prepare_resume(
    output_dir: Path,
    conditions: Sequence[Mapping[str, Any]],
    episodes: int,
    seed: int,
    enabled: bool,
    run_spec: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], set[str]]:
    episode_csv = output_dir / "episodes.csv"
    summary_json = output_dir / "summary.json"
    spec_path = output_dir / "run_spec.json"
    if spec_path.exists():
        if not enabled:
            raise SystemExit("Output exists; use --resume or a new --output-dir")
        if json.loads(spec_path.read_text()) != run_spec:
            raise SystemExit("Resume configuration/checkpoint mismatch; use a new --output-dir")
    else:
        if episode_csv.exists() or summary_json.exists():
            raise SystemExit("Legacy output has no run_spec.json; use a new --output-dir")
        atomic_write_json(spec_path, run_spec)
    previous = json.loads(summary_json.read_text()).get("tasks", []) if summary_json.exists() else []
    expected = {c["condition_id"]: c for c in conditions}
    results = []
    for item in previous:
        cid = item.get("condition_id")
        if cid in expected and all(item.get(k) == v for k, v in expected[cid].items()):
            if item.get("episodes") == episodes and item.get("seed_start") == seed:
                results.append(dict(item))
    if len({r["condition_id"] for r in results}) != len(results):
        raise SystemExit("Duplicate completed conditions in summary.json")
    rows = []
    if episode_csv.exists():
        with episode_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
                raise SystemExit("Unexpected episodes.csv schema")
            rows = list(reader)
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition_id"], []).append(row)
    completed = set()
    for result in results:
        cid = result["condition_id"]
        records = by_condition.get(cid, [])
        if len(records) == episodes and all(
            int(row["episode"]) == i + 1 and int(row["seed"]) == seed + i
            and row["task_id"] == result["task_id"]
            and all(row[k] == str(v) for k, v in expected[cid].items())
            for i, row in enumerate(records)
        ):
            completed.add(cid)
    results = [r for r in results if r["condition_id"] in completed]
    # Discard interrupted conditions: their memory is reconstructed by replaying
    # the whole condition, never by continuing with a fresh empty store midway.
    if episode_csv.exists():
        temporary = episode_csv.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            writer.writerows(r for r in rows if r["condition_id"] in completed)
        temporary.replace(episode_csv)
    return results, completed


def main() -> int:
    cli = parse_args()
    if cli.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if cli.seed < 0 or cli.episode_limit <= 0:
        raise SystemExit("seed must be nonnegative and episode-limit positive")
    modes = load_teammate_modes(cli.teammate_config)
    config = load_yaml_config(cli.config)
    data_cfg = dict(config["data"])
    retrieval_cfg = dict(config["retrieval"])
    training_cfg = dict(config.get("training", {}))
    evaluation_cfg = dict(config.get("evaluation", {}))
    manifest = TabxManifest(Path(str(data_cfg["manifest"])))
    model_cfg = config.get("model", {})
    if cli.max_allies is None:
        cli.max_allies = int(model_cfg.get("max_agents", manifest.max_agents))
    if cli.max_enemies is None:
        cli.max_enemies = int(model_cfg.get("max_enemies", cli.max_allies))
    if cli.max_zones is None:
        cli.max_zones = int(model_cfg.get("n_zones", 4))
    tasks = (
        [manifest.task(task_id) for task_id in cli.task_ids]
        if cli.task_ids else list(manifest.tasks_for_split(
            getattr(cli, "split", None) or evaluation_cfg.get("task_split", "train")))
    )
    return_targets = {task.task_id: resolve_return_target(manifest, task, evaluation_cfg)
                      for task in tasks}
    raw_data_root = infer_raw_data_root(manifest, tasks, cli.raw_data_root)
    definitions = load_task_definitions(
        manifest=manifest,
        tasks=tasks,
        raw_data_root=raw_data_root,
        task_package_roots=cli.task_package_root,
    )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise SystemExit("Duplicate requested task IDs")
    # Validate all conditions before loading the checkpoint or starting rollouts.
    for task in tasks:
        scenario = definitions[task.task_id].task["scenario"]
        teams = np.asarray(scenario["teams"]).reshape(-1)
        disabled = np.asarray(scenario.get("is_disabled", np.zeros_like(teams))).reshape(-1)
        active = set(np.flatnonzero(~disabled[teams == 0].astype(bool)).tolist())
        for mode in modes:
            replaced = set(mode["policies"])
            if replaced and (not replaced < active or max(replaced) >= cli.max_allies):
                raise SystemExit(
                    f"Invalid mode {mode['name']} for {task.task_id}: "
                    f"replace={sorted(replaced)}, enabled allies={sorted(active)}; "
                    "retain at least one model ally"
                )
    conditions = [condition_metadata(t.task_id, m) for t in tasks for m in modes]
    output_dir = cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = cli.checkpoint.expanduser().resolve() / "agent.th"
    checkpoint_stat = checkpoint_file.stat()
    run_spec = json.loads(json.dumps({
        "schema": "tabx_teammates_v2", "config": config,
        "source_identity": source_identity(),
        "checkpoint_sha256": file_sha256(checkpoint_file),
        "index_manifest_sha256": file_sha256(retrieval_cfg["index_manifest"]),
        "checkpoint": str(checkpoint_file),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "conditions": conditions, "episodes": cli.episodes, "seed": cli.seed,
        "online_memory": not cli.disable_online_memory,
        "capacities": [cli.max_allies, cli.max_enemies, cli.max_zones],
        "episode_limit": cli.episode_limit,
        "tabx_root": str(resolve_tabx_root(cli.tabx_root)),
        "task_definitions": {t.task_id: definitions[t.task_id].task for t in tasks},
        "task_settings": {
            t.task_id: {
                "physics": definitions[t.task_id].physics_name,
                "enemy_heuristic": definitions[t.task_id].enemy_heuristic_name,
                **return_targets[t.task_id],
            } for t in tasks
        },
    }))
    results, completed_condition_ids = prepare_resume(
        output_dir, conditions, cli.episodes, cli.seed, cli.resume, run_spec
    )
    normalizer = ObservationNormalizer.load(
        Path(str(data_cfg["normalization"])),
        manifest.obs_dim,
        clip=float(data_cfg.get("normalization_clip", 10.0)),
        preserve_zeros=bool(data_cfg.get("preserve_zero_padding", True)),
        manifest_path=manifest.path,
    )
    device = select_device(str(training_cfg.get("device", "auto")))
    args, mac, _learner, _logger = create_model_and_learner(
        config, manifest, device
    )
    mac.load_models(str(cli.checkpoint.expanduser().resolve()))
    mac.agent.eval()
    import torch as th
    invalid_parameters = []
    with th.no_grad():
        for name, parameter in mac.agent.named_parameters():
            if parameter.is_floating_point() and not bool(
                th.isfinite(parameter).all().item()
            ):
                invalid_parameters.append(
                    (name, int((~th.isfinite(parameter)).sum().item()))
                )
    if invalid_parameters:
        raise RuntimeError(
            "Checkpoint contains non-finite parameters: "
            + ", ".join(
                f"{name}={count}" for name, count in invalid_parameters[:32]
            )
        )
    print("[model-check] all checkpoint parameters are finite", flush=True)
    store = TabxRetrievalStore(
        index_manifest=Path(str(retrieval_cfg["index_manifest"])),
        data_manifest=manifest,
        normalizer=normalizer,
        k=args.k,
        search_k=args.l,
        device=device,
        seed=cli.seed,
        h5_lru_size=int(data_cfg.get("h5_lru_size", 16)),
        enable_online_memory=not cli.disable_online_memory,
        test_nepisode=cli.episodes,
        index_load_mode=str(retrieval_cfg.get("index_load_mode", "ram")),
        faiss_threads=int(retrieval_cfg.get("faiss_threads", 4)),
        trace_search=bool(retrieval_cfg.get("trace_search", False)),
        weight=float(retrieval_cfg.get("weight", 0.8)),
        ind_weight=float(retrieval_cfg.get("ind_weight", 0.2)),
        max_similarity=args.max_similarity,
        index_cache_size=int(retrieval_cfg.get("index_cache_size", 2)),
        allow_approximate_index=bool(retrieval_cfg.get("allow_approximate_index", False)),
        use_utility=bool(retrieval_cfg.get("use_utility", True)),
        expected_encoder_sha256=encoder_sha256(mac.agent.state_dict()),
        normalization_path=Path(str(data_cfg["normalization"])),
    )
    mac.buffer = store

    episode_csv = output_dir / "episodes.csv"
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    def save_progress(status):
        add_baseline_deltas(results)
        atomic_write_json(summary_json, {
            "status": status,
            "requested_tasks": len(tasks),
            "requested_conditions": len(conditions),
            "completed_conditions": len(results),
            "episodes_per_condition": cli.episodes,
            "online_memory": not cli.disable_online_memory,
            "tasks": results,
        })
        write_summary_csv(summary_csv, results)

    try:
        save_progress("running")
        for task in tasks:
            for mode in modes:
                condition = condition_metadata(task.task_id, mode)
                if condition["condition_id"] in completed_condition_ids:
                    continue
                # Every condition starts from identical seeds and empty online memory.
                seed_start = cli.seed
                seed_everything(seed_start, device)
                store.reset_online_memory(seed=seed_start)
                target = return_targets[task.task_id]
                args.online_return_scale = target["return_scale"]
                args.online_target_rtg = target["target_return"] / target["return_scale"]
                definition = definitions[task.task_id]
                env = build_environment(
                    tabx_root=cli.tabx_root, definition=definition, seed=seed_start,
                    max_allies=cli.max_allies, max_enemies=cli.max_enemies,
                    max_zones=cli.max_zones, episode_limit=cli.episode_limit,
                )
                env.configure_teammates(mode["policies"])
                print(
                    f"[condition] task={task.task_id} mode={mode['name']} "
                    f"policies={mode['policies']} episodes={cli.episodes} "
                    f"seed_start={seed_start} memory=reset", flush=True,
                )
                result = evaluate_task(
                    task, definition, env, mac, store, normalizer, device,
                    cli.episodes, seed_start, episode_csv, condition,
                )
                results.append(result)
                rank = {c["condition_id"]: i for i, c in enumerate(conditions)}
                results.sort(key=lambda r: rank[r["condition_id"]])
                save_progress("running")
                print(
                    f"[condition-complete] task={task.task_id} mode={mode['name']} "
                    f"wins={result['wins']}/{cli.episodes} "
                    f"win_rate={result['win_rate']:.6f}", flush=True,
                )
    finally:
        store.close()

    total_episodes = sum(int(item["episodes"]) for item in results)
    total_wins = sum(int(item["wins"]) for item in results)
    final = {
        "status": "complete",
        "requested_tasks": len(tasks),
        "requested_conditions": len(conditions),
        "completed_conditions": len(results),
        "episodes_per_condition": cli.episodes,
        "total_episodes": total_episodes,
        "total_wins": total_wins,
        "overall_win_rate": total_wins / max(1, total_episodes),
        "macro_win_rate": float(
            np.mean([float(item["win_rate"]) for item in results])
        ),
        "online_memory": not cli.disable_online_memory,
        "tasks": results,
    }
    atomic_write_json(summary_json, final)
    write_summary_csv(summary_csv, results)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote episode results: {episode_csv}")
    print(f"Wrote task summary: {summary_csv}")
    print(f"Wrote full summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
