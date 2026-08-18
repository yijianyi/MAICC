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

from tabx.config import load_yaml_config  # noqa: E402
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--task-id", action="append", dest="task_ids",
        help="Evaluate only this manifest task. Repeatable; default is all tasks.",
    )
    parser.add_argument("--episodes", type=int, default=1000)
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
    parser.add_argument("--max-allies", type=int, default=10)
    parser.add_argument("--max-enemies", type=int, default=10)
    parser.add_argument("--max-zones", type=int, default=4)
    parser.add_argument("--episode-limit", type=int, default=512)
    parser.add_argument("--disable-online-memory", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Keep fully completed tasks from summary.json and restart only "
            "the interrupted task so its online memory remains consistent."
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
) -> None:
    fields = ("task_id", "episode", "seed", "win", "return", "length")
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
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
            f"[episode-start] task={task.task_id} "
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
            final_win = bool(info.get("is_win", False))
            completion = bool(done and final_win)
            batch.update(
                {
                    "actions": chosen[None, :, None],
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
        normalized_rtg = normalized_rtg.astype(np.float32) / task.return_scale
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
            seed_start + episode, final_win, total_return, timestep,
        )
        print(
            f"[online] task={task.task_id} episode={episode + 1}/{episodes} "
            f"win={int(final_win)} running_win_rate={np.mean(wins):.4f} "
            f"return={total_return:.6f} length={timestep} "
            f"offline_p={store.offline_probability:.4f}",
            flush=True,
        )

    return {
        "task_id": task.task_id,
        "task_package": str(definition.path),
        "physics": definition.physics_name,
        "enemy_heuristic": definition.enemy_heuristic_name,
        "episodes": episodes,
        "seed_start": seed_start,
        "seed_end": seed_start + episodes - 1,
        "wins": int(np.sum(wins)),
        "win_rate": float(np.mean(wins)),
        "mean_return": float(np.mean(episode_returns)),
        "mean_length": float(np.mean(lengths)),
    }


def write_summary_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    fields = (
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


def ordered_results(
    results: Sequence[Mapping[str, Any]], tasks: Sequence[TabxTask]
) -> list[Mapping[str, Any]]:
    rank = {task.task_id: index for index, task in enumerate(tasks)}
    return sorted(results, key=lambda item: rank[str(item["task_id"])])


def prepare_resume(
    output_dir: Path,
    tasks: Sequence[TabxTask],
    episodes: int,
    enabled: bool,
) -> tuple[list[Dict[str, Any]], set[str]]:
    episode_csv = output_dir / "episodes.csv"
    summary_json = output_dir / "summary.json"
    if not enabled:
        if episode_csv.exists() or summary_json.exists():
            raise SystemExit(
                f"Evaluation output already exists in {output_dir}. Use "
                "--resume to retain fully completed tasks, or choose a new "
                "--output-dir."
            )
        return [], set()
    if not summary_json.is_file():
        # No task reached an atomic completion point. Restart cleanly.
        if episode_csv.exists():
            episode_csv.unlink()
        return [], set()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    previous = payload.get("tasks", [])
    if not isinstance(previous, list):
        raise SystemExit(f"Invalid task list in {summary_json}")
    requested_ids = {task.task_id for task in tasks}
    results = [
        dict(item) for item in previous
        if isinstance(item, dict)
        and str(item.get("task_id")) in requested_ids
        and int(item.get("episodes", -1)) == episodes
    ]
    completed = {str(item["task_id"]) for item in results}
    if len(completed) != len(results):
        raise SystemExit(f"Duplicate completed tasks in {summary_json}")

    # Drop rows from a task that was interrupted before its atomic summary was
    # written. It will be rerun from episode 1 with an empty online memory.
    if episode_csv.is_file():
        with episode_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = [row for row in reader if row.get("task_id") in completed]
        expected_fields = (
            "task_id", "episode", "seed", "win", "return", "length"
        )
        if fields != expected_fields:
            raise SystemExit(f"Unexpected episode CSV schema in {episode_csv}")
        temporary = episode_csv.with_suffix(episode_csv.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=expected_fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(episode_csv)
    return results, completed


def main() -> int:
    cli = parse_args()
    if cli.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    config = load_yaml_config(cli.config)
    data_cfg = dict(config["data"])
    retrieval_cfg = dict(config["retrieval"])
    training_cfg = dict(config.get("training", {}))
    manifest = TabxManifest(Path(str(data_cfg["manifest"])))
    tasks = (
        [manifest.task(task_id) for task_id in cli.task_ids]
        if cli.task_ids else list(manifest.tasks)
    )
    raw_data_root = infer_raw_data_root(manifest, tasks, cli.raw_data_root)
    definitions = load_task_definitions(
        manifest=manifest,
        tasks=tasks,
        raw_data_root=raw_data_root,
        task_package_roots=cli.task_package_root,
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
        online_search_k=int(retrieval_cfg.get("online_search_k", 16)),
        device=device,
        seed=cli.seed,
        h5_lru_size=int(data_cfg.get("h5_lru_size", 16)),
        enable_online_memory=not cli.disable_online_memory,
        test_nepisode=cli.episodes,
        index_load_mode=str(retrieval_cfg.get("index_load_mode", "ram")),
        faiss_threads=int(retrieval_cfg.get("faiss_threads", 4)),
        trace_search=bool(retrieval_cfg.get("trace_search", False)),
    )
    mac.buffer = store

    output_dir = cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = output_dir / "episodes.csv"
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    results, completed_task_ids = prepare_resume(
        output_dir, tasks, cli.episodes, cli.resume
    )
    if completed_task_ids:
        print(
            "[resume] keeping completed tasks: "
            + ", ".join(sorted(completed_task_ids)),
            flush=True,
        )
    try:
        for task_number, task in enumerate(tasks):
            if task.task_id in completed_task_ids:
                continue
            # Each task owns a disjoint deterministic seed interval.
            seed_start = cli.seed + task_number * cli.episodes
            seed_everything(seed_start, device)
            store.reset_online_memory(seed=seed_start)
            args.online_return_scale = task.return_scale
            args.online_target_rtg = task.target_return / task.return_scale
            definition = definitions[task.task_id]
            env = build_environment(
                tabx_root=cli.tabx_root,
                definition=definition,
                seed=seed_start,
                max_allies=cli.max_allies,
                max_enemies=cli.max_enemies,
                max_zones=cli.max_zones,
                episode_limit=cli.episode_limit,
            )
            print(
                f"[task] {task_number + 1}/{len(tasks)} id={task.task_id} "
                f"package={definition.path} episodes={cli.episodes} "
                f"seed_start={seed_start}",
                flush=True,
            )
            result = evaluate_task(
                task, definition, env, mac, store, normalizer, device,
                cli.episodes, seed_start, episode_csv,
            )
            results.append(result)
            results = list(ordered_results(results, tasks))
            combined = {
                "status": "running",
                "requested_tasks": len(tasks),
                "completed_tasks": len(results),
                "episodes_per_task": cli.episodes,
                "online_memory": not cli.disable_online_memory,
                "tasks": results,
            }
            atomic_write_json(summary_json, combined)
            write_summary_csv(summary_csv, results)
            print(
                f"[task-complete] id={task.task_id} "
                f"wins={result['wins']}/{cli.episodes} "
                f"win_rate={result['win_rate']:.6f}",
                flush=True,
            )
    finally:
        store.close()

    total_episodes = sum(int(item["episodes"]) for item in results)
    total_wins = sum(int(item["wins"]) for item in results)
    final = {
        "status": "complete",
        "requested_tasks": len(tasks),
        "completed_tasks": len(results),
        "episodes_per_task": cli.episodes,
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
