"""Synchronous MAICC-facing adapter for a TABX JAX environment instance."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


class TabxOnlineWrapper:
    """Expose TABX through the small interface used by MAICC runners.

    Pass an environment already wrapped with TABXEnemyHeuristicWrapper when
    the enemy side should use the TABX oracle/heuristic policy. This adapter
    deliberately calls env.get_avail_actions(state) online; it never rebuilds
    cooldown rules from observations.
    """

    IDLE_ACTION = 7

    def __init__(self, env: Any, env_params: Mapping[str, Any], seed: int = 0):
        self.env = env
        self.env_params = env_params
        self.seed = int(seed)
        self.episode_index = 0
        self._rng = None
        self.state = None
        self._obs = None
        self.ally_keys = tuple(env.ally_keys)
        self.max_allies = int(env.max_n_ally)
        self.max_enemies = int(env.max_n_enemy)
        self.n_actions = int(env.action_spaces[self.ally_keys[0]].n)
        self.obs_dim = int(env.observation_spaces[self.ally_keys[0]].shape[0])
        self.episode_limit = int(getattr(env, "max_episode_steps", 512))
        self.present = np.ones(self.max_allies, dtype=bool)
        self.teammate_policies = {}
        self._teammate_params = {}
        self._teammate_targets = {}
        self._teammate_rngs = {}
        self.last_executed_actions = None

    def configure_teammates(self, policies: Mapping[int, str]) -> None:
        """Configure zero-based ally slots before resetting the environment."""
        from .teammate_modes import validate_policies

        self.teammate_policies = validate_policies(policies)
        if any(i >= self.max_allies for i in self.teammate_policies):
            raise ValueError("Teammate agent ID exceeds max_allies")
        self._teammate_params = {}
        if self.teammate_policies:
            from src.tabx.heuristic_policy import build_batched_heuristic_params

            self._teammate_params = {
                i: build_batched_heuristic_params(name)
                for i, name in self.teammate_policies.items()
            }

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        return np.asarray(value)

    def _reset_key(self):
        import jax

        self._rng = jax.random.PRNGKey(self.seed + self.episode_index)
        self.episode_index += 1
        self._rng, key = jax.random.split(self._rng)
        return key

    def _step_key(self):
        import jax

        self._rng, key = jax.random.split(self._rng)
        return key

    def _stack_obs(self, observations: Mapping[str, Any]) -> np.ndarray:
        values = [self._numpy(observations[key]) for key in self.ally_keys]
        result = np.stack(values, axis=0).astype(np.float32, copy=False)
        if result.shape != (self.max_allies, self.obs_dim):
            raise RuntimeError(
                f"TABX obs shape {result.shape}, expected "
                f"{(self.max_allies, self.obs_dim)}"
            )
        result[~self.present] = 0.0
        return result

    def _detect_present(self) -> np.ndarray:
        # The underlying TABX state keeps max_health for dead real units, while
        # never-present padded slots have max_health==0.
        raw = self.state.get("state", self.state)
        values = []
        for key in self.ally_keys:
            maximum = self._numpy(raw[key].status.max_health).reshape(-1)[0]
            values.append(float(maximum) > 0.0)
        return np.asarray(values, dtype=bool)

    def reset(self):
        episode_seed = self.seed + self.episode_index
        observations, self.state = self.env.reset(
            self._reset_key(), self.env_params
        )
        self.present = self._detect_present()
        self.last_executed_actions = None
        self._teammate_targets = {}
        self._teammate_rngs = {}
        if self.teammate_policies:
            import jax
            from src.tabx.heuristic_policy import LastVisibleTarget

            raw = self.state["state"]
            active = {
                i for i, unit in enumerate(self.ally_keys)
                if self.present[i] and not bool(np.asarray(raw[unit].status.is_disabled).item())
            }
            if not set(self.teammate_policies) < active:
                raise ValueError(
                    "Replace only present, enabled allies and retain at least one model ally; "
                    f"requested={sorted(self.teammate_policies)}, active={sorted(active)}"
                )
            for i in self.teammate_policies:
                self._teammate_targets[i] = LastVisibleTarget()
                # Independent from the environment/enemy stream, and from other allies.
                root = jax.random.fold_in(jax.random.PRNGKey(episode_seed), 0x5445414D)
                self._teammate_rngs[i] = jax.random.fold_in(root, i)
        self._obs = self._stack_obs(observations)
        return self._obs.copy()

    def _replace_teammate_actions(self, values: np.ndarray) -> None:
        if not self.teammate_policies:
            return
        import jax
        import jax.numpy as jnp
        from src.tabx.heuristic_policy import heuristic_policy

        raw = self.state["state"]
        for i in self.teammate_policies:
            status = raw[self.ally_keys[i]].status
            if not bool(np.asarray(status.is_alive).item()) or bool(
                np.asarray(status.is_disabled).item()
            ):
                values[i] = self.IDLE_ACTION
                continue
            self._teammate_rngs[i], key = jax.random.split(self._teammate_rngs[i])
            action, self._teammate_targets[i] = heuristic_policy(
                key, jnp.asarray(self._obs[i]), self._teammate_targets[i],
                len(self.env.unit_keys), self.env.max_n_zone,
                self._teammate_params[i], self.state["physics_params"],
            )
            values[i] = int(np.asarray(action).item())

    def step(self, actions: Sequence[int]):
        import jax.numpy as jnp

        values = np.asarray(actions, dtype=np.int64).reshape(-1)
        if values.size != self.max_allies:
            raise ValueError(
                f"Expected {self.max_allies} ally actions, got {values.size}"
            )
        if np.any((values < 0) | (values >= self.n_actions)):
            raise ValueError(f"Action outside [0, {self.n_actions})")
        values = values.copy()
        self._replace_teammate_actions(values)
        values[~self.present] = self.IDLE_ACTION
        if self.teammate_policies:
            avail = self.get_avail_actions()
            if np.any((values < 0) | (values >= self.n_actions)) or not np.all(
                avail[np.arange(self.max_allies), values]
            ):
                raise ValueError(f"Unavailable executed ally action: {values.tolist()}")
        self.last_executed_actions = values.copy()
        jax_values = jnp.asarray(values, dtype=jnp.int32)
        action_dict = {
            key: jax_values[index] for index, key in enumerate(self.ally_keys)
        }
        observations, self.state, rewards, dones, info = self.env.step(
            self._step_key(), self.state, action_dict
        )
        self._obs = self._stack_obs(observations)
        if isinstance(rewards, Mapping):
            reward = float(np.asarray(rewards.get("__all__", rewards[self.ally_keys[0]])))
        else:
            reward = float(np.asarray(rewards).reshape(-1)[0])
        done = bool(np.asarray(dones["__all__"]))
        result_info = dict(info)
        if "is_win" in result_info:
            result_info["is_win"] = bool(
                np.asarray(result_info["is_win"]).reshape(-1)[0]
            )
        if "truncation" in result_info:
            result_info["truncation"] = bool(
                np.asarray(result_info["truncation"]).reshape(-1)[0]
            )
        result_info["episode_limit"] = bool(
            result_info.get("truncation", False)
        )
        return reward, done, result_info

    def get_obs(self) -> np.ndarray:
        if self._obs is None:
            raise RuntimeError("reset() must be called before get_obs()")
        return self._obs.copy()

    def get_state(self) -> np.ndarray:
        # Concat state, computed on demand; it is not materialized in replay.
        return self.get_obs().reshape(-1)

    def get_avail_actions(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("reset() must be called before get_avail_actions()")
        values = self.env.get_avail_actions(self.state)
        result = np.stack(
            [self._numpy(values[key]) for key in self.ally_keys], axis=0
        ).astype(np.int8, copy=False)
        result[~self.present] = 0
        result[~self.present, self.IDLE_ACTION] = 1
        return result

    def get_env_info(self) -> Dict[str, int]:
        return {
            "n_agents": self.max_allies,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_dim,
            "state_shape": self.max_allies * self.obs_dim,
            "episode_limit": self.episode_limit,
        }
