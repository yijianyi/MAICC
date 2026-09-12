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
        self.current_episode_seed = None
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

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        return np.asarray(value)

    def _reset_key(self, seed: int | None = None):
        import jax

        episode_seed = (
            self.seed + self.episode_index if seed is None else int(seed)
        )
        if episode_seed < 0:
            raise ValueError(f"Episode seed must be non-negative, got {episode_seed}")
        self.current_episode_seed = episode_seed
        self._rng = jax.random.PRNGKey(episode_seed)
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

    def reset(self, seed: int | None = None):
        """Reset with an explicit seed, or use ``base_seed + episode_index``."""
        observations, self.state = self.env.reset(
            self._reset_key(seed), self.env_params
        )
        self.present = self._detect_present()
        self._obs = self._stack_obs(observations)
        return self._obs.copy()

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
        values[~self.present] = self.IDLE_ACTION
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
