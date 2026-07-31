from __future__ import annotations

from pathlib import Path

from .manifest import BaselineManifest


class TabxOfflineMetadataEnv:
    """Schema-only environment used by the untouched MAICC runner.

    Original IC-MARL training samples its offline replay buffer and never calls
    ``runner.run``. Any attempt to use this object for online evaluation fails
    loudly rather than fabricating an environment result.
    """

    def __init__(
        self,
        manifest,
        time_limit=None,
        key="tabx_original_maicc",
        seed=0,
        common_reward=True,
        reward_scalarisation="sum",
        **_kwargs,
    ):
        if not common_reward:
            raise ValueError("TABX baseline requires the original common reward")
        self.manifest = BaselineManifest(Path(str(manifest)))
        self.manifest.validate_files()
        configured_limit = (
            self.manifest.sequence_length
            if time_limit is None
            else int(time_limit)
        )
        if configured_limit != self.manifest.sequence_length:
            raise ValueError(
                f"time_limit={configured_limit} does not match manifest "
                f"sequence_length={self.manifest.sequence_length}"
            )
        self.episode_limit = configured_limit
        self.key = str(key)
        self.seed = int(seed)
        self.reward_scalarisation = str(reward_scalarisation)

    def get_env_info(self):
        return {
            "state_shape": 1,
            "obs_shape": self.manifest.obs_dim,
            "n_actions": self.manifest.action_dim,
            "n_agents": self.manifest.max_agents,
            "episode_limit": self.manifest.sequence_length,
        }

    def close(self):
        return None

    def save_replay(self):
        raise RuntimeError("TABX has no simulator replay in offline baseline mode")

    def reset(self):
        raise RuntimeError(
            "Online TABX evaluation requires a simulator; this environment "
            "provides metadata only"
        )

    def step(self, _actions):
        return self.reset()


def tabx_offline_env_fn(**kwargs):
    return TabxOfflineMetadataEnv(**kwargs)
