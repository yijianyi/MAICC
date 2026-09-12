import os
import sys

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
from .smacv2_wrapper import SMACv2Wrapper


def smac_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return SMACv2Wrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["sc2v2"] = smac_fn

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

REGISTRY["gymma"] = gymma_fn
