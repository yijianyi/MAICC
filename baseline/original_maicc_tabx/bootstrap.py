from __future__ import annotations

import sys
from pathlib import Path

from .buffer import make_tabx_buffer_class
from .env import tabx_offline_env_fn
from .fidelity import verify_upstream_core


def install(
    repo_root: Path,
    micro_batch_size: int = 0,
    gradient_checkpointing: bool = False,
    padding_compatible: bool = False,
    data_parallel: bool = False,
) -> None:
    """Install data-only adapters before executing the untouched main.py."""

    repo_root = Path(repo_root).resolve()
    verify_upstream_core(repo_root)
    src_root = repo_root / "src"
    for path in (repo_root, src_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    import envs
    import learners
    import modules.agents as agents
    import run as maicc_run
    from modules.agents.transformer_agent import TransformerAgent
    from .vendor.ic_learner import ICLearner as OriginalICLearner

    envs.REGISTRY["tabx"] = tabx_offline_env_fn
    maicc_run.InContextReplayBuffer = make_tabx_buffer_class()
    if int(micro_batch_size) > 0:
        from .accumulating_learner import AccumulatingOriginalICLearner

        learners.REGISTRY["ic_learner"] = AccumulatingOriginalICLearner
    elif padding_compatible:
        from .padding_learner import PaddingCompatibleOriginalICLearner

        learners.REGISTRY["ic_learner"] = (
            PaddingCompatibleOriginalICLearner
        )
    else:
        learners.REGISTRY["ic_learner"] = OriginalICLearner

    agent_class = TransformerAgent
    if gradient_checkpointing:
        class CheckpointedOriginalTransformerAgent(TransformerAgent):
            def __init__(self, input_shape, args, env_info):
                super().__init__(input_shape, args, env_info)
                self.CEM.gradient_checkpointing_enable()
                self.DEM.gradient_checkpointing_enable()
                self.DT.gradient_checkpointing_enable()

        CheckpointedOriginalTransformerAgent.__name__ = "TransformerAgent"
        agent_class = CheckpointedOriginalTransformerAgent

    if data_parallel:
        from .data_parallel import make_data_parallel_agent_class

        agent_class = make_data_parallel_agent_class(agent_class)

    agents.REGISTRY["transformer"] = agent_class
