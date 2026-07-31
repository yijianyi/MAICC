from __future__ import annotations

from typing import Iterable

import torch as th
import torch.nn as nn


def _weighted_replica_mean(
    values: th.Tensor,
    weights: th.Tensor,
) -> th.Tensor:
    """Recover the global masked mean from per-replica means."""

    if values.ndim == 0:
        return values
    values = values.reshape(-1)
    weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if values.numel() != weights.numel():
        raise RuntimeError(
            "Data-parallel loss/weight replica counts differ: "
            f"{values.numel()} != {weights.numel()}"
        )
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise RuntimeError("Data-parallel loss has no valid samples")
    return (values * weights).sum() / denominator


class _AgentDispatch(nn.Module):
    """Expose the original agent's non-forward training methods to DP."""

    def __init__(self, agent: nn.Module):
        super().__init__()
        self.agent = agent

    def forward(self, operation: str, *inputs):
        if operation == "train_em":
            losses = self.agent.train_EM(*inputs)
            mask = inputs[-1]
            valid = mask.sum().detach().to(dtype=th.float32)
            shifted_valid = (
                mask[:, 1:].sum().detach().to(dtype=th.float32)
            )
            return (*losses, valid, shifted_valid)
        if operation == "embedding":
            embedding, individual_return = (
                self.agent.get_dem_traj_embedding(*inputs)
            )
            # Upstream returns CPU zeros for a length-one prefix. DataParallel
            # can only gather replica outputs that live on replica devices.
            if individual_return.device != embedding.device:
                individual_return = individual_return.to(embedding.device)
            return embedding, individual_return
        if operation == "train_dt":
            loss = self.agent.train_DT(*inputs)
            seq_mask = inputs[-1]
            valid = (
                seq_mask[:, -1].sum().detach().to(dtype=th.float32)
            )
            return loss, valid
        if operation == "agent_forward":
            return self.agent(*inputs)
        raise ValueError(f"Unknown parallel agent operation {operation!r}")


def make_data_parallel_agent_class(base_agent_class):
    """Wrap the untouched TransformerAgent in single-process DataParallel."""

    class DataParallelOriginalTransformerAgent(nn.Module):
        def __init__(self, input_shape, args, env_info):
            super().__init__()
            self._dispatch = _AgentDispatch(
                base_agent_class(input_shape, args, env_info)
            )
            self._parallel = None
            self._device_ids = ()

        @property
        def _agent(self):
            return self._dispatch.agent

        @property
        def parallel_device_count(self) -> int:
            return len(self._device_ids)

        def _require_parallel(self):
            if self._parallel is None:
                raise RuntimeError(
                    "Data-parallel MAICC agent was used before cuda()"
                )
            return self._parallel

        def cuda(self, device=None):
            if device is not None:
                raise ValueError(
                    "The baseline data-parallel agent selects all visible "
                    "CUDA devices; control them with CUDA_VISIBLE_DEVICES"
                )
            count = th.cuda.device_count()
            if count < 2:
                raise RuntimeError(
                    "--data-parallel requires at least two visible CUDA "
                    "devices; set CUDA_VISIBLE_DEVICES=0,1 (or more)"
                )
            self._device_ids = tuple(range(count))
            self._dispatch.to(th.device("cuda", self._device_ids[0]))
            self._parallel = nn.DataParallel(
                self._dispatch,
                device_ids=list(self._device_ids),
                output_device=self._device_ids[0],
                dim=0,
            )
            return self

        def train_EM(
            self,
            seq_pre,
            seq_obs,
            seq_action,
            seq_post,
            mask,
        ):
            outputs = self._require_parallel()(
                "train_em",
                seq_pre,
                seq_obs,
                seq_action,
                seq_post,
                mask,
            )
            losses = outputs[:8]
            valid = outputs[8]
            shifted_valid = outputs[9]
            obs_valid = valid if self.args.use_rtg else shifted_valid
            weights: Iterable[th.Tensor] = (
                valid,
                valid,
                obs_valid,
                valid,
                obs_valid,
                valid,
                valid,
                valid,
            )
            return tuple(
                _weighted_replica_mean(loss, weight)
                for loss, weight in zip(losses, weights)
            )

        def get_dem_traj_embedding(
            self,
            seq_pre,
            seq_obs,
            seq_action,
            seq_post,
            mask=None,
        ):
            if mask is None:
                mask = th.ones(
                    seq_pre.shape[0],
                    seq_pre.shape[1],
                    1,
                    device=seq_pre.device,
                )
            return self._require_parallel()(
                "embedding",
                seq_pre,
                seq_obs,
                seq_action,
                seq_post,
                mask,
            )

        def train_DT(
            self,
            context_pre,
            context_obs,
            context_action,
            context_post,
            context_mask,
            seq_pre,
            seq_obs,
            seq_action,
            seq_post,
            seq_mask,
        ):
            loss, valid = self._require_parallel()(
                "train_dt",
                context_pre,
                context_obs,
                context_action,
                context_post,
                context_mask,
                seq_pre,
                seq_obs,
                seq_action,
                seq_post,
                seq_mask,
            )
            return _weighted_replica_mean(loss, valid)

        def forward(
            self,
            context_pre,
            context_obs,
            context_action,
            context_post,
            context_mask,
            seq_pre,
            seq_obs,
            seq_action,
            seq_post,
            seq_mask=None,
        ):
            if seq_mask is None:
                seq_mask = th.ones(
                    seq_pre.shape[0],
                    seq_pre.shape[1],
                    1,
                    device=seq_pre.device,
                )
            return self._require_parallel()(
                "agent_forward",
                context_pre,
                context_obs,
                context_action,
                context_post,
                context_mask,
                seq_pre,
                seq_obs,
                seq_action,
                seq_post,
                seq_mask,
            )

        def state_dict(self, *args, **kwargs):
            # Keep checkpoint keys compatible with the original agent.
            return self._agent.state_dict(*args, **kwargs)

        def load_state_dict(self, state_dict, strict=True):
            return self._agent.load_state_dict(state_dict, strict=strict)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                dispatch = super().__getattr__("_dispatch")
                return getattr(dispatch.agent, name)

    DataParallelOriginalTransformerAgent.__name__ = "TransformerAgent"
    DataParallelOriginalTransformerAgent.__qualname__ = "TransformerAgent"
    return DataParallelOriginalTransformerAgent
