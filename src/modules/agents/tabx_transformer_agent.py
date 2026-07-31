"""TABX TransformerAgent with a factorized centralized embedding model."""

from __future__ import annotations

from typing import Dict

import torch as th
import torch.nn as nn

from .transformer_agent import TransformerAgent


class FactorizedCEM(nn.Module):
    """Spatial-then-temporal CEM with the original token/output contract.

    The original CEM attends over L * (2 * A + 1) tokens at once. This module
    first attends over agents inside each timestep, then attends over time for
    each token slot. Its output has exactly the same flat token ordering, so
    TransformerAgent.train_EM can keep its original loss implementation.
    """

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        n_head: int,
        dropout: float,
        max_agents: int,
        max_timesteps: int,
        use_rtg: bool,
    ):
        super().__init__()
        self.n_embd = int(n_embd)
        self.max_agents = int(max_agents)
        self.max_timesteps = int(max_timesteps)
        self.use_rtg = bool(use_rtg)
        self.max_slots = (2 if use_rtg else 1) + 2 * self.max_agents
        spatial_layers = max(1, int(n_layer) // 2)
        temporal_layers = max(1, int(n_layer) - spatial_layers)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=int(n_head),
            dim_feedforward=4 * self.n_embd,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=int(n_head),
            dim_feedforward=4 * self.n_embd,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.spatial = nn.TransformerEncoder(
            spatial_layer, num_layers=spatial_layers
        )
        self.temporal = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers
        )
        self.slot_embedding = nn.Embedding(self.max_slots, self.n_embd)
        self.time_embedding = nn.Embedding(self.max_timesteps, self.n_embd)
        self.output_norm = nn.LayerNorm(self.n_embd)

    def _spatial_attention_mask(
        self, slots: int, n_agents: int, device: th.device
    ) -> th.Tensor:
        # True means "masked" for nn.TransformerEncoder. Recreate the
        # same-time information boundary of CustomGPT2Model: observation
        # prediction can see every current observation but no current action;
        # action tokens can see current observations/actions; the post token
        # can see the complete timestep.
        allowed = th.tril(
            th.ones(slots, slots, dtype=th.bool, device=device)
        )
        offset = 1 if self.use_rtg else 0
        obs_indices = [offset + 2 * i for i in range(n_agents)]
        action_indices = [offset + 1 + 2 * i for i in range(n_agents)]
        for obs_index in obs_indices:
            allowed[obs_index, obs_indices] = True
            allowed[obs_index, action_indices] = False
        for action_index in action_indices:
            allowed[action_index, obs_indices] = True
            allowed[action_index, action_indices] = True
        return ~allowed

    def custom_forward(
        self, inputs_embeds: th.Tensor, n_agents: int
    ) -> Dict[str, th.Tensor]:
        batch_size, flat_length, width = inputs_embeds.shape
        slots = (2 if self.use_rtg else 1) + 2 * int(n_agents)
        if n_agents > self.max_agents:
            raise ValueError(
                f"n_agents={n_agents} exceeds max_agents={self.max_agents}"
            )
        if flat_length % slots != 0:
            raise ValueError(
                f"CEM token length {flat_length} is not divisible by slots={slots}"
            )
        timesteps = flat_length // slots
        if timesteps > self.max_timesteps:
            raise ValueError(
                f"timesteps={timesteps} exceeds configured "
                f"max_timesteps={self.max_timesteps}"
            )
        values = inputs_embeds.reshape(
            batch_size, timesteps, slots, width
        )
        slot_ids = th.arange(slots, device=values.device)
        time_ids = th.arange(timesteps, device=values.device)
        values = (
            values
            + self.slot_embedding(slot_ids)[None, None, :, :]
            + self.time_embedding(time_ids)[None, :, None, :]
        )

        spatial_input = values.reshape(
            batch_size * timesteps, slots, width
        )
        spatial_mask = self._spatial_attention_mask(
            slots, int(n_agents), values.device
        )
        spatial_output = self.spatial(
            spatial_input, mask=spatial_mask
        ).reshape(
            batch_size, timesteps, slots, width
        )
        temporal_input = spatial_output.permute(0, 2, 1, 3).reshape(
            batch_size * slots, timesteps, width
        )
        causal_mask = th.triu(
            th.ones(
                timesteps,
                timesteps,
                dtype=th.bool,
                device=values.device,
            ),
            diagonal=1,
        )
        temporal_output = self.temporal(
            temporal_input, mask=causal_mask
        )
        output = temporal_output.reshape(
            batch_size, slots, timesteps, width
        ).permute(0, 2, 1, 3)
        output = self.output_norm(output + spatial_output)
        return {
            "last_hidden_state": output.reshape(
                batch_size, flat_length, width
            )
        }


class TabxTransformerAgent(TransformerAgent):
    """Drop-in TransformerAgent that only replaces the unsafe dense CEM."""

    def __init__(self, input_shape, args, env_info):
        super().__init__(input_shape, args, env_info)
        self.CEM = FactorizedCEM(
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            dropout=self.dropout,
            max_agents=self.max_agents,
            max_timesteps=self.traj_len,
            use_rtg=self.args.use_rtg,
        )
