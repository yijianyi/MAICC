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
        self.n_head = int(n_head)
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

    def _batched_attention_mask(
        self,
        base_mask: th.Tensor,
        valid_tokens: th.Tensor,
    ) -> th.Tensor:
        """Mask invalid keys without ever creating an all-masked query row.

        ``src_key_padding_mask`` only masks keys.  Combining it with the
        causal/spatial mask can therefore leave a padded query with no valid
        key at all, which makes attention softmax return NaN.  A per-example
        attention mask lets valid queries ignore padding while invalid queries
        attend to themselves solely to keep the numerical path finite.  Their
        outputs are zeroed immediately after the encoder.
        """
        valid = valid_tokens.bool()
        batch_size, length = valid.shape
        mask = base_mask.unsqueeze(0).expand(
            batch_size, length, length
        ).clone()
        mask |= (~valid).unsqueeze(1).expand(-1, length, -1)
        invalid_queries = ~valid
        if invalid_queries.any():
            batch_ids, query_ids = th.nonzero(
                invalid_queries, as_tuple=True
            )
            mask[batch_ids, query_ids, :] = True
            mask[batch_ids, query_ids, query_ids] = False
        return mask.repeat_interleave(self.n_head, dim=0)

    def custom_forward(
        self, inputs_embeds: th.Tensor, n_agents: int,
        present_mask: th.Tensor | None = None,
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
        spatial_valid = th.ones(
            batch_size, timesteps, slots,
            dtype=th.bool, device=values.device,
        )
        if present_mask is not None:
            present = present_mask.bool()
            offset = 1 if self.use_rtg else 0
            for agent_id in range(int(n_agents)):
                spatial_valid[:, :, offset + 2 * agent_id] = present[
                    :, :, agent_id
                ]
                spatial_valid[:, :, offset + 2 * agent_id + 1] = present[
                    :, :, agent_id
                ]
        spatial_valid_flat = spatial_valid.reshape(
            batch_size * timesteps, slots
        )
        spatial_mask = self._batched_attention_mask(
            spatial_mask, spatial_valid_flat
        )
        spatial_output = self.spatial(
            spatial_input, mask=spatial_mask,
        ).reshape(
            batch_size, timesteps, slots, width
        )
        spatial_output = spatial_output.masked_fill(
            ~spatial_valid.unsqueeze(-1), 0.0
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
        temporal_valid = spatial_valid.permute(0, 2, 1).reshape(
            batch_size * slots, timesteps
        )
        causal_mask = self._batched_attention_mask(
            causal_mask, temporal_valid
        )
        temporal_output = self.temporal(
            temporal_input, mask=causal_mask
        )
        output = temporal_output.reshape(
            batch_size, slots, timesteps, width
        ).permute(0, 2, 1, 3)
        output = output.masked_fill(
            ~spatial_valid.unsqueeze(-1), 0.0
        )
        output = self.output_norm(output + spatial_output)
        output = output.masked_fill(
            ~spatial_valid.unsqueeze(-1), 0.0
        )
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
