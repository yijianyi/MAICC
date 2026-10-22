from transformers import GPT2Config, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch as th
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

LOG_VAR_MIN = -2
LOG_VAR_MAX = 0


class CustomGPT2Model(GPT2Model):
    def __init__(self, config, use_rtg):
        super().__init__(config)
        self.use_rtg = use_rtg
        for block in self.h:
            block.attn.bias[:] = True

    def custom_forward(self, inputs_embeds, n_agents, present_mask=None):
        batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        mask = th.tril(th.ones((1, seq_len, seq_len), dtype=th.bool)).to(
            inputs_embeds.device)
        if self.use_rtg:
            o = [1 + 2 * i for i in range(n_agents)]
            a = [2 + 2 * i for i in range(n_agents)]
            traj_len = 2 + 2 * n_agents
        else:
            o = [2 * i for i in range(n_agents)]
            a = [1 + 2 * i for i in range(n_agents)]
            traj_len = 1 + 2 * n_agents
        for i in range(seq_len):
            if i % traj_len in o:
                if self.use_rtg:
                    start_idx = i - i % traj_len + 1
                    end_idx = i - i % traj_len + 1 + 2 * n_agents
                else:
                    start_idx = i - i % traj_len
                    end_idx = i - i % traj_len + 2 * n_agents
                mask[0][i][start_idx:end_idx:2] = True
                mask[0][i][start_idx+1:end_idx:2] = False
            if i % traj_len in a:
                if self.use_rtg:
                    start_idx = i - i % traj_len + 1
                    end_idx = i - i % traj_len + 1 + 2 * n_agents
                else:
                    start_idx = i - i % traj_len
                    end_idx = i - i % traj_len + 2 * n_agents
                mask[0][i][start_idx:end_idx] = True
        valid = None
        if present_mask is not None:
            timesteps = seq_len // traj_len
            present = present_mask.reshape(batch_size, timesteps, n_agents).bool()
            valid = th.ones(batch_size, timesteps, traj_len, dtype=th.bool,
                            device=inputs_embeds.device)
            valid[:, :, o] = present
            valid[:, :, a] = present
            valid = valid.reshape(batch_size, seq_len)
            mask = mask.expand(batch_size, -1, -1).clone()
            mask &= valid[:, None, :]
            # Padded query rows attend only to themselves, avoiding an empty
            # softmax. Their outputs are discarded; valid tokens cannot read them.
            batch_ids, query_ids = th.where(~valid)
            mask[batch_ids, query_ids, :] = False
            mask[batch_ids, query_ids, query_ids] = True
        result = self.forward(inputs_embeds=inputs_embeds, attention_mask=mask)
        if valid is not None:
            result.last_hidden_state = result.last_hidden_state.masked_fill(
                ~valid.unsqueeze(-1), 0.0)
        return result

    def forward(
        self,
        input_ids: Optional[th.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[th.Tensor]]] = None,
        attention_mask: Optional[th.FloatTensor] = None,
        token_type_ids: Optional[th.LongTensor] = None,
        position_ids: Optional[th.LongTensor] = None,
        head_mask: Optional[th.FloatTensor] = None,
        inputs_embeds: Optional[th.FloatTensor] = None,
        encoder_hidden_states: Optional[th.Tensor] = None,
        encoder_attention_mask: Optional[th.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = th.arange(
                past_length, input_shape[-1] + past_length, dtype=th.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        # Modified
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            assert attention_mask.size(0) in (1, batch_size)
            assert (attention_mask.size(1) == seq_len)
            assert (attention_mask.size(2) == seq_len)
            attention_mask = attention_mask[:, None, :, :]

            attention_mask = attention_mask.to(
                dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * th.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = th.ones(
                    encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                th.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device)
                                       for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, th.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_reentrant=False,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + \
                    (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + \
                        (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            th.tensor(init_value, dtype=th.float32)
        )

    def forward(self):
        return self.constant


class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args, env_info):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.k = args.k
        self.traj_len = args.env_args["time_limit"]
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.dropout = args.dropout
        if self.args.env == "sc2v2":
            self.max_agents = 12
            self.max_actions = 6 + 13
        else:
            self.max_agents = self.n_agents
            self.max_actions = self.n_actions

        if self.args.use_rtg:
            CEM_config = GPT2Config(
                # [pre, obs * n, action * n, post]
                n_positions=(2 + 2 * self.max_agents) * self.traj_len,
                n_embd=self.n_embd,
                n_layer=self.n_layer,
                n_head=self.n_head,
                resid_pdrop=self.dropout,
                embd_pdrop=self.dropout,
                attn_pdrop=self.dropout,
                use_cache=False
            )
        else:
            CEM_config = GPT2Config(
                # [obs * n, action * n, post]
                n_positions=(1 + 2 * self.max_agents) * self.traj_len,
                n_embd=self.n_embd,
                n_layer=self.n_layer,
                n_head=self.n_head,
                resid_pdrop=self.dropout,
                embd_pdrop=self.dropout,
                attn_pdrop=self.dropout,
                use_cache=False
            )

        if self.args.use_rtg:
            DEM_config = GPT2Config(
                n_positions=4 * self.traj_len,  # [pre, obs, action, post]
                n_embd=self.n_embd,
                n_layer=self.n_layer,
                n_head=self.n_head,
                resid_pdrop=self.dropout,
                embd_pdrop=self.dropout,
                attn_pdrop=self.dropout,
                use_cache=False
            )
        else:
            DEM_config = GPT2Config(
                n_positions=3 * self.traj_len,  # [obs, action, post]
                n_embd=self.n_embd,
                n_layer=self.n_layer,
                n_head=self.n_head,
                resid_pdrop=self.dropout,
                embd_pdrop=self.dropout,
                attn_pdrop=self.dropout,
                use_cache=False
            )

        DT_config = GPT2Config(
            n_positions=4 * self.traj_len *
            (self.k + 1),  # [pre, obs, action, post]
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False
        )

        if self.args.use_rtg:
            self.EM_pre_embd = nn.Linear(1, self.n_embd)  # [rtg]
        self.EM_obs_embd = nn.Sequential(
            nn.Linear(input_shape, self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, self.n_embd))
        # TABX reserves one internal-only action token for a causally unknown
        # current action. Environment logits/actions still have max_actions=8.
        self.action_vocab_size = self.max_actions + (1 if args.env == "tabx" else 0)
        self.unknown_action_id = self.max_actions if args.env == "tabx" else None
        self.EM_action_embd = nn.Embedding(self.action_vocab_size, self.n_embd)
        self.EM_post_embd = nn.Linear(
            3, self.n_embd)  # [rew, term, completion]

        self.DT_pre_embd = nn.Linear(1, self.n_embd)  # [rtg]
        self.DT_obs_embd = nn.Sequential(
            nn.Linear(input_shape, self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, self.n_embd))
        self.DT_action_embd = nn.Embedding(self.action_vocab_size, self.n_embd)
        self.DT_post_embd = nn.Linear(
            3, self.n_embd)  # [rew, term, completion]

        # CustomGPT2Model supplies the original nonstandard same-time mask.
        # Keep its explicit attention path independent of library auto-selection.
        CEM_config._attn_implementation = "eager"
        self.CEM = CustomGPT2Model(CEM_config, self.args.use_rtg)
        self.DEM = GPT2Model(DEM_config)
        self.DT = GPT2Model(DT_config)
        self.EM_pred_actions = nn.Linear(self.n_embd, self.max_actions)
        self.DT_pred_actions = nn.Linear(self.n_embd, self.max_actions)
        self.EM_pred_reward = nn.Linear(self.n_embd, 1)
        self.EM_pred_obs = nn.Sequential(
            nn.Linear(self.n_embd, self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, input_shape * 2))
        self.CEM_pred_obs = nn.Sequential(
            nn.Linear(self.n_embd + input_shape, self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU(), nn.Linear(self.args.hidden_dim, self.n_embd))

        self.log_var_multiplier = Scalar(1.0)
        self.log_var_offset = Scalar(-1.0)
        self.gauss_nll_loss = nn.GaussianNLLLoss(reduction='none')

    def train_EM(self, seq_pre, seq_obs, seq_action, seq_post, mask, present_mask=None):
        batch_size, seq_len, n_agents = seq_obs.shape[0], seq_obs.shape[1], seq_obs.shape[2]
        time_mask = mask.squeeze(-1).bool()
        if present_mask is None:
            present_mask = th.ones(
                batch_size, seq_len, n_agents, device=seq_obs.device,
                dtype=th.bool,
            )
        else:
            present_mask = present_mask.squeeze(-1).bool()
        agent_mask = time_mask.unsqueeze(-1) & present_mask
        if not agent_mask.any():
            raise ValueError(
                "train_EM received a batch with no valid present-agent "
                "action timestep"
            )
        idx_mask = th.nonzero(time_mask.reshape(-1)).reshape(-1)
        if self.args.env == "sc2v2":
            ori_dim = seq_obs.shape[-1]
            seq_obs = F.pad(seq_obs, (0, self.input_shape - ori_dim))
        target_action = seq_action[agent_mask]
        target_reward = seq_post[:, :, 0].reshape(-1)[idx_mask]
        target_obs = seq_obs
        obs_feature_weight = th.ones_like(target_obs)
        if self.args.env == "tabx":
            # Invisible/padded other-unit blocks are all zero in TABX. Keep
            # them as valid transition targets, but upweight visible blocks so
            # the Gaussian loss cannot be minimized mostly by predicting zero.
            n_zones = int(getattr(self.args, "n_zones", 4))
            remainder = self.input_shape - 14 - 6 * n_zones
            if remainder < 0 or remainder % 16:
                raise ValueError("TABX observation layout/n_zones mismatch")
            other_blocks = remainder // 16
            visible_weight = float(
                getattr(self.args, "obs_visible_weight", 1.0)
            )
            for block in range(other_blocks):
                start = 14 + 16 * block
                end = start + 16
                visible = (
                    target_obs[..., start:end].abs().sum(-1, keepdim=True) > 0
                ).float()
                obs_feature_weight[..., start:end] = (
                    1.0 + (visible_weight - 1.0) * visible
                )

        if self.args.train_CEM:
            CEM_inputs = []
            for t in range(seq_len):
                if self.args.use_rtg:
                    CEM_inputs.append(self.EM_pre_embd(seq_pre[:, t]))
                for i in range(n_agents):
                    CEM_inputs.append(self.EM_obs_embd(seq_obs[:, t, i]))
                    CEM_inputs.append(self.EM_action_embd(seq_action[:, t, i]))
                CEM_inputs.append(self.EM_post_embd(seq_post[:, t]))
            CEM_inputs = th.stack(CEM_inputs, dim=1)
            if self.args.env == "tabx":
                cem_result = self.CEM.custom_forward(
                    CEM_inputs, n_agents, present_mask=present_mask
                )
            else:
                cem_result = self.CEM.custom_forward(CEM_inputs, n_agents)
            CEM_outputs = cem_result[
                'last_hidden_state']
            CEM_z_os = []
            for i in range(n_agents):
                if self.args.use_rtg:
                    CEM_z_os.append(
                        CEM_outputs[:, 1 + i * 2::(2 + 2 * n_agents)])
                else:
                    CEM_z_os.append(
                        CEM_outputs[:, i * 2::(1 + 2 * n_agents)])
            CEM_z_os = th.stack(CEM_z_os, dim=2)
            CEM_action_out = self.EM_pred_actions(CEM_z_os)[agent_mask]
            CEM_action_loss = F.cross_entropy(CEM_action_out, target_action)

            CEM_z_as = []
            for i in range(n_agents):
                if self.args.use_rtg:
                    CEM_z_as.append(
                        CEM_outputs[:, 2 + i * 2::(2 + 2 * n_agents)])
                else:
                    CEM_z_as.append(
                        CEM_outputs[:, 1 + i * 2::(1 + 2 * n_agents)])
            CEM_z_as = th.stack(CEM_z_as, dim=2)
            CEM_reward_all = self.EM_pred_reward(CEM_z_as).squeeze(-1)
            CEM_reward_out = (
                CEM_reward_all * present_mask.float()
            ).sum(-1).reshape(-1)[idx_mask]
            CEM_reward_loss = F.mse_loss(CEM_reward_out, target_reward)

            if self.args.use_rtg:
                CEM_z_rtgs = []
                for i in range(n_agents):
                    CEM_z_rtgs.append(CEM_outputs[:, 0::(2 + 2 * n_agents)])
                CEM_z_rtgs = th.stack(CEM_z_rtgs, dim=2)
                pre_obs = th.cat([th.zeros(batch_size, 1, n_agents, self.input_shape).to(
                    seq_obs.device), seq_obs[:, :-1]], dim=1)
                CEM_z_rtgs = th.cat([CEM_z_rtgs, pre_obs], dim=-1)
                CEM_z_rtgs = self.CEM_pred_obs(CEM_z_rtgs)
            else:
                CEM_z_posts = []
                for i in range(n_agents):
                    CEM_z_posts.append(
                        CEM_outputs[:, 2 * n_agents::(1 + 2 * n_agents)])
                CEM_z_posts = th.stack(CEM_z_posts, dim=2)
                pre_obs = th.cat([seq_obs[:, :-1], th.zeros(batch_size, 1, n_agents, self.input_shape).to(
                    seq_obs.device)], dim=1)
                CEM_z_posts = th.cat([CEM_z_posts, pre_obs], dim=-1)
                CEM_z_posts = self.CEM_pred_obs(CEM_z_posts)

            if self.args.use_rtg:
                CEM_obs_out_mean, CEM_obs_out_logvar = th.split(
                    self.EM_pred_obs(CEM_z_rtgs), self.input_shape, dim=-1)
            else:
                CEM_obs_out_mean, CEM_obs_out_logvar = th.split(
                    self.EM_pred_obs(CEM_z_posts), self.input_shape, dim=-1)
            CEM_obs_out_logvar = self.log_var_multiplier() * CEM_obs_out_logvar + \
                self.log_var_offset()
            CEM_obs_out_logvar = th.clamp(
                CEM_obs_out_logvar, LOG_VAR_MIN, LOG_VAR_MAX)
            CEM_obs_out_var = th.exp(CEM_obs_out_logvar)

            if self.args.use_rtg:
                temp_mask = (
                    agent_mask.unsqueeze(-1).expand_as(CEM_obs_out_mean)
                    * obs_feature_weight
                )
                masked_CEM_obs_loss = self.gauss_nll_loss(
                    CEM_obs_out_mean, target_obs, CEM_obs_out_var) * temp_mask
                CEM_obs_loss = (
                    masked_CEM_obs_loss.sum()
                    / temp_mask.sum().clamp_min(1.0)
                )
            else:
                temp_mask = (
                    agent_mask[:, 1:].unsqueeze(-1).expand_as(target_obs[:, 1:])
                    * obs_feature_weight[:, 1:]
                )
                masked_CEM_obs_loss = self.gauss_nll_loss(
                    CEM_obs_out_mean[:, :-1], target_obs[:, 1:], CEM_obs_out_var[:, :-1]) * temp_mask
                CEM_obs_loss = (
                    masked_CEM_obs_loss.sum()
                    / temp_mask.sum().clamp_min(1.0)
                )
        else:
            CEM_action_loss, CEM_reward_loss, CEM_obs_loss = 0, 0, 0

        DEM_z_os = []
        DEM_z_as = []
        if self.args.use_rtg:
            DEM_z_rtgs = []
        else:
            DEM_z_posts = []
        for i in range(n_agents):
            DEM_inputs = []
            for t in range(seq_len):
                if self.args.use_rtg:
                    DEM_inputs.append(self.EM_pre_embd(seq_pre[:, t]))
                DEM_inputs.append(self.EM_obs_embd(seq_obs[:, t, i]))
                DEM_inputs.append(self.EM_action_embd(seq_action[:, t, i]))
                DEM_inputs.append(self.EM_post_embd(seq_post[:, t]))
            DEM_inputs = th.stack(DEM_inputs, dim=1)
            DEM_outputs = self.DEM(inputs_embeds=DEM_inputs)[
                'last_hidden_state']
            if self.args.use_rtg:
                DEM_z_os.append(DEM_outputs[:, 1::4])
                DEM_z_as.append(DEM_outputs[:, 2::4])
                DEM_z_rtgs.append(DEM_outputs[:, 0::4])
            else:
                DEM_z_os.append(DEM_outputs[:, 0::3])
                DEM_z_as.append(DEM_outputs[:, 1::3])
                DEM_z_posts.append(DEM_outputs[:, 2::3])
        DEM_z_os = th.stack(DEM_z_os, dim=2)
        DEM_z_as = th.stack(DEM_z_as, dim=2)
        if self.args.use_rtg:
            DEM_z_rtgs = th.stack(DEM_z_rtgs, dim=2)
        else:
            DEM_z_posts = th.stack(DEM_z_posts, dim=2)

        DEM_action_out = self.EM_pred_actions(DEM_z_os)[agent_mask]
        DEM_action_loss = F.cross_entropy(DEM_action_out, target_action)

        if self.args.use_rtg:
            DEM_obs_out_mean, DEM_obs_out_logvar = th.split(
                self.EM_pred_obs(DEM_z_rtgs), self.input_shape, dim=-1)
        else:
            DEM_obs_out_mean, DEM_obs_out_logvar = th.split(
                self.EM_pred_obs(DEM_z_posts), self.input_shape, dim=-1)
        DEM_obs_out_logvar = self.log_var_multiplier() * DEM_obs_out_logvar + \
            self.log_var_offset()
        DEM_obs_out_logvar = th.clamp(
            DEM_obs_out_logvar, LOG_VAR_MIN, LOG_VAR_MAX)
        DEM_obs_out_var = th.exp(DEM_obs_out_logvar)
        if self.args.use_rtg:
            temp_mask = (
                agent_mask.unsqueeze(-1).expand_as(DEM_obs_out_mean)
                * obs_feature_weight
            )
            masked_DEM_obs_loss = self.gauss_nll_loss(
                DEM_obs_out_mean, target_obs, DEM_obs_out_var) * temp_mask
            DEM_obs_loss = (
                masked_DEM_obs_loss.sum()
                / temp_mask.sum().clamp_min(1.0)
            )
        else:
            temp_mask = (
                agent_mask[:, 1:].unsqueeze(-1).expand_as(target_obs[:, 1:])
                * obs_feature_weight[:, 1:]
            )
            masked_DEM_obs_loss = self.gauss_nll_loss(
                DEM_obs_out_mean[:, :-1], target_obs[:, 1:], DEM_obs_out_var[:, :-1]) * temp_mask
            DEM_obs_loss = (
                masked_DEM_obs_loss.sum()
                / temp_mask.sum().clamp_min(1.0)
            )

        if self.args.train_CEM:
            DEM_z_os = DEM_z_os[agent_mask]
            DEM_z_as = DEM_z_as[agent_mask]
            if self.args.use_rtg:
                DEM_z_rtgs = DEM_z_rtgs[agent_mask]
            else:
                DEM_z_posts = DEM_z_posts[agent_mask]

            CEM_z_os = CEM_z_os[agent_mask]
            CEM_z_as = CEM_z_as[agent_mask]
            if self.args.use_rtg:
                CEM_z_rtgs = CEM_z_rtgs[agent_mask]
            else:
                CEM_z_posts = CEM_z_posts[agent_mask]

            MSE_loss1 = F.mse_loss(DEM_z_os, CEM_z_os.detach())
            MSE_loss2 = F.mse_loss(DEM_z_as, CEM_z_as.detach())
            if self.args.use_rtg:
                MSE_loss3 = F.mse_loss(DEM_z_rtgs, CEM_z_rtgs.detach())
            else:
                MSE_loss3 = F.mse_loss(DEM_z_posts, CEM_z_posts.detach())
        else:
            MSE_loss1, MSE_loss2, MSE_loss3 = 0, 0, 0
        return CEM_action_loss, CEM_reward_loss, CEM_obs_loss, DEM_action_loss, DEM_obs_loss, MSE_loss1, MSE_loss2, MSE_loss3

    def get_dem_traj_embedding(self, seq_pre, seq_obs, seq_action, seq_post, mask=None):
        batch_size, seq_len = seq_pre.shape[0], seq_pre.shape[1]
        if mask is None:
            mask = th.ones(batch_size, seq_len, 1).to(seq_pre.device)
        if self.args.use_rtg:
            pre = self.EM_pre_embd(seq_pre)
        obs = self.EM_obs_embd(seq_obs)
        action = self.EM_action_embd(seq_action)
        post = self.EM_post_embd(seq_post)
        if self.args.use_rtg:
            inputs = th.stack([pre, obs, action, post],
                              dim=2).reshape(-1, seq_len * 4, self.n_embd)
        else:
            inputs = th.stack([obs, action, post],
                              dim=2).reshape(-1, seq_len * 3, self.n_embd)
        outputs = self.DEM(inputs_embeds=inputs)['last_hidden_state']
        mask = mask.squeeze(-1)
        flipped_mask = th.flip(mask, dims=[1])
        indices = th.argmax(flipped_mask, dim=1)
        last_indices = seq_len - 1 - indices
        if self.args.use_rtg:
            z_o = outputs[:, 1::4]
            z_a = outputs[:, 2::4]
            z_rtg = outputs[:, 0::4]
        else:
            z_o = outputs[:, 0::3]
            z_a = outputs[:, 1::3]
            z_post = outputs[:, 2::3]

        if seq_len == 1 or not self.args.train_CEM:
            if self.args.use_rtg:
                return z_o[th.arange(batch_size), last_indices].detach() + z_rtg[th.arange(batch_size), last_indices].detach(), th.zeros(batch_size, 1)
            else:
                return z_o[th.arange(batch_size), last_indices].detach(), th.zeros(batch_size, 1)
        else:
            ind_reward = self.EM_pred_reward(
                z_a)[th.arange(batch_size), last_indices - 1].detach()
            if self.args.use_rtg:
                return z_o[th.arange(batch_size), last_indices].detach() + z_rtg[th.arange(batch_size), last_indices].detach() + z_a[th.arange(batch_size), last_indices - 1].detach(), ind_reward
            else:
                return z_o[th.arange(batch_size), last_indices].detach() + z_post[th.arange(batch_size), last_indices - 1].detach() + z_a[th.arange(batch_size), last_indices - 1].detach(), ind_reward

    def train_DT(self, context_pre, context_obs, context_action, context_post, context_mask, seq_pre, seq_obs, seq_action, seq_post, seq_mask):
        action_out = self.forward(context_pre, context_obs, context_action, context_post,
                                  context_mask, seq_pre, seq_obs, seq_action, seq_post, seq_mask)
        idx_mask = th.nonzero(seq_mask[:, -1].reshape(-1)).reshape(-1)
        action_out = action_out[idx_mask]
        target_action = seq_action[:, -1][idx_mask]
        DT_loss = F.cross_entropy(
            action_out, target_action, label_smoothing=0.1)
        return DT_loss

    def forward(self, context_pre, context_obs, context_action, context_post, context_mask, seq_pre, seq_obs, seq_action, seq_post, seq_mask=None):
        batch_size, seq_len = seq_pre.shape[0], seq_pre.shape[1]
        if seq_mask is None:
            seq_mask = th.ones(batch_size, seq_len, 1).to(seq_pre.device)
        inputs_pre = th.cat([context_pre.reshape(
            batch_size, self.args.k * self.traj_len, -1), seq_pre], dim=1)
        inputs_obs = th.cat([context_obs.reshape(
            batch_size, self.args.k * self.traj_len, -1), seq_obs], dim=1)
        inputs_action = th.cat([context_action.reshape(
            batch_size, self.args.k * self.traj_len), seq_action], dim=1)
        inputs_post = th.cat([context_post.reshape(
            batch_size, self.args.k * self.traj_len, -1), seq_post], dim=1)
        inputs_mask = th.cat([context_mask.reshape(
            batch_size, self.args.k * self.traj_len, -1), seq_mask], dim=1).squeeze(-1)
        inputs_mask = th.repeat_interleave(inputs_mask, repeats=4, dim=1)
        pre = self.DT_pre_embd(inputs_pre)
        obs = self.DT_obs_embd(inputs_obs)
        action = self.DT_action_embd(inputs_action)
        post = self.DT_post_embd(inputs_post)
        inputs = th.stack([pre, obs, action, post], dim=2).reshape(-1,
                                                                   4 * (self.args.k * self.traj_len + seq_len), self.n_embd)
        transformer_outputs = self.DT(inputs_embeds=inputs, attention_mask=inputs_mask)[
            'last_hidden_state'][:, -3]
        action_out = self.DT_pred_actions(transformer_outputs)
        return action_out
