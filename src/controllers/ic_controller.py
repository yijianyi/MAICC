from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn.functional as F


# This multi-agent controller shares parameters between agents
class ICMAC:
    def __init__(self, env_info, buffer, args):
        self.n_agents = args.n_agents
        self.args = args
        self.k = self.args.k
        self.env_info = env_info
        self.input_shape = self._get_input_shape(env_info)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type
        self.buffer = buffer

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(
            ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        assert (test_mode == True)
        self.agent.eval()
        self._build_inputs(ep_batch, t)
        present = self.seq_present[:, -1, 0].bool()
        active_ids = th.nonzero(present, as_tuple=False).reshape(-1)
        if active_ids.numel() == 0:
            raise RuntimeError("Online action selection has no present agents")
        active_pre = self.seq_pre[active_ids]
        active_obs = self.seq_obs[active_ids]
        active_action = self.seq_action[active_ids]
        active_post = self.seq_post[active_ids]
        for label, values in (
            ("rtg", active_pre),
            ("observation", active_obs),
            ("post", active_post),
        ):
            if values.is_floating_point() and not bool(
                th.isfinite(values).all().item()
            ):
                invalid = int((~th.isfinite(values)).sum().item())
                raise RuntimeError(
                    f"Non-finite active-agent {label} before DEM: "
                    f"shape={tuple(values.shape)} invalid_values={invalid}"
                )
        if self.args.env == "sc2v2":
            ori_dim = active_obs.shape[-1]
            seq_smac_obs = F.pad(active_obs, (0, self.input_shape - ori_dim))
            query_embeddings, _ = self.agent.get_dem_traj_embedding(
                active_pre, seq_smac_obs, active_action.squeeze(-1), active_post)
        else:
            query_embeddings, _ = self.agent.get_dem_traj_embedding(
                active_pre, active_obs, active_action.squeeze(-1), active_post)
        query_embeddings = query_embeddings.cpu().numpy()
        retrieval = self.buffer.mips(
            query_embeddings, t, test_mode=test_mode)
        if len(retrieval) == 7:
            co, cr, ca, ct, cm, context_rtg, context_completion = retrieval
        else:
            co, cr, ca, ct, cm = retrieval
            seq_len = co.shape[1]
            context_rtg = th.zeros_like(cr)
            context_rtg[:, 0] = cr.sum(1)
            for tt in range(1, seq_len):
                context_rtg[:, tt] = context_rtg[:, tt-1] - cr[:, tt-1]
            context_ret = th.cumsum(cr, dim=1)
            context_completion = (
                context_ret > (self.args.max_return - 1e-1)
            )
        context_obs = co
        context_a = ca.squeeze(-1)
        context_post = th.cat([cr, ct, context_completion], dim=-1)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.env == "sc2v2":
            active_outs = self.agent(context_rtg, context_obs, context_a, context_post, cm,
                                     active_pre, seq_smac_obs, active_action.squeeze(-1), active_post)
        else:
            active_outs = self.agent(context_rtg, context_obs, context_a, context_post, cm,
                                     active_pre, active_obs, active_action.squeeze(-1), active_post)

        # Keep the fixed-capacity agent dimension required by EpisodeBatch,
        # while ensuring padded/nonexistent slots never enter DEM retrieval or
        # DT inference. Their only valid online action is IDLE.
        agent_outs = th.full(
            (self.n_agents, active_outs.shape[-1]),
            -1e10,
            dtype=active_outs.dtype,
            device=active_outs.device,
        )
        agent_outs[active_ids] = active_outs
        idle_action = int(getattr(self.agent, "idle_action_id", 7))
        if 0 <= idle_action < agent_outs.shape[-1]:
            agent_outs[~present, idle_action] = 0.0

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = F.pad(reshaped_avail_actions, (
                    0, self.agent.max_actions - reshaped_avail_actions.shape[1], 0, 0), mode="constant", value=0)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.seq_pre = None
        self.seq_obs = None
        self.seq_action = None
        self.seq_post = None
        self.seq_present = None

    def parameters(self):
        return self.agent.parameters()

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        checkpoint = "{}/agent.th".format(path)
        try:
            state_dict = th.load(
                checkpoint,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
        except TypeError:
            # PyTorch versions before weights_only was introduced.
            state_dict = th.load(
                checkpoint, map_location=lambda storage, loc: storage
            )
        self.agent.load_state_dict(state_dict)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](
            input_shape, self.args, self.env_info)

    def _build_inputs(self, batch, t):
        if self.seq_pre is None:
            target = float(
                getattr(self.args, "online_target_rtg", self.args.max_return)
            )
            rtg = th.ones(
                self.n_agents, 1, 1, dtype=th.float32
            ).to(batch.device) * target
            self.seq_pre = rtg
        else:
            reward = batch["reward"][:, t-1].expand(self.n_agents, 1, 1)
            reward_scale = float(getattr(self.args, "online_return_scale", 1.0))
            rtg = self.seq_pre[:, -1:] - reward / max(reward_scale, 1e-6)
            self.seq_pre = th.cat([self.seq_pre, rtg], dim=1)

        obs = batch["obs"][:, t].reshape(self.n_agents, 1, -1)
        if self.seq_obs is None:
            self.seq_obs = obs
        else:
            self.seq_obs = th.cat([self.seq_obs, obs], dim=1)

        placeholder = int(
            getattr(self.agent, "unknown_action_id", 0) or 0
        )
        action = th.full(
            (self.n_agents, 1, 1), placeholder,
            dtype=th.long, device=batch.device,
        )
        if self.seq_action is None:
            self.seq_action = action
        else:
            self.seq_action = th.cat([self.seq_action, action], dim=1)

        post = th.zeros(self.n_agents, 1, 3).to(th.float32).to(batch.device)
        if self.seq_post is None:
            self.seq_post = post
        else:
            self.seq_post = th.cat([self.seq_post, post], dim=1)

        if "present" in batch.data.transition_data:
            present = batch["present"][:, t].reshape(
                self.n_agents, 1, 1
            ).bool()
        else:
            present = th.ones(
                self.n_agents, 1, 1, dtype=th.bool, device=batch.device
            )
        if self.seq_present is None:
            self.seq_present = present
        else:
            self.seq_present = th.cat([self.seq_present, present], dim=1)

        # Online episodes may be longer than the transformer training window.
        # Keep exactly the manifest-configured rolling history at inference.
        history = int(getattr(self.args, "context_length", 32))
        self.seq_pre = self.seq_pre[:, -history:]
        self.seq_obs = self.seq_obs[:, -history:]
        self.seq_action = self.seq_action[:, -history:]
        self.seq_post = self.seq_post[:, -history:]
        self.seq_present = self.seq_present[:, -history:]

    def update_context(self, batch, t):
        action = batch["actions"][:, t].reshape(self.n_agents, 1, 1)
        self.seq_action[:, -1:] = action
        reward = batch["reward"][:, t].expand(self.n_agents, 1, 1)
        term = batch["terminated"][:, t].expand(self.n_agents, 1, 1)
        if "completion" in batch.data.transition_data:
            completion = batch["completion"][:, t].expand(
                self.n_agents, 1, 1
            ).float()
        else:
            completion = th.zeros_like(reward)
        self.seq_post[:, -1:] = th.cat([reward, term, completion], dim=-1)

    def _get_input_shape(self, env_info):
        if self.args.env == "sc2v2":
            input_shape = 227
        else:
            input_shape = env_info["obs_shape"]
        assert (self.args.obs_agent_id is False)
        return input_shape
