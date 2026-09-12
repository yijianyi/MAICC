import math
import random

import torch as th
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch


class ICLearner:
    """MAICC learner with TABX causal/masking semantics.

    TABX encoder modules (CEM/DEM) use args.use_rtg=False. The policy DT keeps
    its original four-token [RTG, obs, action, post] structure and consumes the
    full-episode RTG supplied by the converted batch.
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = mac.input_shape
        self.mac = mac
        self.logger = logger
        self.training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.use_amp = bool(getattr(args, "use_amp", False) and args.use_cuda)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.train_mode = None
        self.configure_train_mode(0)

    def configure_train_mode(self, train_mode: int) -> None:
        """Select an explicit stage and rebuild its isolated optimiser."""
        if train_mode not in (0, 1):
            raise ValueError(f"Unsupported MAICC train_mode={train_mode}")
        self.train_mode = train_mode
        for name, parameter in self.mac.agent.named_parameters():
            is_dt = name == "DT" or name.startswith("DT.") or name.startswith("DT_")
            parameter.requires_grad_(is_dt if train_mode == 1 else not is_dt)
        self.params = [p for p in self.mac.parameters() if p.requires_grad]
        if not self.params:
            raise RuntimeError(f"No trainable parameters for train_mode={train_mode}")
        self.optimiser = Adam(params=self.params, lr=self.args.lr)

    def _optimise(self, loss: th.Tensor) -> None:
        if not th.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.detach().cpu()}")
        self.optimiser.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimiser)
        grad_norm = clip_grad_norm_(
            self.params, float(getattr(self.args, "grad_norm_clip", 10.0))
        )
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
        self.scaler.step(self.optimiser)
        self.scaler.update()

    def _sequences(self, batch: EpisodeBatch):
        obs = batch["obs"][:, :-1]
        reward = batch["reward"][:, :-1]
        action = batch["actions"][:, :-1].squeeze(-1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] *= 1.0 - terminated[:, :-1]

        if self.args.env == "tabx":
            rtg = batch["rtg"][:, :-1].float()
            completion = batch["completion"][:, :-1].float()
            present = batch["present"][:, :-1].float()
        else:
            rtg = th.zeros_like(reward)
            rtg[:, 0] = reward.sum(1)
            for t in range(1, reward.shape[1]):
                rtg[:, t] = rtg[:, t - 1] - reward[:, t - 1]
            returns = th.cumsum(reward, dim=1)
            completion = (returns > (self.args.max_return - 1e-1)).float()
            present = th.ones_like(obs[..., :1])

        post = th.cat([reward, terminated, completion], dim=-1)
        returns = th.cumsum(reward, dim=1)
        return rtg, obs, action, post, returns, mask, present

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, buffer, train_mode):
        if train_mode != self.train_mode:
            self.configure_train_mode(train_mode)
        if train_mode == 1:
            # A stationary FAISS space requires deterministic frozen encoders,
            # not just requires_grad=False. Keep dropout only in the policy.
            self.mac.agent.eval()
            self.mac.agent.DT.train()
            for name, module in self.mac.agent.named_children():
                if name.startswith("DT_"):
                    module.train()
        else:
            self.mac.agent.train()
        seq_rtg, seq_obs, seq_action, seq_post, seq_ret, mask, present = (
            self._sequences(batch)
        )
        _batch_size, seq_len, n_agents = seq_obs.shape[:3]
        policy_obs = (
            buffer.process_smac_obs(seq_obs)
            if self.args.env == "sc2v2" and buffer is not None
            else seq_obs
        )

        if train_mode == 0:
            with autocast(enabled=self.use_amp):
                losses = self.mac.agent.train_EM(
                    seq_rtg, seq_obs, seq_action, seq_post, mask,
                    present_mask=present,
                )
                (
                    cem_action, cem_reward, cem_obs, dem_action, dem_obs,
                    kl_action, kl_reward, kl_obs,
                ) = losses
                named_losses = {
                    "CEM_action_loss": cem_action,
                    "CEM_reward_loss": cem_reward,
                    "CEM_obs_loss": cem_obs,
                    "DEM_action_loss": dem_action,
                    "DEM_obs_loss": dem_obs,
                    "KL_action_loss": kl_action,
                    "KL_reward_loss": kl_reward,
                    "KL_obs_loss": kl_obs,
                }
                bad_losses = {
                    name: value
                    for name, value in named_losses.items()
                    if isinstance(value, th.Tensor) and not th.isfinite(value)
                }
                if bad_losses:
                    details = ", ".join(
                        f"{name}={value.detach().cpu()}"
                        for name, value in bad_losses.items()
                    )
                    valid_times = int(mask.sum().detach().cpu())
                    valid_agent_steps = int(
                        (mask.unsqueeze(2) * present).sum().detach().cpu()
                    )
                    raise FloatingPointError(
                        "Non-finite Stage-0 loss component(s): "
                        f"{details}; valid_times={valid_times}, "
                        f"valid_agent_steps={valid_agent_steps}"
                    )
                if self.args.train_CEM:
                    loss = (
                        cem_action + cem_reward + cem_obs
                        + kl_action + kl_reward + kl_obs
                    )
                else:
                    loss = dem_action + dem_obs
            self._optimise(loss)
            self.training_steps += 1
            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                if self.args.train_CEM:
                    for key, value in (
                        ("CEM_action_loss", cem_action),
                        ("CEM_reward_loss", cem_reward),
                        ("CEM_obs_loss", cem_obs),
                        ("KL_action_loss", kl_action),
                        ("KL_reward_loss", kl_reward),
                        ("KL_obs_loss", kl_obs),
                    ):
                        self.logger.log_stat(key, float(value), t_env)
                self.logger.log_stat("DEM_action_loss", float(dem_action), t_env)
                self.logger.log_stat("DEM_obs_loss", float(dem_obs), t_env)
                self.log_stats_t = t_env
            return {
                name: (
                    float(value.detach().cpu())
                    if isinstance(value, th.Tensor)
                    else float(value)
                )
                for name, value in named_losses.items()
            }

        if buffer is None:
            raise RuntimeError("DT training requires a retrieval store")
        losses = []
        for agent_id in range(n_agents):
            agent_mask = mask * present[:, :, agent_id]
            valid_lengths = agent_mask.squeeze(-1).sum(1).long()
            max_valid = int(valid_lengths.max().item())
            if max_valid <= 0:
                continue
            rand_t = random.randint(1, min(seq_len, max_valid))
            query_mask = agent_mask[:, :rand_t]
            with th.no_grad():
                query_embedding = self.mac.agent.get_dem_traj_embedding(
                    seq_rtg[:, :rand_t], policy_obs[:, :rand_t, agent_id],
                    seq_action[:, :rand_t, agent_id], seq_post[:, :rand_t],
                    query_mask,
                )[0].cpu().numpy()
            retrieval = buffer.mips(
                query_embedding, rand_t - 1, query_return=seq_ret[:, -1]
            )
            if len(retrieval) == 7:
                co, cr, ca, ct, cm, context_rtg, context_completion = retrieval
            else:
                co, cr, ca, ct, cm = retrieval
                context_rtg = th.zeros_like(cr)
                context_rtg[:, 0] = cr.sum(1)
                for context_t in range(1, cr.shape[1]):
                    context_rtg[:, context_t] = (
                        context_rtg[:, context_t - 1] - cr[:, context_t - 1]
                    )
                context_ret = th.cumsum(cr, dim=1)
                context_completion = (
                    context_ret > (self.args.max_return - 1e-1)
                ).float()
            context_post = th.cat([cr, ct, context_completion], dim=-1)
            with autocast(enabled=self.use_amp):
                agent_loss = self.mac.agent.train_DT(
                    context_rtg, co, ca.squeeze(-1), context_post, cm,
                    seq_rtg[:, :rand_t], policy_obs[:, :rand_t, agent_id],
                    seq_action[:, :rand_t, agent_id], seq_post[:, :rand_t],
                    query_mask,
                )
            losses.append(agent_loss)

        if not losses:
            raise RuntimeError("No present TABX agents were available for DT loss")
        # Mean over valid agents keeps gradient scale stable from 10 to 20 slots.
        dt_loss = th.stack(losses).mean()
        self._optimise(dt_loss)
        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("DT_loss", float(dt_loss), t_env)
            self.log_stats_t = t_env
        return {"DT_loss": float(dt_loss.detach().cpu())}

    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")
        th.save({"train_mode": self.train_mode}, f"{path}/learner_meta.th")

    def load_models(self, path):
        self.mac.load_models(path)
        meta_path = f"{path}/learner_meta.th"
        try:
            meta = th.load(meta_path, map_location="cpu")
            self.configure_train_mode(int(meta.get("train_mode", self.train_mode)))
        except FileNotFoundError:
            pass
        state = th.load(
            f"{path}/opt.th", map_location=lambda storage, loc: storage
        )
        try:
            self.optimiser.load_state_dict(state)
        except ValueError:
            # Legacy MAICC checkpoints used one optimiser over every module;
            # model weights remain loadable, but their optimiser groups cannot
            # be mapped safely to the explicit stage optimiser.
            print("[checkpoint] skipped incompatible legacy optimiser state")
