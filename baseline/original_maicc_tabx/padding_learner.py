from __future__ import annotations

import random

import torch as th
from torch.cuda.amp import autocast

from .vendor.ic_learner import ICLearner as OriginalICLearner


class PaddingCompatibleOriginalICLearner(OriginalICLearner):
    """Original learner with the minimum Stage-1 padding safety change.

    Stage-0 delegates byte-for-byte to the frozen learner. Stage-1 retains the
    upstream loss, optimiser, AMP, retrieval call, and logging, but samples the
    shared query prefix only from timesteps filled for every row in the batch.
    """

    def train(self, batch, t_env, episode_num, buffer, train_mode):
        if train_mode == 0:
            return super().train(
                batch,
                t_env,
                episode_num,
                buffer,
                train_mode,
            )

        assert train_mode == 1
        self.mac.agent.train()
        obss = batch["obs"][:, :-1]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        _, seq_len, n_agents = obss.shape[:3]
        seq_rtg = th.zeros_like(rewards)
        seq_rtg[:, 0] = rewards.sum(1)
        for timestep in range(1, seq_len):
            seq_rtg[:, timestep] = (
                seq_rtg[:, timestep - 1] - rewards[:, timestep - 1]
            )
        seq_action = actions.squeeze(-1)
        seq_ret = th.zeros_like(rewards)
        seq_ret[:, 0] = rewards[:, 0]
        for timestep in range(1, seq_len):
            seq_ret[:, timestep] = (
                seq_ret[:, timestep - 1] + rewards[:, timestep]
            )
        seq_completion = seq_ret > (self.args.max_return - 1e-1)
        seq_post = th.cat(
            [rewards, terminated, seq_completion],
            dim=-1,
        )

        valid_lengths = mask.squeeze(-1).sum(dim=1).long()
        max_rand_t = int(valid_lengths.min().item())
        if max_rand_t < 1:
            raise ValueError(
                "Stage-1 padded batch contains an empty trajectory"
            )

        dt_loss = 0
        if self.args.env == "sc2v2":
            seq_smac_obs = buffer.process_smac_obs(obss)
        for agent_id in range(n_agents):
            rand_t = random.randint(1, max_rand_t)
            if self.args.env == "sc2v2":
                query_obs = seq_smac_obs[:, :rand_t, agent_id]
            else:
                query_obs = obss[:, :rand_t, agent_id]
            query_embeddings = (
                self.mac.agent.get_dem_traj_embedding(
                    seq_rtg[:, :rand_t],
                    query_obs,
                    seq_action[:, :rand_t, agent_id],
                    seq_post[:, :rand_t],
                    mask[:, :rand_t],
                )[0]
                .cpu()
                .numpy()
            )
            co, cr, ca, ct, cm = buffer.mips(
                query_embeddings,
                rand_t - 1,
                query_return=seq_ret[:, -1],
            )
            context_rtg = th.zeros_like(cr)
            context_rtg[:, 0] = cr.sum(1)
            context_len = int(cr.shape[1])
            for timestep in range(1, context_len):
                context_rtg[:, timestep] = (
                    context_rtg[:, timestep - 1] - cr[:, timestep - 1]
                )
            context_action = ca.squeeze(-1)
            context_ret = th.zeros_like(cr)
            context_ret[:, 0] = cr[:, 0]
            for timestep in range(1, context_len):
                context_ret[:, timestep] = (
                    context_ret[:, timestep - 1] + cr[:, timestep]
                )
            context_completion = context_ret > (
                self.args.max_return - 1e-1
            )
            context_post = th.cat(
                [cr, ct, context_completion],
                dim=-1,
            )
            with autocast():
                dt_loss += self.mac.agent.train_DT(
                    context_rtg,
                    co,
                    context_action,
                    context_post,
                    cm,
                    seq_rtg[:, :rand_t],
                    query_obs,
                    seq_action[:, :rand_t, agent_id],
                    seq_post[:, :rand_t],
                    mask[:, :rand_t],
                )

        self.optimiser.zero_grad()
        self.scaler.scale(dt_loss).backward()
        self.scaler.step(self.optimiser)
        self.scaler.update()
        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(
                "DT_loss",
                dt_loss.item() / n_agents,
                t_env,
            )
            self.log_stats_t = t_env
