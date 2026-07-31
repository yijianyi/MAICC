from __future__ import annotations

import random

import torch as th
from torch.cuda.amp import autocast

from .vendor.ic_learner import ICLearner as OriginalICLearner


def _prepare_batch(batch, max_return):
    obss = batch["obs"][:, :-1]
    rewards = batch["reward"][:, :-1]
    actions = batch["actions"][:, :-1]
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    _, seq_len, _ = obss.shape[:3]

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
    seq_completion = seq_ret > (max_return - 1e-1)
    seq_post = th.cat(
        [rewards, terminated, seq_completion],
        dim=-1,
    )
    return seq_rtg, obss, seq_action, seq_post, seq_ret, mask


class AccumulatingOriginalICLearner(OriginalICLearner):
    """Memory-only execution change with the original effective update.

    A complete MAICC batch is sampled by the untouched run loop. The batch is
    split only while constructing/backpropagating the dense model graph, and
    Adam is stepped once after the weighted gradients for the full effective
    batch have accumulated.
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.micro_batch_size = int(
            getattr(args, "baseline_micro_batch_size", 0)
        )
        if self.micro_batch_size <= 0:
            raise ValueError(
                "baseline_micro_batch_size must be positive for the "
                "accumulating learner"
            )

    def _slices(self, batch_size):
        for start in range(0, batch_size, self.micro_batch_size):
            yield start, min(start + self.micro_batch_size, batch_size)

    def train(self, batch, t_env, episode_num, buffer, train_mode):
        if self.micro_batch_size >= batch.batch_size:
            return super().train(
                batch,
                t_env,
                episode_num,
                buffer,
                train_mode,
            )
        if train_mode == 0:
            return self._train_stage0(batch, t_env)
        assert train_mode == 1
        return self._train_stage1(batch, t_env, buffer)

    def _train_stage0(self, batch, t_env):
        self.mac.agent.train()
        effective_size = int(batch.batch_size)
        names = (
            "CEM_action_loss",
            "CEM_reward_loss",
            "CEM_obs_loss",
            "DEM_action_loss",
            "DEM_obs_loss",
            "KL_action_loss",
            "KL_reward_loss",
            "KL_obs_loss",
        )
        totals = {name: 0.0 for name in names}
        self.optimiser.zero_grad()
        for start, end in self._slices(effective_size):
            micro = batch[start:end]
            (
                seq_rtg,
                seq_obs,
                seq_action,
                seq_post,
                _seq_ret,
                mask,
            ) = _prepare_batch(micro, self.args.max_return)
            weight = float(end - start) / effective_size
            with autocast():
                losses = self.mac.agent.train_EM(
                    seq_rtg,
                    seq_obs,
                    seq_action,
                    seq_post,
                    mask,
                )
                if self.args.train_CEM:
                    optimise_loss = (
                        losses[0]
                        + losses[1]
                        + losses[2]
                        + losses[5]
                        + losses[6]
                        + losses[7]
                    )
                else:
                    optimise_loss = losses[3] + losses[4]
            self.scaler.scale(optimise_loss * weight).backward()
            for name, value in zip(names, losses):
                totals[name] += (
                    float(value.detach().item()) * weight
                    if th.is_tensor(value)
                    else float(value) * weight
                )
        self.scaler.step(self.optimiser)
        self.scaler.update()
        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if self.args.train_CEM:
                for name in (
                    "CEM_action_loss",
                    "CEM_reward_loss",
                    "CEM_obs_loss",
                    "KL_action_loss",
                    "KL_reward_loss",
                    "KL_obs_loss",
                ):
                    self.logger.log_stat(name, totals[name], t_env)
            self.logger.log_stat(
                "DEM_action_loss",
                totals["DEM_action_loss"],
                t_env,
            )
            self.logger.log_stat(
                "DEM_obs_loss",
                totals["DEM_obs_loss"],
                t_env,
            )
            self.log_stats_t = t_env

    def _train_stage1(self, batch, t_env, buffer):
        self.mac.agent.train()
        effective_size = int(batch.batch_size)
        seq_len = int(batch["obs"].shape[1] - 1)
        n_agents = int(batch["obs"].shape[2])
        slices = list(self._slices(effective_size))
        prepared = [
            _prepare_batch(batch[start:end], self.args.max_return)
            for start, end in slices
        ]
        self.optimiser.zero_grad()
        total_dt_loss = 0.0

        # The upstream code draws one prefix length per agent, shared by every
        # row in the effective batch. Keep that ordering while splitting only
        # the model graph.
        max_rand_t = seq_len
        if (
            getattr(self.args, "baseline_padding_mode", "strict")
            == "terminal_zero_fill"
        ):
            max_rand_t = min(
                int(values[5].squeeze(-1).sum(dim=1).min().item())
                for values in prepared
            )
            if max_rand_t < 1:
                raise ValueError(
                    "Stage-1 padded batch contains an empty trajectory"
                )
        for agent_id in range(n_agents):
            rand_t = random.randint(1, max_rand_t)
            for (start, end), values in zip(slices, prepared):
                (
                    seq_rtg,
                    seq_obs,
                    seq_action,
                    seq_post,
                    seq_ret,
                    mask,
                ) = values
                query_embeddings = (
                    self.mac.agent.get_dem_traj_embedding(
                        seq_rtg[:, :rand_t],
                        seq_obs[:, :rand_t, agent_id],
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
                        context_rtg[:, timestep - 1]
                        - cr[:, timestep - 1]
                    )
                context_action = ca.squeeze(-1)
                context_ret = th.zeros_like(cr)
                context_ret[:, 0] = cr[:, 0]
                for timestep in range(1, context_len):
                    context_ret[:, timestep] = (
                        context_ret[:, timestep - 1]
                        + cr[:, timestep]
                    )
                context_completion = context_ret > (
                    self.args.max_return - 1e-1
                )
                context_post = th.cat(
                    [cr, ct, context_completion],
                    dim=-1,
                )
                with autocast():
                    loss = self.mac.agent.train_DT(
                        context_rtg,
                        co,
                        context_action,
                        context_post,
                        cm,
                        seq_rtg[:, :rand_t],
                        seq_obs[:, :rand_t, agent_id],
                        seq_action[:, :rand_t, agent_id],
                        seq_post[:, :rand_t],
                        mask[:, :rand_t],
                    )
                weight = float(end - start) / effective_size
                self.scaler.scale(loss * weight).backward()
                total_dt_loss += float(loss.detach().item()) * weight

        self.scaler.step(self.optimiser)
        self.scaler.update()
        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(
                "DT_loss",
                total_dt_loss / n_agents,
                t_env,
            )
            self.log_stats_t = t_env
