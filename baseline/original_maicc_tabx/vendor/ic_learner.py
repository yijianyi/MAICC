import random
import torch as th
from torch.optim import Adam
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from components.episode_buffer import EpisodeBatch


class ICLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = mac.input_shape
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.scaler = GradScaler()

        self.training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, buffer, train_mode):
        self.mac.agent.train()
        # Get the relevant quantities
        obss = batch["obs"][:, :-1]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        batch_size, seq_len, n_agents = obss.shape[0], obss.shape[1], obss.shape[2]
        # pre
        seq_rtg = th.zeros_like(rewards)
        seq_rtg[:, 0] = rewards.sum(1)
        for t in range(1, seq_len):
            seq_rtg[:, t] = seq_rtg[:, t-1] - rewards[:, t-1]

        # obs
        seq_obs = obss

        # action
        seq_action = actions.squeeze(-1)

        # post
        seq_reward = rewards
        seq_term = terminated
        seq_ret = th.zeros_like(rewards)
        seq_ret[:, 0] = rewards[:, 0]
        for t in range(1, seq_len):
            seq_ret[:, t] = seq_ret[:, t-1] + rewards[:, t]
        seq_completion = (seq_ret > (self.args.max_return - 1e-1))
        seq_post = th.cat([seq_reward, seq_term, seq_completion], dim=-1)
        if train_mode == 0:
            with autocast():
                CEM_action_loss, CEM_reward_loss, CEM_obs_loss, DEM_action_loss, DEM_obs_loss, KL_action_loss, KL_reward_loss, KL_obs_loss = self.mac.agent.train_EM(
                    seq_rtg, seq_obs, seq_action, seq_post, mask)
                if self.args.train_CEM:
                    EM_loss = CEM_action_loss + CEM_reward_loss + CEM_obs_loss + \
                        KL_action_loss + KL_reward_loss + KL_obs_loss
                else:
                    EM_loss = DEM_action_loss + DEM_obs_loss
            self.optimiser.zero_grad()
            self.scaler.scale(EM_loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()

            self.training_steps += 1
            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                if self.args.train_CEM:
                    self.logger.log_stat(
                        "CEM_action_loss", CEM_action_loss.item(), t_env)
                    self.logger.log_stat(
                        "CEM_reward_loss", CEM_reward_loss.item(), t_env)
                    self.logger.log_stat(
                        "CEM_obs_loss", CEM_obs_loss.item(), t_env)
                    self.logger.log_stat(
                        "KL_action_loss", KL_action_loss.item(), t_env)
                    self.logger.log_stat(
                        "KL_reward_loss", KL_reward_loss.item(), t_env)
                    self.logger.log_stat(
                        "KL_obs_loss", KL_obs_loss.item(), t_env)
                self.logger.log_stat(
                    "DEM_action_loss", DEM_action_loss.item(), t_env)
                self.logger.log_stat(
                    "DEM_obs_loss", DEM_obs_loss.item(), t_env)
                self.log_stats_t = t_env
            return
        else:
            assert (train_mode == 1)
            DT_loss = 0
            if self.args.env == "sc2v2":
                seq_smac_obs = buffer.process_smac_obs(seq_obs)
            for n in range(n_agents):
                rand_t = random.randint(1, seq_len)
                if self.args.env == "sc2v2":
                    query_embeddings = self.mac.agent.get_dem_traj_embedding(
                        seq_rtg[:, :rand_t], seq_smac_obs[:, :rand_t, n], seq_action[:, :rand_t, n], seq_post[:, :rand_t], mask[:, :rand_t])[0].cpu().numpy()
                else:
                    query_embeddings = self.mac.agent.get_dem_traj_embedding(
                        seq_rtg[:, :rand_t], seq_obs[:, :rand_t, n], seq_action[:, :rand_t, n], seq_post[:, :rand_t], mask[:, :rand_t])[0].cpu().numpy()
                co, cr, ca, ct, cm = buffer.mips(
                    query_embeddings, rand_t - 1, query_return=seq_ret[:, -1])
                context_rtg = th.zeros_like(cr)
                context_rtg[:, 0] = cr.sum(1)
                for t in range(1, seq_len):
                    context_rtg[:, t] = context_rtg[:, t-1] - cr[:, t-1]
                context_obs = co
                context_a = ca.squeeze(-1)
                context_ret = th.zeros_like(cr)
                context_ret[:, 0] = cr[:, 0]
                for t in range(1, seq_len):
                    context_ret[:, t] = context_ret[:, t-1] + cr[:, t]
                context_completion = (context_ret > (
                    self.args.max_return - 1e-1))
                context_post = th.cat([cr, ct, context_completion], dim=-1)
                with autocast():
                    if self.args.env == "sc2v2":
                        DT_loss += self.mac.agent.train_DT(context_rtg, context_obs, context_a, context_post, cm,
                                                           seq_rtg[:, :rand_t], seq_smac_obs[:, :rand_t, n], seq_action[:, :rand_t, n], seq_post[:, :rand_t], mask[:, :rand_t])
                    else:
                        DT_loss += self.mac.agent.train_DT(context_rtg, context_obs, context_a, context_post, cm,
                                                           seq_rtg[:, :rand_t], seq_obs[:, :rand_t, n], seq_action[:, :rand_t, n], seq_post[:, :rand_t], mask[:, :rand_t])
            self.optimiser.zero_grad()
            self.scaler.scale(DT_loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.training_steps += 1
            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                self.logger.log_stat(
                    "DT_loss", DT_loss.item() / n_agents, t_env)
                self.log_stats_t = t_env
            return

    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path),
                    map_location=lambda storage, loc: storage)
        )
