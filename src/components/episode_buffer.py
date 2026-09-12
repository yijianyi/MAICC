import torch as th
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace as SN
import faiss
from copy import deepcopy
from .epsilon_schedules import DecayThenFlatSchedule
from .transforms import OneHot


class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups,
                             batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(
                field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(
                    group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros(
                    (batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros(
                    (batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups,
                         self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError(
                    "{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            if type(v) == list:
                v = th.tensor(np.array(v), dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError(
                        "Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size,
                               self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs,
                               ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
                or isinstance(items, int)  # int i
                # [a,b,c]
                or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))
                ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            # TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size,
                                           max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index +
                              ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(
                self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(
                self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


class InContextReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, args, max_seq_length, preprocess=None, device="cpu"):
        super(InContextReplayBuffer, self).__init__(scheme, groups,
                                                    args.test_nepisode, max_seq_length, preprocess=preprocess, device=device)
        self.offline_buffers = []
        if args.env == "gymma":
            for _ in range(args.task_nums):
                self.offline_buffers.append(EpisodeBatch(scheme, groups, args.trajs_per_task,
                                                         max_seq_length, preprocess=preprocess, device=device))
        elif args.env == "sc2v2":
            self.tasks = ["protoss_5_vs_5", "protoss_7_vs_7", "protoss_10_vs_11", "terran_5_vs_5",
                          "terran_7_vs_7", "terran_10_vs_11", "zerg_5_vs_5", "zerg_7_vs_7", "zerg_10_vs_11"]
            self.nagents2task = {}
            for t in self.tasks:
                race, n_ally, n_enemy = t.split("_")[0], int(
                    t.split("_")[1]), int(t.split("_")[3])

                t_scheme = deepcopy(scheme)
                if race == "protoss":
                    t_scheme["state"].update(
                        {"vshape": 7 * n_enemy + 8 * n_ally + (6 + n_enemy) * n_ally})
                    t_scheme["obs"].update(
                        {"vshape": (n_enemy + n_ally - 1) * 9 + 11})
                else:
                    t_scheme["state"].update(
                        {"vshape": 6 * n_enemy + 7 * n_ally + (6 + n_enemy) * n_ally})
                    t_scheme["obs"].update(
                        {"vshape": (n_enemy + n_ally - 1) * 8 + 10})
                t_scheme["avail_actions"].update({"vshape": (6 + n_enemy, )})
                t_group = {"agents": n_ally}
                t_preprocess = {"actions": (
                    "actions_onehot", [OneHot(out_dim=6 + n_enemy)])}
                self.nagents2task.update({t: n_ally})
                self.offline_buffers.append(EpisodeBatch(t_scheme, t_group, args.trajs_per_task,
                                                         max_seq_length, preprocess=t_preprocess, device=device))
        else:
            assert (0)
        self.args = args
        self.n_agents = args.n_agents
        self.task_nums = args.task_nums
        self.trajs_per_task = args.trajs_per_task
        self.offline_trajs_num = self.task_nums * self.trajs_per_task
        self.buffer_size = args.test_nepisode
        assert (self.offline_trajs_num % 20 == 0)
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.schedule = DecayThenFlatSchedule(1, 0.01, 1, decay="exp")
        self.add_offline_data()

    def add_offline_data(self):
        import h5py
        if self.args.env == "sc2v2":
            for i, t in enumerate(self.tasks):
                hdFile_r = h5py.File("datasets/sc2v2/" + t + ".h5", 'r')
                for key in hdFile_r.keys():
                    value = th.tensor(hdFile_r.get(key)[:])
                    if key == "obs":
                        if t.startswith("protoss"):
                            all_dim = value.shape[-1]
                            ind = 4
                            while ind + 9 < all_dim:
                                value[:, :, :, ind+6:ind+9] /= 3
                                ind += 9
                            value[:, :, :, -3:] /= 3
                        elif t.startswith("terran"):
                            all_dim = value.shape[-1]
                            ind = 4
                            while ind + 8 < all_dim:
                                value[:, :, :, ind+5:ind+8] /= 3/2
                                ind += 8
                            value[:, :, :, -3:] /= 3/2
                    self.offline_buffers[i][key].data[:] = value
                hdFile_r.close()
        elif self.args.env == "gymma":
            if self.args.env_args['time_limit'] == 15:
                hdFile_r = h5py.File(
                    "datasets/" + self.args.env + "_small.h5", 'r')
            else:
                hdFile_r = h5py.File("datasets/" + self.args.env + ".h5", 'r')
            for key in hdFile_r.keys():
                value = th.tensor(hdFile_r.get(key)[:])
                for i in range(self.task_nums):
                    self.offline_buffers[i][key].data[:] = value[i *
                                                                 self.trajs_per_task: (i+1) * self.trajs_per_task]
            hdFile_r.close()
        else:
            assert (0)

    def insert_episode_batch(self, ep_batch):
        assert (ep_batch.batch_size == 1)
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index +
                              ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.update_online_embedding(self.buffer_index)
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(
                self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            assert (0)
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.offline_trajs_num >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        # Sample from a task
        t = np.random.choice(self.task_nums)
        ep_ids = np.random.choice(
            self.trajs_per_task, batch_size, replace=False)
        return self.offline_buffers[t][ep_ids]

    def transfer_id(self, indices):
        if self.args.env == "gymma":
            task_id = indices // (self.n_agents * self.trajs_per_task)
            traj_id = (indices % (self.n_agents *
                                  self.trajs_per_task)) // self.n_agents
            agent_id = (indices % (self.n_agents *
                        self.trajs_per_task)) % self.n_agents
        elif self.args.env == "sc2v2":
            def func_task_id(x):
                trajs = 0
                for i, t in enumerate(self.tasks):
                    trajs += self.nagents2task[t] * self.trajs_per_task
                    if x < trajs:
                        return i

            def func_traj_id(x):
                trajs = 0
                for i, t in enumerate(self.tasks):
                    trajs += self.nagents2task[t] * self.trajs_per_task
                    if x < trajs:
                        trajs -= self.nagents2task[t] * self.trajs_per_task
                        return (x - trajs) // self.nagents2task[t]

            def func_agent_id(x):
                trajs = 0
                for i, t in enumerate(self.tasks):
                    trajs += self.nagents2task[t] * self.trajs_per_task
                    if x < trajs:
                        trajs -= self.nagents2task[t] * self.trajs_per_task
                        return (x - trajs) % self.nagents2task[t]
            vectorized_function1 = np.vectorize(func_task_id)
            vectorized_function2 = np.vectorize(func_traj_id)
            vectorized_function3 = np.vectorize(func_agent_id)
            task_id = vectorized_function1(indices)
            traj_id = vectorized_function2(indices)
            agent_id = vectorized_function3(indices)
        else:
            assert (0)
        return task_id, traj_id, agent_id

    def mips(self, querys, t, query_return=None, test_mode=False):
        if test_mode:
            ratio = self.episodes_in_buffer / self.args.test_nepisode
            if self.episodes_in_buffer < self.args.k:
                epsilon = 1
            else:
                epsilon = self.schedule.eval(ratio)
            random_numbers = th.rand(self.n_agents)

            z_d = []
            for t_id in range(self.task_nums):
                z_d.append(
                    self.offline_key_embeddings[t_id][:, t].reshape(-1, self.args.n_embd))
            z_d = np.concatenate(z_d, axis=0)

            obss = []
            rewards = []
            actions = []
            terminateds = []
            masks = []
            for i in range(self.n_agents):
                qi = deepcopy(querys[i:i+1])
                if epsilon == 1 or random_numbers[i] < epsilon:
                    faiss.normalize_L2(z_d)
                    faiss.normalize_L2(qi)
                    index = faiss.IndexFlatIP(self.args.n_embd)
                    index.add(z_d)
                    D, I = index.search(qi, self.args.l)
                    task_id, traj_id, agent_id = self.transfer_id(I)
                    s_rel = (D + 1) / 2
                    s_ret = self.offline_traj_returns[task_id *
                                                      self.trajs_per_task + traj_id]
                    s_ret = (s_ret - self.args.min_return) / \
                        (self.args.max_return - self.args.min_return)
                    S = s_rel + self.args.weight * s_ret
                    retrieve_indices = np.argsort(-S, axis=1)[:, :self.args.k]
                    traj_indices = np.take_along_axis(
                        I, retrieve_indices, axis=1).reshape(-1)
                    task_id, traj_id, agent_id = self.transfer_id(traj_indices)

                    for j in range(self.args.k):
                        batch = self.offline_buffers[task_id[j]
                                                     ][traj_id[j]:traj_id[j]+1]
                        batch.to(self.args.device)
                        if self.args.env == "sc2v2":
                            smac_obs = self.process_smac_obs(
                                batch["obs"][:, :-1, agent_id[j]:agent_id[j] + 1]).squeeze(0, 2)
                            obss.append(smac_obs)
                        else:
                            obss.append(batch["obs"][0, :-1, agent_id[j]])
                        rewards.append(batch["reward"][0, :-1])
                        actions.append(batch["actions"][0, :-1, agent_id[j]])
                        terminateds.append(batch["terminated"][0, :-1].float())
                        masks.append(batch["filled"][0, :-1].float())
                else:
                    z_d_online = self.online_key_embeddings[:self.episodes_in_buffer, t, i]
                    z_d_online = np.ascontiguousarray(z_d_online)
                    faiss.normalize_L2(z_d_online)
                    faiss.normalize_L2(qi)
                    index = faiss.IndexFlatIP(self.args.n_embd)
                    index.add(z_d_online)
                    D, I = index.search(qi, self.args.l)
                    s_rel = (D + 1) / 2
                    s_ret = self.online_traj_returns[I]
                    s_ret = (s_ret - self.args.min_return) / \
                        (self.args.max_return - self.args.min_return)
                    s_ind_ret = self.online_traj_ind_returns[I,
                                                             i] * self.n_agents
                    s_ind_ret = (s_ind_ret - self.args.min_return) / \
                        (self.args.max_return - self.args.min_return)
                    S = s_rel + self.args.weight * s_ret + self.args.ind_weight * s_ind_ret
                    retrieve_indices = np.argsort(-S,
                                                  axis=1)[:, :self.args.k].reshape(-1)
                    for j in range(self.args.k):
                        traj_id = I[0, retrieve_indices[j]]
                        batch = self[traj_id:traj_id+1]
                        batch.to(self.args.device)
                        if self.args.env == "sc2v2":
                            smac_obs = self.process_smac_obs(
                                batch["obs"][:, :-1, i:i + 1], test_mode=True).squeeze(0, 2)
                            obss.append(smac_obs)
                        else:
                            obss.append(batch["obs"][0, :-1, i])
                        rewards.append(batch["reward"][0, :-1])
                        actions.append(batch["actions"][0, :-1, i])
                        terminateds.append(batch["terminated"][0, :-1].float())
                        masks.append(batch["filled"][0, :-1].float())

            obss = th.stack(obss, dim=0)
            rewards = th.stack(rewards, dim=0)
            actions = th.stack(actions, dim=0)
            terminateds = th.stack(terminateds, dim=0)
            masks = th.stack(masks, dim=0)
            masks[:, 1:] = masks[:, 1:] * (1 - terminateds[:, :-1])
            return obss, rewards, actions, terminateds, masks
        else:
            z_d = []
            for t_id in range(self.task_nums):
                z_d.append(
                    self.offline_key_embeddings[t_id][:, t].reshape(-1, self.args.n_embd))
            z_d = np.concatenate(z_d, axis=0)
            q = deepcopy(querys)
            faiss.normalize_L2(z_d)
            faiss.normalize_L2(q)
            index = faiss.IndexFlatIP(self.args.n_embd)
            index.add(z_d)
            D, I = index.search(q, self.args.l)
            task_id, traj_id, agent_id = self.transfer_id(I)
            s_rel = (D + 1) / 2
            s_ret = np.logical_and(
                D < self.args.max_similarity, self.offline_traj_returns[task_id * self.trajs_per_task + traj_id] <= query_return.cpu().numpy())
            S = s_rel + s_ret
            retrieve_indices = np.argsort(-S, axis=1)[:, :self.args.k]
            traj_indices = np.take_along_axis(I, retrieve_indices, axis=1)
            for i in range(traj_indices.shape[0]):
                traj_indices[i] = np.random.permutation(traj_indices[i])
            traj_indices = traj_indices.reshape(-1)
            task_id, traj_id, agent_id = self.transfer_id(traj_indices)

            obss = []
            rewards = []
            actions = []
            terminateds = []
            masks = []
            for i in range(self.args.batch_size * self.args.k):
                batch = self.offline_buffers[task_id[i]
                                             ][traj_id[i]:traj_id[i]+1]
                batch.to(self.args.device)
                if self.args.env == "sc2v2":
                    smac_obs = self.process_smac_obs(
                        batch["obs"][:, :-1, agent_id[i]:agent_id[i] + 1]).squeeze(0, 2)
                    obss.append(smac_obs)
                else:
                    obss.append(batch["obs"][0, :-1, agent_id[i]])
                rewards.append(batch["reward"][0, :-1])
                actions.append(batch["actions"][0, :-1, agent_id[i]])
                terminateds.append(batch["terminated"][0, :-1].float())
                masks.append(batch["filled"][0, :-1].float())
            obss = th.stack(obss, dim=0)
            rewards = th.stack(rewards, dim=0)
            actions = th.stack(actions, dim=0)
            terminateds = th.stack(terminateds, dim=0)
            masks = th.stack(masks, dim=0)
            masks[:, 1:] = masks[:, 1:] * (1 - terminateds[:, :-1])
            return obss, rewards, actions, terminateds, masks

    def init_embedding(self):
        import os
        import pickle
        emb_path = "datasets/" + self.args.env + "_emb.pkl"
        ret_path = "datasets/" + self.args.env + "_ret.pkl"
        if os.path.exists(emb_path) and os.path.exists(ret_path):
            with open(emb_path, 'rb') as file:
                self.offline_key_embeddings = pickle.load(file)
            with open(ret_path, 'rb') as file:
                self.offline_traj_returns = pickle.load(file)
        else:
            self.agent.eval()
            self.offline_key_embeddings = []
            self.offline_traj_returns = np.zeros((self.offline_trajs_num))
            if self.args.env == "gymma":
                for t_id in range(self.task_nums):
                    self.offline_key_embeddings.append(np.zeros(
                        (self.trajs_per_task, self.args.env_args["time_limit"], self.n_agents, self.args.n_embd)).astype('float32'))
                    for i in range(self.trajs_per_task // 20):
                        batch = self.offline_buffers[t_id][i * 20:(i + 1) * 20]
                        batch.to(self.args.device)
                        seq_rtg, seq_obs, seq_action, seq_post, seq_ret, mask = self.process_batch(
                            batch)
                        self.offline_traj_returns[t_id * self.trajs_per_task + i * 20:t_id * self.trajs_per_task + (i + 1) * 20] = (
                            seq_ret[:, -1, 0]).cpu().numpy()
                        for t in range(self.args.env_args["time_limit"]):
                            for n in range(self.n_agents):
                                emb, ind_r = self.agent.get_dem_traj_embedding(
                                    seq_rtg[:, :t+1], seq_obs[:, :t+1, n], seq_action[:, :t+1, n], seq_post[:, :t+1], mask[:, :t+1])
                                self.offline_key_embeddings[t_id][i *
                                                                  20:(i + 1) * 20, t, n] = emb.cpu().numpy()
            elif self.args.env == "sc2v2":
                for t_id, t in enumerate(self.tasks):
                    n_agents = self.nagents2task[t]
                    self.offline_key_embeddings.append(np.zeros(
                        (self.trajs_per_task, self.args.env_args["time_limit"], n_agents, self.args.n_embd)).astype('float32'))
                    for i in range(self.trajs_per_task // 20):
                        batch = self.offline_buffers[t_id][i * 20:(i + 1) * 20]
                        batch.to(self.args.device)
                        seq_rtg, seq_obs, seq_action, seq_post, seq_ret, mask = self.process_batch(
                            batch)
                        seq_smac_obs = self.process_smac_obs(seq_obs)
                        self.offline_traj_returns[t_id * self.trajs_per_task + i * 20:t_id * self.trajs_per_task + (i + 1) * 20] = (
                            seq_ret[:, -1, 0]).cpu().numpy()
                        for t in range(self.args.env_args["time_limit"]):
                            for n in range(n_agents):
                                emb, ind_r = self.agent.get_dem_traj_embedding(
                                    seq_rtg[:, :t+1], seq_smac_obs[:, :t+1, n], seq_action[:, :t+1, n], seq_post[:, :t+1], mask[:, :t+1])
                                self.offline_key_embeddings[t_id][i *
                                                                  20:(i + 1) * 20, t, n] = emb.cpu().numpy()
            else:
                assert (0)

            with open(emb_path, 'wb') as file:
                pickle.dump(self.offline_key_embeddings, file)
            with open(ret_path, 'wb') as file:
                pickle.dump(self.offline_traj_returns, file)

        self.online_key_embeddings = np.zeros(
            (self.buffer_size, self.args.env_args["time_limit"], self.n_agents, self.args.n_embd)).astype('float32')
        self.online_traj_returns = np.zeros((self.buffer_size))
        self.online_traj_ind_returns = np.zeros(
            (self.buffer_size, self.n_agents))

    def process_smac_obs(self, seq_obs, test_mode=False):
        ori_dim = seq_obs.shape[-1]
        return F.pad(seq_obs, (0, self.agent.input_shape - ori_dim))

    def set_agent(self, agent):
        self.agent = agent

    def process_batch(self, batch):
        obss = batch["obs"][:, :-1]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        batch_size, seq_len = obss.shape[0], obss.shape[1]
        seq_rtg = th.zeros_like(rewards)
        seq_rtg[:, 0] = rewards.sum(1)
        for t in range(1, seq_len):
            seq_rtg[:, t] = seq_rtg[:, t-1] - rewards[:, t-1]
        seq_obs = obss
        seq_action = actions.squeeze(-1)
        seq_reward = rewards
        seq_term = terminated
        seq_ret = th.zeros_like(rewards)
        seq_ret[:, 0] = rewards[:, 0]
        for t in range(1, seq_len):
            seq_ret[:, t] = seq_ret[:, t-1] + rewards[:, t]
        seq_completion = (seq_ret > (self.args.max_return - 1e-1))
        seq_post = th.cat([seq_reward, seq_term, seq_completion], dim=-1)
        return seq_rtg, seq_obs, seq_action, seq_post, seq_ret, mask

    def clean_online(self):
        self.episodes_in_buffer = 0
        self.buffer_index = 0

    def update_online_embedding(self, index):
        self.agent.eval()
        batch = self[index:index+1]
        batch.to(self.args.device)
        seq_rtg, seq_obs, seq_action, seq_post, seq_ret, mask = self.process_batch(
            batch)

        if self.args.env == "sc2v2":
            seq_smac_obs = self.process_smac_obs(seq_obs, test_mode=True)
        self.online_traj_returns[index:index +
                                 1] = (seq_ret[:, -1, 0]).cpu().numpy()

        for t in range(self.args.env_args["time_limit"]):
            for n in range(self.n_agents):
                if self.args.env == "sc2v2":
                    emb, ind_r = self.agent.get_dem_traj_embedding(
                        seq_rtg[:, :t+1], seq_smac_obs[:, :t+1, n], seq_action[:, :t+1, n], seq_post[:, :t+1], mask[:, :t+1])
                else:
                    emb, ind_r = self.agent.get_dem_traj_embedding(
                        seq_rtg[:, :t+1], seq_obs[:, :t+1, n], seq_action[:, :t+1, n], seq_post[:, :t+1], mask[:, :t+1])
                self.online_key_embeddings[index:index +
                                           1, t, n] = emb.cpu().numpy()
                self.online_traj_ind_returns[index:index +
                                             1, n:n+1] += ind_r.cpu().numpy()

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())
