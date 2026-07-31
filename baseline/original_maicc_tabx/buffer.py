from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .manifest import BaselineManifest, BaselineTask
from .padding import (
    TerminalPaddingError,
    valid_transition_length,
    validate_zero_padding,
)


class H5EpisodeStore:
    """Lazy HDF5 storage with the indexing contract of an EpisodeBatch."""

    REQUIRED_FIELDS = ("obs", "actions", "reward", "terminated", "filled")

    def __init__(
        self,
        task: BaselineTask,
        scheme,
        preprocess,
        max_seq_length: int,
        allow_terminal_padding: bool = False,
        device: str = "cpu",
    ):
        from components.episode_buffer import EpisodeBatch

        self.task = task
        self.scheme = deepcopy(scheme)
        self.groups = {"agents": task.n_agents}
        self.preprocess = preprocess
        self.max_seq_length = int(max_seq_length)
        self.allow_terminal_padding = bool(allow_terminal_padding)
        self.device = str(device)
        self._handle = None
        self._episode_batch_cls = EpisodeBatch

    def __len__(self) -> int:
        return self.task.n_trajectories

    def _open(self):
        import h5py

        if self._handle is None:
            self._handle = h5py.File(str(self.task.h5_path), "r")
            missing = [
                field
                for field in self.REQUIRED_FIELDS
                if field not in self._handle
            ]
            if missing:
                self.close()
                raise ValueError(
                    f"{self.task.h5_path} is missing fields {missing}"
                )
        return self._handle

    def _indices(self, item) -> np.ndarray:
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            values = np.arange(start, stop, step, dtype=np.int64)
        elif np.isscalar(item):
            value = int(item)
            if value < 0:
                value += len(self)
            values = np.asarray([value], dtype=np.int64)
        else:
            values = np.asarray(item, dtype=np.int64).reshape(-1)
            values = np.where(values < 0, values + len(self), values)
        if values.size == 0:
            raise IndexError("Cannot construct an empty MAICC EpisodeBatch")
        if values.min() < 0 or values.max() >= len(self):
            raise IndexError(
                f"Rows outside task {self.task.task_id}: {values.tolist()}"
            )
        return values

    def __getitem__(self, item):
        import torch as th

        rows = self._indices(item)
        handle = self._open()
        arrays = {
            field: np.stack(
                [np.asarray(handle[field][int(row)]) for row in rows],
                axis=0,
            )
            for field in self.REQUIRED_FIELDS
        }
        expected_time = self.task.sequence_length + 1
        expected = {
            "obs": (
                rows.size,
                expected_time,
                self.task.n_agents,
                self.task.obs_dim,
            ),
            "actions": (
                rows.size,
                expected_time,
                self.task.n_agents,
                1,
            ),
            "reward": (rows.size, expected_time, 1),
            "terminated": (rows.size, expected_time, 1),
            "filled": (rows.size, expected_time, 1),
        }
        for field, shape in expected.items():
            if arrays[field].shape != shape:
                raise ValueError(
                    f"Unexpected {field} shape {arrays[field].shape} in "
                    f"{self.task.h5_path}; expected {shape}"
                )
        for row_index in range(rows.size):
            row_arrays = {
                field: arrays[field][row_index]
                for field in self.REQUIRED_FIELDS
            }
            try:
                transitions = valid_transition_length(
                    row_arrays["filled"],
                    row_arrays["terminated"],
                    self.task.sequence_length,
                    allow_terminal_padding=self.allow_terminal_padding,
                )
                validate_zero_padding(
                    row_arrays,
                    transitions,
                    self.task.sequence_length,
                )
            except TerminalPaddingError as exc:
                raise ValueError(
                    f"Invalid baseline row={int(rows[row_index])} in "
                    f"{self.task.h5_path}: {exc}"
                ) from exc
        if not np.isfinite(arrays["obs"]).all():
            raise ValueError(f"NaN/Inf observation in {self.task.h5_path}")
        if not np.isfinite(arrays["reward"]).all():
            raise ValueError(f"NaN/Inf reward in {self.task.h5_path}")

        batch = self._episode_batch_cls(
            scheme=self.scheme,
            groups=self.groups,
            batch_size=int(rows.size),
            max_seq_length=self.max_seq_length,
            preprocess=self.preprocess,
            device=self.device,
        )
        batch.update(
            {
                "obs": th.as_tensor(arrays["obs"], dtype=th.float32),
                "actions": th.as_tensor(arrays["actions"], dtype=th.long),
                "reward": th.as_tensor(arrays["reward"], dtype=th.float32),
                "terminated": th.as_tensor(
                    arrays["terminated"], dtype=th.uint8
                ),
                "filled": th.as_tensor(arrays["filled"], dtype=th.long),
            },
            mark_filled=False,
        )
        return batch

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _checkpoint_signature(args) -> str:
    root_value = str(getattr(args, "checkpoint_path", "") or "")
    if not root_value:
        return "stage0-uninitialized"
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        return str(root)
    candidates = sorted(
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not candidates:
        return str(root)
    requested = int(getattr(args, "load_step", 0))
    selected = (
        max(candidates)
        if requested == 0
        else min(candidates, key=lambda value: abs(value - requested))
    )
    agent_path = root / str(selected) / "agent.th"
    if not agent_path.is_file():
        return f"{root}:{selected}"
    digest = hashlib.sha256()
    with agent_path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"{agent_path}:{digest.hexdigest()}"


def _cache_fingerprint(manifest: BaselineManifest, args) -> str:
    payload = {
        "manifest": manifest.fingerprint,
        "checkpoint": _checkpoint_signature(args),
        "n_embd": int(args.n_embd),
        "train_CEM": bool(args.train_CEM),
        "use_rtg": bool(args.use_rtg),
        "time_limit": int(args.env_args["time_limit"]),
        "upstream": "e1640a9",
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def make_tabx_buffer_class():
    """Build the adapter class after the untouched src tree is importable."""

    import faiss
    import torch as th
    from components.episode_buffer import EpisodeBatch, InContextReplayBuffer
    from components.epsilon_schedules import DecayThenFlatSchedule

    class OriginalMaiccTabxBuffer(InContextReplayBuffer):
        """Original replay/retrieval logic backed by strict TABX HDF5 rows."""

        def __init__(
            self,
            scheme,
            groups,
            args,
            max_seq_length,
            preprocess=None,
            device="cpu",
        ):
            if args.env != "tabx":
                super().__init__(
                    scheme,
                    groups,
                    args,
                    max_seq_length,
                    preprocess=preprocess,
                    device=device,
                )
                return
            EpisodeBatch.__init__(
                self,
                scheme,
                groups,
                args.test_nepisode,
                max_seq_length,
                preprocess=preprocess,
                device=device,
            )
            self.args = args
            self.manifest = BaselineManifest(
                Path(str(args.env_args["manifest"]))
            )
            self.manifest.validate_files()
            if max_seq_length != self.manifest.sequence_length + 1:
                raise ValueError(
                    f"runner max_seq_length={max_seq_length} does not match "
                    f"manifest sequence length {self.manifest.sequence_length}"
                )
            expected_args = {
                "n_agents": self.manifest.max_agents,
                "n_actions": self.manifest.action_dim,
                "task_nums": self.manifest.task_nums,
                "trajs_per_task": self.manifest.trajs_per_task,
            }
            for name, expected in expected_args.items():
                actual = int(getattr(args, name))
                if actual != expected:
                    raise ValueError(
                        f"args.{name}={actual} does not match manifest "
                        f"value {expected}"
                    )

            self.tasks = [task.task_id for task in self.manifest.tasks]
            self.nagents2task = {
                task.task_id: task.n_agents for task in self.manifest.tasks
            }
            self.offline_buffers = [
                H5EpisodeStore(
                    task=task,
                    scheme=scheme,
                    preprocess=preprocess,
                    max_seq_length=max_seq_length,
                    allow_terminal_padding=(
                        self.manifest.allows_terminal_padding
                    ),
                    device=device,
                )
                for task in self.manifest.tasks
            ]
            self.n_agents = args.n_agents
            self.task_nums = args.task_nums
            self.trajs_per_task = args.trajs_per_task
            self.offline_trajs_num = self.task_nums * self.trajs_per_task
            self.buffer_size = args.test_nepisode
            if self.offline_trajs_num % 20 != 0:
                raise ValueError(
                    "Original MAICC embedding initialization requires the "
                    "offline trajectory count to be divisible by 20"
                )
            self.buffer_index = 0
            self.episodes_in_buffer = 0
            self.schedule = DecayThenFlatSchedule(
                1, 0.01, 1, decay="exp"
            )

        def can_sample(self, batch_size):
            return self.offline_trajs_num >= batch_size

        def sample(self, batch_size):
            assert self.can_sample(batch_size)
            task_id = np.random.choice(self.task_nums)
            episode_ids = np.random.choice(
                self.trajs_per_task,
                batch_size,
                replace=False,
            )
            return self.offline_buffers[int(task_id)][episode_ids]

        def transfer_id(self, indices):
            return self.manifest.index_mapper.decode(indices)

        def mips(
            self,
            querys,
            t,
            query_return=None,
            test_mode=False,
        ):
            query_count = int(np.asarray(querys).shape[0])
            if (
                not self.manifest.allows_terminal_padding
                and (test_mode or query_count == int(self.args.batch_size))
            ):
                return super().mips(
                    querys,
                    t,
                    query_return=query_return,
                    test_mode=test_mode,
                )
            if test_mode:
                raise RuntimeError(
                    "Padding-compatible TABX baseline has no online simulator"
                )
            # This is the original offline scoring branch. For padded data,
            # only the candidate bank construction changes: a trajectory is
            # present at timestep t iff t is a real transition for that row.
            candidate_embeddings = []
            candidate_ids = []
            timestep = int(t)
            if not 0 <= timestep < self.manifest.sequence_length:
                raise ValueError(
                    f"retrieval timestep={timestep} outside "
                    f"[0, {self.manifest.sequence_length})"
                )
            for task_id, task in enumerate(self.manifest.tasks):
                if self.manifest.allows_terminal_padding:
                    rows = np.flatnonzero(
                        self.offline_valid_steps[task_id][:, timestep]
                    )
                else:
                    rows = np.arange(
                        self.trajs_per_task,
                        dtype=np.int64,
                    )
                if rows.size == 0:
                    continue
                values = np.asarray(
                    self.offline_key_embeddings[task_id][rows, timestep],
                    dtype=np.float32,
                )
                candidate_embeddings.append(
                    values.reshape(-1, self.args.n_embd)
                )
                trajectories = np.repeat(rows, task.n_agents)
                agents = np.tile(
                    np.arange(task.n_agents, dtype=np.int64),
                    rows.size,
                )
                candidate_ids.append(
                    self.manifest.index_mapper.encode(
                        np.full(
                            trajectories.shape,
                            task_id,
                            dtype=np.int64,
                        ),
                        trajectories,
                        agents,
                    )
                )
            if not candidate_embeddings:
                raise ValueError(
                    f"No valid retrieval candidates at timestep={timestep}"
                )
            z_d = np.ascontiguousarray(
                np.concatenate(candidate_embeddings, axis=0),
                dtype=np.float32,
            )
            flat_candidate_ids = np.concatenate(candidate_ids, axis=0)
            if z_d.shape[0] < int(self.args.l):
                raise ValueError(
                    f"Only {z_d.shape[0]} valid retrieval candidates at "
                    f"timestep={timestep}; original l={self.args.l}"
                )
            q = np.ascontiguousarray(
                deepcopy(querys),
                dtype=np.float32,
            )
            faiss.normalize_L2(z_d)
            faiss.normalize_L2(q)
            index = faiss.IndexFlatIP(self.args.n_embd)
            index.add(z_d)
            distances, candidates = index.search(q, self.args.l)
            flat_candidates = flat_candidate_ids[candidates]
            task_id, trajectory_id, agent_id = self.transfer_id(
                flat_candidates
            )
            similarity = (distances + 1) / 2
            if query_return is None:
                raise ValueError(
                    "Original offline retrieval requires query_return"
                )
            query_return_values = (
                query_return.cpu().numpy()
                if hasattr(query_return, "cpu")
                else np.asarray(query_return)
            )
            return_filter = np.logical_and(
                distances < self.args.max_similarity,
                self.offline_traj_returns[
                    task_id * self.trajs_per_task + trajectory_id
                ]
                <= query_return_values,
            )
            score = similarity + return_filter
            selected = np.argsort(-score, axis=1)[:, : self.args.k]
            flat_ids = np.take_along_axis(
                flat_candidates,
                selected,
                axis=1,
            )
            for row in range(flat_ids.shape[0]):
                flat_ids[row] = np.random.permutation(flat_ids[row])
            flat_ids = flat_ids.reshape(-1)
            task_id, trajectory_id, agent_id = self.transfer_id(flat_ids)

            observations = []
            rewards = []
            actions = []
            terminated = []
            masks = []
            for index_id in range(query_count * self.args.k):
                batch = self.offline_buffers[int(task_id[index_id])][
                    int(trajectory_id[index_id]) : int(
                        trajectory_id[index_id]
                    )
                    + 1
                ]
                batch.to(self.args.device)
                observations.append(
                    batch["obs"][0, :-1, int(agent_id[index_id])]
                )
                rewards.append(batch["reward"][0, :-1])
                actions.append(
                    batch["actions"][0, :-1, int(agent_id[index_id])]
                )
                terminated.append(batch["terminated"][0, :-1].float())
                masks.append(batch["filled"][0, :-1].float())
            observations = th.stack(observations, dim=0)
            rewards = th.stack(rewards, dim=0)
            actions = th.stack(actions, dim=0)
            terminated = th.stack(terminated, dim=0)
            masks = th.stack(masks, dim=0)
            masks[:, 1:] = masks[:, 1:] * (1 - terminated[:, :-1])
            return observations, rewards, actions, terminated, masks

        def init_embedding(self):
            self.agent.eval()
            configured_cache = self.args.env_args.get("embedding_cache")
            cache_root = Path(
                str(
                    configured_cache
                    or (
                        self.manifest.path.parent
                        / ".original_maicc_embedding_cache"
                    )
                )
            ).expanduser().resolve()
            cache_root.mkdir(parents=True, exist_ok=True)
            fingerprint = _cache_fingerprint(self.manifest, self.args)
            cache_dir = cache_root / fingerprint
            cache_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = cache_dir / "metadata.json"
            returns_path = cache_dir / "returns.npy"
            embedding_paths = [
                cache_dir / f"task_{task.index:04d}_embeddings.npy"
                for task in self.manifest.tasks
            ]
            validity_paths = [
                cache_dir / f"task_{task.index:04d}_valid_steps.npy"
                for task in self.manifest.tasks
            ]
            expected_metadata = {
                "fingerprint": fingerprint,
                "manifest": str(self.manifest.path),
                "manifest_sha256": self.manifest.fingerprint,
                "checkpoint": _checkpoint_signature(self.args),
                "upstream_commit": "e1640a9",
            }
            if self.manifest.allows_terminal_padding:
                expected_metadata["padding_mode"] = (
                    self.manifest.padding_mode
                )
            cache_ready = (
                metadata_path.is_file()
                and returns_path.is_file()
                and all(path.is_file() for path in embedding_paths)
                and (
                    not self.manifest.allows_terminal_padding
                    or all(path.is_file() for path in validity_paths)
                )
            )
            if cache_ready:
                actual_metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                cache_ready = actual_metadata == expected_metadata

            if cache_ready:
                self.offline_key_embeddings = [
                    np.load(path, mmap_mode="r")
                    for path in embedding_paths
                ]
                self.offline_traj_returns = np.load(returns_path)
                self.offline_valid_steps = (
                    [
                        np.load(path, mmap_mode="r")
                        for path in validity_paths
                    ]
                    if self.manifest.allows_terminal_padding
                    else None
                )
                self._validate_embedding_cache()
            else:
                self.offline_traj_returns = np.zeros(
                    self.offline_trajs_num, dtype=np.float64
                )
                writable_embeddings = []
                writable_validity = []
                for task, path in zip(
                    self.manifest.tasks, embedding_paths
                ):
                    values = np.lib.format.open_memmap(
                        path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(
                            self.trajs_per_task,
                            self.args.env_args["time_limit"],
                            task.n_agents,
                            self.args.n_embd,
                        ),
                    )
                    writable_embeddings.append(values)
                if self.manifest.allows_terminal_padding:
                    for path in validity_paths:
                        valid = np.lib.format.open_memmap(
                            path,
                            mode="w+",
                            dtype=np.bool_,
                            shape=(
                                self.trajs_per_task,
                                self.args.env_args["time_limit"],
                            ),
                        )
                        writable_validity.append(valid)

                for task_id, task in enumerate(self.manifest.tasks):
                    values = writable_embeddings[task_id]
                    valid_values = (
                        writable_validity[task_id]
                        if self.manifest.allows_terminal_padding
                        else None
                    )
                    return_start = task_id * self.trajs_per_task
                    for start in range(0, self.trajs_per_task, 20):
                        end = min(start + 20, self.trajs_per_task)
                        batch = self.offline_buffers[task_id][start:end]
                        batch.to(self.args.device)
                        (
                            seq_rtg,
                            seq_obs,
                            seq_action,
                            seq_post,
                            seq_ret,
                            mask,
                        ) = self.process_batch(batch)
                        self.offline_traj_returns[
                            return_start + start : return_start + end
                        ] = seq_ret[:, -1, 0].cpu().numpy()
                        if valid_values is not None:
                            valid_values[start:end] = (
                                mask.squeeze(-1).bool().cpu().numpy()
                            )
                        for timestep in range(
                            self.args.env_args["time_limit"]
                        ):
                            for agent_id in range(task.n_agents):
                                embedding, _ = (
                                    self.agent.get_dem_traj_embedding(
                                        seq_rtg[:, : timestep + 1],
                                        seq_obs[
                                            :, : timestep + 1, agent_id
                                        ],
                                        seq_action[
                                            :, : timestep + 1, agent_id
                                        ],
                                        seq_post[:, : timestep + 1],
                                        mask[:, : timestep + 1],
                                    )
                                )
                                values[
                                    start:end, timestep, agent_id
                                ] = embedding.cpu().numpy()
                    values.flush()
                    if valid_values is not None:
                        valid_values.flush()
                np.save(returns_path, self.offline_traj_returns)
                temp_metadata = metadata_path.with_suffix(".json.tmp")
                temp_metadata.write_text(
                    json.dumps(
                        expected_metadata,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                os.replace(temp_metadata, metadata_path)
                self.offline_key_embeddings = [
                    np.load(path, mmap_mode="r")
                    for path in embedding_paths
                ]
                self.offline_valid_steps = (
                    [
                        np.load(path, mmap_mode="r")
                        for path in validity_paths
                    ]
                    if self.manifest.allows_terminal_padding
                    else None
                )
                self._validate_embedding_cache()

            self.online_key_embeddings = np.zeros(
                (
                    self.buffer_size,
                    self.args.env_args["time_limit"],
                    self.n_agents,
                    self.args.n_embd,
                ),
                dtype=np.float32,
            )
            self.online_traj_returns = np.zeros(self.buffer_size)
            self.online_traj_ind_returns = np.zeros(
                (self.buffer_size, self.n_agents)
            )

        def _validate_embedding_cache(self):
            if self.offline_traj_returns.shape != (
                self.offline_trajs_num,
            ):
                raise ValueError("Invalid cached trajectory return shape")
            for task, values in zip(
                self.manifest.tasks, self.offline_key_embeddings
            ):
                expected = (
                    self.trajs_per_task,
                    self.args.env_args["time_limit"],
                    task.n_agents,
                    self.args.n_embd,
                )
                if values.shape != expected:
                    raise ValueError(
                        f"Invalid embedding cache for {task.task_id}: "
                        f"{values.shape}, expected {expected}"
                    )
            if self.manifest.allows_terminal_padding:
                for task, valid in zip(
                    self.manifest.tasks,
                    self.offline_valid_steps,
                ):
                    expected = (
                        self.trajs_per_task,
                        self.args.env_args["time_limit"],
                    )
                    if valid.shape != expected:
                        raise ValueError(
                            f"Invalid cached valid-step mask for "
                            f"{task.task_id}: {valid.shape}, expected "
                            f"{expected}"
                        )
                for timestep in range(
                    self.args.env_args["time_limit"]
                ):
                    candidates = sum(
                        int(valid[:, timestep].sum()) * task.n_agents
                        for task, valid in zip(
                            self.manifest.tasks,
                            self.offline_valid_steps,
                        )
                    )
                    if candidates < int(self.args.l):
                        raise ValueError(
                            f"Only {candidates} valid retrieval candidates "
                            f"at timestep={timestep}; original l={self.args.l}"
                        )

        def close(self):
            for store in getattr(self, "offline_buffers", []):
                close = getattr(store, "close", None)
                if close is not None:
                    close()

    OriginalMaiccTabxBuffer.__name__ = "OriginalMaiccTabxBuffer"
    return OriginalMaiccTabxBuffer
