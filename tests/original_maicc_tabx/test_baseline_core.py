from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.original_maicc_tabx.dataset_builder import (
    build_baseline_dataset,
    select_non_overlapping,
    split_episode_groups,
)
from baseline.original_maicc_tabx.fidelity import (
    EXPECTED_VENDOR_LEARNER_HASH,
    verify_upstream_core,
)
from baseline.original_maicc_tabx.manifest import (
    BaselineManifest,
    BaselineManifestError,
    FlatIndexMapper,
)
from baseline.original_maicc_tabx.padding import (
    TerminalPaddingError,
    valid_transition_length,
)
from tools.run_original_maicc_tabx import build_sacred_argv


class FidelityTests(unittest.TestCase):
    def test_upstream_core_and_frozen_learner_match_e1640a9(self):
        verify_upstream_core(REPO_ROOT)

    def test_frozen_learner_exact_hash(self):
        path = (
            REPO_ROOT
            / "baseline"
            / "original_maicc_tabx"
            / "vendor"
            / "ic_learner.py"
        )
        content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.assertEqual(actual, EXPECTED_VENDOR_LEARNER_HASH)


class MappingTests(unittest.TestCase):
    def test_variable_agent_flat_ids_round_trip(self):
        mapper = FlatIndexMapper([2, 4, 3], trajectories_per_task=20)
        values = np.asarray([0, 1, 2, 39, 40, 41, 119, 120, 179])
        task, trajectory, agent = mapper.decode(values)
        np.testing.assert_array_equal(
            task,
            np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2]),
        )
        np.testing.assert_array_equal(
            trajectory,
            np.asarray([0, 0, 1, 19, 0, 0, 19, 0, 19]),
        )
        np.testing.assert_array_equal(
            agent,
            np.asarray([0, 1, 0, 1, 0, 1, 3, 0, 2]),
        )
        encoded = mapper.encode(task, trajectory, agent)
        np.testing.assert_array_equal(encoded, values)

    def test_out_of_range_flat_id_is_rejected(self):
        mapper = FlatIndexMapper([2], trajectories_per_task=20)
        with self.assertRaises(IndexError):
            mapper.decode([40])


class StrictWindowTests(unittest.TestCase):
    @staticmethod
    def entry(row, record, episode, start, end):
        return {
            "row": row,
            "source_record": record,
            "source_episode_id": episode,
            "source_start": start,
            "source_end": end,
        }

    def test_terminal_aligned_overlap_is_removed(self):
        entries = [
            self.entry(0, "r", 1, 0, 64),
            self.entry(1, "r", 1, 64, 128),
            self.entry(2, "r", 1, 86, 150),
            self.entry(3, "r", 2, 0, 64),
        ]
        result = select_non_overlapping(entries)
        self.assertEqual([item["row"] for item in result], [0, 1, 3])

    def test_source_episode_never_crosses_splits(self):
        entries = [
            self.entry(index, "record", index // 2, index * 64, index * 64 + 64)
            for index in range(20)
        ]
        splits = split_episode_groups(
            entries,
            task_id="task-a",
            seed=7,
            fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        )
        owners = {}
        for split, values in splits.items():
            for value in values:
                key = (
                    value["source_record"],
                    value["source_episode_id"],
                )
                previous = owners.setdefault(key, split)
            self.assertEqual(previous, split)


class PaddingTests(unittest.TestCase):
    def test_terminal_padding_reports_real_transition_length(self):
        length = valid_transition_length(
            filled=np.asarray([1, 1, 1, 0, 0]),
            terminated=np.asarray([0, 1, 0, 0, 0]),
            sequence_length=4,
            allow_terminal_padding=True,
        )
        self.assertEqual(length, 2)

    def test_nonterminal_padding_is_rejected(self):
        with self.assertRaises(TerminalPaddingError):
            valid_transition_length(
                filled=np.asarray([1, 1, 1, 0, 0]),
                terminated=np.zeros(5),
                sequence_length=4,
                allow_terminal_padding=True,
            )


class ManifestTests(unittest.TestCase):
    def _payload(self):
        return {
            "schema": "original_maicc_tabx_baseline_v1",
            "split": "train",
            "sequence_length": 64,
            "trajs_per_task": 20,
            "global_schema": {
                "obs_dim": 12,
                "action_dim": 8,
                "max_agents": 4,
            },
            "return_stats": {"min": -1, "max": 2, "mean": 0},
            "baseline_contract": {
                "strict_full_length": True,
                "non_overlapping": True,
                "normalization": "none",
                "equal_trajectories_per_task": True,
            },
            "tasks": [
                {
                    "task_id": "a",
                    "h5_file": "a.h5",
                    "index_file": "a.index.jsonl",
                    "n_agents": 3,
                    "n_trajectories": 20,
                    "obs_dim": 12,
                    "action_dim": 8,
                    "sequence_length": 64,
                    "return_stats": {"min": -1, "max": 2},
                }
            ],
        }

    def test_strict_manifest_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(self._payload()), encoding="utf-8")
            manifest = BaselineManifest(path)
            self.assertEqual(manifest.max_agents, 4)
            self.assertEqual(manifest.trajs_per_task, 20)

    def test_normalized_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._payload()
            payload["baseline_contract"]["normalization"] = "rms"
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BaselineManifestError):
                BaselineManifest(path)

    def test_terminal_padding_manifest_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._payload()
            payload["baseline_contract"]["strict_full_length"] = False
            payload["baseline_contract"][
                "padding_mode"
            ] = "terminal_zero_fill"
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = BaselineManifest(path)
            self.assertTrue(manifest.allows_terminal_padding)
            self.assertEqual(
                manifest.padding_mode,
                "terminal_zero_fill",
            )


class LauncherTests(unittest.TestCase):
    def test_sacred_config_declares_launcher_audit_fields(self):
        config = (
            REPO_ROOT / "src" / "config" / "envs" / "tabx_baseline.yaml"
        ).read_text(encoding="utf-8")
        for key in (
            "embedding_cache:",
            "baseline_micro_batch_size:",
            "baseline_gradient_checkpointing:",
            "baseline_upstream_commit:",
            "baseline_manifest_sha256:",
            "baseline_execution:",
            "baseline_padding_mode:",
            "baseline_data_parallel:",
            "baseline_parallel_devices:",
        ):
            self.assertIn(key, config)

    def test_launcher_forces_original_model_and_loss_switches(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = ManifestTests()._payload()
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = BaselineManifest(path)
            cli = Namespace(
                batch_size=32,
                micro_batch_size=1,
                stage=0,
                checkpoint_path=None,
                extra_updates=[],
                seed=7,
                embedding_cache=None,
                results_dir=None,
                t_max=None,
                device="cpu",
                load_step=0,
                gradient_checkpointing=True,
                data_parallel=False,
                parallel_devices=1,
            )
            argv = build_sacred_argv(cli, manifest)
            updates = set(argv[4:])
            self.assertIn('agent="transformer"', updates)
            self.assertIn("n_layer=8", updates)
            self.assertIn("n_head=8", updates)
            self.assertIn("hidden_dim=64", updates)
            self.assertIn("l=50", updates)
            self.assertIn("train_CEM=True", updates)
            self.assertIn("use_rtg=False", updates)
            self.assertIn("baseline_micro_batch_size=1", updates)
            self.assertIn("baseline_gradient_checkpointing=True", updates)

    def test_launcher_selects_padding_compatible_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = ManifestTests()._payload()
            payload["baseline_contract"]["strict_full_length"] = False
            payload["baseline_contract"][
                "padding_mode"
            ] = "terminal_zero_fill"
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = BaselineManifest(path)
            cli = Namespace(
                batch_size=32,
                micro_batch_size=0,
                stage=0,
                checkpoint_path=None,
                extra_updates=[],
                seed=7,
                embedding_cache=None,
                results_dir=None,
                t_max=None,
                device="cpu",
                load_step=0,
                gradient_checkpointing=False,
                data_parallel=False,
                parallel_devices=1,
            )
            updates = set(build_sacred_argv(cli, manifest)[4:])
            self.assertIn(
                'baseline_execution="padding_compatible_original"',
                updates,
            )
            self.assertIn(
                'baseline_padding_mode="terminal_zero_fill"',
                updates,
            )

    def test_launcher_selects_data_parallel_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = ManifestTests()._payload()
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = BaselineManifest(path)
            cli = Namespace(
                batch_size=32,
                micro_batch_size=0,
                stage=0,
                checkpoint_path=None,
                extra_updates=[],
                seed=7,
                embedding_cache=None,
                results_dir=None,
                t_max=None,
                device="cuda",
                load_step=0,
                gradient_checkpointing=False,
                data_parallel=True,
                parallel_devices=4,
            )
            updates = set(build_sacred_argv(cli, manifest)[4:])
            self.assertIn(
                'baseline_execution="data_parallel_equivalent"',
                updates,
            )
            self.assertIn("baseline_data_parallel=True", updates)
            self.assertIn("baseline_parallel_devices=4", updates)

    def test_data_parallel_rejects_microbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = ManifestTests()._payload()
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = BaselineManifest(path)
            cli = Namespace(
                batch_size=32,
                micro_batch_size=1,
                stage=0,
                checkpoint_path=None,
                extra_updates=[],
                seed=7,
                embedding_cache=None,
                results_dir=None,
                t_max=None,
                device="cuda",
                load_step=0,
                gradient_checkpointing=False,
                data_parallel=True,
                parallel_devices=2,
            )
            with self.assertRaises(SystemExit):
                build_sacred_argv(cli, manifest)


class DataParallelTests(unittest.TestCase):
    def test_replica_means_are_weighted_by_valid_samples(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this test runtime")

        from baseline.original_maicc_tabx.data_parallel import (
            _weighted_replica_mean,
        )

        values = torch.tensor([2.0, 8.0], requires_grad=True)
        weights = torch.tensor([3.0, 1.0])
        result = _weighted_replica_mean(values, weights)
        self.assertAlmostEqual(result.item(), 3.5)
        result.backward()
        np.testing.assert_allclose(
            values.grad.numpy(),
            np.asarray([0.75, 0.25]),
        )


class H5BuilderTests(unittest.TestCase):
    def test_strict_builder_produces_equal_full_length_tasks(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is not installed in this test runtime")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = []
            length = 2
            for task_index, agents in enumerate((2, 3)):
                task_id = f"task-{task_index}"
                h5_path = root / f"{task_id}.h5"
                index_path = root / f"{task_id}.index.jsonl"
                rows = 60
                with h5py.File(str(h5_path), "w") as handle:
                    obs = np.arange(
                        rows * (length + 1) * agents * 4,
                        dtype=np.float32,
                    ).reshape(rows, length + 1, agents, 4)
                    actions = np.zeros(
                        (rows, length + 1, agents, 1),
                        dtype=np.int64,
                    )
                    reward = np.zeros(
                        (rows, length + 1, 1),
                        dtype=np.float32,
                    )
                    reward[:, 0, 0] = np.arange(rows, dtype=np.float32)
                    handle.create_dataset("obs", data=obs)
                    handle.create_dataset("actions", data=actions)
                    handle.create_dataset("reward", data=reward)
                    handle.create_dataset(
                        "terminated",
                        data=np.zeros(
                            (rows, length + 1, 1),
                            dtype=np.uint8,
                        ),
                    )
                    handle.create_dataset(
                        "filled",
                        data=np.ones(
                            (rows, length + 1, 1),
                            dtype=np.int64,
                        ),
                    )
                with index_path.open("w", encoding="utf-8") as handle:
                    for row in range(rows):
                        handle.write(
                            json.dumps(
                                {
                                    "row": row,
                                    "task_id": task_id,
                                    "source_record": f"record-{row}",
                                    "source_episode_id": row,
                                    "source_start": 0,
                                    "source_end": length,
                                }
                            )
                            + "\n"
                        )
                tasks.append(
                    {
                        "task_id": task_id,
                        "h5_file": h5_path.name,
                        "index_file": index_path.name,
                        "n_agents": agents,
                        "n_trajectories": rows,
                        "obs_dim": 4,
                        "action_dim": 8,
                        "sequence_length": length,
                        "split": "train",
                    }
                )
            converted = root / "manifest.json"
            converted.write_text(
                json.dumps(
                    {
                        "schema": "maicc_tabx_manifest_v2",
                        "sequence_length": length,
                        "global_schema": {
                            "obs_dim": 4,
                            "action_dim": 8,
                            "max_agents": 3,
                        },
                        "tasks": tasks,
                    }
                ),
                encoding="utf-8",
            )
            outputs = build_baseline_dataset(
                input_manifest=converted,
                output_dir=root / "baseline",
                trajectories_per_task=40,
                train_fraction=1.0,
                validation_fraction=0.0,
                test_fraction=0.0,
            )
            manifest = BaselineManifest(outputs["train"])
            self.assertEqual(manifest.trajs_per_task, 40)
            self.assertEqual(
                [task.n_trajectories for task in manifest.tasks],
                [40, 40],
            )
            for task in manifest.tasks:
                with h5py.File(str(task.h5_path), "r") as handle:
                    self.assertTrue(np.all(handle["filled"][:] == 1))

    def test_padding_builder_retains_short_natural_terminals(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is not installed in this test runtime")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = []
            sequence_length = 4
            rows = 60
            for task_index, agents in enumerate((2, 3)):
                task_id = f"padded-{task_index}"
                h5_path = root / f"{task_id}.h5"
                index_path = root / f"{task_id}.index.jsonl"
                with h5py.File(str(h5_path), "w") as handle:
                    obs = np.zeros(
                        (
                            rows,
                            sequence_length + 1,
                            agents,
                            4,
                        ),
                        dtype=np.float32,
                    )
                    actions = np.zeros(
                        (
                            rows,
                            sequence_length + 1,
                            agents,
                            1,
                        ),
                        dtype=np.int64,
                    )
                    reward = np.zeros(
                        (rows, sequence_length + 1, 1),
                        dtype=np.float32,
                    )
                    terminated = np.zeros(
                        (rows, sequence_length + 1, 1),
                        dtype=np.uint8,
                    )
                    filled = np.zeros(
                        (rows, sequence_length + 1, 1),
                        dtype=np.int64,
                    )
                    for row in range(rows):
                        actual = sequence_length if row % 2 == 0 else 2
                        obs[row, :actual] = float(row + 1)
                        actions[row, :actual, :, 0] = row % 8
                        reward[row, 0, 0] = float(row + 1)
                        filled[row, : actual + 1, 0] = 1
                        if actual < sequence_length:
                            terminated[row, actual - 1, 0] = 1
                    handle.create_dataset("obs", data=obs)
                    handle.create_dataset("actions", data=actions)
                    handle.create_dataset("reward", data=reward)
                    handle.create_dataset(
                        "terminated",
                        data=terminated,
                    )
                    handle.create_dataset("filled", data=filled)
                with index_path.open("w", encoding="utf-8") as handle:
                    for row in range(rows):
                        actual = sequence_length if row % 2 == 0 else 2
                        handle.write(
                            json.dumps(
                                {
                                    "row": row,
                                    "task_id": task_id,
                                    "source_record": f"record-{row}",
                                    "source_episode_id": row,
                                    "source_start": 0,
                                    "source_end": actual,
                                    "boundary": (
                                        "window_boundary"
                                        if actual == sequence_length
                                        else "natural_terminal"
                                    ),
                                }
                            )
                            + "\n"
                        )
                tasks.append(
                    {
                        "task_id": task_id,
                        "h5_file": h5_path.name,
                        "index_file": index_path.name,
                        "n_agents": agents,
                        "n_trajectories": rows,
                        "obs_dim": 4,
                        "action_dim": 8,
                        "sequence_length": sequence_length,
                        "split": "train",
                    }
                )
            converted = root / "manifest.json"
            converted.write_text(
                json.dumps(
                    {
                        "schema": "maicc_tabx_manifest_v2",
                        "sequence_length": sequence_length,
                        "global_schema": {
                            "obs_dim": 4,
                            "action_dim": 8,
                            "max_agents": 3,
                        },
                        "tasks": tasks,
                    }
                ),
                encoding="utf-8",
            )
            outputs = build_baseline_dataset(
                input_manifest=converted,
                output_dir=root / "padded-baseline",
                trajectories_per_task=40,
                train_fraction=1.0,
                validation_fraction=0.0,
                test_fraction=0.0,
                include_short_terminal=True,
            )
            manifest = BaselineManifest(outputs["train"])
            self.assertTrue(manifest.allows_terminal_padding)
            for task in manifest.tasks:
                with h5py.File(str(task.h5_path), "r") as handle:
                    filled = np.asarray(handle["filled"])
                    self.assertTrue(np.any(filled == 0))
                    self.assertTrue(np.any(filled == 1))


if __name__ == "__main__":
    unittest.main()
