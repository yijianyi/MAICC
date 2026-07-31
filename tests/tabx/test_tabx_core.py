from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tabx.manifest import ManifestError, TabxManifest
from tabx.normalization import ObservationNormalizer
from tools.convert_tabx_to_maicc import (
    ConversionError,
    discover_record_dirs,
)
from tools.scan_tabx_dataset import infer_record_shape_metadata


class ManifestTests(unittest.TestCase):
    def test_v2_global_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "schema": "maicc_tabx_manifest_v2",
                "sequence_length": 64,
                "global_schema": {
                    "obs_dim": 758,
                    "action_dim": 8,
                    "max_agents": 20,
                },
                "tasks": [
                    {
                        "task_id": "task_a",
                        "h5_file": "task_a.h5",
                        "n_agents": 3,
                        "n_trajectories": 4,
                        "obs_dim": 758,
                        "action_dim": 8,
                        "sequence_length": 64,
                        "return_stats": {"min": -1, "max": 2},
                    }
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = TabxManifest(path)
            self.assertEqual(manifest.obs_dim, 758)
            self.assertEqual(manifest.action_dim, 8)
            self.assertEqual(manifest.max_agents, 20)
            self.assertEqual(len(manifest.tasks_for_split("train")), 1)

    def test_rejects_cross_task_obs_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "schema": "maicc_tabx_manifest_v2",
                "sequence_length": 64,
                "tasks": [
                    {
                        "task_id": "a",
                        "h5_file": "a.h5",
                        "n_agents": 2,
                        "n_trajectories": 1,
                        "obs_dim": 10,
                        "action_dim": 8,
                    },
                    {
                        "task_id": "b",
                        "h5_file": "b.h5",
                        "n_agents": 2,
                        "n_trajectories": 1,
                        "obs_dim": 11,
                        "action_dim": 8,
                    },
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ManifestError):
                TabxManifest(path)


class NormalizationTests(unittest.TestCase):
    def test_zero_padding_is_preserved(self) -> None:
        normalizer = ObservationNormalizer(
            mean=np.asarray([1.0, 0.0], dtype=np.float32),
            scale=np.asarray([2.0, 4.0], dtype=np.float32),
            clip=10.0,
            preserve_zeros=True,
        )
        result = normalizer.apply(
            np.asarray([[0.0, 8.0], [5.0, 0.0]], dtype=np.float32)
        )
        np.testing.assert_allclose(
            result,
            np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32),
        )


class RawDiscoveryTests(unittest.TestCase):
    def test_empty_root_has_clear_conversion_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ConversionError,
                "No record-\\* directories",
            ):
                discover_record_dirs(Path(directory))

    def test_discovers_multi_job_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = (
                root
                / "job_0"
                / "records"
                / "task_00001_deadbeef"
                / "record-000000"
            )
            record.mkdir(parents=True)
            (record / "meta.json").write_text("{}", encoding="utf-8")
            self.assertEqual(discover_record_dirs(root), [record])

    def test_infers_capacities_and_active_enemies_without_task_package(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory)
            (record / "meta.json").write_text(
                json.dumps(
                    {
                        "n_ally": 3,
                        "n_units": 5,
                        "max_n_zone": 4,
                    }
                ),
                encoding="utf-8",
            )
            health = np.asarray(
                [
                    [100, 100, 0, 80, 80],
                    [100, 100, 0, 80, 80],
                ],
                dtype=np.float32,
            )
            teams = np.asarray(
                [
                    [0, 0, -1, 1, 1],
                    [0, 0, -1, 1, 1],
                ],
                dtype=np.int8,
            )
            np.save(record / "unit_max_health.npy", health)
            np.save(record / "unit_team.npy", teams)

            shape = infer_record_shape_metadata(record, [0, 1], 3)

            self.assertEqual(shape["ally_capacity"], 3)
            self.assertEqual(shape["enemy_capacity"], 2)
            self.assertEqual(shape["unit_capacity"], 5)
            self.assertEqual(shape["zone_capacity"], 4)
            self.assertEqual(shape["actual_enemies"], 2)
            self.assertIsNone(shape["actual_zones"])


if __name__ == "__main__":
    unittest.main()
