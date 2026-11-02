"""Task-level grouping, zero-copy manifests and isolated training orchestration."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tabx.config import load_yaml_config
from tabx.manifest import TabxManifest
from tools.convert_tabx_to_maicc import ConversionError
from tools.split_tabx_label_groups import prepare_groups, scan_tasks
from tools.run_tabx_label_groups import commands_for_group, main as run_groups, summarize_metrics


class LabelGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tabx label groups ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.output = self.root / "groups"
        self.converted = self.root / "converted"
        self.converted.mkdir()
        self.record("rl", 0, True)
        self.record("rl", 1, True)
        self.record("advance", 0, False)
        tasks = []
        for task_id in ("advance", "rl"):
            (self.converted / f"{task_id}.h5").write_bytes(b"unchanged source dataset")
            (self.converted / f"{task_id}.jsonl").write_text(
                json.dumps({"source_record": f"job/records/{task_id}/record-000000"}) + "\n")
            tasks.append(dict(task_id=task_id, h5_file=f"{task_id}.h5",
                              index_file=f"{task_id}.jsonl", n_agents=10,
                              n_trajectories=3, obs_dim=342, action_dim=8,
                              split="train", return_stats={"min": -1, "max": 2}))
        self.payload = dict(schema="maicc_tabx_manifest_v2", source_root=str(self.raw),
                            sequence_length=32, tasks=tasks)
        self.manifest = self.converted / "manifest.json"
        self.manifest.write_text(json.dumps(self.payload))
        self.template = dict(data={"normalization": "/old/normalization.npz"},
                             model={"n_zones": 4, "max_enemies": 10, "train_cem": True},
                             training={"total_steps": 123, "learning_rate": 0.0005,
                                       "metrics_file": "/old/shared.csv"},
                             retrieval={"index_manifest": "/old/shared-index.json"})

    def record(self, task_id, number, distribution):
        directory = self.raw / "job" / "records" / task_id / f"record-{number:06d}"
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text(json.dumps({"task_id": task_id}))
        if distribution:
            # Discovery checks availability and deliberately does not load large arrays.
            (directory / "actions_reference_distribution.npy").write_bytes(b"distribution")
        return directory

    def prepare(self, **kwargs):
        return prepare_groups(None, self.output, self.manifest, **kwargs)

    def test_every_record_is_checked_and_tasks_are_not_split(self):
        tasks = scan_tasks(self.raw)
        self.assertEqual(tasks["rl"]["records_with_distribution"], 2)
        self.assertFalse(tasks["advance"]["has_reference_distribution"])
        self.record("rl", 2, False)
        with self.assertRaisesRegex(ConversionError, "Task rl has mixed"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_group_manifests_rebase_paths_and_preserve_source_bytes(self):
        originals = {path: path.read_bytes() for path in self.converted.iterdir()}
        self.payload["tasks"][0]["split"] = "test"
        self.manifest.write_text(json.dumps(self.payload))
        originals[self.manifest] = self.manifest.read_bytes()
        report = self.prepare()
        for group, task_id in (("rl_label", "rl"), ("advance_label", "advance")):
            entry = report["groups"][group]
            self.assertEqual(entry["task_ids"], [task_id])
            manifest = TabxManifest(Path(entry["manifest"]))
            manifest.validate_files()
            self.assertEqual(manifest.tasks[0].h5_path, self.converted / f"{task_id}.h5")
            self.assertEqual(manifest.tasks[0].index_path, self.converted / f"{task_id}.jsonl")
            self.assertEqual(manifest.payload["label_grouping"]["action_supervision"], "behavior")
            self.assertEqual(manifest.tasks[0].split, "test" if task_id == "advance" else "train")
        self.assertEqual(list(self.output.rglob("*.h5")), [])
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_missing_raw_task_is_not_mistaken_for_advance(self):
        self.payload["tasks"][0]["task_id"] = "missing"
        self.manifest.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ConversionError, "No raw records.*missing"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_unrelated_raw_tasks_are_excluded_and_mapping_is_configurable(self):
        self.record("unrelated", 0, True)
        self.record("unrelated", 1, False)
        report = self.prepare(distribution_group="advance_label")
        self.assertEqual(report["groups"]["advance_label"]["task_ids"], ["rl"])
        self.assertEqual(report["groups"]["rl_label"]["task_ids"], ["advance"])

    def test_raw_only_scan_and_empty_group(self):
        only_rl = self.raw / "job" / "records" / "rl"
        report = prepare_groups(only_rl, self.output)
        self.assertEqual(report["groups"]["rl_label"]["n_tasks"], 1)
        self.assertEqual(report["groups"]["advance_label"]["n_tasks"], 0)
        self.assertIsNone(report["groups"]["advance_label"]["manifest"])
        self.assertEqual((self.output / "rl_label/task_ids.txt").read_text(), "rl\n")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            run_groups(["--groups-dir", str(self.output), "--group", "advance_label"])
        self.assertEqual(exc.exception.code, 1)

    def test_unavailable_raw_root_errors_instead_of_assigning_all_advance(self):
        with self.assertRaisesRegex(ConversionError, "Raw data root does not exist"):
            prepare_groups(self.root / "unavailable", self.output, self.manifest)

    def test_relative_source_root_is_resolved_from_original_manifest(self):
        self.payload["source_root"] = "../raw"
        self.manifest.write_text(json.dumps(self.payload))
        self.assertEqual(self.prepare()["raw_root"], str(self.raw))

    def test_generated_configs_isolate_every_training_artifact(self):
        original = copy.deepcopy(self.template)
        report = self.prepare(stage0_template=self.template, stage1_template=self.template,
                              steps=17, seed=11, run_root=self.root / "runs")
        norms, outputs, indexes = set(), set(), set()
        for group, entry in report["groups"].items():
            zero = load_yaml_config(Path(entry["stage0_config"]))
            one = load_yaml_config(Path(entry["stage1_config"]))
            self.assertEqual(zero["data"]["manifest"], entry["manifest"])
            self.assertEqual(one["data"], zero["data"])
            self.assertEqual(zero["model"], original["model"])
            self.assertEqual(zero["training"]["learning_rate"], 0.0005)
            self.assertEqual(zero["training"]["total_steps"], 17)
            self.assertEqual(one["training"]["seed"], 11)
            self.assertEqual(zero["training"]["metrics_file"], "training_metrics.csv")
            self.assertEqual(one["retrieval"]["stage0_checkpoint"],
                             str(self.root / "runs" / group / "stage0/step_000000017"))
            norms.add(zero["data"]["normalization"])
            outputs.update([zero["training"]["output_dir"], one["training"]["output_dir"]])
            indexes.add(one["retrieval"]["index_manifest"])
        self.assertEqual((len(norms), len(outputs), len(indexes)), (2, 4, 2))
        self.assertEqual(self.template, original)

    def test_overwrite_conflicts_are_checked_before_any_writes(self):
        path = self.output / "advance_label/manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("existing")
        with self.assertRaisesRegex(ConversionError, "Output exists"):
            self.prepare()
        self.assertFalse((self.output / "rl_label/task_ids.txt").exists())
        self.assertEqual(path.read_text(), "existing")

    def test_read_only_dry_run_plans_both_full_pipelines(self):
        report = self.prepare(stage0_template=self.template, stage1_template=self.template)
        with patch("tools.run_tabx_label_groups.subprocess.run") as run:
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(run_groups(["--groups-dir", str(self.output),
                                             "--stage", "all", "--dry-run"]), 0)
            run.assert_not_called()
        for group in report["groups"]:
            self.assertEqual(printed.getvalue().count(f"[{group}]"), 5)
        self.assertFalse((self.output / "runs").exists())

    def test_runner_invokes_existing_trainers_with_separate_configs(self):
        report = self.prepare(stage0_template=self.template)
        with patch("tools.run_tabx_label_groups.subprocess.run") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                run_groups(["--groups-dir", str(self.output), "--stage", "stage0"])
        self.assertEqual(run.call_count, 2)
        configs = {call.args[0][-1] for call in run.call_args_list}
        self.assertEqual(configs, {entry["stage0_config"] for entry in report["groups"].values()})
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))

    def test_runner_stops_if_a_training_process_fails(self):
        self.prepare(stage0_template=self.template)
        with patch("tools.run_tabx_label_groups.subprocess.run",
                   side_effect=subprocess.CalledProcessError(1, ["trainer"])) as run:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_groups(["--groups-dir", str(self.output), "--stage", "stage0"])
        self.assertEqual(run.call_count, 1)

    def test_missing_stage1_config_fails_before_starting_stage0(self):
        self.prepare(stage0_template=self.template)
        with patch("tools.run_tabx_label_groups.subprocess.run") as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_groups(["--groups-dir", str(self.output), "--stage", "all"])
            run.assert_not_called()

    def test_comparison_uses_last_logged_rows_and_preserves_budget_context(self):
        report = self.prepare()
        for group, entry in report["groups"].items():
            path = Path(entry["run_dir"]) / "stage0/training_metrics.csv"
            path.parent.mkdir(parents=True)
            path.write_text("step,CEM_action_loss,DT_loss\n10,9,\n20,2,\n30,4,\n")
            values = summarize_metrics(path, group, "stage0", 2)
            self.assertEqual(values[0]["tail_mean"], 3)
            self.assertEqual(values[0]["last_step"], 30)
            self.assertAlmostEqual(values[0]["tail_std"], 2 ** 0.5)
        with contextlib.redirect_stdout(io.StringIO()):
            run_groups(["--groups-dir", str(self.output), "--stage", "compare", "--tail-window", "2"])
        with (self.output / "comparison_stage0.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["group"] for row in rows}, {"rl_label", "advance_label"})
        self.assertEqual(len(rows), 2)

    def test_comparison_rejects_duplicate_steps(self):
        path = self.root / "metrics.csv"
        path.write_text("step,CEM_action_loss\n1,2\n1,3\n")
        with self.assertRaisesRegex(ValueError, "duplicate steps"):
            summarize_metrics(path, "rl_label", "stage0", 1000)

    def test_cli_creates_loadable_group_manifests(self):
        completed = subprocess.run([
            sys.executable, str(ROOT / "tools/split_tabx_label_groups.py"),
            "--manifest", str(self.manifest), "--output-dir", str(self.output),
        ], capture_output=True, text=True, check=True)
        self.assertIn("rl_label: tasks=1 records=2", completed.stdout)
        self.assertEqual(TabxManifest(self.output / "advance_label/manifest.json").tasks[0].task_id,
                         "advance")


if __name__ == "__main__":
    unittest.main()
