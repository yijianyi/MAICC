from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.eval_tabx_online import (
    EPISODE_FIELDS,
    append_episode,
    prepare_resume,
)


class OnlineEvalReportingTests(unittest.TestCase):
    def test_episode_csv_records_result_and_running_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            append_episode(path, "task", 1, 40, True, 2.5, 12, 1, 0)
            append_episode(path, "task", 2, 42, False, -1.0, 20, 1, 1)

            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(tuple(rows[0]), EPISODE_FIELDS)
            self.assertEqual([row["seed"] for row in rows], ["40", "42"])
            self.assertEqual([row["result"] for row in rows], ["WIN", "LOSS"])
            self.assertEqual(rows[-1]["cumulative_wins"], "1")
            self.assertEqual(rows[-1]["cumulative_losses"], "1")
            self.assertEqual(rows[-1]["running_win_rate"], "0.5")

    def test_resume_does_not_keep_results_from_a_different_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "task",
                                "episodes": 2,
                                "seed_start": 10,
                                "seed_end": 11,
                                "seed_stride": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "episodes.csv").write_text(
                "task_id,episode,seed,win,return,length\n"
                "task,1,10,1,1.0,10\n"
                "task,2,11,0,0.0,20\n",
                encoding="utf-8",
            )

            results, completed = prepare_resume(
                output_dir,
                [SimpleNamespace(task_id="task")],
                episodes=2,
                base_seed=20,
                seed_stride=1,
                enabled=True,
            )

            self.assertEqual(results, [])
            self.assertEqual(completed, set())
            with (output_dir / "episodes.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [])

    def test_each_task_uses_the_same_seed_interval_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            task_results = [
                {
                    "task_id": task_id,
                    "episodes": 2,
                    "seed_start": 0,
                    "seed_end": 1,
                    "seed_stride": 1,
                    "wins": 1,
                }
                for task_id in ("a", "b")
            ]
            (output_dir / "summary.json").write_text(
                json.dumps({"tasks": task_results}), encoding="utf-8"
            )
            (output_dir / "episodes.csv").write_text(
                "task_id,episode,seed,win,return,length\n"
                "a,1,0,1,1.0,10\n"
                "a,2,1,0,0.0,20\n"
                "b,1,0,0,0.0,15\n"
                "b,2,1,1,1.0,12\n",
                encoding="utf-8",
            )

            results, completed = prepare_resume(
                output_dir,
                [SimpleNamespace(task_id="a"), SimpleNamespace(task_id="b")],
                episodes=2,
                base_seed=0,
                seed_stride=1,
                enabled=True,
            )

            self.assertEqual(completed, {"a", "b"})
            self.assertEqual(len(results), 2)
            self.assertTrue(all(item["seed_start"] == 0 for item in results))
            self.assertTrue(all(item["seed_end"] == 1 for item in results))


if __name__ == "__main__":
    unittest.main()
