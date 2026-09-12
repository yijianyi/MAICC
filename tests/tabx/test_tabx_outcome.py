from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTCOME_PATH = REPO_ROOT / "TABX" / "src" / "tabx" / "outcome.py"
SPEC = importlib.util.spec_from_file_location("tabx_outcome", OUTCOME_PATH)
assert SPEC is not None and SPEC.loader is not None
OUTCOME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OUTCOME)


class TabxOutcomeTests(unittest.TestCase):
    def assert_ally_win(self, team_dones, truncation, expected) -> None:
        actual = OUTCOME.did_ally_win(
            np.asarray(team_dones, dtype=bool), np.asarray(truncation, dtype=bool)
        )
        self.assertEqual(bool(actual), expected)

    def test_ally_wins_when_enemy_is_eliminated(self) -> None:
        self.assert_ally_win([False, True], False, True)

    def test_ally_loses_when_only_ally_is_eliminated(self) -> None:
        self.assert_ally_win([True, False], False, False)

    def test_timeout_is_an_ally_loss(self) -> None:
        self.assert_ally_win([False, False], True, False)
        self.assert_ally_win([False, True], True, False)

    def test_simultaneous_elimination_is_an_ally_win(self) -> None:
        self.assert_ally_win([True, True], False, True)


if __name__ == "__main__":
    unittest.main()
