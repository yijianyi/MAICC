from __future__ import annotations

from typing import Mapping

import numpy as np


class TerminalPaddingError(ValueError):
    pass


def valid_transition_length(
    filled,
    terminated,
    sequence_length: int,
    allow_terminal_padding: bool,
) -> int:
    """Validate one MAICC row and return its number of real transitions."""

    filled_values = np.asarray(filled).reshape(-1)
    terminated_values = np.asarray(terminated).reshape(-1)
    expected = int(sequence_length) + 1
    if filled_values.shape != (expected,):
        raise TerminalPaddingError(
            f"filled shape {filled_values.shape}, expected {(expected,)}"
        )
    if terminated_values.shape != (expected,):
        raise TerminalPaddingError(
            "terminated shape "
            f"{terminated_values.shape}, expected {(expected,)}"
        )
    if not np.all(np.logical_or(filled_values == 0, filled_values == 1)):
        raise TerminalPaddingError("filled must contain only 0/1 values")

    zero_positions = np.flatnonzero(filled_values == 0)
    valid_slots = (
        int(zero_positions[0]) if zero_positions.size else expected
    )
    if not np.all(filled_values[:valid_slots] == 1) or not np.all(
        filled_values[valid_slots:] == 0
    ):
        raise TerminalPaddingError(
            "filled must be a contiguous prefix of ones followed by zeros"
        )

    transitions = valid_slots - 1
    if transitions < 1:
        raise TerminalPaddingError(
            "a baseline row must contain at least one real transition"
        )
    if transitions > sequence_length:
        raise TerminalPaddingError(
            f"valid transition length {transitions} exceeds {sequence_length}"
        )
    if transitions < sequence_length:
        if not allow_terminal_padding:
            raise TerminalPaddingError(
                "terminal padding is not allowed by the strict manifest"
            )
        if terminated_values[transitions - 1] != 1:
            raise TerminalPaddingError(
                "a padded row must terminate on its final real transition"
            )
        if np.any(terminated_values[transitions:] != 0):
            raise TerminalPaddingError(
                "terminated must be zero after the final real transition"
            )
    return transitions


def validate_zero_padding(
    arrays: Mapping[str, np.ndarray],
    transitions: int,
    sequence_length: int,
) -> None:
    """Require converter-created invalid slots to contain only zero values."""

    if transitions >= int(sequence_length):
        return
    # The slot at ``transitions`` is the terminal next-observation slot. TABX
    # has no post-terminal observation, so the converter intentionally leaves
    # it and all later invalid slots at zero.
    for field in ("obs", "actions", "reward", "terminated"):
        values = np.asarray(arrays[field])
        if np.any(values[transitions:] != 0):
            raise TerminalPaddingError(
                f"{field} contains non-zero values in terminal padding"
            )
