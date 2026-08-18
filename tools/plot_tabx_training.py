#!/usr/bin/env python3
"""Plot TABX step-wise training losses and sampled offline rewards."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


LOSS_COLUMNS = (
    "CEM_action_loss",
    "CEM_reward_loss",
    "CEM_obs_loss",
    "DEM_action_loss",
    "DEM_obs_loss",
    "KL_action_loss",
    "KL_reward_loss",
    "KL_obs_loss",
    "DT_loss",
)
REWARD_COLUMNS = (
    ("batch_mean_reward", "Mean team reward per valid timestep"),
    ("batch_team_return", "Mean team return per sampled window"),
    ("batch_win_fraction", "Terminal-win fraction in sampled windows"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training_metrics.csv produced by TABX training."
    )
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=200,
        help="Number of logged steps in the moving average (default: 200).",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def read_metrics(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Metric CSV has no data rows: {path}")
    steps = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
    columns: Dict[str, np.ndarray] = {}
    names: Sequence[str] = tuple(LOSS_COLUMNS) + tuple(
        name for name, _ in REWARD_COLUMNS
    )
    for name in names:
        values: List[float] = []
        for row in rows:
            raw = row.get(name, "")
            try:
                values.append(float(raw) if raw not in (None, "") else math.nan)
            except ValueError:
                values.append(math.nan)
        columns[name] = np.asarray(values, dtype=np.float64)
    return steps, columns


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    width = max(1, int(window))
    if width == 1 or values.size <= 1:
        return values.copy()
    finite = np.isfinite(values)
    numeric = np.where(finite, values, 0.0)
    kernel = np.ones(width, dtype=np.float64)
    totals = np.convolve(numeric, kernel, mode="full")[: values.size]
    counts = np.convolve(finite.astype(np.float64), kernel, mode="full")[
        : values.size
    ]
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def draw_series(
    axis,
    steps: np.ndarray,
    values: np.ndarray,
    title: str,
    smooth_window: int,
) -> None:
    finite = np.isfinite(values)
    axis.plot(
        steps[finite], values[finite],
        color="#9ecae1", linewidth=0.6, alpha=0.35, label="raw",
    )
    smooth = moving_average(values, smooth_window)
    smooth_finite = np.isfinite(smooth)
    axis.plot(
        steps[smooth_finite], smooth[smooth_finite],
        color="#08519c", linewidth=1.6,
        label=f"moving average ({smooth_window})",
    )
    axis.set_title(title)
    axis.set_xlabel("Training step")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def plot_losses(
    steps: np.ndarray,
    columns: Dict[str, np.ndarray],
    output: Path,
    smooth_window: int,
    dpi: int,
) -> None:
    active = [
        name for name in LOSS_COLUMNS
        if np.isfinite(columns[name]).any()
    ]
    if not active:
        raise ValueError("No finite loss columns were found in the metric CSV")
    n_columns = 3
    n_rows = int(math.ceil(len(active) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(5.4 * n_columns, 3.7 * n_rows),
        squeeze=False,
    )
    for axis, name in zip(axes.flat, active):
        draw_series(axis, steps, columns[name], name, smooth_window)
        axis.set_ylabel("Loss")
    for axis in list(axes.flat)[len(active):]:
        axis.set_visible(False)
    figure.suptitle("TABX training losses", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_rewards(
    steps: np.ndarray,
    columns: Dict[str, np.ndarray],
    output: Path,
    smooth_window: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(REWARD_COLUMNS), 1,
        figsize=(12, 3.6 * len(REWARD_COLUMNS)),
        squeeze=False,
    )
    for axis, (name, title) in zip(axes.flat, REWARD_COLUMNS):
        draw_series(axis, steps, columns[name], title, smooth_window)
        axis.set_ylabel(name)
    figure.suptitle(
        "TABX sampled offline rewards (diagnostic, not online performance)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    metrics = args.metrics.expanduser().resolve()
    output_dir = (
        metrics.parent
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, columns = read_metrics(metrics)
    loss_path = output_dir / "loss_curves.png"
    reward_path = output_dir / "reward_curves.png"
    plot_losses(
        steps, columns, loss_path,
        smooth_window=args.smooth_window, dpi=args.dpi,
    )
    plot_rewards(
        steps, columns, reward_path,
        smooth_window=args.smooth_window, dpi=args.dpi,
    )
    print(f"Wrote loss plot: {loss_path}")
    print(f"Wrote reward plot: {reward_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
