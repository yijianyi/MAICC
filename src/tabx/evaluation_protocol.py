"""Evaluation targets derived only from permitted training information."""
import numpy as np


def resolve_return_target(manifest, task, evaluation):
    mode = evaluation.get("target_return_mode", "train_only")
    train_tasks = manifest.tasks_for_split("train")
    if mode == "fixed":
        target = float(evaluation["target_return"])
        scale = float(evaluation["return_scale"])
        source = "fixed_config"
    elif mode == "train_only":
        if task.split == "train":
            target, scale = task.target_return, task.return_scale
            source = "training_task_statistics"
        else:
            # No outcomes, successful trajectories or reward statistics from
            # the held-out task may determine its target or normalization.
            scale = float(np.median([item.return_scale for item in train_tasks]))
            normalized_target = float(np.median([
                item.target_return / item.return_scale for item in train_tasks]))
            target = normalized_target * scale
            source = "training_tasks_median"
    else:
        raise ValueError(f"Unknown target_return_mode: {mode}")
    if not np.isfinite([target, scale]).all() or scale <= 0:
        raise ValueError("Evaluation target must be finite and return_scale positive")
    return {"target_return": float(target), "return_scale": float(scale),
            "target_source": source, "task_split": task.split}
