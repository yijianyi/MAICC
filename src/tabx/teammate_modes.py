"""Validated teammate evaluation conditions; independent of JAX and model loading."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

from .config import load_yaml_config

POLICIES = {"random", "novice", "medium", "advanced", "expert"}


def validate_policies(policies: Mapping) -> dict[int, str]:
    if not isinstance(policies, Mapping):
        raise ValueError("policies must map zero-based ally IDs to policy names")
    result = {}
    for key, name in policies.items():
        if isinstance(key, bool) or not re.fullmatch(r"[0-9]+", str(key)):
            raise ValueError(f"Invalid teammate agent ID: {key!r}")
        agent_id = int(key)
        if agent_id in result or not isinstance(name, str) or name not in POLICIES:
            raise ValueError(f"Duplicate agent ID or unknown policy: {key!r}: {name!r}")
        result[agent_id] = name
    return dict(sorted(result.items()))


def load_teammate_modes(path: Path | None) -> list[dict]:
    if path is None:
        return [{"name": "all_model", "policies": {}}]
    modes = load_yaml_config(path).get("teammate_modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError("teammate_modes must be a nonempty list")
    result = []
    names = set()
    for mode in modes:
        if not isinstance(mode, dict) or set(mode) != {"name", "policies"}:
            raise ValueError("Each mode requires exactly name and policies")
        name = mode["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"Invalid or duplicate teammate mode name: {name!r}")
        policies = validate_policies(mode["policies"])
        if (name == "all_model") != (not policies):
            raise ValueError("The empty policy mapping must be named all_model, exclusively")
        names.add(name)
        result.append({"name": name, "policies": policies})
    return result


def condition_metadata(task_id: str, mode: dict) -> dict:
    return {
        "condition_id": json.dumps([task_id, mode["name"]], separators=(",", ":")),
        "teammate_mode": mode["name"],
        "teammate_policies": json.dumps(mode["policies"], sort_keys=True),
        "replaced_agent_ids": json.dumps(list(mode["policies"])),
        "replacement_count": len(mode["policies"]),
    }


def add_baseline_deltas(results: list[dict]) -> None:
    baselines = {r["task_id"]: r for r in results if r["teammate_mode"] == "all_model"}
    for result in results:
        baseline = baselines.get(result["task_id"])
        for metric in ("win_rate", "mean_return"):
            result[f"delta_{metric}"] = (
                result[metric] - baseline[metric] if baseline else None
            )
