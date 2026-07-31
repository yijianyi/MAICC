from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping


class ConfigError(RuntimeError):
    pass


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to load TABX configs. Install requirements.txt."
        ) from exc

    config_path = path.expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"Cannot read config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Expected a YAML mapping in {config_path}")
    result = _expand(payload)
    result["_config_path"] = str(config_path)
    return result


def deep_update(
    target: MutableMapping[str, Any], source: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in source.items():
        if (
            isinstance(value, Mapping)
            and isinstance(target.get(key), MutableMapping)
        ):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        value = value[part]
    return value


def resolve_path(value: str, config: Mapping[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_path = Path(str(config["_config_path"]))
    return (config_path.parent / path).resolve()
