from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            key = value[2:-1]
            return os.environ.get(key, value)
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @property
    def results_dir(self) -> Path:
        root = Path(self.raw["paths"]["results_root"])
        name = self.raw["experiment"]["name"]
        return root / name

    @property
    def cache_dir(self) -> Path:
        return Path(self.raw["paths"]["cache_root"])

    def save_resolved(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.raw, sort_keys=False), encoding="utf-8")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = _expand_env(cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate_config(cfg)
    return ExperimentConfig(cfg)


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ["experiment", "paths", "data", "protocol", "model", "loss", "train", "verification"]:
        if section not in cfg:
            raise ConfigError(f"Missing config section: {section}")
    raw_root = cfg["paths"].get("bed_raw_root")
    manifest = cfg["paths"].get("manifest_csv")
    if raw_root in (None, "", "${BED_RAW_ROOT}") and not manifest:
        raise ConfigError("Set paths.bed_raw_root or paths.manifest_csv before running.")
    if cfg["data"]["target_sampling_rate"] <= 0:
        raise ConfigError("data.target_sampling_rate must be positive.")
    if cfg["data"]["window_sec"] <= 0 or cfg["data"]["step_sec"] <= 0:
        raise ConfigError("data.window_sec and data.step_sec must be positive.")
