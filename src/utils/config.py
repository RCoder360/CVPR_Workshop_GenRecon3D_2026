"""
Configuration loader for KAN-Refine.

Loads YAML config files and provides a nested attribute-access dict.
"""

import yaml
import os
import copy
from pathlib import Path


class ConfigDict(dict):
    """Dictionary subclass allowing attribute-style access."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        if isinstance(val, dict) and not isinstance(val, ConfigDict):
            val = ConfigDict(val)
            self[key] = val
        return val

    def __setattr__(self, key, value):
        self[key] = value

    def __deepcopy__(self, memo):
        return ConfigDict(copy.deepcopy(dict(self), memo))


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place) and return base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str, default_path: str | None = None) -> ConfigDict:
    """Load a YAML config, optionally merging on top of a default config.

    Parameters
    ----------
    config_path : str
        Path to the experiment-specific YAML file.
    default_path : str, optional
        Path to the base / default YAML. If *None*, we look for
        ``configs/default.yaml`` relative to the repo root.

    Returns
    -------
    ConfigDict
        Merged configuration object.
    """
    repo_root = Path(__file__).resolve().parents[2]  # src/utils -> repo root

    # Load default
    if default_path is None:
        default_path = repo_root / "configs" / "default.yaml"
    if os.path.isfile(default_path):
        with open(default_path, "r") as f:
            base_cfg = yaml.safe_load(f) or {}
    else:
        base_cfg = {}

    # Load experiment config
    with open(config_path, "r") as f:
        exp_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(copy.deepcopy(base_cfg), exp_cfg)
    return ConfigDict(merged)


def save_config(cfg: dict, save_path: str) -> None:
    """Save a config dict to a YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)
