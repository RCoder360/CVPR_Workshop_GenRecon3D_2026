"""
Configuration loader for KAN-Refine.

Loads YAML config files and provides a nested attribute-access dict.
"""

import yaml
import os
import copy
from pathlib import Path
from yaml.constructor import ConstructorError


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


def _to_plain_dict(obj):
    """Recursively convert ConfigDict / nested mappings to plain Python dicts."""
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain_dict(v) for v in obj)
    return obj


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
            try:
                base_cfg = yaml.safe_load(f) or {}
            except ConstructorError:
                # Backward compatibility for previously dumped ConfigDict YAML tags.
                f.seek(0)
                base_cfg = yaml.unsafe_load(f) or {}
    else:
        base_cfg = {}

    # Load experiment config
    with open(config_path, "r") as f:
        try:
            exp_cfg = yaml.safe_load(f) or {}
        except ConstructorError:
            # Backward compatibility for previously dumped ConfigDict YAML tags.
            f.seek(0)
            exp_cfg = yaml.unsafe_load(f) or {}

    base_cfg = _to_plain_dict(base_cfg)
    exp_cfg = _to_plain_dict(exp_cfg)

    merged = _deep_merge(copy.deepcopy(base_cfg), exp_cfg)
    return ConfigDict(merged)


def save_config(cfg: dict, save_path: str) -> None:
    """Save a config dict to a YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        plain_cfg = _to_plain_dict(cfg)
        yaml.safe_dump(plain_cfg, f, default_flow_style=False, sort_keys=False)
