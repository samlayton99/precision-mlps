"""YAML config loading and sweep expansion.

Usage:
    config = load_config("experiments/exp01/config.yaml")
    configs = expand_sweep(config)
"""

from __future__ import annotations

import yaml
from dataclasses import asdict, fields
from itertools import product
from copy import deepcopy
from typing import Any

from src.config.schema import ExperimentConfig


def load_config(yaml_path: str, overrides: dict | None = None) -> ExperimentConfig:
    """Load YAML, merge with defaults, apply overrides.

    Nested YAML keys map to sub-dataclass fields:
        model:
          width: 128
    becomes config.model.width = 128.
    """
    # TODO: implement
    raise NotImplementedError


def config_to_yaml(config: ExperimentConfig, path: str | None = None) -> str:
    """Serialize config to YAML. Write to file if path given."""
    # TODO: implement
    raise NotImplementedError


def expand_sweep(config: ExperimentConfig) -> list[ExperimentConfig]:
    """Cartesian product over config.sweep axes.

    config.sweep = {"construction.lambda_star": [1.0, 1.5], "model.width": [32, 64]}
    returns 4 configs. If sweep is empty, returns [config].
    """
    # TODO: implement
    raise NotImplementedError


def _set_nested(obj: Any, dotted_key: str, value: Any) -> None:
    """Set a nested field: _set_nested(config, "model.width", 128)."""
    # TODO: implement
    raise NotImplementedError
