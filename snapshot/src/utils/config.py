"""Configuration loading and validation for VariationSampler."""

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config is not a valid dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(config).__name__}")

    logger.info("Loaded config: %s", path)
    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override config into base config.

    Override values replace base values. Nested dicts are merged recursively.

    Args:
        base: Base config dictionary.
        override: Override values to apply.

    Returns:
        New merged config dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


_REQUIRED_SECTIONS = ["model", "masking", "sampling", "training"]

_VALIDATION_RULES: list[tuple[str, type, Any, Any]] = [
    # (dotted_path, expected_type, min_value, max_value)
    ("model.d_model", int, 32, 2048),
    ("model.n_layers", int, 1, 48),
    ("model.n_heads", int, 1, 64),
    ("masking.p_tail", float, 0.0, 1.0),
    ("masking.p_attack", float, 0.0, 1.0),
    ("sampling.temperature", float, 0.01, 5.0),
    ("sampling.top_p", float, 0.0, 1.0),
    ("training.learning_rate", float, 1e-7, 1.0),
    ("training.batch_size", int, 1, 4096),
]


def _get_nested(config: dict, dotted_key: str) -> Any:
    """Get a value from a nested dict using dot notation."""
    keys = dotted_key.split(".")
    value = config
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            return None
        value = value[k]
    return value


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate config structure and value ranges.

    Args:
        config: Config dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    for section in _REQUIRED_SECTIONS:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")

    for dotted_key, expected_type, min_val, max_val in _VALIDATION_RULES:
        value = _get_nested(config, dotted_key)
        if value is None:
            errors.append(f"Missing required key: '{dotted_key}'")
            continue
        if not isinstance(value, expected_type):
            # Allow int where float is expected
            if expected_type is float and isinstance(value, int):
                value = float(value)
            else:
                errors.append(
                    f"'{dotted_key}' must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
                continue
        if value < min_val or value > max_val:
            errors.append(f"'{dotted_key}' = {value} out of range [{min_val}, {max_val}]")

    if errors:
        logger.warning("Config validation found %d error(s)", len(errors))
    return errors
