"""Configuration utilities."""

import yaml
from pathlib import Path
from loguru import logger
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)

    if not path.exists():
        logger.warning("Configuration file not found: {}", config_path)
        return {}

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded configuration from {}", config_path)
        return config or {}

    except Exception as e:
        logger.error("Failed to load configuration: {}", e)
        return {}
