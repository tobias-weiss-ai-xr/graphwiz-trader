"""Qlib configuration for GraphWiz Trader integration."""

from pathlib import Path
from typing import Dict, Any, Optional
import os


class QlibConfig:
    """Configuration for Qlib integration with GraphWiz Trader."""

    def __init__(
        self,
        provider: str = "ccxt",
        region: str = "crypto",
        freq: str = "1h",
        data_dir: Optional[str] = None,
    ):
        """
        Initialize Qlib configuration.

        Args:
            provider: Data provider (ccxt, local, etc.)
            region: Market region (crypto, us, cn, etc.)
            freq: Data frequency (1h, 1d, 5m, etc.)
            data_dir: Directory for Qlib data storage
        """
        self.provider = provider
        self.region = region
        self.freq = freq

        # Set default data directory
        if data_dir is None:
            # Default to project_root/data/qlib
            project_root = Path(__file__).parent.parent.parent.parent
            data_dir = project_root / "data" / "qlib"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_qlib_config(self) -> Dict[str, Any]:
        """
        Get Qlib configuration dictionary.

        Returns:
            Dictionary with Qlib configuration
        """
        return {
            "provider": self.provider,
            "region": self.region,
            "freq": self.freq,
            "data_dir": str(self.data_dir),
        }

    def get_feature_config(self) -> Dict[str, Any]:
        """
        Get feature engineering configuration.

        Returns:
            Dictionary with feature configuration
        """
        return {
            "alpha158": True,
            "alpha360": False,  # Can be enabled later
            "graph_features": True,  # Enable Neo4j graph features
            "feature_cache": True,
            "feature_cache_dir": str(self.data_dir / "features"),
        }

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dictionary with model configuration
        """
        return {
            "model_type": "lightgbm",
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 256,
            "epochs": 100,
            "early_stopping_rounds": 10,
        }


# Default configuration instance
default_config = QlibConfig()
