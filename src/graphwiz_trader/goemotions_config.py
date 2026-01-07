"""
GoEmotions trading configuration loader.

This module provides configuration management for GoEmotions-based trading.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigError(Exception):
    pass


class ConfigLoader:
    """Configuration loader with environment variable substitution."""
    
    @staticmethod
    def load(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_str = f.read()
            
            config_str = ConfigLoader._substitute_env_vars(config_str)
            config = yaml.safe_load(config_str)
            
            return config if config else {}
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration: {e}")
    
    @staticmethod
    def _substitute_env_vars(text: str) -> str:
        """Substitute environment variables in configuration string."""
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ''
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replacer, text)
    
    @staticmethod
    def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get nested value from configuration dictionary using dot notation."""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


class GoEmotionsConfig:
    """GoEmotions trading configuration manager."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "goemotions_trading.yaml"
        
        self.config_path = Path(config_path)
        self._config = ConfigLoader.load(config_path)
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration sections."""
        required_sections = ['goemotions', 'technical_indicators', 'signal_generation']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigError(f"Missing required configuration section: {section}")
    
    @property
    def trading_mode(self) -> str:
        """Get trading mode (live or paper)."""
        return self._config.get('trading_mode', {}).get('type', 'live')
    
    @property
    def is_live_trading(self) -> bool:
        """Check if running in live trading mode."""
        return self.trading_mode == 'live'
    
    @property
    def symbols(self) -> list:
        """Get trading symbols for current mode."""
        mode = self.trading_mode
        return self._config.get('symbols', {}).get(mode, ['BTC/EUR'])
    
    @property
    def goemotions_enabled(self) -> bool:
        """Check if GoEmotions is enabled."""
        return self._config.get('goemotions', {}).get('enabled', True)
    
    def get_goemotions_config(self) -> Dict[str, Any]:
        """Get GoEmotions strategy configuration."""
        return self._config.get('goemotions', {})
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get GoEmotions strategy configuration."""
        return self._config.get('goemotions', {}).get('strategy', {})
    
    def get_emotions_config(self) -> Dict[str, Any]:
        """Get emotions configuration."""
        return self._config.get('goemotions', {}).get('emotions', {})
    
    def get_live_trading_config(self) -> Dict[str, Any]:
        """Get live trading configuration."""
        return self._config.get('live_trading', {})
    
    def get_paper_trading_config(self) -> Dict[str, Any]:
        """Get paper trading configuration."""
        return self._config.get('paper_trading', {})
    
    def get_technical_indicators_config(self) -> Dict[str, Any]:
        """Get technical indicators configuration."""
        return self._config.get('technical_indicators', {})
    
    def get_signal_generation_config(self) -> Dict[str, Any]:
        """Get signal generation configuration."""
        return self._config.get('signal_generation', {})
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get risk limits configuration."""
        return self._config.get('risk_limits', {})
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        return ConfigLoader.get_nested_value(self._config, key_path, default)
    
    def reload(self):
        """Reload configuration from file."""
        self._config = ConfigLoader.load(self.config_path)
        self._validate_config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> GoEmotionsConfig:
    """Convenience function to load GoEmotions configuration."""
    return GoEmotionsConfig(config_path)
