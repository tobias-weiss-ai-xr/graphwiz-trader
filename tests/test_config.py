"""Tests for configuration utilities."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from graphwiz_trader.utils.config import load_config


class TestLoadConfig:
    """Test suite for load_config function."""

    def test_load_valid_yaml(self, temp_config_file):
        """Test loading a valid YAML configuration file."""
        config = load_config(temp_config_file)

        assert config is not None
        assert isinstance(config, dict)
        assert config.get("version") == "0.1.0"
        assert "neo4j" in config
        assert "trading" in config
        assert "exchanges" in config
        assert "agents" in config

    def test_load_nonexistent_file(self):
        """Test loading a non-existent configuration file."""
        config = load_config("nonexistent_config.yaml")

        assert config == {}

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config == {}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_yaml_with_comments(self):
        """Test loading YAML with comments."""
        yaml_content = """
# This is a comment
version: "0.1.0"

# Neo4j configuration
neo4j:
  uri: bolt://localhost:7687
  username: neo4j
  password: password123
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["version"] == "0.1.0"
            assert config["neo4j"]["uri"] == "bolt://localhost:7687"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_nested_config(self):
        """Test loading configuration with nested structures."""
        yaml_content = """
trading:
  risk:
    per_trade: 0.02
    max_drawdown: 0.2
  strategies:
    - name: momentum
      params:
        period: 14
    - name: mean_reversion
      params:
        period: 20
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["trading"]["risk"]["per_trade"] == 0.02
            assert len(config["trading"]["strategies"]) == 2
            assert config["trading"]["strategies"][0]["name"] == "momentum"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_config_with_special_types(self):
        """Test loading configuration with various data types."""
        yaml_content = """
string_value: "hello"
number_value: 42
float_value: 3.14
bool_value: true
null_value: null
list_value:
  - item1
  - item2
dict_value:
  key: value
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["string_value"] == "hello"
            assert config["number_value"] == 42
            assert config["float_value"] == 3.14
            assert config["bool_value"] is True
            assert config["null_value"] is None
            assert len(config["list_value"]) == 2
            assert config["dict_value"]["key"] == "value"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch('graphwiz_trader.utils.config.logger')
    def test_load_invalid_yaml(self, mock_logger):
        """Test loading an invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("{ invalid yaml content [[[")

            temp_path = f.name

        try:
            config = load_config(temp_path)
            # Should return empty dict on error
            assert config == {}
            # Should log error
            mock_logger.error.assert_called()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_config_with_env_variables(self):
        """Test loading configuration with environment variable-like placeholders."""
        yaml_content = """
database:
  host: localhost
  port: 7687
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["database"]["host"] == "localhost"
            assert config["database"]["port"] == 7687
        finally:
            Path(temp_path).unlink(missing_ok=True)
