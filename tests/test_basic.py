"""Basic tests for graphwiz-trader."""

import pytest


def test_imports():
    """Test that basic imports work."""
    from graphwiz_trader import __version__

    assert __version__ is not None


def test_config_loading():
    """Test that configuration can be loaded."""
    from graphwiz_trader.utils.config import load_config

    # Test with a non-existent config (should return empty dict)
    config = load_config("nonexistent.yml")
    assert config == {}
