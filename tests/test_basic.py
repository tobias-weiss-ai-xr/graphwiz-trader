"""Basic tests for graphwiz-trader."""

import pytest


def test_imports():
    """Test that basic imports work."""
    from graphwiz_trader import __version__

    assert __version__ is not None
    assert __version__ == "0.1.0"


def test_version():
    """Test package version."""
    from graphwiz_trader import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_main_import():
    """Test that main GraphWizTrader class can be imported."""
    from graphwiz_trader import GraphWizTrader

    assert GraphWizTrader is not None
    assert callable(GraphWizTrader)


def test_config_loading():
    """Test that configuration can be loaded."""
    from graphwiz_trader.utils.config import load_config

    # Test with a non-existent config (should return empty dict)
    config = load_config("nonexistent.yml")
    assert config == {}


def test_knowledge_graph_import():
    """Test that KnowledgeGraph can be imported."""
    from graphwiz_trader.graph import KnowledgeGraph

    assert KnowledgeGraph is not None


def test_trading_engine_import():
    """Test that TradingEngine can be imported."""
    from graphwiz_trader.trading import TradingEngine

    assert TradingEngine is not None


def test_agent_orchestrator_import():
    """Test that AgentOrchestrator can be imported."""
    from graphwiz_trader.agents import AgentOrchestrator

    assert AgentOrchestrator is not None


def test_package_metadata():
    """Test package metadata."""
    from graphwiz_trader import __version__, __author__, __email__

    assert __version__ == "0.1.0"
    assert __author__ == "Tobias Weiss"
    assert "@" in __email__
