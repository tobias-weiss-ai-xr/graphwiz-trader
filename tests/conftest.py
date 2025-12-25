"""Pytest configuration and fixtures for graphwiz-trader tests."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()

    # Setup mock result
    mock_result.data.return_value = [{"key": "value"}]
    mock_result.consume.return_value = MagicMock()

    # Setup mock session
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=False)
    mock_session.run.return_value = mock_result

    # Setup mock driver
    mock_driver.session.return_value = mock_session
    mock_driver.verify_connectivity.return_value = None
    mock_driver.close.return_value = None

    return mock_driver


@pytest.fixture
def neo4j_config():
    """Neo4j configuration for testing."""
    return {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "test_password",
        "database": "neo4j"
    }


@pytest.fixture
def trading_config():
    """Trading configuration for testing."""
    return {
        "max_position_size": 1000,
        "risk_per_trade": 0.02,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.15
    }


@pytest.fixture
def exchanges_config():
    """Exchange configuration for testing."""
    return {
        "binance": {
            "enabled": True,
            "api_key": "test_key",
            "api_secret": "test_secret",
            "sandbox": True,
            "test_mode": True
        },
        "kraken": {
            "enabled": False,
            "api_key": "test_key_kraken",
            "api_secret": "test_secret_kraken"
        }
    }


@pytest.fixture
def agents_config():
    """Agent configuration for testing."""
    return {
        "technical": {
            "enabled": True,
            "model": "gpt-4",
            "temperature": 0.7
        },
        "sentiment": {
            "enabled": True,
            "model": "gpt-3.5-turbo",
            "temperature": 0.5
        },
        "risk": {
            "enabled": False,
            "model": "gpt-4"
        }
    }


@pytest.fixture
def mock_kg():
    """Mock knowledge graph instance."""
    kg = MagicMock()
    kg.query.return_value = [{"result": "data"}]
    kg.write.return_value = MagicMock()
    return kg


@pytest.fixture
def mock_agent_orchestrator():
    """Mock agent orchestrator instance."""
    orchestrator = MagicMock()
    orchestrator.get_decision.return_value = {
        "action": "buy",
        "confidence": 0.75,
        "reason": "Positive momentum"
    }
    return orchestrator


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    config_content = """
version: "0.1.0"

neo4j:
  uri: bolt://localhost:7687
  username: neo4j
  password: test_password
  database: neo4j

trading:
  max_position_size: 1000
  risk_per_trade: 0.02

exchanges:
  binance:
    enabled: true
    api_key: test_key
    api_secret: test_secret
    sandbox: true

agents:
  technical:
    enabled: true
    model: gpt-4
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_ccxt_exchange():
    """Mock CCXT exchange instance."""
    exchange = MagicMock()
    exchange.fetch_ticker.return_value = {
        "symbol": "BTC/USDT",
        "last": 50000,
        "bid": 49999,
        "ask": 50001
    }
    exchange.create_order.return_value = {
        "id": "12345",
        "symbol": "BTC/USDT",
        "type": "market",
        "side": "buy",
        "price": 50000,
        "amount": 0.1
    }
    exchange.close.return_value = None
    return exchange
