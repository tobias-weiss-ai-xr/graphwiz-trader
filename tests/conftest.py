"""Pytest configuration and fixtures for graphwiz-trader tests."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, may use external services)")
    config.addinivalue_line("markers", "slow: Slow-running tests (property-based, large datasets)")
    config.addinivalue_line("markers", "property: Property-based tests using Hypothesis")
    config.addinivalue_line("markers", "hft: High-frequency trading tests")


@pytest.fixture(scope="session")
def mock_neo4j_driver():
    """Mock Neo4j driver (session-scoped for performance)."""
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


@pytest.fixture(scope="session")
def neo4j_config():
    """Neo4j configuration for testing (session-scoped)."""
    return {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "test_password",
        "database": "neo4j"
    }


@pytest.fixture(scope="session")
def trading_config():
    """Trading configuration for testing (session-scoped)."""
    return {
        "max_position_size": 1000,
        "risk_per_trade": 0.02,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.15
    }


@pytest.fixture(scope="session")
def exchanges_config():
    """Exchange configuration for testing (session-scoped)."""
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


@pytest.fixture(scope="session")
def agents_config():
    """Agent configuration for testing (session-scoped)."""
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


@pytest.fixture(scope="session")
def sample_prices():
    """Sample price data for testing (session-scoped)."""
    import numpy as np
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    return prices[:100]


@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """Sample OHLCV data for testing (session-scoped)."""
    from datetime import datetime, timedelta
    import numpy as np

    np.random.seed(42)
    data = []
    base_price = 50000
    now = datetime.now()

    for i in range(100):
        open_price = base_price + np.random.randn() * 100
        high = open_price + abs(np.random.randn() * 50)
        low = open_price - abs(np.random.randn() * 50)
        close = open_price + np.random.randn() * 20
        volume = 1000000 + np.random.randn() * 100000

        data.append({
            "timestamp": now + timedelta(minutes=i),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })

    return data


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
