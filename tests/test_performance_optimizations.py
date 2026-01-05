#!/usr/bin/env python3
"""Performance optimization tests for GraphWiz Trader.

This script tests and validates the performance optimizations made to:
1. Neo4j connection pooling and query caching
2. Trading engine ticker caching and parallel execution
3. Query batching for bulk writes
"""

import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graphwiz_trader.graph.neo4j_graph import KnowledgeGraph
from graphwiz_trader.trading.engine import TradingEngine
from loguru import logger


def test_neo4j_caching():
    """Test Neo4j query caching performance."""
    logger.info("Testing Neo4j query caching...")

    # Mock the driver
    with patch('graphwiz_trader.graph.neo4j_graph.GraphDatabase') as mock_driver:
        mock_instance = Mock()
        mock_driver.driver.return_value = mock_instance

        kg = KnowledgeGraph({
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "test",
            "query_cache_enabled": True,
            "query_cache_ttl": 300
        })

        kg.driver = mock_instance

        # Mock session
        mock_session = Mock()
        mock_result = Mock()
        mock_result.data.return_value = [{"test": "data"}]
        mock_session.run.return_value = mock_result
        mock_instance.session.return_value.__enter__.return_value = mock_session
        mock_instance.session.return_value.__exit__.return_value = None

        # Test cache performance
        query = "MATCH (n) RETURN n LIMIT 10"

        # First call - cache miss
        start = time.time()
        result1 = kg.query(query)
        first_call_time = time.time() - start

        # Second call - cache hit
        start = time.time()
        result2 = kg.query(query)
        second_call_time = time.time() - start

        # Verify cache is faster (should be significantly faster)
        logger.info("First call: {:.4f}s, Cached call: {:.4f}s, Speedup: {:.1f}x",
                   first_call_time, second_call_time, first_call_time / second_call_time if second_call_time > 0 else float('inf'))

        # Verify results are the same
        assert result1 == result2, "Cached results should match original results"

        # Check metrics
        metrics = kg.get_metrics()
        logger.info("Cache size: {}", metrics["cache_size"])

        assert metrics["cache_size"] == 1, f"Expected 1 cached entry, got {metrics['cache_size']}"

        logger.success("✅ Neo4j caching test passed")


def test_neo4j_batch_operations():
    """Test Neo4j batch operation performance."""
    logger.info("Testing Neo4j batch operations...")

    with patch('graphwiz_trader.graph.neo4j_graph.GraphDatabase') as mock_driver:
        mock_instance = Mock()
        mock_driver.driver.return_value = mock_instance

        kg = KnowledgeGraph({
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "test",
            "batch_size": 100
        })

        kg.driver = mock_instance

        # Mock session and transaction
        mock_session = Mock()
        mock_tx = Mock()
        mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
        mock_session.begin_transaction.return_value.__exit__.return_value = None
        mock_instance.session.return_value.__enter__.return_value = mock_session
        mock_instance.session.return_value.__exit__.return_value = None

        # Add multiple operations to batch
        num_operations = 50
        for i in range(num_operations):
            kg.add_to_batch(
                "CREATE (n:Test {id: $id})",
                id=i
            )

        # Flush batch
        start = time.time()
        kg.flush_batch()
        batch_time = time.time() - start

        logger.info("Batched {} operations in {:.4f}s ({:.1f} ops/s)",
                   num_operations, batch_time, num_operations / batch_time if batch_time > 0 else 0)

        # Verify all operations were executed
        assert mock_tx.run.call_count == num_operations, \
            f"Expected {num_operations} calls, got {mock_tx.run.call_count}"

        logger.success("✅ Batch operations test passed")


def test_trading_engine_ticker_caching():
    """Test trading engine ticker caching."""
    logger.info("Testing trading engine ticker caching...")

    # Mock ccxt exchange
    mock_exchange = Mock()
    mock_exchange.id = "binance"
    mock_ticker = {
        "symbol": "BTC/USDT",
        "last": 50000.0,
        "bid": 49999.0,
        "ask": 50001.0
    }
    mock_exchange.fetch_ticker.return_value = mock_ticker

    # Create trading engine with mocks
    engine = TradingEngine(
        trading_config={},
        exchanges_config={},
        knowledge_graph=None,
        agent_orchestrator=None,
        alert_manager=None
    )

    engine.exchanges = {"binance": mock_exchange}

    # First call - cache miss
    start = time.time()
    ticker1 = engine._get_ticker(mock_exchange, "BTC/USDT")
    first_call_time = time.time() - start

    # Second call within cache TTL - cache hit
    start = time.time()
    ticker2 = engine._get_ticker(mock_exchange, "BTC/USDT")
    second_call_time = time.time() - start

    logger.info("First call: {:.4f}s, Cached call: {:.4f}s, Speedup: {:.1f}x",
               first_call_time, second_call_time, first_call_time / second_call_time if second_call_time > 0 else float('inf'))

    # Verify only one API call was made (cached on second call)
    assert mock_exchange.fetch_ticker.call_count == 1, \
        f"Expected 1 API call (cached), got {mock_exchange.fetch_ticker.call_count}"

    # Verify ticker data
    assert ticker1["last"] == 50000.0
    assert ticker2["last"] == 50000.0

    logger.success("✅ Ticker caching test passed")


def test_parallel_ticker_fetching():
    """Test parallel ticker fetching performance."""
    logger.info("Testing parallel ticker fetching...")

    # Mock ccxt exchange
    mock_exchange = Mock()
    mock_exchange.id = "binance"

    def mock_fetch_ticker(symbol):
        """Simulate network delay."""
        time.sleep(0.1)  # Simulate 100ms network delay
        return {
            "symbol": symbol,
            "last": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0
        }

    mock_exchange.fetch_ticker.side_effect = mock_fetch_ticker

    # Create trading engine
    engine = TradingEngine(
        trading_config={},
        exchanges_config={},
        knowledge_graph=None,
        agent_orchestrator=None,
        alert_manager=None
    )

    engine.exchanges = {"binance": mock_exchange}

    # Test parallel fetching
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "BCH/USDT"]

    start = time.time()
    tickers = engine.fetch_tickers_parallel(symbols, "binance")
    parallel_time = time.time() - start

    # Sequential time would be ~0.5s (5 symbols * 0.1s each)
    # Parallel should be much closer to 0.1s (concurrent)
    sequential_estimate = len(symbols) * 0.1

    logger.info("Fetched {} symbols in {:.4f}s (sequential estimate: {:.4f}s), Speedup: {:.1f}x",
               len(symbols), parallel_time, sequential_estimate, sequential_estimate / parallel_time if parallel_time > 0 else 0)

    # Verify all symbols were fetched
    assert len(tickers) == len(symbols), f"Expected {len(symbols)} tickers, got {len(tickers)}"

    # Verify parallel was faster than sequential
    # Allow some margin for thread overhead, but should still be significantly faster
    assert parallel_time < sequential_estimate * 0.7, \
        f"Parallel execution should be faster: {parallel_time:.4f}s vs sequential ~{sequential_estimate:.4f}s"

    logger.success("✅ Parallel ticker fetching test passed")


def test_connection_pooling():
    """Test Neo4j connection pooling configuration."""
    logger.info("Testing connection pooling configuration...")

    config = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "test",
        "max_connection_pool_size": 50,
        "max_connection_lifetime": 3600,
        "connection_acquisition_timeout": 60
    }

    with patch('graphwiz_trader.graph.neo4j_graph.GraphDatabase') as mock_driver:
        mock_instance = Mock()
        mock_driver.driver.return_value = mock_instance

        kg = KnowledgeGraph(config)

        # Connect and verify pool configuration
        kg.connect()

        # Verify driver was called with correct pool parameters
        assert mock_driver.driver.called, "Driver should be created"

        call_kwargs = mock_driver.driver.call_args[1]
        assert call_kwargs["max_connection_pool_size"] == 50
        assert call_kwargs["max_connection_lifetime"] == 3600
        assert call_kwargs["connection_acquisition_timeout"] == 60

        logger.info("✅ Connection pool configured: size={}, lifetime={}s",
                   config["max_connection_pool_size"],
                   config["max_connection_lifetime"])

        logger.success("✅ Connection pooling test passed")


def run_all_tests():
    """Run all performance optimization tests."""
    logger.info("=" * 60)
    logger.info("Running Performance Optimization Tests")
    logger.info("=" * 60)

    tests = [
        ("Neo4j Query Caching", test_neo4j_caching),
        ("Neo4j Batch Operations", test_neo4j_batch_operations),
        ("Trading Engine Ticker Caching", test_trading_engine_ticker_caching),
        ("Parallel Ticker Fetching", test_parallel_ticker_fetching),
        ("Connection Pooling", test_connection_pooling),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running: {name}")
            logger.info('=' * 60)
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test '{name}' failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed out of {passed + failed} total")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
