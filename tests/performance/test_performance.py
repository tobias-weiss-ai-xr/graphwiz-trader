"""Performance tests and benchmarks for graphwiz-trader.

This module tests performance characteristics including:
- Order execution latency
- Agent response time
- Knowledge graph query performance
- Backtesting speed
- Resource usage benchmarks
"""

import pytest
import asyncio
import time
import psutil
import os
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import after checking availability
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


@pytest.mark.performance
class TestOrderExecutionPerformance:
    """Test order execution performance metrics."""

    @pytest.mark.asyncio
    async def test_order_execution_latency(self):
        """Test order execution latency requirements."""
        from graphwiz_trader.trading.engine import TradingEngine

        exchange = MagicMock()
        exchange.create_order = AsyncMock(return_value={"id": "test-order", "status": "open"})

        engine = TradingEngine(
            exchanges={"binance": exchange},
            knowledge_graph=MagicMock(),
            risk_params={"max_position_size": 1000}
        )

        # Measure execution time
        start_time = time.perf_counter()

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "buy",
                "confidence": 0.85,
                "position_size": 0.1
            }

            await engine.execute_trade_signal({
                "symbol": "BTC/USDT",
                "price": 50000
            })

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assert execution time is under 500ms for normal trading
        assert execution_time < 500, f"Execution time {execution_time}ms exceeds 500ms threshold"

    @pytest.mark.asyncio
    async def test_high_frequency_order_latency(self):
        """Test latency for high-frequency trading scenarios."""
        exchange = MagicMock()
        exchange.create_order = AsyncMock(return_value={"id": "hft-order", "status": "open"})

        engine = TradingEngine(
            exchanges={"binance": exchange},
            knowledge_graph=MagicMock(),
            risk_params={"max_position_size": 1000}
        )

        latencies = []

        for i in range(10):
            start_time = time.perf_counter()

            with patch.object(engine, 'get_agent_decision') as mock_decision:
                mock_decision.return_value = {"action": "buy", "confidence": 0.80}

                await engine.execute_trade_signal({
                    "symbol": "BTC/USDT",
                    "price": 50000 + i
                })

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        # HFT requirements
        assert avg_latency < 100, f"Average latency {avg_latency}ms exceeds 100ms for HFT"
        assert p95_latency < 200, f"P95 latency {p95_latency}ms exceeds 200ms for HFT"

    @pytest.mark.asyncio
    async def test_concurrent_order_throughput(self):
        """Test throughput of concurrent orders."""
        num_orders = 20

        exchange = MagicMock()
        exchange.create_order = AsyncMock(return_value={"id": "concurrent-order", "status": "open"})

        engine = TradingEngine(
            exchanges={"binance": exchange},
            knowledge_graph=MagicMock(),
            risk_params={"max_position_size": 1000}
        )

        async def execute_order(i):
            with patch.object(engine, 'get_agent_decision') as mock_decision:
                mock_decision.return_value = {"action": "buy", "confidence": 0.80}
                return await engine.execute_trade_signal({
                    "symbol": "BTC/USDT",
                    "price": 50000 + i
                })

        start_time = time.perf_counter()
        results = await asyncio.gather(*[execute_order(i) for i in range(num_orders)])
        end_time = time.perf_counter()

        total_time = end_time - start_time
        throughput = num_orders / total_time

        # Should handle at least 10 orders per second
        assert throughput >= 10, f"Throughput {throughput} orders/s below 10 orders/s requirement"

        # All orders should succeed
        assert all(r["status"] == "success" for r in results)


@pytest.mark.performance
class TestAgentPerformance:
    """Test AI agent performance metrics."""

    @pytest.mark.asyncio
    async def test_agent_response_time(self):
        """Test agent decision-making response time."""
        from graphwiz_trader.agents.technical_agent import TechnicalAgent

        agent = TechnicalAgent(model="gpt-4", temperature=0.7)

        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "volume": 1000.0,
            "indicators": {
                "rsi": 65,
                "macd": 0.5,
                "sma_20": 49000,
                "sma_50": 48000
            }
        }

        start_time = time.perf_counter()

        # Mock the LLM call for testing
        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = '{"action": "buy", "confidence": 0.75, "reason": "Trend following"}'
            decision = await agent.analyze(market_data)

        end_time = time.perf_counter()
        response_time = (end_time - start_time) * 1000  # Convert to ms

        # Agent should respond within 2 seconds
        assert response_time < 2000, f"Agent response time {response_time}ms exceeds 2000ms"

    @pytest.mark.asyncio
    async def test_multi_agent_orchestration_time(self):
        """Test multi-agent consensus performance."""
        from graphwiz_trader.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Mock agents
        mock_agents = {
            "technical": MagicMock(analyze=AsyncMock(return_value={
                "action": "buy", "confidence": 0.85, "weight": 0.4
            })),
            "sentiment": MagicMock(analyze=AsyncMock(return_value={
                "action": "buy", "confidence": 0.75, "weight": 0.3
            })),
            "risk": MagicMock(analyze=AsyncMock(return_value={
                "action": "hold", "confidence": 0.60, "weight": 0.3
            }))
        }

        orchestrator.agents = mock_agents

        start_time = time.perf_counter()
        decision = await orchestrator.get_consensus({"symbol": "BTC/USDT"})
        end_time = time.perf_counter()

        orchestration_time = (end_time - start_time) * 1000

        # Multi-agent decision should complete within 5 seconds
        assert orchestration_time < 5000, f"Orchestration time {orchestration_time}ms exceeds 5000ms"

    @pytest.mark.asyncio
    async def test_agent_memory_usage(self):
        """Test agent memory consumption."""
        if not MEMORY_PROFILER_AVAILABLE:
            pytest.skip("memory_profiler not available")

        from graphwiz_trader.agents.technical_agent import TechnicalAgent

        agent = TechnicalAgent(model="gpt-4")

        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Perform analysis
        with patch.object(agent, '_call_llm') as mock_llm:
            mock_llm.return_value = '{"action": "buy", "confidence": 0.75}'
            await agent.analyze({"symbol": "BTC/USDT", "price": 50000})

        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Agent should use less than 100MB
        assert memory_used < 100, f"Agent memory usage {memory_used}MB exceeds 100MB"


@pytest.mark.performance
class TestKnowledgeGraphPerformance:
    """Test knowledge graph query performance."""

    @pytest.mark.asyncio
    async def test_simple_query_performance(self, mock_neo4j_driver):
        """Test simple query performance."""
        from graphwiz_trader.knowledge_graph.neo4j_graph import Neo4jKnowledgeGraph

        kg = Neo4jKnowledgeGraph(mock_neo4j_driver)

        # Mock query response
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = [{"price": 50000}]
        mock_session.run.return_value = mock_result
        mock_neo4j_driver.session.return_value = mock_session

        start_time = time.perf_counter()
        result = await kg.query("MATCH (n:Market) RETURN n LIMIT 10")
        end_time = time.perf_counter()

        query_time = (end_time - start_time) * 1000

        # Simple queries should complete in under 100ms
        assert query_time < 100, f"Query time {query_time}ms exceeds 100ms"

    @pytest.mark.asyncio
    async def test_complex_query_performance(self, mock_neo4j_driver):
        """Test complex analytical query performance."""
        kg = Neo4jKnowledgeGraph(mock_neo4j_driver)

        # Mock complex query response
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = [
            {"pattern": "double_top", "count": 5}
            for _ in range(100)
        ]
        mock_session.run.return_value = mock_result
        mock_neo4j_driver.session.return_value = mock_session

        start_time = time.perf_counter()
        result = await kg.query("""
            MATCH (m:Market)-[:HAS_PATTERN]->(p:Pattern)
            WHERE m.timestamp > datetime() - duration('P30D')
            RETURN p.type, count(*) as count
            ORDER BY count DESC
        """)
        end_time = time.perf_counter()

        query_time = (end_time - start_time) * 1000

        # Complex queries should complete in under 1 second
        assert query_time < 1000, f"Complex query time {query_time}ms exceeds 1000ms"

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, mock_neo4j_driver):
        """Test bulk data insertion performance."""
        kg = Neo4jKnowledgeGraph(mock_neo4j_driver)

        num_records = 1000
        records = [
            {
                "timestamp": datetime.now().isoformat(),
                "price": 50000 + i,
                "volume": 1000 + i
            }
            for i in range(num_records)
        ]

        start_time = time.perf_counter()
        await kg.bulk_insert(records)
        end_time = time.perf_counter()

        insert_time = end_time - start_time
        throughput = num_records / insert_time

        # Should handle at least 100 records per second
        assert throughput >= 100, f"Bulk insert throughput {throughput} records/s below 100 records/s"


@pytest.mark.performance
class TestBacktestingPerformance:
    """Test backtesting engine performance."""

    def test_backtesting_speed(self):
        """Test backtesting processing speed."""
        from graphwiz_trader.backtesting.engine import BacktestEngine

        # Generate test data
        num_candles = 10000
        dates = pd.date_range(start="2023-01-01", periods=num_candles, freq="1H")
        data = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.normal(50000, 1000, num_candles),
            "high": np.random.normal(51000, 1000, num_candles),
            "low": np.random.normal(49000, 1000, num_candles),
            "close": np.random.normal(50000, 1000, num_candles),
            "volume": np.random.normal(1000, 100, num_candles)
        })

        engine = BacktestEngine(initial_capital=100000)

        start_time = time.perf_counter()
        results = engine.run(data)
        end_time = time.perf_counter()

        processing_time = end_time - start_time
        candles_per_second = num_candles / processing_time

        # Should process at least 1000 candles per second
        assert candles_per_second >= 1000, f"Backtesting speed {candles_per_second} candles/s below 1000 candles/s"

    def test_multi_strategy_backtesting(self):
        """Test backtesting with multiple strategies."""
        from graphwiz_trader.backtesting.engine import BacktestEngine

        num_candles = 5000
        dates = pd.date_range(start="2023-01-01", periods=num_candles, freq="1H")
        data = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.normal(50000, 1000, num_candles),
            "volume": np.random.normal(1000, 100, num_candles)
        })

        engine = BacktestEngine(initial_capital=100000)

        start_time = time.perf_counter()

        # Run multiple strategies
        strategies = ["sma_cross", "mean_reversion", "momentum"]
        results = {}
        for strategy in strategies:
            results[strategy] = engine.run(data, strategy=strategy)

        end_time = time.perf_counter()

        total_time = end_time - start_time
        strategies_per_second = len(strategies) / total_time

        # Should handle multiple strategies efficiently
        assert total_time < 10, f"Multi-strategy backtesting {total_time}s exceeds 10s"

    def test_parameter_optimization_speed(self):
        """Test parameter optimization performance."""
        from graphwiz_trader.backtesting.optimizer import ParameterOptimizer

        num_candles = 2000
        dates = pd.date_range(start="2023-01-01", periods=num_candles, freq="1H")
        data = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.normal(50000, 1000, num_candles)
        })

        optimizer = ParameterOptimizer()
        param_grid = {
            "short_window": [10, 20, 30],
            "long_window": [50, 100, 200],
            "threshold": [0.01, 0.02, 0.03]
        }

        start_time = time.perf_counter()
        best_params = optimizer.optimize(data, param_grid, max_iterations=10)
        end_time = time.perf_counter()

        optimization_time = end_time - start_time

        # Should complete optimization in reasonable time
        assert optimization_time < 30, f"Optimization time {optimization_time}s exceeds 30s"


@pytest.mark.performance
class TestResourceUsage:
    """Test system resource usage benchmarks."""

    def test_memory_usage_during_trading(self):
        """Test memory usage during trading operations."""
        process = psutil.Process(os.getpid())

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Simulate trading operations
        trading_data = [
            {
                "symbol": "BTC/USDT",
                "price": 50000 + i,
                "volume": 1000 + i
            }
            for i in range(1000)
        ]

        # Process data
        processed = [dict(d) for d in trading_data]

        # Peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_used = peak_memory - baseline_memory

        # Should use less than 500MB for 1000 trades
        assert memory_used < 500, f"Memory usage {memory_used}MB exceeds 500MB for 1000 trades"

    def test_cpu_usage_during_backtesting(self):
        """Test CPU usage during backtesting."""
        from graphwiz_trader.backtesting.engine import BacktestEngine

        process = psutil.Process()

        num_candles = 5000
        dates = pd.date_range(start="2023-01-01", periods=num_candles, freq="1H")
        data = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.normal(50000, 1000, num_candles)
        })

        engine = BacktestEngine(initial_capital=100000)

        # Monitor CPU during backtesting
        cpu_before = process.cpu_percent(interval=0.1)
        start_time = time.perf_counter()

        engine.run(data)

        end_time = time.perf_counter()
        cpu_after = process.cpu_percent(interval=0.1)

        # CPU usage should be reasonable (not spike to 100%)
        assert cpu_after < 95, f"CPU usage {cpu_after}% indicates potential performance issue"

    def test_database_connection_pool_efficiency(self):
        """Test database connection pool performance."""
        from graphwiz_trader.knowledge_graph.neo4j_graph import Neo4jKnowledgeGraph

        mock_driver = MagicMock()

        # Simulate multiple concurrent queries
        kg = Neo4jKnowledgeGraph(mock_driver)

        async def run_query(i):
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.data.return_value = [{"id": i}]
            mock_session.run.return_value = mock_result
            mock_driver.session.return_value = mock_session

            return await kg.query(f"MATCH (n) RETURN n LIMIT {i}")

        # Run concurrent queries
        start_time = time.perf_counter()
        results = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*[run_query(i) for i in range(50)])
        )
        end_time = time.perf_counter()

        query_time = end_time - start_time
        avg_time_per_query = query_time / 50

        # Connection pool should efficiently handle concurrent queries
        assert avg_time_per_query < 0.1, f"Avg query time {avg_time_per_query}s indicates pool bottleneck"


@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Test scalability with increasing load."""

    @pytest.mark.parametrize("num_symbols", [1, 5, 10, 20])
    @pytest.mark.asyncio
    async def test_multi_symbol_scalability(self, num_symbols):
        """Test scalability with multiple trading symbols."""
        exchange = MagicMock()
        exchange.create_order = AsyncMock(return_value={"id": "order", "status": "open"})

        engine = TradingEngine(
            exchanges={"binance": exchange},
            knowledge_graph=MagicMock(),
            risk_params={"max_position_size": 1000}
        )

        start_time = time.perf_counter()

        for i in range(num_symbols):
            with patch.object(engine, 'get_agent_decision') as mock_decision:
                mock_decision.return_value = {"action": "buy", "confidence": 0.80}
                await engine.execute_trade_signal({
                    "symbol": f"SYMB{i}/USDT",
                    "price": 100 + i
                })

        end_time = time.perf_counter()
        time_per_symbol = (end_time - start_time) / num_symbols

        # Time per symbol should not increase significantly
        assert time_per_symbol < 0.5, f"Time per symbol {time_per_symbol}s indicates scalability issue"

    def test_historical_data_scaling(self):
        """Test performance with increasing historical data size."""
        from graphwiz_trader.backtesting.engine import BacktestEngine

        data_sizes = [1000, 5000, 10000, 50000]
        processing_times = []

        for size in data_sizes:
            dates = pd.date_range(start="2023-01-01", periods=size, freq="1H")
            data = pd.DataFrame({
                "timestamp": dates,
                "close": np.random.normal(50000, 1000, size)
            })

            engine = BacktestEngine(initial_capital=100000)

            start_time = time.perf_counter()
            engine.run(data)
            end_time = time.perf_counter()

            processing_times.append(end_time - start_time)

        # Check linear scalability (O(n))
        # Time should scale roughly linearly with data size
        ratio_5k_to_1k = processing_times[1] / processing_times[0]
        ratio_10k_to_1k = processing_times[2] / processing_times[0]
        ratio_50k_to_1k = processing_times[3] / processing_times[0]

        # Allow some overhead but check for reasonable scaling
        assert ratio_5k_to_1k < 6, "5x data should take less than 6x time"
        assert ratio_10k_to_1k < 12, "10x data should take less than 12x time"
        assert ratio_50k_to_1k < 60, "50x data should take less than 60x time"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--performance"])
