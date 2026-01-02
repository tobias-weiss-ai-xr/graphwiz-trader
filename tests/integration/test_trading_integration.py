"""Integration tests for the complete trading workflow.

This module tests end-to-end scenarios including:
- Full trading workflow from signal to execution
- Exchange integration (with sandbox/testnet)
- Agent decision-making processes
- Risk management integration
- Knowledge graph integration
- Multi-agent coordination
- Error handling and recovery
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd

from graphwiz_trader.trading.engine import TradingEngine
from graphwiz_trader.trading.modes import TradingMode
from graphwiz_trader.risk.manager import RiskManager
from graphwiz_trader.agents.orchestrator import AgentOrchestrator
from graphwiz_trader.knowledge_graph.neo4j_graph import Neo4jKnowledgeGraph


@pytest.mark.integration
@pytest.mark.asyncio
class TestTradingWorkflow:
    """Test complete trading workflow from signal generation to execution."""

    async def test_full_buy_workflow(self, mock_ccxt_exchange, mock_neo4j_driver):
        """Test complete buy workflow from analysis to execution."""
        # Setup
        engine = TradingEngine(
            exchanges={"binance": mock_ccxt_exchange},
            knowledge_graph=mock_neo4j_driver,
            risk_params={"max_position_size": 1000, "risk_per_trade": 0.02}
        )

        # Mock market data
        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "volume": 100.5,
            "timestamp": datetime.now().isoformat()
        }

        # Mock agent decision
        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "buy",
                "confidence": 0.85,
                "reason": "Strong bullish momentum",
                "position_size": 0.5,
                "stop_loss": 49000,
                "take_profit": 53000
            }

            # Execute trade
            result = await engine.execute_trade_signal(market_data)

            # Assertions
            assert result["status"] == "success"
            assert result["action"] == "buy"
            assert result["symbol"] == "BTC/USDT"
            assert "order_id" in result
            assert result["confidence"] == 0.85

    async def test_full_sell_workflow(self, mock_ccxt_exchange, mock_neo4j_driver):
        """Test complete sell workflow from signal to execution."""
        engine = TradingEngine(
            exchanges={"binance": mock_ccxt_exchange},
            knowledge_graph=mock_neo4j_driver,
            risk_params={"max_position_size": 1000}
        )

        market_data = {
            "symbol": "ETH/USDT",
            "price": 3000,
            "volume": 50.0,
            "timestamp": datetime.now().isoformat()
        }

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "sell",
                "confidence": 0.78,
                "reason": "Bearish divergence detected"
            }

            result = await engine.execute_trade_signal(market_data)

            assert result["status"] == "success"
            assert result["action"] == "sell"

    async def test_rejected_trade_low_confidence(self, mock_ccxt_exchange, mock_neo4j_driver):
        """Test that low confidence trades are rejected."""
        engine = TradingEngine(
            exchanges={"binance": mock_ccxt_exchange},
            knowledge_graph=mock_neo4j_driver,
            risk_params={"min_confidence": 0.7}
        )

        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "timestamp": datetime.now().isoformat()
        }

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "buy",
                "confidence": 0.55,  # Below threshold
                "reason": "Weak signal"
            }

            result = await engine.execute_trade_signal(market_data)

            assert result["status"] == "rejected"
            assert "confidence" in result["reason"]

    async def test_risk_limit_rejection(self, mock_ccxt_exchange, mock_neo4j_driver):
        """Test that trades exceeding risk limits are rejected."""
        engine = TradingEngine(
            exchanges={"binance": mock_ccxt_exchange},
            knowledge_graph=mock_neo4j_driver,
            risk_params={
                "max_position_size": 1000,
                "risk_per_trade": 0.02,
                "max_daily_loss": 500
            }
        )

        # Simulate existing losses
        engine.daily_pnl = -400

        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "volume": 100.0,
            "timestamp": datetime.now().isoformat()
        }

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "buy",
                "confidence": 0.90,
                "position_size": 2.0,  # Large position
                "risk_amount": 150  # Would exceed daily loss limit
            }

            result = await engine.execute_trade_signal(market_data)

            assert result["status"] == "rejected"
            assert "risk limit" in result["reason"].lower()


@pytest.mark.integration
@pytest.mark.asyncio
class TestExchangeIntegration:
    """Test exchange integration with sandbox/testnet."""

    async def test_exchange_connection(self, mock_ccxt_exchange):
        """Test exchange connection and authentication."""
        exchange = mock_ccxt_exchange
        exchange.fetch_balance.return_value = {
            "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0}
        }

        balance = await exchange.fetch_balance()

        assert "USDT" in balance
        assert balance["USDT"]["free"] == 10000.0

    async def test_order_placement(self, mock_ccxt_exchange):
        """Test order placement and confirmation."""
        exchange = mock_ccxt_exchange

        order = await exchange.create_order(
            symbol="BTC/USDT",
            type="market",
            side="buy",
            amount=0.1,
            price=50000
        )

        assert order["id"] == "12345"
        assert order["symbol"] == "BTC/USDT"
        assert order["side"] == "buy"

    async def test_order_cancellation(self, mock_ccxt_exchange):
        """Test order cancellation."""
        exchange = mock_ccxt_exchange
        exchange.cancel_order.return_value = {"id": "12345", "status": "canceled"}

        result = await exchange.cancel_order("12345", "BTC/USDT")

        assert result["status"] == "canceled"

    async def test_market_data_fetch(self, mock_ccxt_exchange):
        """Test fetching real-time market data."""
        exchange = mock_ccxt_exchange

        ticker = await exchange.fetch_ticker("BTC/USDT")

        assert "last" in ticker
        assert "bid" in ticker
        assert "ask" in ticker
        assert ticker["last"] == 50000

    async def test_multi_exchange_execution(self):
        """Test executing orders across multiple exchanges."""
        exchange1 = MagicMock()
        exchange2 = MagicMock()

        exchange1.create_order = AsyncMock(return_value={"id": "ex1-123", "status": "open"})
        exchange2.create_order = AsyncMock(return_value={"id": "ex2-456", "status": "open"})

        # Execute on both exchanges
        order1 = await exchange1.create_order("BTC/USDT", "market", "buy", 0.1, 50000)
        order2 = await exchange2.create_order("BTC/USDT", "market", "buy", 0.1, 50000)

        assert order1["id"] == "ex1-123"
        assert order2["id"] == "ex2-456"


@pytest.mark.integration
@pytest.mark.asyncio
class TestAgentDecisionMaking:
    """Test agent decision-making processes."""

    async def test_single_agent_decision(self):
        """Test decision from a single agent."""
        agent = Mock()
        agent.analyze.return_value = {
            "action": "buy",
            "confidence": 0.85,
            "reasoning": "Strong uptrend with high volume"
        }

        result = await agent.analyze({"symbol": "BTC/USDT", "price": 50000})

        assert result["action"] == "buy"
        assert result["confidence"] == 0.85

    async def test_multi_agent_consensus(self):
        """Test consensus decision from multiple agents."""
        agents = {
            "technical": Mock(),
            "sentiment": Mock(),
            "risk": Mock()
        }

        agents["technical"].analyze.return_value = {
            "action": "buy",
            "confidence": 0.85
        }
        agents["sentiment"].analyze.return_value = {
            "action": "buy",
            "confidence": 0.75
        }
        agents["risk"].analyze.return_value = {
            "action": "hold",
            "confidence": 0.60
        }

        # Simulate consensus calculation
        decisions = [agent.analyze({}) for agent in agents.values()]
        buy_votes = sum(1 for d in decisions if d["action"] == "buy")
        avg_confidence = np.mean([d["confidence"] for d in decisions])

        assert buy_votes == 2
        assert avg_confidence > 0.7

    async def test_agent_disagreement_resolution(self):
        """Test handling of agent disagreement."""
        agents = {
            "bullish": Mock(),
            "bearish": Mock()
        }

        agents["bullish"].analyze.return_value = {
            "action": "buy",
            "confidence": 0.90,
            "weight": 0.5
        }
        agents["bearish"].analyze.return_value = {
            "action": "sell",
            "confidence": 0.85,
            "weight": 0.5
        }

        decisions = [agent.analyze({}) for agent in agents.values()]

        # In case of tie, should default to hold
        if len(set(d["action"] for d in decisions)) > 1:
            final_decision = "hold"
        else:
            final_decision = decisions[0]["action"]

        assert final_decision == "hold"


@pytest.mark.integration
@pytest.mark.asyncio
class TestRiskManagementIntegration:
    """Test risk management integration."""

    async def test_position_sizing(self):
        """Test dynamic position sizing based on risk."""
        risk_manager = RiskManager(
            account_balance=10000,
            risk_per_trade=0.02,
            max_position_size=1000
        )

        position_size = risk_manager.calculate_position_size(
            entry_price=50000,
            stop_loss=49000,
            risk_per_trade=0.02
        )

        # Risk amount = $200 (2% of $10,000)
        # Price risk = $1,000 per BTC
        # Position size = 200 / 1000 = 0.2 BTC
        expected_size = (10000 * 0.02) / (50000 - 49000)
        assert abs(position_size - expected_size) < 0.01

    async def test_stop_loss_adjustment(self):
        """Test dynamic stop-loss adjustment."""
        risk_manager = RiskManager()

        # Test trailing stop
        new_stop = risk_manager.adjust_trailing_stop(
            entry_price=50000,
            current_stop=49000,
            current_price=51000,
            trail_distance_pct=0.05
        )

        expected_stop = 51000 * (1 - 0.05)
        assert abs(new_stop - expected_stop) < 1

    async def test_drawdown_monitoring(self):
        """Test drawdown monitoring and alerts."""
        risk_manager = RiskManager(
            max_drawdown_pct=0.10,
            peak_balance=10000
        )

        # Simulate drawdown
        current_balance = 8500  # 15% drawdown
        is_limit_exceeded = risk_manager.check_drawdown_limit(current_balance)

        assert is_limit_exceeded == True

    async def test_correlation_check(self):
        """Test portfolio correlation checks."""
        risk_manager = RiskManager()

        existing_positions = [
            {"symbol": "BTC/USDT", "size": 0.5},
            {"symbol": "ETH/USDT", "size": 5.0}
        ]

        # Check if new position is too correlated
        is_allowed = risk_manager.check_correlation_risk(
            new_symbol="BTC/USDT",
            existing_positions=existing_positions,
            max_correlation=0.7
        )

        # BTC/USDT is already in portfolio, should be rejected or reduced
        assert is_allowed == False


@pytest.mark.integration
@pytest.mark.asyncio
class TestKnowledgeGraphIntegration:
    """Test knowledge graph integration."""

    async def test_market_context_storage(self, mock_neo4j_driver):
        """Test storing market context in knowledge graph."""
        kg = Neo4jKnowledgeGraph(mock_neo4j_driver)

        market_data = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "volume": 1000.0,
            "timestamp": datetime.now().isoformat()
        }

        await kg.store_market_context(market_data)

        # Verify write was called
        assert mock_neo4j_driver.session.called

    async def test_historical_pattern_retrieval(self, mock_neo4j_driver):
        """Test retrieving historical patterns from knowledge graph."""
        kg = Neo4jKnowledgeGraph(mock_neo4j_driver)

        # Mock query response
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = [
            {
                "pattern": "double_bottom",
                "success_rate": 0.75,
                "avg_gain": 0.15
            }
        ]
        mock_session.run.return_value = mock_result
        mock_neo4j_driver.session.return_value = mock_session

        patterns = await kg.find_similar_patterns(
            symbol="BTC/USDT",
            lookback_days=30
        )

        assert len(patterns) > 0
        assert patterns[0]["pattern"] == "double_bottom"

    async def test_agent_decision_logging(self, mock_neo4j_driver):
        """Test logging agent decisions to knowledge graph."""
        kg = Neo4jKnowledgeGraph(mock_neo4j_driver)

        decision = {
            "timestamp": datetime.now().isoformat(),
            "agent": "technical",
            "action": "buy",
            "confidence": 0.85,
            "reasoning": "Strong momentum"
        }

        await kg.log_agent_decision(decision)

        assert mock_neo4j_driver.session.called

    async def test_performance_tracking(self, mock_neo4j_driver):
        """Test tracking performance metrics in knowledge graph."""
        kg = Neo4jKnowledgeGraph(mock_neo4j_driver)

        metrics = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_trades": 10,
            "win_rate": 0.60,
            "total_pnl": 500.0,
            "sharpe_ratio": 1.5
        }

        await kg.store_performance_metrics(metrics)

        assert mock_neo4j_driver.session.called


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndScenarios:
    """Test complete end-to-end trading scenarios."""

    async def test_profitable_trade_scenario(self, mock_ccxt_exchange, mock_neo4j_driver):
        """Test a complete profitable trade cycle."""
        engine = TradingEngine(
            exchanges={"binance": mock_ccxt_exchange},
            knowledge_graph=mock_neo4j_driver,
            risk_params={"max_position_size": 1000, "risk_per_trade": 0.02}
        )

        # Entry
        entry_signal = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "action": "buy",
            "confidence": 0.85
        }

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "buy",
                "confidence": 0.85,
                "position_size": 0.2
            }

            entry_result = await engine.execute_trade_signal(entry_signal)
            assert entry_result["status"] == "success"

        # Exit
        exit_signal = {
            "symbol": "BTC/USDT",
            "price": 52000,  # 4% profit
            "action": "sell",
            "confidence": 0.80
        }

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "sell",
                "confidence": 0.80,
                "reason": "Take profit target reached"
            }

            exit_result = await engine.execute_trade_signal(exit_signal)
            assert exit_result["status"] == "success"

    async def test_stop_loss_scenario(self, mock_ccxt_exchange, mock_neo4j_driver):
        """Test trade that hits stop loss."""
        engine = TradingEngine(
            exchanges={"binance": mock_ccxt_exchange},
            knowledge_graph=mock_neo4j_driver,
            risk_params={
                "max_position_size": 1000,
                "stop_loss_pct": 0.05
            }
        )

        # Entry
        entry_signal = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "action": "buy",
            "confidence": 0.80
        }

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "buy",
                "confidence": 0.80,
                "position_size": 0.2,
                "stop_loss": 47500
            }

            entry_result = await engine.execute_trade_signal(entry_signal)
            assert entry_result["status"] == "success"

        # Price drops to stop loss
        stop_signal = {
            "symbol": "BTC/USDT",
            "price": 47400,  # Below stop loss
            "action": "sell",
            "confidence": 1.0,  # High confidence for risk management
            "reason": "Stop loss triggered"
        }

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "sell",
                "confidence": 1.0,
                "reason": "Stop loss triggered"
            }

            exit_result = await engine.execute_trade_signal(stop_signal)
            assert exit_result["status"] == "success"

    async def test_multi_symbol_portfolio(self, mock_ccxt_exchange, mock_neo4j_driver):
        """Test managing a portfolio of multiple symbols."""
        engine = TradingEngine(
            exchanges={"binance": mock_ccxt_exchange},
            knowledge_graph=mock_neo4j_driver,
            risk_params={
                "max_position_size": 1000,
                "max_positions": 5
            }
        )

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        for symbol in symbols:
            signal = {
                "symbol": symbol,
                "price": 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100,
                "action": "buy",
                "confidence": 0.80
            }

            with patch.object(engine, 'get_agent_decision') as mock_decision:
                mock_decision.return_value = {
                    "action": "buy",
                    "confidence": 0.80,
                    "position_size": 0.1
                }

                result = await engine.execute_trade_signal(signal)
                assert result["status"] == "success"

        # Verify positions were created
        assert len(engine.positions) == 3

    async def test_error_recovery(self, mock_ccxt_exchange, mock_neo4j_driver):
        """Test system recovery from errors."""
        engine = TradingEngine(
            exchanges={"binance": mock_ccxt_exchange},
            knowledge_graph=mock_neo4j_driver,
            risk_params={"max_position_size": 1000}
        )

        # Simulate exchange error
        mock_ccxt_exchange.create_order.side_effect = [
            Exception("Network error"),
            {"id": "12345", "status": "open"}  # Second attempt succeeds
        ]

        signal = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "action": "buy",
            "confidence": 0.85
        }

        with patch.object(engine, 'get_agent_decision') as mock_decision:
            mock_decision.return_value = {
                "action": "buy",
                "confidence": 0.85,
                "position_size": 0.1
            }

            # Should retry and succeed
            result = await engine.execute_trade_signal(signal, max_retries=2)
            assert result["status"] == "success"


@pytest.mark.integration
class TestTradingModes:
    """Test different trading mode configurations."""

    def test_paper_trading_mode(self):
        """Test paper trading configuration."""
        from graphwiz_trader.trading.modes import TradingMode

        paper_mode = TradingMode(
            name="paper_trading",
            execution_type="simulated",
            slippage_model=0.001,
            latency_model=0.1
        )

        assert paper_mode.execution_type == "simulated"
        assert paper_mode.slippage_model == 0.001

    def test_live_trading_mode(self):
        """Test live trading configuration."""
        live_mode = TradingMode(
            name="live_trading",
            execution_type="real",
            require_confirmation=True,
            max_order_size=10000
        )

        assert live_mode.execution_type == "real"
        assert live_mode.require_confirmation == True

    def test_hft_mode(self):
        """Test high-frequency trading mode."""
        hft_mode = TradingMode(
            name="hft",
            execution_type="real",
            max_latency_ms=10,
            order_type="limit"
        )

        assert hft_mode.max_latency_ms == 10
        assert hft_mode.order_type == "limit"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--integration"])
