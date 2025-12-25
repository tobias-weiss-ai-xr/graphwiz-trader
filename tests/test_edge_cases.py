"""Edge case tests for graphwiz-trader components."""

import pytest
from unittest.mock import MagicMock, Mock, patch


class TestKnowledgeGraphEdgeCases:
    """Edge case tests for KnowledgeGraph."""

    # Note: Complex context manager mocking tests skipped
    # The main test suite (test_neo4j_graph.py) already covers these scenarios
    # using the proper fixtures


class TestTradingEngineEdgeCases:
    """Edge case tests for TradingEngine."""

    def test_empty_exchanges_config(self, mock_kg, mock_agent_orchestrator):
        """Test trading engine with no exchanges configured."""
        from graphwiz_trader.trading import TradingEngine

        engine = TradingEngine({}, {}, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()

        assert len(engine.exchanges) == 0

    def test_all_exchanges_disabled(self, trading_config, mock_kg, mock_agent_orchestrator):
        """Test when all exchanges are disabled."""
        from graphwiz_trader.trading import TradingEngine

        exchanges_config = {
            "binance": {"enabled": False},
            "kraken": {"enabled": False}
        }

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()

        assert len(engine.exchanges) == 0

    def test_execute_trade_with_zero_amount(self, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test trade execution with zero amount."""
        from graphwiz_trader.trading import TradingEngine

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)

        result = engine.execute_trade("BTC/USDT", "buy", 0)

        assert result["status"] == "error"
        assert "Invalid amount" in result["message"]

    def test_execute_trade_with_negative_amount(self, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test trade execution with negative amount."""
        from graphwiz_trader.trading import TradingEngine

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)

        result = engine.execute_trade("BTC/USDT", "buy", -1.5)

        assert result["status"] == "error"
        assert "Invalid amount" in result["message"]

    def test_execute_trade_invalid_side(self, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test trade execution with invalid side."""
        from graphwiz_trader.trading import TradingEngine

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)

        result = engine.execute_trade("BTC/USDT", "invalid_side", 1.0)

        assert result["status"] == "error"
        assert "Invalid side" in result["message"]


class TestAgentOrchestratorEdgeCases:
    """Edge case tests for AgentOrchestrator."""

    def test_empty_config(self, mock_kg):
        """Test orchestrator with empty configuration."""
        from graphwiz_trader.agents import AgentOrchestrator

        orchestrator = AgentOrchestrator({}, mock_kg)

        assert len(orchestrator.agents) == 0

    def test_config_with_missing_enabled_field(self, mock_kg):
        """Test agent config without enabled field."""
        from graphwiz_trader.agents import AgentOrchestrator

        config = {
            "agent1": {"model": "gpt-4"},
            "agent2": {}
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        # Agents should not be initialized when enabled field is missing
        assert len(orchestrator.agents) == 0

    def test_get_decision_with_complex_context(self, agents_config, mock_kg):
        """Test get_decision with complex trading context."""
        from graphwiz_trader.agents import AgentOrchestrator

        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        complex_context = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "volume": 1000000,
            "indicators": {
                "rsi": 65,
                "macd": 0.5,
                "ema": [49500, 49800, 50000]
            },
            "market_sentiment": "bullish",
            "news": [
                {"title": "BTC hits new high", "sentiment": "positive"},
                {"title": "Regulatory concerns", "sentiment": "negative"}
            ],
            "time": "2024-01-01T12:00:00Z"
        }

        decision = orchestrator.get_decision(complex_context)

        assert isinstance(decision, dict)
        assert "action" in decision
        assert "confidence" in decision

    def test_agent_config_with_extra_fields(self, mock_kg):
        """Test agent config with additional custom fields."""
        from graphwiz_trader.agents import AgentOrchestrator

        config = {
            "custom_agent": {
                "enabled": True,
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "system_prompt": "You are a trading assistant",
                "custom_field": "custom_value"
            }
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        assert "custom_agent" in orchestrator.agents
        # Agents are objects now, check config attribute
        assert orchestrator.agents["custom_agent"].config["custom_field"] == "custom_value"


class TestConfigEdgeCases:
    """Edge case tests for configuration loading."""

    def test_load_malformed_yaml(self):
        """Test loading malformed YAML."""
        from graphwiz_trader.utils.config import load_config
        import tempfile
        from pathlib import Path

        malformed_yaml = """
key: value
  indented_wrong: true
- list_item
  another_item:
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(malformed_yaml)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            # Should return empty dict on parse error
            assert config == {}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_config_with_unicode(self):
        """Test loading config with unicode characters."""
        from graphwiz_trader.utils.config import load_config
        import tempfile
        from pathlib import Path

        yaml_with_unicode = """
app_name: "交易系统"
description: "Automated trading system with knowledge graphs™"
symbols:
  - "BTC/比特币"
  - "ETH/以太坊"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_with_unicode)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["app_name"] == "交易系统"
            assert "™" in config["description"]
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_very_large_config(self):
        """Test loading a very large configuration file."""
        from graphwiz_trader.utils.config import load_config
        import tempfile
        from pathlib import Path

        # Generate a large config
        large_yaml = "trading:\n"
        for i in range(1000):
            large_yaml += f"  param_{i}: {i}\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(large_yaml)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert len(config["trading"]) == 1000
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestGraphWizTraderIntegrationEdgeCases:
    """Integration edge case tests."""

    def test_multiple_start_stop_cycles(self, temp_config_file):
        """Test multiple start/stop cycles."""
        from graphwiz_trader.main import GraphWizTrader

        with patch('graphwiz_trader.main.KnowledgeGraph'):
            with patch('graphwiz_trader.main.AgentOrchestrator'):
                with patch('graphwiz_trader.main.TradingEngine') as mock_engine:
                    mock_engine_instance = MagicMock()
                    mock_engine.return_value = mock_engine_instance

                    trader = GraphWizTrader(temp_config_file)

                    # Multiple start/stop cycles
                    for i in range(3):
                        trader.start()
                        assert trader.is_running()
                        trader.stop()
                        assert not trader.is_running()

                    assert mock_engine_instance.start.call_count == 3
                    assert mock_engine_instance.stop.call_count == 3

    def test_start_without_stop(self, temp_config_file):
        """Test starting system without explicitly stopping."""
        from graphwiz_trader.main import GraphWizTrader

        with patch('graphwiz_trader.main.KnowledgeGraph') as mock_kg:
            with patch('graphwiz_trader.main.AgentOrchestrator'):
                with patch('graphwiz_trader.main.TradingEngine'):
                    mock_kg_instance = MagicMock()
                    mock_kg.return_value = mock_kg_instance

                    trader = GraphWizTrader(temp_config_file)
                    trader.start()

                    # Let it go out of scope without explicit stop
                    # Should handle cleanup gracefully
                    assert trader._running
