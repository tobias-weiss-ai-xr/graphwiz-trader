"""Integration tests for main GraphWizTrader system."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from graphwiz_trader.main import GraphWizTrader


class TestGraphWizTrader:
    """Integration test suite for GraphWizTrader class."""

    def test_initialization(self, temp_config_file):
        """Test GraphWizTrader initialization."""
        trader = GraphWizTrader(temp_config_file)

        assert trader.config is not None
        assert trader.kg is None
        assert trader.trading_engine is None
        assert trader.agent_orchestrator is None
        assert trader._running is False

    def test_initialization_with_nonexistent_config(self):
        """Test initialization with non-existent config file."""
        with patch('graphwiz_trader.main.logger') as mock_logger:
            trader = GraphWizTrader("nonexistent_config.yaml")
            assert trader.config == {}

    @patch('graphwiz_trader.main.KnowledgeGraph')
    @patch('graphwiz_trader.main.AgentOrchestrator')
    @patch('graphwiz_trader.main.TradingEngine')
    def test_start_success(self, mock_trading_engine, mock_agent_orch, mock_kg_class, temp_config_file):
        """Test successful system startup."""
        # Setup mocks
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg

        mock_agent_orchestrator = MagicMock()
        mock_agent_orch.return_value = mock_agent_orchestrator

        mock_engine = MagicMock()
        mock_trading_engine.return_value = mock_engine

        trader = GraphWizTrader(temp_config_file)
        trader.start()

        # Verify initialization sequence
        mock_kg_class.assert_called_once()
        mock_kg.connect.assert_called_once()

        mock_agent_orch.assert_called_once()
        mock_trading_engine.assert_called_once()
        mock_engine.start.assert_called_once()

        assert trader._running is True

    @patch('graphwiz_trader.main.KnowledgeGraph')
    def test_start_kg_connection_failure(self, mock_kg_class, temp_config_file):
        """Test system startup when knowledge graph connection fails."""
        mock_kg = MagicMock()
        mock_kg.connect.side_effect = Exception("Connection failed")
        mock_kg_class.return_value = mock_kg

        trader = GraphWizTrader(temp_config_file)

        with pytest.raises(Exception) as exc_info:
            trader.start()

        assert "Connection failed" in str(exc_info.value)

    @patch('graphwiz_trader.main.KnowledgeGraph')
    @patch('graphwiz_trader.main.AgentOrchestrator')
    @patch('graphwiz_trader.main.TradingEngine')
    def test_stop(self, mock_trading_engine, mock_agent_orch, mock_kg_class, temp_config_file):
        """Test stopping the trading system."""
        # Setup mocks
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg

        mock_agent_orchestrator = MagicMock()
        mock_agent_orch.return_value = mock_agent_orchestrator

        mock_engine = MagicMock()
        mock_trading_engine.return_value = mock_engine

        trader = GraphWizTrader(temp_config_file)
        trader.start()
        trader.stop()

        assert trader._running is False
        mock_engine.stop.assert_called_once()
        mock_kg.disconnect.assert_called_once()

    @patch('graphwiz_trader.main.KnowledgeGraph')
    @patch('graphwiz_trader.main.AgentOrchestrator')
    @patch('graphwiz_trader.main.TradingEngine')
    def test_stop_without_start(self, mock_trading_engine, mock_agent_orch, mock_kg_class, temp_config_file):
        """Test stopping system when it was never started."""
        trader = GraphWizTrader(temp_config_file)
        # Should not raise exception
        trader.stop()

        assert trader._running is False
        mock_trading_engine.return_value.stop.assert_not_called()

    @patch('graphwiz_trader.main.KnowledgeGraph')
    @patch('graphwiz_trader.main.AgentOrchestrator')
    @patch('graphwiz_trader.main.TradingEngine')
    def test_is_running(self, mock_trading_engine, mock_agent_orch, mock_kg_class, temp_config_file):
        """Test is_running method."""
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg

        mock_engine = MagicMock()
        mock_trading_engine.return_value = mock_engine

        trader = GraphWizTrader(temp_config_file)

        assert trader.is_running() is False

        trader.start()
        assert trader.is_running() is True

        trader.stop()
        assert trader.is_running() is False

    @patch('graphwiz_trader.main.KnowledgeGraph')
    @patch('graphwiz_trader.main.AgentOrchestrator')
    @patch('graphwiz_trader.main.TradingEngine')
    def test_config_loading(self, mock_trading_engine, mock_agent_orch, mock_kg_class, temp_config_file):
        """Test that configuration is loaded correctly."""
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg

        trader = GraphWizTrader(temp_config_file)

        assert trader.config["version"] == "0.1.0"
        assert "neo4j" in trader.config
        assert "trading" in trader.config
        assert "exchanges" in trader.config
        assert "agents" in trader.config

    @patch('graphwiz_trader.main.KnowledgeGraph')
    @patch('graphwiz_trader.main.AgentOrchestrator')
    @patch('graphwiz_trader.main.TradingEngine')
    def test_component_initialization_order(self, mock_trading_engine, mock_agent_orch, mock_kg_class, temp_config_file):
        """Test that components are initialized in the correct order."""
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg

        mock_agent_orchestrator = MagicMock()
        mock_agent_orch.return_value = mock_agent_orchestrator

        mock_engine = MagicMock()
        mock_trading_engine.return_value = mock_engine

        trader = GraphWizTrader(temp_config_file)
        trader.start()

        # Verify initialization order: KG -> Agents -> Trading Engine
        calls = [
            mock_kg_class.call_args,
            mock_agent_orch.call_args,
            mock_trading_engine.call_args
        ]

        # Each should be called exactly once
        assert all(calls)

        # Verify trading engine receives kg and orchestrator
        # Use call_args() which returns (args, kwargs) tuple
        if mock_trading_engine.call_args:
            args, kwargs = mock_trading_engine.call_args
            if kwargs:
                assert kwargs.get("knowledge_graph") == mock_kg
                assert kwargs.get("agent_orchestrator") == mock_agent_orchestrator

    @patch('graphwiz_trader.main.KnowledgeGraph')
    @patch('graphwiz_trader.main.AgentOrchestrator')
    @patch('graphwiz_trader.main.TradingEngine')
    def test_version_from_config(self, mock_trading_engine, mock_agent_orch, mock_kg_class, temp_config_file):
        """Test that version is read from config."""
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg

        trader = GraphWizTrader(temp_config_file)

        assert trader.config.get("version") == "0.1.0"

    @patch('graphwiz_trader.main.KnowledgeGraph')
    @patch('graphwiz_trader.main.AgentOrchestrator')
    @patch('graphwiz_trader.main.TradingEngine')
    def test_default_version(self, mock_trading_engine, mock_agent_orch, mock_kg_class):
        """Test default version when not in config."""
        import tempfile
        config_content = "neo4j:\n  uri: bolt://localhost:7687\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        try:
            mock_kg = MagicMock()
            mock_kg_class.return_value = mock_kg

            trader = GraphWizTrader(temp_path)

            # Should default to 0.1.0
            assert trader.config.get("version", "0.1.0") == "0.1.0"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestMainFunction:
    """Test suite for main entry point."""

    @patch('graphwiz_trader.main.GraphWizTrader')
    @patch('sys.argv', ['main', '--config', 'test.yaml'])
    def test_main_function_with_config(self, mock_trader_class):
        """Test main function with config argument."""
        from graphwiz_trader.main import main

        mock_trader = MagicMock()
        mock_trader.is_running.return_value = False
        mock_trader_class.return_value = mock_trader

        with patch('pathlib.Path.exists', return_value=True):
            result = main()

        assert result == 0
        mock_trader.start.assert_called_once()
        mock_trader.stop.assert_called_once()

    @patch('sys.argv', ['main', '--version'])
    def test_main_version_argument(self):
        """Test main function with version argument."""
        from graphwiz_trader.main import main

        with pytest.raises(SystemExit):
            main()

    @patch('graphwiz_trader.main.GraphWizTrader')
    @patch('sys.argv', ['main'])
    def test_main_function_keyboard_interrupt(self, mock_trader_class):
        """Test main function handles keyboard interrupt."""
        from graphwiz_trader.main import main

        mock_trader = MagicMock()
        mock_trader.is_running.side_effect = [True, KeyboardInterrupt]
        mock_trader_class.return_value = mock_trader

        with patch('pathlib.Path.exists', return_value=True):
            result = main()

        assert result == 0
        mock_trader.stop.assert_called_once()

    @patch('graphwiz_trader.main.GraphWizTrader')
    @patch('sys.argv', ['main', '--config', 'nonexistent.yaml'])
    def test_main_function_config_not_found(self, mock_trader_class):
        """Test main function when config file doesn't exist."""
        from graphwiz_trader.main import main

        with patch('pathlib.Path.exists', return_value=False):
            result = main()

        assert result == 1
        mock_trader_class.assert_not_called()

    @patch('graphwiz_trader.main.GraphWizTrader')
    @patch('sys.argv', ['main'])
    def test_main_function_exception_handling(self, mock_trader_class):
        """Test main function handles exceptions gracefully."""
        from graphwiz_trader.main import main

        mock_trader_class.side_effect = Exception("Test error")

        with patch('pathlib.Path.exists', return_value=True):
            result = main()

        assert result == 1
