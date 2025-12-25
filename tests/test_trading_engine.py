"""Tests for trading engine module."""

import pytest
from unittest.mock import MagicMock, patch, call

from graphwiz_trader.trading import TradingEngine


class TestTradingEngine:
    """Test suite for TradingEngine class."""

    def test_initialization(self, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test TradingEngine initialization."""
        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)

        assert engine.config == trading_config
        assert engine.exchanges_config == exchanges_config
        assert engine.kg == mock_kg
        assert engine.agents == mock_agent_orchestrator
        assert engine.exchanges == {}
        assert engine._running is False

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_initialize_exchanges(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test exchange initialization."""
        # Setup mock exchange
        mock_exchange = MagicMock()
        mock_exchange.close.return_value = None
        mock_ccxt.binance.return_value = mock_exchange

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()

        assert "binance" in engine.exchanges
        assert "kraken" not in engine.exchanges  # Disabled in config
        mock_ccxt.binance.assert_called_once()

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_initialize_exchanges_sandbox_mode(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test that sandbox mode is enabled when configured."""
        mock_exchange = MagicMock()
        mock_exchange.close.return_value = None
        mock_ccxt.binance.return_value = mock_exchange

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()

        # Verify sandbox mode was set
        mock_exchange.set_sandbox_mode.assert_called_once_with(True)

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_initialize_exchanges_failure(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test handling of exchange initialization failures."""
        mock_ccxt.binance.side_effect = Exception("API Error")

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()

        # Should not crash, just log the error
        assert "binance" not in engine.exchanges

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_start(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test starting the trading engine."""
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine.start()

        assert engine._running is True
        assert "binance" in engine.exchanges

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_stop(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test stopping the trading engine."""
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine.start()
        engine.stop()

        assert engine._running is False
        mock_exchange.close.assert_called_once()

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_close_exchanges(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test closing all exchange connections."""
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()
        engine._close_exchanges()

        mock_exchange.close.assert_called_once()

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_close_exchanges_with_error(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test closing exchanges handles errors gracefully."""
        mock_exchange = MagicMock()
        mock_exchange.close.side_effect = Exception("Close error")
        mock_ccxt.binance.return_value = mock_exchange

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()
        # Should not raise exception
        engine._close_exchanges()

    def test_execute_trade_stub(self, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test execute_trade stub implementation."""
        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)

        result = engine.execute_trade("BTC/USDT", "buy", 0.1)

        assert result["status"] == "executed"
        assert result["symbol"] == "BTC/USDT"
        assert result["side"] == "buy"
        assert result["amount"] == 0.1

    def test_execute_trade_sell(self, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test execute_trade with sell order."""
        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)

        result = engine.execute_trade("ETH/USDT", "sell", 1.5)

        assert result["status"] == "executed"
        assert result["symbol"] == "ETH/USDT"
        assert result["side"] == "sell"
        assert result["amount"] == 1.5

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_multiple_exchanges(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test initializing multiple exchanges."""
        exchanges_config["kraken"]["enabled"] = True
        mock_binance = MagicMock()
        mock_kraken = MagicMock()
        mock_ccxt.binance.return_value = mock_binance
        mock_ccxt.kraken.return_value = mock_kraken

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()

        assert "binance" in engine.exchanges
        assert "kraken" in engine.exchanges
        assert len(engine.exchanges) == 2

    @patch('graphwiz_trader.trading.engine.ccxt')
    def test_exchange_config_parameters(self, mock_ccxt, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test that exchange config parameters are passed correctly."""
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange

        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)
        engine._initialize_exchanges()

        # Verify CCXT was called with correct parameters
        call_args = mock_ccxt.binance.call_args
        config_dict = call_args[0][0]

        assert config_dict["apiKey"] == "test_key"
        assert config_dict["secret"] == "test_secret"
        assert config_dict["sandbox"] is True
        assert config_dict["enableRateLimit"] is True

    def test_running_state(self, trading_config, exchanges_config, mock_kg, mock_agent_orchestrator):
        """Test the running state of the trading engine."""
        engine = TradingEngine(trading_config, exchanges_config, mock_kg, mock_agent_orchestrator)

        assert engine._running is False

        with patch.object(engine, '_initialize_exchanges'):
            engine.start()
            assert engine._running is True

        with patch.object(engine, '_close_exchanges'):
            engine.stop()
            assert engine._running is False
