"""
Tests for HFT (High-Frequency Trading) module.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from graphwiz_trader.hft import (
    CrossExchangeArbitrage,
    FastOrderExecutor,
    HFTRiskManager,
    OrderBook,
    OrderBookManager,
    StatisticalArbitrage,
    WebSocketMarketData,
)


# OrderBook Tests
@pytest.mark.hft
class TestOrderBook:
    """Test OrderBook class."""

    def test_orderbook_initialization(self) -> None:
        """Test order book initialization."""
        ob = OrderBook("BTC/USDT", max_depth=20)
        assert ob.symbol == "BTC/USDT"
        assert ob.max_depth == 20
        assert ob.best_bid is None
        assert ob.best_ask is None

    def test_orderbook_update(self) -> None:
        """Test order book update."""
        ob = OrderBook("BTC/USDT")
        orderbook_data = {
            "bids": [[50000.0, 1.0], [49999.0, 2.0], [49998.0, 3.0]],
            "asks": [[50001.0, 1.5], [50002.0, 2.5], [50003.0, 3.5]],
            "timestamp": 1234567890,
        }
        ob.update(orderbook_data)

        assert ob.best_bid == 50000.0
        assert ob.best_ask == 50001.0
        assert len(ob.bids) == 3
        assert len(ob.asks) == 3

    def test_orderbook_spread(self) -> None:
        """Test spread calculation."""
        ob = OrderBook("BTC/USDT")
        orderbook_data = {
            "bids": [[50000.0, 1.0]],
            "asks": [[50010.0, 1.0]],
            "timestamp": 1234567890,
        }
        ob.update(orderbook_data)

        assert ob.spread is not None
        assert ob.spread_bps is not None
        assert ob.spread == (50010.0 - 50000.0) / 50010.0
        assert abs(ob.spread_bps - 2.0) < 0.01  # ~2 bps

    def test_orderbook_mid_price(self) -> None:
        """Test mid price calculation."""
        ob = OrderBook("BTC/USDT")
        orderbook_data = {
            "bids": [[50000.0, 1.0]],
            "asks": [[50010.0, 1.0]],
            "timestamp": 1234567890,
        }
        ob.update(orderbook_data)

        assert ob.mid_price == 50005.0

    def test_orderbook_imbalance(self) -> None:
        """Test order book imbalance calculation."""
        ob = OrderBook("BTC/USDT")
        orderbook_data = {
            "bids": [[50000.0, 10.0], [49999.0, 5.0]],
            "asks": [[50001.0, 2.0], [50002.0, 3.0]],
            "timestamp": 1234567890,
        }
        ob.update(orderbook_data)

        # Bid volume = 15, Ask volume = 5, Imbalance = (15-5)/(15+5) = 0.5
        assert ob.imbalance == 0.5

    def test_orderbook_vwap(self) -> None:
        """Test VWAP calculation."""
        ob = OrderBook("BTC/USDT")
        orderbook_data = {
            "bids": [[50000.0, 1.0], [49999.0, 2.0]],
            "asks": [[50001.0, 1.0], [50002.0, 2.0]],
            "timestamp": 1234567890,
        }
        ob.update(orderbook_data)

        # Buy VWAP for 2 units: (50001*1 + 50002*1) / 2 = 50001.5
        vwap_buy = ob.get_vwap("buy", 2.0)
        assert vwap_buy == 50001.5

        # Sell VWAP for 2 units: (50000*1 + 49999*1) / 2 = 49999.5
        vwap_sell = ob.get_vwap("sell", 2.0)
        assert vwap_sell == 49999.5

    def test_orderbook_liquidity(self) -> None:
        """Test liquidity calculation."""
        ob = OrderBook("BTC/USDT")
        orderbook_data = {
            "bids": [[50000.0, 1.0], [49999.0, 2.0], [49998.0, 3.0]],
            "asks": [[50001.0, 1.5], [50002.0, 2.5], [50003.0, 3.5]],
            "timestamp": 1234567890,
        }
        ob.update(orderbook_data)

        # Total bid liquidity (depth 3) = 1 + 2 + 3 = 6
        assert ob.get_liquidity("sell", depth=3) == 6.0

        # Total ask liquidity (depth 3) = 1.5 + 2.5 + 3.5 = 7.5
        assert ob.get_liquidity("buy", depth=3) == 7.5


# OrderBookManager Tests
@pytest.mark.hft
class TestOrderBookManager:
    """Test OrderBookManager class."""

    def test_orderbook_manager_initialization(self) -> None:
        """Test order book manager initialization."""
        manager = OrderBookManager(max_depth=20)
        assert manager.max_depth == 20
        assert len(manager.books) == 0

    def test_orderbook_manager_update(self) -> None:
        """Test updating order books."""
        manager = OrderBookManager()
        orderbook_data = {
            "bids": [[50000.0, 1.0]],
            "asks": [[50010.0, 1.0]],
            "timestamp": 1234567890,
        }
        manager.update("binance", "BTC/USDT", orderbook_data)

        assert "BTC/USDT" in manager.books
        assert "binance" in manager.books["BTC/USDT"]
        assert manager.books["BTC/USDT"]["binance"].best_bid == 50000.0

    def test_arbitrage_opportunities(self) -> None:
        """Test arbitrage opportunity detection."""
        manager = OrderBookManager()

        # Binance: bid=50000, ask=50010
        manager.update(
            "binance",
            "BTC/USDT",
            {
                "bids": [[50000.0, 1.0]],
                "asks": [[50010.0, 1.0]],
                "timestamp": 1234567890,
            },
        )

        # OKX: bid=50020, ask=50005 (arbitrage opportunity!)
        # Buy on OKX at 50005, sell on Binance at 50000 - NO opportunity (would lose money)
        # Buy on Binance at 50010, sell on OKX at 50020 - YES! Profit!
        manager.update(
            "okx",
            "BTC/USDT",
            {
                "bids": [[50020.0, 1.0]],
                "asks": [[50005.0, 1.0]],
                "timestamp": 1234567890,
            },
        )

        opportunities = manager.get_arbitrage_opportunities("BTC/USDT", min_profit_bps=1)
        assert len(opportunities) > 0

        # Should be able to buy on binance at 50010 and sell on okx at 50020
        # Profit = (50020 - 50010) / 50010 * 10000 = ~2 bps
        arb = opportunities[0]
        assert arb["profit_bps"] > 1

    def test_get_book(self) -> None:
        """Test getting specific order book."""
        manager = OrderBookManager()
        orderbook_data = {
            "bids": [[50000.0, 1.0]],
            "asks": [[50010.0, 1.0]],
            "timestamp": 1234567890,
        }
        manager.update("binance", "BTC/USDT", orderbook_data)

        book = manager.get_book("binance", "BTC/USDT")
        assert book is not None
        assert book.symbol == "BTC/USDT"

        book = manager.get_book("okx", "BTC/USDT")
        assert book is None


# HFTRiskManager Tests
@pytest.mark.hft
class TestHFTRiskManager:
    """Test HFTRiskManager class."""

    def test_risk_manager_initialization(self) -> None:
        """Test risk manager initialization."""
        config = {
            "max_position_size": 1.0,
            "max_exposure": 10000,
            "max_orders_per_sec": 10,
            "circuit_breaker": -0.05,
        }
        rm = HFTRiskManager(config)

        assert rm.max_position_size == 1.0
        assert rm.max_exposure == 10000
        assert rm.circuit_breaker_tripped is False

    @pytest.mark.asyncio
    async def test_check_order_valid(self) -> None:
        """Test valid order check."""
        config = {
            "max_position_size": 1.0,
            "max_exposure": 100000,  # Increased to accommodate order value
            "start_balance": 100000,
        }
        rm = HFTRiskManager(config)

        order = {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "side": "buy",
            "amount": 0.5,
            "price": 50000,
        }

        approved, reason = await rm.check_order(order)
        if not approved:
            print(f"Order rejected: {reason}")
        assert approved is True
        assert reason == "OK"

    @pytest.mark.asyncio
    async def test_check_order_position_limit(self) -> None:
        """Test position limit violation."""
        config = {"max_position_size": 0.5}
        rm = HFTRiskManager(config)

        # Set existing position
        rm.positions["BTC/USDT"]["binance"] = 0.4

        order = {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "side": "buy",
            "amount": 0.3,  # Would result in 0.7 total
            "price": 50000,
        }

        approved, reason = await rm.check_order(order)
        assert approved is False
        assert "Position size limit exceeded" in reason

    @pytest.mark.asyncio
    async def test_check_order_rate_limit(self) -> None:
        """Test order rate limiting."""
        config = {"max_orders_per_sec": 2}
        rm = HFTRiskManager(config)

        order = {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "side": "buy",
            "amount": 0.1,
            "price": 50000,
        }

        # First two orders should pass
        approved1, _ = await rm.check_order(order)
        approved2, _ = await rm.check_order(order)
        assert approved1 is True
        assert approved2 is True

        # Third order should fail
        approved3, reason = await rm.check_order(order)
        assert approved3 is False
        assert "rate limit" in reason.lower()

    def test_update_position(self) -> None:
        """Test position update."""
        config = {"max_position_size": 1.0}
        rm = HFTRiskManager(config)

        fill = {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "side": "buy",
            "filled": 0.5,
            "pnl": 100.0,
        }

        rm.update_position(fill)
        assert rm.positions["BTC/USDT"]["binance"] == 0.5
        assert rm.daily_pnl == 100.0

    def test_circuit_breaker_trigger(self) -> None:
        """Test circuit breaker triggering."""
        config = {
            "max_exposure": 10000,
            "circuit_breaker": -0.05,  # -5% loss
        }
        rm = HFTRiskManager(config)

        # Simulate large loss
        fill = {
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "side": "buy",
            "filled": 1.0,
            "pnl": -600.0,  # More than 5% of 10000
        }

        rm.update_position(fill)
        assert rm.circuit_breaker_tripped is True

    def test_get_risk_metrics(self) -> None:
        """Test risk metrics retrieval."""
        config = {"max_position_size": 1.0, "max_exposure": 10000}
        rm = HFTRiskManager(config)

        metrics = rm.get_risk_metrics()
        assert "daily_pnl" in metrics
        assert "total_exposure" in metrics
        assert "circuit_breaker_tripped" in metrics
        assert metrics["circuit_breaker_tripped"] is False


# StatisticalArbitrage Tests
@pytest.mark.hft
class TestStatisticalArbitrage:
    """Test StatisticalArbitrage strategy."""

    @pytest.mark.asyncio
    async def test_stat_arb_initialization(self) -> None:
        """Test statistical arbitrage initialization."""
        config = {"lookback": 100, "z_threshold": 2.0}
        kg_mock = Mock()
        strategy = StatisticalArbitrage(config, kg_mock)

        assert strategy.lookback_period == 100
        assert strategy.z_score_threshold == 2.0
        assert strategy.running is False

    @pytest.mark.asyncio
    async def test_stat_arb_on_market_data(self) -> None:
        """Test processing market data."""
        config = {"lookback": 10, "z_threshold": 2.0}
        kg_mock = Mock()
        strategy = StatisticalArbitrage(config, kg_mock)

        # Add market data
        for i in range(15):
            data = {
                "symbol": "BTC/USDT",
                "bid": 50000 + i * 10,
                "ask": 50010 + i * 10,
                "timestamp": 1234567890 + i,
            }
            await strategy.on_market_data(data)

        assert len(strategy.price_history["BTC/USDT"]) == 10  # Limited by lookback

    @pytest.mark.asyncio
    async def test_stat_arb_generate_signal(self) -> None:
        """Test signal generation."""
        config = {"lookback": 10, "z_threshold": 1.5}
        kg_mock = Mock()
        strategy = StatisticalArbitrage(config, kg_mock)

        # Add prices: mean=50000, then add outlier
        for i in range(10):
            strategy.price_history["BTC/USDT"].append(
                {"price": 50000.0, "timestamp": 1234567890 + i}
            )

        # Add high outlier
        strategy.price_history["BTC/USDT"].append(
            {"price": 52000.0, "timestamp": 1234567900}
        )

        signal = await strategy.generate_signal("BTC/USDT")
        # Should generate sell signal due to high z-score
        assert signal is not None
        assert signal["action"] == "sell"
        assert signal["z_score"] > 1.5


# CrossExchangeArbitrage Tests
@pytest.mark.hft
class TestCrossExchangeArbitrage:
    """Test CrossExchangeArbitrage strategy."""

    @pytest.mark.asyncio
    async def test_cross_exchange_arb_initialization(self) -> None:
        """Test cross-exchange arbitrage initialization."""
        config = {"min_profit_bps": 5.0, "max_position_size": 0.1}
        kg_mock = Mock()
        ob_manager = OrderBookManager()
        strategy = CrossExchangeArbitrage(config, kg_mock, ob_manager)

        assert strategy.min_profit_bps == 5.0
        assert strategy.max_position_size == 0.1

    @pytest.mark.asyncio
    async def test_cross_exchange_arb_generate_signal(self) -> None:
        """Test arbitrage signal generation."""
        config = {"min_profit_bps": 1.0}  # Lowered threshold to match actual spread
        kg_mock = Mock()
        ob_manager = OrderBookManager()

        # Setup order books with arbitrage opportunity
        ob_manager.update(
            "binance",
            "BTC/USDT",
            {
                "bids": [[50000.0, 1.0]],
                "asks": [[50010.0, 1.0]],
                "timestamp": 1234567890,
            },
        )
        ob_manager.update(
            "okx",
            "BTC/USDT",
            {
                "bids": [[50020.0, 1.0]],
                "asks": [[50005.0, 1.0]],
                "timestamp": 1234567890,
            },
        )

        strategy = CrossExchangeArbitrage(config, kg_mock, ob_manager)
        signal = await strategy.generate_signal("BTC/USDT")

        assert signal is not None
        assert signal["profit_bps"] > 1.0


# FastOrderExecutor Tests
@pytest.mark.hft
class TestFastOrderExecutor:
    """Test FastOrderExecutor class."""

    def test_executor_initialization(self) -> None:
        """Test executor initialization."""
        exchanges = {
            "binance": {
                "enabled": False,  # Disabled to avoid actual connection
                "api_key": "test_key",
                "api_secret": "test_secret",
            }
        }
        executor = FastOrderExecutor(exchanges)
        assert len(executor.exchanges) == 0  # Not initialized because disabled

    @pytest.mark.asyncio
    async def test_get_order_history(self) -> None:
        """Test order history retrieval."""
        exchanges = {}
        executor = FastOrderExecutor(exchanges)

        # Manually add order to history
        executor.order_history.append(
            {
                "order_id": "12345",
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.1,
                "latency_ms": 5.5,
            }
        )

        history = executor.get_order_history()
        assert len(history) == 1
        assert history[0]["order_id"] == "12345"

    @pytest.mark.asyncio
    async def test_get_average_latency(self) -> None:
        """Test average latency calculation."""
        exchanges = {}
        executor = FastOrderExecutor(exchanges)

        # Add mock orders with latencies
        executor.order_history.extend(
            [
                {"exchange": "binance", "latency_ms": 5.0},
                {"exchange": "binance", "latency_ms": 10.0},
                {"exchange": "okx", "latency_ms": 15.0},
            ]
        )

        avg_all = executor.get_average_latency()
        assert avg_all == 10.0

        avg_binance = executor.get_average_latency("binance")
        assert avg_binance == 7.5


# WebSocketMarketData Tests
@pytest.mark.hft
class TestWebSocketMarketData:
    """Test WebSocketMarketData class."""

    def test_websocket_initialization(self) -> None:
        """Test WebSocket initialization."""
        exchanges = {
            "binance": {
                "api_key": "test_key",
                "api_secret": "test_secret",
            }
        }
        ws = WebSocketMarketData(exchanges)
        assert ws.running is False
        assert len(ws.callbacks) == 0

    @pytest.mark.asyncio
    async def test_register_callback(self) -> None:
        """Test callback registration."""
        exchanges = {}
        ws = WebSocketMarketData(exchanges)

        async def test_callback(data: dict) -> None:
            pass

        ws.register_callback("ticker", test_callback)
        assert "ticker" in ws.callbacks

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test start and stop."""
        exchanges = {}
        ws = WebSocketMarketData(exchanges)

        await ws.start()
        assert ws.running is True

        await ws.stop()
        assert ws.running is False
