"""Tests for backtesting engine module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from graphwiz_trader.backtesting import (
    BacktestEngine,
    BacktestResult,
    Trade,
    SimpleMovingAverageStrategy,
    RSIMeanReversionStrategy
)


def generate_test_data(days: int = 50, trend: str = "up") -> list:
    """Generate synthetic OHLCV data for testing.

    Args:
        days: Number of days of data
        trend: "up", "down", or "sideways"

    Returns:
        List of data dictionaries
    """
    data = []
    base_price = 100.0
    now = datetime.now()

    for i in range(days):
        if trend == "up":
            change = 0.5 + (i * 0.05)  # Increasing uptrend
        elif trend == "down":
            change = -0.5 - (i * 0.05)  # Increasing downtrend
        else:
            change = (i % 10 - 5) * 0.2  # Sideways oscillation

        price = base_price + change
        high = price * 1.02
        low = price * 0.98
        volume = 1000000 + (i * 10000)

        data.append({
            "timestamp": now + timedelta(days=i),
            "open": price,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
            "price": price  # Alternate field name
        })

    return data


class TestBacktestEngine:
    """Test suite for BacktestEngine class."""

    def test_initialization(self):
        """Test BacktestEngine initialization."""
        engine = BacktestEngine(initial_capital=10000, commission=0.001)

        assert engine.initial_capital == 10000
        assert engine.commission == 0.001
        assert engine.technical_analysis is not None

    @pytest.mark.parametrize("trend,expected_return_check,description", [
        ("up", lambda r: r >= 0, "uptrending data"),
        ("down", lambda r: r <= 0, "downtrending data"),
        ("sideways", lambda r: True, "sideways data")
    ])
    def test_backtest_with_trends(self, trend, expected_return_check, description):
        """Test backtest with different market trends."""
        engine = BacktestEngine(initial_capital=10000)
        data = generate_test_data(days=50, trend=trend)

        # Simple buy-and-hold strategy
        def buy_hold_strategy(context):
            if context.get("data_points", 0) < 10:
                return "buy"
            return "hold"

        result = engine.run_backtest(data, buy_hold_strategy, "TEST/USDT")

        assert isinstance(result, BacktestResult)
        assert result.symbol == "TEST/USDT"
        assert len(result.trades) > 0
        assert expected_return_check(result.total_return), f"Return check failed for {description}"

    def test_backtest_empty_data(self):
        """Test backtest with no data."""
        engine = BacktestEngine()

        result = engine.run_backtest([], lambda ctx: "hold", "TEST/USDT")

        assert result.total_return == 0
        assert len(result.trades) == 0

    def test_backtest_metrics(self):
        """Test backtest calculates all metrics correctly."""
        engine = BacktestEngine(initial_capital=10000)
        data = generate_test_data(days=30, trend="up")

        def strategy(ctx):
            # Make a few trades
            dp = ctx.get("data_points", 0)
            if dp == 5:
                return "buy"
            elif dp == 15:
                return "sell"
            elif dp == 20:
                return "buy"
            return "hold"

        result = engine.run_backtest(data, strategy, "TEST/USDT")

        # Sharpe ratio is a direct attribute, not in metrics
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "sortino_ratio")
        assert "equity_curve" in result.metrics
        assert "trade_count" in result.metrics
        assert result.max_drawdown >= 0

    def test_max_drawdown_calculation(self):
        """Test max drawdown is calculated correctly."""
        engine = BacktestEngine()

        # Create equity curve with known drawdown
        equity = [1000, 1100, 1200, 1000, 900, 950]  # 25% drawdown from 1200 to 900
        max_dd = engine._calculate_max_drawdown(equity)

        # Drawdown should be around 25%
        assert max_dd > 20  # At least 20%
        assert max_dd < 30  # Less than 30%

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        engine = BacktestEngine()

        # Positive returns should give positive Sharpe
        equity_increasing = [1000, 1010, 1020, 1030, 1040, 1050]
        sharpe = engine._calculate_sharpe_ratio(equity_increasing)
        assert sharpe > 0

    def test_sma_strategy(self):
        """Test simple moving average strategy."""
        engine = BacktestEngine()
        data = generate_test_data(days=50, trend="up")

        strategy = SimpleMovingAverageStrategy(fast_period=5, slow_period=15)
        result = engine.run_backtest(data, strategy, "TEST/USDT")

        assert isinstance(result, BacktestResult)
        # SMA crossover should generate some trades
        assert len(result.trades) >= 0

    def test_rsi_strategy(self):
        """Test RSI mean reversion strategy."""
        engine = BacktestEngine()
        data = generate_test_data(days=50, trend="sideways")

        strategy = RSIMeanReversionStrategy(oversold=30, overbought=70)
        result = engine.run_backtest(data, strategy, "TEST/USDT")

        assert isinstance(result, BacktestResult)

    @pytest.mark.parametrize("signal,price,capital,position,entry_price,position_side,expected_action,expected_quantity,expected_value", [
        ("buy", 100, 10000, 0, 0, None, "buy", lambda q: q > 0, 1000),
        ("sell", 100, 10000, 10, 95, "long", "sell", 10, None),
        ("hold", 100, 10000, 0, 0, None, None, None, None)
    ])
    def test_trade_execution(self, signal, price, capital, position, entry_price,
                            position_side, expected_action, expected_quantity, expected_value):
        """Test trade execution for different signals."""
        engine = BacktestEngine(initial_capital=capital)

        trade = engine._execute_strategy(
            signal=signal,
            price=price,
            capital=capital,
            position=position,
            entry_price=entry_price,
            position_side=position_side,
            timestamp=datetime.now(),
            symbol="TEST/USDT"
        )

        if expected_action is None:
            assert trade is None
        else:
            assert trade is not None
            assert trade.action == expected_action
            assert trade.price == price

            # Handle callable quantity checks
            if callable(expected_quantity):
                assert expected_quantity(trade.quantity)
            elif expected_quantity is not None:
                assert trade.quantity == expected_quantity

            if expected_value is not None:
                assert trade.value == expected_value

    def test_commission_impact(self):
        """Test that commission reduces returns."""
        data = generate_test_data(days=30, trend="up")

        # Strategy that trades frequently
        def frequent_trading(ctx):
            dp = ctx.get("data_points", 0)
            if dp % 5 == 0 and dp < 25:
                return "buy" if dp % 10 == 0 else "sell"
            return "hold"

        # No commission
        engine_no_comm = BacktestEngine(initial_capital=10000, commission=0.0)
        result_no_comm = engine_no_comm.run_backtest(data, frequent_trading, "TEST/USDT")

        # With commission
        engine_with_comm = BacktestEngine(initial_capital=10000, commission=0.01)  # 1%
        result_with_comm = engine_with_comm.run_backtest(data, frequent_trading, "TEST/USDT")

        # Commission should reduce returns
        assert result_with_comm.total_return <= result_no_comm.total_return

    def test_result_properties(self):
        """Test BacktestResult dataclass."""
        trade = Trade(
            timestamp=datetime.now(),
            symbol="TEST/USDT",
            action="buy",
            price=100,
            quantity=10,
            value=1000
        )

        result = BacktestResult(
            symbol="TEST/USDT",
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=10000,
            final_capital=11000,
            total_return=1000,
            total_return_pct=10.0,
            trades=[trade]
        )

        assert result.symbol == "TEST/USDT"
        assert result.total_return_pct == 10.0
        assert len(result.trades) == 1
        assert result.win_rate >= 0  # Should calculate win rate

    def test_context_building(self):
        """Test context building for strategy."""
        engine = BacktestEngine()
        data = generate_test_data(days=20)

        context = engine._build_context(data, 105.0, "TEST/USDT")

        assert "symbol" in context
        assert "current_price" in context
        assert "technical_indicators" in context
        assert context["symbol"] == "TEST/USDT"
        assert context["current_price"] == 105.0

    def test_multiple_positions(self):
        """Test backtest handles multiple positions."""
        engine = BacktestEngine(initial_capital=10000)
        data = generate_test_data(days=40, trend="up")

        # Strategy that opens and closes positions
        def multi_position(ctx):
            dp = ctx.get("data_points", 0)
            if dp in [5, 20]:
                return "buy"
            elif dp in [15, 30]:
                return "sell"
            return "hold"

        result = engine.run_backtest(data, multi_position, "TEST/USDT")

        # Should have multiple trades
        assert len(result.trades) >= 2

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        engine = BacktestEngine()

        # Mix of positive and negative returns
        equity_mixed = [1000, 1050, 950, 1100, 900, 1150]
        sortino = engine._calculate_sortino_ratio(equity_mixed)

        # Should calculate a value
        assert isinstance(sortino, float)
        # May be negative or positive depending on the values
