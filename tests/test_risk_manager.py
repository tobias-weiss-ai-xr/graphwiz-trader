"""Tests for risk management system."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from graphwiz_trader.risk import (
    RiskManager,
    RiskLimits,
    RiskLimitsConfig,
    calculate_position_size,
    calculate_portfolio_risk,
    calculate_correlation_matrix,
    calculate_max_drawdown,
    StopLossCalculator,
    PositionSizingStrategy,
)


@pytest.fixture
def sample_prices():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Create correlated price series
    returns = np.random.multivariate_normal(
        mean=[0.001, 0.0005, 0.0008],
        cov=[[0.02, 0.01, 0.005], [0.01, 0.015, 0.003], [0.005, 0.003, 0.01]],
        size=100,
    )

    prices = pd.DataFrame(
        (1 + returns).cumprod(axis=0) * 100,
        index=dates,
        columns=["BTC", "ETH", "SOL"],
    )

    return prices


@pytest.fixture
def risk_manager():
    """Create a RiskManager instance for testing."""
    config = RiskLimitsConfig(
        max_position_size=0.20,  # 20% max per position
        max_total_exposure=1.0,  # 100% max exposure
        max_daily_loss_pct=0.10,  # 10% max daily loss
        max_correlated_exposure=0.40,  # 40% max correlated exposure
    )

    return RiskManager(account_balance=100000.0, limits_config=config)


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_fixed_fractional_position_size(self):
        """Test fixed fractional position sizing."""
        result = calculate_position_size(
            account_balance=100000.0,
            entry_price=100.0,
            stop_loss_price=98.0,  # 2% stop
            risk_per_trade=0.02,
            strategy=PositionSizingStrategy.FIXED_FRACTIONAL,
        )

        assert result["position_value"] > 0
        assert result["dollar_risk"] == 2000.0  # 2% of 100k
        assert result["position_size"] > 0
        assert result["stop_distance"] == 0.02

    def test_kelly_criterion_position_size(self):
        """Test Kelly Criterion position sizing."""
        result = calculate_position_size(
            account_balance=100000.0,
            entry_price=100.0,
            stop_loss_price=98.0,
            risk_per_trade=0.02,
            strategy=PositionSizingStrategy.KELLY_CRITERION,
            strategy_params={
                "win_rate": 0.55,
                "avg_win": 1.5,
                "avg_loss": 1.0,
                "kelly_fraction": 0.5,
            },
        )

        assert "kelly_percentage" in result
        assert result["kelly_percentage"] >= 0
        assert result["position_value"] > 0

    def test_fixed_dollar_position_size(self):
        """Test fixed dollar position sizing."""
        result = calculate_position_size(
            account_balance=100000.0,
            entry_price=100.0,
            stop_loss_price=98.0,
            strategy=PositionSizingStrategy.FIXED_DOLLAR,
            strategy_params={"fixed_amount": 5000.0},
        )

        assert result["dollar_risk"] == 5000.0
        assert result["position_value"] > 0

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            calculate_position_size(
                account_balance=-1000.0,
                entry_price=100.0,
                stop_loss_price=98.0,
            )

        with pytest.raises(ValueError):
            calculate_position_size(
                account_balance=100000.0,
                entry_price=0.0,
                stop_loss_price=98.0,
            )


class TestRiskLimits:
    """Test risk limits and validation."""

    @pytest.fixture
    def limits(self):
        """Create RiskLimits instance."""
        config = RiskLimitsConfig(
            max_position_size=0.10,
            min_position_size=0.001,
            max_total_exposure=1.0,
            max_daily_loss_pct=0.05,
        )
        return RiskLimits(config)

    def test_position_size_check_pass(self, limits):
        """Test position size check that passes."""
        allowed, message = limits.check_position_size(
            position_value=5000.0, portfolio_value=100000.0, symbol="BTC"
        )

        assert allowed is True
        assert "approved" in message.lower()

    def test_position_size_check_fail_too_large(self, limits):
        """Test position size check that fails (too large)."""
        allowed, message = limits.check_position_size(
            position_value=15000.0, portfolio_value=100000.0, symbol="BTC", hard_limit=True
        )

        assert allowed is False
        assert "too large" in message.lower()

    def test_position_size_check_fail_too_small(self, limits):
        """Test position size check that fails (too small)."""
        allowed, message = limits.check_position_size(
            position_value=50.0, portfolio_value=100000.0, symbol="BTC"
        )

        assert allowed is False
        assert "too small" in message.lower()

    def test_total_exposure_check(self, limits):
        """Test total exposure check."""
        allowed, message = limits.check_total_exposure(
            total_exposure=95000.0, portfolio_value=100000.0
        )

        assert allowed is True

        # Test exceeding limit
        allowed, message = limits.check_total_exposure(
            total_exposure=105000.0, portfolio_value=100000.0, hard_limit=True
        )

        assert allowed is False

    def test_daily_loss_check(self, limits):
        """Test daily loss check."""
        allowed, message = limits.check_daily_loss(
            daily_pnl=-3000.0, portfolio_value=100000.0
        )

        assert allowed is True

        # Test exceeding limit
        allowed, message = limits.check_daily_loss(
            daily_pnl=-6000.0, portfolio_value=100000.0, hard_limit=True
        )

        assert allowed is False


class TestStopLossCalculator:
    """Test stop-loss and take-profit calculations."""

    @pytest.fixture
    def calculator(self):
        """Create StopLossCalculator instance."""
        return StopLossCalculator(
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.06,
            risk_reward_ratio=3.0,
        )

    def test_calculate_stop_loss_long(self, calculator):
        """Test stop-loss calculation for long position."""
        entry_price = 100.0
        stop_loss = calculator.calculate_stop_loss(
            entry_price=entry_price,
            side="long",
            stop_loss_pct=0.05,
        )

        assert stop_loss < entry_price
        assert abs(stop_loss - 95.0) < 0.01  # 5% below entry

    def test_calculate_stop_loss_short(self, calculator):
        """Test stop-loss calculation for short position."""
        entry_price = 100.0
        stop_loss = calculator.calculate_stop_loss(
            entry_price=entry_price,
            side="short",
            stop_loss_pct=0.05,
        )

        assert stop_loss > entry_price
        assert abs(stop_loss - 105.0) < 0.01  # 5% above entry

    def test_calculate_take_profit_with_rr_ratio(self, calculator):
        """Test take-profit calculation using risk/reward ratio."""
        entry_price = 100.0
        stop_loss = 98.0
        take_profit = calculator.calculate_take_profit(
            entry_price=entry_price, stop_loss_price=stop_loss, side="long"
        )

        # Risk is $2, R:R is 3:1, so take-profit should be at $106
        assert take_profit > entry_price
        assert abs(take_profit - 106.0) < 0.01

    def test_calculate_trailing_stop_long(self, calculator):
        """Test trailing stop for long position."""
        entry_price = 100.0
        current_price = 110.0

        # Initial trailing stop
        trailing_stop_1 = calculator.calculate_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            side="long",
            trailing_distance_pct=0.03,
            best_price=110.0,
        )

        # When price moves up, trailing stop should move up
        assert trailing_stop_1 > entry_price * 0.97  # Above initial stop

        # When price stays same, trailing stop should stay same or move up
        trailing_stop_2 = calculator.calculate_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            side="long",
            trailing_distance_pct=0.03,
            best_price=110.0,
        )

        assert trailing_stop_2 >= trailing_stop_1

    def test_calculate_position_size_from_risk(self, calculator):
        """Test position size calculation from risk amount."""
        account_balance = 100000.0
        entry_price = 100.0
        stop_loss_price = 98.0
        risk_per_trade = 0.02

        position_size = calculator.calculate_position_size_from_risk(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            risk_per_trade_pct=risk_per_trade,
        )

        # Risk amount = $2000, stop distance = $2, so position = 1000 shares
        assert position_size > 0
        assert abs(position_size - 1000.0) < 0.01


class TestPortfolioRisk:
    """Test portfolio risk calculations."""

    def test_calculate_portfolio_risk_empty(self):
        """Test portfolio risk calculation with no positions."""
        result = calculate_portfolio_risk(
            positions=[],
            prices=pd.DataFrame(),
            confidence_level=0.95,
        )

        assert result["portfolio_value"] == 0.0
        assert result["var_95"] == 0.0

    def test_calculate_portfolio_risk_with_positions(self, sample_prices):
        """Test portfolio risk calculation with positions."""
        positions = [
            {"symbol": "BTC", "quantity": 1.0, "entry_price": sample_prices["BTC"].iloc[-1]},
            {"symbol": "ETH", "quantity": 10.0, "entry_price": sample_prices["ETH"].iloc[-1]},
        ]

        result = calculate_portfolio_risk(
            positions=positions,
            prices=sample_prices,
            confidence_level=0.95,
            method="historical",
        )

        assert result["portfolio_value"] > 0
        assert result["var_95"] > 0
        assert result["cvar_95"] >= result["var_95"]  # CVaR should be >= VaR
        assert result["portfolio_std"] > 0


class TestCorrelationMatrix:
    """Test correlation matrix calculation."""

    def test_calculate_correlation_matrix(self, sample_prices):
        """Test correlation matrix calculation."""
        corr_matrix = calculate_correlation_matrix(sample_prices, method="pearson")

        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        assert all(corr_matrix.diagonal() == 1.0)  # Diagonal should be 1.0

    def test_correlation_symmetry(self, sample_prices):
        """Test that correlation matrix is symmetric."""
        corr_matrix = calculate_correlation_matrix(sample_prices)

        # Matrix should be symmetric
        assert np.allclose(corr_matrix.values, corr_matrix.values.T)


class TestMaxDrawdown:
    """Test maximum drawdown calculations."""

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create price series with known drawdown
        prices = pd.Series([100, 110, 105, 95, 90, 100, 110, 120])

        result = calculate_max_drawdown(prices)

        assert result["max_drawdown"] > 0
        assert result["max_drawdown"] <= 1.0  # Should be a percentage
        assert result["trough_date"] is not pd.NaT
        assert result["peak_date"] is not pd.NaT

    def test_max_drawdown_empty_series(self):
        """Test drawdown calculation with empty series."""
        prices = pd.Series(dtype=float)
        result = calculate_max_drawdown(prices)

        assert result["max_drawdown"] == 0.0
        assert result["current_drawdown"] == 0.0

    def test_max_drawdown_recovery(self):
        """Test drawdown recovery calculation."""
        # Create series that recovers from drawdown
        prices = pd.Series([100, 90, 85, 90, 95, 100, 105])

        result = calculate_max_drawdown(prices)

        assert result["max_drawdown"] > 0
        assert result["recovery_date"] is not pd.NaT  # Should recover


class TestRiskManager:
    """Test RiskManager integration."""

    def test_risk_manager_initialization(self, risk_manager):
        """Test RiskManager initialization."""
        assert risk_manager.account_balance == 100000.0
        assert risk_manager.initial_balance == 100000.0
        assert len(risk_manager.positions) == 0

    def test_add_position(self, risk_manager):
        """Test adding a position."""
        position = risk_manager.add_position(
            symbol="BTC",
            quantity=1.0,
            entry_price=50000.0,
            side="long",
            sector="Crypto",
        )

        assert position.symbol == "BTC"
        assert position.quantity == 1.0
        assert "BTC" in risk_manager.positions

    def test_add_position_exceeds_limit(self, risk_manager):
        """Test that large positions are rejected."""
        with pytest.raises(ValueError):
            risk_manager.add_position(
                symbol="BTC",
                quantity=50.0,  # This would be > 20% limit
                entry_price=50000.0,
                side="long",
            )

    def test_update_position_price(self, risk_manager):
        """Test updating position price."""
        risk_manager.add_position(symbol="BTC", quantity=1.0, entry_price=50000.0)

        risk_manager.update_position_price("BTC", 51000.0)

        assert risk_manager.positions["BTC"].current_price == 51000.0

    def test_close_position(self, risk_manager):
        """Test closing a position."""
        risk_manager.add_position(symbol="BTC", quantity=1.0, entry_price=50000.0)
        risk_manager.update_position_price("BTC", 51000.0)

        pnl = risk_manager.close_position("BTC")

        assert pnl == 1000.0  # $51000 - $50000
        assert "BTC" not in risk_manager.positions
        assert risk_manager.account_balance > risk_manager.initial_balance

    def test_get_portfolio_state(self, risk_manager):
        """Test getting portfolio state."""
        risk_manager.add_position(symbol="BTC", quantity=1.0, entry_price=50000.0)
        risk_manager.add_position(symbol="ETH", quantity=10.0, entry_price=3000.0)

        state = risk_manager.get_portfolio_state()

        assert state.total_value == 100000.0
        assert len(state.positions) == 2
        assert state.daily_pnl == 0.0

    def test_calculate_position_size_with_risk_check(self, risk_manager):
        """Test position size calculation with risk checks."""
        result = risk_manager.calculate_position_size(
            symbol="BTC",
            entry_price=50000.0,
            stop_loss_price=49000.0,  # 2% stop loss
            strategy=PositionSizingStrategy.FIXED_FRACTIONAL,
        )

        assert result["position_value"] > 0
        assert result["dollar_risk"] == 2000.0  # 2% of 100k

    def test_position_size_rejected_when_exceeds_exposure(self, risk_manager):
        """Test that position size is rejected when it would exceed exposure."""
        # Add positions that use up most of the exposure
        risk_manager.add_position(symbol="BTC", quantity=1.5, entry_price=50000.0)

        # Try to add another large position
        with pytest.raises(ValueError):
            risk_manager.calculate_position_size(
                symbol="ETH",
                entry_price=3000.0,
                stop_loss_price=2940.0,
                strategy=PositionSizingStrategy.FIXED_FRACTIONAL,
            )

    def test_daily_metrics_reset(self, risk_manager):
        """Test daily metrics reset."""
        risk_manager.add_position(symbol="BTC", quantity=1.0, entry_price=50000.0)
        risk_manager.trades_today = 10
        risk_manager.daily_pnl = 500.0

        risk_manager.reset_daily_metrics()

        assert risk_manager.trades_today == 0
        assert risk_manager.daily_pnl == 0.0

    def test_get_risk_summary(self, risk_manager):
        """Test getting risk summary."""
        risk_manager.add_position(symbol="BTC", quantity=1.0, entry_price=50000.0)

        summary = risk_manager.get_risk_summary()

        assert "account_balance" in summary
        assert "daily_pnl" in summary
        assert "num_positions" in summary
        assert summary["num_positions"] == 1
