"""Tests for portfolio management module."""

import pytest
from datetime import datetime, timedelta

from graphwiz_trader.portfolio import PortfolioManager, Asset, PortfolioSnapshot


class TestAsset:
    """Test suite for Asset dataclass."""

    def test_asset_creation(self):
        """Test creating an asset."""
        asset = Asset(
            symbol="BTC/USDT",
            quantity=1.5,
            entry_price=50000,
            current_price=55000
        )

        assert asset.symbol == "BTC/USDT"
        assert asset.quantity == 1.5
        assert asset.entry_price == 50000
        assert asset.value == 82500  # 1.5 * 55000
        assert asset.unrealized_pnl == 7500  # (55000 - 50000) * 1.5
        assert asset.unrealized_pnl_pct == 10.0  # (55000 - 50000) / 50000 * 100

    def test_asset_loss(self):
        """Test asset with loss."""
        asset = Asset(
            symbol="ETH/USDT",
            quantity=10,
            entry_price=3000,
            current_price=2500
        )

        assert asset.unrealized_pnl == -5000  # (2500 - 3000) * 10
        assert asset.unrealized_pnl_pct < 0


class TestPortfolioManager:
    """Test suite for PortfolioManager class."""

    def test_initialization(self):
        """Test portfolio manager initialization."""
        pm = PortfolioManager(initial_capital=100000)

        assert pm.initial_capital == 100000
        assert pm.cash == 100000
        assert len(pm.assets) == 0
        assert pm.max_positions == 10

    def test_add_position(self):
        """Test adding a position."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.6)  # Allow larger positions

        result = pm.add_position("BTC/USDT", 1.0, 50000)

        assert result is True
        assert "BTC/USDT" in pm.assets
        assert pm.cash == 50000  # 100000 - 50000
        assert pm.assets["BTC/USDT"].quantity == 1.0

    def test_add_position_insufficient_cash(self):
        """Test adding position with insufficient cash."""
        pm = PortfolioManager(initial_capital=1000)

        result = pm.add_position("BTC/USDT", 1.0, 50000)

        assert result is False
        assert len(pm.assets) == 0
        assert pm.cash == 1000  # Cash unchanged

    @pytest.mark.parametrize("max_positions,positions_to_add,expected_count", [
        (2, 2, 2),  # Exactly at limit
        (2, 3, 2),  # Exceeds limit
        (5, 3, 3),  # Under limit
    ])
    def test_add_position_max_positions(self, max_positions, positions_to_add, expected_count):
        """Test max positions limit."""
        pm = PortfolioManager(initial_capital=100000, max_positions=max_positions)

        # Add positions
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
        for i in range(positions_to_add):
            pm.add_position(symbols[i], 0.1, 10000)

        assert len(pm.assets) == expected_count

    @pytest.mark.parametrize("max_position_size,trade_cost,should_succeed", [
        (0.1, 50000, False),  # 50% trade, 10% max - fails
        (0.2, 15000, True),   # 15% trade, 20% max - succeeds
        (0.5, 50000, True),   # 50% trade, 50% max - succeeds
    ])
    def test_add_position_max_size(self, max_position_size, trade_cost, should_succeed):
        """Test max position size limit."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=max_position_size)

        quantity = trade_cost / 50000  # Calculate quantity to match cost
        result = pm.add_position("BTC/USDT", quantity, 50000)

        assert result is should_succeed
        assert len(pm.assets) == (1 if should_succeed else 0)

    def test_add_to_existing_position(self):
        """Test adding to an existing position."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.8)

        pm.add_position("BTC/USDT", 1.0, 50000)
        pm.add_position("BTC/USDT", 0.5, 52000)

        asset = pm.assets["BTC/USDT"]
        assert asset.quantity == 1.5

        # Entry price should be weighted average
        # (50000 * 1.0 + 52000 * 0.5) / 1.5 = 50666.67
        assert 50600 < asset.entry_price < 50700

    def test_remove_position(self):
        """Test removing a position."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.6)

        pm.add_position("BTC/USDT", 1.0, 50000)
        result = pm.remove_position("BTC/USDT", 0.5, 52000)

        assert result is True
        assert pm.cash == 76000  # 50000 + 52000 * 0.5
        assert pm.assets["BTC/USDT"].quantity == 0.5

    def test_remove_full_position(self):
        """Test removing entire position."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.6)

        pm.add_position("BTC/USDT", 1.0, 50000)
        result = pm.remove_position("BTC/USDT", 1.0, 52000)

        assert result is True
        assert "BTC/USDT" not in pm.assets
        assert pm.cash == 102000  # 50000 + 52000

    def test_remove_nonexistent_position(self):
        """Test removing position that doesn't exist."""
        pm = PortfolioManager(initial_capital=100000)

        result = pm.remove_position("BTC/USDT", 1.0, 50000)

        assert result is False

    def test_update_prices(self):
        """Test updating prices."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.8)

        pm.add_position("BTC/USDT", 1.0, 50000)
        pm.add_position("ETH/USDT", 10.0, 3000)

        pm.update_prices({
            "BTC/USDT": 55000,
            "ETH/USDT": 3500
        })

        assert pm.assets["BTC/USDT"].current_price == 55000
        assert pm.assets["BTC/USDT"].unrealized_pnl == 5000
        assert pm.assets["ETH/USDT"].current_price == 3500
        assert pm.assets["ETH/USDT"].unrealized_pnl == 5000

    def test_get_total_value(self):
        """Test calculating total portfolio value."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.6)

        assert pm.get_total_value() == 100000

        pm.add_position("BTC/USDT", 1.0, 50000)
        assert pm.get_total_value() == 100000  # 50000 cash + 50000 BTC

        pm.update_prices({"BTC/USDT": 55000})
        assert pm.get_total_value() == 105000  # 50000 cash + 55000 BTC

    def test_get_positions(self):
        """Test getting all positions."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.8)

        pm.add_position("BTC/USDT", 1.0, 50000)
        pm.add_position("ETH/USDT", 10.0, 3000)

        positions = pm.get_positions()

        assert len(positions) == 2
        assert any(p["symbol"] == "BTC/USDT" for p in positions)
        assert any(p["symbol"] == "ETH/USDT" for p in positions)

    def test_get_portfolio_summary(self):
        """Test getting portfolio summary."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.6)

        pm.add_position("BTC/USDT", 1.0, 50000)
        pm.update_prices({"BTC/USDT": 55000})

        summary = pm.get_portfolio_summary()

        assert summary["total_value"] == 105000
        assert summary["cash"] == 50000
        assert summary["invested_value"] == 55000
        assert summary["total_pnl"] == 5000
        assert summary["position_count"] == 1
        assert summary["total_return"] == 5000

    def test_set_target_weights(self):
        """Test setting target allocation weights."""
        pm = PortfolioManager(initial_capital=100000)

        pm.set_target_weights({
            "BTC/USDT": 0.6,
            "ETH/USDT": 0.3,
            "USDT": 0.1
        })

        assert pm.target_weights["BTC/USDT"] == 0.6
        assert pm.target_weights["ETH/USDT"] == 0.3
        assert pm.target_weights["USDT"] == 0.1

    def test_get_rebalance_trades(self):
        """Test calculating rebalancing trades."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.8)
        pm.rebalance_threshold = 0.05

        # Add positions that deviate from targets
        pm.add_position("BTC/USDT", 1.0, 50000)  # 50% weight
        pm.add_position("ETH/USDT", 5.0, 3000)   # 15% weight

        pm.set_target_weights({
            "BTC/USDT": 0.4,  # Should sell
            "ETH/USDT": 0.4,  # Should buy
            "USDT": 0.2
        })

        trades = pm.get_rebalance_trades()

        assert len(trades) > 0
        assert any(t["symbol"] == "BTC/USDT" and t["action"] == "sell" for t in trades)
        assert any(t["symbol"] == "ETH/USDT" and t["action"] == "buy" for t in trades)

    def test_portfolio_snapshots(self):
        """Test creating portfolio snapshots."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.6)
        timestamp = datetime.now()

        pm.add_position("BTC/USDT", 1.0, 50000, timestamp=timestamp)

        assert len(pm.snapshots) == 1

        snapshot = pm.snapshots[0]
        assert isinstance(snapshot, PortfolioSnapshot)
        assert snapshot.total_value == 100000
        assert snapshot.cash == 50000
        assert snapshot.invested_value == 50000

    def test_get_snapshots(self):
        """Test retrieving snapshots."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.8)

        pm.add_position("BTC/USDT", 1.0, 50000, timestamp=datetime.now())
        pm.add_position("ETH/USDT", 5.0, 3000, timestamp=datetime.now() + timedelta(seconds=1))

        snapshots = pm.get_snapshots()

        assert len(snapshots) == 2
        assert all("timestamp" in s for s in snapshots)
        assert all("total_value" in s for s in snapshots)

    def test_clear_positions(self):
        """Test clearing all positions."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=0.8)

        pm.add_position("BTC/USDT", 1.0, 50000)
        pm.add_position("ETH/USDT", 5.0, 3000)

        pm.clear_positions()

        assert len(pm.assets) == 0
        # Cash should be 100000 - 50000 - 15000 = 35000
        assert pm.cash == 35000

    def test_reset(self):
        """Test resetting portfolio."""
        pm = PortfolioManager(initial_capital=100000)

        pm.add_position("BTC/USDT", 1.0, 50000)
        pm.set_target_weights({"BTC/USDT": 1.0})

        pm.reset()

        assert pm.cash == 100000  # Back to initial
        assert len(pm.assets) == 0
        assert len(pm.target_weights) == 0
        assert len(pm.snapshots) == 0

    def test_reset_with_new_capital(self):
        """Test resetting with new capital amount."""
        pm = PortfolioManager(initial_capital=100000)

        pm.add_position("BTC/USDT", 1.0, 50000)

        pm.reset(initial_capital=200000)

        assert pm.initial_capital == 200000
        assert pm.cash == 200000

    def test_portfolio_metrics(self):
        """Test calculating portfolio metrics."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=1.0)  # Allow any size

        # Create snapshots with growth - add new positions instead of adding to existing
        for i in range(10):
            # Reset and add fresh position for each snapshot
            if i > 0:
                pm.clear_positions()
            value = 100000 + (i * 1000)  # Growing portfolio
            pm.cash = value
            pm.add_position(f"BTC{i}/USDT", 0.5, value / 2, timestamp=datetime.now() + timedelta(minutes=i))

        metrics = pm.calculate_portfolio_metrics()

        assert "total_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert metrics["total_return"] > 0  # Portfolio grew

    def test_weight_calculation(self):
        """Test portfolio weight calculation."""
        pm = PortfolioManager(initial_capital=100000, max_position_size=1.0)  # Disable size limit

        pm.add_position("BTC/USDT", 1.0, 50000)
        pm.add_position("ETH/USDT", 10.0, 2500)  # Changed from 25000 to 2500

        # Check weights - weights are based on total portfolio value
        positions = pm.get_positions()
        btc_position = next(p for p in positions if p["symbol"] == "BTC/USDT")
        eth_position = next(p for p in positions if p["symbol"] == "ETH/USDT")

        # Total invested = 50000 + 25000 = 75000, Cash = 25000, Total = 100000
        # Based on the implementation, weights are calculated as asset.value / total_value
        # where total_value = cash + sum(asset.values) = 25000 + 75000 = 100000
        # So BTC weight = 50000 / 100000 = 0.5, ETH weight = 25000 / 100000 = 0.25
        assert btc_position["weight"] == 50000 / 100000
        assert eth_position["weight"] == 25000 / 100000
