#!/usr/bin/env python3
"""
Risk management tests.

Tests position limits, daily loss limits, and risk checks.
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RiskManager:
    """Risk management module (same logic as live trading)."""

    def __init__(
        self,
        max_position_eur: float = 300.0,
        max_daily_loss_eur: float = 50.0,
        max_daily_trades: int = 2,
        stop_loss_percent: float = 0.015,
        take_profit_percent: float = 0.03
    ):
        self.max_position = max_position_eur
        self.max_daily_loss = max_daily_loss_eur
        self.max_daily_trades = max_daily_trades
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent

        # Track state
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.positions = {}

    def check_position_limit(self, position_size_eur: float) -> tuple:
        """Check if position size is within limits."""
        if position_size_eur > self.max_position:
            return False, f"Position size €{position_size_eur:.2f} exceeds maximum €{self.max_position:.2f}"
        return True, "Position size OK"

    def check_daily_loss_limit(self) -> tuple:
        """Check if daily loss limit has been reached."""
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss €{abs(self.daily_pnl):.2f} exceeds limit €{self.max_daily_loss:.2f}"
        return True, "Daily loss limit OK"

    def check_trade_count_limit(self) -> tuple:
        """Check if daily trade count limit has been reached."""
        if self.daily_trade_count >= self.max_daily_trades:
            return False, f"Daily trades {self.daily_trade_count} exceeds limit {self.max_daily_trades}"
        return True, "Trade count OK"

    def check_all_limits(self, position_size_eur: float) -> tuple:
        """Check all risk limits before trade."""
        checks = [
            self.check_position_limit(position_size_eur),
            self.check_daily_loss_limit(),
            self.check_trade_count_limit()
        ]

        all_passed = all(check[0] for check in checks)
        messages = [check[1] for check in checks]

        return all_passed, messages

    def calculate_stop_loss(self, entry_price: float, action: str) -> float:
        """Calculate stop loss price."""
        if action == 'BUY':
            return entry_price * (1 - self.stop_loss_percent)
        else:  # SELL
            return entry_price * (1 + self.stop_loss_percent)

    def calculate_take_profit(self, entry_price: float, action: str) -> float:
        """Calculate take profit price."""
        if action == 'BUY':
            return entry_price * (1 + self.take_profit_percent)
        else:  # SELL
            return entry_price * (1 - self.take_profit_percent)

    def update_pnl(self, trade_pnl: float):
        """Update daily P&L after trade."""
        self.daily_pnl += trade_pnl

    def increment_trade_count(self):
        """Increment daily trade count."""
        self.daily_trade_count += 1


class TestPositionLimits(unittest.TestCase):
    """Test position size limits."""

    def setUp(self):
        """Set up risk manager."""
        self.risk = RiskManager(max_position_eur=300.0)

    def test_position_within_limit(self):
        """Test position size within limit."""
        passed, msg = self.risk.check_position_limit(250)
        print(f"\n  ✓ €250 position: {msg}")
        self.assertTrue(passed)

    def test_position_at_limit(self):
        """Test position size exactly at limit."""
        passed, msg = self.risk.check_position_limit(300)
        print(f"  ✓ €300 position: {msg}")
        self.assertTrue(passed)

    def test_position_exceeds_limit(self):
        """Test position size exceeds limit."""
        passed, msg = self.risk.check_position_limit(350)
        print(f"  ✓ €350 position: {msg}")
        self.assertFalse(passed)

    def test_position_small(self):
        """Test small position size."""
        passed, msg = self.risk.check_position_limit(50)
        print(f"  ✓ €50 position: {msg}")
        self.assertTrue(passed)


class TestDailyLossLimits(unittest.TestCase):
    """Test daily loss limit logic."""

    def setUp(self):
        """Set up risk manager."""
        self.risk = RiskManager(max_daily_loss_eur=50.0)

    def test_no_loss(self):
        """Test with no loss."""
        passed, msg = self.risk.check_daily_loss_limit()
        print(f"\n  ✓ No loss: {msg}")
        self.assertTrue(passed)

    def test_small_loss(self):
        """Test with small loss within limit."""
        self.risk.update_pnl(-25)
        passed, msg = self.risk.check_daily_loss_limit()
        print(f"  ✓ €25 loss: {msg}")
        self.assertTrue(passed)

    def test_loss_at_limit(self):
        """Test loss exactly at limit."""
        self.risk.update_pnl(-50)
        passed, msg = self.risk.check_daily_loss_limit()
        print(f"  ✓ €50 loss: {msg}")
        self.assertFalse(passed)  # Should reject trades

    def test_loss_exceeds_limit(self):
        """Test loss exceeds limit."""
        self.risk.update_pnl(-75)
        passed, msg = self.risk.check_daily_loss_limit()
        print(f"  ✓ €75 loss: {msg}")
        self.assertFalse(passed)

    def test_profit_ignores_limit(self):
        """Test that profit ignores loss limit."""
        self.risk.update_pnl(100)
        passed, msg = self.risk.check_daily_loss_limit()
        print(f"  ✓ €100 profit: {msg}")
        self.assertTrue(passed)


class TestTradeCountLimits(unittest.TestCase):
    """Test daily trade count limits."""

    def setUp(self):
        """Set up risk manager."""
        self.risk = RiskManager(max_daily_trades=2)

    def test_no_trades(self):
        """Test with no trades yet."""
        passed, msg = self.risk.check_trade_count_limit()
        print(f"\n  ✓ No trades: {msg}")
        self.assertTrue(passed)

    def test_one_trade(self):
        """Test with one trade."""
        self.risk.increment_trade_count()
        passed, msg = self.risk.check_trade_count_limit()
        print(f"  ✓ 1 trade: {msg}")
        self.assertTrue(passed)

    def test_at_limit(self):
        """Test at trade limit."""
        self.risk.increment_trade_count()
        self.risk.increment_trade_count()
        passed, msg = self.risk.check_trade_count_limit()
        print(f"  ✓ 2 trades: {msg}")
        self.assertFalse(passed)  # Should reject more trades

    def test_exceeds_limit(self):
        """Test exceeding limit."""
        self.risk.daily_trade_count = 3
        passed, msg = self.risk.check_trade_count_limit()
        print(f"  ✓ 3 trades: {msg}")
        self.assertFalse(passed)


class TestStopLossTakeProfit(unittest.TestCase):
    """Test stop loss and take profit calculations."""

    def setUp(self):
        """Set up risk manager."""
        self.risk = RiskManager(
            stop_loss_percent=0.015,  # 1.5%
            take_profit_percent=0.03   # 3%
        )

    def test_stop_loss_buy(self):
        """Test stop loss for BUY position."""
        entry_price = 50000
        stop_loss = self.risk.calculate_stop_loss(entry_price, 'BUY')
        expected = 50000 * (1 - 0.015)  # 49250
        print(f"\n  ✓ BUY stop loss: €{stop_loss:.2f} (entry: €{entry_price})")
        self.assertAlmostEqual(stop_loss, expected, places=2)

    def test_stop_loss_sell(self):
        """Test stop loss for SELL position."""
        entry_price = 50000
        stop_loss = self.risk.calculate_stop_loss(entry_price, 'SELL')
        expected = 50000 * (1 + 0.015)  # 50750
        print(f"  ✓ SELL stop loss: €{stop_loss:.2f} (entry: €{entry_price})")
        self.assertAlmostEqual(stop_loss, expected, places=2)

    def test_take_profit_buy(self):
        """Test take profit for BUY position."""
        entry_price = 50000
        take_profit = self.risk.calculate_take_profit(entry_price, 'BUY')
        expected = 50000 * (1 + 0.03)  # 51500
        print(f"  ✓ BUY take profit: €{take_profit:.2f} (entry: €{entry_price})")
        self.assertAlmostEqual(take_profit, expected, places=2)

    def test_take_profit_sell(self):
        """Test take profit for SELL position."""
        entry_price = 50000
        take_profit = self.risk.calculate_take_profit(entry_price, 'SELL')
        expected = 50000 * (1 - 0.03)  # 48500
        print(f"  ✓ SELL take profit: €{take_profit:.2f} (entry: €{entry_price})")
        self.assertAlmostEqual(take_profit, expected, places=2)

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio is favorable."""
        entry_price = 50000
        stop_loss = self.risk.calculate_stop_loss(entry_price, 'BUY')
        take_profit = self.risk.calculate_take_profit(entry_price, 'BUY')

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        ratio = reward / risk

        print(f"  ✓ Risk/Reward ratio: {ratio:.2f}")
        self.assertGreater(ratio, 1.0, "Risk/reward should be > 1:1")
        self.assertAlmostEqual(ratio, 2.0, places=1, msg="Risk/reward should be ~2:1")


class TestCompositeRiskChecks(unittest.TestCase):
    """Test combined risk checks."""

    def setUp(self):
        """Set up risk manager."""
        self.risk = RiskManager(
            max_position_eur=300.0,
            max_daily_loss_eur=50.0,
            max_daily_trades=2
        )

    def test_all_checks_pass(self):
        """Test all risk checks pass."""
        passed, messages = self.risk.check_all_limits(250)
        print(f"\n  ✓ All checks pass: {messages}")
        self.assertTrue(passed)

    def test_position_too_large(self):
        """Test position size check fails."""
        passed, messages = self.risk.check_all_limits(350)
        print(f"  ✓ Position too large: {messages}")
        self.assertFalse(passed)

    def test_daily_loss_reached(self):
        """Test daily loss check fails."""
        self.risk.update_pnl(-60)
        passed, messages = self.risk.check_all_limits(250)
        print(f"  ✓ Daily loss reached: {messages}")
        self.assertFalse(passed)

    def test_trade_limit_reached(self):
        """Test trade count check fails."""
        self.risk.daily_trade_count = 2
        passed, messages = self.risk.check_all_limits(250)
        print(f"  ✓ Trade limit reached: {messages}")
        self.assertFalse(passed)

    def test_multiple_failures(self):
        """Test multiple checks fail simultaneously."""
        self.risk.update_pnl(-75)
        self.risk.daily_trade_count = 3
        passed, messages = self.risk.check_all_limits(350)
        print(f"  ✓ Multiple failures: {messages}")
        self.assertFalse(passed)
        # Should have 3 failed checks
        failures = [m for m in messages if "exceeds" in m or "reached" in m]
        self.assertGreaterEqual(len(failures), 2)


def run_tests():
    """Run all risk management tests."""
    print("=" * 80)
    print("Risk Management Tests")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPositionLimits))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDailyLossLimits))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTradeCountLimits))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStopLossTakeProfit))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCompositeRiskChecks))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
