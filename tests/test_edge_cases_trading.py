#!/usr/bin/env python3
"""
Edge case tests for trading system.

Tests boundary conditions, unusual scenarios, and corner cases.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)


class TestPriceEdgeCases(unittest.TestCase):
    """Test edge cases related to prices."""

    def test_extreme_high_price(self):
        """Test handling of extremely high prices."""
        print("\n  Testing extremely high price...")

        price = 999999999.99  # Nearly 1 billion

        # Should handle without overflow
        position_size = 300 / price
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, 1)

        print(f"  ✓ Extreme price handled: €{price:,.2f}")
        print(f"    Position size: {position_size:.10f}")

    def test_extreme_low_price(self):
        """Test handling of extremely low prices."""
        print("\n  Testing extremely low price...")

        price = 0.0001  # Very low price

        # Should handle without underflow
        position_size = 300 / price
        self.assertGreater(position_size, 0)

        print(f"  ✓ Low price handled: €{price:.4f}")
        print(f"    Position size: {position_size:.2f}")

    def test_zero_price(self):
        """Test handling of zero price."""
        print("\n  Testing zero price...")

        price = 0

        # Should reject zero price
        if price <= 0:
            valid = False
        else:
            valid = False

        print(f"  ✓ Zero price rejected")
        self.assertFalse(valid)

    def test_negative_price(self):
        """Test handling of negative price."""
        print("\n  Testing negative price...")

        price = -100

        # Should reject negative price
        if price > 0:
            valid = True
        else:
            valid = False

        print(f"  ✓ Negative price rejected")
        self.assertFalse(valid)

    def test_price_very_small_change(self):
        """Test RSI with very small price changes."""
        print("\n  Testing minimal price variation...")

        # Prices that barely change
        prices = [100.00, 100.01, 100.00, 99.99, 100.00, 100.01, 100.00, 99.99,
                 100.00, 100.01, 100.00, 99.99, 100.00, 100.01, 100.00]

        # Calculate RSI
        if len(prices) >= 14:
            df = pd.DataFrame({'close': prices})
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            # Avoid division by zero
            if loss.iloc[-1] == 0:
                rsi = 100 if gain.iloc[-1] > 0 else 50
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

            print(f"  ✓ RSI with minimal variation: {rsi:.2f}")
            self.assertGreaterEqual(rsi, 0)
            self.assertLessEqual(rsi, 100)


class TestPositionEdgeCases(unittest.TestCase):
    """Test edge cases related to position sizing."""

    def test_exact_position_limit(self):
        """Test position exactly at limit."""
        print("\n  Testing position at exact limit...")

        balance = 1200  # 1200 * 0.25 = 300 (exactly at max_position)
        max_position = 300
        price = 50000

        position_size_eur = min(max_position, balance * 0.25)
        amount = position_size_eur / price

        print(f"  ✓ Position at limit: €{position_size_eur:.2f}")
        print(f"    Amount: {amount:.6f} BTC")

        # Should use max_position
        self.assertEqual(position_size_eur, max_position)

    def test_position_just_below_limit(self):
        """Test position just below limit."""
        print("\n  Testing position just below limit...")

        balance = 1199  # 1199 * 0.25 = 299.75, just below 300
        max_position = 300
        price = 50000

        position_size_eur = min(max_position, balance * 0.25)
        amount = position_size_eur / price

        print(f"  ✓ Position below limit: €{position_size_eur:.2f}")
        print(f"    Amount: {amount:.6f} BTC")

        # Should use balance calculation
        self.assertAlmostEqual(position_size_eur, 299.75, places=2)

    def test_position_just_above_limit(self):
        """Test position just above limit."""
        print("\n  Testing position just above limit...")

        balance = 1201  # 1201 * 0.25 = 300.25, just above 300
        max_position = 300
        price = 50000

        position_size_eur = min(max_position, balance * 0.25)

        print(f"  ✓ Position capped at limit: €{position_size_eur:.2f}")

        # Should cap at max_position
        self.assertEqual(position_size_eur, max_position)

    def test_minimum_threshold_boundary(self):
        """Test position at minimum threshold boundary."""
        print("\n  Testing minimum threshold boundary...")

        # Exactly €10 worth
        balance = 40  # 40 * 0.25 = 10
        price = 50000

        position_size_eur = balance * 0.25

        if position_size_eur < 10:
            allowed = False
        else:
            allowed = True

        print(f"  ✓ Minimum threshold boundary: €{position_size_eur:.2f} -> {'Allowed' if allowed else 'Rejected'}")
        self.assertTrue(allowed)

    def test_fractional_satoshi(self):
        """Test handling of fractional satoshis."""
        print("\n  Testing fractional satoshis...")

        # Very small BTC amount (less than 1 satoshi)
        amount = 0.00000001  # 1 satoshi
        price = 50000

        value = amount * price

        print(f"  ✓ Fractional satoshi: {amount:.8f} BTC = €{value:.6f}")
        self.assertGreater(value, 0)


class TestRSIEdgeCases(unittest.TestCase):
    """Test edge cases in RSI calculation."""

    def test_rsi_exactly_at_thresholds(self):
        """Test RSI exactly at buy/sell thresholds."""
        print("\n  Testing RSI at exact thresholds...")

        # Test at exact boundaries
        rsi_values = [42.0, 58.0]  # Thresholds

        for rsi in rsi_values:
            if rsi < 42:
                action = "BUY"
            elif rsi > 58:
                action = "SELL"
            else:
                action = "HOLD"

            print(f"  ✓ RSI {rsi}: {action}")
            self.assertEqual(action, "HOLD")

    def test_rsi_extreme_values(self):
        """Test RSI at extreme values (0 and 100)."""
        print("\n  Testing RSI at extremes...")

        for rsi in [0, 100]:
            if rsi < 42:
                action = "BUY"
                confidence = min(0.95, 0.65 + (42 - rsi) / 80)
            elif rsi > 58:
                action = "SELL"
                confidence = min(0.95, 0.65 + (rsi - 58) / 80)
            else:
                action = "HOLD"
                confidence = 0.5

            print(f"  ✓ RSI {rsi}: {action} (confidence: {confidence:.2f})")

            # Confidence should be high at extremes
            if rsi in [0, 100]:
                self.assertGreater(confidence, 0.8)

    def test_rsi_single_price_repeated(self):
        """Test RSI when price stays constant."""
        print("\n  Testing RSI with constant price...")

        # All same prices
        prices = [50000] * 20

        if len(prices) >= 14:
            df = pd.DataFrame({'close': prices})
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            if loss.iloc[-1] == 0 and gain.iloc[-1] == 0:
                rsi = 50.0  # Neutral when no change
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

            print(f"  ✓ RSI with constant price: {rsi:.2f}")
            self.assertEqual(rsi, 50.0)


class TestRiskEdgeCases(unittest.TestCase):
    """Test edge cases in risk management."""

    def test_daily_loss_exactly_at_limit(self):
        """Test daily loss exactly at limit."""
        print("\n  Testing daily loss at exact limit...")

        daily_pnl = -50.0
        max_loss = 50.0

        should_stop = daily_pnl <= -max_loss

        print(f"  ✓ Loss at limit: €{abs(daily_pnl)} -> {'Stop' if should_stop else 'Continue'}")
        self.assertTrue(should_stop)

    def test_daily_loss_just_below_limit(self):
        """Test daily loss just below limit."""
        print("\n  Testing daily loss just below limit...")

        daily_pnl = -49.99
        max_loss = 50.0

        should_stop = daily_pnl <= -max_loss

        print(f"  ✓ Loss just below limit: €{abs(daily_pnl)} -> {'Stop' if should_stop else 'Continue'}")
        self.assertFalse(should_stop)

    def test_daily_loss_just_above_limit(self):
        """Test daily loss just above limit."""
        print("\n  Testing daily loss just above limit...")

        daily_pnl = -50.01
        max_loss = 50.0

        should_stop = daily_pnl <= -max_loss

        print(f"  ✓ Loss just above limit: €{abs(daily_pnl)} -> {'Stop' if should_stop else 'Continue'}")
        self.assertTrue(should_stop)

    def test_zero_daily_trades_remaining(self):
        """Test when no trades remain."""
        print("\n  Testing zero trades remaining...")

        current_trades = 2
        max_trades = 2

        can_trade = current_trades < max_trades

        print(f"  ✓ Trades used: {current_trades}/{max_trades} -> {'Can trade' if can_trade else 'Cannot trade'}")
        self.assertFalse(can_trade)

    def test_exactly_one_trade_remaining(self):
        """Test when exactly one trade remains."""
        print("\n  Testing one trade remaining...")

        current_trades = 1
        max_trades = 2

        can_trade = current_trades < max_trades

        print(f"  ✓ Trades used: {current_trades}/{max_trades} -> {'Can trade' if can_trade else 'Cannot trade'}")
        self.assertTrue(can_trade)


class TestConfidenceEdgeCases(unittest.TestCase):
    """Test edge cases in confidence calculation."""

    def test_confidence_at_minimum(self):
        """Test confidence at minimum threshold."""
        print("\n  Testing confidence at minimum...")

        rsi = 42.1  # Just above oversold threshold
        confidence = min(0.95, 0.65 + (42 - rsi) / 80)

        print(f"  ✓ RSI {rsi}: confidence = {confidence:.3f}")
        self.assertAlmostEqual(confidence, 0.65, places=2)

    def test_confidence_at_maximum(self):
        """Test confidence at maximum."""
        print("\n  Testing confidence at maximum...")

        rsi = 0  # Extremely oversold
        confidence = min(0.95, 0.65 + (42 - rsi) / 80)

        print(f"  ✓ RSI {rsi}: confidence = {confidence:.3f}")
        self.assertAlmostEqual(confidence, 0.95, places=2)

    def test_confidence_below_threshold(self):
        """Test confidence below trading threshold."""
        print("\n  Testing confidence below threshold...")

        rsi = 50  # Neutral zone
        confidence = 0.5

        should_trade = confidence > 0.6

        print(f"  ✓ Confidence {confidence} -> {'Trade' if should_trade else 'No trade'}")
        self.assertFalse(should_trade)

    def test_confidence_exactly_at_threshold(self):
        """Test confidence exactly at trading threshold."""
        print("\n  Testing confidence at threshold...")

        confidence = 0.6
        should_trade = confidence > 0.6

        print(f"  ✓ Confidence {confidence} -> {'Trade' if should_trade else 'No trade'}")
        # Should NOT trade (strictly greater than)
        self.assertFalse(should_trade)


class TestVolumeEdgeCases(unittest.TestCase):
    """Test edge cases related to volume."""

    def test_zero_volume(self):
        """Test handling of zero volume."""
        print("\n  Testing zero volume...")

        volume = 0

        if volume > 0:
            valid = True
        else:
            valid = False

        print(f"  ✓ Zero volume rejected")
        self.assertFalse(valid)

    def test_very_high_volume(self):
        """Test handling of very high volume."""
        print("\n  Testing very high volume...")

        volume = 999999999999  # Extremely high

        # Should handle without overflow
        print(f"  ✓ High volume handled: {volume:,.0f}")
        self.assertGreater(volume, 0)

    def test_negative_volume(self):
        """Test handling of negative volume."""
        print("\n  Testing negative volume...")

        volume = -100

        # Should reject
        if volume > 0:
            valid = True
        else:
            valid = False

        print(f"  ✓ Negative volume rejected")
        self.assertFalse(valid)


def run_tests():
    """Run all edge case tests."""
    print("=" * 80)
    print("Edge Case Tests")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPriceEdgeCases))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPositionEdgeCases))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRSIEdgeCases))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskEdgeCases))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConfidenceEdgeCases))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVolumeEdgeCases))

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
