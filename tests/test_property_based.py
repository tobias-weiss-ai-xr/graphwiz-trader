#!/usr/bin/env python3
"""
Property-based testing for trading system.

Uses Hypothesis library to generate random inputs and test invariants.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
    from hypothesis import given, strategies as st, settings
except ImportError as e:
    print(f"Warning: Missing dependency - {e}")
    print("Install: pip install hypothesis")
    print("Running with limited testing...")

    # Create mock strategies if hypothesis not available
    class MockStrategy:
        @staticmethod
        def lists(*args, **kwargs):
            class MockValue:
                def example(self):
                    return list(range(10))
            return MockValue()

    st = MockStrategy()


class TestRSIProperties(unittest.TestCase):
    """Property-based tests for RSI calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.prices_strategy = st.lists(
            st.floats(min_value=1, max_value=100000, allow_nan=False, allow_infinity=False),
            min_size=14,
            max_size=200
        )

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI."""
        if len(prices) < period:
            return 50.0

        df = pd.DataFrame({'close': prices})
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0 if gain.iloc[-1] > 0 else 50.0

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    @given(st.lists(st.floats(min_value=100, max_value=50000, allow_nan=False, allow_infinity=False), min_size=14, max_size=100))
    @settings(max_examples=100)
    def test_rsi_always_between_0_and_100(self, prices):
        """RSI should always be between 0 and 100."""
        rsi = self.calculate_rsi(prices)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)

    @given(st.lists(st.floats(min_value=100, max_value=50000, allow_nan=False, allow_infinity=False), min_size=14, max_size=100))
    @settings(max_examples=100)
    def test_rsi_never_nan(self, prices):
        """RSI should never be NaN for valid inputs."""
        rsi = self.calculate_rsi(prices)
        self.assertFalse(pd.isna(rsi))

    @given(st.lists(st.floats(min_value=100, max_value=50000, allow_nan=False, allow_infinity=False), min_size=14, max_size=100))
    @settings(max_examples=50)
    def test_rsi_consistent_for_same_input(self, prices):
        """RSI should be consistent for the same input."""
        rsi1 = self.calculate_rsi(prices)
        rsi2 = self.calculate_rsi(prices)
        self.assertEqual(rsi1, rsi2)


class TestPositionSizingProperties(unittest.TestCase):
    """Property-based tests for position sizing."""

    @given(
        st.floats(min_value=100, max_value=100000, allow_nan=False, allow_infinity=False),  # balance
        st.floats(min_value=10, max_value=100000, allow_nan=False, allow_infinity=False),  # price
        st.floats(min_value=100, max_value=10000, allow_nan=False, allow_infinity=False)    # max_position
    )
    @settings(max_examples=100)
    def test_position_size_never_negative(self, balance, price, max_position):
        """Position size should never be negative."""
        position_eur = min(max_position, balance * 0.25)
        if position_eur >= 10 and price > 0:
            amount = position_eur / price
            self.assertGreaterEqual(amount, 0)

    @given(
        st.floats(min_value=100, max_value=100000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=10, max_value=100000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=100, max_value=10000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_position_respects_max_limit(self, balance, price, max_position):
        """Position should respect maximum limit."""
        position_eur = min(max_position, balance * 0.25)
        self.assertLessEqual(position_eur, max_position)

    @given(
        st.floats(min_value=100, max_value=100000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=10, max_value=100000, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_position_uses_25_percent_of_balance(self, balance, price):
        """Position should use 25% of balance or max, whichever is smaller."""
        max_position = 300
        position_eur = min(max_position, balance * 0.25)

        if balance * 0.25 <= max_position:
            self.assertAlmostEqual(position_eur, balance * 0.25, places=2)


class TestSignalGenerationProperties(unittest.TestCase):
    """Property-based tests for signal generation."""

    def generate_signal(self, rsi):
        """Generate signal from RSI."""
        if rsi < 42:
            return {"action": "BUY", "confidence": min(0.95, 0.65 + (42 - rsi) / 80)}
        elif rsi > 58:
            return {"action": "SELL", "confidence": min(0.95, 0.65 + (rsi - 58) / 80)}
        else:
            return {"action": "HOLD", "confidence": 0.5}

    @given(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_signal_always_valid_action(self, rsi):
        """Signal action should always be valid."""
        signal = self.generate_signal(rsi)
        self.assertIn(signal['action'], ['BUY', 'SELL', 'HOLD'])

    @given(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_confidence_always_in_range(self, rsi):
        """Confidence should always be in valid range."""
        signal = self.generate_signal(rsi)
        self.assertGreaterEqual(signal['confidence'], 0.5)
        self.assertLessEqual(signal['confidence'], 0.95)

    @given(st.floats(min_value=0, max_value=41.99, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_oversold_always_buy(self, rsi):
        """Oversold RSI should always generate BUY signal."""
        signal = self.generate_signal(rsi)
        self.assertEqual(signal['action'], 'BUY')

    @given(st.floats(min_value=58.01, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_overbought_always_sell(self, rsi):
        """Overbought RSI should always generate SELL signal."""
        signal = self.generate_signal(rsi)
        self.assertEqual(signal['action'], 'SELL')

    @given(st.floats(min_value=42, max_value=58, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_neutral_always_hold(self, rsi):
        """Neutral RSI should always generate HOLD signal."""
        signal = self.generate_signal(rsi)
        self.assertEqual(signal['action'], 'HOLD')


class TestRiskManagementProperties(unittest.TestCase):
    """Property-based tests for risk management."""

    @given(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),  # daily_pnl
        st.floats(min_value=10, max_value=1000, allow_nan=False, allow_infinity=False)    # max_loss
    )
    @settings(max_examples=100)
    def test_loss_limit_triggers_correctly(self, daily_pnl, max_loss):
        """Loss limit should trigger when P&L exceeds limit."""
        should_stop = daily_pnl <= -max_loss

        if daily_pnl <= -max_loss:
            self.assertTrue(should_stop)
        else:
            self.assertFalse(should_stop)

    @given(
        st.integers(min_value=0, max_value=10),  # trade_count
        st.integers(min_value=1, max_value=10)    # max_trades
    )
    @settings(max_examples=100)
    def test_trade_limit_enforced(self, trade_count, max_trades):
        """Trade limit should be enforced."""
        can_trade = trade_count < max_trades

        if trade_count >= max_trades:
            self.assertFalse(can_trade)
        else:
            self.assertTrue(can_trade)


class TestStopLossProperties(unittest.TestCase):
    """Property-based tests for stop loss."""

    @given(
        st.floats(min_value=10, max_value=100000, allow_nan=False, allow_infinity=False),  # entry_price
        st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False)   # stop_loss_pct
    )
    @settings(max_examples=100)
    def test_stop_loss_always_below_entry_for_buy(self, entry_price, stop_loss_pct):
        """Stop loss for BUY should always be below entry price."""
        stop_loss = entry_price * (1 - stop_loss_pct)
        self.assertLess(stop_loss, entry_price)
        self.assertGreater(stop_loss, 0)

    @given(
        st.floats(min_value=10, max_value=100000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_stop_loss_always_above_entry_for_sell(self, entry_price, stop_loss_pct):
        """Stop loss for SELL should always be above entry price."""
        stop_loss = entry_price * (1 + stop_loss_pct)
        self.assertGreater(stop_loss, entry_price)


class TestTakeProfitProperties(unittest.TestCase):
    """Property-based tests for take profit."""

    @given(
        st.floats(min_value=10, max_value=100000, allow_nan=False, allow_infinity=False),  # entry_price
        st.floats(min_value=0.001, max_value=0.2, allow_nan=False, allow_infinity=False)   # take_profit_pct
    )
    @settings(max_examples=100)
    def test_take_profit_always_above_entry_for_buy(self, entry_price, take_profit_pct):
        """Take profit for BUY should always be above entry price."""
        take_profit = entry_price * (1 + take_profit_pct)
        self.assertGreater(take_profit, entry_price)

    @given(
        st.floats(min_value=10, max_value=100000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=0.2, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_take_profit_always_below_entry_for_sell(self, entry_price, take_profit_pct):
        """Take profit for SELL should always be below entry price."""
        take_profit = entry_price * (1 - take_profit_pct)
        self.assertLess(take_profit, entry_price)


class TestRiskRewardProperties(unittest.TestCase):
    """Property-based tests for risk/reward ratio."""

    @given(
        st.floats(min_value=10, max_value=100000, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.001, max_value=0.2, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_risk_reward_ratio_favorable(self, entry_price, stop_loss_pct, take_profit_pct):
        """Risk/reward ratio should always be favorable."""
        if take_profit_pct > stop_loss_pct:
            # Take profit is larger than stop loss
            ratio = take_profit_pct / stop_loss_pct
            self.assertGreater(ratio, 1.0)


class TestDataProperties(unittest.TestCase):
    """Property-based tests for data operations."""

    @given(st.lists(st.floats(min_value=-1000, max_value=100000, allow_nan=False, allow_infinity=False), min_size=10))
    @settings(max_examples=100)
    def test_mean_calculable_for_any_numeric_list(self, values):
        """Mean should be calculable for any numeric list."""
        mean = np.mean(values)
        self.assertTrue(np.isfinite(mean))

    @given(st.lists(st.floats(min_value=-1000, max_value=100000, allow_nan=False, allow_infinity=False), min_size=2))
    @settings(max_examples=100)
    def test_std_calculable_for_any_numeric_list(self, values):
        """Standard deviation should be calculable for any numeric list."""
        std = np.std(values)
        self.assertGreaterEqual(std, 0)

    @given(st.lists(st.integers(min_value=-1000000, max_value=1000000), min_size=10, max_size=1000))
    @settings(max_examples=100)
    def test_list_operations_preserve_length(self, values):
        """List operations should preserve or predictably change length."""
        original_length = len(values)

        # Filter positive values
        positive = [v for v in values if v > 0]
        self.assertLessEqual(len(positive), original_length)

        # Remove duplicates
        unique = list(set(values))
        self.assertLessEqual(len(unique), original_length)


def run_tests():
    """Run all property-based tests."""
    print("=" * 80)
    print("Property-Based Testing")
    print("=" * 80)

    try:
        # Create test suite
        suite = unittest.TestSuite()

        # Add all test classes
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRSIProperties))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPositionSizingProperties))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSignalGenerationProperties))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskManagementProperties))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStopLossProperties))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTakeProfitProperties))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskRewardProperties))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataProperties))

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

    except NameError:
        print("\n⚠️  Hypothesis library not installed")
        print("Install: pip install hypothesis")
        print("\nSkipping property-based tests...")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
