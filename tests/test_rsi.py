#!/usr/bin/env python3
"""
Unit tests for RSI calculation and trading signals.

Tests the core technical analysis logic used in live trading.
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
    print("Install: pip install pandas numpy")
    sys.exit(1)


class TestRSICalculation(unittest.TestCase):
    """Test RSI indicator calculation."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple price series
        self.prices_up = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                         120, 122, 124, 126, 128]  # Consistent uptrend
        self.prices_down = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102,
                           100, 98, 96, 94, 92]  # Consistent downtrend
        self.prices_flat = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                           100, 100, 100, 100, 100]  # Flat
        self.prices_volatile = [100, 105, 95, 110, 90, 115, 85, 120, 80, 125,
                               75, 130, 70, 135, 65]  # High volatility

    def calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate RSI indicator (same as live trading script)."""
        if len(prices) < period:
            return 50.0

        df = pd.DataFrame({'close': prices})
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def test_rsi_uptrend(self):
        """Test RSI in uptrend should be high (>70)."""
        rsi = self.calculate_rsi(self.prices_up)
        print(f"\n✓ Uptrend RSI: {rsi:.2f}")
        self.assertGreater(rsi, 70, "RSI should be high in uptrend")
        self.assertLessEqual(rsi, 100, "RSI cannot exceed 100")

    def test_rsi_downtrend(self):
        """Test RSI in downtrend should be low (<30)."""
        rsi = self.calculate_rsi(self.prices_down)
        print(f"✓ Downtrend RSI: {rsi:.2f}")
        self.assertLess(rsi, 30, "RSI should be low in downtrend")
        self.assertGreaterEqual(rsi, 0, "RSI cannot be below 0")

    def test_rsi_flat(self):
        """Test RSI in flat market should be around 50."""
        rsi = self.calculate_rsi(self.prices_flat)
        print(f"✓ Flat market RSI: {rsi:.2f}")
        self.assertAlmostEqual(rsi, 50, delta=10, msg="RSI should be around 50 in flat market")

    def test_rsi_volatile(self):
        """Test RSI handles volatility."""
        rsi = self.calculate_rsi(self.prices_volatile)
        print(f"✓ Volatile market RSI: {rsi:.2f}")
        self.assertGreaterEqual(rsi, 0, "RSI should be >= 0")
        self.assertLessEqual(rsi, 100, "RSI should be <= 100")

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data returns neutral."""
        short_prices = [100, 102, 104]  # Only 3 data points
        rsi = self.calculate_rsi(short_prices, period=14)
        print(f"✓ Insufficient data RSI: {rsi:.2f}")
        self.assertEqual(rsi, 50.0, "RSI should return 50 with insufficient data")

    def test_rsi_bounds(self):
        """Test RSI always stays within bounds."""
        # Random price series
        np.random.seed(42)
        for _ in range(10):
            prices = list(np.random.uniform(90, 110, 50))
            rsi = self.calculate_rsi(prices)
            self.assertGreaterEqual(rsi, 0, f"RSI {rsi} below 0")
            self.assertLessEqual(rsi, 100, f"RSI {rsi} above 100")
        print("✓ RSI bounds test passed for 10 random series")


class TestSignalGeneration(unittest.TestCase):
    """Test trading signal generation logic."""

    def generate_signal(self, rsi: float) -> dict:
        """Generate trading signal based on RSI (same as live trading)."""
        if rsi < 42:
            action = "BUY"
            confidence = min(0.95, 0.65 + (42 - rsi) / 80)
            reason = f"RSI oversold ({rsi:.1f})"
        elif rsi > 58:
            action = "SELL"
            confidence = min(0.95, 0.65 + (rsi - 58) / 80)
            reason = f"RSI overbought ({rsi:.1f})"
        else:
            action = "HOLD"
            confidence = 0.5
            reason = f"RSI neutral ({rsi:.1f})"

        return {'action': action, 'confidence': confidence, 'reason': reason, 'rsi': rsi}

    def test_signal_extreme_oversold(self):
        """Test signal at extreme oversold (RSI = 20)."""
        signal = self.generate_signal(20)
        print(f"\n✓ RSI 20 signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        self.assertEqual(signal['action'], 'BUY')
        self.assertGreater(signal['confidence'], 0.7)

    def test_signal_oversold_threshold(self):
        """Test signal just below oversold threshold (RSI = 41.9)."""
        signal = self.generate_signal(41.9)
        print(f"✓ RSI 41.9 signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        self.assertEqual(signal['action'], 'BUY')
        self.assertAlmostEqual(signal['confidence'], 0.65, places=2)

    def test_signal_at_oversold_threshold(self):
        """Test signal exactly at oversold threshold (RSI = 42)."""
        signal = self.generate_signal(42)
        print(f"✓ RSI 42 signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        self.assertEqual(signal['action'], 'HOLD')  # Exactly at threshold should HOLD

    def test_signal_neutral(self):
        """Test signal in neutral zone (RSI = 50)."""
        signal = self.generate_signal(50)
        print(f"✓ RSI 50 signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        self.assertEqual(signal['action'], 'HOLD')
        self.assertEqual(signal['confidence'], 0.5)

    def test_signal_overbought_threshold(self):
        """Test signal just above overbought threshold (RSI = 58.1)."""
        signal = self.generate_signal(58.1)
        print(f"✓ RSI 58.1 signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        self.assertEqual(signal['action'], 'SELL')
        self.assertAlmostEqual(signal['confidence'], 0.65, places=2)

    def test_signal_at_overbought_threshold(self):
        """Test signal exactly at overbought threshold (RSI = 58)."""
        signal = self.generate_signal(58)
        print(f"✓ RSI 58 signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        self.assertEqual(signal['action'], 'HOLD')  # Exactly at threshold should HOLD

    def test_signal_extreme_overbought(self):
        """Test signal at extreme overbought (RSI = 80)."""
        signal = self.generate_signal(80)
        print(f"✓ RSI 80 signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
        self.assertEqual(signal['action'], 'SELL')
        self.assertGreater(signal['confidence'], 0.8)

    def test_signal_confidence_bounds(self):
        """Test confidence never exceeds bounds."""
        for rsi in [0, 20, 40, 50, 60, 80, 100]:
            signal = self.generate_signal(rsi)
            self.assertGreaterEqual(signal['confidence'], 0.5)
            self.assertLessEqual(signal['confidence'], 0.95)
        print("✓ Confidence bounds test passed")


class TestPositionSizing(unittest.TestCase):
    """Test position sizing calculations."""

    def calculate_position_size(self, available_eur: float, max_position: float, price: float) -> float:
        """Calculate position size (same as live trading)."""
        position_size_eur = min(max_position, available_eur * 0.25)
        if position_size_eur < 10:
            return 0
        return position_size_eur / price

    def test_position_size_max_limit(self):
        """Test position size respects maximum limit."""
        # Large balance, should use max_position
        amount = self.calculate_position_size(available_eur=10000, max_position=300, price=50000)
        expected = 300 / 50000  # 0.006 BTC
        print(f"\n✓ Max limit: {amount:.6f} BTC (€300)")
        self.assertAlmostEqual(amount, expected, places=6)

    def test_position_size_balance_limited(self):
        """Test position size limited by balance."""
        # Small balance, should use 25% of balance
        amount = self.calculate_position_size(available_eur=500, max_position=300, price=50000)
        expected = (500 * 0.25) / 50000  # 0.0025 BTC
        print(f"✓ Balance limited: {amount:.6f} BTC (€125)")
        self.assertAlmostEqual(amount, expected, places=6)

    def test_position_size_insufficient_funds(self):
        """Test position size with insufficient funds."""
        amount = self.calculate_position_size(available_eur=30, max_position=300, price=50000)
        print(f"✓ Insufficient funds: {amount:.6f} BTC")
        self.assertEqual(amount, 0, "Should return 0 with insufficient funds")

    def test_position_size_minimum_threshold(self):
        """Test position size exactly at minimum threshold."""
        # Exactly €10 worth
        amount = self.calculate_position_size(available_eur=40, max_position=300, price=50000)
        expected = 10 / 50000
        print(f"✓ Minimum threshold: {amount:.6f} BTC (€10)")
        self.assertAlmostEqual(amount, expected, places=6)


def run_tests():
    """Run all tests."""
    print("=" * 80)
    print("RSI & Signal Generation Tests")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRSICalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionSizing))

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
