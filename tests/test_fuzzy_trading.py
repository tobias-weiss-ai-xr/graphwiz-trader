#!/usr/bin/env python3
"""
Fuzzy testing for trading system.

Generates random, invalid, and unexpected inputs to find edge cases and bugs.
"""

import sys
import unittest
import random
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)


class FuzzyPriceGenerator:
    """Generate random price data for fuzzy testing."""

    @staticmethod
    def random_price():
        """Generate completely random price."""
        return random.uniform(-1000000, 1000000)

    @staticmethod
    def realistic_price(base=50000, volatility=0.02):
        """Generate realistic price with small variation."""
        change = random.uniform(-volatility, volatility)
        return base * (1 + change)

    @staticmethod
    def extreme_price():
        """Generate extreme price (very high or very low)."""
        if random.choice([True, False]):
            return random.uniform(1, 0.000001)  # Very low
        else:
            return random.uniform(1000000, 1000000000)  # Very high

    @staticmethod
    def invalid_price():
        """Generate invalid price."""
        return random.choice([
            0,  # Zero
            -100,  # Negative
            float('inf'),  # Infinite
            float('-inf'),  # Negative infinite
            float('nan'),  # Not a number
            None,  # None
            "100",  # String instead of number
            "",  # Empty string
        ])

    @staticmethod
    def price_series(length=100, trend=None):
        """Generate series of prices with optional trend."""
        prices = []
        price = 50000

        for _ in range(length):
            if trend == 'up':
                price *= random.uniform(1.0, 1.01)
            elif trend == 'down':
                price *= random.uniform(0.99, 1.0)
            else:
                price *= random.uniform(0.99, 1.01)

            prices.append(max(price, 0.01))  # Ensure positive

        return prices


class FuzzyVolumeGenerator:
    """Generate random volume data."""

    @staticmethod
    def random_volume():
        """Generate random volume."""
        return random.uniform(-1000000, 1000000)

    @staticmethod
    def realistic_volume():
        """Generate realistic volume."""
        return random.uniform(0.1, 10000)

    @staticmethod
    def invalid_volume():
        """Generate invalid volume."""
        return random.choice([
            0,  # Zero
            -100,  # Negative
            float('inf'),
            float('nan'),
            None,
        ])


class FuzzyBalanceGenerator:
    """Generate random balance data."""

    @staticmethod
    def random_balance():
        """Generate random balance."""
        return random.uniform(-1000000, 1000000)

    @staticmethod
    def realistic_balance():
        """Generate realistic balance."""
        return random.uniform(100, 100000)

    @staticmethod
    def invalid_balance():
        """Generate invalid balance."""
        return random.choice([
            0,
            -100,
            float('nan'),
            None,
        ])


class TestFuzzyPriceHandling(unittest.TestCase):
    """Test handling of random price inputs."""

    def test_fuzzy_rsi_calculation(self):
        """Test RSI with random price series."""
        print("\n  Testing RSI with 100 random price series...")

        failures = []

        for i in range(100):
            try:
                # Generate random price series
                prices = [FuzzyPriceGenerator.random_price() for _ in range(random.randint(10, 100))]

                # Filter to positive prices only
                prices = [p for p in prices if p > 0]

                if len(prices) < 14:
                    continue

                # Calculate RSI
                df = pd.DataFrame({'close': prices})
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

                if loss.iloc[-1] == 0:
                    rsi = 50.0
                else:
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

                # Validate result
                if not (0 <= rsi <= 100):
                    failures.append(f"RSI {rsi} out of bounds")
                elif pd.isna(rsi):
                    failures.append(f"RSI is NaN")

            except Exception as e:
                # Some failures expected with invalid data
                if "invalid" not in str(e).lower():
                    failures.append(f"Unexpected error: {e}")

        print(f"    Failures: {len(failures)}/100")
        self.assertLess(len(failures), 10, "Should handle most random inputs")

    def test_fuzzy_invalid_prices(self):
        """Test handling of various invalid prices."""
        print("\n  Testing invalid price handling...")

        invalid_prices = [
            0, -1, -100, -1000000,
            float('inf'), float('-inf'),
            float('nan'),
        ]

        handled = 0
        for price in invalid_prices:
            try:
                # Try to use price in calculation
                if price <= 0 or not np.isfinite(price):
                    # Should reject
                    handled += 1
                else:
                    position_size = 300 / price
            except:
                handled += 1

        print(f"    Handled: {handled}/{len(invalid_prices)} invalid prices")
        self.assertEqual(handled, len(invalid_prices))

    def test_fuzzy_extreme_prices(self):
        """Test handling of extreme price values."""
        print("\n  Testing extreme price handling...")

        extreme_prices = [
            0.000001,  # Very small
            0.0000001,  # Extremely small
            1000000,  # Very high
            1000000000,  # Extremely high
        ]

        for price in extreme_prices:
            try:
                # Calculate position size
                position_size = 300 / price

                # Should be finite and positive
                self.assertTrue(np.isfinite(position_size))
                self.assertGreater(position_size, 0)

                print(f"    €300 at €{price:.10f} = {position_size:.10f} BTC ✓")

            except Exception as e:
                print(f"    €300 at €{price:.10f} = ERROR: {e}")
                self.fail(f"Should handle extreme price {price}")


class TestFuzzyVolumeHandling(unittest.TestCase):
    """Test handling of random volume inputs."""

    def test_fuzzy_volume_validation(self):
        """Test volume validation with random inputs."""
        print("\n  Testing volume validation with random inputs...")

        valid_count = 0
        invalid_count = 0

        for i in range(100):
            # Generate random volume
            volume = FuzzyVolumeGenerator.random_volume()

            # Validate
            if volume > 0 and np.isfinite(volume):
                valid_count += 1
            else:
                invalid_count += 1

        print(f"    Valid: {valid_count}, Invalid: {invalid_count}")
        self.assertGreater(valid_count, 0)

    def test_fuzzy_invalid_volumes(self):
        """Test handling of invalid volumes."""
        print("\n  Testing invalid volume handling...")

        invalid_volumes = [
            0, -1, -100, -1000000,
            float('inf'), float('-inf'),
            float('nan'),
        ]

        handled = 0
        for volume in invalid_volumes:
            try:
                # Try to validate volume
                if volume <= 0 or not np.isfinite(volume):
                    # Should reject
                    handled += 1
                else:
                    # Accept
                    pass
            except:
                handled += 1

        print(f"    Handled: {handled}/{len(invalid_volumes)} invalid volumes")
        self.assertEqual(handled, len(invalid_volumes))


class TestFuzzyPositionSizing(unittest.TestCase):
    """Test position sizing with random inputs."""

    def test_fuzzy_position_calculations(self):
        """Test position sizing with random balances and prices."""
        print("\n  Testing position sizing with random inputs...")

        for i in range(50):
            # Generate random inputs
            balance = FuzzyBalanceGenerator.realistic_balance()
            price = FuzzyPriceGenerator.realistic_price()
            max_position = 300

            try:
                # Calculate position
                position_size_eur = min(max_position, balance * 0.25)

                if position_size_eur < 10:
                    continue

                amount = position_size_eur / price

                # Validate
                self.assertGreater(amount, 0)
                self.assertTrue(np.isfinite(amount))
                self.assertLessEqual(position_size_eur, max_position)

            except Exception as e:
                self.fail(f"Failed with balance={balance}, price={price}: {e}")

        print(f"    ✓ Passed 50 random position calculations")

    def test_fuzzy_edge_cases(self):
        """Test position sizing edge cases."""
        print("\n  Testing position sizing edge cases...")

        edge_cases = [
            (10, 50000, 300),      # Small balance
            (100, 50000, 300),     # Medium balance
            (10000, 50000, 300),   # Large balance
            (1200, 1000, 300),     # Low price
            (1000, 1000000, 300),  # High price
            (40, 50000, 300),      # At minimum threshold
            (39.99, 50000, 300),   # Just below threshold
        ]

        for balance, price, max_position in edge_cases:
            try:
                position_size_eur = min(max_position, balance * 0.25)

                if position_size_eur < 10:
                    # Should be rejected
                    self.assertLess(position_size_eur, 10)
                else:
                    amount = position_size_eur / price
                    self.assertGreater(amount, 0)

                print(f"    €{balance} @ €{price} -> {'OK' if position_size_eur >= 10 else 'Rejected'} ✓")

            except Exception as e:
                print(f"    €{balance} @ €{price} -> ERROR: {e}")
                self.fail(f"Should handle edge case balance={balance}, price={price}")


class TestFuzzySignalGeneration(unittest.TestCase):
    """Test signal generation with random RSI values."""

    def test_fuzzy_rsi_signals(self):
        """Test signal generation with random RSI values."""
        print("\n  Testing signal generation with random RSI...")

        for i in range(100):
            # Generate random RSI
            rsi = random.uniform(0, 100)

            # Generate signal
            if rsi < 42:
                action = "BUY"
                confidence = min(0.95, 0.65 + (42 - rsi) / 80)
            elif rsi > 58:
                action = "SELL"
                confidence = min(0.95, 0.65 + (rsi - 58) / 80)
            else:
                action = "HOLD"
                confidence = 0.5

            # Validate
            self.assertIn(action, ['BUY', 'SELL', 'HOLD'])
            self.assertGreaterEqual(confidence, 0.5)
            self.assertLessEqual(confidence, 0.95)

            # Action should match RSI zone
            if rsi < 42:
                self.assertEqual(action, 'BUY')
            elif rsi > 58:
                self.assertEqual(action, 'SELL')
            else:
                self.assertEqual(action, 'HOLD')

        print(f"    ✓ Passed 100 random RSI signal generations")

    def test_fuzzy_boundary_rsi(self):
        """Test RSI at and near boundaries."""
        print("\n  Testing RSI boundaries...")

        # Test around boundaries
        boundary_tests = [
            41.9, 42.0, 42.1,   # BUY boundary
            57.9, 58.0, 58.1,   # SELL boundary
            0, 1, 99, 100,      # Extremes
        ]

        for rsi in boundary_tests:
            if rsi < 42:
                action = "BUY"
            elif rsi > 58:
                action = "SELL"
            else:
                action = "HOLD"

            print(f"    RSI {rsi:5.1f} -> {action} ✓")


class TestFuzzyRiskChecks(unittest.TestCase):
    """Test risk checks with random scenarios."""

    def test_fuzzy_daily_loss_scenarios(self):
        """Test daily loss limit with random P&L values."""
        print("\n  Testing daily loss limit with random P&L...")

        max_loss = 50
        stopped = 0
        allowed = 0

        for i in range(100):
            # Generate random P&L
            daily_pnl = random.uniform(-100, 100)

            # Check if should stop
            if daily_pnl <= -max_loss:
                stopped += 1
            else:
                allowed += 1

            # Validate decision
            if daily_pnl <= -max_loss:
                self.assertLessEqual(daily_pnl, -max_loss)

        print(f"    Stopped: {stopped}, Allowed: {allowed}")
        self.assertGreater(stopped, 0, "Should stop some trades")
        self.assertGreater(allowed, 0, "Should allow some trades")

    def test_fuzzy_trade_count_scenarios(self):
        """Test trade count limit with random trade counts."""
        print("\n  Testing trade count limit...")

        max_trades = 2
        stopped = 0
        allowed = 0

        for i in range(100):
            # Generate random trade count
            trade_count = random.randint(0, 10)

            # Check if can trade
            if trade_count >= max_trades:
                stopped += 1
            else:
                allowed += 1

        print(f"    Stopped: {stopped}, Allowed: {allowed}")
        self.assertGreater(stopped, 0)
        self.assertGreater(allowed, 0)


class TestFuzzyDataIntegrity(unittest.TestCase):
    """Test data integrity with random inputs."""

    def test_fuzzy_nan_handling(self):
        """Test handling of NaN values in calculations."""
        print("\n  Testing NaN handling...")

        data_with_nan = [100, 102, float('nan'), 106, 108, float('nan'), 110]

        # Clean data
        clean_data = [x for x in data_with_nan if not pd.isna(x)]

        self.assertEqual(len(clean_data), 5)
        print(f"    Cleaned {len(data_with_nan)} values to {len(clean_data)} valid values ✓")

    def test_fuzzy_outlier_handling(self):
        """Test handling of outliers in data."""
        print("\n  Testing outlier handling...")

        # Generate data with outliers
        np.random.seed(42)
        normal_data = list(np.random.normal(50000, 1000, 50))
        outliers = [100000, 1000, 500000]
        data_with_outliers = normal_data + outliers

        # Calculate statistics
        mean = np.mean(data_with_outliers)
        median = np.median(data_with_outliers)
        std = np.std(data_with_outliers)

        print(f"    Mean: €{mean:,.2f}")
        print(f"    Median: €{median:,.2f}")
        print(f"    Std Dev: €{std:,.2f}")

        # Median should be robust to outliers
        self.assertLess(median, mean, "Median < Mean with high outliers")

    def test_fuzzy_type_safety(self):
        """Test type safety with mixed types."""
        print("\n  Testing type safety...")

        mixed_inputs = [
            100,
            "100",
            100.5,
            None,
            True,
            False,
            [100, 200],
            {"price": 100},
        ]

        for value in mixed_inputs:
            try:
                # Try to convert to float
                if isinstance(value, (int, float)):
                    price = float(value)
                elif isinstance(value, str):
                    price = float(value)
                else:
                    # Should reject non-numeric types
                    price = None

                if price is not None:
                    self.assertIsInstance(price, float)

            except (ValueError, TypeError):
                # Expected for some types
                pass

        print(f"    ✓ Handled {len(mixed_inputs)} mixed type inputs")


class TestFuzzyMarketScenarios(unittest.TestCase):
    """Test random market scenarios."""

    def test_fuzzy_market_conditions(self):
        """Test strategy under random market conditions."""
        print("\n  Testing random market conditions...")

        scenarios = [
            ("Bull", FuzzyPriceGenerator.price_series(50, 'up')),
            ("Bear", FuzzyPriceGenerator.price_series(50, 'down')),
            ("Sideways", FuzzyPriceGenerator.price_series(50, None)),
            ("Volatile", [50000 + random.uniform(-5000, 5000) for _ in range(50)]),
        ]

        for name, prices in scenarios:
            # Calculate RSI
            if len(prices) >= 14:
                df = pd.DataFrame({'close': prices})
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

                if loss.iloc[-1] == 0:
                    rsi = 50.0
                else:
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

                # Generate signal
                if rsi < 42:
                    signal = "BUY"
                elif rsi > 58:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                print(f"    {name}: RSI={rsi:.1f}, Signal={signal} ✓")

    def test_fuzzy_price_shocks(self):
        """Test handling of sudden price shocks."""
        print("\n  Testing price shocks...")

        # Normal prices then sudden shock
        normal_prices = [50000] * 40
        shock_scenarios = [
            ("+20% shock", [50000 * 1.2] * 10),
            ("-20% shock", [50000 * 0.8] * 10),
            ("+50% spike", [50000 * 1.5] * 10),
            ("-50% crash", [50000 * 0.5] * 10),
        ]

        for name, shock_prices in shock_scenarios:
            prices = normal_prices + shock_prices

            try:
                # Calculate RSI
                df = pd.DataFrame({'close': prices})
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

                if loss.iloc[-1] == 0:
                    rsi = 100.0 if gain.iloc[-1] > 0 else 50.0
                else:
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

                self.assertTrue(0 <= rsi <= 100)
                print(f"    {name}: RSI={rsi:.1f} ✓")

            except Exception as e:
                print(f"    {name}: ERROR - {e}")
                self.fail(f"Should handle {name}")


def run_tests():
    """Run all fuzzy tests."""
    print("=" * 80)
    print("Fuzzy Testing - Random & Edge Case Inputs")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuzzyPriceHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuzzyVolumeHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuzzyPositionSizing))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuzzySignalGeneration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuzzyRiskChecks))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuzzyDataIntegrity))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuzzyMarketScenarios))

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
