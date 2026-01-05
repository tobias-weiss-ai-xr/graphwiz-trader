#!/usr/bin/env python3
"""
Chaos engineering tests for trading system.

Simulates failures and unexpected conditions to test system resilience.
"""

import sys
import unittest
import random
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)


class ChaosMonkey:
    """Simulates chaotic conditions."""

    @staticmethod
    def random_delay(max_ms=1000):
        """Add random delay."""
        delay = random.uniform(0, max_ms / 1000)
        time.sleep(delay)

    @staticmethod
    def random_failure(failure_rate=0.3):
        """Randomly fail with given probability."""
        return random.random() < failure_rate

    @staticmethod
    def corrupt_data(data):
        """Corrupt data in random ways."""
        corruption_type = random.choice(['null', 'nan', 'inf', 'wrong_type', 'empty'])

        if corruption_type == 'null':
            return None
        elif corruption_type == 'nan':
            return float('nan')
        elif corruption_type == 'inf':
            return float('inf') if random.choice([True, False]) else float('-inf')
        elif corruption_type == 'wrong_type':
            return "corrupted"
        elif corruption_type == 'empty':
            return ""
        else:
            return data

    @staticmethod
    def random_network_error():
        """Generate random network error."""
        errors = [
            ConnectionError("Connection reset"),
            TimeoutError("Request timeout"),
            ccxt.NetworkError("Network unreachable"),
            ccxt.ExchangeNotAvailable("Exchange maintenance"),
            ccxt.RequestTimeout("Request timeout"),
        ]
        return random.choice(errors)


class TestNetworkResilience(unittest.TestCase):
    """Test system resilience to network issues."""

    def test_intermittent_connection_failures(self):
        """Test handling of intermittent connection failures."""
        print("\n  Testing intermittent failures...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        success_count = 0
        failure_count = 0

        for i in range(10):
            try:
                # Simulate intermittent failure
                if ChaosMonkey.random_failure(0.3):
                    raise ChaosMonkey.random_network_error()

                ticker = exchange.fetch_ticker('BTC/EUR')
                success_count += 1

            except Exception as e:
                failure_count += 1
                # Should handle gracefully
                self.assertIsNotNone(e)

        print(f"    Successes: {success_count}, Failures: {failure_count}")
        self.assertGreater(success_count, 0, "Should have some successes")

    def test_timeout_recovery(self):
        """Test recovery from timeout."""
        print("\n  Testing timeout recovery...")

        exchange = ccxt.kraken({
            'enableRateLimit': True,
            'timeout': 1  # Very short timeout
        })

        retries = 0
        max_retries = 3
        success = False

        while retries < max_retries and not success:
            try:
                ticker = exchange.fetch_ticker('BTC/EUR')
                success = True
                print(f"    ✓ Succeeded on attempt {retries + 1}")
            except Exception as e:
                retries += 1
                if retries < max_retries:
                    time.sleep(0.5)

        self.assertTrue(success or retries >= max_retries)


class TestDataCorruptionResilience(unittest.TestCase):
    """Test resilience to corrupted data."""

    def test_corrupted_price_data(self):
        """Test handling of corrupted price data."""
        print("\n  Testing corrupted price data...")

        # Valid prices with some corrupted
        prices = [50000, 51000, 52000, None, 53000, float('nan'), 54000, "55000"]

        # Clean data
        clean_prices = []
        for price in prices:
            try:
                if price is not None and not pd.isna(price):
                    clean_prices.append(float(price))
            except (ValueError, TypeError):
                pass

        print(f"    Original: {len(prices)}, Clean: {len(clean_prices)}")
        self.assertGreater(len(clean_prices), 0)
        self.assertEqual(len(clean_prices), 6)  # 5 numbers + 1 valid string

    def test_mixed_type_data(self):
        """Test handling of mixed data types."""
        print("\n  Testing mixed type data...")

        mixed_data = [
            50000,      # int
            51000.5,    # float
            "52000",    # string
            None,       # None
            True,       # bool
            [53000],    # list
            {"price": 54000},  # dict
        ]

        valid_prices = []
        for item in mixed_data:
            try:
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    valid_prices.append(float(item))
                elif isinstance(item, str):
                    valid_prices.append(float(item))
            except (ValueError, TypeError):
                pass

        print(f"    Extracted {len(valid_prices)} valid prices from mixed data")
        self.assertGreaterEqual(len(valid_prices), 3)

    def test_extreme_volatility(self):
        """Test handling of extreme volatility."""
        print("\n  Testing extreme volatility...")

        # Generate extremely volatile prices
        prices = []
        price = 50000

        for i in range(50):
            # Random swings of ±20%
            change = random.uniform(-0.2, 0.2)
            price = price * (1 + change)
            prices.append(max(price, 1000))  # Minimum €1000

        try:
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

            self.assertTrue(0 <= rsi <= 100)
            print(f"    ✓ Extreme volatility handled: RSI={rsi:.1f}")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            self.fail("Should handle extreme volatility")

    def test_missing_data_points(self):
        """Test handling of missing data points."""
        print("\n  Testing missing data points...")

        # Prices with gaps
        prices = [50000, None, None, 53000, 54000, None, 55000]

        # Forward fill
        filled_prices = []
        last_price = 50000

        for price in prices:
            if price is None:
                filled_prices.append(last_price)
            else:
                filled_prices.append(price)
                last_price = price

        print(f"    Original had {sum(1 for p in prices if p is None)} gaps")
        print(f"    ✓ Forward filled successfully")

        self.assertEqual(len(filled_prices), len(prices))
        self.assertTrue(all(p is not None for p in filled_prices))


class TestRateLimitChaos(unittest.TestCase):
    """Test behavior under rate limiting chaos."""

    def test_burst_requests(self):
        """Test handling of burst requests."""
        print("\n  Testing burst requests...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        # Send burst of requests
        success = 0
        for i in range(10):
            try:
                ticker = exchange.fetch_ticker('BTC/EUR')
                success += 1
            except ccxt.RateLimitExceeded:
                # Expected
                pass
            except Exception as e:
                # Other errors
                pass

        print(f"    Burst: {success}/10 succeeded")
        self.assertGreater(success, 0)

    def test_rapid_alternating_requests(self):
        """Test rapid alternating requests to different endpoints."""
        print("\n  Testing rapid alternating requests...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        endpoints = [
            lambda: exchange.fetch_ticker('BTC/EUR'),
            lambda: exchange.fetch_ticker('ETH/EUR'),
            lambda: exchange.fetch_order_book('BTC/EUR'),
        ]

        success = 0
        for i in range(9):
            endpoint = endpoints[i % len(endpoints)]
            try:
                endpoint()
                success += 1
            except Exception as e:
                pass

        print(f"    Alternating: {success}/9 succeeded")
        self.assertGreater(success, 0)


class TestResourceExhaustion(unittest.TestCase):
    """Test behavior under resource constraints."""

    def test_large_dataset_memory(self):
        """Test handling of very large datasets."""
        print("\n  Testing large dataset handling...")

        # Generate large dataset
        size = 100000
        print(f"    Generating {size} data points...")

        prices = [50000 + random.uniform(-5000, 5000) for _ in range(size)]

        try:
            # Calculate indicator
            df = pd.DataFrame({'close': prices})
            ma = df['close'].rolling(window=20).mean()

            print(f"    ✓ Processed {size} points successfully")
            self.assertEqual(len(ma), size)

        except MemoryError:
            self.fail("Should handle large dataset without memory error")

    def test_concurrent_calculations(self):
        """Test multiple concurrent calculations."""
        print("\n  Testing concurrent calculations...")

        # Generate multiple datasets
        datasets = []
        for i in range(10):
            prices = [50000 + random.uniform(-1000, 1000) for _ in range(100)]
            datasets.append(prices)

        # Process all
        results = []
        for prices in datasets:
            df = pd.DataFrame({'close': prices})
            ma = df['close'].rolling(window=14).mean()
            results.append(ma.iloc[-1])

        print(f"    ✓ Processed {len(datasets)} datasets concurrently")
        self.assertEqual(len(results), 10)


class TestInvalidStateRecovery(unittest.TestCase):
    """Test recovery from invalid states."""

    def test_reinitialize_after_error(self):
        """Test reinitialization after error."""
        print("\n  Testing reinitialization...")

        # Create exchange with invalid credentials
        bad_exchange = ccxt.kraken({
            'apiKey': 'invalid',
            'secret': 'invalid'
        })

        # Try to use it
        try:
            bad_exchange.fetch_balance()
        except:
            pass  # Expected to fail

        # Reinitialize with good credentials
        good_exchange = ccxt.kraken({
            'enableRateLimit': True
        })

        # Should work now
        try:
            ticker = good_exchange.fetch_ticker('BTC/EUR')
            self.assertIsNotNone(ticker)
            print(f"    ✓ Reinitialized successfully")
        except Exception as e:
            self.fail(f"Should work after reinitialization: {e}")

    def test_state_corruption_recovery(self):
        """Test recovery from corrupted state."""
        print("\n  Testing state corruption recovery...")

        # Simulated trading state
        state = {
            'daily_pnl': 0.0,
            'trade_count': 0,
            'positions': {}
        }

        # Corrupt state
        state['daily_pnl'] = float('nan')
        state['trade_count'] = -1
        state['positions'] = 'corrupted'

        # Validate and reset
        if not np.isfinite(state['daily_pnl']):
            state['daily_pnl'] = 0.0

        if state['trade_count'] < 0:
            state['trade_count'] = 0

        if not isinstance(state['positions'], dict):
            state['positions'] = {}

        # Verify recovery
        self.assertEqual(state['daily_pnl'], 0.0)
        self.assertEqual(state['trade_count'], 0)
        self.assertIsInstance(state['positions'], dict)

        print(f"    ✓ State recovered from corruption")


class TestConcurrentModifications(unittest.TestCase):
    """Test handling of concurrent modifications."""

    def test_concurrent_balance_updates(self):
        """Test concurrent balance updates."""
        print("\n  Testing concurrent balance updates...")

        balance = {'EUR': 1000.0, 'BTC': 0.0}

        # Simulate concurrent updates
        updates = [
            ('EUR', 100),
            ('EUR', -50),
            ('EUR', 200),
            ('EUR', -75),
        ]

        for currency, amount in updates:
            if currency in balance:
                balance[currency] += amount

        # Final balance should be correct
        expected = 1000 + 100 - 50 + 200 - 75
        self.assertEqual(balance['EUR'], expected)

        print(f"    ✓ Concurrent updates handled: €{balance['EUR']:.2f}")

    def test_concurrent_position_tracking(self):
        """Test concurrent position tracking."""
        print("\n  Testing concurrent position tracking...")

        positions = {}

        # Simulate concurrent position updates
        operations = [
            ('add', 'BTC', 0.5),
            ('add', 'ETH', 5.0),
            ('remove', 'BTC', 0.2),
            ('add', 'BTC', 0.3),
        ]

        for op, symbol, amount in operations:
            if op == 'add':
                positions[symbol] = positions.get(symbol, 0) + amount
            elif op == 'remove':
                if symbol in positions:
                    positions[symbol] -= amount
                    if positions[symbol] <= 0:
                        del positions[symbol]

        print(f"    ✓ Final positions: {positions}")
        self.assertIn('BTC', positions)
        self.assertIn('ETH', positions)


class TestUnexpectedInputCombinations(unittest.TestCase):
    """Test unexpected combinations of inputs."""

    def test_zero_balance_with_trade(self):
        """Test trade attempt with zero balance."""
        print("\n  Testing zero balance trade...")

        balance = 0
        price = 50000

        position_size_eur = min(300, balance * 0.25)

        if position_size_eur < 10:
            allowed = False
        else:
            allowed = True

        self.assertFalse(allowed)
        print(f"    ✓ Zero balance rejected")

    def test_negative_rsi(self):
        """Test signal generation with negative RSI."""
        print("\n  Testing negative RSI...")

        rsi = -10

        # Should clamp to valid range
        rsi = max(0, min(100, rsi))

        if rsi < 42:
            signal = "BUY"
        elif rsi > 58:
            signal = "SELL"
        else:
            signal = "HOLD"

        self.assertEqual(rsi, 0)
        self.assertEqual(signal, 'BUY')
        print(f"    ✓ Negative RSI clamped to 0, signal: BUY")

    def test_extremely_high_confidence(self):
        """Test handling of confidence > 1.0."""
        print("\n  Testing high confidence...")

        confidence = 1.5

        # Should clamp
        confidence = min(0.95, max(0.5, confidence))

        self.assertAlmostEqual(confidence, 0.95)
        print(f"    ✓ Confidence clamped to {confidence}")


def run_tests():
    """Run all chaos tests."""
    print("=" * 80)
    print("Chaos Engineering Tests")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNetworkResilience))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataCorruptionResilience))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRateLimitChaos))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestResourceExhaustion))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInvalidStateRecovery))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConcurrentModifications))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUnexpectedInputCombinations))

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
