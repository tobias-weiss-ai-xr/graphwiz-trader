#!/usr/bin/env python3
"""
Stress tests for trading system.

Tests system under heavy load and extreme conditions.
"""

import sys
import unittest
import time
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)


class TestAPILoad(unittest.TestCase):
    """Test API under heavy load."""

    def setUp(self):
        """Set up exchange."""
        self.exchange = ccxt.kraken({'enableRateLimit': True})

    def test_sequential_requests(self):
        """Test many sequential requests."""
        print("\n  Testing 50 sequential requests...")

        start = time.time()
        success = 0

        for i in range(50):
            try:
                ticker = self.exchange.fetch_ticker('BTC/EUR')
                success += 1
            except Exception as e:
                pass

        elapsed = time.time() - start

        print(f"    Completed: {success}/50 in {elapsed:.1f}s")
        print(f"    Average: {(elapsed/50)*1000:.1f}ms per request")

        self.assertGreater(success, 40, "Should succeed on most requests")

    def test_parallel_requests(self):
        """Test parallel API requests."""
        print("\n  Testing 20 parallel requests...")

        def fetch_ticker():
            try:
                exchange = ccxt.kraken({'enableRateLimit': True})
                ticker = exchange.fetch_ticker('BTC/EUR')
                return True
            except:
                return False

        start = time.time()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_ticker) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]

        elapsed = time.time() - start
        success = sum(results)

        print(f"    Completed: {success}/20 in {elapsed:.1f}s")

        self.assertGreater(success, 10, "Should succeed on most parallel requests")


class TestComputationStress(unittest.TestCase):
    """Test computational performance under stress."""

    def test_massive_rsi_calculations(self):
        """Test massive number of RSI calculations."""
        print("\n  Testing 10,000 RSI calculations...")

        # Generate large dataset
        prices = [50000 + random.uniform(-5000, 5000) for _ in range(10020)]

        start = time.time()

        # Calculate RSI for 10,000 different windows
        results = []
        for i in range(10000):
            window = prices[i:i+14]
            df = pd.DataFrame({'close': window})
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            if loss.iloc[-1] == 0:
                rsi = 50.0
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

            results.append(rsi)

        elapsed = time.time() - start

        print(f"    Completed: {len(results)} calculations in {elapsed:.2f}s")
        print(f"    Throughput: {len(results)/elapsed:.0f} calculations/second")

        self.assertEqual(len(results), 10000)
        self.assertLess(elapsed, 60, "Should complete in under 60 seconds")

    def test_large_dataframe_operations(self):
        """Test operations on very large DataFrames."""
        print("\n  Testing 1M row DataFrame...")

        print(f"    Generating 1,000,000 rows...")
        data = {
            'price': np.random.uniform(45000, 55000, 1000000),
            'volume': np.random.uniform(100, 1000, 1000000)
        }

        start = time.time()

        df = pd.DataFrame(data)

        # Perform multiple operations
        df['returns'] = df['price'].pct_change()
        df['ma_50'] = df['price'].rolling(window=50).mean()
        df['volatility'] = df['returns'].rolling(window=50).std()

        elapsed = time.time() - start

        print(f"    Completed in {elapsed:.2f}s")
        print(f"    Memory: ~{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        self.assertEqual(len(df), 1000000)
        self.assertLess(elapsed, 30, "Should process 1M rows in under 30 seconds")

    def test_rapid_signal_generation(self):
        """Test rapid signal generation."""
        print("\n  Testing 100,000 signal generations...")

        start = time.time()

        for i in range(100000):
            rsi = random.uniform(0, 100)

            if rsi < 42:
                action = "BUY"
                confidence = min(0.95, 0.65 + (42 - rsi) / 80)
            elif rsi > 58:
                action = "SELL"
                confidence = min(0.95, 0.65 + (rsi - 58) / 80)
            else:
                action = "HOLD"
                confidence = 0.5

        elapsed = time.time() - start

        print(f"    Generated 100,000 signals in {elapsed*1000:.1f}ms")
        print(f"    Throughput: {100000/elapsed:.0f} signals/second")

        self.assertLess(elapsed, 1, "Should generate 100K signals in under 1 second")


class TestMemoryStress(unittest.TestCase):
    """Test memory management under stress."""

    def test_repeated_dataframe_creation(self):
        """Test repeated DataFrame creation and cleanup."""
        print("\n  Testing 1,000 DataFrame cycles...")

        import gc

        for i in range(1000):
            # Create DataFrame
            df = pd.DataFrame({
                'price': np.random.uniform(45000, 55000, 1000),
                'volume': np.random.uniform(100, 1000, 1000)
            })

            # Perform operations
            ma = df['price'].rolling(window=20).mean()

            # Cleanup
            del df, ma

            if i % 100 == 0:
                gc.collect()

        print(f"    ✓ Completed 1,000 cycles without memory issues")

    def test_large_dataset_memory(self):
        """Test memory usage with large datasets."""
        print("\n  Testing memory with 5M rows...")

        # Create large dataset
        size = 5000000
        print(f"    Generating {size:,} rows...")

        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=size, freq='1s'),
            'price': np.random.uniform(45000, 55000, size)
        })

        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        print(f"    Memory usage: {memory_mb:.1f} MB")

        # Should be reasonable (less than 1GB)
        self.assertLess(memory_mb, 1000, "Should use less than 1GB")

        # Clean up
        del df


class TestConcurrentStress(unittest.TestCase):
    """Test concurrent operations under stress."""

    def test_concurrent_rsi_calculations(self):
        """Test concurrent RSI calculations."""
        print("\n  Testing 100 concurrent RSI calculations...")

        def calculate_rsi(prices):
            df = pd.DataFrame({'close': prices})
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            if loss.iloc[-1] == 0:
                return 50.0
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        # Generate datasets
        datasets = [
            [50000 + random.uniform(-5000, 5000) for _ in range(50)]
            for _ in range(100)
        ]

        start = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(calculate_rsi, datasets))

        elapsed = time.time() - start

        print(f"    Completed 100 concurrent calculations in {elapsed:.2f}s")
        print(f"    Average: {(elapsed/100)*1000:.1f}ms per calculation")

        self.assertEqual(len(results), 100)
        self.assertTrue(all(0 <= r <= 100 for r in results))


class TestLongRunningStability(unittest.TestCase):
    """Test system stability over extended periods."""

    def test_extended_calculation_loop(self):
        """Test extended calculation loop."""
        print("\n  Testing 10,000 iteration loop...")

        start = time.time()

        for i in range(10000):
            # Generate random data
            prices = [50000 + random.uniform(-1000, 1000) for _ in range(20)]

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

            # Generate signal
            if rsi < 42:
                action = "BUY"
            elif rsi > 58:
                action = "SELL"
            else:
                action = "HOLD"

            if i % 1000 == 0:
                elapsed = time.time() - start
                print(f"    Iteration {i}: {elapsed:.1f}s elapsed")

        elapsed = time.time() - start

        print(f"    ✓ Completed 10,000 iterations in {elapsed:.1f}s")
        self.assertLess(elapsed, 60, "Should complete in under 60 seconds")


class TestBoundaryStress(unittest.TestCase):
    """Test system at computational boundaries."""

    def test_minimum_period_rsi(self):
        """Test RSI with minimum period."""
        print("\n  Testing RSI with minimum data points...")

        prices = [50000] * 14  # Exactly 14 points

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

        print(f"    RSI with minimum data: {rsi:.2f}")
        self.assertTrue(0 <= rsi <= 100)

    def test_maximum_precision_prices(self):
        """Test handling of maximum precision prices."""
        print("\n  Testing high precision prices...")

        # Very precise prices - base is 50000.123456789 with ±0.0001 noise
        # Range: 50000.123356789 to 50000.123556789
        prices = [50000.123456789 + random.uniform(-0.0001, 0.0001) for _ in range(20)]

        df = pd.DataFrame({'close': prices})

        # Should handle high precision without error
        # Check that price is in reasonable range (base 50000.12 + small deviation)
        price = df['close'].iloc[0]
        self.assertGreater(price, 50000)
        self.assertLess(price, 50000.2)  # Allow up to 50000.2 (base is ~50000.1235)

        print(f"    ✓ High precision handled: {price:.6f}")


def run_tests():
    """Run all stress tests."""
    print("=" * 80)
    print("Stress Tests")
    print("=" * 80)
    print("\n⚠️  These tests may take several minutes to complete")

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAPILoad))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestComputationStress))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMemoryStress))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConcurrentStress))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLongRunningStability))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBoundaryStress))

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
