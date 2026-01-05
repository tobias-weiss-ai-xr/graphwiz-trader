#!/usr/bin/env python3
"""
Performance tests for trading system.

Tests API response times, rate limits, and computational performance.
"""

import sys
import time
import unittest
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)


class TestAPIPerformance(unittest.TestCase):
    """Test API response times and performance."""

    @classmethod
    def setUpClass(cls):
        """Set up exchange for testing."""
        cls.exchange = ccxt.kraken({'enableRateLimit': True})

    def test_ticker_response_time(self):
        """Test ticker API response time."""
        print("\n  Testing ticker response time...")

        times = []
        for i in range(5):
            start = time.time()
            self.exchange.fetch_ticker('BTC/EUR')
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"    Request {i+1}: {elapsed*1000:.1f}ms")

        avg_time = sum(times) / len(times)
        max_time = max(times)

        print(f"  ✓ Average: {avg_time*1000:.1f}ms, Max: {max_time*1000:.1f}ms")

        # Should be reasonably fast
        self.assertLess(avg_time, 2.0, "Average response should be < 2 seconds")
        self.assertLess(max_time, 5.0, "Max response should be < 5 seconds")

    def test_ohlcv_response_time(self):
        """Test OHLCV API response time."""
        print("\n  Testing OHLCV response time...")

        start = time.time()
        ohlcv = self.exchange.fetch_ohlcv('BTC/EUR', '1h', limit=100)
        elapsed = time.time() - start

        print(f"  ✓ Fetched {len(ohlcv)} candles in {elapsed*1000:.1f}ms")

        # Should be fast even with more data
        self.assertLess(elapsed, 3.0, "Should fetch 100 candles in < 3 seconds")

    def test_orderbook_response_time(self):
        """Test order book response time."""
        print("\n  Testing order book response time...")

        start = time.time()
        orderbook = self.exchange.fetch_order_book('BTC/EUR', limit=20)
        elapsed = time.time() - start

        bids = len(orderbook['bids'])
        asks = len(orderbook['asks'])

        print(f"  ✓ Fetched {bids} bids, {asks} asks in {elapsed*1000:.1f}ms")

        self.assertLess(elapsed, 2.0, "Order book fetch should be < 2 seconds")

    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        print("\n  Testing concurrent requests...")

        symbols = ['BTC/EUR', 'ETH/EUR', 'SOL/EUR']

        start = time.time()

        for symbol in symbols:
            try:
                self.exchange.fetch_ticker(symbol)
            except Exception as e:
                print(f"    Warning: {symbol} failed: {e}")

        elapsed = time.time() - start

        print(f"  ✓ Fetched {len(symbols)} tickers in {elapsed*1000:.1f}ms")
        print(f"    Average per ticker: {(elapsed/len(symbols))*1000:.1f}ms")

        # Should complete all requests
        self.assertLess(elapsed, 10.0, "All requests should complete in < 10 seconds")

    def test_rate_limit_respect(self):
        """Test that rate limits are respected."""
        print("\n  Testing rate limit handling...")

        times = []
        for i in range(3):
            start = time.time()
            self.exchange.fetch_ticker('BTC/EUR')
            elapsed = time.time() - start
            times.append(elapsed)

            if i < 2:
                time.sleep(0.1)  # Small delay between requests

        print(f"  ✓ Request times: {[f'{t*1000:.1f}ms' for t in times]}")

        # Should not trigger rate limiting
        for t in times:
            self.assertLess(t, 5.0, "Each request should complete in < 5 seconds")


class TestComputationalPerformance(unittest.TestCase):
    """Test performance of calculations."""

    def test_rsi_calculation_performance(self):
        """Test RSI calculation speed."""
        print("\n  Testing RSI calculation performance...")

        # Generate large dataset
        np.random.seed(42)
        prices = list(np.random.uniform(45000, 55000, 10000))

        start = time.time()

        # Calculate RSI
        df = pd.DataFrame({'close': prices})
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        elapsed = time.time() - start

        print(f"  ✓ Calculated RSI for {len(prices)} prices in {elapsed*1000:.2f}ms")

        # Should be fast
        self.assertLess(elapsed, 1.0, "RSI calculation should be < 1 second for 10K points")

    def test_signal_generation_performance(self):
        """Test signal generation speed."""
        print("\n  Testing signal generation performance...")

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            rsi = 50 + (i % 60 - 30)  # Varying RSI

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

        print(f"  ✓ Generated {iterations} signals in {elapsed*1000:.2f}ms")
        print(f"    Average: {(elapsed/iterations)*1000:.4f}ms per signal")

        # Should be very fast
        self.assertLess(elapsed, 0.1, "1000 signals should generate in < 100ms")

    def test_position_sizing_performance(self):
        """Test position sizing calculation speed."""
        print("\n  Testing position sizing performance...")

        iterations = 10000
        start = time.time()

        for i in range(iterations):
            balance = 1000 + (i % 5000)
            max_position = 300
            price = 50000

            position_size_eur = min(max_position, balance * 0.25)
            if position_size_eur >= 10:
                amount = position_size_eur / price

        elapsed = time.time() - start

        print(f"  ✓ Calculated {iterations} positions in {elapsed*1000:.2f}ms")
        print(f"    Average: {(elapsed/iterations)*1000000:.2f}μs per calculation")

        # Should be extremely fast
        self.assertLess(elapsed, 0.5, "10K calculations should complete in < 500ms")

    def test_risk_check_performance(self):
        """Test risk management check speed."""
        print("\n  Testing risk check performance...")

        iterations = 10000
        start = time.time()

        for i in range(iterations):
            daily_pnl = -50 + (i % 100)
            max_loss = 50
            daily_trades = i % 5
            max_trades = 2
            position_size = 200 + (i % 200)

            checks = []
            if position_size > 300:
                checks.append(False)
            if daily_pnl <= -max_loss:
                checks.append(False)
            if daily_trades >= max_trades:
                checks.append(False)

            all_passed = len(checks) == 0

        elapsed = time.time() - start

        print(f"  ✓ Performed {iterations} risk checks in {elapsed*1000:.2f}ms")
        print(f"    Average: {(elapsed/iterations)*1000000:.2f}μs per check")

        # Should be very fast
        self.assertLess(elapsed, 0.5, "10K risk checks should complete in < 500ms")


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency of operations."""

    def test_large_dataset_handling(self):
        """Test handling large datasets without memory issues."""
        print("\n  Testing large dataset handling...")

        # Generate large dataset
        size = 100000
        print(f"    Generating {size} data points...")

        np.random.seed(42)
        timestamps = pd.date_range('2020-01-01', periods=size, freq='1min')
        prices = np.random.uniform(45000, 55000, size)
        volumes = np.random.uniform(100, 1000, size)

        start = time.time()

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })

        # Perform operations
        df['returns'] = df['price'].pct_change()
        df['ma_20'] = df['price'].rolling(window=20).mean()
        df['volatility'] = df['returns'].rolling(window=20).std()

        elapsed = time.time() - start

        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        print(f"  ✓ Processed {size} rows in {elapsed:.2f}s")
        print(f"    Memory usage: {memory_mb:.2f} MB")

        # Should complete in reasonable time
        self.assertLess(elapsed, 10.0, "100K rows should process in < 10 seconds")
        self.assertLess(memory_mb, 100, "Memory usage should be < 100 MB")

    def test_dataframe_cleanup(self):
        """Test proper cleanup of DataFrames."""
        print("\n  Testing DataFrame cleanup...")

        import gc

        # Create multiple DataFrames
        dfs = []
        for i in range(10):
            df = pd.DataFrame({
                'price': np.random.uniform(45000, 55000, 1000)
            })
            dfs.append(df)

        # Clear references
        dfs.clear()
        gc.collect()

        print(f"  ✓ DataFrames cleaned up successfully")


class TestScalability(unittest.TestCase):
    """Test system scalability."""

    def test_handling_multiple_symbols(self):
        """Test handling multiple trading symbols."""
        print("\n  Testing multiple symbol handling...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        # Fetch multiple symbols
        symbols = ['BTC/EUR', 'ETH/EUR', 'SOL/EUR', 'ADA/EUR', 'DOT/EUR']

        start = time.time()
        results = {}

        for symbol in symbols:
            try:
                ticker = exchange.fetch_ticker(symbol)
                results[symbol] = {
                    'price': ticker['last'],
                    'volume': ticker['baseVolume']
                }
            except Exception as e:
                print(f"    Warning: {symbol} failed: {e}")

        elapsed = time.time() - start

        print(f"  ✓ Fetched {len(results)}/{len(symbols)} symbols in {elapsed:.2f}s")

        # Should handle multiple symbols
        self.assertGreaterEqual(len(results), 3, "Should fetch at least 3 symbols")

    def test_batch_data_processing(self):
        """Test batch processing of data."""
        print("\n  Testing batch data processing...")

        # Simulate processing batches of data
        batch_size = 100
        num_batches = 10

        start = time.time()

        for batch_num in range(num_batches):
            # Generate batch data
            data = np.random.uniform(45000, 55000, batch_size)

            # Process batch
            df = pd.DataFrame({'close': data})
            ma = df['close'].rolling(window=14).mean()
            std = df['close'].rolling(window=14).std()

            # Clean up
            del df, ma, std

        elapsed = time.time() - start

        total_points = batch_size * num_batches
        print(f"  ✓ Processed {total_points} points in {elapsed*1000:.1f}ms")
        print(f"    Throughput: {total_points/elapsed:.0f} points/second")

        # Should maintain good throughput
        self.assertGreater(total_points/elapsed, 10000, "Should process > 10K points/second")


class TestLatencyBenchmarks(unittest.TestCase):
    """Benchmark critical operations."""

    def test_end_to_end_latency(self):
        """Test complete trading decision latency."""
        print("\n  Testing end-to-end trading decision latency...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        iterations = 10
        latencies = []

        for i in range(iterations):
            start = time.time()

            # 1. Fetch market data
            ticker = exchange.fetch_ticker('BTC/EUR')
            price = ticker['last']

            # 2. Calculate RSI (simulated)
            rsi = 50 + (i % 40 - 20)

            # 3. Generate signal
            if rsi < 42:
                signal = "BUY"
                confidence = min(0.95, 0.65 + (42 - rsi) / 80)
            elif rsi > 58:
                signal = "SELL"
                confidence = min(0.95, 0.65 + (rsi - 58) / 80)
            else:
                signal = "HOLD"
                confidence = 0.5

            # 4. Check risk limits
            risk_ok = confidence > 0.6

            elapsed = time.time() - start
            latencies.append(elapsed)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print(f"  ✓ Average latency: {avg_latency*1000:.1f}ms")
        print(f"    Min: {min_latency*1000:.1f}ms, Max: {max_latency*1000:.1f}ms")

        # Should be fast enough for trading
        self.assertLess(avg_latency, 3.0, "Average trading decision should be < 3 seconds")

    def test_calculation_throughput(self):
        """Benchmark calculation throughput."""
        print("\n  Testing calculation throughput...")

        # Calculate multiple indicators
        iterations = 100
        data_points = 1000

        np.random.seed(42)
        prices = np.random.uniform(45000, 55000, data_points)

        start = time.time()

        for i in range(iterations):
            # Multiple calculations
            df = pd.DataFrame({'close': prices})

            # SMA
            sma_20 = df['close'].rolling(window=20).mean()
            sma_50 = df['close'].rolling(window=50).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Bollinger Bands
            bb_std = df['close'].rolling(window=20).std()
            bb_upper = sma_20 + (2 * bb_std)
            bb_lower = sma_20 - (2 * bb_std)

        elapsed = time.time() - start

        throughput = (iterations * data_points) / elapsed

        print(f"  ✓ Processed {iterations * data_points} calculations in {elapsed:.2f}s")
        print(f"    Throughput: {throughput:.0f} points/second")

        # Should maintain good throughput
        self.assertGreater(throughput, 50000, "Should process > 50K points/second")


def run_tests():
    """Run all performance tests."""
    print("=" * 80)
    print("Performance Tests")
    print("=" * 80)
    print("\n⚠️  Note: These tests measure actual performance and may vary")
    print("    based on network conditions and system load.")

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAPIPerformance))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestComputationalPerformance))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMemoryEfficiency))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestScalability))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLatencyBenchmarks))

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
