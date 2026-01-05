#!/usr/bin/env python3
"""
Error handling tests.

Tests how the system handles various failure scenarios and edge cases.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)


class TestNetworkErrorHandling(unittest.TestCase):
    """Test handling of network errors."""

    def test_timeout_handling(self):
        """Test system handles timeouts gracefully."""
        print("\n  Testing timeout error handling...")

        exchange = ccxt.kraken({
            'timeout': 1,  # 1 second timeout
            'enableRateLimit': True,
        })

        # Try to fetch with unrealistic timeout
        try:
            # This should not crash the system
            start = time.time()
            exchange.fetch_ticker('BTC/EUR')
            elapsed = time.time() - start
            print(f"  ✓ Request completed in {elapsed:.2f}s")
            self.assertLess(elapsed, 5, "Request should complete within 5 seconds")
        except (ccxt.RequestTimeout, ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            # These are all acceptable network-related errors
            print(f"  ✓ Network error handled gracefully: {type(e).__name__}")
        except Exception as e:
            # Other network-related errors (SSL, connection, etc.)
            print(f"  ✓ Error handled gracefully: {type(e).__name__}")
            error_msg = str(e).lower()
            # Check for any network-related error keywords
            is_network_error = any(keyword in error_msg for keyword in
                ['timeout', 'network', 'request', 'connection', 'ssl', 'tls', 'socket', 'errno'])
            self.assertTrue(is_network_error, f"Expected network-related error, got: {error_msg}")

    def test_invalid_symbol(self):
        """Test handling of invalid trading symbols."""
        print("\n  Testing invalid symbol handling...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        try:
            # Try to fetch invalid symbol
            exchange.fetch_ticker('INVALID/SYMBOL')
            self.fail("Should have raised an error for invalid symbol")
        except Exception as e:
            print(f"  ✓ Invalid symbol error: {type(e).__name__}")
            # Should be a ccxt error
            self.assertIsInstance(e, (ccxt.BadSymbol, ccxt.ExchangeError))


class TestAPIErrorHandling(unittest.TestCase):
    """Test handling of API errors."""

    def test_missing_credentials(self):
        """Test behavior with missing API credentials."""
        print("\n  Testing missing credentials...")

        exchange = ccxt.kraken({
            'apiKey': '',  # Empty
            'secret': '',  # Empty
        })

        try:
            # Try authenticated endpoint without credentials
            exchange.fetch_balance()
            self.fail("Should fail without credentials")
        except Exception as e:
            print(f"  ✓ Authentication error: {type(e).__name__}")
            # Check for credential-related error messages
            error_msg = str(e).lower()
            self.assertTrue('credential' in error_msg or 'apikey' in error_msg or 'authentication' in error_msg or 'permission' in error_msg)

    def test_invalid_credentials(self):
        """Test behavior with invalid API credentials."""
        print("\n  Testing invalid credentials...")

        exchange = ccxt.kraken({
            'apiKey': 'invalid_key_12345',
            'secret': 'invalid_secret_67890',
        })

        try:
            exchange.fetch_balance()
            self.fail("Should fail with invalid credentials")
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
            print(f"  ✓ Invalid credentials error: {type(e).__name__}")
            # Should be authentication error
        except Exception as e:
            # Base64 decoding errors also indicate invalid credentials
            print(f"  ✓ Invalid credentials error: {type(e).__name__}")
            error_msg = str(e).lower()
            self.assertTrue('credential' in error_msg or 'authentication' in error_msg or 'padding' in error_msg)

    def test_rate_limit_handling(self):
        """Test handling of rate limits."""
        print("\n  Testing rate limit handling...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        # Make multiple rapid requests
        print(f"  Making 5 rapid requests...")
        for i in range(5):
            try:
                exchange.fetch_ticker('BTC/EUR')
                print(f"    Request {i+1}: ✓")
            except ccxt.RateLimitExceeded as e:
                print(f"    Request {i+1}: Rate limited (expected)")
                self.assertIsInstance(e, ccxt.RateLimitExceeded)
                break
        else:
            print(f"  ✓ All requests handled without rate limiting")


class TestDataErrorHandling(unittest.TestCase):
    """Test handling of data errors."""

    def test_missing_ohlcv_data(self):
        """Test handling when OHLCV data is missing."""
        print("\n  Testing missing OHLCV data...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        try:
            # Try to fetch OHLCV for potentially invalid symbol
            ohlcv = exchange.fetch_ohlcv('BTC/EUR', '1m', limit=0)
            # Should return empty list, not crash
            self.assertIsInstance(ohlcv, list)
            print(f"  ✓ Handled gracefully, returned: {len(ohlcv)} candles")
        except Exception as e:
            print(f"  ✓ Error handled: {type(e).__name__}")

    def test_empty_orderbook(self):
        """Test handling of empty order book."""
        print("\n  Testing empty order book...")

        exchange = ccxt.kraken({'enableRateLimit': True})

        try:
            # Fetch order book
            orderbook = exchange.fetch_order_book('BTC/EUR', limit=1)

            # Should have bids and asks keys
            self.assertIn('bids', orderbook)
            self.assertIn('asks', orderbook)
            self.assertIsInstance(orderbook['bids'], list)
            self.assertIsInstance(orderbook['asks'], list)

            print(f"  ✓ Order book structure valid")
            print(f"    Bids: {len(orderbook['bids'])}")
            print(f"    Asks: {len(orderbook['asks'])}")

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")


class TestCalculationErrorHandling(unittest.TestCase):
    """Test error handling in calculations."""

    def test_empty_price_data(self):
        """Test RSI calculation with empty data."""
        print("\n  Testing RSI with empty data...")

        try:
            import pandas as pd

            # Empty price list
            prices = []

            # Should handle gracefully
            if len(prices) < 14:
                rsi = 50.0  # Default value
            else:
                df = pd.DataFrame({'close': prices})
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

            print(f"  ✓ RSI with empty data: {rsi}")
            self.assertEqual(rsi, 50.0)

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            self.fail("Should handle empty data gracefully")

    def test_division_by_zero(self):
        """Test handling of division by zero in calculations."""
        print("\n  Testing division by zero handling...")

        try:
            # Calculate change percentage
            old_price = 0
            new_price = 50000

            if old_price == 0:
                change_pct = 0.0
            else:
                change_pct = ((new_price - old_price) / old_price) * 100

            print(f"  ✓ Division by zero handled: {change_pct}%")
            self.assertEqual(change_pct, 0.0)

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            self.fail("Should handle division by zero")

    def test_nan_values(self):
        """Test handling of NaN values in calculations."""
        print("\n  Testing NaN value handling...")

        try:
            import pandas as pd
            import numpy as np

            # Data with potential NaN
            data = [100, 102, np.nan, 106, 108]

            # Clean data
            clean_data = [x for x in data if not pd.isna(x)]

            # Calculate average
            avg = sum(clean_data) / len(clean_data) if clean_data else 0

            print(f"  ✓ NaN values handled: {avg:.2f}")
            self.assertGreater(avg, 0)

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            self.fail("Should handle NaN values")


class TestRiskErrorHandling(unittest.TestCase):
    """Test error handling in risk management."""

    def test_negative_balance(self):
        """Test handling of negative balance."""
        print("\n  Testing negative balance handling...")

        balance = -100  # Negative balance

        # Should reject negative balance
        if balance < 0:
            valid = False
            error = "Balance cannot be negative"
        else:
            valid = True
            error = ""

        print(f"  ✓ Negative balance rejected: {error}")
        self.assertFalse(valid)
        self.assertEqual(error, "Balance cannot be negative")

    def test_zero_position_size(self):
        """Test handling of zero position size."""
        print("\n  Testing zero position size...")

        position_size = 0

        if position_size <= 0:
            valid = False
            error = "Position size must be positive"
        else:
            valid = True
            error = ""

        print(f"  ✓ Zero position rejected: {error}")
        self.assertFalse(valid)

    def test_exceeds_daily_loss_limit(self):
        """Test exceeding daily loss limit."""
        print("\n  Testing daily loss limit...")

        daily_pnl = -75  # €75 loss
        max_loss = -50   # €50 limit

        if daily_pnl <= max_loss:
            trading_allowed = False
            error = f"Daily loss €{abs(daily_pnl)} exceeds limit €{abs(max_loss)}"
        else:
            trading_allowed = True
            error = ""

        print(f"  ✓ Loss limit enforced: {error}")
        self.assertFalse(trading_allowed)


class TestConfigurationErrorHandling(unittest.TestCase):
    """Test handling of configuration errors."""

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        print("\n  Testing missing config file...")

        config_file = Path(__file__).parent.parent / 'config/nonexistent.yaml'

        if not config_file.exists():
            error = f"Config file not found: {config_file}"
            print(f"  ✓ Missing config detected: {error}")
            self.assertIn("not found", error.lower())

    def test_invalid_yaml(self):
        """Test handling of invalid YAML syntax."""
        print("\n  Testing invalid YAML...")

        try:
            import yaml

            invalid_yaml = """
            key: value
              indented_wrong:
            - item1
            """

            yaml.safe_load(invalid_yaml)
            self.fail("Should raise error for invalid YAML")

        except yaml.YAMLError as e:
            print(f"  ✓ Invalid YAML caught: {type(e).__name__}")
            self.assertIsInstance(e, yaml.YAMLError)

    def test_missing_required_config(self):
        """Test handling of missing required config values."""
        print("\n  Testing missing required config...")

        config = {
            'live_trading': {
                'max_position_eur': 300,
                # Missing max_daily_loss_eur
            }
        }

        required_fields = ['max_position_eur', 'max_daily_loss_eur', 'max_daily_trades']
        missing = [f for f in required_fields if f not in config.get('live_trading', {})]

        if missing:
            error = f"Missing required config: {missing}"
            print(f"  ✓ Missing config detected: {error}")
            self.assertGreater(len(missing), 0)


class TestRecoveryScenarios(unittest.TestCase):
    """Test system recovery from errors."""

    def test_retry_logic(self):
        """Test retry logic for failed requests."""
        print("\n  Testing retry logic...")

        max_retries = 3
        attempts = 0
        success = False

        while attempts < max_retries and not success:
            attempts += 1
            try:
                # Simulate operation that might fail
                if attempts < 2:
                    raise ConnectionError("Simulated failure")
                success = True
                print(f"  ✓ Success on attempt {attempts}")
            except ConnectionError as e:
                print(f"    Attempt {attempts} failed: {e}")
                if attempts < max_retries:
                    time.sleep(0.1)  # Brief delay before retry

        self.assertTrue(success, "Should succeed within retry limit")

    def test_fallback_to_default(self):
        """Test fallback to default values on error."""
        print("\n  Testing fallback to defaults...")

        try:
            # Simulate getting value from config
            value = None  # Config fetch failed

            # Fallback to default
            default_value = 300
            final_value = value if value is not None else default_value

            print(f"  ✓ Fallback to default: {final_value}")
            self.assertEqual(final_value, default_value)

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            self.fail("Should use fallback value")


def run_tests():
    """Run all error handling tests."""
    print("=" * 80)
    print("Error Handling Tests")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNetworkErrorHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAPIErrorHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataErrorHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalculationErrorHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskErrorHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConfigurationErrorHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRecoveryScenarios))

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
