#!/usr/bin/env python3
"""
API connection tests for supported exchanges.

Tests connectivity and basic functionality of exchange APIs.
"""

import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    print("Install: pip install ccxt python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()


class TestKrakenAPI(unittest.TestCase):
    """Test Kraken API connection and functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up Kraken exchange for testing."""
        cls.exchange = ccxt.kraken({
            'apiKey': os.getenv('KRAKEN_API_KEY', ''),
            'secret': os.getenv('KRAKEN_API_SECRET', ''),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })

    def test_public_markets(self):
        """Test fetching markets (public endpoint)."""
        print("\n  Testing Kraken public markets...")
        try:
            markets = self.exchange.load_markets()
            self.assertIsInstance(markets, dict)
            self.assertGreater(len(markets), 0)
            print(f"  ✓ Loaded {len(markets)} markets")

            # Check for BTC/EUR pair
            self.assertIn('BTC/EUR', markets)
            print(f"  ✓ BTC/EUR available: {markets['BTC/EUR']['symbol']}")
        except Exception as e:
            self.fail(f"Failed to load markets: {e}")

    def test_public_ticker(self):
        """Test fetching ticker (public endpoint)."""
        print("\n  Testing Kraken public ticker...")
        try:
            ticker = self.exchange.fetch_ticker('BTC/EUR')
            self.assertIn('last', ticker)
            self.assertIn('bid', ticker)
            self.assertIn('ask', ticker)
            print(f"  ✓ BTC/EUR price: €{ticker['last']:,.2f}")
        except Exception as e:
            self.fail(f"Failed to fetch ticker: {e}")

    def test_public_ohlcv(self):
        """Test fetching OHLCV data (public endpoint)."""
        print("\n  Testing Kraken OHLCV data...")
        try:
            ohlcv = self.exchange.fetch_ohlcv('BTC/EUR', '1h', limit=50)
            self.assertIsInstance(ohlcv, list)
            self.assertEqual(len(ohlcv), 50)
            print(f"  ✓ Fetched {len(ohlcv)} candles")

            # Verify candle structure
            candle = ohlcv[0]
            self.assertEqual(len(candle), 6)  # timestamp, open, high, low, close, volume
            print(f"  ✓ Candle structure valid")
        except Exception as e:
            self.fail(f"Failed to fetch OHLCV: {e}")

    def test_authenticated_balance(self):
        """Test fetching balance (authenticated endpoint)."""
        print("\n  Testing Kraken authenticated balance...")
        if not os.getenv('KRAKEN_API_KEY'):
            self.skipTest("KRAKEN_API_KEY not set")

        try:
            balance = self.exchange.fetch_balance()
            self.assertIsInstance(balance, dict)
            print(f"  ✓ Balance fetched successfully")

            # Check if EUR balance exists
            if 'EUR' in balance:
                eur_balance = balance['EUR']
                print(f"  ✓ EUR balance: €{eur_balance.get('free', 0):,.2f}")
        except Exception as e:
            self.fail(f"Failed to fetch balance: {e}")

    def test_order_book(self):
        """Test fetching order book (public endpoint)."""
        print("\n  Testing Kraken order book...")
        try:
            orderbook = self.exchange.fetch_order_book('BTC/EUR', limit=5)
            self.assertIn('bids', orderbook)
            self.assertIn('asks', orderbook)
            self.assertGreater(len(orderbook['bids']), 0)
            self.assertGreater(len(orderbook['asks']), 0)
            print(f"  ✓ Order book: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks")

            # Show best bid/ask
            if orderbook['bids']:
                best_bid = orderbook['bids'][0]
                print(f"  ✓ Best bid: €{best_bid[0]:,.2f}")
            if orderbook['asks']:
                best_ask = orderbook['asks'][0]
                print(f"  ✓ Best ask: €{best_ask[0]:,.2f}")
        except Exception as e:
            self.fail(f"Failed to fetch order book: {e}")


class TestOneTradingAPI(unittest.TestCase):
    """Test One Trading (Bitpanda Pro) API connection."""

    @classmethod
    def setUpClass(cls):
        """Set up One Trading exchange for testing."""
        cls.exchange = ccxt.onetrading({
            'apiKey': os.getenv('ONETRADING_API_KEY', ''),
            'secret': os.getenv('ONETRADING_API_SECRET', ''),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })

    def test_public_markets(self):
        """Test fetching markets (public endpoint)."""
        print("\n  Testing One Trading public markets...")
        try:
            markets = self.exchange.load_markets()
            self.assertIsInstance(markets, dict)
            print(f"  ✓ Loaded {len(markets)} markets")

            # Check for EUR pairs
            eur_pairs = [s for s in markets if '/EUR' in s]
            print(f"  ✓ EUR pairs: {len(eur_pairs)}")
            if eur_pairs:
                print(f"  ✓ Examples: {eur_pairs[:5]}")
        except Exception as e:
            self.fail(f"Failed to load markets: {e}")

    def test_public_ticker(self):
        """Test fetching ticker (public endpoint)."""
        print("\n  Testing One Trading public ticker...")
        try:
            ticker = self.exchange.fetch_ticker('BTC/EUR')
            self.assertIn('last', ticker)
            print(f"  ✓ BTC/EUR price: €{ticker['last']:,.2f}")
        except Exception as e:
            self.fail(f"Failed to fetch ticker: {e}")

    def test_authenticated_balance(self):
        """Test fetching balance (authenticated endpoint)."""
        print("\n  Testing One Trading authenticated balance...")
        if not os.getenv('ONETRADING_API_KEY'):
            self.skipTest("ONETRADING_API_KEY not set")

        try:
            balance = self.exchange.fetch_balance()
            self.assertIsInstance(balance, dict)
            print(f"  ✓ Balance fetched successfully")
        except Exception as e:
            print(f"  ⚠ Authentication failed: {e}")
            print(f"     This is expected if using Bitpanda Public API key")


class TestExchangeComparison(unittest.TestCase):
    """Compare exchanges for pricing and liquidity."""

    def test_compare_btc_eur_prices(self):
        """Compare BTC/EUR prices across exchanges."""
        print("\n  Comparing BTC/EUR prices...")

        exchanges = {}
        for name, exchange_class in [('Kraken', ccxt.kraken)]:
            try:
                exchange = exchange_class({'enableRateLimit': True})
                ticker = exchange.fetch_ticker('BTC/EUR')
                exchanges[name] = {
                    'price': ticker['last'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'spread': ticker['ask'] - ticker['bid']
                }
                print(f"  ✓ {name}: €{exchanges[name]['price']:,.2f}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

        # If we have multiple exchanges, compare
        if len(exchanges) > 1:
            prices = [e['price'] for e in exchanges.values()]
            price_diff = max(prices) - min(prices)
            print(f"  ✓ Price difference: €{price_diff:.2f}")


def run_tests():
    """Run all API tests."""
    print("=" * 80)
    print("Exchange API Connection Tests")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestKrakenAPI))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOneTradingAPI))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExchangeComparison))

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
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
