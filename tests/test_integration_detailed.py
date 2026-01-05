#!/usr/bin/env python3
"""
Detailed integration tests for Kraken and One Trading APIs.

Comprehensive test of all API endpoints with detailed output.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title):
    """Print formatted section."""
    print(f"\n{'─' * 80}")
    print(f" {title}")
    print(f"{'─' * 80}")


def test_kraken():
    """Comprehensive Kraken API test."""
    print_header("Kraken Integration Tests")

    # Initialize exchange
    api_key = os.getenv('KRAKEN_API_KEY', '')
    api_secret = os.getenv('KRAKEN_API_SECRET', '')

    print(f"\nCredentials:")
    print(f"  API Key: {'✓ Set' if api_key else '✗ Not set'} ({len(api_key)} chars)")
    print(f"  API Secret: {'✓ Set' if api_secret else '✗ Not set'} ({len(api_secret)} chars)")

    exchange = ccxt.kraken({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })

    # Test 1: Public Markets
    print_section("1. Public Markets")
    try:
        markets = exchange.load_markets()
        print(f"✓ Loaded {len(markets)} markets")

        # Find EUR pairs
        eur_pairs = [s for s in markets if '/EUR' in s]
        print(f"✓ Found {len(eur_pairs)} EUR trading pairs")
        print(f"  Examples: {eur_pairs[:10]}")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 2: BTC/EUR Ticker
    print_section("2. BTC/EUR Ticker")
    try:
        ticker = exchange.fetch_ticker('BTC/EUR')
        print(f"  Last Price: €{ticker['last']:,.2f}")
        print(f"  24h High: €{ticker['high']:,.2f}")
        print(f"  24h Low: €{ticker['low']:,.2f}")
        print(f"  24h Volume: {ticker['baseVolume']:,.2f} BTC")
        print(f"  24h Change: {ticker['percentage']:>+.2f}%")
        print(f"✓ Ticker fetched successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 3: Order Book
    print_section("3. Order Book (BTC/EUR)")
    try:
        orderbook = exchange.fetch_order_book('BTC/EUR', limit=5)

        print(f"  Bids (Buy Orders):")
        for i, bid in enumerate(orderbook['bids'][:5], 1):
            print(f"    {i}. €{bid[0]:>10,.2f} | {bid[1]:>10.4f} BTC")

        print(f"\n  Asks (Sell Orders):")
        for i, ask in enumerate(orderbook['asks'][:5], 1):
            print(f"    {i}. €{ask[0]:>10,.2f} | {ask[1]:>10.4f} BTC")

        spread = orderbook['asks'][0][0] - orderbook['bids'][0][0]
        spread_pct = (spread / orderbook['bids'][0][0]) * 100
        print(f"\n  Spread: €{spread:.2f} ({spread_pct:.3f}%)")
        print(f"✓ Order book fetched successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 4: OHLCV Data
    print_section("4. OHLCV Data (BTC/EUR - 1h candles)")
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/EUR', '1h', limit=10)

        print(f"  Latest 10 candles:")
        print(f"  {'Timestamp':<20} | {'Open':>10} | {'High':>10} | {'Low':>10} | {'Close':>10} | {'Volume':>10}")
        print(f"  {'-' * 20} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}")

        for candle in reversed(ohlcv[-5:]):
            timestamp = datetime.fromtimestamp(candle[0] / 1000).strftime('%Y-%m-%d %H:%M')
            print(f"  {timestamp} | €{candle[1]:>9,.2f} | €{candle[2]:>9,.2f} | €{candle[3]:>9,.2f} | €{candle[4]:>9,.2f} | {candle[5]:>10.4f}")

        print(f"✓ OHLCV data fetched successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 5: Authenticated Balance
    if api_key and api_secret:
        print_section("5. Account Balance (Authenticated)")
        try:
            balance = exchange.fetch_balance()

            print(f"  Asset Balances:")
            for currency, data in sorted(balance.items()):
                if isinstance(data, dict) and 'total' in data:
                    if data['total'] > 0:
                        print(f"    {currency}: {data['free']:>10.4f} (frozen: {data['used']:>.4f})")

            print(f"✓ Balance fetched successfully")

        except Exception as e:
            print(f"✗ Authentication failed: {e}")
    else:
        print_section("5. Account Balance (Skipped)")
        print("  ⚠️  No API credentials configured")

    return True


def test_onetrading():
    """Comprehensive One Trading API test."""
    print_header("One Trading (Bitpanda Pro) Integration Tests")

    # Initialize exchange
    api_key = os.getenv('ONETRADING_API_KEY', '')
    api_secret = os.getenv('ONETRADING_API_SECRET', '')

    print(f"\nCredentials:")
    print(f"  API Key: {'✓ Set' if api_key else '✗ Not set'} ({len(api_key)} chars)")
    print(f"  API Secret: {'✓ Set' if api_secret else '✗ Not set'} ({len(api_secret)} chars)")

    exchange = ccxt.onetrading({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })

    # Test 1: Public Markets
    print_section("1. Public Markets")
    try:
        markets = exchange.load_markets()
        print(f"✓ Loaded {len(markets)} markets")

        # Find EUR pairs
        eur_pairs = [s for s in markets if '/EUR' in s]
        print(f"✓ Found {len(eur_pairs)} EUR trading pairs")
        print(f"  All EUR pairs: {eur_pairs}")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 2: BTC/EUR Ticker
    print_section("2. BTC/EUR Ticker")
    try:
        ticker = exchange.fetch_ticker('BTC/EUR')
        print(f"  Last Price: €{ticker['last']:,.2f}")
        print(f"  24h High: €{ticker['high']:,.2f}")
        print(f"  24h Low: €{ticker['low']:,.2f}")
        print(f"  24h Volume: {ticker['baseVolume']:,.2f} BTC")
        print(f"  24h Change: {ticker.get('percentage', 0):>+.2f}%")
        print(f"✓ Ticker fetched successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 3: Multiple Tickers
    print_section("3. Multiple Prices")
    try:
        for symbol in ['BTC/EUR', 'ETH/EUR', 'SOL/EUR']:
            if symbol in markets:
                ticker = exchange.fetch_ticker(symbol)
                print(f"  {symbol}: €{ticker['last']:,.2f}")

        print(f"✓ All tickers fetched successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test 4: Authenticated Balance
    if api_key and api_secret:
        print_section("4. Account Balance (Authenticated)")
        try:
            balance = exchange.fetch_balance()

            print(f"  Asset Balances:")
            for currency, data in sorted(balance.items()):
                if isinstance(data, dict) and 'total' in data:
                    if data['total'] > 0:
                        print(f"    {currency}: {data['free']:>10.4f} (frozen: {data['used']:>.4f})")

            print(f"✓ Balance fetched successfully")

        except Exception as e:
            print(f"✗ Authentication failed: {e}")
            print(f"  This usually means:")
            print(f"    • You're using Bitpanda Public API key (read-only)")
            print(f"    • One Trading requires separate exchange API keys")
            print(f"    • Generate keys at: https://exchange.onetrading.com/")
    else:
        print_section("4. Account Balance (Skipped)")
        print("  ⚠️  No ONETRADING_API credentials configured")
        print("  To enable:")
        print("    1. Create account at https://exchange.onetrading.com/")
        print("    2. Generate API keys from exchange settings")
        print("    3. Add to .env:")
        print("       ONETRADING_API_KEY=your_key")
        print("       ONETRADING_API_SECRET=your_secret")

    return True


def compare_exchanges():
    """Compare prices between exchanges."""
    print_header("Exchange Comparison")

    try:
        # Kraken
        kraken = ccxt.kraken({'enableRateLimit': True})
        kraken_ticker = kraken.fetch_ticker('BTC/EUR')
        kraken_price = kraken_ticker['last']

        # One Trading
        onetrading = ccxt.onetrading({'enableRateLimit': True})
        onetrading_ticker = onetrading.fetch_ticker('BTC/EUR')
        onetrading_price = onetrading_ticker['last']

        # Comparison
        print(f"\nBTC/EUR Price Comparison:")
        print(f"  Kraken:     €{kraken_price:,.2f}")
        print(f"  One Trading: €{onetrading_price:,.2f}")

        diff = abs(kraken_price - onetrading_price)
        diff_pct = (diff / kraken_price) * 100

        print(f"\n  Difference: €{diff:.2f} ({diff_pct:.3f}%)")

        if kraken_price < onetrading_price:
            print(f"  ✓ Kraken is cheaper by €{onetrading_price - kraken_price:.2f}")
        else:
            print(f"  ✓ One Trading is cheaper by €{kraken_price - onetrading_price:.2f}")

        print(f"\n  Recommendation:")
        if diff_pct > 0.1:
            cheaper = "Kraken" if kraken_price < onetrading_price else "One Trading"
            print(f"    • Buy on {cheaper} for better price")
        else:
            print(f"    • Prices are very similar, either exchange is fine")

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    return True


def main():
    """Run all integration tests."""
    print("=" * 80)
    print(" GraphWiz Trader - Integration Tests")
    print(" Kraken & One Trading APIs")
    print("=" * 80)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Test Kraken
    try:
        results['Kraken'] = test_kraken()
    except Exception as e:
        print(f"\n✗ Kraken tests failed with error: {e}")
        results['Kraken'] = False

    # Test One Trading
    try:
        results['One Trading'] = test_onetrading()
    except Exception as e:
        print(f"\n✗ One Trading tests failed with error: {e}")
        results['One Trading'] = False

    # Compare exchanges
    try:
        compare_exchanges()
    except Exception as e:
        print(f"\n✗ Exchange comparison failed: {e}")

    # Print summary
    print_header("Integration Test Summary")

    for exchange, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {exchange}: {status}")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 All integration tests passed!")
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"\n⚠️  Some tests failed: {', '.join(failed)}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
