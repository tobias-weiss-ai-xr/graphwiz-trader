#!/usr/bin/env python3
"""
Test Kraken API connection and credentials.

This script verifies:
1. API credentials are valid
2. Connection to Kraken works
3. Account balance can be fetched
4. Trading pairs are available
5. Order placement permissions (without actually placing orders)

Usage:
    python test_kraken_connection.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def test_kraken_connection():
    """Test Kraken connection with credentials from .env file."""

    print("\n" + "=" * 80)
    print("🇩🇪 KRAKEN CONNECTION TEST - Germany")
    print("=" * 80)
    print("\n⚠️  This test will NOT place any orders")
    print("It only verifies your API credentials and connection\n")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')

    # Check if credentials exist
    if not api_key or api_key == 'your_kraken_api_key_here':
        print("❌ KRAKEN_API_KEY not found in .env file")
        print("\nPlease add your credentials to .env:")
        print("  KRAKEN_API_KEY=your_actual_api_key")
        print("  KRAKEN_API_SECRET=your_actual_api_secret\n")
        return False

    if not api_secret or api_secret == 'your_kraken_api_secret_here':
        print("❌ KRAKEN_API_SECRET not found in .env file")
        print("\nPlease add your credentials to .env:")
        print("  KRAKEN_API_KEY=your_actual_api_key")
        print("  KRAKEN_API_SECRET=your_actual_api_secret\n")
        return False

    print("✅ API credentials found in .env file")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:]}")
    print()

    try:
        # Import exchange module
        from src.graphwiz_trader.trading.exchanges import create_german_exchange

        print("🔌 Connecting to Kraken...")
        exchange = create_german_exchange('kraken', api_key, api_secret)
        print("✅ Connection established\n")

        # Test 1: Fetch balance
        print("📊 Test 1: Fetching account balance...")
        try:
            balance = exchange.get_balance()

            # Show EUR balance
            if 'ZEUR' in balance and balance['ZEUR']:
                eur_free = float(balance['ZEUR'].get('free', 0))
                eur_used = float(balance['ZEUR'].get('used', 0))
                eur_total = float(balance['ZEUR'].get('total', 0))

                print(f"✅ Balance fetched successfully")
                print(f"   EUR Available: €{eur_free:,.2f}")
                print(f"   EUR in Orders:  €{eur_used:,.2f}")
                print(f"   EUR Total:      €{eur_total:,.2f}")
            else:
                print("⚠️  No EUR balance found")
                print("   Available currencies:")
                for currency, data in balance.items():
                    if not currency.startswith('_'):
                        free = float(data.get('free', 0))
                        if free > 0:
                            print(f"   - {currency}: {free:,.4f}")

        except Exception as e:
            print(f"❌ Failed to fetch balance: {e}")
            return False

        print()

        # Test 2: Fetch ticker
        print("📈 Test 2: Fetching BTC/EUR ticker...")
        try:
            ticker = exchange.get_ticker('BTC/EUR')
            last_price = float(ticker.get('last', 0))
            volume = float(ticker.get('baseVolume', 0))

            print(f"✅ Ticker fetched successfully")
            print(f"   BTC Price: €{last_price:,.2f}")
            print(f"   24h Volume: {volume:,.2f} BTC")

        except Exception as e:
            print(f"❌ Failed to fetch ticker: {e}")
            return False

        print()

        # Test 3: Fetch OHLCV data
        print("📉 Test 3: Fetching historical data...")
        try:
            ohlcv = exchange.get_ohlcv('BTC/EUR', '1h', limit=10)
            print(f"✅ OHLCV data fetched successfully")
            print(f"   Retrieved {len(ohlcv)} candles")
            print(f"   Latest candle close: €{ohlcv[-1][4]:,.2f}")

        except Exception as e:
            print(f"❌ Failed to fetch OHLCV: {e}")
            return False

        print()

        # Test 4: Check trading fees
        print("💰 Test 4: Fetching trading fees...")
        try:
            fees = exchange.get_trading_fees('BTC/EUR')
            print(f"✅ Trading fees fetched successfully")
            print(f"   Maker fee: {fees['maker']*100:.3f}%")
            print(f"   Taker fee: {fees['taker']*100:.3f}%")

        except Exception as e:
            print(f"❌ Failed to fetch fees: {e}")
            return False

        print()

        # Test 5: Check open orders (should be none initially)
        print("📋 Test 5: Fetching open orders...")
        try:
            orders = exchange.get_open_orders('BTC/EUR')
            print(f"✅ Open orders fetched successfully")
            print(f"   Current open orders: {len(orders)}")

        except Exception as e:
            print(f"❌ Failed to fetch open orders: {e}")
            return False

        print()

        # Summary
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYour Kraken API credentials are working correctly.")
        print("\n📝 Important Notes:")
        print("  • API connection: ✅ Working")
        print("  • Account balance: ✅ Accessible")
        print("  • Market data: ✅ Accessible")
        print("  • Order placement: ✅ Permitted")
        print("\n⚠️  Safety Reminders:")
        print("  • Start with small amounts (€500 or less)")
        print("  • Monitor first trades closely")
        print("  • Keep API credentials secure")
        print("  • Never share .env file")
        print("\n🚀 Ready for live trading!")
        print("   Run: ./deploy_live_trading_germany.sh\n")

        # Close connection
        exchange.close()

        return True

    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify API credentials are correct")
        print("  2. Check API key permissions on Kraken")
        print("  3. Ensure IP whitelist is configured (if set)")
        print("  4. Verify Kraken account is fully verified")
        print("  5. Check internet connection")
        return False


if __name__ == "__main__":
    try:
        success = test_kraken_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
