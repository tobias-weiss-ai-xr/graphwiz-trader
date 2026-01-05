#!/usr/bin/env python3
"""
Test One Trading (Bitpanda Pro) API connection.

This tests if the Bitpanda API key works with One Trading exchange.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import ccxt
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

def test_onetrading():
    """Test One Trading API connection."""

    # Use provided Bitpanda API key
    api_key = os.getenv('ONETRADING_API_KEY') or "2985be5c272ecae3eab42a102255c4db3d050446c5ec0bda05225350162d5d6a7a07a0b212c850824a760f7ac3840915a94a8b9529bb65270f7600363dd87a23"
    api_secret = os.getenv('ONETRADING_API_SECRET') or ""

    print("=" * 80)
    print("Testing One Trading (Bitpanda Pro) API")
    print("=" * 80)
    print(f"API Key: {api_key[:20]}...")
    print(f"API Secret: {'Set' if api_secret else 'Not set'}")
    print()

    try:
        # Initialize One Trading exchange
        print("Initializing One Trading exchange...")
        exchange = ccxt.onetrading({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })

        # Test public endpoint (no auth required)
        print("\n1. Testing public endpoint (markets)...")
        markets = exchange.load_markets()
        print(f"   ✅ Public API working! Found {len(markets)} markets")

        # Check for EUR pairs
        eur_pairs = [s for s in markets if '/EUR' in s]
        print(f"   EUR trading pairs: {len(eur_pairs)}")
        if eur_pairs:
            print("   Examples:", eur_pairs[:5])

        # Get ticker for BTC/EUR
        if 'BTC/EUR' in markets:
            print("\n2. Testing BTC/EUR ticker...")
            ticker = exchange.fetch_ticker('BTC/EUR')
            print(f"   ✅ BTC/EUR Price: €{ticker['last']:,.2f}")

        # Test authenticated endpoint
        print("\n3. Testing authenticated endpoint (balance)...")
        try:
            balance = exchange.fetch_balance()
            print("   ✅ Authenticated API working!")
            print(f"   Balance types: {list(balance.keys())}")

            # Show EUR balance if available
            if 'EUR' in balance:
                eur_balance = balance['EUR']
                print(f"   EUR Balance: €{eur_balance.get('free', 0):,.2f}")

        except Exception as auth_error:
            print(f"   ❌ Authentication failed: {auth_error}")
            print("\n   This usually means:")
            print("   • The API key is from Bitpanda Public API (read-only)")
            print("   • One Trading requires separate API keys")
            print("   • Need to create API keys at: https://exchange.onetrading.com/")
            return False

        print("\n" + "=" * 80)
        print("✅ One Trading API connection successful!")
        print("=" * 80)
        print("\nYour Bitpanda API key works with One Trading.")
        print("You can use One Trading for live trading.")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if API key is correct")
        print("2. One Trading (Bitpanda Pro) may require separate API keys")
        print("3. Generate API keys at: https://exchange.onetrading.com/api")
        return False

if __name__ == "__main__":
    success = test_onetrading()
    sys.exit(0 if success else 1)
