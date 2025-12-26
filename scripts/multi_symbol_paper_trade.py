#!/usr/bin/env python3
"""
Multi-symbol paper trading script.

Run paper trading on multiple symbols simultaneously.
"""

import sys
from pathlib import Path
import subprocess
import time
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Recommended symbols for paper trading
RECOMMENDED_SYMBOLS = [
    "BTC/USDT",  # Bitcoin - Base asset
    "ETH/USDT",  # Ethereum - High liquidity
    "SOL/USDT",  # Solana - High performance
    "BNB/USDT",  # Binance Coin - Low risk
    "XRP/USDT",  # Ripple - High volume
    "DOGE/USDT", # Dogecoin - High volatility
]

def run_paper_trading(symbol: str, capital: float = 10000):
    """Run paper trading for a single symbol.

    Args:
        symbol: Trading pair symbol
        capital: Starting capital
    """
    cmd = [
        "python", "scripts/paper_trade.py",
        "--symbol", symbol,
        "--capital", str(capital),
        "--continuous",
        "--interval", "3600",  # Check every hour
    ]

    print(f"Starting paper trading for {symbol}...")
    subprocess.Popen(cmd)

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-symbol paper trading"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        help="Symbols to trade (default: BTC, ETH, SOL)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Capital per symbol (default: 10000)"
    )
    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Use all recommended symbols"
    )

    args = parser.parse_args()

    # Select symbols
    if args.recommended:
        symbols = RECOMMENDED_SYMBOLS
    else:
        symbols = args.symbols

    print("\n" + "=" * 80)
    print("MULTI-SYMBOL PAPER TRADING")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Capital per symbol: ${args.capital:,.2f}")
    print(f"Total virtual capital: ${args.capital * len(symbols):,.2f}")
    print("=" * 80 + "\n")

    # Start paper trading for each symbol
    processes = []
    for symbol in symbols:
        run_paper_trading(symbol, args.capital)
        time.sleep(2)  # Stagger starts

    print("\n✅ All paper trading sessions started!")
    print(f"\nMonitoring {len(symbols)} symbols:")
    for symbol in symbols:
        print(f"  • {symbol}")

    print("\nResults will be saved to data/paper_trading/")
    print("\nPress Ctrl+C to stop all sessions\n")

    # Wait indefinitely
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping all paper trading sessions...")


if __name__ == "__main__":
    main()
