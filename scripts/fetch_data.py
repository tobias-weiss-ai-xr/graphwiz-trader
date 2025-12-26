#!/usr/bin/env python3
"""
Fetch historical market data from exchanges for backtesting.

Supports fetching OHLCV data from various exchanges via CCXT.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt
except ImportError:
    logger.error("CCXT not installed. Install with: pip install ccxt")
    sys.exit(1)


def fetch_ohlcv(
    exchange_name: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 30,
    limit: int = 1000
):
    """Fetch OHLCV data from exchange.

    Args:
        exchange_name: Exchange name (binance, kraken, etc.)
        symbol: Trading pair symbol
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        days: Number of days of history
        limit: Max candles per request

    Returns:
        List of OHLCV dictionaries
    """
    logger.info(f"Fetching {symbol} data from {exchange_name}")

    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    # Calculate timeframe in milliseconds
    timeframe_ms = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }.get(timeframe, 60 * 60 * 1000)

    # Calculate since timestamp
    since = datetime.now() - timedelta(days=days)
    since_ms = int(since.timestamp() * 1000)

    all_ohlcv = []
    current_since = since_ms

    # Fetch data in batches
    while True:
        try:
            logger.debug(f"Fetching {limit} candles from {datetime.fromtimestamp(current_since/1000)}")

            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=current_since,
                limit=limit
            )

            if not ohlcv:
                logger.info("No more data available")
                break

            all_ohlcv.extend(ohlcv)
            logger.info(f"Fetched {len(ohlcv)} candles, total: {len(all_ohlcv)}")

            # Move to next batch
            current_since = ohlcv[-1][0] + timeframe_ms

            # Check if we've fetched enough
            if current_since >= datetime.now().timestamp() * 1000:
                break

            # Check if we're getting duplicate data (end of available data)
            if len(ohlcv) < limit:
                break

        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit exceeded, waiting...")
            import time
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break

    # Convert to standard format
    data = []
    for candle in all_ohlcv:
        data.append({
            'timestamp': datetime.fromtimestamp(candle[0] / 1000),
            'open': candle[1],
            'high': candle[2],
            'low': candle[3],
            'close': candle[4],
            'volume': candle[5]
        })

    logger.success(f"Successfully fetched {len(data)} candles")
    return data


def save_data(data, symbol: str, timeframe: str, output_dir: str = "data"):
    """Save fetched data to CSV file.

    Args:
        data: List of OHLCV dictionaries
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        output_dir: Output directory
    """
    import pandas as pd
    from pathlib import Path

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    # Generate filename
    symbol_clean = symbol.replace('/', '_')
    filename = f"{symbol_clean}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
    filepath = Path(output_dir) / filename

    # Save to CSV
    df.to_csv(filepath)
    logger.success(f"Data saved to {filepath}")

    return filepath


def load_data(symbol: str, timeframe: str, data_dir: str = "data"):
    """Load previously saved data from CSV.

    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        data_dir: Data directory

    Returns:
        List of OHLCV dictionaries
    """
    import pandas as pd
    from pathlib import Path

    symbol_clean = symbol.replace('/', '_')

    # Find most recent file
    files = list(Path(data_dir).glob(f"{symbol_clean}_{timeframe}_*.csv"))

    if not files:
        logger.warning(f"No data files found for {symbol} {timeframe}")
        return None

    # Sort by modification time and get most recent
    filepath = max(files, key=lambda f: f.stat().st_mtime)

    # Load from CSV
    df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')

    # Convert to list of dicts
    data = df.reset_index().to_dict('records')

    logger.info(f"Loaded {len(data)} candles from {filepath}")
    return data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch historical market data for backtesting"
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange name (default: binance)"
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair symbol (default: BTC/USDT)"
    )
    parser.add_argument(
        "--timeframe",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        default="1h",
        help="Candle timeframe (default: 1h)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data (default: 30)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save data to CSV file"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for saved data (default: data/)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max candles per request (default: 1000)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    try:
        # Fetch data
        data = fetch_ohlcv(
            exchange_name=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            limit=args.limit
        )

        if not data:
            logger.error("No data fetched")
            return 1

        # Show summary
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Exchange:    {args.exchange}")
        print(f"Symbol:      {args.symbol}")
        print(f"Timeframe:   {args.timeframe}")
        print(f"Candles:     {len(data)}")
        print(f"Date Range:  {data[0]['timestamp']} to {data[-1]['timestamp']}")
        print(f"Price Range: ${min(d['low'] for d in data):,.2f} - "
              f"${max(d['high'] for d in data):,.2f}")
        print("="*60 + "\n")

        # Save if requested
        if args.save:
            filepath = save_data(data, args.symbol, args.timeframe, args.output_dir)
            print(f"💾 Saved to: {filepath}")

        return 0

    except Exception as e:
        logger.exception("Failed to fetch data: {}", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
