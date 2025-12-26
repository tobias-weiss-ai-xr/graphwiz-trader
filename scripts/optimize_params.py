#!/usr/bin/env python3
"""
Parameter optimization for backtesting strategies.

Tests multiple parameter combinations to find optimal settings.
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from itertools import product

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphwiz_trader.backtesting import BacktestEngine, SimpleMovingAverageStrategy, RSIMeanReversionStrategy


def load_csv_data(filepath: str):
    """Load historical data from CSV file."""
    logger.info(f"Loading {filepath}")

    df = pd.read_csv(filepath, parse_dates=['timestamp'])

    data = []
    for _, row in df.iterrows():
        data.append({
            'timestamp': row['timestamp'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })

    logger.info(f"Loaded {len(data)} candles")
    return data


def optimize_sma_params(data, symbol: str = "BTC/USDT"):
    """Optimize SMA crossover strategy parameters.

    Tests different fast/sow period combinations.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimizing SMA Strategy - {symbol}")
    logger.info(f"{'='*60}")

    # Parameter grid
    fast_periods = [5, 10, 15, 20]
    slow_periods = [20, 30, 40, 50]

    results = []
    total = len(fast_periods) * len(slow_periods)
    current = 0

    for fast, slow in product(fast_periods, slow_periods):
        current += 1
        logger.info(f"[{current}/{total}] Testing fast={fast}, slow={slow}")

        if fast >= slow:
            logger.debug("Skipping: fast >= slow")
            continue

        try:
            engine = BacktestEngine(
                initial_capital=10000,
                commission=0.001
            )

            strategy = SimpleMovingAverageStrategy(fast_period=fast, slow_period=slow)
            result = engine.run_backtest(data, strategy, symbol)

            results.append({
                'fast': fast,
                'slow': slow,
                'return': result.total_return_pct,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown,
                'trades': len(result.trades),
                'win_rate': result.win_rate * 100
            })

        except Exception as e:
            logger.error(f"Error testing fast={fast}, slow={slow}: {e}")

    # Find best parameters
    if not results:
        logger.error("No results obtained")
        return

    # Sort by return
    best_return = max(results, key=lambda x: x['return'])
    best_sharpe = max(results, key=lambda x: x['sharpe'])

    # Print top 5 by return
    print(f"\n{'='*80}")
    print(f"SMA PARAMETER OPTIMIZATION RESULTS - {symbol}")
    print(f"{'='*80}")
    print(f"{'Fast':<8} {'Slow':<8} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<10} {'Win Rate %':<12} {'Trades':<8}")
    print("-"*80)

    top_by_return = sorted(results, key=lambda x: x['return'], reverse=True)[:10]
    for r in top_by_return:
        print(f"{r['fast']:<8} {r['slow']:<8} {r['return']:<12.2f} "
              f"{r['sharpe']:<10.2f} {r['max_dd']:<10.2f} "
              f"{r['win_rate']:<12.2f} {r['trades']:<8}")

    print("="*80)
    print(f"\n🏆 Best Return: fast={best_return['fast']}, slow={best_return['slow']}")
    print(f"   Return: {best_return['return']:.2f}%, Sharpe: {best_return['sharpe']:.2f}")
    print(f"   Max DD: {best_return['max_dd']:.2f}%, Trades: {best_return['trades']}")
    print(f"\n📊 Best Sharpe: fast={best_sharpe['fast']}, slow={best_sharpe['slow']}")
    print(f"   Sharpe: {best_sharpe['sharpe']:.2f}, Return: {best_sharpe['return']:.2f}%")

    return results


def optimize_rsi_params(data, symbol: str = "BTC/USDT"):
    """Optimize RSI strategy parameters.

    Tests different oversold/overbought combinations.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimizing RSI Strategy - {symbol}")
    logger.info(f"{'='*60}")

    # Parameter grid
    oversold_levels = [20, 25, 30, 35]
    overbought_levels = [65, 70, 75, 80]

    results = []
    total = len(oversold_levels) * len(overbought_levels)
    current = 0

    for oversold, overbought in product(oversold_levels, overbought_levels):
        current += 1
        logger.info(f"[{current}/{total}] Testing oversold={oversold}, overbought={overbought}")

        if oversold >= overbought:
            logger.debug("Skipping: oversold >= overbought")
            continue

        try:
            engine = BacktestEngine(
                initial_capital=10000,
                commission=0.001
            )

            strategy = RSIMeanReversionStrategy(
                oversold=oversold,
                overbought=overbought
            )
            result = engine.run_backtest(data, strategy, symbol)

            results.append({
                'oversold': oversold,
                'overbought': overbought,
                'return': result.total_return_pct,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown,
                'trades': len(result.trades),
                'win_rate': result.win_rate * 100
            })

        except Exception as e:
            logger.error(f"Error testing oversold={oversold}, overbought={overbought}: {e}")

    # Find best parameters
    if not results:
        logger.error("No results obtained")
        return

    # Sort by return
    best_return = max(results, key=lambda x: x['return'])
    best_sharpe = max(results, key=lambda x: x['sharpe'])

    # Print top 5 by return
    print(f"\n{'='*80}")
    print(f"RSI PARAMETER OPTIMIZATION RESULTS - {symbol}")
    print(f"{'='*80}")
    print(f"{'Oversold':<12} {'Overbought':<12} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<10} {'Win Rate %':<12} {'Trades':<8}")
    print("-"*80)

    top_by_return = sorted(results, key=lambda x: x['return'], reverse=True)[:10]
    for r in top_by_return:
        print(f"{r['oversold']:<12} {r['overbought']:<12} {r['return']:<12.2f} "
              f"{r['sharpe']:<10.2f} {r['max_dd']:<10.2f} "
              f"{r['win_rate']:<12.2f} {r['trades']:<8}")

    print("="*80)
    print(f"\n🏆 Best Return: oversold={best_return['oversold']}, overbought={best_return['overbought']}")
    print(f"   Return: {best_return['return']:.2f}%, Sharpe: {best_return['sharpe']:.2f}")
    print(f"   Max DD: {best_return['max_dd']:.2f}%, Trades: {best_return['trades']}")
    print(f"\n📊 Best Sharpe: oversold={best_sharpe['oversold']}, overbought={best_sharpe['overbought']}")
    print(f"   Sharpe: {best_sharpe['sharpe']:.2f}, Return: {best_sharpe['return']:.2f}%")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize backtesting strategy parameters"
    )
    parser.add_argument(
        "--file",
        help="CSV file to optimize on"
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Symbol name (default: BTC/USDT)"
    )
    parser.add_argument(
        "--strategy",
        choices=["sma", "rsi", "all"],
        default="all",
        help="Strategy to optimize (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data/)"
    )

    args = parser.parse_args()

    # Find data file
    if args.file:
        filepath = args.file
    else:
        # Find most recent file for symbol
        data_dir = Path(args.data_dir)
        symbol_clean = args.symbol.replace("/", "_")
        files = list(data_dir.glob(f"{symbol_clean}_*.csv"))

        if not files:
            logger.error(f"No data files found for {args.symbol}")
            logger.info("Run: python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1d --days 30 --save")
            return 1

        # Use most recent file
        filepath = str(max(files, key=lambda f: f.stat().st_mtime))

    try:
        # Load data
        data = load_csv_data(filepath)

        # Run optimizations
        if args.strategy in ["sma", "all"]:
            optimize_sma_params(data, args.symbol)

        if args.strategy in ["rsi", "all"]:
            optimize_rsi_params(data, args.symbol)

        logger.success("\n✅ Optimization complete!")

    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
