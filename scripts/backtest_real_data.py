#!/usr/bin/env python3
"""
Run backtests on real historical market data.

Usage:
    python scripts/backtest_real_data.py --symbol BTC/USDT
    python scripts/backtest_real_data.py --all
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphwiz_trader.backtesting import BacktestEngine, SimpleMovingAverageStrategy, RSIMeanReversionStrategy
from loguru import logger


def load_csv_data(filepath: str):
    """Load historical data from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        List of OHLCV dictionaries
    """
    logger.info(f"Loading data from {filepath}")

    # Read CSV
    df = pd.read_csv(filepath, parse_dates=['timestamp'])

    # Convert to list of dicts
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

    logger.info(f"Loaded {len(data)} candles from {data[0]['timestamp']} to {data[-1]['timestamp']}")
    return data


def run_backtest_on_file(csv_file: str, strategies: list = None):
    """Run backtest on a CSV file.

    Args:
        csv_file: Path to CSV file
        strategies: List of strategy names to run
    """
    if strategies is None:
        strategies = ['sma', 'rsi']

    # Load data
    try:
        data = load_csv_data(csv_file)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Extract symbol from filename
    filename = Path(csv_file).stem
    symbol = filename.replace('_', '/').replace(f"_{datetime.now().strftime('%Y%m%d')}", "")

    logger.info(f"\n{'='*60}")
    logger.info(f"Backtesting {symbol}")
    logger.info(f"{'='*60}")

    results = {}

    # Run each strategy
    for strategy_name in strategies:
        try:
            logger.info(f"\nRunning {strategy_name.upper()} strategy...")

            engine = BacktestEngine(
                initial_capital=10000,
                commission=0.001  # 0.1% Binance fee
            )

            # Select strategy
            if strategy_name == 'sma':
                strategy = SimpleMovingAverageStrategy(fast_period=10, slow_period=30)
            elif strategy_name == 'rsi':
                strategy = RSIMeanReversionStrategy(oversold=30, overbought=70)
            else:
                logger.warning(f"Unknown strategy: {strategy_name}")
                continue

            # Run backtest
            result = engine.run_backtest(data, strategy, symbol)
            results[strategy_name] = result

            # Print results
            print(f"\n{'='*60}")
            print(f"BACKTEST RESULTS: {strategy_name.upper()} - {symbol}")
            print(f"{'='*60}")
            print(f"Period:           {result.start_date} to {result.end_date}")
            print(f"Initial Capital:  ${result.initial_capital:,.2f}")
            print(f"Final Capital:    ${result.final_capital:,.2f}")
            print(f"-"*60)
            print(f"Total Return:     ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
            print(f"Win Rate:         {result.win_rate*100:.2f}%")
            print(f"Total Trades:     {len(result.trades)}")
            print(f"-"*60)
            print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
            print(f"Sortino Ratio:    {result.sortino_ratio:.2f}")
            print(f"Max Drawdown:     {result.max_drawdown:.2f}%")
            print(f"{'='*60}\n")

        except Exception as e:
            logger.exception(f"Error running {strategy_name} strategy: {e}")

    # Comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"STRATEGY COMPARISON - {symbol}")
        print(f"{'='*80}")
        print(f"{'Strategy':<15} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<10} {'Win Rate %':<12} {'Trades':<10}")
        print("-"*80)

        for name, result in results.items():
            print(f"{name.upper():<15} {result.total_return_pct:<12.2f} "
                  f"{result.sharpe_ratio:<10.2f} {result.max_drawdown:<10.2f} "
                  f"{result.win_rate*100:<12.2f} {len(result.trades):<10}")

        print("="*80)

        # Find best
        best_return = max(results.items(), key=lambda x: x[1].total_return_pct)
        best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)

        print(f"\n🏆 Best Return: {best_return[0].upper()} ({best_return[1].total_return_pct:.2f}%)")
        print(f"📊 Best Sharpe: {best_sharpe[0].upper()} ({best_sharpe[1].sharpe_ratio:.2f})")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest on real historical market data"
    )
    parser.add_argument(
        "--symbol",
        help="Specific symbol to backtest (e.g., BTC/USDT)"
    )
    parser.add_argument(
        "--file",
        help="Specific CSV file to backtest"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backtest all available data files"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["sma", "rsi"],
        choices=["sma", "rsi"],
        help="Strategies to run"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data/)"
    )

    args = parser.parse_args()

    # Find data files
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run: python scripts/fetch_data.py --save")
        return 1

    if args.file:
        # Specific file
        csv_files = [Path(args.file)]
    elif args.symbol:
        # Find files for symbol
        symbol_clean = args.symbol.replace("/", "_")
        csv_files = list(data_dir.glob(f"{symbol_clean}_*.csv"))
        if not csv_files:
            logger.error(f"No data files found for {args.symbol}")
            logger.info("Available files:")
            for f in data_dir.glob("*.csv"):
                logger.info(f"  {f.name}")
            return 1
    elif args.all:
        # All CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"No data files found in {data_dir}")
            logger.info("Run: python scripts/fetch_data.py --save")
            return 1
    else:
        # No arguments, show available files
        logger.info("Available data files:")
        for f in data_dir.glob("*.csv"):
            logger.info(f"  {f.name}")
        logger.info("\nUsage:")
        logger.info("  python scripts/backtest_real_data.py --symbol BTC/USDT")
        logger.info("  python scripts/backtest_real_data.py --all")
        return 0

    # Run backtests
    all_results = {}
    for csv_file in csv_files:
        logger.info(f"\n\nProcessing {csv_file.name}...")
        results = run_backtest_on_file(str(csv_file), args.strategies)
        all_results[csv_file.name] = results

    logger.success("\n✅ All backtests completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
