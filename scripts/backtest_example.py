#!/usr/bin/env python3
"""
Example backtesting script for GraphWiz Trader.

This script demonstrates how to:
1. Fetch historical data
2. Define trading strategies
3. Run backtests
4. Analyze results
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphwiz_trader.backtesting import BacktestEngine, SimpleMovingAverageStrategy, RSIMeanReversionStrategy
from graphwiz_trader.analysis import TechnicalAnalysis


def fetch_sample_data(days: int = 30, symbol: str = "BTC/USDT"):
    """Fetch sample historical data.

    In production, you would fetch from an exchange API.
    For now, we'll generate synthetic data.
    """
    logger.info(f"Generating {days} days of sample data for {symbol}")

    import numpy as np

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate realistic price movement
    data = []
    base_price = 50000 if "BTC" in symbol else 3000
    now = datetime.now()

    # Simulate trending market with noise
    trend = 0.0001  # Slight uptrend
    volatility = 0.02  # 2% daily volatility

    for i in range(days * 24 * 60):  # Minute data
        if i % 10000 == 0:
            logger.debug(f"Generated {i} candles...")

        # Random walk with trend
        return_pct = np.random.normal(trend, volatility / np.sqrt(1440))

        open_price = base_price * (1 + return_pct)
        high = open_price * (1 + abs(np.random.normal(0, 0.005)))
        low = open_price * (1 - abs(np.random.normal(0, 0.005)))
        close = open_price * (1 + np.random.normal(0, 0.001))
        volume = 1000000 + np.random.normal(0, 100000)

        data.append({
            "timestamp": now - timedelta(minutes=days*24*60 - i),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": max(0, volume)
        })

        base_price = close

    logger.info(f"Generated {len(data)} candles")
    return data


def custom_strategy(context):
    """Custom trading strategy example.

    This strategy combines:
    - RSI for overbought/oversold conditions
    - SMA crossover for trend direction
    - Volume confirmation

    Returns: 'buy', 'sell', or 'hold'
    """
    indicators = context.get('technical_indicators', {})

    if not indicators:
        return 'hold'

    rsi = indicators.get('rsi', {}).get('value')
    sma_fast = indicators.get('sma_fast', {}).get('value')
    sma_slow = indicators.get('sma_slow', {}).get('value')
    volume = context.get('volume', 0)
    avg_volume = context.get('avg_volume', 1)

    # Buy signal: RSI oversold and price below SMA
    if rsi and rsi < 30 and sma_fast and sma_slow:
        if sma_fast > sma_slow and volume > avg_volume * 1.2:
            return 'buy'

    # Sell signal: RSI overbought or price below stop loss
    if rsi and rsi > 70:
        return 'sell'

    if sma_fast and sma_slow and sma_fast < sma_slow * 0.98:
        return 'sell'

    return 'hold'


def run_simple_backtest(symbol: str = "BTC/USDT", days: int = 30):
    """Run a simple backtest with built-in SMA strategy."""
    logger.info(f"=== Running Simple Backtest for {symbol} ===")

    # Fetch data
    data = fetch_sample_data(days=days, symbol=symbol)

    # Initialize engine
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001  # 0.1% Binance fee
    )

    # Run with built-in SMA strategy
    logger.info("Running SMA Crossover strategy...")
    strategy = SimpleMovingAverageStrategy(fast_period=10, slow_period=30)
    result = engine.run_backtest(data, strategy, symbol)

    # Print results
    print_results(result, "SMA Crossover")

    return result


def run_rsi_backtest(symbol: str = "BTC/USDT", days: int = 30):
    """Run backtest with RSI mean reversion strategy."""
    logger.info(f"=== Running RSI Backtest for {symbol} ===")

    # Fetch data
    data = fetch_sample_data(days=days, symbol=symbol)

    # Initialize engine
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001
    )

    # Run with RSI strategy
    logger.info("Running RSI Mean Reversion strategy...")
    strategy = RSIMeanReversionStrategy(oversold=30, overbought=70)
    result = engine.run_backtest(data, strategy, symbol)

    # Print results
    print_results(result, "RSI Mean Reversion")

    return result


def run_custom_backtest(symbol: str = "BTC/USDT", days: int = 30):
    """Run backtest with custom strategy."""
    logger.info(f"=== Running Custom Backtest for {symbol} ===")

    # Fetch data
    data = fetch_sample_data(days=days, symbol=symbol)

    # Initialize engine
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001
    )

    # Run with custom strategy
    logger.info("Running custom strategy...")
    result = engine.run_backtest(data, custom_strategy, symbol)

    # Print results
    print_results(result, "Custom Strategy")

    return result


def print_results(result, strategy_name: str):
    """Pretty print backtest results."""
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS: {strategy_name}")
    print("="*60)
    print(f"Symbol:           {result.symbol}")
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
    print("="*60)

    # Show trades if not too many
    if len(result.trades) <= 10:
        print("\nTrade History:")
        for i, trade in enumerate(result.trades, 1):
            print(f"  {i}. {trade.action.upper()} {trade.quantity:.4f} @ ${trade.price:.2f}")

    # Show equity curve summary
    if 'equity_curve' in result.metrics:
        equity = result.metrics['equity_curve']
        print(f"\nEquity Curve:")
        print(f"  Start:  ${equity[0]:,.2f}")
        print(f"  End:    ${equity[-1]:,.2f}")
        print(f"  Peak:   ${max(equity):,.2f}")
        print(f"  Low:    ${min(equity):,.2f}")


def compare_strategies(symbol: str = "BTC/USDT", days: int = 30):
    """Compare multiple strategies side by side."""
    logger.info(f"=== Comparing Strategies for {symbol} ===")

    results = {}

    # Run all strategies
    results['SMA'] = run_simple_backtest(symbol, days)
    results['RSI'] = run_rsi_backtest(symbol, days)
    results['Custom'] = run_custom_backtest(symbol, days)

    # Comparison table
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(f"{'Strategy':<15} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<10} {'Win Rate %':<12}")
    print("-"*80)

    for name, result in results.items():
        print(f"{name:<15} {result.total_return_pct:<12.2f} "
              f"{result.sharpe_ratio:<10.2f} {result.max_drawdown:<10.2f} "
              f"{result.win_rate*100:<12.2f}")

    print("="*80)

    # Find best strategy
    best = max(results.items(), key=lambda x: x[1].total_return_pct)
    print(f"\n🏆 Best Return: {best[0]} ({best[1].total_return_pct:.2f}%)")

    best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"📊 Best Sharpe: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GraphWiz Trader Backtesting")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--strategy", choices=["sma", "rsi", "custom", "all"],
                      default="all", help="Strategy to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    try:
        if args.strategy == "all":
            # Compare all strategies
            compare_strategies(args.symbol, args.days)
        elif args.strategy == "sma":
            run_simple_backtest(args.symbol, args.days)
        elif args.strategy == "rsi":
            run_rsi_backtest(args.symbol, args.days)
        elif args.strategy == "custom":
            run_custom_backtest(args.symbol, args.days)

        logger.success("Backtesting completed!")

    except Exception as e:
        logger.exception("Backtesting failed: {}", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
