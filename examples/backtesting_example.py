#!/usr/bin/env python3
"""
Example backtesting script for graphwiz-trader.

This script demonstrates how to use the backtesting framework
to test different trading strategies.
"""

from datetime import datetime, timedelta

from graphwiz_trader.backtesting import (
    BacktestEngine,
    MomentumStrategy,
    MeanReversionStrategy,
    GridTradingStrategy,
    DCAStrategy,
)


def main():
    """Run example backtests."""
    print("=" * 60)
    print("Graphwiz Trader - Backtesting Example")
    print("=" * 60)

    # Initialize backtest engine
    engine = BacktestEngine(
        config_path="/opt/git/graphwiz-trader/config/backtesting.yaml",
        output_dir="/opt/git/graphwiz-trader/backtests",
    )

    # Define backtest parameters
    symbol = "BTC/USDT"
    start_date = datetime.now() - timedelta(days=90)  # Last 90 days
    end_date = datetime.now()
    timeframe = "1h"
    initial_capital = 10000.0

    print(f"\nBacktest Parameters:")
    print(f"  Symbol: {symbol}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print()

    # Initialize strategies
    strategies = [
        MomentumStrategy(
            lookback_period=20,
            momentum_threshold=0.02,
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005,
        ),
        MeanReversionStrategy(
            lookback_period=20,
            std_dev_threshold=2.0,
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005,
        ),
        GridTradingStrategy(
            grid_levels=10,
            grid_spacing=0.01,
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005,
        ),
        DCAStrategy(
            purchase_interval="1D",
            purchase_amount=500.0,
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005,
        ),
    ]

    # Run backtests
    print("Running backtests...\n")
    results = engine.run_multiple_backtests(
        strategies=strategies,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        initial_capital=initial_capital,
    )

    # Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    for strategy_name, result in results.items():
        metrics = result["metrics"]

        print(f"\n{strategy_name}:")
        print(f"  Final Capital: ${result['final_capital']:,.2f}")
        print(f"  Total Return: {metrics.total_return * 100:,.2f}%")
        print(f"  Annualized Return: {metrics.annualized_return * 100:,.2f}%")
        print(f"  Volatility: {metrics.volatility * 100:,.2f}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
        print(f"  Max Drawdown: {metrics.max_drawdown * 100:,.2f}%")
        print(f"  Win Rate: {metrics.win_rate * 100:.2f}%")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Profit Factor: {metrics.profit_factor:.3f}")

    # Find best strategy by Sharpe ratio
    best_strategy = engine.get_best_strategy(metric="sharpe_ratio")
    print(f"\nBest strategy by Sharpe ratio: {best_strategy}")

    # Generate reports
    print("\nGenerating reports...")
    for strategy_name in results.keys():
        try:
            engine.generate_report(strategy_name)
            print(f"  Generated report for {strategy_name}")
        except Exception as e:
            print(f"  Error generating report for {strategy_name}: {e}")

    # Generate comparison report
    try:
        engine.generate_comparison_report()
        print("  Generated comparison report")
    except Exception as e:
        print(f"  Error generating comparison report: {e}")

    # Save results
    try:
        engine.save_results()
        print("  Saved results to file")
    except Exception as e:
        print(f"  Error saving results: {e}")

    print("\n" + "=" * 60)
    print("Backtesting complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
