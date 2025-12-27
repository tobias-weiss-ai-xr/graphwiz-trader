"""Quick start demo for Qlib Phase 2: Portfolio Optimization.

This script demonstrates the new Phase 2 features:
- Portfolio optimization
- Dynamic position sizing
- Advanced backtesting
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphwiz_trader.qlib import (
    PortfolioOptimizer,
    DynamicPositionSizer,
    PortfolioConstraints,
    OptimizerConfig,
    BacktestEngine,
    BacktestConfig,
    create_portfolio_optimizer,
)


async def demo_portfolio_optimization():
    """Demonstrate portfolio optimization."""
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     Portfolio Optimization Demo                        ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Generate sample returns data
    np.random.seed(42)
    n_assets = 5
    n_periods = 252

    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.02,
        columns=['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT'],
    )

    logger.info(f"\nGenerated returns data for {n_assets} assets over {n_periods} periods")

    # Test different optimization methods
    methods = ['equal_weight', 'mean_variance', 'max_sharpe', 'min_variance', 'risk_parity']

    logger.info("\n" + "─" * 60)
    logger.info("Comparing Optimization Methods")
    logger.info("─" * 60)

    results = {}

    for method in methods:
        optimizer = create_portfolio_optimizer(
            method=method,
            constraints=PortfolioConstraints(
                max_position_weight=0.4,
            ),
        )

        weights = optimizer.optimize(returns)
        metrics = optimizer.calculate_portfolio_metrics(weights, returns)

        results[method] = {
            'weights': weights,
            'return': metrics['annualized_return'],
            'volatility': metrics['volatility'],
            'sharpe': metrics['sharpe_ratio'],
        }

        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Weights: {dict(weights.round(3))}")
        logger.info(f"  Return:  {metrics['annualized_return']:>8.2%}")
        logger.info(f"  Risk:    {metrics['volatility']:>8.2%}")
        logger.info(f"  Sharpe:  {metrics['sharpe_ratio']:>8.2f}")

    # Find best Sharpe ratio
    best_method = max(results.keys(), key=lambda k: results[k]['sharpe'])
    logger.info(f"\n✓ Best method by Sharpe ratio: {best_method}")

    return True


async def demo_dynamic_position_sizing():
    """Demonstrate dynamic position sizing."""
    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║     Dynamic Position Sizing Demo                       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    sizer = DynamicPositionSizer(
        base_position_size=0.1,
        max_position_size=0.3,
        min_position_size=0.05,
        risk_tolerance=0.02,
    )

    logger.info("\nPosition Sizes for Different Scenarios:")
    logger.info("─" * 60)

    # Test scenarios
    portfolio_value = 100000
    asset_price = 50000

    scenarios = [
        ("Very High Confidence", 0.95, 0.2),
        ("High Confidence", 0.80, 0.3),
        ("Medium Confidence", 0.65, 0.4),
        ("Low Confidence", 0.55, 0.5),
        ("Very Low Confidence", 0.51, 0.6),
    ]

    for name, confidence, volatility in scenarios:
        size = sizer.calculate_position_size(
            signal_confidence=confidence,
            portfolio_value=portfolio_value,
            asset_price=asset_price,
            asset_volatility=volatility,
        )

        pct_of_portfolio = size / portfolio_value

        logger.info(f"{name:20s} | Conf: {confidence:.2%} | Vol: {volatility:.2%} → ${size:>7,.0f} ({pct_of_portfolio:>5.1%})")

    logger.info("\n✓ Position sizes adapt to confidence and risk!")

    return True


async def demo_backtesting():
    """Demonstrate advanced backtesting."""
    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║     Advanced Backtesting Demo                          ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Generate sample price data
    np.random.seed(42)
    n_periods = 500

    returns = np.random.randn(n_periods) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate a simple strategy
    signals = pd.Series(0, index=pd.date_range(start='2024-01-01', periods=n_periods, freq='1h'))

    # Buy when price is below moving average
    ma = prices.rolling(50).mean()
    signals[prices < ma] = 1  # Buy signal

    price_data = pd.DataFrame({'close': prices}, index=signals.index)

    logger.info(f"\nGenerated sample data:")
    logger.info(f"  Periods: {n_periods}")
    logger.info(f"  Buy signals: {(signals == 1).sum()}")
    logger.info(f"  Sell signals: {(signals == 0).sum()}")

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
    )

    engine = BacktestEngine(config=config)

    logger.info("\nRunning backtest...")
    result = engine.run_backtest(signals, price_data)

    # Display key metrics
    logger.info("\n" + "─" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("─" * 60)
    logger.info(f"Total Return:         {result.total_return:>10.2%}")
    logger.info(f"Annualized Return:    {result.annualized_return:>10.2%}")
    logger.info(f"Volatility:           {result.volatility:>10.2%}")
    logger.info(f"Sharpe Ratio:         {result.sharpe_ratio:>10.2f}")
    logger.info(f"Sortino Ratio:        {result.sortino_ratio:>10.2f}")
    logger.info(f"Max Drawdown:         {result.max_drawdown:>10.2%}")
    logger.info(f"Win Rate:             {result.win_rate:>10.2%}")
    logger.info(f"Profit Factor:        {result.profit_factor:>10.2f}")
    logger.info(f"Total Trades:         {result.total_trades:>10}")

    logger.info("\n✓ Backtesting complete!")

    return True


async def demo_end_to_end():
    """Demonstrate end-to-end workflow."""
    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║     End-to-End Workflow Demo                            ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Step 1: Generate market data
    logger.info("\n[Step 1] Generating market data for 5 assets...")
    np.random.seed(42)
    n_periods = 252

    assets = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    returns = pd.DataFrame(
        np.random.randn(n_periods, len(assets)) * 0.02,
        columns=assets,
    )

    logger.info(f"✓ Generated {n_periods} periods of returns for {len(assets)} assets")

    # Step 2: Optimize portfolio
    logger.info("\n[Step 2] Optimizing portfolio weights...")
    optimizer = create_portfolio_optimizer(
        method='max_sharpe',
        constraints=PortfolioConstraints(max_position_weight=0.4),
    )

    weights = optimizer.optimize(returns)
    logger.info(f"✓ Optimal weights: {dict(weights.round(3))}")

    # Step 3: Calculate position sizes
    logger.info("\n[Step 3] Calculating position sizes...")
    portfolio_value = 100000
    sizer = DynamicPositionSizer(base_position_size=0.1)

    position_sizes = {}
    for asset in assets:
        size = sizer.calculate_position_size_by_weight(
            optimal_weight=weights[asset],
            portfolio_value=portfolio_value,
            asset_price=50000,  # Simplified
        )
        position_sizes[asset] = size

    logger.info(f"✓ Position sizes (from ${portfolio_value:,}):")
    for asset, size in position_sizes.items():
        logger.info(f"    {asset}: ${size:>8,.2f}")

    # Step 4: Calculate expected performance
    logger.info("\n[Step 4] Calculating expected performance...")
    metrics = optimizer.calculate_portfolio_metrics(weights, returns)

    logger.info(f"✓ Expected portfolio performance:")
    logger.info(f"    Annualized Return: {metrics['annualized_return']:>8.2%}")
    logger.info(f"    Volatility:        {metrics['volatility']:>8.2%}")
    logger.info(f"    Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
    logger.info(f"    Max Drawdown:      {metrics['max_drawdown']:>8.2%}")

    logger.info("\n✓ End-to-end workflow complete!")

    return True


async def main():
    """Run all demos."""
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║                                                          ║")
    logger.info("║         QLIB PHASE 2 - PORTFOLIO OPTIMIZATION            ║")
    logger.info("║                     QUICK START DEMO                     ║")
    logger.info("║                                                          ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    demos = [
        ("Portfolio Optimization", demo_portfolio_optimization),
        ("Dynamic Position Sizing", demo_dynamic_position_sizing),
        ("Advanced Backtesting", demo_backtesting),
        ("End-to-End Workflow", demo_end_to_end),
    ]

    results = []

    for name, demo_func in demos:
        try:
            success = await demo_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"\n✗ {name} demo failed: {e}")
            results.append((name, False))

    # Summary
    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║                        SUMMARY                           ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")

    passed = sum(1 for _, s in results if s)
    logger.info(f"\nTotal: {passed}/{len(results)} demos completed successfully")

    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║                    Demo Complete!                       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    logger.info("\nNext Steps:")
    logger.info("  1. Run the full test suite:")
    logger.info("     python tests/integration/test_qlib_phase2.py")
    logger.info("\n  2. Use in your strategy:")
    logger.info("     from graphwiz_trader.strategies import create_qlib_strategy_v2")
    logger.info("\n  3. Read the documentation:")
    logger.info("     docs/QLIB_PHASE2_DOCUMENTATION.md")


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Run demos
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"\n\nDemo failed with error: {e}")
        sys.exit(1)
