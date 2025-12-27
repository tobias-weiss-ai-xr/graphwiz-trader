"""
Interactive demo for 2025 Advanced Trading Strategies.

This demo showcases cutting-edge strategies based on latest research:
- Advanced Mean Reversion (5 types)
- Pairs Trading with PCA
- Momentum with Volatility Filtering
- Multi-Factor Models
- Confidence Threshold Framework

Run this demo to see the strategies in action with sample data.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import sys

from graphwiz_trader.strategies.advanced_strategies import (
    AdvancedMeanReversionStrategy,
    PairsTradingStrategy,
    MomentumVolatilityFilteringStrategy,
    MultiFactorStrategy,
    ConfidenceThresholdStrategy,
    MeanReversionType,
    create_advanced_strategy,
)


def generate_sample_data(symbol: str = 'BTC/USDT', days: int = 30) -> pd.DataFrame:
    """Generate realistic sample OHLCV data for testing.

    Args:
        symbol: Trading symbol
        days: Number of days of data to generate

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Generating {days} days of sample data for {symbol}...")

    np.random.seed(42)

    # Generate hourly candles
    periods = days * 24
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=periods,
        freq='1h'
    )

    # Simulate realistic price movements
    # with trends, mean reversion, and volatility changes
    base_price = 50000 if 'BTC' in symbol else 3000

    # Create price path with mean-reverting behavior
    price_path = []
    price = base_price

    for i in range(periods):
        # Add trend component
        trend = np.sin(i / 100) * 0.01  # Cyclical trend

        # Add mean reversion component
        reversion = (base_price - price) / base_price * 0.05

        # Add random noise
        noise = np.random.normal(0, 0.01)

        # Calculate return
        ret = trend + reversion + noise
        price = price * (1 + ret)
        price_path.append(price)

    # Create OHLCV DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [p * (1 + np.random.uniform(-0.003, 0.003)) for p in price_path],
        'high': [p * (1 + abs(np.random.uniform(0, 0.008))) for p in price_path],
        'low': [p * (1 - abs(np.random.uniform(0, 0.008))) for p in price_path],
        'close': price_path,
        'volume': np.random.uniform(100, 1000, periods),
    })

    df.set_index('timestamp', inplace=True)

    logger.info(f"Generated {len(df)} candles of data")
    logger.info(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")

    return df


def demo_mean_reversion():
    """Demonstrate advanced mean reversion strategies."""
    print("\n" + "="*80)
    print("DEMO 1: Advanced Mean Reversion Strategy")
    print("="*80)
    print("\nBased on 2025 research from Stoic.ai, OKX, and Robuxio")
    print("Tests 5 different mean reversion types\n")

    # Generate sample data
    df = generate_sample_data('BTC/USDT', days=30)

    # Test different mean reversion types
    types_to_test = [
        (MeanReversionType.BOLLINGER, "Bollinger Bands"),
        (MeanReversionType.RSI, "RSI-Based"),
        (MeanReversionType.ZSCORE, "Z-Score"),
        (MeanReversionType.ENVELOPE, "Moving Average Envelope"),
        (MeanReversionType.MULTI, "Multi-Indicator Combined"),
    ]

    results = {}

    for reversion_type, name in types_to_test:
        print(f"\n{'─'*80}")
        print(f"Testing: {name}")
        print(f"{'─'*80}")

        strategy = AdvancedMeanReversionStrategy(
            reversion_type=reversion_type,
            entry_threshold=2.0,
            exit_threshold=0.5,
            lookback_period=20,
            volatility_filter=True,
        )

        signals = strategy.generate_signals(df)

        # Analyze signals
        total_signals = signals['signal'].sum()
        total_exit_signals = signals['exit_signal'].sum()

        print(f"Total entry signals: {total_signals}")
        print(f"Total exit signals: {total_exit_signals}")

        # Show recent signals
        recent_signals = signals[signals['signal'] == 1].tail(5)
        if len(recent_signals) > 0:
            print(f"\nRecent entry signals:")
            for idx, row in recent_signals.iterrows():
                print(f"  {idx}: Price=${df.loc[idx, 'close']:,.2f}, "
                      f"Position Size={row.get('position_size', 'N/A')}")

        results[name] = {
            'entry_signals': total_signals,
            'exit_signals': total_exit_signals,
        }

    # Summary
    print(f"\n{'='*80}")
    print("MEAN REVERSION SUMMARY")
    print(f"{'='*80}")

    for name, stats in results.items():
        print(f"{name:30s}: {stats['entry_signals']:3d} entry, "
              f"{stats['exit_signals']:3d} exit signals")


def demo_pairs_trading():
    """Demonstrate pairs trading strategy with PCA."""
    print("\n" + "="*80)
    print("DEMO 2: Pairs Trading with PCA")
    print("="*80)
    print("\nBased on 2025 statistical arbitrage research")
    print("Uses PCA for cointegration-based pair selection\n")

    # Generate data for multiple symbols
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    price_data = {}

    print("Generating price data for multiple symbols...")
    for symbol in symbols:
        price_data[symbol] = generate_sample_data(symbol, days=30)['close']

    # Initialize strategy
    strategy = PairsTradingStrategy(
        lookback_period=30,
        entry_zscore=2.0,
        exit_zscore=0.0,
        n_components=3,
    )

    # Select best pairs
    print(f"\n{'─'*80}")
    print("Selecting trading pairs using PCA...")
    print(f"{'─'*80}")

    pairs = strategy.select_pairs(price_data)

    if len(pairs) > 0:
        print(f"\nFound {len(pairs)} correlated pairs:")
        for i, (sym1, sym2, corr) in enumerate(pairs[:5], 1):
            print(f"  {i}. {sym1:12s} ↔ {sym2:12s} (correlation: {corr:.4f})")

        # Generate signals for best pair
        best_pair = pairs[0][:2]
        print(f"\n{'─'*80}")
        print(f"Analyzing best pair: {best_pair[0]} ↔ {best_pair[1]}")
        print(f"{'─'*80}")

        # Create combined DataFrame
        df = pd.DataFrame({
            best_pair[0]: price_data[best_pair[0]],
            best_pair[1]: price_data[best_pair[1]],
        })

        signals = strategy.generate_signals(best_pair, df)

        # Calculate hedge ratio
        hedge_ratio = signals['hedge_ratio'].iloc[-1]
        print(f"\nHedge Ratio: {hedge_ratio:.4f}")
        print(f"  (1 unit of {best_pair[0]} = {hedge_ratio:.4f} units of {best_pair[1]})")

        # Show signals
        total_signals = signals['signal'].sum()
        print(f"\nTotal trading signals: {total_signals}")

        # Recent spread statistics
        recent_spread = signals['spread'].tail(20)
        recent_zscore = signals['spread_zscore'].tail(20)

        print(f"\nRecent spread statistics:")
        print(f"  Mean spread: ${recent_spread.mean():,.2f}")
        print(f"  Std spread: ${recent_spread.std():,.2f}")
        print(f"  Current z-score: {recent_zscore.iloc[-1]:.2f}")

        if abs(recent_zscore.iloc[-1]) > 2.0:
            print(f"  ⚠️  Current z-score exceeds entry threshold!")

    else:
        print("\n⚠️  No highly correlated pairs found in current data")


def demo_momentum_volatility():
    """Demonstrate momentum with volatility filtering."""
    print("\n" + "="*80)
    print("DEMO 3: Momentum with Volatility Filtering")
    print("="*80)
    print("\nBased on 2025 systematic crypto trading research")
    print("Only trades momentum when volatility is low\n")

    df = generate_sample_data('BTC/USDT', days=30)

    strategy = MomentumVolatilityFilteringStrategy(
        momentum_period=50,
        volatility_period=20,
        volatility_threshold=0.06,
        momentum_threshold=0.02,
    )

    print(f"Strategy Parameters:")
    print(f"  Momentum period: {strategy.momentum_period} hours")
    print(f"  Volatility period: {strategy.volatility_period} hours")
    print(f"  Volatility threshold: {strategy.volatility_threshold:.1%}")
    print(f"  Momentum threshold: {strategy.momentum_threshold:.1%}")

    signals = strategy.generate_signals(df)

    # Analyze results
    total_signals = signals['signal'].sum()

    print(f"\n{'─'*80}")
    print("Signal Analysis:")
    print(f"{'─'*80}")

    print(f"Total momentum signals: {total_signals}")

    # Calculate what percentage of time volatility was too high
    high_vol_periods = (signals['volatility'] > strategy.volatility_threshold).sum()
    total_periods = signals['volatility'].notna().sum()

    print(f"\nVolatility Filtering:")
    print(f"  High volatility periods: {high_vol_periods} ({high_vol_periods/total_periods*100:.1f}%)")
    print(f"  Low volatility periods: {total_periods - high_vol_periods} ({(total_periods-high_vol_periods)/total_periods*100:.1f}%)")

    # Show recent signals
    if total_signals > 0:
        recent_signals = signals[signals['signal'] == 1].tail(5)
        print(f"\nRecent momentum signals:")
        for idx, row in recent_signals.iterrows():
            print(f"  {idx}: Momentum={row['momentum']:.2%}, "
                  f"Volatility={row['volatility']:.2%}, "
                  f"Position Size={row['position_size']:.2f}")

    # Show momentum vs volatility relationship
    print(f"\n{'─'*80}")
    print("Momentum vs Volatility Analysis:")
    print(f"{'─'*80}")

    positive_momentum = signals['momentum'] > 0
    low_volatility = signals['volatility'] < strategy.volatility_threshold

    tradable_moments = (positive_momentum & low_volatility).sum()
    print(f"Positive momentum periods: {positive_momentum.sum()}")
    print(f"Low volatility periods: {low_volatility.sum()}")
    print(f"Tradable moments (both): {tradable_moments}")


def demo_multi_factor():
    """Demonstrate multi-factor strategy."""
    print("\n" + "="*80)
    print("DEMO 4: Multi-Factor Strategy")
    print("="*80)
    print("\nBased on 2025 multi-factor ML research (ACM)")
    print("Combines traditional + on-chain factors\n")

    df = generate_sample_data('BTC/USDT', days=30)

    strategy = MultiFactorStrategy(
        factors=['momentum', 'mean_reversion', 'volatility', 'volume', 'on_chain_activity'],
        factor_weights={
            'momentum': 0.3,
            'mean_reversion': 0.2,
            'volatility': 0.2,
            'volume': 0.15,
            'on_chain_activity': 0.15,
        },
    )

    print(f"Strategy Factors:")
    for factor, weight in strategy.factor_weights.items():
        print(f"  {factor:20s}: {weight:.2%}")

    # Calculate factors
    factors_df = strategy.calculate_factors(df)

    print(f"\n{'─'*80}")
    print("Recent Factor Values (Normalized):")
    print(f"{'─'*80}")

    recent_factors = factors_df.tail(5)
    for idx, row in recent_factors.iterrows():
        if idx == recent_factors.index[0]:
            print(f"\n{idx}:")
            for factor in strategy.factors:
                norm_factor = f'{factor}_norm'
                if norm_factor in row:
                    print(f"  {factor:20s}: {row[norm_factor]:.3f}")

    # Generate signals
    signals = strategy.generate_signals(df)

    print(f"\n{'─'*80}")
    print("Signal Analysis:")
    print(f"{'─'*80}")

    total_signals = signals['signal'].sum()
    print(f"Total signals: {total_signals}")

    # Show factor score distribution
    print(f"\nFactor Score Statistics:")
    print(f"  Mean: {signals['factor_score'].mean():.3f}")
    print(f"  Std: {signals['factor_score'].std():.3f}")
    print(f"  Min: {signals['factor_score'].min():.3f}")
    print(f"  Max: {signals['factor_score'].max():.3f}")

    # Show top scoring periods
    top_scores = signals['factor_score'].nlargest(5)
    print(f"\nTop 5 Factor Scores:")
    for idx, score in top_scores.items():
        print(f"  {idx}: {score:.3f}")


def demo_confidence_threshold():
    """Demonstrate confidence threshold strategy."""
    print("\n" + "="*80)
    print("DEMO 5: Confidence Threshold Framework")
    print("="*80)
    print("\nBased on 2025 confidence threshold research (MDPI)")
    print("Dynamically adjusts threshold based on performance\n")

    # Test different modes
    modes = ['conservative', 'normal', 'aggressive']

    for mode in modes:
        strategy = ConfidenceThresholdStrategy(
            base_threshold=0.6,
            mode=mode,
        )

        print(f"\n{mode.capitalize()} Mode:")
        print(f"  Base Threshold: {strategy.threshold:.2f}")

        # Simulate threshold adjustment
        recent_performance = pd.Series([
            0.01, 0.02, 0.015, 0.01, -0.005, 0.02, 0.01, 0.015, 0.02, 0.01
        ])

        # Test with different volatility regimes
        volatilities = [0.01, 0.05, 0.10]

        print(f"  Dynamic Threshold Adjustment:")
        for vol in volatilities:
            adjusted = strategy.adjust_threshold(
                recent_performance=recent_performance,
                market_volatility=vol,
            )
            print(f"    Volatility {vol:.2%}: threshold = {adjusted:.3f}")

    # Demonstrate threshold evolution
    print(f"\n{'─'*80}")
    print("Simulated Threshold Evolution Over Time:")
    print(f"{'─'*80}")

    strategy = ConfidenceThresholdStrategy(base_threshold=0.6, mode='normal')

    # Simulate trading performance over time
    performance_windows = []
    thresholds = []

    for i in range(20):
        # Generate random performance
        perf = pd.Series([np.random.normal(0.01, 0.02) for _ in range(10)])
        vol = np.random.uniform(0.02, 0.08)

        threshold = strategy.adjust_threshold(perf, vol)

        performance_windows.append(perf.mean())
        thresholds.append(threshold)

    print(f"  Initial threshold: {thresholds[0]:.3f}")
    print(f"  Final threshold: {thresholds[-1]:.3f}")
    print(f"  Average threshold: {np.mean(thresholds):.3f}")
    print(f"  Threshold range: {np.min(thresholds):.3f} - {np.max(thresholds):.3f}")


def demo_comparison():
    """Compare all strategies side by side."""
    print("\n" + "="*80)
    print("DEMO 6: Strategy Comparison")
    print("="*80)
    print("\nComparing signal generation across all strategies\n")

    df = generate_sample_data('BTC/USDT', days=30)

    strategies = {
        'Mean Reversion (Bollinger)': create_advanced_strategy(
            "mean_reversion",
            reversion_type=MeanReversionType.BOLLINGER,
        ),
        'Momentum + Vol Filter': create_advanced_strategy(
            "momentum_volatility",
        ),
        'Multi-Factor': create_advanced_strategy(
            "multi_factor",
        ),
    }

    print(f"{'─'*80}")
    print(f"{'Strategy':30s} {'Signals':>10s} {'Exit Signals':>15s}")
    print(f"{'─'*80}")

    for name, strategy in strategies.items():
        signals = strategy.generate_signals(df)

        total_signals = signals['signal'].sum()
        total_exits = signals.get('exit_signal', pd.Series([0])).sum()

        print(f"{name:30s} {total_signals:>10d} {total_exits:>15d}")

    print(f"{'─'*80}")


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print(" "*15 + "2025 ADVANCED TRADING STRATEGIES DEMO")
    print("="*80)
    print("\nThis demo showcases cutting-edge strategies based on latest research:")
    print("  • Advanced Mean Reversion (5 types)")
    print("  • Pairs Trading with PCA")
    print("  • Momentum with Volatility Filtering")
    print("  • Multi-Factor Models")
    print("  • Confidence Threshold Framework")
    print("\n" + "="*80)

    try:
        # Run each demo
        demo_mean_reversion()
        demo_pairs_trading()
        demo_momentum_volatility()
        demo_multi_factor()
        demo_confidence_threshold()
        demo_comparison()

        print("\n" + "="*80)
        print(" "*25 + "DEMO COMPLETED")
        print("="*80)
        print("\nAll strategies demonstrated successfully!")
        print("\nNext Steps:")
        print("  1. Run tests: pytest tests/integration/test_advanced_strategies.py")
        print("  2. Integrate with your trading engine")
        print("  3. Backtest with historical data")
        print("  4. Optimize parameters for your symbols")
        print("  5. Deploy to paper trading for validation")

    except Exception as e:
        logger.exception(f"Error running demo: {e}")
        return 1

    return 0


if __name__ == '__main__':
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Run demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
