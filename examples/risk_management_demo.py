#!/usr/bin/env python3
"""Risk management system demonstration.

This script demonstrates the key features of the risk management system.
Note: This is a demonstration script and requires all dependencies to be installed.
"""

import sys
sys.path.insert(0, '/opt/git/graphwiz-trader/src')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# These imports will work when dependencies are installed
try:
    from graphwiz_trader.risk import (
        RiskManager,
        RiskLimits,
        RiskLimitsConfig,
        StopLossCalculator,
        calculate_position_size,
        calculate_portfolio_risk,
        calculate_correlation_matrix,
        calculate_max_drawdown,
        PositionSizingStrategy,
    )
    from graphwiz_trader.risk.alerts import AlertSeverity, AlertType
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please install required dependencies:")
    print("  pip install numpy pandas scipy loguru aiohttp")
    sys.exit(1)


def demo_position_sizing():
    """Demonstrate position sizing strategies."""
    print("\n" + "=" * 60)
    print("POSITION SIZING DEMO")
    print("=" * 60)

    account_balance = 100000.0
    entry_price = 50000.0  # BTC
    stop_loss_price = 49000.0  # 2% stop loss

    print(f"\nAccount Balance: ${account_balance:,.2f}")
    print(f"Entry Price: ${entry_price:,.2f}")
    print(f"Stop Loss: ${stop_loss_price:,.2f} ({(stop_loss_price/entry_price - 1)*100:.2f}%)")

    # Fixed Fractional
    result = calculate_position_size(
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        risk_per_trade=0.02,
        strategy=PositionSizingStrategy.FIXED_FRACTIONAL,
    )
    print(f"\nFixed Fractional (2% risk):")
    print(f"  Position Size: {result['position_size']:.4f} BTC")
    print(f"  Position Value: ${result['position_value']:,.2f}")
    print(f"  Dollar Risk: ${result['dollar_risk']:,.2f}")

    # Kelly Criterion
    result = calculate_position_size(
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        risk_per_trade=0.02,
        strategy=PositionSizingStrategy.KELLY_CRITERION,
        strategy_params={
            "win_rate": 0.55,
            "avg_win": 1.5,
            "avg_loss": 1.0,
            "kelly_fraction": 0.5,
        },
    )
    print(f"\nKelly Criterion (55% win rate, 1.5:1 R:R, half-Kelly):")
    print(f"  Position Size: {result['position_size']:.4f} BTC")
    print(f"  Position Value: ${result['position_value']:,.2f}")
    print(f"  Kelly Percentage: {result['kelly_percentage']:.2%}")


def demo_stop_loss_calculator():
    """Demonstrate stop-loss calculator."""
    print("\n" + "=" * 60)
    print("STOP-LOSS CALCULATOR DEMO")
    print("=" * 60)

    calculator = StopLossCalculator(
        default_stop_loss_pct=0.02,
        default_take_profit_pct=0.06,
        risk_reward_ratio=3.0,
    )

    entry_price = 100.0
    stop_loss_price = 98.0

    print(f"\nEntry Price: ${entry_price:.2f}")
    print(f"Stop Loss Price: ${stop_loss_price:.2f}")
    print(f"Risk per Share: ${entry_price - stop_loss_price:.2f} ({(1 - stop_loss_price/entry_price)*100:.2f}%)")

    # Calculate take profit
    take_profit = calculator.calculate_take_profit(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        side="long",
    )
    print(f"\nTake Profit (3:1 R:R): ${take_profit:.2f}")
    print(f"Reward: ${take_profit - entry_price:.2f} ({(take_profit/entry_price - 1)*100:.2f}%)")

    # Calculate position size from risk
    account_balance = 100000.0
    risk_per_trade = 0.02
    position_size = calculator.calculate_position_size_from_risk(
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        risk_per_trade_pct=risk_per_trade,
    )
    print(f"\nPosition Size (2% risk on ${account_balance:,.0f} account):")
    print(f"  {position_size:.0f} shares")
    print(f"  Total Value: ${position_size * entry_price:,.2f}")


def demo_correlation_analysis():
    """Demonstrate correlation analysis."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS DEMO")
    print("=" * 60)

    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Create correlated returns
    returns = np.random.multivariate_normal(
        mean=[0.001, 0.0005, 0.0008],
        cov=[[0.02, 0.01, 0.005], [0.01, 0.015, 0.003], [0.005, 0.003, 0.01]],
        size=100,
    )

    prices = pd.DataFrame(
        (1 + returns).cumprod(axis=0) * 100,
        index=dates,
        columns=["BTC", "ETH", "SOL"],
    )

    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(prices, method="pearson")

    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))

    print("\nKey Correlations:")
    print(f"  BTC-ETH: {corr_matrix.loc['BTC', 'ETH']:.3f}")
    print(f"  BTC-SOL: {corr_matrix.loc['BTC', 'SOL']:.3f}")
    print(f"  ETH-SOL: {corr_matrix.loc['ETH', 'SOL']:.3f}")


def demo_drawdown_analysis():
    """Demonstrate drawdown analysis."""
    print("\n" + "=" * 60)
    print("DRAWDOWN ANALYSIS DEMO")
    print("=" * 60)

    # Create price series with known patterns
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    # Simulate price path
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * (1 + returns).cumprod()
    price_series = pd.Series(prices, index=dates)

    # Calculate drawdown
    drawdown_analysis = calculate_max_drawdown(price_series)

    print(f"\nStarting Price: ${prices[0]:.2f}")
    print(f"Ending Price: ${prices[-1]:.2f}")
    print(f"Total Return: {(prices[-1]/prices[0] - 1)*100:.2f}%")

    print(f"\nMaximum Drawdown:")
    print(f"  Depth: {drawdown_analysis['max_drawdown']:.2%}")
    print(f"  Absolute: ${drawdown_analysis['max_drawdown_abs']:.2f}")
    print(f"  Duration: {drawdown_analysis['max_drawdown_duration']} days")
    print(f"  Peak Date: {drawdown_analysis['peak_date'].strftime('%Y-%m-%d')}")
    print(f"  Trough Date: {drawdown_analysis['trough_date'].strftime('%Y-%m-%d')}")
    print(f"  Current Drawdown: {drawdown_analysis['current_drawdown']:.2%}")


def demo_risk_manager():
    """Demonstrate RiskManager."""
    print("\n" + "=" * 60)
    print("RISK MANAGER DEMO")
    print("=" * 60)

    # Create risk manager with custom limits
    config = RiskLimitsConfig(
        max_position_size=0.20,  # 20% max per position
        max_total_exposure=1.0,  # 100% max exposure
        max_daily_loss_pct=0.10,  # 10% max daily loss
        max_correlated_exposure=0.40,  # 40% max correlated exposure
    )

    rm = RiskManager(
        account_balance=100000.0,
        limits_config=config,
    )

    print(f"\nInitial Account Balance: ${rm.account_balance:,.2f}")

    # Add positions
    print("\n--- Adding Positions ---")

    try:
        pos1 = rm.add_position(
            symbol="BTC",
            quantity=1.0,
            entry_price=50000.0,
            side="long",
            sector="Crypto",
        )
        print(f"✓ Added BTC: 1.0 @ $50,000")
    except ValueError as e:
        print(f"✗ Failed to add BTC: {e}")

    try:
        pos2 = rm.add_position(
            symbol="ETH",
            quantity=10.0,
            entry_price=3000.0,
            side="long",
            sector="Crypto",
        )
        print(f"✓ Added ETH: 10.0 @ $3,000")
    except ValueError as e:
        print(f"✗ Failed to add ETH: {e}")

    # Try to add position that exceeds limit
    print("\n--- Testing Limits ---")
    try:
        pos3 = rm.add_position(
            symbol="SOL",
            quantity=1000.0,  # This would exceed limits
            entry_price=100.0,
            side="long",
            sector="Crypto",
        )
        print(f"✓ Added SOL: 1000.0 @ $100")
    except ValueError as e:
        print(f"✗ SOL rejected (expected): {str(e)[:80]}...")

    # Get portfolio state
    state = rm.get_portfolio_state()
    print(f"\n--- Portfolio State ---")
    print(f"Total Value: ${state.total_value:,.2f}")
    print(f"Cash Balance: ${state.cash_balance:,.2f}")
    print(f"Number of Positions: {len(state.positions)}")
    print(f"Daily P&L: ${state.daily_pnl:,.2f}")

    # Calculate position size for new trade
    print(f"\n--- Calculating Position Size ---")
    try:
        size_result = rm.calculate_position_size(
            symbol="BTC",
            entry_price=52000.0,
            stop_loss_price=50960.0,  # 2% stop loss
            strategy=PositionSizingStrategy.FIXED_FRACTIONAL,
        )
        print(f"Proposed BTC position:")
        print(f"  Size: {size_result['position_size']:.4f} BTC")
        print(f"  Value: ${size_result['position_value']:,.2f}")
        print(f"  Risk: ${size_result['dollar_risk']:,.2f}")
    except ValueError as e:
        print(f"Cannot size position: {e}")

    # Update prices
    print(f"\n--- Updating Prices ---")
    rm.update_position_price("BTC", 51000.0)
    rm.update_position_price("ETH", 3100.0)
    print("✓ Updated BTC to $51,000")
    print("✓ Updated ETH to $3,100")

    state = rm.get_portfolio_state()
    print(f"Unrealized P&L: ${state.unrealized_pnl:,.2f}")

    # Close position
    print(f"\n--- Closing Position ---")
    pnl = rm.close_position("BTC")
    print(f"Closed BTC P&L: ${pnl:,.2f}")

    state = rm.get_portfolio_state()
    print(f"Account Balance: ${state.total_value:,.2f}")
    print(f"Realized P&L: ${state.realized_pnl:,.2f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("GRAPHWIZ-TRADER RISK MANAGEMENT SYSTEM DEMONSTRATION")
    print("=" * 60)

    try:
        demo_position_sizing()
        demo_stop_loss_calculator()
        demo_correlation_analysis()
        demo_drawdown_analysis()
        demo_risk_manager()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("  ✓ Multiple position sizing strategies (Kelly, Fixed Fractional, etc.)")
        print("  ✓ Stop-loss and take-profit calculations")
        print("  ✓ Correlation analysis and matrix calculation")
        print("  ✓ Drawdown analysis and metrics")
        print("  ✓ Comprehensive risk limits and validation")
        print("  ✓ Portfolio state management")
        print("  ✓ Position sizing with risk checks")
        print("\nAll risk metrics are tracked in the knowledge graph for analysis.")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
