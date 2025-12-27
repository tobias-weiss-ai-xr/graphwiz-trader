"""
Interactive demo for modern trading strategies (2025).

This demo showcases cutting-edge strategies based on latest 2025 research:
- Grid Trading (AI-enhanced)
- Smart Dollar-Cost Averaging (DCA)
- Automated Market Making (AMM)
- Triangular Arbitrage

Run this demo to see the strategies in action.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import sys

from graphwiz_trader.strategies.modern_strategies import (
    GridTradingStrategy,
    GridTradingMode,
    SmartDCAStrategy,
    AutomatedMarketMakingStrategy,
    TriangularArbitrageStrategy,
    create_modern_strategy,
)


def generate_ranging_market_data(days: int = 7) -> pd.DataFrame:
    """Generate ranging market data (ideal for grid trading)."""
    np.random.seed(42)

    periods = days * 24
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=periods,
        freq='1h'
    )

    base_price = 50000
    prices = []

    for i in range(periods):
        # Mean-reverting price (ranging market)
        trend = np.sin(i / 20) * 2000
        noise = np.random.normal(0, 300)
        price = base_price + trend + noise
        prices.append(price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [p * (1 + np.random.uniform(-0.001, 0.001)) for p in prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 500, periods),
    })
    df.set_index('timestamp', inplace=True)

    return df


def demo_grid_trading():
    """Demonstrate AI-enhanced grid trading strategy."""
    print("\n" + "="*80)
    print("DEMO 1: AI-Enhanced Grid Trading Strategy")
    print("="*80)
    print("\nBased on 2025 research:")
    print("  - arXiv:2506.11921 (Dynamic Grid Trading)")
    print("  - Zignaly Grid Trading Guide")
    print("  - Coinrule 2025 Grid Bot Guide")
    print()

    # Generate ranging market data
    df = generate_ranging_market_data(days=7)

    print(f"Generated {len(df)} hours of ranging market data")
    print(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    print()

    # Test different grid modes
    modes = [
        (GridTradingMode.ARITHMETIC, "Arithmetic (Equal Spacing)"),
        (GridTradingMode.GEOMETRIC, "Geometric (Percentage Spacing)"),
        (GridTradingMode.AI_ENHANCED, "AI-Enhanced (ML Optimized)"),
    ]

    for mode, name in modes:
        print(f"\n{'─'*80}")
        print(f"Testing: {name}")
        print(f"{'─'*80}")

        strategy = GridTradingStrategy(
            symbol='BTC/USDT',
            upper_price=53000,
            lower_price=47000,
            num_grids=10,
            grid_mode=mode,
            investment_amount=10000,
            dynamic_rebalancing=True,
            trailing_profit=True,
        )

        # Generate signals
        current_price = df['close'].iloc[-1]
        signals = strategy.generate_signals(current_price, df)

        print(f"Current price: ${current_price:,.2f}")
        print(f"Grid levels: {len(strategy.grid_levels)}")
        print(f"Orders to place: {len(signals['orders_to_place'])}")

        if signals['orders_to_place']:
            print(f"\nFirst 3 orders:")
            for order in signals['orders_to_place'][:3]:
                print(f"  {order['side'].upper():4s} {order['quantity']:.6f} @ ${order['price']:,.2f}")

        if signals['rebalance_needed']:
            print(f"\n⚠️  Rebalance recommended due to high volatility")

        if signals['trailing_profit_active']:
            print(f"🎯 Trailing profit active at ${signals['trailing_sell_price']:,.2f}")


def demo_smart_dca():
    """Demonstrate smart DCA strategy."""
    print("\n" + "="*80)
    print("DEMO 2: Smart Dollar-Cost Averaging Strategy")
    print("="*80)
    print("\nBased on 2025 research:")
    print("  - AlgosOne DCA Analysis")
    print("  - Altrady Trading Tools 2025")
    print()

    strategy = SmartDCAStrategy(
        symbol='BTC/USDT',
        total_investment=10000,
        purchase_frequency='weekly',
        purchase_amount=500,
        volatility_adjustment=True,
        momentum_boost=0.5,
        price_threshold=0.05,
    )

    print(f"Smart DCA Configuration:")
    print(f"  Total investment: ${strategy.total_investment:,.2f}")
    print(f"  Base purchase: ${strategy.base_purchase_amount:.2f}")
    print(f"  Frequency: {strategy.purchase_frequency}")
    print(f"  Volatility adjustment: {strategy.volatility_adjustment}")
    print(f"  Momentum boost: {strategy.momentum_boost:.1%}")
    print()

    # Simulate price history
    price_history = [55000, 53000, 51000, 49000, 51000, 53000, 55000]

    print('-' * 80)
    print("Simulating DCA Purchases Over Time:")
    print('-' * 80)

    for i, price in enumerate(price_history, 1):
        purchase = strategy.calculate_next_purchase(price)

        print(f"\nWeek {i}: BTC @ ${price:,.2f}")
        print(f"  Purchase amount: ${purchase['amount']:.2f}")
        print(f"  Quantity: {purchase['quantity']:.6f} BTC")
        print(f"  Reason: {purchase['reason']}")

        strategy.execute_purchase(purchase)

    # Show final status
    current_price = 55000
    status = strategy.get_portfolio_status(current_price)

    print(f"\n{'='*80}")
    print("Final Portfolio Status:")
    print(f"{'='*80}")
    print(f"Total invested: ${status['total_invested']:,.2f}")
    print(f"Total quantity: {status['total_quantity']:.6f} BTC")
    print(f"Avg purchase price: ${status['avg_purchase_price']:,.2f}")
    print(f"Current value: ${status['current_value']:,.2f}")
    print(f"P&L: ${status['pnl']:+,.2f} ({status['pnl_pct']:+.2f}%)")


def demo_amm():
    """Demonstrate automated market making strategy."""
    print("\n" + "="*80)
    print("DEMO 3: Automated Market Making (AMM) Strategy")
    print("="*80)
    print("\nBased on 2025 DeFi research:")
    print("  - ScienceDirect: DeFi and Automated Market Making")
    print("  - ACM: Toward More Profitable Liquidity Provisioning")
    print("  - arXiv: AMM and Decentralized Finance")
    print()

    strategy = AutomatedMarketMakingStrategy(
        token_a='ETH',
        token_b='USDT',
        pool_price=3000,
        price_range=(2400, 3600),
        base_fee_rate=0.003,
        inventory_target_ratio=0.5,
        rebalance_threshold=0.1,
    )

    print(f"AMM Pool Configuration:")
    print(f"  Pair: {strategy.token_a}/{strategy.token_b}")
    print(f"  Pool price: ${strategy.pool_price:,.2f}")
    print(f"  Price range: ${strategy.price_range[0]:,.2f} - ${strategy.price_range[1]:,.2f}")
    print(f"  Fee rate: {strategy.base_fee_rate:.3%}")
    print(f"  Inventory target: {strategy.inventory_target_ratio:.1%} each")
    print()

    # Calculate optimal positions
    print(f"{'─'*80}")
    print("Initial Position Analysis:")
    print(f"{'─'*80}")

    recommendations = strategy.calculate_optimal_positions(
        current_inventory_a=10,
        current_inventory_b=30000,
        current_price=3000,
    )

    print(f"Current {strategy.token_a} ratio: {recommendations['current_ratio_a']:.2%}")
    print(f"Target ratio: {recommendations['target_ratio']:.2%}")
    print(f"Needs rebalance: {recommendations['needs_rebalance']}")

    if recommendations['actions']:
        print(f"\nRebalancing actions:")
        for action in recommendations['actions']:
            print(f"  {action}")

    # Simulate trades
    print(f"\n{'─'*80}")
    print("Simulating Incoming Trades:")
    print(f"{'─'*80}")

    trades = [
        {'side': 'buy', 'amount': 1.0, 'price': 3000},
        {'side': 'sell', 'amount': 0.5, 'price': 3000},
        {'side': 'buy', 'amount': 2.0, 'price': 3050},  # Large trade
    ]

    for i, trade in enumerate(trades, 1):
        print(f"\nTrade {i}: {trade['side'].upper()} {trade['amount']} {strategy.token_a}")
        result = strategy.simulate_trade(trade)

        print(f"  Fee earned: ${result['fee_earned']:.2f}")
        print(f"  Price impact: {result['price_impact']:.4%}")
        if result['is_adverse_selection']:
            print(f"  ⚠️  Adverse selection detected!")

    # Pool metrics
    metrics = strategy.get_pool_metrics()

    print(f"\n{'='*80}")
    print("Pool Performance Metrics:")
    print(f"{'='*80}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Total fees earned: ${metrics['total_fees']:.2f}")
    print(f"Adverse selection rate: {metrics['adverse_selection_rate']:.2%}")
    print(f"Avg price impact: {metrics['avg_price_impact']:.4%}")


def demo_triangular_arbitrage():
    """Demonstrate triangular arbitrage strategy."""
    print("\n" + "="*80)
    print("DEMO 4: Triangular Arbitrage Strategy")
    print("="*80)
    print("\nBased on 2025 research:")
    print("  - WunderTrading Crypto Arbitrage Guide")
    print("  - Crustlab Best Arbitrage Bots")
    print("  - Bitunix Arbitrage Day Trading")
    print()

    strategy = TriangularArbitrageStrategy(
        exchanges=['binance', 'okx'],
        trading_pairs=['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
        min_profit_threshold=0.005,
        fee_rate=0.001,
    )

    print(f"Arbitrage Configuration:")
    print(f"  Exchanges: {', '.join(strategy.exchanges)}")
    print(f"  Trading pairs: {', '.join(strategy.trading_pairs)}")
    print(f"  Min profit threshold: {strategy.min_profit_threshold:.2%}")
    print(f"  Fee rate: {strategy.fee_rate:.2%}")
    print()

    # Simulate price data
    print(f"{'─'*80}")
    print("Price Data Across Exchanges:")
    print(f"{'─'*80}")

    price_data = {
        'binance': {
            'BTC/USDT': 50000,
            'ETH/BTC': 0.060,
            'ETH/USDT': 3000,
        },
        'okx': {
            'BTC/USDT': 50100,  # Slightly higher
            'ETH/BTC': 0.061,    # Slightly higher
            'ETH/USDT': 2990,    # Slightly lower
        },
    }

    for exchange, prices in price_data.items():
        strategy.update_prices({exchange: prices})

        print(f"\n{exchange.upper()}:")
        for pair, price in prices.items():
            print(f"  {pair}: ${price:,.2f}")

    # Find opportunities
    print(f"\n{'─'*80}")
    print("Scanning for Arbitrage Opportunities:")
    print(f"{'─'*80}")

    opportunities = strategy.find_arbitrage_opportunities()

    if opportunities:
        print(f"\nFound {len(opportunities)} opportunities:")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"\n{i}. {opp['exchange'].upper()} - {opp['profit_pct']:.2%} profit")
            print(f"   Path: {' → '.join(opp['path'])}")
            print(f"   Est. profit: ${opp['estimated_profit']:.2f}")

        # Execute best opportunity
        if opportunities:
            best = opportunities[0]
            print(f"\n{'='*80}")
            print("Executing Best Opportunity:")
            print(f"{'='*80}")

            result = strategy.execute_arbitrage(best, trade_size=10000)

            if result['success']:
                print(f"✅ Execution successful!")
                print(f"Initial amount: ${result['initial_amount']:,.2f}")
                print(f"Final amount: ${result['final_amount']:,.2f}")
                print(f"Profit: ${result['profit']:+,.2f} ({result['profit_pct']:+.2%})")

                if result['trades']:
                    print(f"\nExecution path:")
                    for trade in result['trades']:
                        print(f"  {trade['side'].upper():4s} {trade['amount']:,.2f} @ ${trade['price']:,.2f} ({trade['pair']})")
    else:
        print("\nNo profitable opportunities found at current prices")
        print("(This is normal - opportunities are rare in efficient markets)")


def demo_comparison():
    """Compare all modern strategies."""
    print("\n" + "="*80)
    print("DEMO 5: Strategy Comparison")
    print("="*80)
    print("\nComparing key characteristics of all modern strategies:\n")

    strategies = [
        {
            'name': 'Grid Trading',
            'market_type': 'Ranging/Sideways',
            'risk_level': 'Low-Medium',
            'complexity': 'Low',
            'best_for': 'Consistent profits in ranging markets',
        },
        {
            'name': 'Smart DCA',
            'market_type': 'All (Long-term)',
            'risk_level': 'Low',
            'complexity': 'Low',
            'best_for': 'Long-term accumulation',
        },
        {
            'name': 'AMM',
            'market_type': 'DeFi Pools',
            'risk_level': 'Medium',
            'complexity': 'High',
            'best_for': 'Liquidity provision, fee income',
        },
        {
            'name': 'Triangular Arbitrage',
            'market_type': 'Cross-exchange',
            'risk_level': 'Medium',
            'complexity': 'High',
            'best_for': 'Risk-free profit from price discrepancies',
        },
    ]

    print(f"{'Strategy':<25s} {'Market':<20s} {'Risk':<12s} {'Complexity':<12s}")
    print(f"{'─'*80}")

    for s in strategies:
        print(f"{s['name']:<25s} {s['market_type']:<20s} {s['risk_level']:<12s} {s['complexity']:<12s}")

    print(f"\n{'─'*80}")
    print("Best Use Cases:")
    print(f"{'─'*80}")

    for s in strategies:
        print(f"\n{s['name']}:")
        print(f"  {s['best_for']}")


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print(" "*20 + "2025 MODERN TRADING STRATEGIES DEMO")
    print("="*80)
    print("\nThis demo showcases cutting-edge strategies based on latest 2025 research:")
    print("  • Grid Trading (AI-enhanced)")
    print("  • Smart Dollar-Cost Averaging (DCA)")
    print("  • Automated Market Making (AMM)")
    print("  • Triangular Arbitrage")
    print("\n" + "="*80)

    try:
        demo_grid_trading()
        demo_smart_dca()
        demo_amm()
        demo_triangular_arbitrage()
        demo_comparison()

        print("\n" + "="*80)
        print(" "*30 + "DEMO COMPLETED")
        print("="*80)
        print("\nAll modern strategies demonstrated successfully!")
        print("\nNext Steps:")
        print("  1. Run tests: pytest tests/integration/test_modern_strategies.py")
        print("  2. Integrate with your trading engine")
        print("  3. Backtest with historical data")
        print("  4. Deploy to paper trading for validation")

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
