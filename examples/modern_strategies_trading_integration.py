"""
Example: Integrating Modern Strategies with Trading Engine

This example demonstrates how to use modern trading strategies
with the GraphWiz trading engine for paper trading and live trading.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import sys

from graphwiz_trader.strategies.modern_strategies import (
    GridTradingStrategy,
    GridTradingMode,
    SmartDCAStrategy,
    AutomatedMarketMakingStrategy,
    TriangularArbitrageStrategy,
)
from graphwiz_trader.strategies.modern_strategy_adapter import (
    ModernStrategyAdapter,
    create_modern_strategy_adapter,
)


def example_grid_trading_integration():
    """Example: Grid trading integration with trading engine."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Grid Trading with Trading Engine")
    print("="*80)

    # Create grid trading strategy
    strategy = GridTradingStrategy(
        symbol='BTC/USDT',
        upper_price=55000,
        lower_price=45000,
        num_grids=10,
        grid_mode=GridTradingMode.GEOMETRIC,
        investment_amount=10000,
        dynamic_rebalancing=True,
    )

    # Create adapter
    adapter = ModernStrategyAdapter(strategy)

    # Simulate market data
    current_price = 50000
    historical_data = pd.DataFrame({
        'close': [48000, 48500, 49000, 49500, 50000],
        'volume': [1000, 1100, 1050, 1200, 1150],
    })

    # Generate trading signals
    signals = adapter.generate_trading_signals(current_price, historical_data)

    print(f"\nStrategy: {signals['strategy']}")
    print(f"Current price: ${signals['current_price']:,.2f}")
    print(f"Grid levels: {len(signals['grid_levels'])}")
    print(f"Orders to place: {len(signals['orders'])}")

    if signals['orders']:
        print(f"\nFirst 3 orders:")
        for order in signals['orders'][:3]:
            print(f"  {order['side'].upper():4s} {order['amount']:.6f} @ ${order['price']:,.2f}")

    # Simulate trade execution
    if signals['orders']:
        first_order = signals['orders'][0]
        trade_result = {
            'status': 'executed',
            'order_id': 'ORD-001',
            'symbol': first_order['symbol'],
            'side': first_order['side'],
            'amount': first_order['amount'],
            'price': first_order['price'],
            'metadata': first_order.get('metadata', {}),
        }

        print(f"\n✅ Executed trade: {trade_result['side']} {trade_result['amount']:.6f} @ ${trade_result['price']:,.2f}")

        # Get strategy status
        status = adapter.get_strategy_status(current_price)
        print(f"\nStrategy Status:")
        print(f"  Type: {status['strategy_type']}")
        print(f"  Symbol: {status['symbol']}")
        print(f"  Range: ${status['lower_price']:,.2f} - ${status['upper_price']:,.2f}")


def example_smart_dca_integration():
    """Example: Smart DCA integration with trading engine."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Smart DCA with Trading Engine")
    print("="*80)

    # Create smart DCA strategy
    strategy = SmartDCAStrategy(
        symbol='BTC/USDT',
        total_investment=10000,
        purchase_frequency='weekly',
        purchase_amount=500,
        volatility_adjustment=True,
        momentum_boost=0.5,
    )

    # Create adapter
    adapter = ModernStrategyAdapter(strategy)

    # Simulate price history
    price_history = [55000, 53000, 51000, 49000, 51000]

    print(f"\nSmart DCA Configuration:")
    print(f"  Total investment: ${strategy.total_investment:,.2f}")
    print(f"  Base purchase: ${strategy.base_purchase_amount:.2f}")
    print(f"  Frequency: {strategy.purchase_frequency}")
    print(f"\nSimulating purchases over time:")

    for i, price in enumerate(price_history, 1):
        # Generate trading signal
        signals = adapter.generate_trading_signals(price)

        if signals['should_execute']:
            order = signals['order']
            print(f"\nWeek {i}: BTC @ ${price:,.2f}")
            print(f"  Purchase: {order['amount']:.6f} BTC (${order['metadata']['amount_usd']:.2f})")
            print(f"  Reason: {order['metadata']['reason']}")

            # Simulate trade execution
            trade_result = {
                'status': 'executed',
                'order_id': f'ORD-{i}',
                'symbol': order['symbol'],
                'side': order['side'],
                'amount': order['amount'],
                'price': order['price'],
                'metadata': order.get('metadata', {}),
            }

            # Update strategy state
            adapter.execute_trade(trade_result)

    # Get final portfolio status
    final_price = 51000
    status = adapter.get_strategy_status(final_price)

    print(f"\n{'='*60}")
    print("Final Portfolio Status:")
    print(f"{'='*60}")
    print(f"Total invested: ${status['total_invested']:,.2f}")
    print(f"Total quantity: {status['total_quantity']:.6f} BTC")
    print(f"Avg purchase price: ${status['avg_purchase_price']:,.2f}")
    print(f"Current value: ${status['current_value']:,.2f}")
    print(f"P&L: ${status['pnl']:+,.2f} ({status['pnl_pct']:+.2f}%)")


def example_amm_integration():
    """Example: AMM integration with trading engine."""
    print("\n" + "="*80)
    print("EXAMPLE 3: AMM with Trading Engine")
    print("="*80)

    # Create AMM strategy
    strategy = AutomatedMarketMakingStrategy(
        token_a='ETH',
        token_b='USDT',
        pool_price=3000,
        price_range=(2400, 3600),
        base_fee_rate=0.003,
        inventory_target_ratio=0.5,
    )

    # Create adapter
    adapter = ModernStrategyAdapter(strategy)

    print(f"\nAMM Pool Configuration:")
    print(f"  Pair: {strategy.token_a}/{strategy.token_b}")
    print(f"  Pool price: ${strategy.pool_price:,.2f}")
    print(f"  Price range: ${strategy.price_range[0]:,.2f} - ${strategy.price_range[1]:,.2f}")

    # Initial inventory analysis
    current_price = 3000
    signals = adapter.generate_trading_signals(
        current_price=current_price,
        current_inventory_a=10,
        current_inventory_b=30000,
    )

    print(f"\nInitial Position Analysis:")
    print(f"  Current ETH ratio: {signals['current_ratio_a']:.2%}")
    print(f"  Target ratio: {signals['target_ratio']:.2%}")
    print(f"  Needs rebalance: {signals['needs_rebalance']}")

    # Simulate incoming trades
    print(f"\nSimulating Trades:")
    trades = [
        {'side': 'buy', 'amount': 1.0, 'price': 3000},
        {'side': 'sell', 'amount': 0.5, 'price': 3000},
        {'side': 'buy', 'amount': 2.0, 'price': 3050},
    ]

    for i, trade in enumerate(trades, 1):
        # Simulate trade execution
        trade_result = {
            'status': 'executed',
            'order_id': f'ORD-{i}',
            'symbol': f"{strategy.token_a}/{strategy.token_b}",
            'side': trade['side'],
            'amount': trade['amount'],
            'price': trade['price'],
            'metadata': {},
        }

        # Update strategy state
        adapter.execute_trade(trade_result)

        print(f"\nTrade {i}: {trade['side'].upper()} {trade['amount']} {strategy.token_a} @ ${trade['price']:,.2f}")

    # Get pool metrics
    metrics = adapter.get_strategy_status(current_price)
    print(f"\n{'='*60}")
    print("Pool Performance Metrics:")
    print(f"{'='*60}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Total fees earned: ${metrics['total_fees']:.2f}")
    print(f"Adverse selection rate: {metrics['adverse_selection_rate']:.2%}")
    print(f"Avg price impact: {metrics['avg_price_impact']:.4%}")


def example_arbitrage_integration():
    """Example: Triangular arbitrage integration with trading engine."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Triangular Arbitrage with Trading Engine")
    print("="*80)

    # Create arbitrage strategy
    strategy = TriangularArbitrageStrategy(
        exchanges=['binance', 'okx'],
        trading_pairs=['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
        min_profit_threshold=0.005,
        fee_rate=0.001,
    )

    # Create adapter
    adapter = ModernStrategyAdapter(strategy)

    print(f"\nArbitrage Configuration:")
    print(f"  Exchanges: {', '.join(strategy.exchanges)}")
    print(f"  Trading pairs: {', '.join(strategy.trading_pairs)}")
    print(f"  Min profit threshold: {strategy.min_profit_threshold:.2%}")

    # Simulate price data
    price_data = {
        'binance': {
            'BTC/USDT': 50000,
            'ETH/BTC': 0.060,
            'ETH/USDT': 3000,
        },
        'okx': {
            'BTC/USDT': 50100,
            'ETH/BTC': 0.061,
            'ETH/USDT': 2990,
        },
    }

    # Generate arbitrage signals (current_price not used for arbitrage)
    signals = adapter.generate_trading_signals(current_price=0, price_data=price_data)

    print(f"\nScanning for opportunities...")
    print(f"Found {signals['opportunities_found']} opportunities")

    if signals['orders']:
        opportunity = signals['orders'][0]['metadata']
        print(f"\nBest opportunity:")
        print(f"  Exchange: {opportunity['exchange'].upper()}")
        print(f"  Path: {' → '.join(opportunity['path'])}")
        print(f"  Profit: {opportunity['profit_pct']:.2%} (${opportunity['estimated_profit']:.2f})")

        print(f"\n✅ Would execute arbitrage trade")


def example_factory_function():
    """Example: Using factory function to create strategies."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Factory Function for Easy Strategy Creation")
    print("="*80)

    strategies_to_create = [
        {
            'strategy_type': 'grid_trading',
            'name': 'Grid Trading BTC',
            'params': {
                'symbol': 'BTC/USDT',
                'upper_price': 55000,
                'lower_price': 45000,
                'num_grids': 10,
            }
        },
        {
            'strategy_type': 'smart_dca',
            'name': 'Smart DCA ETH',
            'params': {
                'symbol': 'ETH/USDT',
                'total_investment': 5000,
                'purchase_amount': 100,
            }
        },
        {
            'strategy_type': 'amm',
            'name': 'AMM SOL/USDT',
            'params': {
                'token_a': 'SOL',
                'token_b': 'USDT',
                'pool_price': 100,
                'price_range': (80, 120),
            }
        },
    ]

    print(f"\nCreating strategies using factory function:\n")

    for config in strategies_to_create:
        adapter = create_modern_strategy_adapter(
            strategy_type=config['strategy_type'],
            **config['params']
        )

        status = adapter.get_strategy_status(current_price=0)
        print(f"✅ Created: {config['name']}")
        print(f"   Strategy type: {status.get('strategy_type', 'N/A')}")
        print(f"   Symbol: {config['params'].get('symbol') or config['params'].get('token_a') + '/' + config['params'].get('token_b', '')}")
        print()


async def main():
    """Run all integration examples."""
    print("\n" + "="*80)
    print(" "*20 + "MODERN STRATEGIES TRADING INTEGRATION EXAMPLES")
    print("="*80)
    print("\nThis demonstrates how to integrate modern strategies with the trading engine")

    try:
        example_grid_trading_integration()
        example_smart_dca_integration()
        example_amm_integration()
        example_arbitrage_integration()
        example_factory_function()

        print("\n" + "="*80)
        print(" "*30 + "EXAMPLES COMPLETED")
        print("="*80)
        print("\nNext Steps:")
        print("  1. Integrate with paper trading engine for backtesting")
        print("  2. Connect to live exchange for paper trading")
        print("  3. Deploy to production with real trading (caution!)")
        print("\nKey Features Demonstrated:")
        print("  ✅ Unified adapter interface for all strategies")
        print("  ✅ Signal generation compatible with trading engine")
        print("  ✅ State management after trade execution")
        print("  ✅ Strategy status and performance tracking")
        print("  ✅ Factory function for easy strategy creation")

    except Exception as e:
        logger.exception(f"Error running examples: {e}")
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

    # Run examples
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
