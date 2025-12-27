"""
Comprehensive Backtesting Suite for Modern Trading Strategies

This script performs:
1. Longer backtests with historical data
2. Parameter comparison across different configurations
3. Performance metrics analysis and reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

from graphwiz_trader.strategies import (
    GridTradingStrategy,
    GridTradingMode,
    SmartDCAStrategy,
    AutomatedMarketMakingStrategy,
    TriangularArbitrageStrategy,
    ModernStrategyAdapter,
)
from graphwiz_trader.trading.exchange import create_exchange


class ModernStrategyBacktester:
    """Comprehensive backtester for modern strategies."""

    def __init__(self, exchange_name: str = "binance"):
        """Initialize backtester.

        Args:
            exchange_name: Exchange for historical data
        """
        self.exchange = create_exchange(exchange_name)
        self.results = []

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        days: int = 30
    ) -> pd.DataFrame:
        """Fetch historical market data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching {days} days of {symbol} data ({timeframe})...")

        # Calculate number of candles needed
        candles_needed = days * 24  # For hourly data

        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=candles_needed)

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            print(f"✅ Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            return df

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise

    def backtest_grid_trading(
        self,
        symbol: str,
        data: pd.DataFrame,
        configurations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Backtest grid trading with multiple configurations.

        Args:
            symbol: Trading pair symbol
            data: Historical data
            configurations: List of strategy configurations

        Returns:
            List of backtest results
        """
        print("\n" + "="*80)
        print("BACKTESTING: GRID TRADING STRATEGY")
        print("="*80)

        results = []

        for i, config in enumerate(configurations, 1):
            print(f"\n--- Configuration {i}/{len(configurations)} ---")
            print(f"Mode: {config.get('grid_mode', 'GEOMETRIC')}")
            print(f"Grids: {config.get('num_grids', 10)}")
            print(f"Range: ${config.get('lower_price', 0):,.0f} - ${config.get('upper_price', 0):,.0f}")

            # Create strategy
            strategy = GridTradingStrategy(
                symbol=symbol,
                **config
            )

            adapter = ModernStrategyAdapter(strategy)

            # Simulate trading over time
            trades_executed = 0
            total_profit = 0
            initial_price = data['close'].iloc[0]
            final_price = data['close'].iloc[-1]

            # Sample every 10 candles for speed
            sample_data = data.iloc[::10]

            for idx, row in sample_data.iterrows():
                current_price = row['close']

                # Get window of historical data
                historical_window = data.loc[:idx].tail(100) if idx in data.index else data.tail(100)

                # Generate signals
                signals = adapter.generate_trading_signals(current_price, historical_window)

                # Count orders near current price (simulate fills)
                for order in signals['orders']:
                    if abs(order['price'] - current_price) / current_price < 0.02:  # Within 2%
                        trades_executed += 1
                        # Estimate profit (simplified)
                        if order['side'] == 'sell':
                            profit = (current_price - strategy.lower_price) * order['amount']
                            total_profit += profit

            # Calculate metrics
            price_change_pct = ((final_price - initial_price) / initial_price) * 100

            result = {
                'strategy': 'Grid Trading',
                'config': config,
                'trades_executed': trades_executed,
                'total_profit': total_profit,
                'initial_price': initial_price,
                'final_price': final_price,
                'price_change_pct': price_change_pct,
                'data_points': len(sample_data),
            }

            results.append(result)

            print(f"Trades executed: {trades_executed}")
            print(f"Estimated profit: ${total_profit:,.2f}")
            print(f"Price change: {price_change_pct:+.2f}%")

        return results

    def backtest_smart_dca(
        self,
        symbol: str,
        data: pd.DataFrame,
        configurations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Backtest smart DCA with multiple configurations.

        Args:
            symbol: Trading pair symbol
            data: Historical data
            configurations: List of strategy configurations

        Returns:
            List of backtest results
        """
        print("\n" + "="*80)
        print("BACKTESTING: SMART DCA STRATEGY")
        print("="*80)

        results = []

        for i, config in enumerate(configurations, 1):
            print(f"\n--- Configuration {i}/{len(configurations)} ---")
            print(f"Purchase amount: ${config.get('purchase_amount', 500)}")
            print(f"Volatility adjustment: {config.get('volatility_adjustment', False)}")
            print(f"Momentum boost: {config.get('momentum_boost', 0):.1%}")

            # Create strategy
            strategy = SmartDCAStrategy(
                symbol=symbol,
                **config
            )

            adapter = ModernStrategyAdapter(strategy)

            # Simulate DCA purchases over time (weekly)
            # Sample every 7 days worth of hourly candles (168 candles)
            purchase_candles = data.iloc[::168]

            initial_price = data['close'].iloc[0]
            final_price = data['close'].iloc[-1]

            for idx, row in purchase_candles.iterrows():
                current_price = row['close']

                # Get historical window
                historical_window = data.loc[:idx].tail(100) if idx in data.index else data.tail(100)

                # Generate signal
                signals = adapter.generate_trading_signals(current_price, historical_window)

                # Execute purchase
                if signals['should_execute']:
                    order = signals['order']

                    # Record purchase
                    trade_result = {
                        'status': 'executed',
                        'symbol': symbol,
                        'side': order['side'],
                        'amount': order['amount'],
                        'price': order['price'],
                        'metadata': order.get('metadata', {}),
                    }

                    adapter.execute_trade(trade_result)

            # Get final portfolio status
            status = adapter.get_strategy_status(final_price)

            result = {
                'strategy': 'Smart DCA',
                'config': config,
                'total_invested': status['total_invested'],
                'total_quantity': status['total_quantity'],
                'avg_purchase_price': status['avg_purchase_price'],
                'current_value': status['current_value'],
                'pnl': status['pnl'],
                'pnl_pct': status['pnl_pct'],
                'num_purchases': status['num_purchases'],
                'initial_price': initial_price,
                'final_price': final_price,
                'price_change_pct': ((final_price - initial_price) / initial_price) * 100,
            }

            results.append(result)

            print(f"Total invested: ${status['total_invested']:,.2f}")
            print(f"Total quantity: {status['total_quantity']:.6f}")
            print(f"Avg price: ${status['avg_purchase_price']:,.2f}")
            print(f"P&L: ${status['pnl']:+,.2f} ({status['pnl_pct']:+.2f}%)")

        return results

    def backtest_amm(
        self,
        token_a: str,
        token_b: str,
        data: pd.DataFrame,
        configurations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Backtest AMM with multiple configurations.

        Args:
            token_a: First token
            token_b: Second token
            data: Historical price data
            configurations: List of strategy configurations

        Returns:
            List of backtest results
        """
        print("\n" + "="*80)
        print("BACKTESTING: AMM STRATEGY")
        print("="*80)

        results = []

        for i, config in enumerate(configurations, 1):
            print(f"\n--- Configuration {i}/{len(configurations)} ---")
            print(f"Price range: ${config.get('price_range', (0, 0))[0]:,.0f} - ${config.get('price_range', (0, 0))[1]:,.0f}")
            print(f"Fee rate: {config.get('base_fee_rate', 0.003):.3%}")

            # Create strategy
            strategy = AutomatedMarketMakingStrategy(
                token_a=token_a,
                token_b=token_b,
                **config
            )

            adapter = ModernStrategyAdapter(strategy)

            # Simulate random trades coming in
            np.random.seed(42)
            num_trades = 100

            initial_price = data['close'].iloc[0]
            final_price = data['close'].iloc[-1]

            for trade_idx in range(num_trades):
                # Random price within range
                price_pct = np.random.uniform(-0.1, 0.1)
                current_price = final_price * (1 + price_pct)

                # Random trade side and amount
                side = np.random.choice(['buy', 'sell'])
                amount = np.random.uniform(0.5, 2.0)

                # Simulate trade
                signals = adapter.generate_trading_signals(
                    current_price=current_price,
                    current_inventory_a=10,
                    current_inventory_b=10 * current_price,
                )

                # Record trade
                trade_result = {
                    'status': 'executed',
                    'symbol': f"{token_a}/{token_b}",
                    'side': side,
                    'amount': amount,
                    'price': current_price,
                    'metadata': {},
                }

                adapter.execute_trade(trade_result)

            # Get pool metrics
            metrics = adapter.get_strategy_status(current_price)

            result = {
                'strategy': 'AMM',
                'config': config,
                'total_trades': metrics['total_trades'],
                'total_fees': metrics['total_fees'],
                'adverse_selection_rate': metrics['adverse_selection_rate'],
                'avg_price_impact': metrics['avg_price_impact'],
                'initial_price': initial_price,
                'final_price': final_price,
                'price_change_pct': ((final_price - initial_price) / initial_price) * 100,
            }

            results.append(result)

            print(f"Total trades: {metrics['total_trades']}")
            print(f"Total fees: ${metrics['total_fees']:.2f}")
            print(f"Adverse selection: {metrics['adverse_selection_rate']:.2%}")
            print(f"Avg price impact: {metrics['avg_price_impact']:.4%}")

        return results

    def compare_configurations(self, results: List[Dict[str, Any]]) -> None:
        """Compare and rank different configurations.

        Args:
            results: List of backtest results
        """
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)

        strategy_name = results[0]['strategy']

        if strategy_name == 'Grid Trading':
            # Sort by trades executed
            sorted_results = sorted(results, key=lambda x: x['trades_executed'], reverse=True)

            print(f"\n{'Rank':<6} {'Grids':<8} {'Mode':<12} {'Trades':<10} {'Profit':<12}")
            print("-" * 80)

            for i, result in enumerate(sorted_results, 1):
                mode = result['config'].get('grid_mode', 'GEOMETRIC')
                grids = result['config'].get('num_grids', 10)
                trades = result['trades_executed']
                profit = result['total_profit']

                print(f"{i:<6} {grids:<8} {mode:<12} {trades:<10} ${profit:>10,.2f}")

        elif strategy_name == 'Smart DCA':
            # Sort by P&L
            sorted_results = sorted(results, key=lambda x: x['pnl_pct'], reverse=True)

            print(f"\n{'Rank':<6} {'Vol Adj':<10} {'Mom Boost':<12} {'Invested':<12} {'P&L %':<10}")
            print("-" * 80)

            for i, result in enumerate(sorted_results, 1):
                vol_adj = result['config'].get('volatility_adjustment', False)
                mom_boost = result['config'].get('momentum_boost', 0)
                invested = result['total_invested']
                pnl_pct = result['pnl_pct']

                print(f"{i:<6} {str(vol_adj):<10} {mom_boost:<12} ${invested:>10,.2f} {pnl_pct:>+8.2f}%")

        elif strategy_name == 'AMM':
            # Sort by total fees
            sorted_results = sorted(results, key=lambda x: x['total_fees'], reverse=True)

            print(f"\n{'Rank':<6} {'Fee Rate':<10} {'Range':<20} {'Trades':<10} {'Fees':<12}")
            print("-" * 80)

            for i, result in enumerate(sorted_results, 1):
                fee_rate = result['config'].get('base_fee_rate', 0.003)
                range_str = f"{result['config'].get('price_range', (0, 0))[0]:,.0f}-{result['config'].get('price_range', (0, 0))[1]:,.0f}"
                trades = result['total_trades']
                fees = result['total_fees']

                print(f"{i:<6} {fee_rate:<10.3%} {range_str:<20} {trades:<10} ${fees:>10,.2f}")

    def generate_report(self, all_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate comprehensive backtesting report.

        Args:
            all_results: Dictionary mapping strategy names to results

        Returns:
            Report file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"data/backtesting/modern_strategies_backtest_{timestamp}.json"

        # Create directory
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n✅ Report saved to: {report_path}")
        return report_path


def main():
    """Run comprehensive backtesting suite."""
    print("\n" + "="*80)
    print(" "*15 + "MODERN STRATEGIES COMPREHENSIVE BACKTESTING")
    print("="*80)
    print("\nThis will:")
    print("  1. Fetch historical market data")
    print("  2. Backtest all modern strategies")
    print("  3. Compare different configurations")
    print("  4. Generate performance reports")
    print()

    backtester = ModernStrategyBacktester()

    all_results = {}

    # ============================================================================
    # GRID TRADING BACKTESTS
    # ============================================================================
    print("\n" + "▶" * 40)
    print("Starting Grid Trading Backtests...")
    print("▶" * 40)

    # Fetch BTC data
    btc_data = backtester.fetch_historical_data('BTC/USDT', days=30)

    # Get current price to center grid properly
    current_btc_price = btc_data['close'].iloc[-1]

    print(f"\nCurrent BTC price: ${current_btc_price:,.2f}")
    print(f"Setting grid range around current price (±10%, ±15%, ±20%)")

    # Define configurations to test - FIXED: Centered around current price
    grid_configs = [
        {
            'upper_price': current_btc_price * 1.10,  # +10%
            'lower_price': current_btc_price * 0.90,  # -10%
            'num_grids': 5,
            'grid_mode': GridTradingMode.ARITHMETIC,
            'investment_amount': 10000,
        },
        {
            'upper_price': current_btc_price * 1.15,  # +15%
            'lower_price': current_btc_price * 0.85,  # -15%
            'num_grids': 10,
            'grid_mode': GridTradingMode.ARITHMETIC,
            'investment_amount': 10000,
        },
        {
            'upper_price': current_btc_price * 1.15,  # +15%
            'lower_price': current_btc_price * 0.85,  # -15%
            'num_grids': 10,
            'grid_mode': GridTradingMode.GEOMETRIC,
            'investment_amount': 10000,
        },
        {
            'upper_price': current_btc_price * 1.20,  # +20%
            'lower_price': current_btc_price * 0.80,  # -20%
            'num_grids': 15,
            'grid_mode': GridTradingMode.GEOMETRIC,
            'investment_amount': 10000,
        },
    ]

    grid_results = backtester.backtest_grid_trading('BTC/USDT', btc_data, grid_configs)
    backtester.compare_configurations(grid_results)
    all_results['grid_trading'] = grid_results

    # ============================================================================
    # SMART DCA BACKTESTS
    # ============================================================================
    print("\n" + "▶" * 40)
    print("Starting Smart DCA Backtests...")
    print("▶" * 40)

    # Fetch ETH data
    eth_data = backtester.fetch_historical_data('ETH/USDT', days=30)

    # Define configurations to test
    dca_configs = [
        {
            'total_investment': 5000,
            'purchase_amount': 100,
            'purchase_frequency': 'daily',
            'volatility_adjustment': False,
            'momentum_boost': 0.0,
        },
        {
            'total_investment': 5000,
            'purchase_amount': 100,
            'purchase_frequency': 'daily',
            'volatility_adjustment': True,
            'momentum_boost': 0.0,
        },
        {
            'total_investment': 5000,
            'purchase_amount': 100,
            'purchase_frequency': 'daily',
            'volatility_adjustment': True,
            'momentum_boost': 0.5,
        },
        {
            'total_investment': 5000,
            'purchase_amount': 200,
            'purchase_frequency': 'daily',
            'volatility_adjustment': True,
            'momentum_boost': 0.5,
        },
    ]

    dca_results = backtester.backtest_smart_dca('ETH/USDT', eth_data, dca_configs)
    backtester.compare_configurations(dca_results)
    all_results['smart_dca'] = dca_results

    # ============================================================================
    # AMM BACKTESTS
    # ============================================================================
    print("\n" + "▶" * 40)
    print("Starting AMM Backtests...")
    print("▶" * 40)

    # Fetch SOL data
    sol_data = backtester.fetch_historical_data('SOL/USDT', days=30)

    # Define configurations to test
    amm_configs = [
        {
            'pool_price': 100,
            'price_range': (80, 120),
            'base_fee_rate': 0.001,
        },
        {
            'pool_price': 100,
            'price_range': (80, 120),
            'base_fee_rate': 0.003,
        },
        {
            'pool_price': 100,
            'price_range': (70, 130),
            'base_fee_rate': 0.003,
        },
        {
            'pool_price': 100,
            'price_range': (90, 110),
            'base_fee_rate': 0.005,
        },
    ]

    amm_results = backtester.backtest_amm('SOL', 'USDT', sol_data, amm_configs)
    backtester.compare_configurations(amm_results)
    all_results['amm'] = amm_results

    # ============================================================================
    # GENERATE REPORT
    # ============================================================================
    print("\n" + "="*80)
    print(" "*25 + "BACKTESTING COMPLETE")
    print("="*80)

    report_path = backtester.generate_report(all_results)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nGrid Trading: {len(grid_results)} configurations tested")
    print(f"Smart DCA: {len(dca_results)} configurations tested")
    print(f"AMM: {len(amm_results)} configurations tested")
    print(f"\nTotal configurations: {len(grid_results) + len(dca_results) + len(amm_results)}")
    print(f"\nReport saved: {report_path}")

    # Best configurations summary
    print("\n" + "-"*80)
    print("BEST CONFIGURATIONS")
    print("-"*80)

    # Grid Trading - most trades
    best_grid = max(grid_results, key=lambda x: x['trades_executed'])
    print(f"\nGrid Trading:")
    print(f"  Config: {best_grid['config']['num_grids']} grids, {best_grid['config']['grid_mode']}")
    print(f"  Trades: {best_grid['trades_executed']}")
    print(f"  Profit: ${best_grid['total_profit']:,.2f}")

    # Smart DCA - best P&L
    best_dca = max(dca_results, key=lambda x: x['pnl_pct'])
    print(f"\nSmart DCA:")
    print(f"  Config: Vol adj={best_dca['config']['volatility_adjustment']}, Mom boost={best_dca['config']['momentum_boost']}")
    print(f"  P&L: {best_dca['pnl_pct']:+.2f}%")
    print(f"  Invested: ${best_dca['total_invested']:,.2f}")

    # AMM - most fees
    best_amm = max(amm_results, key=lambda x: x['total_fees'])
    print(f"\nAMM:")
    print(f"  Config: Fee rate={best_amm['config']['base_fee_rate']:.3%}")
    print(f"  Fees earned: ${best_amm['total_fees']:.2f}")
    print(f"  Trades: {best_amm['total_trades']}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
