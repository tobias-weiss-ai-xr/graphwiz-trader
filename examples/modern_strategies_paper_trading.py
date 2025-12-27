"""
Example: Modern Strategies with Paper Trading Engine

This example demonstrates how to use modern trading strategies
with the GraphWiz paper trading engine for backtesting and validation.
"""

import time
import pandas as pd
from datetime import datetime
from loguru import logger
from typing import Dict, Any

from graphwiz_trader.strategies import (
    GridTradingStrategy,
    GridTradingMode,
    SmartDCAStrategy,
    ModernStrategyAdapter,
)
from graphwiz_trader.trading.exchange import create_exchange


class ModernStrategyPaperTrader:
    """Paper trading engine with modern strategies."""

    def __init__(
        self,
        strategy_adapter: ModernStrategyAdapter,
        exchange_name: str = "binance",
        symbol: str = "BTC/USDT",
        initial_capital: float = 10000.0,
        commission: float = 0.001,
    ):
        """Initialize paper trader with modern strategy.

        Args:
            strategy_adapter: Modern strategy adapter
            exchange_name: Exchange to use for market data
            symbol: Trading pair symbol
            initial_capital: Starting virtual capital
            commission: Trading commission rate
        """
        self.adapter = strategy_adapter
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.commission = commission

        # Create exchange (read-only for market data)
        self.exchange = create_exchange(exchange_name)

        # Initialize portfolio
        self.portfolio = {
            "capital": initial_capital,
            "position": 0.0,
            "avg_price": 0.0,
        }

        # Trade history
        self.trades = []
        self.equity_curve = []

        logger.info(f"Initialized paper trading: {symbol} with ${initial_capital:,.2f}")
        logger.info(f"Strategy: {self.adapter.strategy_type}")

    def fetch_latest_data(self, limit: int = 100) -> pd.DataFrame:
        """Fetch latest market data.

        Args:
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, "1h", limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            logger.debug(f"Fetched {len(df)} candles for {self.symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise

    def generate_and_execute_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals from strategy and execute trades.

        Args:
            data: Market data DataFrame

        Returns:
            Execution result dictionary
        """
        try:
            # Get current price
            current_price = data["close"].iloc[-1]
            current_time = data.index[-1]

            # Generate signals based on strategy type
            if isinstance(self.adapter.strategy, GridTradingStrategy):
                return self._handle_grid_trading(current_price, data)

            elif isinstance(self.adapter.strategy, SmartDCAStrategy):
                return self._handle_dca(current_price, data)

            else:
                return {
                    "status": "error",
                    "message": f"Strategy {self.adapter.strategy_type} not yet supported"
                }

        except Exception as e:
            logger.error(f"Error generating/executing signals: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_grid_trading(
        self,
        current_price: float,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Handle grid trading strategy.

        Args:
            current_price: Current market price
            data: Historical data

        Returns:
            Execution result
        """
        # Generate grid signals
        signals = self.adapter.generate_trading_signals(current_price, data)

        # Execute orders if signal exists
        executed_orders = []
        for order in signals['orders']:
            # Only execute orders near current price (simulating limit order fills)
            if abs(order['price'] - current_price) / current_price < 0.01:  # Within 1%
                result = self._execute_virtual_order(order, current_price)
                if result:
                    executed_orders.append(result)

        # Update equity
        self._update_equity(current_price)

        return {
            "status": "success",
            "strategy": "grid_trading",
            "current_price": current_price,
            "orders_executed": len(executed_orders),
            "portfolio_value": self._calculate_portfolio_value(current_price),
        }

    def _handle_dca(
        self,
        current_price: float,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Handle smart DCA strategy.

        Args:
            current_price: Current market price
            data: Historical data

        Returns:
            Execution result
        """
        # Generate DCA signal
        signals = self.adapter.generate_trading_signals(current_price, data)

        # Execute purchase if signal indicates
        if signals['should_execute']:
            order = signals['order']
            result = self._execute_virtual_order(order, current_price)
            if result:
                # Update strategy state
                trade_result = {
                    'status': 'executed',
                    'side': order['side'],
                    'amount': order['amount'],
                    'price': order['price'],
                    'metadata': order.get('metadata', {}),
                }
                self.adapter.execute_trade(trade_result)
        else:
            result = None

        # Update equity
        self._update_equity(current_price)

        return {
            "status": "success",
            "strategy": "smart_dca",
            "current_price": current_price,
            "order_executed": result is not None,
            "portfolio_value": self._calculate_portfolio_value(current_price),
        }

    def _execute_virtual_order(self, order: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Execute a virtual order.

        Args:
            order: Order dictionary
            current_price: Current market price

        Returns:
            Trade dictionary if executed, None otherwise
        """
        side = order['side']
        quantity = order['amount']
        price = order.get('price', current_price)

        capital = self.portfolio["capital"]
        position = self.portfolio["position"]

        trade = None

        if side == "buy":
            # Check if we have enough capital
            cost = quantity * price * (1 + self.commission)
            if cost > capital:
                logger.warning(f"Insufficient capital: need ${cost:.2f}, have ${capital:.2f}")
                return None

            # Execute buy
            self.portfolio["capital"] -= cost
            self.portfolio["position"] += quantity

            # Update average price
            if position > 0:
                total_value = (position * self.portfolio["avg_price"]) + (quantity * price)
                total_quantity = position + quantity
                self.portfolio["avg_price"] = total_value / total_quantity
            else:
                self.portfolio["avg_price"] = price

            trade = {
                "timestamp": datetime.now(),
                "action": "buy",
                "price": price,
                "quantity": quantity,
                "value": quantity * price,
                "cost": cost,
                "capital_after": self.portfolio["capital"],
                "position_after": self.portfolio["position"],
            }

            logger.success(
                f"🟢 BOUGHT {quantity:.6f} @ ${price:,.2f} "
                f"(Value: ${quantity * price:,.2f}, Cost: ${cost:,.2f})"
            )

        elif side == "sell":
            # Check if we have enough position
            if quantity > position:
                logger.warning(f"Insufficient position: need {quantity:.6f}, have {position:.6f}")
                return None

            # Execute sell
            proceeds = quantity * price * (1 - self.commission)
            pnl = proceeds - (quantity * self.portfolio["avg_price"])
            pnl_pct = (pnl / (quantity * self.portfolio["avg_price"])) * 100

            self.portfolio["capital"] += proceeds
            self.portfolio["position"] -= quantity

            trade = {
                "timestamp": datetime.now(),
                "action": "sell",
                "price": price,
                "quantity": quantity,
                "value": quantity * price,
                "proceeds": proceeds,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "capital_after": self.portfolio["capital"],
                "position_after": self.portfolio["position"],
            }

            if pnl >= 0:
                logger.success(
                    f"🔵 SOLD {quantity:.6f} @ ${price:,.2f} "
                    f"(P&L: ${pnl:,.2f}, {pnl_pct:+.2f}%)"
                )
            else:
                logger.warning(
                    f"🔴 SOLD {quantity:.6f} @ ${price:,.2f} "
                    f"(P&L: ${pnl:,.2f}, {pnl_pct:+.2f}%)"
                )

        # Record trade
        if trade:
            self.trades.append(trade)

        return trade

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value.

        Args:
            current_price: Current market price

        Returns:
            Total portfolio value
        """
        capital = self.portfolio["capital"]
        position = self.portfolio["position"]
        position_value = position * current_price
        return capital + position_value

    def _update_equity(self, current_price: float):
        """Update equity curve.

        Args:
            current_price: Current market price
        """
        total_value = self._calculate_portfolio_value(current_price)

        self.equity_curve.append({
            "timestamp": datetime.now(),
            "capital": self.portfolio["capital"],
            "position": self.portfolio["position"],
            "position_value": self.portfolio["position"] * current_price,
            "total_value": total_value,
        })

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics.

        Returns:
            Performance metrics dictionary
        """
        if not self.equity_curve:
            return {}

        initial_value = self.initial_capital
        final_value = self.equity_curve[-1]["total_value"]

        total_return = final_value - initial_value
        total_return_pct = (total_return / initial_value) * 100

        sell_trades = [t for t in self.trades if t["action"] == "sell"]
        winning_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0

        total_pnl = sum(t.get("pnl", 0) for t in sell_trades)

        return {
            "initial_capital": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_pnl": total_pnl,
            "total_trades": len(self.trades),
            "sell_trades": len(sell_trades),
            "win_rate": win_rate * 100,
            "winning_trades": len(winning_trades),
            "current_position": self.portfolio["position"],
        }

    def print_summary(self):
        """Print performance summary."""
        metrics = self.get_performance_metrics()

        if not metrics:
            logger.warning("No performance metrics available")
            return

        print("\n" + "=" * 80)
        print("PAPER TRADING PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Symbol:         {self.symbol}")
        print(f"Strategy:       {self.adapter.strategy_type}")
        print(f"Period:         {self.equity_curve[0]['timestamp']} to {self.equity_curve[-1]['timestamp']}")
        print("-" * 80)
        print(f"Initial Capital:  ${metrics['initial_capital']:,.2f}")
        print(f"Final Value:      ${metrics['final_value']:,.2f}")
        print(f"Total Return:     ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:+.2f}%)")
        print(f"Total P&L:        ${metrics['total_pnl']:,.2f}")
        print("-" * 80)
        print(f"Total Trades:     {metrics['total_trades']}")
        print(f"  Sell Trades:    {metrics['sell_trades']}")
        print(f"Win Rate:         {metrics['win_rate']:.2f}%")
        print(f"Winning Trades:   {metrics['winning_trades']}/{metrics['sell_trades']}")
        print("-" * 80)
        print(f"Current Position: {metrics['current_position']:.6f} {self.symbol.split('/')[0]}")
        print("=" * 80 + "\n")


def example_grid_trading_paper_trading():
    """Example: Grid trading with paper trading."""
    print("\n" + "="*80)
    print("EXAMPLE: Grid Trading Paper Trading")
    print("="*80)

    # Create grid trading strategy
    strategy = GridTradingStrategy(
        symbol='BTC/USDT',
        upper_price=55000,
        lower_price=45000,
        num_grids=10,
        grid_mode=GridTradingMode.GEOMETRIC,
        investment_amount=10000,
    )

    # Create adapter
    adapter = ModernStrategyAdapter(strategy)

    # Create paper trader
    trader = ModernStrategyPaperTrader(
        strategy_adapter=adapter,
        symbol='BTC/USDT',
        initial_capital=10000,
    )

    # Run for a few iterations
    for i in range(5):
        print(f"\n--- Iteration {i+1} ---")

        # Fetch data and execute
        data = trader.fetch_latest_data(limit=100)
        result = trader.generate_and_execute_signals(data)

        if result["status"] == "success":
            print(f"Price: ${result['current_price']:,.2f}")
            print(f"Portfolio: ${result['portfolio_value']:,.2f}")

            # Get strategy status
            status = adapter.get_strategy_status(result['current_price'])
            if 'current_position' in status:
                print(f"Position: {status['current_position']:.6f} BTC")

    # Print summary
    trader.print_summary()


def example_smart_dca_paper_trading():
    """Example: Smart DCA with paper trading."""
    print("\n" + "="*80)
    print("EXAMPLE: Smart DCA Paper Trading")
    print("="*80)

    # Create smart DCA strategy
    strategy = SmartDCAStrategy(
        symbol='ETH/USDT',
        total_investment=5000,
        purchase_frequency='daily',
        purchase_amount=100,
        volatility_adjustment=True,
        momentum_boost=0.5,
    )

    # Create adapter
    adapter = ModernStrategyAdapter(strategy)

    # Create paper trader
    trader = ModernStrategyPaperTrader(
        strategy_adapter=adapter,
        symbol='ETH/USDT',
        initial_capital=5000,
    )

    # Run for a few iterations
    for i in range(5):
        print(f"\n--- Iteration {i+1} ---")

        # Fetch data and execute
        data = trader.fetch_latest_data(limit=100)
        result = trader.generate_and_execute_signals(data)

        if result["status"] == "success":
            print(f"Price: ${result['current_price']:,.2f}")
            print(f"Portfolio: ${result['portfolio_value']:,.2f}")
            print(f"Order executed: {result.get('order_executed', False)}")

            # Get DCA status
            status = adapter.get_strategy_status(result['current_price'])
            print(f"Total invested: ${status['total_invested']:,.2f}")
            print(f"Avg price: ${status['avg_purchase_price']:,.2f}")

    # Print summary
    trader.print_summary()


if __name__ == '__main__':
    print("\n" + "="*80)
    print(" "*15 + "MODERN STRATEGIES PAPER TRADING EXAMPLES")
    print("="*80)

    try:
        # Run examples
        example_grid_trading_paper_trading()
        example_smart_dca_paper_trading()

        print("\n" + "="*80)
        print(" "*30 + "EXAMPLES COMPLETED")
        print("="*80)
        print("\nThese examples demonstrate:")
        print("  ✅ Grid trading with automatic limit order placement")
        print("  ✅ Smart DCA with volatility-adjusted purchases")
        print("  ✅ Virtual portfolio tracking and P&L calculation")
        print("  ✅ Performance metrics and reporting")
        print("\nNext steps:")
        print("  1. Run longer backtests with historical data")
        print("  2. Compare different strategy parameters")
        print("  3. Deploy to paper trading with live market data")

    except Exception as e:
        logger.exception(f"Error running examples: {e}")
