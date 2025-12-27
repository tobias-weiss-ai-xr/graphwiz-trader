"""
Grid Trading Paper Trading Deployment

Deploy the optimized Grid Trading strategy to paper trading with:
- Optimal configuration (10 grids, Geometric, ±15% range)
- Real-time market data from Binance
- Continuous trading with automatic execution
- Performance monitoring and reporting

Usage:
    python examples/grid_trading_paper_trading_deploy.py
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import sys

from graphwiz_trader.strategies import (
    GridTradingStrategy,
    GridTradingMode,
    ModernStrategyAdapter,
)
from graphwiz_trader.trading.exchange import create_exchange


class GridTradingPaperTrader:
    """Grid trading paper trading engine with optimal configuration."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        grid_range_pct: float = 0.15,  # ±15% from current price
        num_grids: int = 10,
        grid_mode: GridTradingMode = GridTradingMode.GEOMETRIC,
    ):
        """Initialize grid trading paper trader.

        Args:
            symbol: Trading pair symbol
            initial_capital: Starting virtual capital
            commission: Trading commission rate
            grid_range_pct: Grid range as percentage (default 0.15 = ±15%)
            num_grids: Number of grid levels
            grid_mode: Grid spacing mode
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.commission = commission
        self.grid_range_pct = grid_range_pct

        # Create exchange connection
        self.exchange = create_exchange("binance")

        # Fetch current price
        logger.info(f"Fetching current price for {symbol}...")
        ticker = self.exchange.fetch_ticker(symbol)
        self.current_price = ticker["last"]

        # Create grid trading strategy with optimal configuration
        upper_price = self.current_price * (1 + grid_range_pct)
        lower_price = self.current_price * (1 - grid_range_pct)

        logger.info(f"Current {symbol} price: ${self.current_price:,.2f}")
        logger.info(f"Grid range: ${lower_price:,.2f} - ${upper_price:,.2f} (±{grid_range_pct:.1%})")

        self.strategy = GridTradingStrategy(
            symbol=symbol,
            upper_price=upper_price,
            lower_price=lower_price,
            num_grids=num_grids,
            grid_mode=grid_mode,
            investment_amount=initial_capital,
            dynamic_rebalancing=True,
        )

        self.adapter = ModernStrategyAdapter(self.strategy)

        # Initialize portfolio
        self.portfolio = {
            "capital": initial_capital,
            "position": 0.0,
            "avg_price": 0.0,
            "grid_orders_placed": [],
        }

        # Trade history
        self.trades = []
        self.equity_curve = []
        self.start_time = datetime.now()

        # Statistics
        self.stats = {
            "total_signals": 0,
            "orders_executed": 0,
            "buy_orders": 0,
            "sell_orders": 0,
            "last_rebalance": None,
        }

        logger.success(f"Grid Trading Paper Trader initialized")
        logger.success(f"  Capital: ${initial_capital:,.2f}")
        logger.success(f"  Grids: {num_grids} {grid_mode.value}")
        logger.success(f"  Range: ±{grid_range_pct:.1%}")

    def fetch_market_data(self, limit: int = 100) -> pd.DataFrame:
        """Fetch latest market data.

        Args:
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, "1h", limit=limit)

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise

    def check_grid_range(self) -> bool:
        """Check if current price is within grid range.

        Returns:
            True if price within range, False otherwise
        """
        if self.current_price < self.strategy.lower_price:
            logger.warning(
                f"Price ${self.current_price:,.2f} BELOW grid range "
                f"(${self.strategy.lower_price:,.2f} - ${self.strategy.upper_price:,.2f})"
            )
            return False

        if self.current_price > self.strategy.upper_price:
            logger.warning(
                f"Price ${self.current_price:,.2f} ABOVE grid range "
                f"(${self.strategy.lower_price:,.2f} - ${self.strategy.upper_price:,.2f})"
            )
            return False

        return True

    def generate_and_execute_signals(self, data: pd.DataFrame) -> dict:
        """Generate signals and execute grid orders.

        Args:
            data: Historical market data

        Returns:
            Execution result dictionary
        """
        try:
            # Update current price
            self.current_price = data["close"].iloc[-1]

            # Check if price is within grid range
            in_range = self.check_grid_range()

            if not in_range:
                return {
                    "status": "warning",
                    "message": "Price outside grid range - no trades executed",
                    "current_price": self.current_price,
                    "portfolio_value": self._calculate_portfolio_value(),
                }

            # Generate signals
            signals = self.adapter.generate_trading_signals(
                current_price=self.current_price,
                historical_data=data,
            )

            self.stats["total_signals"] += 1

            # Execute orders near current price (within 1%)
            orders_executed = 0
            for order in signals['orders']:
                # Check if order price is close to current price (within 1%)
                if abs(order['price'] - self.current_price) / self.current_price < 0.01:
                    result = self._execute_grid_order(order)

                    if result:
                        orders_executed += 1
                        self.stats["orders_executed"] += 1

                        if order['side'] == 'buy':
                            self.stats["buy_orders"] += 1
                        else:
                            self.stats["sell_orders"] += 1

            # Check for rebalancing
            if signals.get('rebalance_needed'):
                logger.warning("⚠️  Rebalance recommended due to high volatility")
                self.stats["last_rebalance"] = datetime.now()

            # Update equity
            self._update_equity_curve()

            return {
                "status": "success",
                "current_price": self.current_price,
                "orders_executed": orders_executed,
                "portfolio_value": self._calculate_portfolio_value(),
                "in_range": in_range,
                "total_signals": self.stats["total_signals"],
                "total_orders": self.stats["orders_executed"],
            }

        except Exception as e:
            logger.error(f"Error generating/executing signals: {e}")
            return {
                "status": "error",
                "message": str(e),
                "current_price": self.current_price,
            }

    def _execute_grid_order(self, order: dict) -> dict:
        """Execute a grid order.

        Args:
            order: Order dictionary

        Returns:
            Trade result if executed, None otherwise
        """
        side = order['side']
        quantity = order['quantity']
        price = order['price']

        # Check if we already have an order at this price level
        for existing_order in self.portfolio["grid_orders_placed"]:
            if existing_order['price'] == price and existing_order['status'] == 'open':
                logger.debug(f"Order already exists at ${price:,.2f}")
                return None

        capital = self.portfolio["capital"]
        position = self.portfolio["position"]

        trade = None

        if side == "buy":
            # Check if we have enough capital
            cost = quantity * price * (1 + self.commission)
            if cost > capital:
                logger.debug(f"Insufficient capital: need ${cost:.2f}, have ${capital:.2f}")
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

            # Add to grid orders
            self.portfolio["grid_orders_placed"].append({
                'side': 'buy',
                'quantity': quantity,
                'price': price,
                'status': 'filled',
                'timestamp': datetime.now(),
            })

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
                logger.debug(f"Insufficient position: need {quantity:.6f}, have {position:.6f}")
                return None

            # Execute sell
            proceeds = quantity * price * (1 - self.commission)
            pnl = proceeds - (quantity * self.portfolio["avg_price"])
            pnl_pct = (pnl / (quantity * self.portfolio["avg_price"])) * 100

            self.portfolio["capital"] += proceeds
            self.portfolio["position"] -= quantity

            # Add to grid orders
            self.portfolio["grid_orders_placed"].append({
                'side': 'sell',
                'quantity': quantity,
                'price': price,
                'status': 'filled',
                'timestamp': datetime.now(),
            })

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

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value.

        Returns:
            Total portfolio value
        """
        capital = self.portfolio["capital"]
        position = self.portfolio["position"]
        position_value = position * self.current_price
        return capital + position_value

    def _update_equity_curve(self):
        """Update equity curve."""
        total_value = self._calculate_portfolio_value()

        self.equity_curve.append({
            "timestamp": datetime.now(),
            "capital": self.portfolio["capital"],
            "position": self.portfolio["position"],
            "position_value": self.portfolio["position"] * self.current_price,
            "total_value": total_value,
            "current_price": self.current_price,
        })

    def get_performance_metrics(self) -> dict:
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

        # Calculate max drawdown
        equity_values = [e["total_value"] for e in self.equity_curve]
        running_max = pd.Series(equity_values).cummax()
        drawdown = pd.Series(equity_values) - running_max
        max_drawdown = (drawdown.min() / initial_value) * 100 if initial_value > 0 else 0

        # Calculate runtime
        runtime = datetime.now() - self.start_time

        return {
            "runtime_hours": runtime.total_seconds() / 3600,
            "initial_capital": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_pnl": total_pnl,
            "total_trades": len(self.trades),
            "buy_orders": self.stats["buy_orders"],
            "sell_orders": self.stats["sell_orders"],
            "win_rate": win_rate * 100,
            "winning_trades": len(winning_trades),
            "max_drawdown": abs(max_drawdown),
            "current_position": self.portfolio["position"],
            "current_price": self.current_price,
            "in_grid_range": self.check_grid_range(),
        }

    def save_results(self, output_dir: str = "data/paper_trading"):
        """Save trading results to files.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_clean = self.symbol.replace("/", "_")

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = output_path / f"{symbol_clean}_grid_trades_{timestamp_str}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved trades to {trades_file}")

        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = output_path / f"{symbol_clean}_grid_equity_{timestamp_str}.csv"
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"Saved equity curve to {equity_file}")

        # Save performance summary
        metrics = self.get_performance_metrics()
        if metrics:
            summary_file = output_path / f"{symbol_clean}_grid_summary_{timestamp_str}.json"
            with open(summary_file, "w") as f:
                import json
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Saved summary to {summary_file}")

    def print_summary(self):
        """Print performance summary."""
        metrics = self.get_performance_metrics()

        if not metrics:
            logger.warning("No performance metrics available")
            return

        print("\n" + "="*80)
        print("GRID TRADING PAPER TRADING PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Symbol:         {self.symbol}")
        print(f"Strategy:       Grid Trading ({self.strategy.num_grids} grids, {self.strategy.grid_mode.value})")
        print(f"Grid Range:      ${self.strategy.lower_price:,.2f} - ${self.strategy.upper_price:,.2f}")
        print(f"Runtime:         {metrics['runtime_hours']:.1f} hours")
        print("-"*80)
        print(f"Initial Capital:  ${metrics['initial_capital']:,.2f}")
        print(f"Final Value:      ${metrics['final_value']:,.2f}")
        print(f"Total Return:     ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:+.2f}%)")
        print(f"Total P&L:        ${metrics['total_pnl']:,.2f}")
        print("-"*80)
        print(f"Total Trades:     {metrics['total_trades']}")
        print(f"  Buy Orders:     {metrics['buy_orders']}")
        print(f"  Sell Orders:    {metrics['sell_orders']}")
        print(f"Win Rate:         {metrics['win_rate']:.2f}%")
        print(f"Winning Trades:   {metrics['winning_trades']}/{metrics['sell_orders']}")
        print("-"*80)
        print(f"Max Drawdown:     {metrics['max_drawdown']:.2f}%")
        print(f"Current Position: {metrics['current_position']:.6f} {self.symbol.split('/')[0]}")
        print(f"Current Price:    ${metrics['current_price']:,.2f}")
        print(f"In Grid Range:    {metrics['in_grid_range']}")
        print("="*80 + "\n")

    def run_iteration(self) -> dict:
        """Run one iteration of paper trading.

        Returns:
            Status dict with current state
        """
        try:
            # Fetch market data
            data = self.fetch_market_data(limit=100)

            # Generate and execute signals
            result = self.generate_and_execute_signals(data)

            return result

        except Exception as e:
            logger.error(f"Error in iteration: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    def run(
        self,
        interval_seconds: int = 3600,
        iterations: int = None,
        auto_save: bool = True,
    ):
        """Run continuous paper trading.

        Args:
            interval_seconds: Seconds between iterations (default: 1 hour)
            iterations: Maximum number of iterations (None = infinite)
            auto_save: Auto-save results every 10 iterations
        """
        logger.info(f"Starting Grid Trading paper trading for {self.symbol}")
        logger.info(f"Interval: {interval_seconds}s, Max iterations: {iterations or 'infinite'}")
        logger.info(f"Auto-save: {auto_save}")

        iteration = 0

        try:
            while True:
                iteration += 1

                if iterations and iteration > iterations:
                    logger.info(f"Reached max iterations ({iterations})")
                    break

                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {iteration}")
                logger.info(f"{'='*60}")

                # Run one iteration
                result = self.run_iteration()

                if result["status"] == "success":
                    logger.info(
                        f"Price: ${result['current_price']:,.2f}, "
                        f"Portfolio: ${result['portfolio_value']:,.2f} "
                    )

                    if result.get('orders_executed', 0) > 0:
                        logger.info(f"Orders executed: {result['orders_executed']}")

                elif result["status"] == "warning":
                    logger.warning(result.get('message', 'Unknown warning'))

                else:
                    logger.error(f"Iteration failed: {result.get('message')}")

                # Auto-save every 10 iterations
                if auto_save and iteration % 10 == 0:
                    logger.info("Auto-saving results...")
                    self.save_results()
                    self.print_summary()

                # Wait for next iteration
                if iteration < (iterations or float('inf')):
                    logger.info(f"Waiting {interval_seconds}s until next iteration...")
                    time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal, shutting down...")

        finally:
            logger.success("Paper trading stopped")
            self.save_results()
            self.print_summary()


def main():
    """Main deployment function."""
    print("\n" + "="*80)
    print(" "*15 + "GRID TRADING PAPER TRADING DEPLOYMENT")
    print("="*80)
    print("\nDeploying Grid Trading strategy with OPTIMAL configuration:")
    print("  • 10 grids (optimal balance)")
    print("  • Geometric mode (+10.6% better than arithmetic)")
    print("  • ±15% range (covers most price movement)")
    print("  • Real-time Binance market data")
    print("  • Automatic order execution")
    print("\n" + "="*80)

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Create trader with optimal configuration
    trader = GridTradingPaperTrader(
        symbol="BTC/USDT",
        initial_capital=10000,
        grid_range_pct=0.15,  # ±15% (optimal)
        num_grids=10,  # Optimal
        grid_mode=GridTradingMode.GEOMETRIC,  # Best performer
    )

    # Run paper trading
    # Options:
    # - Run for 24 iterations (1 day, hourly)
    # - Run for 168 iterations (1 week, hourly)
    # - Run indefinitely (no limit)

    print("\nDeployment Options:")
    print("  1. Quick test (5 iterations, 1 min intervals)")
    print("  2. 1 day test (24 iterations, 1 hour intervals)")
    print("  3. 1 week test (168 iterations, 1 hour intervals)")
    print("  4. Continuous (infinite iterations, 1 hour intervals)")
    print()

    # For demonstration, run 5 iterations with 1-minute intervals
    # Change these parameters for production:
    # - interval_seconds=3600 for hourly checks
    # - iterations=None for infinite running

    trader.run(
        interval_seconds=60,  # 1 minute for demo (use 3600 for production)
        iterations=5,  # 5 iterations for demo (use None for infinite)
        auto_save=True,
    )


if __name__ == '__main__':
    main()
