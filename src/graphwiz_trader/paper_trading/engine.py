"""
Paper trading engine for simulated trading.

This engine simulates live trading without risking real money.
It fetches real market data, generates signals, and tracks virtual trades.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import json

from loguru import logger
from ccxt import Exchange

from ..trading.exchange import create_exchange
from ..backtesting import RSIMeanReversionStrategy


class PaperTradingEngine:
    """Paper trading engine for simulated trading."""

    def __init__(
        self,
        exchange_name: str = "binance",
        symbol: str = "BTC/USDT",
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% Binance fee
        strategy_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize paper trading engine.

        Args:
            exchange_name: Exchange to use (default: binance)
            symbol: Trading pair symbol
            initial_capital: Starting virtual capital
            commission: Trading commission rate
            strategy_config: Strategy configuration
        """
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.commission = commission

        # Create exchange (read-only for market data)
        self.exchange: Exchange = create_exchange(exchange_name)

        # Initialize portfolio
        self.portfolio = {
            "capital": initial_capital,
            "position": 0.0,
            "avg_price": 0.0,
        }

        # Strategy (default: RSI with optimized parameters)
        strategy_config = strategy_config or {"oversold": 25, "overbought": 65}
        self.strategy = RSIMeanReversionStrategy(**strategy_config)

        # Trade history
        self.trades = []
        self.equity_curve = []

        # State
        self.is_running = False
        self.last_signal = None

        logger.info(f"Initialized paper trading: {symbol} with ${initial_capital:,.2f}")

    def fetch_latest_data(self, limit: int = 100) -> pd.DataFrame:
        """Fetch latest market data.

        Args:
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, "1d", limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            logger.debug(f"Fetched {len(df)} candles for {self.symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            raise

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """Generate trading signal using strategy.

        Args:
            data: Market data DataFrame

        Returns:
            Signal: 'buy', 'sell', or None
        """
        # Generate signal using strategy
        signal = self.strategy.generate_signal(data)

        if signal != self.last_signal:
            logger.info(f"Signal generated: {signal}")
            self.last_signal = signal

        return signal

    def execute_virtual_trade(
        self, signal: str, price: float, timestamp: datetime
    ) -> Optional[Dict]:
        """Execute a virtual trade (no real money).

        Args:
            signal: Trading signal ('buy' or 'sell')
            price: Current price
            timestamp: Trade timestamp

        Returns:
            Trade dict if executed, None otherwise
        """
        if not signal or signal not in ["buy", "sell"]:
            return None

        capital = self.portfolio["capital"]
        position = self.portfolio["position"]

        trade = None

        if signal == "buy":
            # Calculate max quantity we can buy
            if capital <= 0:
                logger.warning("No capital available to buy")
                return None

            # Buy with 95% of available capital (keep 5% buffer)
            max_value = capital * 0.95
            cost = max_value * (1 + self.commission)
            quantity = max_value / price

            # Update portfolio
            self.portfolio["capital"] -= cost
            self.portfolio["position"] += quantity

            # Update average price (for P&L calculation)
            total_value = (position * self.portfolio["avg_price"]) + (quantity * price)
            total_quantity = position + quantity
            self.portfolio["avg_price"] = (
                total_value / total_quantity if total_quantity > 0 else price
            )

            trade = {
                "timestamp": timestamp,
                "action": "buy",
                "price": price,
                "quantity": quantity,
                "value": max_value,
                "cost": cost,
                "capital_after": self.portfolio["capital"],
                "position_after": self.portfolio["position"],
            }

            logger.success(
                f"🟢 BOUGHT {quantity:.4f} @ ${price:,.2f} "
                f"(Value: ${max_value:,.2f}, Cost: ${cost:,.2f})"
            )

        elif signal == "sell":
            # Sell entire position
            if position <= 0:
                logger.warning("No position to sell")
                return None

            # Calculate proceeds
            proceeds = position * price * (1 - self.commission)
            pnl = proceeds - (position * self.portfolio["avg_price"])
            pnl_pct = (pnl / (position * self.portfolio["avg_price"])) * 100

            # Update portfolio
            self.portfolio["capital"] += proceeds
            self.portfolio["position"] = 0
            self.portfolio["avg_price"] = 0

            trade = {
                "timestamp": timestamp,
                "action": "sell",
                "price": price,
                "quantity": position,
                "value": position * price,
                "proceeds": proceeds,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "capital_after": self.portfolio["capital"],
                "position_after": 0,
            }

            if pnl >= 0:
                logger.success(
                    f"🔵 SOLD {position:.4f} @ ${price:,.2f} "
                    f"(P&L: ${pnl:,.2f}, {pnl_pct:+.2f}%)"
                )
            else:
                logger.warning(
                    f"🔴 SOLD {position:.4f} @ ${price:,.2f} "
                    f"(P&L: ${pnl:,.2f}, {pnl_pct:+.2f}%)"
                )

        # Record trade
        if trade:
            self.trades.append(trade)

        return trade

    def calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value (capital + position).

        Args:
            current_price: Current market price

        Returns:
            Total portfolio value
        """
        capital = self.portfolio["capital"]
        position = self.portfolio["position"]
        position_value = position * current_price
        return capital + position_value

    def update_equity_curve(self, timestamp: datetime, current_price: float):
        """Update equity curve with current portfolio value.

        Args:
            timestamp: Current timestamp
            current_price: Current market price
        """
        total_value = self.calculate_portfolio_value(current_price)

        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "capital": self.portfolio["capital"],
                "position": self.portfolio["position"],
                "position_value": self.portfolio["position"] * current_price,
                "total_value": total_value,
            }
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return {}

        # Initial and final values
        initial_value = self.initial_capital
        final_value = self.equity_curve[-1]["total_value"]

        # Total return
        total_return = final_value - initial_value
        total_return_pct = (total_return / initial_value) * 100

        # Trades
        buy_trades = [t for t in self.trades if t["action"] == "buy"]
        sell_trades = [t for t in self.trades if t["action"] == "sell"]

        # Win rate (from sell trades)
        winning_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0

        # Total P&L
        total_pnl = sum(t.get("pnl", 0) for t in sell_trades)

        # Calculate drawdown
        equity_values = [e["total_value"] for e in self.equity_curve]
        running_max = pd.Series(equity_values).cummax()
        drawdown = pd.Series(equity_values) - running_max
        max_drawdown = (drawdown.min() / initial_value) * 100 if initial_value > 0 else 0

        return {
            "initial_capital": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_pnl": total_pnl,
            "total_trades": len(self.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "win_rate": win_rate * 100,
            "winning_trades": len(winning_trades),
            "max_drawdown": abs(max_drawdown),
            "current_position": self.portfolio["position"],
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
            trades_file = output_path / f"{symbol_clean}_trades_{timestamp_str}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved trades to {trades_file}")

        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = output_path / f"{symbol_clean}_equity_{timestamp_str}.csv"
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"Saved equity curve to {equity_file}")

        # Save performance summary
        metrics = self.get_performance_metrics()
        if metrics:
            summary_file = output_path / f"{symbol_clean}_summary_{timestamp_str}.json"
            with open(summary_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Saved summary to {summary_file}")

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
        print(f"Strategy:       RSI({self.strategy.oversold}/{self.strategy.overbought})")
        print(
            f"Period:         {self.equity_curve[0]['timestamp']} to {self.equity_curve[-1]['timestamp']}"
        )
        print("-" * 80)
        print(f"Initial Capital:  ${metrics['initial_capital']:,.2f}")
        print(f"Final Value:      ${metrics['final_value']:,.2f}")
        print(
            f"Total Return:     ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:+.2f}%)"
        )
        print(f"Total P&L:        ${metrics['total_pnl']:,.2f}")
        print("-" * 80)
        print(f"Total Trades:     {metrics['total_trades']}")
        print(f"  Buy Trades:     {metrics['buy_trades']}")
        print(f"  Sell Trades:    {metrics['sell_trades']}")
        print(f"Win Rate:         {metrics['win_rate']:.2f}%")
        print(f"Winning Trades:   {metrics['winning_trades']}/{metrics['sell_trades']}")
        print("-" * 80)
        print(f"Max Drawdown:     {metrics['max_drawdown']:.2f}%")
        print(f"Current Position: {metrics['current_position']:.4f} {self.symbol.split('/')[0]}")
        print("=" * 80 + "\n")

    def run_once(self) -> Dict[str, Any]:
        """Run one iteration of paper trading.

        Returns:
            Status dict with current state
        """
        try:
            # Fetch latest data
            data = self.fetch_latest_data(limit=100)

            if data.empty:
                return {"status": "error", "message": "No data available"}

            # Get current price
            current_price = data["close"].iloc[-1]
            current_time = data["timestamp"].iloc[-1]

            # Generate signal
            signal = self.generate_signal(data)

            # Execute trade if signal
            trade = None
            if signal:
                trade = self.execute_virtual_trade(signal, current_price, current_time)

            # Update equity curve
            self.update_equity_curve(current_time, current_price)

            # Get current metrics
            metrics = self.get_performance_metrics()

            return {
                "status": "success",
                "timestamp": current_time,
                "price": current_price,
                "signal": signal,
                "trade": trade,
                "portfolio_value": metrics.get("final_value"),
                "total_return_pct": metrics.get("total_return_pct"),
            }

        except Exception as e:
            logger.error(f"Error in paper trading iteration: {e}")
            return {"status": "error", "message": str(e)}

    def start(self, interval_seconds: int = 3600, max_iterations: Optional[int] = None):
        """Start continuous paper trading.

        Args:
            interval_seconds: Seconds between iterations (default: 1 hour)
            max_iterations: Maximum number of iterations (None = infinite)
        """
        self.is_running = True
        iteration = 0

        logger.info(f"Starting paper trading for {self.symbol}")
        logger.info(
            f"Interval: {interval_seconds}s, Max iterations: {max_iterations or 'infinite'}"
        )

        try:
            while self.is_running:
                iteration += 1

                if max_iterations and iteration > max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break

                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {iteration}")
                logger.info(f"{'='*60}")

                # Run one iteration
                result = self.run_once()

                if result["status"] == "success":
                    logger.info(
                        f"Price: ${result['price']:,.2f}, "
                        f"Portfolio: ${result['portfolio_value']:,.2f} "
                        f"({result['total_return_pct']:+.2f}%)"
                    )
                else:
                    logger.error(f"Iteration failed: {result.get('message')}")

                # Save results every 10 iterations
                if iteration % 10 == 0:
                    self.save_results()

                # Wait for next iteration
                if iteration < (max_iterations or float("inf")):
                    logger.info(f"Waiting {interval_seconds}s until next iteration...")
                    time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal, shutting down...")

        finally:
            self.is_running = False
            self.save_results()
            self.print_summary()
            logger.success("Paper trading stopped")
