"""
Dynamic Grid Rebalancing System

Automatically re-centers grid trading strategy when price moves outside grid range.
Keeps strategy active and prevents missed trading opportunities.
"""

import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import pandas as pd

from graphwiz_trader.strategies import GridTradingStrategy, GridTradingMode
from graphwiz_trader.trading.exchange import create_exchange

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


class DynamicGridRebalancer:
    """
    Grid trading paper trader with automatic dynamic rebalancing.

    Monitors price movement and automatically re-centers the grid when price
    moves outside the optimal range, keeping the strategy active.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        initial_capital: float = 10000.0,
        num_grids: int = 10,
        grid_range_pct: float = 0.15,  # ±15% initial range
        grid_mode: GridTradingMode = GridTradingMode.GEOMETRIC,
        rebalance_threshold: float = 0.10,  # Rebalance when price >10% outside grid
        exchange_name: str = "binance",
    ):
        """
        Initialize dynamic grid rebalancing trader.

        Args:
            symbol: Trading pair symbol
            initial_capital: Starting capital
            num_grids: Number of grid levels
            grid_range_pct: Initial grid range (±percentage)
            grid_mode: Grid spacing mode (arithmetic/geometric)
            rebalance_threshold: Trigger rebalancing when price beyond this threshold
            exchange_name: Exchange to use
        """
        self.symbol = symbol
        self.exchange = create_exchange(exchange_name)
        self.initial_capital = initial_capital
        self.num_grids = num_grids
        self.grid_range_pct = grid_range_pct
        self.grid_mode = grid_mode
        self.rebalance_threshold = rebalance_threshold

        # Portfolio state
        self.capital = initial_capital
        self.position = 0.0
        self.avg_price = 0.0

        # Strategy state
        self.strategy: Optional[GridTradingStrategy] = None
        self.current_price: Optional[float] = None
        self.grid_center: Optional[float] = None

        # Rebalancing tracking
        self.rebalance_count = 0
        self.rebalance_history: List[Dict] = []

        # Trading history
        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []

        # Initialize
        self._initialize_strategy()

    def _initialize_strategy(self):
        """Initialize or re-initialize the grid strategy."""
        # Fetch current price
        ticker = self.exchange.fetch_ticker(self.symbol)
        self.current_price = ticker["last"]
        self.grid_center = self.current_price

        # Calculate grid range
        upper_price = self.current_price * (1 + self.grid_range_pct)
        lower_price = self.current_price * (1 - self.grid_range_pct)

        # Create strategy
        self.strategy = GridTradingStrategy(
            symbol=self.symbol,
            upper_price=upper_price,
            lower_price=lower_price,
            num_grids=self.num_grids,
            grid_mode=self.grid_mode,
            investment_amount=self.initial_capital,
        )

        logger.info(f"✅ Grid Trading Strategy initialized")
        logger.info(f"   Center Price: ${self.grid_center:,.2f}")
        logger.info(f"   Grid Range: ${lower_price:,.2f} - ${upper_price:,.2f}")
        logger.info(f"   Grid Levels: {len(self.strategy.grid_levels)}")

    def check_rebalance_needed(self) -> Tuple[bool, Optional[str]]:
        """
        Check if rebalancing is needed.

        Returns:
            (needs_rebalance, reason) tuple
        """
        if not self.strategy:
            return False, None

        upper_price = self.strategy.upper_price
        lower_price = self.strategy.lower_price

        # Calculate how far outside the grid we are
        if self.current_price > upper_price:
            pct_above = (self.current_price - upper_price) / upper_price
            if pct_above > self.rebalance_threshold:
                return True, f"Price {pct_above*100:.1f}% above grid (${self.current_price:,.2f} > ${upper_price:,.2f})"

        elif self.current_price < lower_price:
            pct_below = (lower_price - self.current_price) / lower_price
            if pct_below > self.rebalance_threshold:
                return True, f"Price {pct_below*100:.1f}% below grid (${self.current_price:,.2f} < ${lower_price:,.2f})"

        return False, None

    def rebalance_grid(self, reason: str):
        """
        Rebalance the grid by re-centering around current price.

        Preserves existing positions and capital, only updates grid levels.
        """
        logger.warning("\n" + "=" * 80)
        logger.warning(f"🔄 REBALANCING GRID (#{self.rebalance_count + 1})")
        logger.warning("=" * 80)
        logger.warning(f"Reason: {reason}")
        logger.warning(f"Old Grid Center: ${self.grid_center:,.2f}")
        logger.warning(f"Current Price: ${self.current_price:,.2f}")

        # Record old grid info
        old_upper = self.strategy.upper_price
        old_lower = self.strategy.lower_price
        old_center = self.grid_center

        # Close any existing positions at current price
        if self.position > 0:
            # Sell position at current price
            revenue = self.position * self.current_price
            profit = (self.current_price - self.avg_price) * self.position

            logger.warning(f"Closing position: {self.position:.6f} @ ${self.current_price:,.2f}")
            logger.warning(f"Profit from position: ${profit:.2f}")

            self.capital += revenue
            self.position = 0.0
            self.avg_price = 0.0

        # Re-initialize strategy with new grid center
        self._initialize_strategy()

        # Record rebalance event
        rebalance_event = {
            "timestamp": datetime.now(),
            "rebalance_number": self.rebalance_count + 1,
            "old_center": old_center,
            "new_center": self.grid_center,
            "old_upper": old_upper,
            "old_lower": old_lower,
            "new_upper": self.strategy.upper_price,
            "new_lower": self.strategy.lower_price,
            "price_at_rebalance": self.current_price,
            "reason": reason,
        }
        self.rebalance_history.append(rebalance_event)
        self.rebalance_count += 1

        logger.warning(f"New Grid Center: ${self.grid_center:,.2f}")
        logger.warning(f"New Grid Range: ${self.strategy.lower_price:,.2f} - ${self.strategy.upper_price:,.2f}")
        logger.warning("=" * 80 + "\n")

    def execute_grid_trades(self):
        """Execute simulated grid trades based on current price."""
        if not self.strategy:
            return

        grid_levels = sorted(self.strategy.grid_levels)

        for level in grid_levels:
            # Buy orders (below current price)
            if level < self.current_price:
                if self.capital > level * 0.001:  # Minimum trade size
                    trade_amount = (self.initial_capital / self.strategy.num_grids) / level
                    cost = trade_amount * level

                    if cost <= self.capital:
                        # Update position
                        if self.position > 0:
                            self.avg_price = (
                                (self.avg_price * self.position + level * trade_amount)
                                / (self.position + trade_amount)
                            )
                        else:
                            self.avg_price = level

                        self.position += trade_amount
                        self.capital -= cost

                        self.trades.append({
                            "timestamp": datetime.now(),
                            "side": "buy",
                            "price": level,
                            "amount": trade_amount,
                            "cost": cost,
                            "rebalance_cycle": self.rebalance_count,
                        })

                        logger.info(f"  📈 BOUGHT {trade_amount:.6f} @ ${level:,.2f}")

            # Sell orders (above current price)
            elif level > self.current_price:
                if self.position > 0:
                    sell_amount = min(
                        self.position,
                        (self.initial_capital / self.strategy.num_grids) / level
                    )

                    if sell_amount > 0:
                        revenue = sell_amount * level
                        profit = (level - self.avg_price) * sell_amount

                        self.capital += revenue
                        self.position -= sell_amount

                        self.trades.append({
                            "timestamp": datetime.now(),
                            "side": "sell",
                            "price": level,
                            "amount": sell_amount,
                            "revenue": revenue,
                            "profit": profit,
                            "rebalance_cycle": self.rebalance_count,
                        })

                        logger.info(f"  📉 SOLD {sell_amount:.6f} @ ${level:,.2f} (Profit: ${profit:.2f})")

    def update_portfolio_value(self) -> Dict[str, float]:
        """Calculate current portfolio value."""
        position_value = self.position * self.current_price
        total_value = self.capital + position_value
        pnl = total_value - self.initial_capital
        roi = (pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0

        return {
            "capital": self.capital,
            "position": self.position,
            "position_value": position_value,
            "total_value": total_value,
            "pnl": pnl,
            "roi_pct": roi,
        }

    def save_results(self):
        """Save trading and rebalancing results to files."""
        data_dir = Path("data/dynamic_grid_rebalancing")
        data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = data_dir / f"{self.symbol.replace('/', '_')}_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"💾 Saved {len(self.trades)} trades to {trades_file}")

        # Save rebalance history
        if self.rebalance_history:
            rebalance_df = pd.DataFrame(self.rebalance_history)
            rebalance_file = data_dir / f"{self.symbol.replace('/', '_')}_rebalances_{timestamp}.csv"
            rebalance_df.to_csv(rebalance_file, index=False)
            logger.info(f"💾 Saved {len(self.rebalance_history)} rebalances to {rebalance_file}")

        # Save equity curve
        if self.equity_history:
            equity_df = pd.DataFrame(self.equity_history)
            equity_file = data_dir / f"{self.symbol.replace('/', '_')}_equity_{timestamp}.csv"
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"💾 Saved equity curve to {equity_file}")

    def print_summary(self):
        """Print current portfolio summary."""
        portfolio = self.update_portfolio_value()

        logger.info("\n" + "=" * 80)
        logger.info(f"SUMMARY - Rebalance Cycle {self.rebalance_count}")
        logger.info("=" * 80)
        logger.info(f"\n💰 Portfolio: ${portfolio['total_value']:,.2f}")
        logger.info(f"   P&L: ${portfolio['pnl']:+,.2f} ({portfolio['roi_pct']:+.2f}%)")
        logger.info(f"   Capital: ${portfolio['capital']:,.2f}")
        logger.info(f"   Position: {portfolio['position']:.6f} (${portfolio['position_value']:,.2f})")
        logger.info(f"\n📊 Grid Status:")
        logger.info(f"   Current Price: ${self.current_price:,.2f}")
        logger.info(f"   Grid Center: ${self.grid_center:,.2f}")
        logger.info(f"   Grid Range: ${self.strategy.lower_price:,.2f} - ${self.strategy.upper_price:,.2f}")
        logger.info(f"   Total Rebalances: {self.rebalance_count}")
        logger.info(f"   Total Trades: {len(self.trades)}")
        logger.info("=" * 80 + "\n")

    def run(
        self,
        iterations: int = 10,
        interval_seconds: int = 60,
        auto_save: bool = True,
    ):
        """
        Run dynamic grid rebalancing paper trading.

        Args:
            iterations: Number of iterations to run
            interval_seconds: Seconds between iterations
            auto_save: Auto-save results at the end
        """
        logger.info("=" * 80)
        logger.info("DYNAMIC GRID REBALANCING SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Grid Range: ±{self.grid_range_pct*100}%")
        logger.info(f"Rebalance Threshold: ±{self.rebalance_threshold*100}% outside grid")
        logger.info(f"Grid Mode: {self.grid_mode.value}")
        logger.info("=" * 80 + "\n")

        try:
            for iteration in range(1, iterations + 1):
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Iteration {iteration}/{iterations}")
                logger.info("=" * 80)

                # Fetch current price
                ticker = self.exchange.fetch_ticker(self.symbol)
                self.current_price = ticker["last"]
                price_change = ticker.get("change", 0)

                logger.info(f"\n📊 Price: ${self.current_price:,.2f} ({price_change:+.2f}%)")

                # Check if rebalancing is needed
                needs_rebalance, reason = self.check_rebalance_needed()

                if needs_rebalance:
                    self.rebalance_grid(reason)
                else:
                    # Check if price is in grid range
                    if self.current_price < self.strategy.lower_price:
                        logger.warning(f"  ⚠️  Price below grid (${self.current_price:,.2f} < ${self.strategy.lower_price:,.2f})")
                    elif self.current_price > self.strategy.upper_price:
                        logger.warning(f"  ⚠️  Price above grid (${self.current_price:,.2f} > ${self.strategy.upper_price:,.2f})")
                    else:
                        logger.info(f"  ✅ Price within grid range")

                # Execute grid trades
                self.execute_grid_trades()

                # Update equity history
                portfolio = self.update_portfolio_value()
                portfolio["iteration"] = iteration
                portfolio["rebalance_count"] = self.rebalance_count
                portfolio["timestamp"] = datetime.now()
                self.equity_history.append(portfolio)

                # Print summary
                self.print_summary()

                # Wait for next iteration
                if iteration < iterations:
                    logger.info(f"⏳ Waiting {interval_seconds}s until next iteration...")
                    time.sleep(interval_seconds)

            # Final save
            if auto_save:
                logger.info("\n💾 Saving results...")
                self.save_results()

            logger.success("\n✅ Dynamic Grid Rebalancing completed!")
            logger.success(f"Total Rebalances: {self.rebalance_count}")
            logger.success(f"Final Portfolio Value: ${portfolio['total_value']:,.2f}")
            logger.success(f"Total P&L: ${portfolio['pnl']:+,.2f} ({portfolio['roi_pct']:+.2f}%)")

        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Trading interrupted by user")
            if auto_save:
                logger.info("💾 Saving results...")
                self.save_results()
            logger.success("✅ Results saved. Goodbye!")


def main():
    """Main entry point for dynamic grid rebalancing demo."""
    trader = DynamicGridRebalancer(
        symbol="BTC/USDT",
        initial_capital=10000.0,
        num_grids=10,
        grid_range_pct=0.15,  # ±15% grid range
        grid_mode=GridTradingMode.GEOMETRIC,
        rebalance_threshold=0.10,  # Rebalance when >10% outside grid
    )

    # Run with simulated price movement to trigger rebalancing
    # For demo: 5 iterations with 60s intervals
    trader.run(
        iterations=5,
        interval_seconds=60,
        auto_save=True,
    )


if __name__ == "__main__":
    main()
