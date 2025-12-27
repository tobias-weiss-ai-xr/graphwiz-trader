"""
Multi-Symbol Grid Trading Paper Trading System

Trade multiple cryptocurrency pairs simultaneously with portfolio-level risk management.
Each symbol has its own grid strategy, managed under a unified portfolio.
"""

import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
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


@dataclass
class SymbolConfig:
    """Configuration for a single symbol in the portfolio."""
    symbol: str
    capital_allocation: float  # Percentage of total capital (0-1)
    num_grids: int = 10
    grid_range_pct: float = 0.15  # ±15%
    grid_mode: GridTradingMode = GridTradingMode.GEOMETRIC


@dataclass
class SymbolState:
    """State tracking for a single symbol."""
    symbol: str
    strategy: GridTradingStrategy
    capital: float
    initial_capital: float = 0.0
    position: float = 0.0
    avg_price: float = 0.0
    trades: List[Dict] = field(default_factory=list)

    def portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value for this symbol."""
        position_value = self.position * current_price
        return self.capital + position_value

    def pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.position == 0:
            return 0.0
        return (current_price - self.avg_price) * self.position

    def investment_amount(self) -> float:
        """Get initial investment amount for this symbol."""
        return self.initial_capital


class MultiSymbolGridTrader:
    """
    Multi-symbol grid trading paper trading system.

    Manages multiple grid trading strategies across different cryptocurrency pairs
    with unified portfolio tracking and risk management.
    """

    def __init__(
        self,
        symbols: List[SymbolConfig],
        total_capital: float = 10000.0,
        exchange_name: str = "binance",
    ):
        """
        Initialize multi-symbol grid trader.

        Args:
            symbols: List of symbol configurations
            total_capital: Total capital across all symbols
            exchange_name: Exchange to use (default: binance)
        """
        self.exchange = create_exchange(exchange_name)
        self.total_capital = total_capital
        self.symbols_config = symbols
        self.symbol_states: Dict[str, SymbolState] = {}
        self.iteration = 0
        self.equity_history: List[Dict] = []

        # Validate capital allocation
        total_allocation = sum(s.capital_allocation for s in symbols)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(
                f"Capital allocation must sum to 1.0, got {total_allocation:.2f}"
            )

        # Initialize each symbol
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize grid trading strategies for all symbols."""
        logger.info("=" * 80)
        logger.info("Initializing Multi-Symbol Grid Trading System")
        logger.info("=" * 80)

        for config in self.symbols_config:
            # Allocate capital
            capital = self.total_capital * config.capital_allocation

            # Fetch current price
            logger.info(f"\nInitializing {config.symbol}...")
            ticker = self.exchange.fetch_ticker(config.symbol)
            current_price = ticker["last"]

            # Calculate grid range
            upper_price = current_price * (1 + config.grid_range_pct)
            lower_price = current_price * (1 - config.grid_range_pct)

            # Create strategy
            strategy = GridTradingStrategy(
                symbol=config.symbol,
                upper_price=upper_price,
                lower_price=lower_price,
                num_grids=config.num_grids,
                grid_mode=config.grid_mode,
                investment_amount=capital,
            )

            # Create state
            self.symbol_states[config.symbol] = SymbolState(
                symbol=config.symbol,
                strategy=strategy,
                capital=capital,
                initial_capital=capital,
            )

            logger.info(f"✅ {config.symbol}: ${current_price:,.2f}")
            logger.info(f"   Grid: ${lower_price:,.2f} - ${upper_price:,.2f}")
            logger.info(f"   Allocation: ${capital:,.2f} ({config.capital_allocation*100:.1f}%)")

        logger.info("\n" + "=" * 80)
        logger.success(f"Multi-Symbol Grid Trader initialized")
        logger.success(f"  Symbols: {len(self.symbols_config)}")
        logger.success(f"  Total Capital: ${self.total_capital:,.2f}")
        logger.info("=" * 80 + "\n")

    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch current market data for a symbol."""
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "price": ticker["last"],
            "volume": ticker["baseVolume"],
            "change": ticker["change"],
        }

    def execute_grid_trades(self, symbol: str, current_price: float):
        """
        Execute grid trades for a symbol.

        Simulates grid trading behavior by checking if price crosses grid levels.
        """
        state = self.symbol_states[symbol]
        strategy = state.strategy

        # Get grid levels
        grid_levels = sorted(strategy.grid_levels)

        # Find nearby grid levels
        for level in grid_levels:
            # Buy condition: price crosses below a grid level
            if level < current_price:
                # Check if we should buy (simulate grid order fill)
                if state.capital > level * 0.001:  # Minimum trade size
                    # Calculate trade amount (equal distribution across grids)
                    trade_amount = (state.investment_amount() / strategy.num_grids) / level

                    # Execute buy
                    cost = trade_amount * level
                    if cost <= state.capital:
                        # Update position
                        total_cost = state.capital * trade_amount * level / state.capital
                        new_position = state.position + trade_amount
                        new_avg_price = (
                            (state.avg_price * state.position + level * trade_amount) / new_position
                            if new_position > 0 else level
                        )

                        state.capital -= cost
                        state.position = new_position
                        state.avg_price = new_avg_price

                        # Record trade
                        state.trades.append({
                            "timestamp": datetime.now(),
                            "side": "buy",
                            "price": level,
                            "amount": trade_amount,
                            "cost": cost,
                            "iteration": self.iteration,
                        })

                        logger.info(f"  📈 {symbol}: BOUGHT {trade_amount:.6f} @ ${level:,.2f}")

            # Sell condition: price crosses above a grid level
            elif level > current_price:
                # Check if we have position to sell
                if state.position > 0:
                    # Calculate sell amount (portion of position)
                    sell_amount = min(state.position, state.investment_amount() / strategy.num_grids / level)

                    # Execute sell
                    revenue = sell_amount * level

                    # Calculate profit
                    profit = (level - state.avg_price) * sell_amount

                    state.capital += revenue
                    state.position -= sell_amount

                    # Record trade
                    state.trades.append({
                        "timestamp": datetime.now(),
                        "side": "sell",
                        "price": level,
                        "amount": sell_amount,
                        "revenue": revenue,
                        "profit": profit,
                        "iteration": self.iteration,
                    })

                    logger.info(f"  📉 {symbol}: SOLD {sell_amount:.6f} @ ${level:,.2f} (Profit: ${profit:.2f})")

    def update_portfolio(self):
        """Update portfolio values for all symbols."""
        portfolio_summary = {
            "timestamp": datetime.now(),
            "iteration": self.iteration,
            "total_value": 0.0,
            "total_capital": 0.0,
            "total_position_value": 0.0,
            "symbols": {},
        }

        for symbol, state in self.symbol_states.items():
            # Fetch current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]

            # Calculate values
            position_value = state.position * current_price
            total_value = state.capital + position_value
            pnl = state.pnl(current_price)

            portfolio_summary["symbols"][symbol] = {
                "price": current_price,
                "capital": state.capital,
                "position": state.position,
                "position_value": position_value,
                "total_value": total_value,
                "pnl": pnl,
                "trades_count": len(state.trades),
            }

            portfolio_summary["total_value"] += total_value
            portfolio_summary["total_capital"] += state.capital
            portfolio_summary["total_position_value"] += position_value

        # Calculate portfolio P&L
        portfolio_summary["total_pnl"] = portfolio_summary["total_value"] - self.total_capital
        portfolio_summary["total_roi_pct"] = (
            portfolio_summary["total_pnl"] / self.total_capital * 100
            if self.total_capital > 0 else 0
        )

        # Add to history
        self.equity_history.append(portfolio_summary)

        return portfolio_summary

    def print_summary(self, portfolio_summary: Dict[str, Any]):
        """Print portfolio summary."""
        logger.info("\n" + "=" * 80)
        logger.info(f"PORTFOLIO SUMMARY - Iteration {self.iteration}")
        logger.info("=" * 80)

        # Overall portfolio
        total_value = portfolio_summary["total_value"]
        total_pnl = portfolio_summary["total_pnl"]
        total_roi = portfolio_summary["total_roi_pct"]

        logger.info(f"\n💰 Total Portfolio Value: ${total_value:,.2f}")
        logger.info(f"   P&L: ${total_pnl:+,.2f} ({total_roi:+.2f}%)")
        logger.info(f"   Capital: ${portfolio_summary['total_capital']:,.2f}")
        logger.info(f"   Positions: ${portfolio_summary['total_position_value']:,.2f}")

        # Individual symbols
        logger.info(f"\n📊 Individual Symbols:")
        for symbol, data in portfolio_summary["symbols"].items():
            pnl = data["pnl"]
            roi = (pnl / (self.total_capital * self._get_symbol_allocation(symbol))) * 100
            logger.info(f"\n  {symbol}:")
            logger.info(f"    Price: ${data['price']:,.2f}")
            logger.info(f"    Value: ${data['total_value']:,.2f}")
            logger.info(f"    P&L: ${pnl:+,.2f}")
            logger.info(f"    Trades: {data['trades_count']}")

        logger.info("\n" + "=" * 80 + "\n")

    def _get_symbol_allocation(self, symbol: str) -> float:
        """Get capital allocation for a symbol."""
        for config in self.symbols_config:
            if config.symbol == symbol:
                return config.capital_allocation
        return 0.0

    def save_results(self, portfolio_summary: Dict[str, Any]):
        """Save trading results to CSV files."""
        # Create data directory
        data_dir = Path("data/multi_symbol_trading")
        data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save equity curve
        equity_df = pd.DataFrame(self.equity_history)
        equity_file = data_dir / f"portfolio_equity_{timestamp}.csv"
        equity_df.to_csv(equity_file, index=False)
        logger.info(f"💾 Saved equity curve to {equity_file}")

        # Save trades for each symbol
        for symbol, state in self.symbol_states.items():
            if state.trades:
                trades_df = pd.DataFrame(state.trades)
                trades_file = data_dir / f"{symbol.replace('/', '_')}_trades_{timestamp}.csv"
                trades_df.to_csv(trades_file, index=False)
                logger.info(f"💾 Saved {len(state.trades)} trades for {symbol} to {trades_file}")

        # Save summary
        summary_file = data_dir / f"portfolio_summary_{timestamp}.json"
        import json
        with open(summary_file, "w") as f:
            # Convert datetime objects to strings
            summary_copy = portfolio_summary.copy()
            summary_copy["timestamp"] = summary_copy["timestamp"].isoformat()
            json.dump(summary_copy, f, indent=2, default=str)
        logger.info(f"💾 Saved portfolio summary to {summary_file}")

    def run(
        self,
        interval_seconds: int = 3600,
        iterations: Optional[int] = None,
        auto_save: bool = True,
        save_interval: int = 10,
    ):
        """
        Run multi-symbol grid trading.

        Args:
            interval_seconds: Seconds between market checks
            iterations: Number of iterations (None = infinite)
            auto_save: Auto-save results every save_interval iterations
            save_interval: Iterations between auto-saves
        """
        logger.info(f"Starting Multi-Symbol Grid Trading")
        logger.info(f"  Interval: {interval_seconds}s")
        logger.info(f"  Max Iterations: {iterations or 'Infinite'}")
        logger.info(f"  Auto-save: {auto_save} (every {save_interval} iterations)")
        logger.info("")

        try:
            while iterations is None or self.iteration < iterations:
                self.iteration += 1

                logger.info("=" * 80)
                logger.info(f"Iteration {self.iteration}")
                logger.info("=" * 80)

                # Process each symbol
                for symbol in self.symbol_states.keys():
                    logger.info(f"\n📊 Processing {symbol}...")

                    # Fetch market data
                    market_data = self.fetch_market_data(symbol)
                    current_price = market_data["price"]
                    price_change = market_data["change"]

                    logger.info(f"  Price: ${current_price:,.2f} ({price_change:+.2f}%)")

                    # Check grid range
                    state = self.symbol_states[symbol]
                    strategy = state.strategy

                    if current_price < strategy.lower_price:
                        logger.warning(f"  ⚠️  Price BELOW grid range (${strategy.lower_price:,.2f})")
                    elif current_price > strategy.upper_price:
                        logger.warning(f"  ⚠️  Price ABOVE grid range (${strategy.upper_price:,.2f})")
                    else:
                        logger.info(f"  ✅ Price within grid range")

                    # Execute grid trades
                    self.execute_grid_trades(symbol, current_price)

                # Update portfolio
                portfolio_summary = self.update_portfolio()
                self.print_summary(portfolio_summary)

                # Auto-save
                if auto_save and self.iteration % save_interval == 0:
                    logger.info("💾 Auto-saving results...")
                    self.save_results(portfolio_summary)

                # Wait for next iteration
                if iterations is None or self.iteration < iterations:
                    logger.info(f"⏳ Waiting {interval_seconds}s until next iteration...")
                    time.sleep(interval_seconds)

            # Final save
            if auto_save:
                logger.info("\n💾 Final save...")
                self.save_results(portfolio_summary)

            logger.success("\n✅ Multi-Symbol Grid Trading completed!")

        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Trading interrupted by user")
            logger.info("💾 Saving final results...")
            self.save_results(portfolio_summary)
            logger.success("✅ Results saved. Goodbye!")


def main():
    """Main entry point for multi-symbol grid trading."""
    print("\n" + "=" * 80)
    print(" " * 20 + "MULTI-SYMBOL GRID TRADING SYSTEM")
    print("=" * 80)
    print("\nConfiguration:")
    print("  • Multiple cryptocurrency pairs")
    print("  • Individual grid strategies per symbol")
    print("  • Unified portfolio management")
    print("  • Real-time market data from Binance")
    print("\n" + "=" * 80 + "\n")

    # Define symbols to trade
    symbols = [
        SymbolConfig(
            symbol="BTC/USDT",
            capital_allocation=0.40,  # 40% of portfolio
            num_grids=10,
            grid_range_pct=0.15,
        ),
        SymbolConfig(
            symbol="ETH/USDT",
            capital_allocation=0.30,  # 30% of portfolio
            num_grids=10,
            grid_range_pct=0.15,
        ),
        SymbolConfig(
            symbol="SOL/USDT",
            capital_allocation=0.20,  # 20% of portfolio
            num_grids=10,
            grid_range_pct=0.20,  # Wider range for higher volatility
        ),
        SymbolConfig(
            symbol="BNB/USDT",
            capital_allocation=0.10,  # 10% of portfolio
            num_grids=10,
            grid_range_pct=0.15,
        ),
    ]

    # Create trader
    trader = MultiSymbolGridTrader(
        symbols=symbols,
        total_capital=10000.0,
    )

    # Run trading
    # For demo: 3 iterations with 60s intervals
    # For production: Use interval_seconds=3600 (1 hour) and iterations=None (infinite)
    trader.run(
        interval_seconds=60,  # 60 seconds for demo
        iterations=3,         # 3 iterations for demo
        auto_save=True,
        save_interval=10,
    )


if __name__ == "__main__":
    main()
