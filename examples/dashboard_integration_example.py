"""
Dashboard Integration Example

Shows how to integrate the real-time performance dashboard with trading strategies.
Run this to launch a multi-symbol grid trading system with live dashboard monitoring.
"""

import sys
import threading
import time
from pathlib import Path

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trading_dashboard import DashboardIntegrator, StrategyState
from multi_symbol_grid_trading import MultiSymbolGridTrader, SymbolConfig
from loguru import logger


class DashboardEnabledTrader(MultiSymbolGridTrader):
    """Multi-symbol trader with real-time dashboard updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dashboard_integrator = DashboardIntegrator()

    def start_dashboard(self, host="0.0.0.0", port=5000):
        """Start the dashboard in a background thread."""
        self.dashboard_integrator.start(host=host, port=port)

    def update_dashboard(self):
        """Update dashboard with current trading state."""
        strategies = {}
        for symbol, state in self.symbol_states.items():
            # Fetch current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]

            # Create strategy state for dashboard
            strategy_state = StrategyState(
                symbol=symbol,
                strategy_type="Grid Trading",
                current_price=current_price,
                portfolio_value=state.portfolio_value(current_price),
                capital=state.capital,
                position=state.position,
                position_value=state.position * current_price,
                pnl=state.pnl(current_price),
                roi_pct=(state.pnl(current_price) / state.initial_capital * 100) if state.initial_capital > 0 else 0,
                grid_upper=state.strategy.upper_price,
                grid_lower=state.strategy.lower_price,
                grid_levels=list(state.strategy.grid_levels),
                trades_count=len(state.trades),
                last_update=time.time(),
            )
            strategies[symbol] = strategy_state

        # Update dashboard
        self.dashboard_integrator.update(
            strategies=strategies,
            equity_history=self.equity_history,
            trades=[trade for state in self.symbol_states.values() for trade in state.trades],
        )

    def run_with_dashboard(
        self,
        iterations: int = 10,
        interval_seconds: int = 60,
        dashboard_host="0.0.0.0",
        dashboard_port=5000,
    ):
        """Run trading with real-time dashboard updates."""
        # Start dashboard
        self.start_dashboard(host=dashboard_host, port=dashboard_port)
        time.sleep(2)  # Let dashboard start

        logger.info("Starting trading with dashboard integration...")

        try:
            while iterations is None or self.iteration < iterations:
                self.iteration += 1

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Iteration {self.iteration}")
                logger.info("=" * 80)

                # Process each symbol
                for symbol in self.symbol_states.keys():
                    logger.info(f"\n📊 Processing {symbol}...")

                    # Fetch market data
                    market_data = self.fetch_market_data(symbol)
                    current_price = market_data["price"]

                    logger.info(f"  Price: ${current_price:,.2f}")

                    # Execute grid trades
                    self.execute_grid_trades(symbol, current_price)

                # Update portfolio
                portfolio_summary = self.update_portfolio()

                # Update dashboard
                self.update_dashboard()

                # Print summary
                self.print_summary(portfolio_summary)

                # Wait for next iteration
                if iterations is None or self.iteration < iterations:
                    logger.info(f"⏳ Waiting {interval_seconds}s until next iteration...")
                    time.sleep(interval_seconds)

            logger.success("\n✅ Trading completed!")

        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Trading interrupted by user")
            logger.success("✅ Goodbye!")


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print(" " * 15 + "MULTI-SYMBOL GRID TRADING WITH DASHBOARD")
    print("=" * 80)
    print("\nConfiguration:")
    print("  • Multiple cryptocurrency pairs")
    print("  • Real-time web dashboard at http://localhost:5000")
    print("  • Live performance monitoring")
    print("  • WebSocket updates")
    print("\n" + "=" * 80 + "\n")

    # Define symbols to trade
    symbols = [
        SymbolConfig(
            symbol="BTC/USDT",
            capital_allocation=0.40,
            num_grids=10,
            grid_range_pct=0.15,
        ),
        SymbolConfig(
            symbol="ETH/USDT",
            capital_allocation=0.30,
            num_grids=10,
            grid_range_pct=0.15,
        ),
        SymbolConfig(
            symbol="SOL/USDT",
            capital_allocation=0.20,
            num_grids=10,
            grid_range_pct=0.20,
        ),
        SymbolConfig(
            symbol="BNB/USDT",
            capital_allocation=0.10,
            num_grids=10,
            grid_range_pct=0.15,
        ),
    ]

    # Create trader
    trader = DashboardEnabledTrader(
        symbols=symbols,
        total_capital=10000.0,
    )

    # Run with dashboard
    trader.run_with_dashboard(
        iterations=3,  # 3 iterations for demo
        interval_seconds=60,
        dashboard_host="0.0.0.0",
        dashboard_port=5000,
    )


if __name__ == "__main__":
    main()
