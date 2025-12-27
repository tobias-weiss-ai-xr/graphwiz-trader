"""
Multi-Strategy Paper Trading System

Deploys Smart DCA and AMM strategies to paper trading alongside Grid Trading.
Demonstrates portfolio-level strategy management and performance comparison.
"""

import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger
import pandas as pd

from graphwiz_trader.strategies import (
    GridTradingStrategy,
    GridTradingMode,
    SmartDCAStrategy,
    AutomatedMarketMakingStrategy,
)
from graphwiz_trader.trading.exchange import create_exchange

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy."""
    strategy_name: str
    strategy_type: str  # 'grid', 'dca', 'amm'
    symbol: str
    capital_allocation: float  # Percentage of total capital
    params: dict = field(default_factory=dict)


@dataclass
class StrategyState:
    """Runtime state of a strategy."""
    name: str
    strategy: object  # GridTradingStrategy, SmartDCAStrategy, or AMMStrategy
    symbol: str  # Store symbol separately since AMM doesn't have it
    capital: float
    initial_capital: float
    position: float = 0.0
    avg_price: float = 0.0
    trades: List[Dict] = field(default_factory=list)

    @property
    def portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        position_value = self.position * current_price
        return self.capital + position_value

    @property
    def pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.position == 0:
            return 0.0
        return (current_price - self.avg_price) * self.position


class MultiStrategyPaperTrader:
    """
    Paper trading system managing multiple strategies simultaneously.

    Supports Grid Trading, Smart DCA, and AMM strategies with unified
    portfolio tracking and performance comparison.
    """

    def __init__(
        self,
        strategies: List[StrategyAllocation],
        total_capital: float = 10000.0,
        exchange_name: str = "binance",
    ):
        """
        Initialize multi-strategy paper trader.

        Args:
            strategies: List of strategy configurations
            total_capital: Total capital across all strategies
            exchange_name: Exchange to use
        """
        self.exchange = create_exchange(exchange_name)
        self.total_capital = total_capital
        self.strategy_allocations = strategies
        self.strategy_states: Dict[str, StrategyState] = {}
        self.iteration = 0
        self.equity_history: List[Dict] = []

        # Validate capital allocation
        total_allocation = sum(s.capital_allocation for s in strategies)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(
                f"Capital allocation must sum to 1.0, got {total_allocation:.2f}"
            )

        # Initialize strategies
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize all trading strategies."""
        logger.info("=" * 80)
        logger.info("Initializing Multi-Strategy Paper Trading System")
        logger.info("=" * 80)

        for alloc in self.strategy_allocations:
            capital = self.total_capital * alloc.capital_allocation

            logger.info(f"\nInitializing {alloc.strategy_name} ({alloc.strategy_type.upper()})...")
            logger.info(f"  Symbol: {alloc.symbol}")
            logger.info(f"  Allocation: ${capital:,.2f} ({alloc.capital_allocation*100:.1f}%)")

            # Fetch current price
            ticker = self.exchange.fetch_ticker(alloc.symbol)
            current_price = ticker["last"]

            # Initialize strategy based on type
            if alloc.strategy_type == "grid":
                strategy = self._init_grid_strategy(alloc, current_price)
            elif alloc.strategy_type == "dca":
                strategy = self._init_dca_strategy(alloc, current_price, capital)
            elif alloc.strategy_type == "amm":
                strategy = self._init_amm_strategy(alloc, current_price)
            else:
                raise ValueError(f"Unknown strategy type: {alloc.strategy_type}")

            # Create state
            self.strategy_states[alloc.strategy_name] = StrategyState(
                name=alloc.strategy_name,
                strategy=strategy,
                symbol=alloc.symbol,
                capital=capital,
                initial_capital=capital,
            )

            logger.info(f"  ✅ {alloc.strategy_name} initialized")

        logger.info("\n" + "=" * 80)
        logger.success(f"Multi-Strategy Paper Trader initialized")
        logger.success(f"  Strategies: {len(self.strategy_allocations)}")
        logger.success(f"  Total Capital: ${self.total_capital:,.2f}")
        logger.info("=" * 80 + "\n")

    def _init_grid_strategy(self, alloc: StrategyAllocation, current_price: float) -> GridTradingStrategy:
        """Initialize grid trading strategy."""
        params = alloc.params
        grid_range_pct = params.get("grid_range_pct", 0.15)
        num_grids = params.get("num_grids", 10)
        grid_mode = params.get("grid_mode", GridTradingMode.GEOMETRIC)

        upper_price = current_price * (1 + grid_range_pct)
        lower_price = current_price * (1 - grid_range_pct)

        strategy = GridTradingStrategy(
            symbol=alloc.symbol,
            upper_price=upper_price,
            lower_price=lower_price,
            num_grids=num_grids,
            grid_mode=grid_mode,
            investment_amount=self.total_capital * alloc.capital_allocation,
        )

        logger.info(f"  Grid Range: ${lower_price:,.2f} - ${upper_price:,.2f}")
        logger.info(f"  Grids: {num_grids} ({grid_mode.value})")

        return strategy

    def _init_dca_strategy(self, alloc: StrategyAllocation, current_price: float, capital: float) -> SmartDCAStrategy:
        """Initialize Smart DCA strategy."""
        params = alloc.params
        purchase_amount = params.get("purchase_amount", capital / 10)
        volatility_adjustment = params.get("volatility_adjustment", True)
        momentum_boost = params.get("momentum_boost", 0.5)
        price_threshold = params.get("price_threshold", 0.05)

        strategy = SmartDCAStrategy(
            symbol=alloc.symbol,
            total_investment=capital,
            purchase_frequency="daily",
            purchase_amount=purchase_amount,
            volatility_adjustment=volatility_adjustment,
            momentum_boost=momentum_boost,
            price_threshold=price_threshold,
        )

        logger.info(f"  Total Investment: ${capital:,.2f}")
        logger.info(f"  Purchase Amount: ${purchase_amount:,.2f}")
        logger.info(f"  Volatility Adjustment: {volatility_adjustment}")
        logger.info(f"  Momentum Boost: {momentum_boost*100:.0f}%")

        return strategy

    def _init_amm_strategy(self, alloc: StrategyAllocation, current_price: float) -> AutomatedMarketMakingStrategy:
        """Initialize AMM strategy."""
        params = alloc.params
        base_fee_rate = params.get("base_fee_rate", 0.005)  # 0.5%
        price_range_pct = params.get("price_range_pct", 0.20)  # ±20%
        inventory_target_ratio = params.get("inventory_target_ratio", 0.5)

        # Parse symbol to get token pair
        symbol_parts = alloc.symbol.split("/")
        token_a = symbol_parts[0]  # e.g., "SOL"
        token_b = symbol_parts[1]  # e.g., "USDT"

        # Calculate price range
        price_range = (
            current_price * (1 - price_range_pct),
            current_price * (1 + price_range_pct),
        )

        strategy = AutomatedMarketMakingStrategy(
            token_a=token_a,
            token_b=token_b,
            pool_price=current_price,
            price_range=price_range,
            base_fee_rate=base_fee_rate,
            inventory_target_ratio=inventory_target_ratio,
        )

        logger.info(f"  Token Pair: {token_a}/{token_b}")
        logger.info(f"  Pool Price: ${current_price:,.2f}")
        logger.info(f"  Price Range: ${price_range[0]:,.2f} - ${price_range[1]:,.2f}")
        logger.info(f"  Fee Rate: {base_fee_rate*100:.2f}%")

        return strategy

    def execute_grid_strategy(self, state: StrategyState, current_price: float):
        """Execute grid trading strategy."""
        strategy = state.strategy
        grid_levels = sorted(strategy.grid_levels)

        for level in grid_levels:
            # Buy orders
            if level < current_price:
                if state.capital > level * 0.001:
                    trade_amount = (state.initial_capital / strategy.num_grids) / level
                    cost = trade_amount * level

                    if cost <= state.capital:
                        if state.position > 0:
                            state.avg_price = (
                                (state.avg_price * state.position + level * trade_amount)
                                / (state.position + trade_amount)
                            )
                        else:
                            state.avg_price = level

                        state.position += trade_amount
                        state.capital -= cost

                        state.trades.append({
                            "timestamp": datetime.now(),
                            "side": "buy",
                            "price": level,
                            "amount": trade_amount,
                            "cost": cost,
                            "iteration": self.iteration,
                        })

            # Sell orders
            elif level > current_price:
                if state.position > 0:
                    sell_amount = min(
                        state.position,
                        (state.initial_capital / strategy.num_grids) / level
                    )

                    if sell_amount > 0:
                        revenue = sell_amount * level
                        profit = (level - state.avg_price) * sell_amount

                        state.capital += revenue
                        state.position -= sell_amount

                        state.trades.append({
                            "timestamp": datetime.now(),
                            "side": "sell",
                            "price": level,
                            "amount": sell_amount,
                            "revenue": revenue,
                            "profit": profit,
                            "iteration": self.iteration,
                        })

    def execute_dca_strategy(self, state: StrategyState, current_price: float, historical_data: pd.DataFrame):
        """Execute Smart DCA strategy."""
        strategy = state.strategy

        # Simple DCA: Purchase fixed amount every iteration
        purchase_amount = strategy.base_purchase_amount

        # Check if we should buy (every iteration for demo)
        if purchase_amount <= state.capital:
            amount = purchase_amount / current_price

            if state.position > 0:
                state.avg_price = (
                    (state.avg_price * state.position + current_price * amount)
                    / (state.position + amount)
                )
            else:
                state.avg_price = current_price

            state.position += amount
            state.capital -= purchase_amount

            state.trades.append({
                "timestamp": datetime.now(),
                "side": "buy",
                "price": current_price,
                "amount": amount,
                "cost": purchase_amount,
                "iteration": self.iteration,
            })

            logger.info(f"  📈 DCA BUY: {amount:.6f} @ ${current_price:,.2f}")

    def execute_amm_strategy(self, state: StrategyState, current_price: float):
        """Execute AMM strategy (simplified simulation)."""
        # For AMM, we just track the current state
        # Real AMM would respond to external trades
        # Here we just log that it's operational
        logger.info(f"  💰 AMM operational at ${current_price:,.2f}")

    def update_portfolio(self):
        """Update portfolio values for all strategies."""
        portfolio_summary = {
            "timestamp": datetime.now(),
            "iteration": self.iteration,
            "total_value": 0.0,
            "strategies": {},
        }

        for name, state in self.strategy_states.items():
            # Fetch current price
            ticker = self.exchange.fetch_ticker(state.symbol)
            current_price = ticker["last"]

            # Calculate values
            position_value = state.position * current_price
            total_value = state.capital + position_value
            pnl = total_value - state.initial_capital
            roi = (pnl / state.initial_capital * 100) if state.initial_capital > 0 else 0

            portfolio_summary["strategies"][name] = {
                "type": self._get_strategy_type(name),
                "symbol": state.symbol,
                "price": current_price,
                "capital": state.capital,
                "position": state.position,
                "position_value": position_value,
                "total_value": total_value,
                "pnl": pnl,
                "roi_pct": roi,
                "trades_count": len(state.trades),
            }

            portfolio_summary["total_value"] += total_value

        # Calculate portfolio P&L
        portfolio_summary["total_pnl"] = portfolio_summary["total_value"] - self.total_capital
        portfolio_summary["total_roi_pct"] = (
            portfolio_summary["total_pnl"] / self.total_capital * 100
            if self.total_capital > 0 else 0
        )

        # Add to history
        self.equity_history.append(portfolio_summary)

        return portfolio_summary

    def _get_strategy_type(self, name: str) -> str:
        """Get strategy type from name."""
        for alloc in self.strategy_allocations:
            if alloc.strategy_name == name:
                return alloc.strategy_type
        return "unknown"

    def print_summary(self, portfolio_summary: Dict):
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

        # Individual strategies
        logger.info(f"\n📊 Strategy Performance:")
        for name, data in portfolio_summary["strategies"].items():
            logger.info(f"\n  {name} ({data['type'].upper()}):")
            logger.info(f"    Symbol: {data['symbol']}")
            logger.info(f"    Value: ${data['total_value']:,.2f}")
            logger.info(f"    P&L: ${data['pnl']:+,.2f} ({data['roi_pct']:+.2f}%)")
            logger.info(f"    Trades: {data['trades_count']}")

        logger.info("\n" + "=" * 80 + "\n")

    def save_results(self, portfolio_summary: Dict):
        """Save trading results to files."""
        data_dir = Path("data/multi_strategy_trading")
        data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save equity curve
        equity_df = pd.DataFrame(self.equity_history)
        equity_file = data_dir / f"portfolio_equity_{timestamp}.csv"
        equity_df.to_csv(equity_file, index=False)
        logger.info(f"💾 Saved equity curve to {equity_file}")

        # Save trades for each strategy
        for name, state in self.strategy_states.items():
            if state.trades:
                trades_df = pd.DataFrame(state.trades)
                trades_file = data_dir / f"{name}_trades_{timestamp}.csv"
                trades_df.to_csv(trades_file, index=False)
                logger.info(f"💾 Saved {len(state.trades)} trades for {name} to {trades_file}")

    def run(
        self,
        iterations: int = 10,
        interval_seconds: int = 60,
        auto_save: bool = True,
    ):
        """Run multi-strategy paper trading."""
        logger.info(f"Starting Multi-Strategy Paper Trading")
        logger.info(f"  Iterations: {iterations}")
        logger.info(f"  Interval: {interval_seconds}s")
        logger.info("")

        try:
            while self.iteration < iterations:
                self.iteration += 1

                logger.info("=" * 80)
                logger.info(f"Iteration {self.iteration}/{iterations}")
                logger.info("=" * 80)

                # Process each strategy
                for name, state in self.strategy_states.items():
                    logger.info(f"\n📊 Processing {name}...")

                    # Fetch current price and historical data
                    ticker = self.exchange.fetch_ticker(state.symbol)
                    current_price = ticker["last"]

                    # Fetch historical data for DCA strategy
                    ohlcv = self.exchange.fetch_ohlcv(state.symbol, "1h", limit=100)
                    historical_data = pd.DataFrame(
                        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )

                    logger.info(f"  Price: ${current_price:,.2f}")

                    # Execute strategy based on type
                    strategy_type = self._get_strategy_type(name)

                    if strategy_type == "grid":
                        self.execute_grid_strategy(state, current_price)
                    elif strategy_type == "dca":
                        self.execute_dca_strategy(state, current_price, historical_data)
                    elif strategy_type == "amm":
                        self.execute_amm_strategy(state, current_price)

                # Update portfolio
                portfolio_summary = self.update_portfolio()
                self.print_summary(portfolio_summary)

                # Auto-save
                if auto_save and self.iteration % 5 == 0:
                    logger.info("💾 Auto-saving results...")
                    self.save_results(portfolio_summary)

                # Wait for next iteration
                if self.iteration < iterations:
                    logger.info(f"⏳ Waiting {interval_seconds}s until next iteration...")
                    time.sleep(interval_seconds)

            # Final save
            if auto_save:
                logger.info("\n💾 Final save...")
                self.save_results(portfolio_summary)

            logger.success("\n✅ Multi-Strategy Paper Trading completed!")

        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Trading interrupted by user")
            if auto_save:
                logger.info("💾 Saving final results...")
                self.save_results(portfolio_summary)
            logger.success("✅ Results saved. Goodbye!")


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print(" " * 15 + "MULTI-STRATEGY PAPER TRADING SYSTEM")
    print("=" * 80)
    print("\nStrategies:")
    print("  • Grid Trading (BTC)")
    print("  • Smart DCA (ETH)")
    print("  • AMM (SOL)")
    print("\n" + "=" * 80 + "\n")

    # Define strategies
    strategies = [
        StrategyAllocation(
            strategy_name="Grid BTC",
            strategy_type="grid",
            symbol="BTC/USDT",
            capital_allocation=0.50,
            params={
                "num_grids": 10,
                "grid_range_pct": 0.15,
                "grid_mode": GridTradingMode.GEOMETRIC,
            },
        ),
        StrategyAllocation(
            strategy_name="Smart DCA ETH",
            strategy_type="dca",
            symbol="ETH/USDT",
            capital_allocation=0.30,
            params={
                "purchase_amount": 300,  # $300 per purchase
                "volatility_adjustment": True,
                "momentum_boost": 0.5,
            },
        ),
        StrategyAllocation(
            strategy_name="AMM SOL",
            strategy_type="amm",
            symbol="SOL/USDT",
            capital_allocation=0.20,
            params={
                "base_fee_rate": 0.005,  # 0.5%
                "price_range_pct": 0.20,  # ±20%
                "inventory_target_ratio": 0.5,
            },
        ),
    ]

    # Create trader
    trader = MultiStrategyPaperTrader(
        strategies=strategies,
        total_capital=10000.0,
    )

    # Run trading
    trader.run(
        iterations=3,  # 3 iterations for demo
        interval_seconds=60,
        auto_save=True,
    )


if __name__ == "__main__":
    main()
