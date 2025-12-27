"""
Advanced Risk Management System for Trading

Implements comprehensive risk controls including:
- Stop-loss protection
- Volatility-based position sizing
- Daily loss limits
- Maximum drawdown limits
- Portfolio-level risk management
"""

import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
class RiskLimits:
    """Risk management limits."""
    max_position_size_pct: float = 0.20  # Max 20% of portfolio in one position
    stop_loss_pct: float = 0.05  # 5% stop loss
    daily_loss_limit_pct: float = 0.03  # 3% daily loss limit
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    volatility_scaling: bool = True  # Adjust position size based on volatility


@dataclass
class RiskState:
    """Current risk state."""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    peak_portfolio_value: float = 0.0
    current_drawdown_pct: float = 0.0
    last_risk_check: Optional[datetime] = None
    stop_loss_triggered: bool = False
    daily_limit_triggered: bool = False
    drawdown_limit_triggered: bool = False


class RiskManagedGridTrader:
    """
    Grid trading paper trader with advanced risk management.

    Implements multiple layers of risk control to protect capital and
    limit losses during adverse market conditions.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        initial_capital: float = 10000.0,
        num_grids: int = 10,
        grid_range_pct: float = 0.15,
        grid_mode: GridTradingMode = GridTradingMode.GEOMETRIC,
        risk_limits: Optional[RiskLimits] = None,
        exchange_name: str = "binance",
    ):
        """
        Initialize risk-managed grid trader.

        Args:
            symbol: Trading pair symbol
            initial_capital: Starting capital
            num_grids: Number of grid levels
            grid_range_pct: Grid range (±percentage)
            grid_mode: Grid spacing mode
            risk_limits: Risk management limits
            exchange_name: Exchange to use
        """
        self.symbol = symbol
        self.exchange = create_exchange(exchange_name)
        self.initial_capital = initial_capital
        self.num_grids = num_grids
        self.grid_range_pct = grid_range_pct
        self.grid_mode = grid_mode

        # Risk management
        self.risk_limits = risk_limits or RiskLimits()
        self.risk_state = RiskState()
        self.risk_state.peak_portfolio_value = initial_capital

        # Portfolio state
        self.capital = initial_capital
        self.position = 0.0
        self.avg_price = 0.0

        # Strategy state
        self.strategy: Optional[GridTradingStrategy] = None
        self.current_price: Optional[float] = None

        # Trading history
        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.risk_events: List[Dict] = []

        # Tracking
        self.iteration = 0
        self.trading_day_start = datetime.now().date()

        # Initialize
        self._initialize_strategy()

    def _initialize_strategy(self):
        """Initialize grid trading strategy."""
        ticker = self.exchange.fetch_ticker(self.symbol)
        self.current_price = ticker["last"]

        upper_price = self.current_price * (1 + self.grid_range_pct)
        lower_price = self.current_price * (1 - self.grid_range_pct)

        self.strategy = GridTradingStrategy(
            symbol=self.symbol,
            upper_price=upper_price,
            lower_price=lower_price,
            num_grids=self.num_grids,
            grid_mode=self.grid_mode,
            investment_amount=self.initial_capital,
        )

        logger.info(f"✅ Grid Trading Strategy initialized")
        logger.info(f"   Grid Range: ${lower_price:,.2f} - ${upper_price:,.2f}")
        logger.info(f"   Risk Limits:")
        logger.info(f"     Stop Loss: {self.risk_limits.stop_loss_pct*100:.1f}%")
        logger.info(f"     Daily Loss Limit: {self.risk_limits.daily_loss_limit_pct*100:.1f}%")
        logger.info(f"     Max Drawdown: {self.risk_limits.max_drawdown_pct*100:.1f}%")

    def check_risk_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check all risk limits.

        Returns:
            (can_trade, violation_reason) tuple
        """
        # Check if it's a new trading day
        current_date = datetime.now().date()
        if current_date != self.trading_day_start:
            # Reset daily limits
            self.risk_state.daily_pnl = 0.0
            self.risk_state.daily_trades = 0
            self.trading_day_start = current_date
            logger.info("📅 New trading day - daily limits reset")

        # Calculate current portfolio value
        portfolio_value = self.capital + (self.position * self.current_price)

        # Update peak value
        if portfolio_value > self.risk_state.peak_portfolio_value:
            self.risk_state.peak_portfolio_value = portfolio_value

        # Calculate drawdown
        drawdown = (self.risk_state.peak_portfolio_value - portfolio_value)
        drawdown_pct = drawdown / self.risk_state.peak_portfolio_value
        self.risk_state.current_drawdown_pct = drawdown_pct

        # Check 1: Stop loss on position
        if self.position > 0:
            position_pnl_pct = (self.current_price - self.avg_price) / self.avg_price
            if position_pnl_pct < -self.risk_limits.stop_loss_pct:
                self.risk_state.stop_loss_triggered = True
                return False, f"Stop loss triggered: Position down {position_pnl_pct*100:.1f}%"

        # Check 2: Daily loss limit
        daily_pnl_pct = self.risk_state.daily_pnl / self.initial_capital
        if daily_pnl_pct < -self.risk_limits.daily_loss_limit_pct:
            self.risk_state.daily_limit_triggered = True
            return False, f"Daily loss limit reached: {daily_pnl_pct*100:.1f}%"

        # Check 3: Maximum drawdown
        if drawdown_pct > self.risk_limits.max_drawdown_pct:
            self.risk_state.drawdown_limit_triggered = True
            return False, f"Max drawdown exceeded: {drawdown_pct*100:.1f}%"

        return True, None

    def calculate_position_size(self, base_size: float) -> float:
        """
        Calculate position size based on volatility.

        Reduces position size in high volatility conditions.
        """
        if not self.risk_limits.volatility_scaling:
            return base_size

        # Fetch recent price data for volatility calculation
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, "1h", limit=24)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

            # Calculate daily volatility (standard deviation of returns)
            df["returns"] = df["close"].pct_change()
            volatility = df["returns"].std()

            # Scale position size inversely with volatility
            # Base adjustment: 1% volatility = full position size
            # 2% volatility = 50% position size, etc.
            vol_multiplier = min(1.0, 0.01 / (volatility + 1e-6))
            vol_multiplier = max(0.25, vol_multiplier)  # Minimum 25% of base size

            adjusted_size = base_size * vol_multiplier

            logger.info(f"  Volatility: {volatility*100:.2f}% → Position size: {adjusted_size/base_size*100:.0f}% of base")

            return adjusted_size

        except Exception as e:
            logger.warning(f"  Could not calculate volatility: {e}, using base size")
            return base_size

    def execute_stop_loss(self):
        """Execute stop loss by closing position."""
        if self.position > 0:
            logger.warning("\n" + "=" * 80)
            logger.warning("🚨 EXECUTING STOP LOSS")
            logger.warning("=" * 80)

            # Close entire position
            revenue = self.position * self.current_price
            loss = (self.avg_price - self.current_price) * self.position

            self.capital += revenue
            self.position = 0.0
            self.avg_price = 0.0

            # Record trade
            self.trades.append({
                "timestamp": datetime.now(),
                "side": "sell",
                "price": self.current_price,
                "amount": revenue / self.current_price,
                "revenue": revenue,
                "profit": -loss,
                "type": "stop_loss",
                "iteration": self.iteration,
            })

            # Record risk event
            self.risk_events.append({
                "timestamp": datetime.now(),
                "type": "stop_loss_triggered",
                "price": self.current_price,
                "loss": loss,
                "iteration": self.iteration,
            })

            logger.warning(f"Closed position: ${revenue:,.2f}")
            logger.warning(f"Loss: ${loss:,.2f}")
            logger.warning("=" * 80 + "\n")

    def execute_grid_trades(self):
        """Execute grid trades with risk management."""
        if not self.strategy:
            return

        # Check risk limits first
        can_trade, violation = self.check_risk_limits()

        if not can_trade:
            logger.warning(f"⚠️  Trading blocked: {violation}")

            # If stop loss triggered, execute it
            if self.risk_state.stop_loss_triggered:
                self.execute_stop_loss()

            return

        grid_levels = sorted(self.strategy.grid_levels)

        for level in grid_levels:
            # Calculate position size with risk adjustment
            base_trade_amount = (self.initial_capital / self.strategy.num_grids) / level
            trade_amount = self.calculate_position_size(base_trade_amount)

            # Buy orders
            if level < self.current_price:
                cost = trade_amount * level

                # Check max position size
                position_value = (self.position + trade_amount) * self.current_price
                portfolio_value = self.capital + position_value
                position_pct = position_value / portfolio_value if portfolio_value > 0 else 0

                if position_pct > self.risk_limits.max_position_size_pct:
                    logger.info(f"  ⚠️  Position size limit reached ({position_pct*100:.1f}%), skipping buy")
                    continue

                if cost <= self.capital:
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
                        "iteration": self.iteration,
                    })

                    logger.info(f"  📈 BOUGHT {trade_amount:.6f} @ ${level:,.2f}")

            # Sell orders
            elif level > self.current_price:
                if self.position > 0:
                    sell_amount = min(
                        trade_amount,
                        self.position
                    )

                    if sell_amount > 0:
                        revenue = sell_amount * level
                        profit = (level - self.avg_price) * sell_amount

                        self.capital += revenue
                        self.position -= sell_amount

                        # Update daily P&L
                        self.risk_state.daily_pnl += profit
                        self.risk_state.daily_trades += 1

                        self.trades.append({
                            "timestamp": datetime.now(),
                            "side": "sell",
                            "price": level,
                            "amount": sell_amount,
                            "revenue": revenue,
                            "profit": profit,
                            "iteration": self.iteration,
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
            "daily_pnl": self.risk_state.daily_pnl,
            "drawdown_pct": self.risk_state.current_drawdown_pct * 100,
        }

    def print_summary(self):
        """Print portfolio summary with risk metrics."""
        portfolio = self.update_portfolio_value()

        logger.info("\n" + "=" * 80)
        logger.info(f"RISK-MANAGED PORTFOLIO - Iteration {self.iteration}")
        logger.info("=" * 80)
        logger.info(f"\n💰 Portfolio: ${portfolio['total_value']:,.2f}")
        logger.info(f"   P&L: ${portfolio['pnl']:+,.2f} ({portfolio['roi_pct']:+.2f}%)")
        logger.info(f"   Daily P&L: ${portfolio['daily_pnl']:+,.2f}")
        logger.info(f"   Drawdown: {portfolio['drawdown_pct']:.2f}%")

        logger.info(f"\n🛡️  Risk Status:")
        logger.info(f"   Position Size: {portfolio['position_value']/portfolio['total_value']*100:.1f}% of portfolio")
        logger.info(f"   Daily Trades: {self.risk_state.daily_trades}")

        if self.risk_state.stop_loss_triggered:
            logger.warning(f"   ⚠️  Stop Loss: TRIGGERED")
        if self.risk_state.daily_limit_triggered:
            logger.warning(f"   ⚠️  Daily Limit: REACHED")
        if self.risk_state.drawdown_limit_triggered:
            logger.warning(f"   ⚠️  Max Drawdown: EXCEEDED")

        if not any([self.risk_state.stop_loss_triggered, self.risk_state.daily_limit_triggered,
                    self.risk_state.drawdown_limit_triggered]):
            logger.info(f"   ✅ All risk limits OK")

        logger.info("=" * 80 + "\n")

    def save_results(self):
        """Save trading results with risk events."""
        data_dir = Path("data/risk_managed_trading")
        data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = data_dir / f"{self.symbol.replace('/', '_')}_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"💾 Saved {len(self.trades)} trades to {trades_file}")

        # Save risk events
        if self.risk_events:
            events_df = pd.DataFrame(self.risk_events)
            events_file = data_dir / f"{self.symbol.replace('/', '_')}_risk_events_{timestamp}.csv"
            events_df.to_csv(events_file, index=False)
            logger.info(f"💾 Saved {len(self.risk_events)} risk events to {events_file}")

    def run(
        self,
        iterations: int = 10,
        interval_seconds: int = 60,
        auto_save: bool = True,
    ):
        """Run risk-managed grid trading."""
        logger.info("=" * 80)
        logger.info("RISK-MANAGED GRID TRADING")
        logger.info("=" * 80)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Risk Limits:")
        logger.info(f"  Stop Loss: {self.risk_limits.stop_loss_pct*100:.1f}%")
        logger.info(f"  Daily Loss Limit: {self.risk_limits.daily_loss_limit_pct*100:.1f}%")
        logger.info(f"  Max Drawdown: {self.risk_limits.max_drawdown_pct*100:.1f}%")
        logger.info(f"  Max Position Size: {self.risk_limits.max_position_size_pct*100:.1f}%")
        logger.info(f"  Volatility Scaling: {self.risk_limits.volatility_scaling}")
        logger.info("=" * 80 + "\n")

        try:
            while self.iteration < iterations:
                self.iteration += 1

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Iteration {self.iteration}/{iterations}")
                logger.info("=" * 80)

                # Fetch current price
                ticker = self.exchange.fetch_ticker(self.symbol)
                self.current_price = ticker["last"]
                price_change = ticker.get("change", 0)

                logger.info(f"\n📊 Price: ${self.current_price:,.2f} ({price_change:+.2f}%)")

                # Check if price in grid range
                if self.current_price < self.strategy.lower_price:
                    logger.warning(f"  ⚠️  Price below grid (${self.current_price:,.2f} < ${self.strategy.lower_price:,.2f})")
                elif self.current_price > self.strategy.upper_price:
                    logger.warning(f"  ⚠️  Price above grid (${self.current_price:,.2f} > ${self.strategy.upper_price:,.2f})")
                else:
                    logger.info(f"  ✅ Price within grid range")

                # Execute trades with risk management
                self.execute_grid_trades()

                # Print summary
                self.print_summary()

                # Wait for next iteration
                if self.iteration < iterations:
                    logger.info(f"⏳ Waiting {interval_seconds}s until next iteration...")
                    time.sleep(interval_seconds)

            # Final save
            if auto_save:
                logger.info("\n💾 Saving results...")
                self.save_results()

            portfolio = self.update_portfolio_value()
            logger.success("\n✅ Risk-Managed Trading completed!")
            logger.success(f"Final Portfolio: ${portfolio['total_value']:,.2f}")
            logger.success(f"Total P&L: ${portfolio['pnl']:+,.2f} ({portfolio['roi_pct']:+.2f}%)")

        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Trading interrupted by user")
            if auto_save:
                logger.info("💾 Saving results...")
                self.save_results()
            logger.success("✅ Results saved. Goodbye!")


def main():
    """Main entry point for risk-managed trading."""
    # Define risk limits
    risk_limits = RiskLimits(
        max_position_size_pct=0.20,  # 20% max position
        stop_loss_pct=0.05,  # 5% stop loss
        daily_loss_limit_pct=0.03,  # 3% daily loss limit
        max_drawdown_pct=0.10,  # 10% max drawdown
        volatility_scaling=True,  # Enable volatility-based position sizing
    )

    # Create trader
    trader = RiskManagedGridTrader(
        symbol="BTC/USDT",
        initial_capital=10000.0,
        num_grids=10,
        grid_range_pct=0.15,
        grid_mode=GridTradingMode.GEOMETRIC,
        risk_limits=risk_limits,
    )

    # Run trading
    trader.run(
        iterations=3,  # 3 iterations for demo
        interval_seconds=60,
        auto_save=True,
    )


if __name__ == "__main__":
    main()
