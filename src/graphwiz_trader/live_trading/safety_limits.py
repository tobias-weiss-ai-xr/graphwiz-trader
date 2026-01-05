"""
Safety limits for live trading.

Implements risk management limits and checks.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta


@dataclass
class SafetyLimits:
    """Safety limits for live trading."""

    # Position limits
    max_position_size: float = 1000.0  # Maximum $ value per position
    max_position_pct: float = 0.02  # Maximum 2% of portfolio per position
    max_total_exposure: float = 0.10  # Maximum 10% of portfolio in trades

    # Daily limits
    max_daily_loss: float = 500.0  # Stop trading if daily loss exceeds $500
    max_daily_loss_pct: float = 0.05  # Stop if daily loss exceeds 5% of portfolio
    max_daily_trades: int = 10  # Maximum trades per day

    # Trade limits
    min_trade_size: float = 10.0  # Minimum $10 per trade
    max_slippage_pct: float = 0.01  # Maximum 1% slippage tolerance

    # Safety switches
    emergency_shutdown: bool = False  # Immediate stop if True
    require_confirmation: bool = True  # Require manual confirmation for first trade

    # Time-based limits
    trading_hours_only: bool = False  # Only trade during specific hours
    trading_start_hour: int = 9  # 9 AM
    trading_end_hour: int = 17  # 5 PM

    # Volatility limits
    max_volatility_pct: float = 0.10  # Don't trade if volatility > 10%
    volatility_window: int = 20  # 20-period volatility window

    def can_open_position(
        self,
        portfolio_value: float,
        current_exposure: float,
        trade_value: float,
        daily_pnl: float,
        daily_trade_count: int,
        current_time: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """Check if a new position can be opened.

        Args:
            portfolio_value: Total portfolio value
            current_exposure: Current total exposure in $
            trade_value: Value of proposed trade in $
            daily_pnl: Daily P&L in $
            daily_trade_count: Number of trades today
            current_time: Current timestamp

        Returns:
            (can_trade, reason) tuple
        """
        # Emergency shutdown
        if self.emergency_shutdown:
            return False, "Emergency shutdown activated"

        # Check position size
        if trade_value > self.max_position_size:
            return (
                False,
                f"Trade size ${trade_value:,.2f} exceeds max ${self.max_position_size:,.2f}",
            )

        if trade_value > portfolio_value * self.max_position_pct:
            return (
                False,
                f"Trade size {trade_value/portfolio_value*100:.1f}% exceeds max {self.max_position_pct*100:.1f}%",
            )

        # Check total exposure
        new_exposure = current_exposure + trade_value
        if new_exposure > portfolio_value * self.max_total_exposure:
            return (
                False,
                f"Total exposure would be {new_exposure/portfolio_value*100:.1f}%, max {self.max_total_exposure*100:.1f}%",
            )

        # Check daily loss limit
        if daily_pnl < -self.max_daily_loss:
            return (
                False,
                f"Daily loss ${abs(daily_pnl):,.2f} exceeds max ${self.max_daily_loss:,.2f}",
            )

        if daily_pnl < -portfolio_value * self.max_daily_loss_pct:
            return (
                False,
                f"Daily loss {abs(daily_pnl/portfolio_value)*100:.1f}% exceeds max {self.max_daily_loss_pct*100:.1f}%",
            )

        # Check daily trade count
        if daily_trade_count >= self.max_daily_trades:
            return (
                False,
                f"Daily trade count {daily_trade_count} exceeds max {self.max_daily_trades}",
            )

        # Check minimum trade size
        if trade_value < self.min_trade_size:
            return (
                False,
                f"Trade size ${trade_value:,.2f} below minimum ${self.min_trade_size:,.2f}",
            )

        # Check trading hours
        if self.trading_hours_only and current_time:
            if not (self.trading_start_hour <= current_time.hour < self.trading_end_hour):
                return (
                    False,
                    f"Outside trading hours ({self.trading_start_hour}:00-{self.trading_end_hour}:00)",
                )

        return True, "All checks passed"

    def should_close_position(
        self,
        entry_price: float,
        current_price: float,
        position_value: float,
        unrealized_pnl_pct: float,
    ) -> tuple[bool, str]:
        """Check if a position should be closed (risk management).

        Args:
            entry_price: Entry price
            current_price: Current price
            position_value: Position value in $
            unrealized_pnl_pct: Unrealized P&L percentage

        Returns:
            (should_close, reason) tuple
        """
        # Emergency shutdown - close all positions
        if self.emergency_shutdown:
            return True, "Emergency shutdown - closing all positions"

        # Stop loss at -10%
        if unrealized_pnl_pct < -10:
            return True, f"Stop loss triggered: {unrealized_pnl_pct:.1f}%"

        # Take profit at +15%
        if unrealized_pnl_pct > 15:
            return True, f"Take profit triggered: {unrealized_pnl_pct:.1f}%"

        return False, "Hold position"

    def trigger_emergency_shutdown(self, reason: str = "Manual"):
        """Trigger emergency shutdown.

        Args:
            reason: Reason for shutdown
        """
        self.emergency_shutdown = True
        from loguru import logger

        logger.error(f"🚨 EMERGENCY SHUTDOWN TRIGGERED: {reason}")

    def reset_emergency_shutdown(self):
        """Reset emergency shutdown flag."""
        self.emergency_shutdown = False
        from loguru import logger

        logger.info("✅ Emergency shutdown reset")
