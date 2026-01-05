"""
Risk manager for live trading.

Handles position sizing, stop-loss, and portfolio risk.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def value(self) -> float:
        """Position value."""
        return self.quantity * self.entry_price

    @property
    def current_value(self, current_price: float) -> float:
        """Current position value."""
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.side == "long":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


class RiskManager:
    """Risk management for live trading."""

    def __init__(self, safety_limits):
        """Initialize risk manager.

        Args:
            safety_limits: SafetyLimits instance
        """
        self.safety_limits = safety_limits
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of each day)."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.last_reset_date = today
            from loguru import logger

            logger.info("Daily risk stats reset")

    def calculate_position_size(
        self,
        portfolio_value: float,
        current_price: float,
        signal: str,
    ) -> float:
        """Calculate safe position size.

        Args:
            portfolio_value: Total portfolio value
            current_price: Current asset price
            signal: Trading signal ("buy" or "sell")

        Returns:
            Quantity to trade
        """
        # Calculate max position value
        max_value_by_pct = portfolio_value * self.safety_limits.max_position_pct
        max_value = min(self.safety_limits.max_position_size, max_value_by_pct)

        # Calculate quantity
        quantity = max_value / current_price

        # Round to reasonable precision
        quantity = round(quantity, 6)

        return quantity

    def can_trade(
        self,
        signal: str,
        symbol: str,
        portfolio_value: float,
        current_price: float,
        current_time: Optional[datetime] = None,
    ) -> tuple[bool, str, float]:
        """Check if trade is allowed and calculate quantity.

        Args:
            signal: Trading signal ("buy" or "sell")
            symbol: Trading pair symbol
            portfolio_value: Total portfolio value
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            (can_trade, reason, quantity) tuple
        """
        # Reset daily stats if needed
        self.reset_daily_stats()

        # Get current exposure
        current_exposure = sum(pos.value for pos in self.positions.values())

        # Calculate position size
        quantity = self.calculate_position_size(portfolio_value, current_price, signal)
        trade_value = quantity * current_price

        # Check safety limits
        can_trade, reason = self.safety_limits.can_open_position(
            portfolio_value=portfolio_value,
            current_exposure=current_exposure,
            trade_value=trade_value,
            daily_pnl=self.daily_pnl,
            daily_trade_count=self.daily_trade_count,
            current_time=current_time,
        )

        return can_trade, reason, quantity

    def add_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Add a position to tracking.

        Args:
            symbol: Trading pair symbol
            side: Position side ("long" or "short")
            quantity: Position quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        from loguru import logger

        logger.info(f"Opened {side} position: {quantity:.4f} {symbol} @ ${entry_price:,.2f}")

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return realized P&L.

        Args:
            symbol: Trading pair symbol
            exit_price: Exit price

        Returns:
            Realized P&L
        """
        if symbol not in self.positions:
            from loguru import logger

            logger.warning(f"No position found for {symbol}")
            return 0.0

        position = self.positions.pop(symbol)
        pnl = position.unrealized_pnl(exit_price)

        # Update daily stats
        self.daily_pnl += pnl
        self.daily_trade_count += 1

        from loguru import logger

        if pnl >= 0:
            logger.success(
                f"Closed position: {position.quantity:.4f} {symbol} @ ${exit_price:,.2f} "
                f"P&L: ${pnl:,.2f} (+{position.unrealized_pnl_pct(exit_price):.2f}%)"
            )
        else:
            logger.warning(
                f"Closed position: {position.quantity:.4f} {symbol} @ ${exit_price:,.2f} "
                f"P&L: ${pnl:,.2f} ({position.unrealized_pnl_pct(exit_price):.2f}%)"
            )

        return pnl

    def check_positions(self, current_prices: Dict[str, float]) -> list:
        """Check all positions for risk management.

        Args:
            current_prices: Current prices for all positions

        Returns:
            List of (symbol, should_close, reason) tuples
        """
        actions = []

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            unrealized_pnl_pct = position.unrealized_pnl_pct(current_price)

            should_close, reason = self.safety_limits.should_close_position(
                entry_price=position.entry_price,
                current_price=current_price,
                position_value=position.value,
                unrealized_pnl_pct=unrealized_pnl_pct,
            )

            if should_close:
                actions.append((symbol, True, reason))

        return actions

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary.

        Returns:
            Portfolio summary dict
        """
        return {
            "positions": {
                symbol: {
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time,
                    "value": pos.value,
                }
                for symbol, pos in self.positions.items()
            },
            "daily_pnl": self.daily_pnl,
            "daily_trade_count": self.daily_trade_count,
            "open_positions": len(self.positions),
        }
