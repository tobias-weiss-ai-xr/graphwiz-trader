"""Portfolio management system for tracking positions and P&L."""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class Position:
    """Represents a trading position in a specific asset."""

    def __init__(
        self,
        symbol: str,
        base_currency: str,
        quote_currency: str
    ):
        """Initialize a position.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            base_currency: Base currency (e.g., "BTC")
            quote_currency: Quote currency (e.g., "USDT")
        """
        self.symbol = symbol
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.amount = Decimal("0")  # Position size (can be negative for short)
        self.avg_entry_price = Decimal("0")
        self.avg_exit_price = Decimal("0")
        self.total_bought = Decimal("0")
        self.total_sold = Decimal("0")
        self.realized_pnl = Decimal("0")
        self.unrealized_pnl = Decimal("0")
        self.total_fees = Decimal("0")
        self.first_trade_time: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        self.trade_count = 0

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.amount > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.amount < 0

    @property
    def is_open(self) -> bool:
        """Check if position has any amount."""
        return self.amount != 0

    @property
    def break_even_price(self) -> Decimal:
        """Calculate break-even price including fees."""
        if self.amount == 0:
            return Decimal("0")

        total_cost = self.total_bought - self.total_sold
        if total_cost == 0:
            return self.avg_entry_price

        return total_cost / abs(self.amount)

    def update_position(
        self,
        side: str,
        amount: Decimal,
        price: Decimal,
        fee: Decimal = Decimal("0")
    ) -> Tuple[Decimal, Decimal]:
        """Update position with a trade.

        Args:
            side: "buy" or "sell"
            amount: Amount traded
            price: Trade price
            fee: Fee paid

        Returns:
            Tuple of (realized_pnl, unrealized_pnl)
        """
        timestamp = datetime.now(timezone.utc)

        if self.first_trade_time is None:
            self.first_trade_time = timestamp
        self.last_trade_time = timestamp
        self.trade_count += 1

        if side.lower() == "buy":
            # Add to position or reduce short
            if self.amount < 0:
                # Reducing short position
                closing_amount = min(amount, abs(self.amount))
                if closing_amount > 0:
                    # Realize P&L on closed portion
                    pnl = (self.avg_entry_price - price) * closing_amount
                    self.realized_pnl += pnl
                    self.amount += closing_amount
                    amount -= closing_amount

            if amount > 0:
                # Opening or adding to long
                total_value = self.total_bought + (amount * price)
                self.total_bought = total_value
                self.amount += amount

                # Recalculate average entry price
                if self.amount > 0:
                    self.avg_entry_price = self.total_bought / self.amount
                else:
                    self.avg_entry_price = Decimal("0")

        else:  # sell
            # Subtract from position or reduce long
            if self.amount > 0:
                # Reducing long position
                closing_amount = min(amount, self.amount)
                if closing_amount > 0:
                    # Realize P&L on closed portion
                    pnl = (price - self.avg_entry_price) * closing_amount
                    self.realized_pnl += pnl
                    self.amount -= closing_amount
                    self.avg_exit_price = price
                    amount -= closing_amount

            if amount > 0:
                # Opening or adding to short
                total_value = self.total_sold + (amount * price)
                self.total_sold = total_value
                self.amount -= amount

                # Recalculate average entry price for short
                if self.amount < 0:
                    self.avg_entry_price = self.total_sold / abs(self.amount)
                else:
                    self.avg_entry_price = Decimal("0")

        self.total_fees += fee

        logger.debug(
            "Updated position {}: side={}, amount={}, price={}, "
            "total_amount={}, realized_pnl={}",
            self.symbol, side, amount, price, self.amount, self.realized_pnl
        )

        return self.realized_pnl, self.unrealized_pnl

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L at current price.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if self.amount == 0:
            self.unrealized_pnl = Decimal("0")
            return self.unrealized_pnl

        if self.amount > 0:
            # Long position
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.amount
        else:
            # Short position
            self.unrealized_pnl = (self.avg_entry_price - current_price) * abs(self.amount)

        return self.unrealized_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "symbol": self.symbol,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "amount": float(self.amount),
            "avg_entry_price": float(self.avg_entry_price),
            "avg_exit_price": float(self.avg_exit_price),
            "total_bought": float(self.total_bought),
            "total_sold": float(self.total_sold),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "total_fees": float(self.total_fees),
            "break_even_price": float(self.break_even_price),
            "is_long": self.is_long,
            "is_short": self.is_short,
            "is_open": self.is_open,
            "first_trade_time": self.first_trade_time.isoformat() if self.first_trade_time else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "trade_count": self.trade_count
        }

    def __repr__(self) -> str:
        return (f"Position(symbol={self.symbol}, amount={self.amount}, "
                f"avg_price={self.avg_entry_price}, "
                f"realized_pnl={self.realized_pnl}, unrealized_pnl={self.unrealized_pnl})")


class PortfolioManager:
    """Manages portfolio, positions, and risk parameters."""

    def __init__(
        self,
        initial_balance: Dict[str, float],
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.3,
        max_portfolio_risk: float = 0.1,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15
    ):
        """Initialize PortfolioManager.

        Args:
            initial_balance: Initial balance per currency (e.g., {"USDT": 10000})
            risk_per_trade: Risk per trade as fraction of portfolio (0.02 = 2%)
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_risk: Maximum total portfolio risk
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
        """
        # Convert balances to Decimal
        self.balances = {k: Decimal(str(v)) for k, v in initial_balance.items()}
        self.initial_balances = self.balances.copy()

        # Risk parameters
        self.risk_per_trade = Decimal(str(risk_per_trade))
        self.max_position_size = Decimal(str(max_position_size))
        self.max_portfolio_risk = Decimal(str(max_portfolio_risk))
        self.stop_loss_pct = Decimal(str(stop_loss_pct))
        self.take_profit_pct = Decimal(str(take_profit_pct))

        # Positions
        self.positions: Dict[str, Position] = {}

        # Performance tracking
        self.total_realized_pnl = Decimal("0")
        self.total_unrealized_pnl = Decimal("0")
        self.total_fees_paid = Decimal("0")
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.start_time = datetime.now(timezone.utc)

        logger.info(
            "Initialized portfolio with balances: {}, risk_per_trade={}, "
            "max_position_size={}",
            self.balances, self.risk_per_trade, self.max_position_size
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def get_or_create_position(self, symbol: str) -> Position:
        """Get existing position or create new one."""
        if symbol not in self.positions:
            # Parse symbol (e.g., "BTC/USDT" -> base="BTC", quote="USDT")
            parts = symbol.replace("/", "-").split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid symbol format: {symbol}")

            base_currency, quote_currency = parts
            self.positions[symbol] = Position(symbol, base_currency, quote_currency)

        return self.positions[symbol]

    def update_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        fee: float = 0.0
    ) -> Tuple[Decimal, Decimal]:
        """Update position with a trade.

        Args:
            symbol: Trading pair symbol
            side: "buy" or "sell"
            amount: Amount traded
            price: Trade price
            fee: Fee paid in quote currency

        Returns:
            Tuple of (realized_pnl, unrealized_pnl)
        """
        position = self.get_or_create_position(symbol)

        try:
            amount_dec = Decimal(str(amount))
            price_dec = Decimal(str(price))
            fee_dec = Decimal(str(fee))

            # Update position
            realized_pnl, unrealized_pnl = position.update_position(
                side, amount_dec, price_dec, fee_dec
            )

            # Update portfolio totals
            self.total_realized_pnl = sum(p.realized_pnl for p in self.positions.values())
            self.total_fees_paid = sum(p.total_fees for p in self.positions.values())
            self.total_trades += 1

            # Update win/loss count
            if realized_pnl > 0:
                self.winning_trades += 1
            elif realized_pnl < 0:
                self.losing_trades += 1

            # Update balance for closed position
            if not position.is_open:
                self._update_balance_from_closed_position(position, symbol)

            logger.info(
                "Trade executed: {} {} {} @ {}, fee={}, "
                "realized_pnl={}, total_trades={}",
                side, amount, symbol, price, fee, realized_pnl, self.total_trades
            )

            return realized_pnl, unrealized_pnl

        except (InvalidOperation, ValueError) as e:
            logger.error("Failed to update position: {}", str(e))
            raise

    def _update_balance_from_closed_position(self, position: Position, symbol: str) -> None:
        """Update balance when position is closed."""
        parts = symbol.replace("/", "-").split("-")
        if len(parts) != 2:
            return

        _, quote_currency = parts

        # Add realized P&L to balance
        if quote_currency not in self.balances:
            self.balances[quote_currency] = Decimal("0")

        self.balances[quote_currency] += position.realized_pnl

        logger.debug(
            "Updated balance for {} due to closed position: {}",
            quote_currency, self.balances[quote_currency]
        )

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        risk_amount: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk parameters.

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional, calculates from default if None)
            risk_amount: Specific risk amount (optional, uses portfolio risk if None)

        Returns:
            Position size in base currency
        """
        try:
            entry_price_dec = Decimal(str(entry_price))
            parts = symbol.replace("/", "-").split("-")

            if len(parts) != 2:
                raise ValueError(f"Invalid symbol format: {symbol}")

            _, quote_currency = parts

            # Get available balance in quote currency
            available_balance = self.balances.get(quote_currency, Decimal("0"))

            if available_balance <= 0:
                logger.warning("No available balance for {}", quote_currency)
                return 0.0

            # Calculate risk amount
            if risk_amount is None:
                risk_amount = available_balance * self.risk_per_trade
            else:
                risk_amount = Decimal(str(risk_amount))

            # Calculate stop loss
            if stop_loss_price is None:
                # Use default stop loss percentage
                if self.side_is_long(symbol):
                    stop_loss_price = entry_price_dec * (1 - self.stop_loss_pct)
                else:
                    stop_loss_price = entry_price_dec * (1 + self.stop_loss_pct)
            else:
                stop_loss_price = Decimal(str(stop_loss_price))

            # Calculate risk per unit
            risk_per_unit = abs(entry_price_dec - stop_loss_price)

            if risk_per_unit == 0:
                logger.warning("Risk per unit is zero, cannot calculate position size")
                return 0.0

            # Calculate position size
            position_size = risk_amount / risk_per_unit

            # Apply maximum position size limit
            max_position_value = available_balance * self.max_position_size
            max_size = max_position_value / entry_price_dec

            position_size = min(position_size, max_size)

            # Ensure we have enough balance
            required_margin = position_size * entry_price_dec
            if required_margin > available_balance:
                position_size = available_balance / entry_price_dec

            result = float(position_size)
            logger.debug(
                "Calculated position size for {}: {} @ {} (risk={})",
                symbol, result, entry_price, risk_amount
            )

            return result

        except (InvalidOperation, ValueError, ZeroDivisionError) as e:
            logger.error("Failed to calculate position size: {}", str(e))
            return 0.0

    def side_is_long(self, symbol: str) -> bool:
        """Determine if we should go long or short (default: long)."""
        # This can be enhanced with strategy signals
        return True

    def calculate_unrealized_pnl(self, prices: Dict[str, float]) -> Decimal:
        """Calculate total unrealized P&L for all open positions.

        Args:
            prices: Dictionary of current prices per symbol

        Returns:
            Total unrealized P&L
        """
        total_unrealized = Decimal("0")

        for symbol, position in self.positions.items():
            if position.is_open and symbol in prices:
                try:
                    current_price = Decimal(str(prices[symbol]))
                    unrealized = position.calculate_unrealized_pnl(current_price)
                    total_unrealized += unrealized
                except (InvalidOperation, ValueError) as e:
                    logger.error("Failed to calculate unrealized P&L for {}: {}", symbol, e)

        self.total_unrealized_pnl = total_unrealized
        return total_unrealized

    def get_portfolio_value(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, Decimal]:
        """Calculate total portfolio value.

        Args:
            prices: Current prices for valuation (optional)

        Returns:
            Portfolio value per currency
        """
        portfolio_value = self.balances.copy()

        # Add value of open positions
        if prices:
            for symbol, position in self.positions.items():
                if position.is_open and symbol in prices:
                    parts = symbol.replace("/", "-").split("-")
                    if len(parts) == 2:
                        base_currency, quote_currency = parts

                        # Value position at current price
                        try:
                            current_price = Decimal(str(prices[symbol]))
                            position_value = position.amount * current_price

                            if quote_currency not in portfolio_value:
                                portfolio_value[quote_currency] = Decimal("0")

                            portfolio_value[quote_currency] += position_value
                        except (InvalidOperation, ValueError):
                            pass

        return portfolio_value

    def get_total_portfolio_value(self, prices: Optional[Dict[str, float]] = None) -> Decimal:
        """Get total portfolio value in base currency (USDT or first currency)."""
        values = self.get_portfolio_value(prices)
        # Return USDT value or first currency
        return values.get("USDT", values.get("USD", Decimal("0")))

    def rebalance_portfolio(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Calculate rebalancing trades to achieve target weights.

        Args:
            target_weights: Target weight per symbol (e.g., {"BTC/USDT": 0.4, "ETH/USDT": 0.6})
            prices: Current prices for valuation
            threshold: Rebalance only if deviation exceeds this (5% default)

        Returns:
            List of rebalancing trades
        """
        trades = []
        total_value = self.get_total_portfolio_value(prices)

        if total_value <= 0:
            logger.warning("Cannot rebalance: portfolio value is zero")
            return trades

        # Calculate current weights
        current_weights = {}
        for symbol, position in self.positions.items():
            if position.is_open and symbol in prices:
                try:
                    current_price = Decimal(str(prices[symbol]))
                    position_value = position.amount * current_price
                    current_weights[symbol] = float(position_value / total_value)
                except (InvalidOperation, ValueError):
                    pass

        # Calculate required trades
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                continue

            current_weight = current_weights.get(symbol, 0.0)
            deviation = abs(current_weight - target_weight)

            if deviation > threshold:
                target_value = total_value * Decimal(str(target_weight))
                current_value = total_value * Decimal(str(current_weight))
                diff = target_value - current_value

                if abs(diff) > 0:  # Only trade if significant difference
                    price = Decimal(str(prices[symbol]))
                    amount = abs(diff) / price

                    side = "buy" if diff > 0 else "sell"

                    trades.append({
                        "symbol": symbol,
                        "side": side,
                        "amount": float(amount),
                        "price": float(price),
                        "reason": f"Rebalance: current={current_weight:.2%}, "
                                 f"target={target_weight:.2%}"
                    })

        logger.info("Calculated {} rebalancing trades", len(trades))
        return trades

    def check_risk_limits(self) -> Dict[str, bool]:
        """Check if portfolio is within risk limits.

        Returns:
            Dictionary with risk limit statuses
        """
        total_value = self.get_total_portfolio_value()
        total_risk = Decimal("0")

        for position in self.positions.values():
            if position.is_open:
                # Estimate risk as unrealized loss if price moves 5%
                estimated_loss = abs(position.amount) * position.avg_entry_price * Decimal("0.05")
                total_risk += estimated_loss

        risk_ratio = total_risk / total_value if total_value > 0 else Decimal("0")

        return {
            "within_risk_per_trade": True,  # Checked at trade time
            "within_max_position_size": True,  # Checked at trade time
            "within_portfolio_risk": risk_ratio <= self.max_portfolio_risk,
            "risk_ratio": float(risk_ratio)
        }

    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics.

        Returns:
            Dictionary with portfolio statistics
        """
        # Calculate performance metrics
        win_rate = (self.winning_trades / self.total_trades
                   if self.total_trades > 0 else Decimal("0"))

        avg_win = Decimal("0")
        avg_loss = Decimal("0")

        # Calculate average win/loss from positions
        total_won = Decimal("0")
        total_lost = Decimal("0")

        for position in self.positions.values():
            if position.realized_pnl > 0:
                total_won += position.realized_pnl
            elif position.realized_pnl < 0:
                total_lost += abs(position.realized_pnl)

        if self.winning_trades > 0:
            avg_win = total_won / self.winning_trades
        if self.losing_trades > 0:
            avg_loss = total_lost / self.losing_trades

        # Profit factor
        profit_factor = total_won / total_lost if total_lost > 0 else Decimal("0")

        # Open positions
        open_positions = [p for p in self.positions.values() if p.is_open]

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": float(win_rate),
            "total_realized_pnl": float(self.total_realized_pnl),
            "total_unrealized_pnl": float(self.total_unrealized_pnl),
            "total_fees_paid": float(self.total_fees_paid),
            "net_pnl": float(self.total_realized_pnl - self.total_fees_paid),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "open_positions": len(open_positions),
            "balances": {k: float(v) for k, v in self.balances.items()},
            "initial_balances": {k: float(v) for k, v in self.initial_balances.items()},
            "start_time": self.start_time.isoformat(),
            "risk_per_trade": float(self.risk_per_trade),
            "max_position_size": float(self.max_position_size),
            "current_risk_status": self.check_risk_limits()
        }

    def close_all_positions(
        self,
        prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Calculate trades to close all positions.

        Args:
            prices: Current prices

        Returns:
            List of closing trades
        """
        trades = []

        for symbol, position in self.positions.items():
            if position.is_open and symbol in prices:
                if position.amount > 0:
                    side = "sell"
                else:
                    side = "buy"

                trades.append({
                    "symbol": symbol,
                    "side": side,
                    "amount": float(abs(position.amount)),
                    "price": prices[symbol],
                    "reason": "Close all positions"
                })

        logger.info("Calculated {} trades to close all positions", len(trades))
        return trades

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary representation."""
        return {
            "balances": {k: float(v) for k, v in self.balances.items()},
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "statistics": self.get_portfolio_statistics()
        }
