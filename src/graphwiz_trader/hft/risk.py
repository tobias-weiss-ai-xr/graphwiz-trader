"""
HFT Risk Management Module.

Provides real-time risk management for high-frequency trading.
"""

import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from loguru import logger


class HFTRiskManager:
    """Real-time risk management for HFT."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize HFT risk manager.

        Args:
            config: Risk management configuration
        """
        self.max_position_size = config.get("max_position_size", 1.0)
        self.max_exposure = config.get("max_exposure", 10000.0)
        self.max_orders_per_second = config.get("max_orders_per_sec", 10)
        self.circuit_breaker_threshold = config.get(
            "circuit_breaker", -0.05
        )  # -5% loss triggers breaker
        self.max_drawdown_pct = config.get("max_drawdown_pct", 10.0)
        self.position_limit_per_symbol = config.get("position_limit_per_symbol", 0.5)

        # State tracking
        self.positions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.daily_pnl = 0.0
        self.peak_pnl = 0.0
        self.order_count = 0
        self.circuit_breaker_tripped = False
        self.last_reset = time.time()
        self.start_balance = config.get("start_balance", 10000.0)
        self.current_balance = self.start_balance

        # Order rate tracking
        self.order_timestamps: list = []
        self.rate_limit_window = 1.0  # 1 second window

        # Statistics
        self.risk_violations: Dict[str, int] = defaultdict(int)
        self.total_orders_checked = 0

    async def check_order(self, order: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if order passes risk controls.

        Args:
            order: Order details

        Returns:
            Tuple of (approved: bool, reason: str)
        """
        self.total_orders_checked += 1

        # Circuit breaker check
        if self.circuit_breaker_tripped:
            self.risk_violations["circuit_breaker"] += 1
            return False, "Circuit breaker tripped - trading halted"

        symbol = order.get("symbol")
        exchange = order.get("exchange")
        side = order.get("side")
        amount = order.get("amount", 0)

        if not all([symbol, exchange, side]):
            return False, "Missing required order fields"

        # Check position size limits
        current_position = self.positions[symbol][exchange]
        new_position = current_position + amount if side == "buy" else current_position - amount

        if abs(new_position) > self.max_position_size:
            self.risk_violations["position_size"] += 1
            return (
                False,
                f"Position size limit exceeded: {abs(new_position):.4f} > {self.max_position_size}",
            )

        # Check per-symbol position limit
        total_symbol_position = sum(abs(pos) for pos in self.positions[symbol].values())
        if total_symbol_position > self.position_limit_per_symbol:
            self.risk_violations["symbol_limit"] += 1
            return (
                False,
                f"Symbol position limit exceeded: {total_symbol_position:.4f} > {self.position_limit_per_symbol}",
            )

        # Check total exposure
        total_exposure = self._calculate_total_exposure()
        estimated_new_exposure = total_exposure + (amount * order.get("price", 0))

        if estimated_new_exposure > self.max_exposure:
            self.risk_violations["exposure"] += 1
            return (
                False,
                f"Total exposure limit exceeded: ${estimated_new_exposure:.2f} > ${self.max_exposure}",
            )

        # Check order rate
        approved, reason = self._check_order_rate()
        if not approved:
            self.risk_violations["rate_limit"] += 1
            return approved, reason

        # Check drawdown
        if self.daily_pnl < 0:
            drawdown_pct = abs(self.daily_pnl) / self.start_balance * 100
            if drawdown_pct > self.max_drawdown_pct:
                self.risk_violations["drawdown"] += 1
                return (
                    False,
                    f"Max drawdown exceeded: {drawdown_pct:.2f}% > {self.max_drawdown_pct}%",
                )

        return True, "OK"

    def _check_order_rate(self) -> Tuple[bool, str]:
        """
        Check if order rate is within limits.

        Returns:
            Tuple of (approved: bool, reason: str)
        """
        now = time.time()

        # Remove old timestamps outside the window
        self.order_timestamps = [
            ts for ts in self.order_timestamps if now - ts < self.rate_limit_window
        ]

        if len(self.order_timestamps) >= self.max_orders_per_second:
            return (
                False,
                f"Order rate limit exceeded: {len(self.order_timestamps)} orders in {self.rate_limit_window}s",
            )

        # Add current timestamp
        self.order_timestamps.append(now)
        return True, "OK"

    def _calculate_total_exposure(self) -> float:
        """
        Calculate total exposure across all positions.

        Returns:
            Total exposure in base currency
        """
        # Simplified calculation - in production would use current market prices
        total = 0.0
        for symbol_positions in self.positions.values():
            for position in symbol_positions.values():
                total += abs(position)
        return total

    def update_position(self, fill: Dict[str, Any]) -> None:
        """
        Update position after order fill.

        Args:
            fill: Fill information
        """
        symbol = fill.get("symbol")
        exchange = fill.get("exchange")
        side = fill.get("side")
        filled = fill.get("filled", 0)
        pnl = fill.get("pnl", 0)

        if not all([symbol, exchange, side]):
            logger.warning("Incomplete fill data, skipping position update")
            return

        # Update position
        if side == "buy":
            self.positions[symbol][exchange] += filled
        else:
            self.positions[symbol][exchange] -= filled

        # Update PnL
        self.daily_pnl += pnl
        self.current_balance = self.start_balance + self.daily_pnl

        # Update peak PnL for drawdown calculation
        if self.daily_pnl > self.peak_pnl:
            self.peak_pnl = self.daily_pnl

        # Check circuit breaker
        loss_threshold = self.max_exposure * self.circuit_breaker_threshold
        if self.daily_pnl < loss_threshold:
            self.circuit_breaker_tripped = True
            logger.critical(
                f"CIRCUIT BREAKER TRIPPED! Daily PnL: ${self.daily_pnl:.2f}, "
                f"Threshold: ${loss_threshold:.2f}"
            )

        logger.debug(
            f"Position updated: {symbol} on {exchange} = {self.positions[symbol][exchange]:.4f}, "
            f"Daily PnL: ${self.daily_pnl:.2f}"
        )

    def get_position(self, symbol: str, exchange: str) -> float:
        """
        Get current position for symbol on exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange identifier

        Returns:
            Position size
        """
        return self.positions[symbol][exchange]

    def get_all_positions(self) -> Dict[str, Dict[str, float]]:
        """
        Get all current positions.

        Returns:
            Nested dictionary of symbol -> exchange -> position
        """
        return dict(self.positions)

    def get_total_position(self, symbol: str) -> float:
        """
        Get total position across all exchanges for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Total position size
        """
        return sum(self.positions[symbol].values())

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (requires manual intervention)."""
        self.circuit_breaker_tripped = False
        logger.warning("Circuit breaker manually reset")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (typically called at start of trading day)."""
        self.daily_pnl = 0.0
        self.peak_pnl = 0.0
        self.order_count = 0
        self.last_reset = time.time()
        self.circuit_breaker_tripped = False
        self.order_timestamps.clear()
        self.current_balance = self.start_balance
        logger.info("Daily risk statistics reset")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.

        Returns:
            Dictionary of risk metrics
        """
        total_exposure = self._calculate_total_exposure()
        current_drawdown = 0.0

        if self.peak_pnl > 0 and self.daily_pnl < self.peak_pnl:
            current_drawdown = (self.peak_pnl - self.daily_pnl) / self.peak_pnl * 100

        return {
            "daily_pnl": self.daily_pnl,
            "current_balance": self.current_balance,
            "peak_pnl": self.peak_pnl,
            "current_drawdown_pct": current_drawdown,
            "total_exposure": total_exposure,
            "exposure_utilization_pct": (total_exposure / self.max_exposure * 100)
            if self.max_exposure > 0
            else 0,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "total_orders_checked": self.total_orders_checked,
            "risk_violations": dict(self.risk_violations),
            "current_order_rate": len(self.order_timestamps),
            "positions_count": sum(
                len(exchanges) for exchanges in self.positions.values()
            ),
        }

    def is_safe_to_trade(self) -> Tuple[bool, str]:
        """
        Check if it's safe to trade based on current risk state.

        Returns:
            Tuple of (safe: bool, reason: str)
        """
        if self.circuit_breaker_tripped:
            return False, "Circuit breaker is tripped"

        metrics = self.get_risk_metrics()

        if metrics["current_drawdown_pct"] > self.max_drawdown_pct:
            return False, f"Drawdown exceeds limit: {metrics['current_drawdown_pct']:.2f}%"

        if metrics["exposure_utilization_pct"] > 90:
            return False, f"Exposure near limit: {metrics['exposure_utilization_pct']:.2f}%"

        return True, "Safe to trade"

    async def close_all_positions(self) -> None:
        """Mark all positions as closed (does not execute trades)."""
        logger.info("Closing all positions in risk manager")
        self.positions.clear()

    def __repr__(self) -> str:
        """String representation of risk manager state."""
        metrics = self.get_risk_metrics()
        return (
            f"HFTRiskManager("
            f"daily_pnl=${metrics['daily_pnl']:.2f}, "
            f"exposure=${metrics['total_exposure']:.2f}, "
            f"circuit_breaker={metrics['circuit_breaker_tripped']}, "
            f"positions={metrics['positions_count']})"
        )
