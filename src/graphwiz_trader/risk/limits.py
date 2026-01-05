"""Risk limits and validation checks.

This module defines risk limits, performs validation checks, and provides
stop-loss/take-profit calculation capabilities.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import numpy as np


class RiskLimitType(Enum):
    """Types of risk limits."""

    MAX_POSITION_SIZE = "max_position_size"
    MAX_TOTAL_EXPOSURE = "max_total_exposure"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CORRELATION_EXPOSURE = "max_correlation_exposure"
    MAX_SECTOR_EXPOSURE = "max_sector_exposure"
    MAX_ASSET_CLASS_EXPOSURE = "max_asset_class_exposure"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_CONCENTRATION = "max_concentration"


@dataclass
class RiskLimit:
    """Risk limit configuration."""

    name: str
    limit_type: RiskLimitType
    value: float
    enabled: bool = True
    hard_limit: bool = False  # If True, strictly enforce; if False, warn only
    scope: str = "portfolio"  # 'portfolio', 'asset', 'sector', 'strategy'

    def check(self, current_value: float) -> Tuple[bool, float, float]:
        """Check if current value exceeds limit.

        Args:
            current_value: Current value to check against limit

        Returns:
            Tuple of (is_violated, current_value, limit_value)
        """
        if not self.enabled:
            return False, current_value, self.value

        is_violated = current_value > self.value
        return is_violated, current_value, self.value


@dataclass
class RiskLimitsConfig:
    """Comprehensive risk limits configuration."""

    # Position-level limits
    max_position_size: float = 0.10  # Max 10% of portfolio per position
    max_position_size_abs: Optional[float] = None  # Absolute dollar amount
    min_position_size: float = 0.001  # Min 0.1% of portfolio

    # Portfolio-level limits
    max_total_exposure: float = 1.0  # Max 100% of portfolio (no leverage)
    max_long_exposure: float = 1.0  # Max long exposure
    max_short_exposure: float = 0.5  # Max short exposure

    # Daily loss limits
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_daily_loss_abs: Optional[float] = None  # Absolute dollar amount

    # Correlation and concentration limits
    max_correlated_exposure: float = 0.30  # Max 30% in highly correlated assets
    correlation_threshold: float = 0.70  # Correlation threshold for "highly correlated"

    # Sector and asset class limits
    max_sector_exposure: float = 0.40  # Max 40% per sector
    max_asset_class_exposure: float = 0.60  # Max 60% per asset class
    max_single_asset_concentration: float = 0.20  # Max 20% in single asset

    # Drawdown limits
    max_drawdown_pct: float = 0.20  # Max 20% drawdown
    max_drawdown_duration_days: int = 90  # Max days in drawdown

    # Trading limits
    max_trades_per_day: int = 100
    max_turnover_pct: float = 0.50  # Max 50% daily turnover

    # Leverage limits
    max_gross_exposure: float = 1.5  # Max gross exposure (long + short)
    max_net_exposure: float = 1.0  # Max net exposure (long - short)


class RiskLimits:
    """Risk limits manager with validation checks."""

    def __init__(self, config: Optional[RiskLimitsConfig] = None):
        """Initialize risk limits manager.

        Args:
            config: Risk limits configuration (uses defaults if None)
        """
        self.config = config or RiskLimitsConfig()
        self.violations: List[Dict[str, Any]] = []
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0

    def check_position_size(
        self,
        position_value: float,
        portfolio_value: float,
        symbol: str,
        hard_limit: bool = True,
    ) -> Tuple[bool, str]:
        """Check if position size exceeds limits.

        Args:
            position_value: Value of position
            portfolio_value: Total portfolio value
            symbol: Asset symbol
            hard_limit: If True, enforce limit; if False, warn only

        Returns:
            Tuple of (is_allowed, message)
        """
        if portfolio_value <= 0:
            return False, "Invalid portfolio value"

        position_pct = position_value / portfolio_value

        # Check minimum size
        if position_pct < self.config.min_position_size:
            msg = f"Position too small: {position_pct:.2%} < {self.config.min_position_size:.2%}"
            logger.warning(msg)
            return False, msg

        # Check maximum percentage
        if position_pct > self.config.max_position_size:
            msg = f"Position too large: {position_pct:.2%} > {self.config.max_position_size:.2%}"
            if hard_limit:
                logger.error(msg)
                self._record_violation("max_position_size", position_pct, symbol)
                return False, msg
            else:
                logger.warning(msg)
                self._record_violation("max_position_size", position_pct, symbol, warning=True)

        # Check absolute limit if set
        if self.config.max_position_size_abs and position_value > self.config.max_position_size_abs:
            msg = f"Position exceeds absolute limit: ${position_value:.2f} > ${self.config.max_position_size_abs:.2f}"
            if hard_limit:
                logger.error(msg)
                self._record_violation("max_position_size_abs", position_value, symbol)
                return False, msg
            else:
                logger.warning(msg)

        return True, "Position size approved"

    def check_total_exposure(
        self,
        total_exposure: float,
        portfolio_value: float,
        hard_limit: bool = True,
    ) -> Tuple[bool, str]:
        """Check if total exposure exceeds limits.

        Args:
            total_exposure: Total exposure (long + short absolute values)
            portfolio_value: Total portfolio value
            hard_limit: If True, enforce limit; if False, warn only

        Returns:
            Tuple of (is_allowed, message)
        """
        if portfolio_value <= 0:
            return False, "Invalid portfolio value"

        exposure_ratio = total_exposure / portfolio_value

        if exposure_ratio > self.config.max_total_exposure:
            msg = f"Total exposure too high: {exposure_ratio:.2%} > {self.config.max_total_exposure:.2%}"
            if hard_limit:
                logger.error(msg)
                self._record_violation("max_total_exposure", exposure_ratio)
                return False, msg
            else:
                logger.warning(msg)

        # Check gross exposure limit
        if exposure_ratio > self.config.max_gross_exposure:
            msg = f"Gross exposure too high: {exposure_ratio:.2%} > {self.config.max_gross_exposure:.2%}"
            if hard_limit:
                logger.error(msg)
                self._record_violation("max_gross_exposure", exposure_ratio)
                return False, msg

        return True, "Exposure approved"

    def check_daily_loss(
        self,
        daily_pnl: float,
        portfolio_value: float,
        hard_limit: bool = True,
    ) -> Tuple[bool, str]:
        """Check if daily loss exceeds limits.

        Args:
            daily_pnl: Daily P&L (negative for loss)
            portfolio_value: Total portfolio value
            hard_limit: If True, enforce limit; if False, warn only

        Returns:
            Tuple of (is_allowed, message)
        """
        self.daily_pnl = daily_pnl

        if daily_pnl >= 0:
            return True, "No loss to check"

        loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0

        if loss_pct > self.config.max_daily_loss_pct:
            msg = f"Daily loss exceeds limit: {loss_pct:.2%} > {self.config.max_daily_loss_pct:.2%}"
            if hard_limit:
                logger.error(msg)
                self._record_violation("max_daily_loss", loss_pct, abs(daily_pnl))
                return False, msg
            else:
                logger.warning(msg)

        # Check absolute limit if set
        if self.config.max_daily_loss_abs and abs(daily_pnl) > self.config.max_daily_loss_abs:
            msg = (
                f"Daily loss exceeds absolute limit: ${abs(daily_pnl):.2f} > "
                f"${self.config.max_daily_loss_abs:.2f}"
            )
            if hard_limit:
                logger.error(msg)
                self._record_violation("max_daily_loss_abs", abs(daily_pnl))
                return False, msg

        return True, "Daily loss within limits"

    def check_correlation_exposure(
        self,
        positions: List[Dict[str, Any]],
        correlation_matrix: Dict[Tuple[str, str], float],
        hard_limit: bool = True,
    ) -> Tuple[bool, str]:
        """Check exposure to highly correlated assets.

        Args:
            positions: List of position dicts with 'symbol' and 'value'
            correlation_matrix: Dict of (symbol1, symbol2) -> correlation
            hard_limit: If True, enforce limit; if False, warn only

        Returns:
            Tuple of (is_allowed, message)
        """
        if not positions:
            return True, "No positions to check"

        # Group highly correlated positions
        correlated_groups = self._find_correlated_groups(positions, correlation_matrix)

        for group in correlated_groups:
            group_exposure = sum(pos["value"] for pos in group)
            symbols = [pos["symbol"] for pos in group]

            # Check against portfolio value
            portfolio_value = sum(pos["value"] for pos in positions)
            if portfolio_value > 0:
                exposure_pct = group_exposure / portfolio_value

                if exposure_pct > self.config.max_correlated_exposure:
                    msg = (
                        f"Correlated exposure too high: {exposure_pct:.2%} in {symbols} "
                        f"> {self.config.max_correlated_exposure:.2%}"
                    )
                    if hard_limit:
                        logger.error(msg)
                        self._record_violation("max_correlation_exposure", exposure_pct, symbols)
                        return False, msg
                    else:
                        logger.warning(msg)

        return True, "Correlation exposure within limits"

    def _find_correlated_groups(
        self,
        positions: List[Dict[str, Any]],
        correlation_matrix: Dict[Tuple[str, str], float],
    ) -> List[List[Dict[str, Any]]]:
        """Find groups of highly correlated positions."""
        # Build correlation graph
        symbol_positions = {pos["symbol"]: pos for pos in positions}

        # Find correlated pairs
        correlated_pairs = []
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1 :]:
                symbols = tuple(sorted([pos1["symbol"], pos2["symbol"]]))
                if symbols in correlation_matrix:
                    correlation = correlation_matrix[symbols]
                    if abs(correlation) >= self.config.correlation_threshold:
                        correlated_pairs.append(symbols)

        # Group connected components
        groups = []
        visited = set()

        for symbol in symbol_positions:
            if symbol in visited:
                continue

            # BFS to find all correlated symbols
            group = []
            queue = [symbol]
            visited.add(symbol)

            while queue:
                current = queue.pop(0)
                group.append(symbol_positions[current])

                # Find correlated symbols
                for pair in correlated_pairs:
                    if current in pair:
                        other = pair[0] if pair[1] == current else pair[1]
                        if other not in visited:
                            visited.add(other)
                            queue.append(other)

            if len(group) > 1:  # Only include groups with multiple assets
                groups.append(group)

        return groups

    def check_sector_exposure(
        self,
        positions: List[Dict[str, Any]],
        sector_map: Dict[str, str],
        portfolio_value: float,
        hard_limit: bool = True,
    ) -> Tuple[bool, str]:
        """Check sector concentration limits.

        Args:
            positions: List of position dicts with 'symbol' and 'value'
            sector_map: Dict mapping symbol to sector
            portfolio_value: Total portfolio value
            hard_limit: If True, enforce limit; if False, warn only

        Returns:
            Tuple of (is_allowed, message)
        """
        if portfolio_value <= 0:
            return False, "Invalid portfolio value"

        # Calculate sector exposures
        sector_exposure = {}
        for pos in positions:
            sector = sector_map.get(pos["symbol"], "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pos["value"]

        # Check each sector
        for sector, exposure in sector_exposure.items():
            exposure_pct = exposure / portfolio_value

            if exposure_pct > self.config.max_sector_exposure:
                msg = (
                    f"Sector exposure too high: {sector} {exposure_pct:.2%} > "
                    f"{self.config.max_sector_exposure:.2%}"
                )
                if hard_limit:
                    logger.error(msg)
                    self._record_violation("max_sector_exposure", exposure_pct, sector)
                    return False, msg
                else:
                    logger.warning(msg)

        return True, "Sector exposure within limits"

    def check_drawdown_limit(
        self,
        current_drawdown: float,
        hard_limit: bool = True,
    ) -> Tuple[bool, str]:
        """Check if drawdown exceeds limits.

        Args:
            current_drawdown: Current drawdown as percentage (0.15 = 15%)
            hard_limit: If True, enforce limit; if False, warn only

        Returns:
            Tuple of (is_allowed, message)
        """
        if current_drawdown > self.config.max_drawdown_pct:
            msg = (
                f"Drawdown exceeds limit: {current_drawdown:.2%} > "
                f"{self.config.max_drawdown_pct:.2%}"
            )
            if hard_limit:
                logger.error(msg)
                self._record_violation("max_drawdown", current_drawdown)
                return False, msg
            else:
                logger.warning(msg)

        return True, "Drawdown within limits"

    def check_trading_limits(
        self,
        num_trades_today: int,
        hard_limit: bool = True,
    ) -> Tuple[bool, str]:
        """Check if trading activity exceeds limits.

        Args:
            num_trades_today: Number of trades executed today
            hard_limit: If True, enforce limit; if False, warn only

        Returns:
            Tuple of (is_allowed, message)
        """
        if num_trades_today > self.config.max_trades_per_day:
            msg = f"Trade limit exceeded: {num_trades_today} > {self.config.max_trades_per_day}"
            if hard_limit:
                logger.error(msg)
                self._record_violation("max_trades_per_day", num_trades_today)
                return False, msg
            else:
                logger.warning(msg)

        return True, "Trading within limits"

    def _record_violation(
        self,
        limit_type: str,
        current_value: float,
        context: Any = None,
        warning: bool = False,
    ) -> None:
        """Record a risk limit violation.

        Args:
            limit_type: Type of limit violated
            current_value: Current value that exceeded limit
            context: Additional context (symbol, sector, etc.)
            warning: If True, this is a warning, not a hard violation
        """
        violation = {
            "limit_type": limit_type,
            "current_value": current_value,
            "context": context,
            "warning": warning,
        }
        self.violations.append(violation)

    def get_violations(self, warnings_only: bool = False) -> List[Dict[str, Any]]:
        """Get list of violations.

        Args:
            warnings_only: If True, only return warnings

        Returns:
            List of violation dictionaries
        """
        if warnings_only:
            return [v for v in self.violations if v.get("warning", False)]
        return self.violations

    def clear_violations(self) -> None:
        """Clear all recorded violations."""
        self.violations.clear()


class StopLossCalculator:
    """Calculator for stop-loss and take-profit levels."""

    def __init__(
        self,
        default_stop_loss_pct: float = 0.02,
        default_take_profit_pct: float = 0.06,
        risk_reward_ratio: float = 3.0,
    ):
        """Initialize stop-loss calculator.

        Args:
            default_stop_loss_pct: Default stop-loss percentage (2%)
            default_take_profit_pct: Default take-profit percentage (6%)
            risk_reward_ratio: Target risk/reward ratio (3:1)
        """
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.risk_reward_ratio = risk_reward_ratio

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str = "long",
        stop_loss_pct: Optional[float] = None,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0,
        support_level: Optional[float] = None,
    ) -> float:
        """Calculate stop-loss price.

        Args:
            entry_price: Entry price
            side: Position side ('long' or 'short')
            stop_loss_pct: Stop-loss percentage (uses default if None)
            atr: Average True Range for ATR-based stops
            atr_multiplier: ATR multiplier for stop distance
            support_level: Support/resistance level for placement

        Returns:
            Stop-loss price
        """
        if side.lower() == "long":
            # ATR-based stop
            if atr is not None:
                return entry_price - (atr * atr_multiplier)

            # Support level
            if support_level is not None and support_level < entry_price:
                return support_level

            # Percentage-based stop
            stop_pct = stop_loss_pct or self.default_stop_loss_pct
            return entry_price * (1 - stop_pct)

        else:  # short
            # ATR-based stop
            if atr is not None:
                return entry_price + (atr * atr_multiplier)

            # Resistance level
            if support_level is not None and support_level > entry_price:
                return support_level

            # Percentage-based stop
            stop_pct = stop_loss_pct or self.default_stop_loss_pct
            return entry_price * (1 + stop_pct)

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        side: str = "long",
        take_profit_pct: Optional[float] = None,
        resistance_level: Optional[float] = None,
    ) -> float:
        """Calculate take-profit price.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price (for risk/reward calculation)
            side: Position side ('long' or 'short')
            take_profit_pct: Take-profit percentage (calculates from R:R if None)
            resistance_level: Resistance/support level for placement

        Returns:
            Take-profit price
        """
        # Calculate risk amount
        if side.lower() == "long":
            risk = entry_price - stop_loss_price

            # Risk/reward-based take profit
            if take_profit_pct is None and resistance_level is None:
                return entry_price + (risk * self.risk_reward_ratio)

            # Resistance level
            if resistance_level is not None and resistance_level > entry_price:
                return resistance_level

            # Percentage-based
            take_profit_pct = take_profit_pct or self.default_take_profit_pct
            return entry_price * (1 + take_profit_pct)

        else:  # short
            risk = stop_loss_price - entry_price

            # Risk/reward-based take profit
            if take_profit_pct is None and resistance_level is None:
                return entry_price - (risk * self.risk_reward_ratio)

            # Support level
            if resistance_level is not None and resistance_level < entry_price:
                return resistance_level

            # Percentage-based
            take_profit_pct = take_profit_pct or self.default_take_profit_pct
            return entry_price * (1 - take_profit_pct)

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        side: str = "long",
        trailing_distance_pct: float = 0.03,
        best_price: Optional[float] = None,
    ) -> float:
        """Calculate trailing stop-loss price.

        Args:
            entry_price: Entry price
            current_price: Current market price
            side: Position side ('long' or 'short')
            trailing_distance_pct: Trailing distance percentage
            best_price: Best price since entry (uses entry if None)

        Returns:
            Trailing stop price
        """
        best_price = best_price or entry_price

        if side.lower() == "long":
            # Only move stop up, never down
            trailing_stop = best_price * (1 - trailing_distance_pct)
            return max(trailing_stop, entry_price * (1 - trailing_distance_pct))

        else:  # short
            # Only move stop down, never up
            trailing_stop = best_price * (1 + trailing_distance_pct)
            return min(trailing_stop, entry_price * (1 + trailing_distance_pct))

    def calculate_position_size_from_risk(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade_pct: float = 0.02,
    ) -> float:
        """Calculate position size based on risk amount.

        Args:
            account_balance: Total account balance
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            risk_per_trade_pct: Risk percentage per trade

        Returns:
            Position size (number of units/shares)
        """
        risk_amount = account_balance * risk_per_trade_pct
        stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance == 0:
            return 0.0

        position_size = risk_amount / stop_distance
        return position_size
