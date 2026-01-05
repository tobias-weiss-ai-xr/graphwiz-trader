"""Portfolio management module."""

from loguru import logger
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import numpy as np


@dataclass
class Asset:
    """Represents an asset in the portfolio."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    value: float = 0.0
    weight: float = 0.0  # Portfolio weight
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def __post_init__(self):
        self._update_value()

    def _update_value(self):
        """Update asset value and P&L."""
        self.value = self.quantity * self.current_price
        if self.quantity > 0 and self.entry_price > 0:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (
                (self.current_price - self.entry_price) / self.entry_price
            ) * 100


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""

    timestamp: datetime
    total_value: float
    cash: float
    invested_value: float
    total_pnl: float
    total_pnl_pct: float
    asset_count: int
    weights: Dict[str, float] = field(default_factory=dict)


class PortfolioManager:
    """Manages portfolio allocation and rebalancing."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_positions: int = 10,
        max_position_size: float = 0.2,  # 20% of portfolio
        rebalance_threshold: float = 0.05,  # 5% deviation triggers rebalance
    ):
        """Initialize portfolio manager.

        Args:
            initial_capital: Starting capital
            max_positions: Maximum number of positions
            max_position_size: Maximum size of any position (as fraction of portfolio)
            rebalance_threshold: Threshold for triggering rebalancing
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.rebalance_threshold = rebalance_threshold

        self.assets: Dict[str, Asset] = {}
        self.snapshots: List[PortfolioSnapshot] = []
        self.target_weights: Dict[str, float] = {}

    def add_position(
        self, symbol: str, quantity: float, price: float, timestamp: Optional[datetime] = None
    ) -> bool:
        """Add a position to the portfolio.

        Args:
            symbol: Asset symbol
            quantity: Amount to buy (positive) or sell (negative)
            price: Execution price
            timestamp: Trade timestamp

        Returns:
            True if position added successfully
        """
        if quantity <= 0:
            logger.warning("Invalid quantity: {}", quantity)
            return False

        cost = quantity * price
        if cost > self.cash:
            logger.warning("Insufficient cash for {}: need {}, have {}", symbol, cost, self.cash)
            return False

        # Check max positions
        if symbol not in self.assets and len(self.assets) >= self.max_positions:
            logger.warning("Max positions ({}) reached", self.max_positions)
            return False

        # Check position size
        portfolio_value = self.get_total_value()
        if portfolio_value > 0 and cost / portfolio_value > self.max_position_size:
            logger.warning(
                "Position size exceeds maximum: {} / {}",
                cost,
                portfolio_value * self.max_position_size,
            )
            return False

        # Update or create position
        if symbol in self.assets:
            asset = self.assets[symbol]
            # Update entry price (weighted average)
            total_cost = (asset.quantity * asset.entry_price) + cost
            asset.quantity += quantity
            asset.entry_price = total_cost / asset.quantity if asset.quantity > 0 else price
            asset.current_price = price
            asset._update_value()
        else:
            self.assets[symbol] = Asset(
                symbol=symbol, quantity=quantity, entry_price=price, current_price=price
            )

        self.cash -= cost
        self._update_weights()

        if timestamp:
            self._create_snapshot(timestamp)

        logger.info("Added position: {} {} @ {}", quantity, symbol, price)
        return True

    def remove_position(
        self, symbol: str, quantity: float, price: float, timestamp: Optional[datetime] = None
    ) -> bool:
        """Remove (reduce) a position from the portfolio.

        Args:
            symbol: Asset symbol
            quantity: Amount to sell
            price: Execution price
            timestamp: Trade timestamp

        Returns:
            True if position removed successfully
        """
        if symbol not in self.assets:
            logger.warning("Position not found: {}", symbol)
            return False

        asset = self.assets[symbol]
        if quantity > asset.quantity:
            logger.warning("Cannot sell more than owned: {} > {}", quantity, asset.quantity)
            quantity = asset.quantity

        proceeds = quantity * price
        self.cash += proceeds

        asset.quantity -= quantity
        if asset.quantity <= 0:
            del self.assets[symbol]
        else:
            asset.current_price = price
            asset._update_value()

        self._update_weights()

        if timestamp:
            self._create_snapshot(timestamp)

        logger.info("Removed position: {} {} @ {}", quantity, symbol, price)
        return True

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all assets.

        Args:
            prices: Dictionary mapping symbol to current price
        """
        for symbol, price in prices.items():
            if symbol in self.assets:
                self.assets[symbol].current_price = price
                self.assets[symbol]._update_value()

        self._update_weights()

    def get_total_value(self) -> float:
        """Calculate total portfolio value.

        Returns:
            Total value (cash + invested)
        """
        invested_value = sum(asset.value for asset in self.assets.values())
        return self.cash + invested_value

    def get_invested_value(self) -> float:
        """Get value of invested assets.

        Returns:
            Total value of all positions
        """
        return sum(asset.value for asset in self.assets.values())

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions.

        Returns:
            List of position dictionaries
        """
        return [
            {
                "symbol": asset.symbol,
                "quantity": asset.quantity,
                "entry_price": asset.entry_price,
                "current_price": asset.current_price,
                "value": asset.value,
                "weight": asset.weight,
                "unrealized_pnl": asset.unrealized_pnl,
                "unrealized_pnl_pct": asset.unrealized_pnl_pct,
            }
            for asset in self.assets.values()
        ]

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        total_value = self.get_total_value()
        invested_value = self.get_invested_value()
        total_pnl = sum(asset.unrealized_pnl for asset in self.assets.values())
        total_pnl_pct = (
            (total_pnl / (invested_value - total_pnl)) * 100
            if (invested_value - total_pnl) > 0
            else 0.0
        )

        return {
            "total_value": total_value,
            "cash": self.cash,
            "invested_value": invested_value,
            "cash_ratio": self.cash / total_value if total_value > 0 else 1.0,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "position_count": len(self.assets),
            "initial_capital": self.initial_capital,
            "total_return": total_value - self.initial_capital,
            "total_return_pct": ((total_value - self.initial_capital) / self.initial_capital) * 100,
        }

    def set_target_weights(self, weights: Dict[str, float]) -> None:
        """Set target allocation weights.

        Args:
            weights: Dictionary mapping symbol to target weight (0-1)
        """
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            self.target_weights = {k: v / total for k, v in weights.items()}
        else:
            self.target_weights = {}

        logger.info("Set target weights: {}", self.target_weights)

    def get_rebalance_trades(self) -> List[Dict[str, Any]]:
        """Calculate trades needed to rebalance to target weights.

        Returns:
            List of trade dictionaries with symbol, action, quantity
        """
        if not self.target_weights:
            return []

        total_value = self.get_total_value()
        trades = []

        # Check existing positions
        for symbol, asset in self.assets.items():
            target_weight = self.target_weights.get(symbol, 0)
            target_value = total_value * target_weight
            current_value = asset.value
            deviation = abs(current_value - target_value) / total_value if total_value > 0 else 0

            if deviation > self.rebalance_threshold:
                # Need to rebalance
                if current_value < target_value:
                    # Buy more
                    buy_value = target_value - current_value
                    quantity = buy_value / asset.current_price
                    trades.append(
                        {
                            "symbol": symbol,
                            "action": "buy",
                            "quantity": quantity,
                            "value": buy_value,
                            "reason": f"Rebalance: current weight {asset.weight:.2%}, target {target_weight:.2%}",
                        }
                    )
                else:
                    # Sell some
                    sell_value = current_value - target_value
                    quantity = sell_value / asset.current_price
                    trades.append(
                        {
                            "symbol": symbol,
                            "action": "sell",
                            "quantity": quantity,
                            "value": sell_value,
                            "reason": f"Rebalance: current weight {asset.weight:.2%}, target {target_weight:.2%}",
                        }
                    )

        # Check for missing positions
        for symbol, target_weight in self.target_weights.items():
            if symbol not in self.assets:
                target_value = total_value * target_weight
                # Need to add position (estimate price)
                trades.append(
                    {
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": 0,  # Will be calculated with actual price
                        "value": target_value,
                        "reason": f"New position: target weight {target_weight:.2%}",
                    }
                )

        return trades

    def calculate_portfolio_metrics(
        self, snapshot_history: Optional[List[PortfolioSnapshot]] = None
    ) -> Dict[str, Any]:
        """Calculate portfolio performance metrics.

        Args:
            snapshot_history: Historical snapshots (uses self.snapshots if None)

        Returns:
            Dictionary with performance metrics
        """
        snapshots = snapshot_history or self.snapshots

        if len(snapshots) < 2:
            return {}

        values = [s.total_value for s in snapshots]
        returns = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                ret = (values[i] - values[i - 1]) / values[i - 1]
                returns.append(ret)

        if not returns:
            return {}

        # Calculate metrics
        total_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0

        # Volatility (std dev of returns)
        volatility = np.std(returns) * np.sqrt(252) if returns else 0

        # Sharpe ratio (assuming 2% risk-free rate)
        avg_return = np.mean(returns) * 252 if returns else 0
        sharpe_ratio = (avg_return - 0.02) / volatility if volatility > 0 else 0

        # Maximum drawdown
        max_drawdown = 0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_return": total_return * 100,  # Percentage
            "volatility": volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown * 100,
            "current_value": values[-1],
            "peak_value": peak,
            "snapshot_count": len(snapshots),
        }

    def _update_weights(self) -> None:
        """Update portfolio weights for all assets."""
        total_value = self.get_total_value()
        if total_value > 0:
            for asset in self.assets.values():
                asset.weight = asset.value / total_value

    def _create_snapshot(self, timestamp: datetime) -> None:
        """Create a portfolio snapshot.

        Args:
            timestamp: Snapshot timestamp
        """
        total_value = self.get_total_value()
        invested_value = self.get_invested_value()
        total_pnl = sum(asset.unrealized_pnl for asset in self.assets.values())
        total_pnl_pct = (
            (total_pnl / (invested_value - total_pnl)) * 100
            if (invested_value - total_pnl) > 0
            else 0.0
        )

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=self.cash,
            invested_value=invested_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            asset_count=len(self.assets),
            weights={symbol: asset.weight for symbol, asset in self.assets.items()},
        )

        self.snapshots.append(snapshot)

        # Limit snapshot history
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]

    def get_snapshots(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get portfolio snapshot history.

        Args:
            limit: Maximum number of snapshots to return

        Returns:
            List of snapshot dictionaries
        """
        recent_snapshots = (
            self.snapshots[-limit:] if len(self.snapshots) > limit else self.snapshots
        )

        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "total_value": s.total_value,
                "cash": s.cash,
                "invested_value": s.invested_value,
                "total_pnl": s.total_pnl,
                "total_pnl_pct": s.total_pnl_pct,
                "asset_count": s.asset_count,
            }
            for s in recent_snapshots
        ]

    def clear_positions(self) -> None:
        """Clear all positions (sell everything)."""
        self.assets.clear()
        self._update_weights()
        logger.info("Cleared all positions")

    def reset(self, initial_capital: Optional[float] = None) -> None:
        """Reset portfolio to initial state.

        Args:
            initial_capital: New initial capital (uses original if None)
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital

        self.cash = self.initial_capital
        self.assets.clear()
        self.snapshots.clear()
        self.target_weights.clear()

        logger.info("Reset portfolio to initial capital: {}", self.initial_capital)
