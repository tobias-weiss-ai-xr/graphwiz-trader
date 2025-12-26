"""
Paper trading trade representation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PaperTrade:
    """Represents a paper trade."""

    timestamp: datetime
    symbol: str
    action: str  # "buy" or "sell"
    price: float
    quantity: float
    value: float
    cost: Optional[float] = None  # For buys: includes commission
    proceeds: Optional[float] = None  # For sells: after commission
    pnl: Optional[float] = None  # For sells: profit/loss
    pnl_pct: Optional[float] = None  # For sells: profit/loss percentage

    def to_dict(self) -> dict:
        """Convert trade to dictionary.

        Returns:
            Trade data as dictionary
        """
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "value": self.value,
            "cost": self.cost,
            "proceeds": self.proceeds,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }
