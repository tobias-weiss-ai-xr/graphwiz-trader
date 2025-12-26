"""
Paper trading portfolio management.
"""

from typing import Dict, Any


class PaperPortfolio:
    """Paper trading portfolio for tracking virtual positions."""

    def __init__(self, initial_capital: float = 10000.0):
        """Initialize paper portfolio.

        Args:
            initial_capital: Starting virtual capital
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, float] = {}

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value.

        Args:
            current_prices: Current prices for all assets

        Returns:
            Total portfolio value
        """
        total = self.capital
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                total += quantity * current_prices[symbol]
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary.

        Returns:
            Portfolio state as dictionary
        """
        return {
            "initial_capital": self.initial_capital,
            "capital": self.capital,
            "positions": self.positions.copy(),
        }
