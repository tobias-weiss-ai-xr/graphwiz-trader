"""
Paper trading module for testing strategies without real money.

Paper trading simulates live trading by:
- Fetching real market data
- Generating trading signals
- Tracking virtual portfolio and trades
- Logging performance metrics
- NOT executing real trades
"""

from .engine import PaperTradingEngine
from .portfolio import PaperPortfolio
from .trade import PaperTrade

__all__ = [
    "PaperTradingEngine",
    "PaperPortfolio",
    "PaperTrade",
]
