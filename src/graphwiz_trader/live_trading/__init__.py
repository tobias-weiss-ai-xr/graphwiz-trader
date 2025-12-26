"""
Live trading module for real money trading.

⚠️ WARNING: This module executes REAL trades with REAL money.
Always test thoroughly with paper trading first.

Safety features:
- Position size limits
- Daily loss limits
- Emergency shutdown
- Pre-trade checks
- Comprehensive logging
"""

from .engine import LiveTradingEngine
from .risk_manager import RiskManager
from .safety_limits import SafetyLimits

__all__ = [
    "LiveTradingEngine",
    "RiskManager",
    "SafetyLimits",
]
