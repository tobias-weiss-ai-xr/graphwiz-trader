"""
Paper trading dashboard module.

Provides interactive web-based visualization and analysis of paper trading results.
"""

from .data_loader import (
    load_all_symbols,
    load_equity_curve,
    load_summary,
    get_available_symbols,
)
from .metrics import (
    calculate_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    get_latest_metrics,
)
from .service_monitor import get_service_status, get_latest_log_lines
from .live_data import (
    get_current_price,
    get_all_symbols_prices,
    get_24h_stats,
    fetch_recent_candles,
    get_market_summary,
)

__all__ = [
    # Data loading
    "load_all_symbols",
    "load_equity_curve",
    "load_summary",
    "get_available_symbols",
    # Metrics
    "calculate_drawdown",
    "calculate_returns",
    "calculate_sharpe_ratio",
    "get_latest_metrics",
    # Service monitoring
    "get_service_status",
    "get_latest_log_lines",
    # Live data
    "get_current_price",
    "get_all_symbols_prices",
    "get_24h_stats",
    "fetch_recent_candles",
    "get_market_summary",
]
