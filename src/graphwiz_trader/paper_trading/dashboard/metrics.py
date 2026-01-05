"""
Performance metrics calculator for paper trading dashboard.

Calculates returns, drawdown, Sharpe ratio, and other performance metrics.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def calculate_returns(equity_df: pd.DataFrame) -> pd.Series:
    """Calculate period-over-period returns.

    Args:
        equity_df: DataFrame with total_value column

    Returns:
        Series of returns
    """
    if equity_df.empty or "total_value" not in equity_df.columns:
        return pd.Series([], dtype=float)

    # Calculate percentage change
    returns = equity_df["total_value"].pct_change().fillna(0)
    return returns


def calculate_drawdown(equity_df: pd.DataFrame) -> pd.Series:
    """Calculate drawdown from peak.

    Args:
        equity_df: DataFrame with total_value column

    Returns:
        Series of drawdown values (negative values)
    """
    if equity_df.empty or "total_value" not in equity_df.columns:
        return pd.Series([], dtype=float)

    # Calculate cumulative maximum
    cummax = equity_df["total_value"].cummax()

    # Calculate drawdown
    drawdown = (equity_df["total_value"] - cummax) / cummax

    return drawdown


def calculate_max_drawdown(equity_df: pd.DataFrame) -> float:
    """Calculate maximum drawdown.

    Args:
        equity_df: DataFrame with total_value column

    Returns:
        Maximum drawdown as negative percentage
    """
    drawdown = calculate_drawdown(equity_df)
    if drawdown.empty:
        return 0.0
    return drawdown.min()


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24,  # Hourly data
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year (default hourly)

    Returns:
        Annualized Sharpe ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)

    # Calculate Sharpe ratio
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    return sharpe


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Calculate volatility (standard deviation of returns).

    Args:
        returns: Series of period returns
        annualize: Whether to annualize (default True)

    Returns:
        Volatility
    """
    if returns.empty:
        return 0.0

    vol = returns.std()

    if annualize:
        # Assume hourly data (365 * 24 periods per year)
        vol *= np.sqrt(365 * 24)

    return vol


def get_latest_metrics(
    symbol: str, equity_df: Optional[pd.DataFrame] = None, summary: Optional[dict] = None
) -> dict:
    """Get latest performance metrics for a symbol.

    Args:
        symbol: Trading symbol
        equity_df: Optional pre-loaded equity DataFrame
        summary: Optional pre-loaded summary dict

    Returns:
        Dictionary of metrics
    """
    from .data_loader import load_equity_curve, load_summary

    # Load data if not provided
    if equity_df is None:
        equity_df = load_equity_curve(symbol)

    if summary is None:
        summary = load_summary(symbol)

    if equity_df is None:
        return {
            "symbol": symbol,
            "error": "No data available",
        }

    # Calculate metrics
    returns = calculate_returns(equity_df)
    max_drawdown = calculate_max_drawdown(equity_df)
    sharpe = calculate_sharpe_ratio(returns)
    volatility = calculate_volatility(returns)

    # Get latest values
    latest = equity_df.iloc[-1]
    initial = equity_df.iloc[0]

    total_return = latest["total_value"] - initial["total_value"]
    total_return_pct = (total_return / initial["total_value"]) * 100

    metrics = {
        "symbol": symbol,
        "initial_capital": float(initial["total_value"]),
        "final_value": float(latest["total_value"]),
        "total_return": float(total_return),
        "total_return_pct": float(total_return_pct),
        "max_drawdown": float(max_drawdown * 100),  # Convert to percentage
        "sharpe_ratio": float(sharpe),
        "volatility": float(volatility * 100),  # Convert to percentage
        "current_position": float(latest["position"]),
        "current_position_value": float(latest["position_value"]),
        "data_points": len(equity_df),
        "first_timestamp": str(equity_df.iloc[0]["timestamp"]),
        "last_timestamp": str(equity_df.iloc[-1]["timestamp"]),
    }

    # Add summary data if available
    if summary:
        metrics["total_trades"] = summary.get("total_trades", 0)
        metrics["win_rate"] = summary.get("win_rate", 0)

    return metrics


def calculate_correlation(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate correlation matrix between multiple symbols.

    Args:
        data: Dictionary mapping symbol to equity DataFrame

    Returns:
        Correlation matrix
    """
    if len(data) < 2:
        return pd.DataFrame()

    # Create DataFrame with returns for each symbol
    returns_dict = {}

    for symbol, df in data.items():
        if not df.empty and "total_value" in df.columns:
            returns = calculate_returns(df)
            returns_dict[symbol] = returns

    if not returns_dict:
        return pd.DataFrame()

    # Combine into single DataFrame
    combined = pd.DataFrame(returns_dict)

    # Calculate correlation
    corr = combined.corr()

    return corr
