"""Risk calculation utilities.

This module provides mathematical functions for calculating various risk metrics
including position sizes, portfolio risk, correlations, and drawdowns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from loguru import logger


class PositionSizingStrategy(Enum):
    """Position sizing strategies."""

    KELLY_CRITERION = "kelly_criterion"
    FIXED_FRACTIONAL = "fixed_fractional"
    FIXED_DOLLAR = "fixed_dollar"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"


def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade: float = 0.02,
    strategy: PositionSizingStrategy = PositionSizingStrategy.FIXED_FRACTIONAL,
    strategy_params: Optional[Dict] = None,
) -> Dict[str, float]:
    """Calculate optimal position size based on risk parameters.

    Args:
        account_balance: Total account balance/equity
        entry_price: Entry price for the trade
        stop_loss_price: Stop loss price for the trade
        risk_per_trade: Risk percentage per trade (default 2%)
        strategy: Position sizing strategy to use
        strategy_params: Additional parameters for specific strategies

    Returns:
        Dictionary containing position size details:
            - position_size: Number of units/shares
            - position_value: Total value of position
            - dollar_risk: Dollar amount at risk
            - risk_percentage: Actual risk percentage
            - stop_distance: Distance to stop loss in percentage

    Raises:
        ValueError: If parameters are invalid
    """
    if account_balance <= 0:
        raise ValueError("Account balance must be positive")

    if entry_price <= 0:
        raise ValueError("Entry price must be positive")

    if stop_loss_price <= 0:
        raise ValueError("Stop loss price must be positive")

    strategy_params = strategy_params or {}

    # Calculate stop loss distance
    if entry_price > stop_loss_price:  # Long position
        stop_distance = (entry_price - stop_loss_price) / entry_price
    else:  # Short position
        stop_distance = (stop_loss_price - entry_price) / entry_price

    if stop_distance <= 0:
        raise ValueError("Stop loss must be different from entry price")

    if stop_distance > 0.5:  # Sanity check: stop > 50% is unusual
        logger.warning("Stop loss distance is very large: {:.2%}", stop_distance)

    # Calculate position size based on strategy
    if strategy == PositionSizingStrategy.FIXED_FRACTIONAL:
        result = _calculate_fixed_fractional(
            account_balance, entry_price, stop_distance, risk_per_trade
        )
    elif strategy == PositionSizingStrategy.KELLY_CRITERION:
        result = _calculate_kelly(
            account_balance, entry_price, stop_distance, risk_per_trade, strategy_params
        )
    elif strategy == PositionSizingStrategy.FIXED_DOLLAR:
        result = _calculate_fixed_dollar(
            account_balance, entry_price, stop_distance, strategy_params
        )
    elif strategy == PositionSizingStrategy.VOLATILITY_TARGET:
        result = _calculate_volatility_target(
            account_balance, entry_price, stop_distance, strategy_params
        )
    elif strategy == PositionSizingStrategy.RISK_PARITY:
        result = _calculate_risk_parity(
            account_balance, entry_price, stop_distance, strategy_params
        )
    else:
        raise ValueError(f"Unknown position sizing strategy: {strategy}")

    # Add common fields
    result["stop_distance"] = stop_distance
    result["risk_percentage"] = risk_per_trade

    logger.info(
        "Position size calculated: {} units at ${:.2f}, risking ${:.2f} ({:.2%} of portfolio)",
        result["position_size"],
        entry_price,
        result["dollar_risk"],
        result["risk_percentage"],
    )

    return result


def _calculate_fixed_fractional(
    account_balance: float,
    entry_price: float,
    stop_distance: float,
    risk_per_trade: float,
) -> Dict[str, float]:
    """Fixed fractional position sizing."""
    dollar_risk = account_balance * risk_per_trade
    position_size = dollar_risk / (entry_price * stop_distance)
    position_value = position_size * entry_price

    return {
        "position_size": position_size,
        "position_value": position_value,
        "dollar_risk": dollar_risk,
    }


def _calculate_kelly(
    account_balance: float,
    entry_price: float,
    stop_distance: float,
    risk_per_trade: float,
    params: Dict,
) -> Dict[str, float]:
    """Kelly Criterion position sizing.

    Kelly % = (Win% * AvgWin - Loss% * AvgLoss) / AvgWin

    Uses half-Kelly by default for safety.
    """
    win_rate = params.get("win_rate", 0.55)
    avg_win = params.get("avg_win", 1.5)  # Average win as multiple of risk
    avg_loss = params.get("avg_loss", 1.0)  # Average loss as multiple of risk
    kelly_fraction = params.get("kelly_fraction", 0.5)  # Use half-Kelly

    loss_rate = 1 - win_rate
    kelly_percentage = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
    kelly_percentage = max(0, kelly_percentage * kelly_fraction)  # Ensure non-negative

    # Cap at risk_per_trade maximum
    actual_risk = min(kelly_percentage, risk_per_trade)

    dollar_risk = account_balance * actual_risk
    position_size = dollar_risk / (entry_price * stop_distance)
    position_value = position_size * entry_price

    return {
        "position_size": position_size,
        "position_value": position_value,
        "dollar_risk": dollar_risk,
        "kelly_percentage": kelly_percentage,
    }


def _calculate_fixed_dollar(
    account_balance: float,
    entry_price: float,
    stop_distance: float,
    params: Dict,
) -> Dict[str, float]:
    """Fixed dollar amount position sizing."""
    fixed_dollar_amount = params.get("fixed_amount", account_balance * 0.02)

    dollar_risk = min(fixed_dollar_amount, account_balance)
    position_size = dollar_risk / (entry_price * stop_distance)
    position_value = position_size * entry_price

    return {
        "position_size": position_size,
        "position_value": position_value,
        "dollar_risk": dollar_risk,
    }


def _calculate_volatility_target(
    account_balance: float,
    entry_price: float,
    stop_distance: float,
    params: Dict,
) -> Dict[str, float]:
    """Volatility-targeted position sizing.

    Adjusts position size based on asset volatility.
    """
    volatility = params.get("volatility", 0.02)  # Daily volatility
    target_volatility = params.get("target_volatility", 0.01)  # Target portfolio volatility

    # Scale position by volatility ratio
    vol_scalar = target_volatility / volatility if volatility > 0 else 1.0
    vol_scalar = min(vol_scalar, 2.0)  # Cap at 2x for safety

    dollar_risk = account_balance * 0.02 * vol_scalar
    position_size = dollar_risk / (entry_price * stop_distance)
    position_value = position_size * entry_price

    return {
        "position_size": position_size,
        "position_value": position_value,
        "dollar_risk": dollar_risk,
        "volatility_scalar": vol_scalar,
    }


def _calculate_risk_parity(
    account_balance: float,
    entry_price: float,
    stop_distance: float,
    params: Dict,
) -> Dict[str, float]:
    """Risk parity position sizing.

    Allocates capital based on equal risk contribution.
    """
    num_positions = params.get("num_positions", 10)
    risk_per_position = 1.0 / num_positions

    dollar_risk = account_balance * risk_per_position
    position_size = dollar_risk / (entry_price * stop_distance)
    position_value = position_size * entry_price

    return {
        "position_size": position_size,
        "position_value": position_value,
        "dollar_risk": dollar_risk,
        "risk_per_position": risk_per_position,
    }


def calculate_portfolio_risk(
    positions: List[Dict[str, Union[str, float]]],
    prices: pd.DataFrame,
    confidence_level: float = 0.95,
    method: str = "historical",
) -> Dict[str, float]:
    """Calculate portfolio-level risk metrics including VaR and CVaR.

    Args:
        positions: List of position dicts with 'symbol', 'quantity', 'entry_price'
        prices: DataFrame of historical prices (datetime index, symbols as columns)
        confidence_level: Confidence level for VaR calculation (default 95%)
        method: VaR calculation method ('historical', 'parametric', or 'monte_carlo')

    Returns:
        Dictionary containing:
            - portfolio_value: Total portfolio value
            - var_95: Value at Risk at confidence level
            - cvar_95: Conditional VaR (Expected Shortfall)
            - portfolio_std: Portfolio standard deviation
            - worst_case_loss: Maximum historical loss
            - beta: Portfolio beta (if benchmark provided)
    """
    if not positions:
        return {
            "portfolio_value": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "portfolio_std": 0.0,
            "worst_case_loss": 0.0,
        }

    # Calculate position weights
    portfolio_value = sum(
        pos["quantity"] * pos.get("current_price", pos["entry_price"]) for pos in positions
    )

    if portfolio_value == 0:
        return {
            "portfolio_value": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "portfolio_std": 0.0,
            "worst_case_loss": 0.0,
        }

    weights = pd.Series(
        {
            pos["symbol"]: (
                pos["quantity"] * pos.get("current_price", pos["entry_price"]) / portfolio_value
            )
            for pos in positions
        }
    )

    # Calculate returns
    returns = prices.pct_change().dropna()

    if method == "historical":
        var_result = _calculate_historical_var(returns, weights, confidence_level)
    elif method == "parametric":
        var_result = _calculate_parametric_var(returns, weights, confidence_level)
    elif method == "monte_carlo":
        var_result = _calculate_monte_carlo_var(returns, weights, confidence_level)
    else:
        raise ValueError(f"Unknown VaR method: {method}")

    # Calculate worst case loss from historical data
    portfolio_returns = (returns * weights).sum(axis=1)
    worst_case_loss = portfolio_returns.min() * portfolio_value

    result = {
        "portfolio_value": portfolio_value,
        "worst_case_loss": abs(worst_case_loss),
        **var_result,
    }

    logger.info(
        "Portfolio risk: Value=${:.2f}, VaR({:.0%})=${:.2f}, CVaR=${:.2f}",
        result["portfolio_value"],
        confidence_level,
        result["var_95"],
        result["cvar_95"],
    )

    return result


def _calculate_historical_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float,
) -> Dict[str, float]:
    """Calculate historical VaR."""
    portfolio_value = 1.0  # Normalized
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_value_normalized = portfolio_value

    # Calculate VaR
    var_level = 1 - confidence_level
    var = np.percentile(portfolio_returns, var_level * 100) * portfolio_value_normalized

    # Calculate CVaR (Expected Shortfall)
    cvar = portfolio_returns[portfolio_returns <= var].mean() * portfolio_value_normalized

    # Calculate standard deviation
    portfolio_std = portfolio_returns.std()

    return {
        "var_95": abs(var),
        "cvar_95": abs(cvar),
        "portfolio_std": portfolio_std,
    }


def _calculate_parametric_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float,
) -> Dict[str, float]:
    """Calculate parametric VaR using covariance matrix."""
    # Calculate covariance matrix
    cov_matrix = returns.cov()

    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)

    # Calculate VaR using z-score
    from scipy.stats import norm

    z_score = norm.ppf(1 - confidence_level)
    var = z_score * portfolio_std

    # Calculate CVaR
    cvar = -portfolio_std * norm.pdf(z_score) / (1 - confidence_level)

    return {
        "var_95": abs(var),
        "cvar_95": abs(cvar),
        "portfolio_std": portfolio_std,
    }


def _calculate_monte_carlo_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence_level: float,
    num_simulations: int = 10000,
) -> Dict[str, float]:
    """Calculate Monte Carlo VaR."""
    cov_matrix = returns.cov()
    mean_returns = returns.mean()

    # Generate random scenarios
    np.random.seed(42)
    random_shocks = np.random.multivariate_normal(
        mean_returns.values, cov_matrix.values, num_simulations
    )

    # Calculate portfolio returns for each scenario
    portfolio_returns = random_shocks @ weights.values

    # Calculate VaR
    var_level = 1 - confidence_level
    var = np.percentile(portfolio_returns, var_level * 100)

    # Calculate CVaR
    cvar = portfolio_returns[portfolio_returns <= var].mean()

    # Calculate standard deviation
    portfolio_std = portfolio_returns.std()

    return {
        "var_95": abs(var),
        "cvar_95": abs(cvar),
        "portfolio_std": portfolio_std,
    }


def calculate_correlation_matrix(
    prices: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 10,
) -> pd.DataFrame:
    """Calculate correlation matrix for assets.

    Args:
        prices: DataFrame of historical prices (datetime index, symbols as columns)
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        min_periods: Minimum number of observations required

    Returns:
        Correlation matrix as DataFrame
    """
    if prices.empty:
        return pd.DataFrame()

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Calculate correlation matrix
    corr_matrix = returns.corr(method=method, min_periods=min_periods)

    # Replace NaN with 0 for assets with insufficient data
    corr_matrix = corr_matrix.fillna(0)

    logger.debug(
        "Calculated correlation matrix for {} assets using {} method",
        len(corr_matrix),
        method,
    )

    return corr_matrix


def calculate_max_drawdown(
    prices: pd.Series,
    method: str = "historical",
) -> Dict[str, Union[float, Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Calculate maximum drawdown and related metrics.

    Args:
        prices: Series of prices or portfolio values (datetime index)
        method: Calculation method ('historical' or 'theoretical')

    Returns:
        Dictionary containing:
            - max_drawdown: Maximum drawdown as percentage
            - max_drawdown_abs: Maximum drawdown in absolute terms
            - max_drawdown_duration: Duration of max drawdown in days
            - peak_date: Date of peak before max drawdown
            - trough_date: Date of trough (bottom of drawdown)
            - recovery_date: Date of recovery (or NaT if not recovered)
            - current_drawdown: Current drawdown percentage
            - avg_drawdown: Average drawdown percentage
            - drawdowns: Series of all drawdowns
    """
    if prices.empty or len(prices) < 2:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_abs": 0.0,
            "max_drawdown_duration": 0,
            "peak_date": pd.NaT,
            "trough_date": pd.NaT,
            "recovery_date": pd.NaT,
            "current_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "drawdowns": pd.Series(dtype=float),
        }

    # Calculate cumulative returns
    cumulative = (1 + prices.pct_change()).cumprod()

    # Calculate running maximum (peak)
    running_max = cumulative.expanding().max()

    # Calculate drawdowns
    drawdowns = (cumulative - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdowns.min()
    max_dd_abs = (cumulative - running_max).min()

    # Find peak and trough dates
    trough_idx = drawdowns.idxmin()
    peak_idx = cumulative[:trough_idx].idxmax()

    # Find recovery date (if any)
    recovery_date = pd.NaT
    if max_dd < 0:
        recovery_mask = cumulative[trough_idx:] >= cumulative[peak_idx]
        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()

    # Calculate drawdown duration
    if not pd.isna(recovery_date):
        max_dd_duration = (recovery_date - peak_idx).days
    else:
        max_dd_duration = len(cumulative) - peak_idx  # Still in drawdown

    # Current drawdown
    current_dd = drawdowns.iloc[-1]

    # Average drawdown (only negative drawdowns)
    avg_dd = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0

    result = {
        "max_drawdown": abs(max_dd),
        "max_drawdown_abs": abs(max_dd_abs),
        "max_drawdown_duration": max_dd_duration,
        "peak_date": peak_idx,
        "trough_date": trough_idx,
        "recovery_date": recovery_date,
        "current_drawdown": abs(current_dd) if current_dd < 0 else 0.0,
        "avg_drawdown": abs(avg_dd),
        "drawdowns": drawdowns,
    }

    logger.info(
        "Max drawdown: {:.2%}, Duration: {} days, Current: {:.2%}",
        result["max_drawdown"],
        result["max_drawdown_duration"],
        result["current_drawdown"],
    )

    return result


def calculate_portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate portfolio beta relative to benchmark.

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns

    Returns:
        Beta coefficient
    """
    # Align data
    aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()

    if len(aligned_data) < 2:
        return 1.0  # Default beta

    # Calculate covariance and variance
    covariance = aligned_data.iloc[:, 0].cov(aligned_data.iloc[:, 1])
    variance = aligned_data.iloc[:, 1].var()

    if variance == 0:
        return 1.0

    beta = covariance / variance
    return beta


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily, 52 for weekly)

    Returns:
        Sharpe ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - risk_free_rate / periods_per_year

    # Calculate Sharpe ratio
    mean_excess_return = excess_returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    sharpe = mean_excess_return / std_return * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """Calculate Sortino ratio (downside risk-adjusted return).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        target_return: Target/minimum acceptable return

    Returns:
        Sortino ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - risk_free_rate / periods_per_year

    # Calculate downside deviation (only negative returns relative to target)
    downside_returns = returns[returns < target_return] - target_return
    downside_deviation = downside_returns.std()

    if downside_deviation == 0:
        return 0.0 if excess_returns.mean() <= 0 else float("inf")

    # Calculate Sortino ratio
    mean_excess_return = excess_returns.mean()
    sortino = mean_excess_return / downside_deviation * np.sqrt(periods_per_year)

    return sortino
