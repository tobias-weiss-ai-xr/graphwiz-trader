"""Portfolio optimization using Qlib framework.

This module implements advanced portfolio optimization strategies including:
- Mean-variance optimization
- Risk parity
- Black-Litterman model
- Dynamic position sizing
- Risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    from scipy.optimize import minimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("Scipy not available. Portfolio optimization will be limited.")

from .config import QlibConfig


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio optimization."""

    max_position_weight: float = 0.3  # Maximum weight for single asset
    min_position_weight: float = 0.0  # Minimum weight for single asset
    max_leverage: float = 1.0  # Maximum portfolio leverage
    target_volatility: Optional[float] = None  # Target portfolio volatility
    max_drawdown: Optional[float] = None  # Maximum acceptable drawdown
    turnover_limit: Optional[float] = None  # Maximum portfolio turnover


@dataclass
class OptimizerConfig:
    """Configuration for portfolio optimizer."""

    optimization_method: str = "mean_variance"  # mean_variance, risk_parity, equal_weight
    risk_free_rate: float = 0.02  # Annual risk-free rate
    rebalance_frequency: str = "1d"  # Rebalancing frequency
    lookback_window: int = 60  # Days of historical data for optimization
    min_returns: float = 0.0  # Minimum acceptable return
    max_risk: Optional[float] = None  # Maximum acceptable risk


class PortfolioOptimizer:
    """
    Portfolio optimizer using Qlib-inspired methods.

    This class implements various portfolio optimization strategies
    to determine optimal asset allocation based on expected returns,
    risk estimates, and constraints.
    """

    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
        constraints: Optional[PortfolioConstraints] = None,
    ):
        """
        Initialize portfolio optimizer.

        Args:
            config: Optimizer configuration
            constraints: Portfolio constraints
        """
        self.config = config or OptimizerConfig()
        self.constraints = constraints or PortfolioConstraints()

        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available. Using simplified optimization methods.")

        logger.info(f"Portfolio optimizer initialized with method: {self.config.optimization_method}")

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        covariance_matrix: Optional[pd.DataFrame] = None,
        current_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Optimize portfolio weights.

        Args:
            returns: Historical returns matrix (assets x time)
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            current_weights: Current portfolio weights

        Returns:
            Optimal weights for each asset
        """
        logger.info(f"Running portfolio optimization: {self.config.optimization_method}")

        # Calculate covariance matrix if not provided
        if covariance_matrix is None:
            covariance_matrix = returns.cov() * 252  # Annualized

        # Calculate expected returns if not provided
        if expected_returns is None:
            expected_returns = returns.mean() * 252  # Annualized

        # Run optimization based on method
        if self.config.optimization_method == "mean_variance":
            weights = self._mean_variance_optimization(
                expected_returns,
                covariance_matrix,
                current_weights,
            )
        elif self.config.optimization_method == "risk_parity":
            weights = self._risk_parity_optimization(covariance_matrix, current_weights)
        elif self.config.optimization_method == "equal_weight":
            weights = self._equal_weight(returns.columns)
        elif self.config.optimization_method == "max_sharpe":
            weights = self._max_sharpe_ratio(
                expected_returns,
                covariance_matrix,
                current_weights,
            )
        elif self.config.optimization_method == "min_variance":
            weights = self._minimum_variance(covariance_matrix, current_weights)
        elif self.config.optimization_method == "black_litterman":
            weights = self._black_litterman(
                returns,
                covariance_matrix,
                current_weights,
            )
        else:
            logger.warning(f"Unknown method: {self.config.optimization_method}. Using equal weight.")
            weights = self._equal_weight(returns.columns)

        # Apply constraints
        weights = self._apply_constraints(weights)

        logger.info(f"Optimization complete. Top holdings: {weights.nlargest(3).to_dict()}")

        return weights

    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Mean-variance optimization (Markowitz).

        Maximizes: μ'w - γ * w'Σw
        Where μ is expected returns, Σ is covariance matrix, γ is risk aversion
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available. Using equal weight.")
            return self._equal_weight(expected_returns.index)

        n_assets = len(expected_returns)

        # Objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            # Risk aversion parameter (can be made configurable)
            gamma = 1.0
            return -(portfolio_return - gamma * portfolio_variance)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]

        # Bounds
        bounds = [
            (self.constraints.min_position_weight, self.constraints.max_position_weight)
            for _ in range(n_assets)
        ]

        # Initial guess (equal weight or current weights)
        x0 = (
            current_weights.values
            if current_weights is not None
            else np.array([1.0 / n_assets] * n_assets)
        )

        # Optimize
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9},
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return self._equal_weight(expected_returns.index)

        weights = pd.Series(result.x, index=expected_returns.index)

        return weights

    def _max_sharpe_ratio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Maximize Sharpe ratio optimization.

        Maximizes: (μ'w - rf) / sqrt(w'Σw)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available. Using equal weight.")
            return self._equal_weight(expected_returns.index)

        n_assets = len(expected_returns)
        risk_free_rate = self.config.risk_free_rate

        # Objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe_ratio

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]

        # Bounds
        bounds = [
            (self.constraints.min_position_weight, self.constraints.max_position_weight)
            for _ in range(n_assets)
        ]

        # Initial guess
        x0 = (
            current_weights.values
            if current_weights is not None
            else np.array([1.0 / n_assets] * n_assets)
        )

        # Optimize
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9},
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return self._equal_weight(expected_returns.index)

        weights = pd.Series(result.x, index=expected_returns.index)

        return weights

    def _minimum_variance(
        self,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Minimum variance optimization.

        Minimizes: w'Σw
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available. Using equal weight.")
            return self._equal_weight(covariance_matrix.index)

        n_assets = len(covariance_matrix)

        # Objective function (portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]

        # Bounds
        bounds = [
            (self.constraints.min_position_weight, self.constraints.max_position_weight)
            for _ in range(n_assets)
        ]

        # Initial guess
        x0 = (
            current_weights.values
            if current_weights is not None
            else np.array([1.0 / n_assets] * n_assets)
        )

        # Optimize
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9},
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return self._equal_weight(covariance_matrix.index)

        weights = pd.Series(result.x, index=covariance_matrix.index)

        return weights

    def _risk_parity_optimization(
        self,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Risk parity optimization.

        Allocates weights such that each asset contributes equal risk to portfolio.
        """
        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available. Using equal weight.")
            return self._equal_weight(covariance_matrix.index)

        n_assets = len(covariance_matrix)

        # Objective function (minimize sum of squared risk contribution differences)
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            marginal_contrib = np.dot(covariance_matrix, weights)
            contrib = weights * marginal_contrib
            risk_contrib = contrib / portfolio_variance
            target_risk = 1.0 / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]

        # Bounds
        bounds = [
            (self.constraints.min_position_weight, self.constraints.max_position_weight)
            for _ in range(n_assets)
        ]

        # Initial guess
        x0 = (
            current_weights.values
            if current_weights is not None
            else np.array([1.0 / n_assets] * n_assets)
        )

        # Optimize
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9},
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return self._equal_weight(covariance_matrix.index)

        weights = pd.Series(result.x, index=covariance_matrix.index)

        return weights

    def _black_litterman(
        self,
        returns: pd.DataFrame,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Black-Litterman portfolio optimization.

        Combines market equilibrium with investor views.
        """
        # Market equilibrium weights (use market cap if available, else equal weight)
        market_weights = self._equal_weight(returns.columns)

        # Risk aversion parameter
        risk_aversion = 3.0  # Can be made configurable

        # Implied equilibrium returns
        implied_returns = risk_aversion * np.dot(covariance_matrix, market_weights)

        # For now, use equilibrium returns (can add views later)
        expected_returns = pd.Series(implied_returns, index=returns.columns)

        # Use mean-variance with equilibrium returns
        return self._mean_variance_optimization(expected_returns, covariance_matrix, current_weights)

    def _equal_weight(self, assets: pd.Index) -> pd.Series:
        """Equal weight allocation."""
        n = len(assets)
        return pd.Series(1.0 / n, index=assets)

    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """Apply portfolio constraints to weights."""
        # Ensure weights sum to 1
        weights = weights / weights.sum()

        # Apply max weight constraint
        weights = weights.clip(upper=self.constraints.max_position_weight)

        # Renormalize
        weights = weights / weights.sum()

        # Apply min weight constraint
        weights[weights < self.constraints.min_position_weight] = 0

        # Renormalize again
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # If all weights are zero, use equal weight
            weights = pd.Series(1.0 / len(weights), index=weights.index)

        return weights

    def calculate_portfolio_metrics(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics.

        Args:
            weights: Portfolio weights
            returns: Asset returns

        Returns:
            Dictionary of portfolio metrics
        """
        # Portfolio returns
        portfolio_returns = returns.dot(weights)

        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility

        # Maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Value at Risk (95%)
        var_95 = portfolio_returns.quantile(0.05)

        # Conditional VaR (expected shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'portfolio_returns': portfolio_returns,
        }

        return metrics


class DynamicPositionSizer:
    """
    Dynamic position sizing based on model confidence and risk.

    Adjusts position sizes based on:
    - Model prediction confidence
    - Portfolio volatility
    - Risk limits
    - Market conditions
    """

    def __init__(
        self,
        base_position_size: float = 0.1,
        max_position_size: float = 0.3,
        min_position_size: float = 0.05,
        risk_tolerance: float = 0.02,  # 2% portfolio risk per trade
    ):
        """
        Initialize dynamic position sizer.

        Args:
            base_position_size: Base position as fraction of portfolio
            max_position_size: Maximum position size
            min_position_size: Minimum position size
            risk_tolerance: Portfolio risk per trade (fraction)
        """
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.risk_tolerance = risk_tolerance

    def calculate_position_size(
        self,
        signal_confidence: float,
        portfolio_value: float,
        asset_price: float,
        asset_volatility: float,
    ) -> float:
        """
        Calculate position size based on confidence and risk.

        Args:
            signal_confidence: Model confidence (0-1)
            portfolio_value: Total portfolio value
            asset_price: Current asset price
            asset_volatility: Asset volatility (annualized)

        Returns:
            Position size in base currency
        """
        # Adjust position size based on confidence
        confidence_multiplier = 0.5 + (signal_confidence * 0.5)  # 0.5 to 1.0

        # Calculate base position
        base_position = portfolio_value * self.base_position_size * confidence_multiplier

        # Risk-based position sizing (Kelly Criterion inspired)
        # Position size = (confidence * expected_return) / (asset_volatility^2)
        # Simplified: Scale by risk tolerance and volatility
        risk_adjusted_size = (self.risk_tolerance * portfolio_value) / (asset_volatility * np.sqrt(252))

        # Take minimum of base and risk-adjusted
        position_size = min(base_position, risk_adjusted_size)

        # Apply limits
        position_size = max(self.min_position_size * portfolio_value, position_size)
        position_size = min(self.max_position_size * portfolio_value, position_size)

        return position_size

    def calculate_position_size_by_weight(
        self,
        optimal_weight: float,
        portfolio_value: float,
        asset_price: float,
    ) -> float:
        """
        Calculate position size based on optimal portfolio weight.

        Args:
            optimal_weight: Optimal weight from portfolio optimization
            portfolio_value: Total portfolio value
            asset_price: Current asset price

        Returns:
            Position size in base currency
        """
        position_value = optimal_weight * portfolio_value

        # Apply limits
        position_value = max(self.min_position_size * portfolio_value, position_value)
        position_value = min(self.max_position_size * portfolio_value, position_value)

        return position_value


def create_portfolio_optimizer(
    method: str = "mean_variance",
    constraints: Optional[PortfolioConstraints] = None,
) -> PortfolioOptimizer:
    """
    Convenience function to create portfolio optimizer.

    Args:
        method: Optimization method
        constraints: Portfolio constraints

    Returns:
        PortfolioOptimizer instance
    """
    config = OptimizerConfig(optimization_method=method)
    return PortfolioOptimizer(config=config, constraints=constraints)
