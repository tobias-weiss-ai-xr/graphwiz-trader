"""Advanced backtesting framework using Qlib methodology.

This module provides comprehensive backtesting capabilities including:
- Strategy backtesting
- Performance metrics calculation
- Model validation and selection
- Risk analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from loguru import logger

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("Scipy not available. Some metrics will be limited.")

from .config import QlibConfig


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    profit_loss: Optional[float]
    return_pct: Optional[float]


@dataclass
class BacktestResult:
    """Results from backtesting."""

    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float  # Compound Annual Growth Rate

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # Total wins / Total losses

    # Trade-level metrics
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Advanced metrics
    information_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    hit_rate: Optional[float] = None


class BacktestEngine:
    """
    Advanced backtesting engine inspired by Qlib.

    Provides comprehensive backtesting with detailed performance metrics
    and analysis capabilities.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.returns: pd.Series = pd.Series(dtype=float)

        logger.info("Backtest engine initialized")

    def run_backtest(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run backtest on signals.

        Args:
            signals: DataFrame with signals (1=buy, 0=sell/hold)
            price_data: DataFrame with price data (must have 'close' column)
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info("Running backtest...")

        # Initialize
        capital = self.config.initial_capital
        position = 0.0
        trades = []
        equity_values = []
        returns_list = []
        entry_time = None
        entry_price = None

        # Align signals and price data
        aligned_data = pd.concat([signals, price_data], axis=1).join(
            pd.concat([signals, price_data], axis=1), how="inner"
        )

        # Ensure we have the signal column
        if "signal" in aligned_data.columns:
            signal_col = "signal"
        else:
            # Use first column as signal
            signal_col = aligned_data.columns[0]

        # Iterate through data
        for i, (timestamp, row) in enumerate(aligned_data.iterrows()):
            close_price = row["close"]
            signal = row[signal_col]

            # Apply slippage
            execution_price = (
                close_price * (1 - self.config.slippage) if position == 0 else close_price
            )

            if signal == 1 and position == 0:
                # Enter long position
                entry_time = timestamp
                entry_price = execution_price
                commission_cost = capital * self.config.commission

                position = (capital - commission_cost) / entry_price
                capital = 0

            elif signal == 0 and position > 0:
                # Exit position
                commission_cost = position * execution_price * self.config.commission
                capital = position * execution_price - commission_cost

                # Record trade
                profit_loss = capital - self.config.initial_capital
                return_pct = (execution_price - entry_price) / entry_price

                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        symbol="BTC/USDT",  # Can be made configurable
                        side="buy",
                        entry_price=entry_price,
                        exit_price=execution_price,
                        quantity=position,
                        profit_loss=profit_loss,
                        return_pct=return_pct,
                    )
                )

                position = 0
                entry_price = None

            # Calculate current equity
            current_equity = capital + (position * execution_price if position > 0 else 0)
            equity_values.append(current_equity)

            # Calculate returns
            if len(equity_values) > 1:
                returns_list.append((current_equity - equity_values[-2]) / equity_values[-2])
            else:
                returns_list.append(0.0)

        # Close any remaining position
        if position > 0 and entry_price is not None:
            final_price = aligned_data["close"].iloc[-1]
            capital = position * final_price
            profit_loss = capital - self.config.initial_capital
            return_pct = (final_price - entry_price) / entry_price

            trades.append(
                Trade(
                    entry_time=entry_time,
                    exit_time=aligned_data.index[-1],
                    symbol="BTC/USDT",
                    side="buy",
                    entry_price=entry_price,
                    exit_price=final_price,
                    quantity=position,
                    profit_loss=profit_loss,
                    return_pct=return_pct,
                )
            )

        # Create equity curve and returns series
        self.equity_curve = pd.Series(equity_values, index=aligned_data.index)
        self.returns = pd.Series(returns_list, index=aligned_data.index)
        self.trades = trades

        # Calculate metrics
        result = self._calculate_metrics(benchmark_returns)

        logger.info(f"Backtest complete. Total return: {result.total_return:.2%}")

        return result

    def _calculate_metrics(
        self,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics."""

        if len(self.equity_curve) == 0:
            raise ValueError("No equity curve data. Run backtest first.")

        # Basic return metrics
        final_equity = self.equity_curve.iloc[-1]
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital

        # Annualized return
        days = len(self.equity_curve)
        years = days / 252  # Assuming 252 trading days per year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # CAGR
        cagr = (final_equity / self.config.initial_capital) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = self.returns.std() * np.sqrt(252) if len(self.returns) > 1 else 0

        # Sharpe Ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino Ratio
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        )
        sortino_ratio = (
            (annualized_return - risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else 0
        )

        # Maximum drawdown
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()

        # Max drawdown duration (in days)
        drawdown_duration = 0
        max_drawdown_duration = 0
        for dd in drawdown:
            if dd < 0:
                drawdown_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
            else:
                drawdown_duration = 0

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade metrics
        total_trades = len(self.trades)
        if total_trades > 0:
            winning_trades = sum(1 for t in self.trades if t.profit_loss and t.profit_loss > 0)
            losing_trades = sum(1 for t in self.trades if t.profit_loss and t.profit_loss < 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            wins = [t.profit_loss for t in self.trades if t.profit_loss and t.profit_loss > 0]
            losses = [t.profit_loss for t in self.trades if t.profit_loss and t.profit_loss < 0]

            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0

            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Advanced metrics (if benchmark provided)
        information_ratio = None
        beta = None
        alpha = None

        if benchmark_returns is not None and SCIPY_AVAILABLE:
            # Align returns
            aligned_returns = self.returns.align(benchmark_returns, join="inner")
            if len(aligned_returns[0]) > 1:
                excess_returns = aligned_returns[0] - aligned_returns[1]
                information_ratio = (
                    excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                    if excess_returns.std() > 0
                    else 0
                )

                # Beta and Alpha (CAPM)
                covariance = np.cov(aligned_returns[0].dropna(), aligned_returns[1].dropna())[0][1]
                benchmark_variance = aligned_returns[1].var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = annualized_return - (
                    risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate)
                )

        # Hit rate (same as win_rate)
        hit_rate = win_rate

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades,
            equity_curve=self.equity_curve,
            returns=self.returns,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha,
            hit_rate=hit_rate,
        )

    def generate_report(self, result: BacktestResult) -> str:
        """Generate human-readable backtest report."""
        report = []
        report.append("=" * 80)
        report.append("BACKTEST REPORT")
        report.append("=" * 80)
        report.append("")

        # Return Metrics
        report.append("RETURN METRICS")
        report.append("-" * 40)
        report.append(f"Total Return:         {result.total_return:>10.2%}")
        report.append(f"Annualized Return:    {result.annualized_return:>10.2%}")
        report.append(f"CAGR:                 {result.cagr:>10.2%}")
        report.append("")

        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"Volatility:           {result.volatility:>10.2%}")
        report.append(f"Sharpe Ratio:         {result.sharpe_ratio:>10.2f}")
        report.append(f"Sortino Ratio:        {result.sortino_ratio:>10.2f}")
        report.append(f"Calmar Ratio:         {result.calmar_ratio:>10.2f}")
        report.append("")

        # Drawdown Metrics
        report.append("DRAWDOWN METRICS")
        report.append("-" * 40)
        report.append(f"Max Drawdown:         {result.max_drawdown:>10.2%}")
        report.append(f"Avg Drawdown:         {result.avg_drawdown:>10.2%}")
        report.append(f"Max DD Duration:      {result.max_drawdown_duration:>10} days")
        report.append("")

        # Trade Metrics
        report.append("TRADE METRICS")
        report.append("-" * 40)
        report.append(f"Total Trades:         {result.total_trades:>10}")
        report.append(f"Winning Trades:       {result.winning_trades:>10}")
        report.append(f"Losing Trades:        {result.losing_trades:>10}")
        report.append(f"Win Rate:             {result.win_rate:>10.2%}")
        report.append(f"Avg Win:              ${result.avg_win:>10.2f}")
        report.append(f"Avg Loss:             ${result.avg_loss:>10.2f}")
        report.append(f"Profit Factor:        {result.profit_factor:>10.2f}")
        report.append("")

        # Advanced Metrics (if available)
        if result.information_ratio is not None:
            report.append("ADVANCED METRICS")
            report.append("-" * 40)
            report.append(f"Information Ratio:    {result.information_ratio:>10.2f}")
            report.append(f"Beta:                 {result.beta:>10.2f}")
            report.append(f"Alpha:                {result.alpha:>10.2%}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)


class ModelValidator:
    """
    Model validation and selection using cross-validation and multiple metrics.

    Helps select the best model and validate performance.
    """

    def __init__(self, n_folds: int = 5):
        """
        Initialize model validator.

        Args:
            n_folds: Number of cross-validation folds
        """
        self.n_folds = n_folds

    def cross_validate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Args:
            model: Model to validate (must have fit/predict methods)
            X: Features
            y: Labels
            metric: Metric to optimize ('accuracy', 'sharpe', etc.)

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Running {self.n_folds}-fold cross-validation...")

        fold_size = len(X) // self.n_folds
        scores = []

        for fold in range(self.n_folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.n_folds - 1 else len(X)

            X_val = X.iloc[start_idx:end_idx]
            y_val = y.iloc[start_idx:end_idx]

            X_train = pd.concat([X.iloc[:start_idx], X.iloc[end_idx:]])
            y_train = pd.concat([y.iloc[:start_idx], y.iloc[end_idx:]])

            # Train model
            model.fit(X_train, y_train)

            # Predict
            predictions = model.predict(X_val)

            # Calculate score
            if metric == "accuracy":
                score = np.mean(predictions == y_val)
            else:
                score = np.mean(predictions == y_val)  # Default to accuracy

            scores.append(score)
            logger.debug(f"Fold {fold + 1}: {score:.4f}")

        results = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "scores": scores,
        }

        logger.info(f"CV Results: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

        return results

    def select_best_model(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select best model from candidates.

        Args:
            models: Dictionary of model_name -> model
            X: Features
            y: Labels

        Returns:
            Tuple of (best_model_name, validation_results)
        """
        logger.info(f"Selecting best model from {len(models)} candidates...")

        best_model = None
        best_score = -np.inf
        best_results = None

        for model_name, model in models.items():
            try:
                results = self.cross_validate(model, X, y)
                score = results["mean_score"]

                logger.info(f"{model_name}: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = model_name
                    best_results = results

            except Exception as e:
                logger.error(f"Error validating {model_name}: {e}")

        logger.info(f"Best model: {best_model} (score: {best_score:.4f})")

        return best_model, best_results


def create_backtest_engine(
    initial_capital: float = 100000.0,
    commission: float = 0.001,
) -> BacktestEngine:
    """
    Convenience function to create backtest engine.

    Args:
        initial_capital: Starting capital
        commission: Commission rate

    Returns:
        BacktestEngine instance
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission=commission,
    )
    return BacktestEngine(config=config)
