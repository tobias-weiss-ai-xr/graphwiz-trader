"""
Performance analysis module for backtesting results.

This module provides comprehensive performance metrics:
- Returns analysis
- Risk metrics (Sharpe ratio, Sortino ratio, max drawdown)
- Trade analysis
- Equity curves
- Strategy comparison
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    total_trades: int
    profitable_trades: int
    losing_trades: int
    avg_hold_time: float
    calmar_ratio: float
    omega_ratio: float


class PerformanceAnalyzer:
    """
    Analyze backtesting performance and generate metrics.

    Provides comprehensive analysis including returns, risk metrics,
    trade statistics, and visualizations.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(
        self,
        equity_curve: pd.Series,
    ) -> pd.Series:
        """
        Calculate returns from equity curve.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Series of returns
        """
        return equity_curve.pct_change().fillna(0)

    def calculate_cumulative_returns(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Calculate cumulative returns.

        Args:
            returns: Series of returns

        Returns:
            Series of cumulative returns
        """
        return (1 + returns).cumprod() - 1

    def calculate_volatility(
        self,
        returns: pd.Series,
        annualize: bool = True,
    ) -> float:
        """
        Calculate volatility (standard deviation of returns).

        Args:
            returns: Series of returns
            annualize: Whether to annualize volatility

        Returns:
            Volatility
        """
        vol = returns.std()

        if annualize:
            # Assuming daily returns, annualize with sqrt(252)
            vol = vol * np.sqrt(252)

        return float(vol)

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        annualize: bool = True,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            annualize: Whether to use annualized values

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - self.risk_free_rate / 252

        if annualize:
            sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe = excess_returns.mean() / excess_returns.std()

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        annualize: bool = True,
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation).

        Args:
            returns: Series of returns
            annualize: Whether to use annualized values

        Returns:
            Sortino ratio
        """
        excess_returns = returns - self.risk_free_rate / 252

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std()

        if annualize:
            sortino = np.sqrt(252) * excess_returns.mean() / downside_deviation
        else:
            sortino = excess_returns.mean() / downside_deviation

        return float(sortino)

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Tuple of (max_drawdown, max_drawdown_duration)
        """
        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Find maximum drawdown
        max_drawdown = drawdown.min()

        # Find duration of max drawdown
        max_drawdown_idx = drawdown.idxmin()
        peak_idx = equity_curve[:max_drawdown_idx].idxmax()

        duration = len(equity_curve[peak_idx:max_drawdown_idx])

        return float(max_drawdown), duration

    def calculate_calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: float,
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            annualized_return: Annualized return
            max_drawdown: Maximum drawdown (as positive number)

        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        return float(annualized_return / abs(max_drawdown))

    def calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0.0,
    ) -> float:
        """
        Calculate Omega ratio.

        Args:
            returns: Series of returns
            threshold: Return threshold

        Returns:
            Omega ratio
        """
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())

        if losses == 0:
            return float("inf") if gains > 0 else 0.0

        return float(gains / losses)

    def analyze_trades(
        self,
        trades: List,
    ) -> Dict[str, float]:
        """
        Analyze completed trades.

        Args:
            trades: List of Trade objects

        Returns:
            Dictionary of trade statistics
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_return": 0.0,
                "avg_hold_time": 0.0,
                "profitable_trades": 0,
                "losing_trades": 0,
            }

        trade_returns = [trade.pnl_percentage for trade in trades]
        hold_times = [
            (trade.exit_time - trade.entry_time).total_seconds() / 3600 for trade in trades
        ]

        profitable_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        gross_profit = sum(t.pnl for t in profitable_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))

        return {
            "total_trades": len(trades),
            "win_rate": len(profitable_trades) / len(trades) if trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
            "avg_trade_return": np.mean(trade_returns) if trade_returns else 0.0,
            "avg_hold_time": np.mean(hold_times) if hold_times else 0.0,
            "profitable_trades": len(profitable_trades),
            "losing_trades": len(losing_trades),
        }

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: Optional[List] = None,
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: Series of portfolio values over time
            trades: List of completed trades

        Returns:
            PerformanceMetrics object
        """
        returns = self.calculate_returns(equity_curve)

        # Calculate duration in years
        duration_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25

        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized return
        annualized_return = (1 + total_return) ** (1 / duration_years) - 1

        # Risk metrics
        volatility = self.calculate_volatility(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        max_drawdown, max_drawdown_duration = self.calculate_max_drawdown(equity_curve)

        # Additional ratios
        calmar_ratio = self.calculate_calmar_ratio(annualized_return, max_drawdown)
        omega_ratio = self.calculate_omega_ratio(returns)

        # Trade analysis
        trade_stats = self.analyze_trades(trades) if trades else {}

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=trade_stats.get("win_rate", 0.0),
            profit_factor=trade_stats.get("profit_factor", 0.0),
            avg_trade_return=trade_stats.get("avg_trade_return", 0.0),
            total_trades=trade_stats.get("total_trades", 0),
            profitable_trades=trade_stats.get("profitable_trades", 0),
            losing_trades=trade_stats.get("losing_trades", 0),
            avg_hold_time=trade_stats.get("avg_hold_time", 0.0),
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
        )

    def generate_equity_curve(
        self,
        initial_capital: float,
        trades: List,
        price_data: pd.Series,
    ) -> pd.Series:
        """
        Generate equity curve from trades.

        Args:
            initial_capital: Starting capital
            trades: List of trades
            price_data: Price series for reference

        Returns:
            Series of portfolio values over time
        """
        # Create index from price data
        equity = pd.Series(index=price_data.index, dtype=float)
        equity.iloc[0] = initial_capital

        capital = initial_capital
        position = 0.0
        last_price = price_data.iloc[0]

        for timestamp, price in price_data.items():
            # Update position value based on current price
            position_value = position * price
            equity[timestamp] = capital + position_value

            # Check for trade execution at this timestamp
            for trade in trades:
                if trade.entry_time == timestamp and trade.side.value == "buy":
                    capital -= trade.entry_price * trade.quantity
                    position += trade.quantity
                    last_price = price
                elif trade.exit_time == timestamp and trade.side.value == "sell":
                    capital += trade.exit_price * trade.quantity
                    position -= trade.quantity
                    last_price = price

        return equity

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive equity curve plot.

        Args:
            equity_curve: Series of portfolio values over time
            save_path: Optional path to save HTML plot

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved equity curve plot to {save_path}")

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.Series,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create drawdown plot.

        Args:
            equity_curve: Series of portfolio values over time
            save_path: Optional path to save HTML plot

        Returns:
            Plotly figure object
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="red", width=1),
            )
        )

        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved drawdown plot to {save_path}")

        return fig

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create returns distribution plot.

        Args:
            returns: Series of returns
            save_path: Optional path to save HTML plot

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                name="Returns",
                marker_color="lightblue",
            )
        )

        # Add mean line
        mean_return = returns.mean() * 100
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_return:.3f}%",
        )

        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template="plotly_white",
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved returns distribution plot to {save_path}")

        return fig

    def compare_strategies(
        self,
        results: Dict[str, PerformanceMetrics],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create strategy comparison plot.

        Args:
            results: Dictionary mapping strategy names to metrics
            save_path: Optional path to save HTML plot

        Returns:
            Plotly figure object
        """
        metrics_to_compare = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
        ]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Total Return",
                "Sharpe Ratio",
                "Max Drawdown",
                "Win Rate",
            ),
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for metric, pos in zip(metrics_to_compare, positions):
            strategies = list(results.keys())
            values = [getattr(results[s], metric) for s in strategies]

            # Convert to percentages for display
            if metric in ["total_return", "max_drawdown", "win_rate"]:
                values = [v * 100 for v in values]

            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=values,
                    name=metric,
                    showlegend=False,
                ),
                row=pos[0],
                col=pos[1],
            )

        fig.update_layout(
            title_text="Strategy Comparison",
            template="plotly_white",
            height=600,
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved strategy comparison plot to {save_path}")

        return fig

    def generate_report(
        self,
        metrics: PerformanceMetrics,
        equity_curve: pd.Series,
        returns: pd.Series,
        output_path: str,
    ) -> None:
        """
        Generate comprehensive HTML report.

        Args:
            metrics: Performance metrics
            equity_curve: Portfolio value series
            returns: Returns series
            output_path: Path to save HTML report
        """
        # Create plots
        equity_fig = self.plot_equity_curve(equity_curve)
        drawdown_fig = self.plot_drawdown(equity_curve)
        returns_fig = self.plot_returns_distribution(returns)

        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtesting Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .metric {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .metric-label {{ font-size: 12px; color: #666; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #f44336; }}
                .plot {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Backtesting Performance Report</h1>

            <h2>Performance Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if metrics.total_return > 0 else 'negative'}">
                        {metrics.total_return * 100:.2f}%
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Annualized Return</div>
                    <div class="metric-value {'positive' if metrics.annualized_return > 0 else 'negative'}">
                        {metrics.annualized_return * 100:.2f}%
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value">{metrics.volatility * 100:.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{metrics.sharpe_ratio:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value">{metrics.sortino_ratio:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{metrics.max_drawdown * 100:.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown Duration</div>
                    <div class="metric-value">{metrics.max_drawdown_duration} periods</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{metrics.win_rate * 100:.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">{metrics.profit_factor:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Calmar Ratio</div>
                    <div class="metric-value">{metrics.calmar_ratio:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Omega Ratio</div>
                    <div class="metric-value">{metrics.omega_ratio:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{metrics.total_trades}</div>
                </div>
            </div>

            <h2>Equity Curve</h2>
            <div class="plot" id="equity-plot"></div>

            <h2>Drawdown</h2>
            <div class="plot" id="drawdown-plot"></div>

            <h2>Returns Distribution</h2>
            <div class="plot" id="returns-plot"></div>

            <script>
                Plotly.newPlot('equity-plot', {equity_fig.to_json()});
                Plotly.newPlot('drawdown-plot', {drawdown_fig.to_json()});
                Plotly.newPlot('returns-plot', {returns_fig.to_json()});
            </script>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Generated report at {output_path}")
