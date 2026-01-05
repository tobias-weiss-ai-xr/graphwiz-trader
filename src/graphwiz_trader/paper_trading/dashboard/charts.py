"""
Chart generation module for paper trading dashboard.

Creates interactive Plotly charts for equity curves, comparisons, and analytics.
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_equity_curve(
    equity_df: pd.DataFrame,
    title: str = "Portfolio Value",
    show_marker: bool = False,
) -> go.Figure:
    """Create equity curve line chart.

    Args:
        equity_df: DataFrame with timestamp and total_value columns
        title: Chart title
        show_marker: Whether to show markers on data points

    Returns:
        Plotly Figure object
    """
    if equity_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_df["timestamp"],
            y=equity_df["total_value"],
            mode="lines" + ("+markers" if show_marker else ""),
            name="Portfolio Value",
            line=dict(color="#2E86AB", width=2),
            hovertemplate="%{x}<br>Value: $%{y:,.2f}<extra></extra>",
        )
    )

    # Add initial capital reference line
    if "capital" in equity_df.columns:
        initial_capital = equity_df["capital"].iloc[0]
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"Initial: ${initial_capital:,.2f}",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_comparison(
    equity_dict: dict[str, pd.DataFrame],
    normalize: bool = True,
    title: str = "Symbol Comparison",
) -> go.Figure:
    """Create comparison chart for multiple symbols.

    Args:
        equity_dict: Dictionary mapping symbol to DataFrame
        normalize: Whether to normalize to starting value (100%)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4B51"]

    for i, (symbol, df) in enumerate(equity_dict.items()):
        if df.empty or "total_value" not in df.columns:
            continue

        x = df["timestamp"]
        if normalize:
            # Normalize to starting value (100%)
            y = (df["total_value"] / df["total_value"].iloc[0]) * 100
            name = f"{symbol} (Normalized)"
            yaxis_title = "Normalized Value (%)"
        else:
            y = df["total_value"]
            name = symbol
            yaxis_title = "Portfolio Value ($)"

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=f"%{{x}}<br>{symbol}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=yaxis_title if not normalize else "Normalized Value (%)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def plot_drawdown(equity_df: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    """Create drawdown chart.

    Args:
        equity_df: DataFrame with timestamp and total_value columns
        title: Chart title

    Returns:
        Plotly Figure object
    """
    from .metrics import calculate_drawdown

    if equity_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Calculate drawdown
    drawdown = calculate_drawdown(equity_df) * 100  # Convert to percentage

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_df["timestamp"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#C73E1D", width=1),
            hovertemplate="%{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_white",
        height=300,
    )

    return fig


def plot_returns_distribution(
    equity_df: pd.DataFrame,
    title: str = "Returns Distribution",
) -> go.Figure:
    """Create histogram of returns.

    Args:
        equity_df: DataFrame with total_value column
        title: Chart title

    Returns:
        Plotly Figure object
    """
    from .metrics import calculate_returns

    if equity_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    returns = calculate_returns(equity_df) * 100  # Convert to percentage

    fig = go.Figure()

    # Color positive returns green, negative red
    colors = ["#2ECC71" if r >= 0 else "#E74C3C" for r in returns]

    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=30,
            marker_color=colors,
            name="Returns",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        )
    )

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=300,
        showlegend=False,
    )

    return fig


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Create correlation heatmap.

    Args:
        correlation_matrix: Correlation DataFrame
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if correlation_matrix.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        width=500,
    )

    return fig


def create_metrics_card(
    metric_name: str,
    value: float | str,
    suffix: str = "",
    prefix: str = "",
    color: str = "blue",
) -> dict:
    """Create a metric card for Streamlit.

    Args:
        metric_name: Name of the metric
        value: Metric value
        suffix: Suffix to add (e.g., "%")
        prefix: Prefix to add (e.g., "$")
        color: Color theme (blue, green, red)

    Returns:
        Dictionary with metric info
    """
    return {
        "name": metric_name,
        "value": value,
        "suffix": suffix,
        "prefix": prefix,
        "color": color,
    }


def format_metric_value(value: float, decimals: int = 2) -> str:
    """Format metric value for display.

    Args:
        value: Value to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def plot_candlestick_with_rsi(
    ohlcv_df: pd.DataFrame,
    rsi_series: Optional[pd.Series] = None,
    title: str = "Price with RSI",
) -> go.Figure:
    """Create candlestick chart with RSI indicator.

    Args:
        ohlcv_df: DataFrame with columns: timestamp, open, high, low, close, volume
        rsi_series: Optional Series of RSI values
        title: Chart title

    Returns:
        Plotly Figure object with two subplots
    """
    if ohlcv_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Calculate RSI if not provided
    if rsi_series is None:
        from .live_data import calculate_rsi

        rsi_series = calculate_rsi(ohlcv_df["close"])

    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "RSI (14)"),
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=ohlcv_df["timestamp"],
            open=ohlcv_df["open"],
            high=ohlcv_df["high"],
            low=ohlcv_df["low"],
            close=ohlcv_df["close"],
            name="OHLC",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
        ),
        row=1,
        col=1,
    )

    # Add RSI line
    fig.add_trace(
        go.Scatter(
            x=ohlcv_df["timestamp"],
            y=rsi_series,
            mode="lines",
            name="RSI",
            line=dict(color="#9C27B0", width=2),
        ),
        row=2,
        col=1,
    )

    # Add RSI reference lines (overbought/oversold)
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        row=2,
        col=1,
        annotation_text="Overbought (70)",
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        row=2,
        col=1,
        annotation_text="Oversold (30)",
    )

    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600,
        xaxis_rangeslider_visible=False,
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)

    return fig


def plot_portfolio_aggregation(
    portfolio_data: dict[str, dict],
    title: str = "Portfolio Overview",
) -> go.Figure:
    """Create portfolio aggregation chart showing combined performance.

    Args:
        portfolio_data: Dict mapping symbol to {equity_df, current_price, etc.}
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if not portfolio_data:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    fig = go.Figure()

    # Create stacked area chart for total portfolio value
    total_values = []

    # Get all timestamps from first symbol
    first_symbol = list(portfolio_data.keys())[0]
    timestamps = portfolio_data[first_symbol]["equity_df"]["timestamp"]

    # Calculate total portfolio value over time
    total_value = pd.Series(0.0, index=range(len(timestamps)))

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

    for i, (symbol, data) in enumerate(portfolio_data.items()):
        equity_df = data["equity_df"]
        if not equity_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=equity_df["timestamp"],
                    y=equity_df["total_value"],
                    mode="lines",
                    name=f"{symbol} Value",
                    line=dict(color=colors[i % len(colors)], width=2),
                    stackgroup="one",  # Create stacked area
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig
