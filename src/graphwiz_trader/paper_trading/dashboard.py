"""
Paper trading dashboard - Main Streamlit application.

Provides interactive visualization and analysis of paper trading results.
"""

import sys
from pathlib import Path

# Add project root to path for imports when run directly by Streamlit
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from loguru import logger

from graphwiz_trader.paper_trading.dashboard.data_loader import (
    get_available_symbols,
    load_all_symbols,
    load_equity_curve,
    load_summary,
)
from graphwiz_trader.paper_trading.dashboard.metrics import (
    calculate_correlation,
    get_latest_metrics,
)
from graphwiz_trader.paper_trading.dashboard.charts import (
    plot_equity_curve,
    plot_comparison,
    plot_drawdown,
    plot_returns_distribution,
    plot_correlation_heatmap,
    format_metric_value,
)
from graphwiz_trader.paper_trading.dashboard.service_monitor import (
    get_service_status,
    get_active_symbols,
    get_log_summary,
    is_service_running,
)
from graphwiz_trader.paper_trading.dashboard.live_data import (
    get_current_price,
    get_all_symbols_prices,
    get_24h_stats,
    fetch_recent_candles,
    get_market_summary,
)
from graphwiz_trader.paper_trading.dashboard.charts import (
    plot_candlestick_with_rsi,
    plot_portfolio_aggregation,
)

# Page config
st.set_page_config(
    page_title="GraphWiz Paper Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .status-running {
        color: #2ECC71;
        font-weight: bold;
    }
    .status-stopped {
        color: #E74C3C;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "selected_symbols" not in st.session_state:
    st.session_state.selected_symbols = []


def load_data():
    """Load all available data."""
    symbols = get_available_symbols()
    service_status = get_service_status()
    config_symbols = get_active_symbols()

    return {
        "symbols": symbols,
        "service_status": service_status,
        "config_symbols": config_symbols,
    }


def render_overview_page(data: dict):
    """Render overview page with service status and portfolio summary."""
    st.title("📊 Service Overview")

    col1, col2, col3 = st.columns(3)

    # Service status
    with col1:
        is_running = is_service_running()
        status_icon = "✅" if is_running else "⚠️"
        status_text = "Running" if is_running else "Stopped"
        st.metric("Service Status", f"{status_icon} {status_text}")

    # Total symbols
    with col2:
        total_symbols = len(data["config_symbols"])
        st.metric("Active Symbols", total_symbols)

    # Running instances
    with col3:
        running_count = len(data["service_status"])
        st.metric("Running Instances", running_count)

    st.markdown("---")

    # Service status details
    st.subheader("Symbol Status")

    if not data["config_symbols"]:
        st.warning("No symbols configured. Add symbols using the service CLI.")
        return

    # Create status table
    status_data = []
    for symbol in data["config_symbols"]:
        status = data["service_status"].get(symbol, {})
        is_running = symbol in data["service_status"]

        status_data.append({
            "Symbol": symbol,
            "Status": "🟢 Running" if is_running else "🔴 Stopped",
            "Uptime": status.get("uptime_str", "-"),
            "Memory": f"{status.get('memory_mb', 0):.1f} MB" if is_running else "-",
        })

    st.dataframe(
        pd.DataFrame(status_data),
        use_container_width=True,
        hide_index=True,
    )

    # Live prices section
    st.markdown("---")
    st.subheader("💰 Live Prices")

    with st.spinner("Fetching live prices..."):
        market_summary = get_market_summary()

    if market_summary:
        price_data = []
        for symbol, stats in market_summary.items():
            price_data.append({
                "Symbol": symbol,
                "Price": f"${stats['price']:,.2f}",
                "24h Change": f"{stats['change_24h_pct']:+.2f}%",
                "24h High": f"${stats['high_24h']:,.2f}",
                "24h Low": f"${stats['low_24h']:,.2f}",
                "Volume": f"${stats['volume_24h']:,.0f}",
            })

        st.dataframe(
            pd.DataFrame(price_data),
            use_container_width=True,
            hide_index=True,
        )

        # Calculate total portfolio value
        st.markdown("---")
        st.subheader("📈 Portfolio Summary")

        total_value = 0
        for symbol, stats in market_summary.items():
            equity_df = load_equity_curve(symbol)
            if equity_df is not None and not equity_df.empty:
                total_value += equity_df.iloc[-1]["total_value"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")

        with col2:
            # Calculate 24h change
            total_24h_change = sum(
                stats.get("change_24h_pct", 0) * stats.get("price", 0)
                for stats in market_summary.values()
            )
            st.metric("24h Change (est.)", f"{total_24h_change:+.2f}%")

        with col3:
            st.metric("Symbols Active", len(market_summary))

    # Quick actions
    st.markdown("---")
    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    with col2:
        st.info("Use service CLI to start/stop symbols")

    with col3:
        st.caption("Run: `python scripts/paper_trading_service.py status`")


def render_symbol_detail_page(data: dict):
    """Render detailed view for a single symbol."""
    st.title("📈 Symbol Performance")

    # Symbol selector
    available_symbols = data["symbols"]
    if not available_symbols:
        st.warning("No data available. Run paper trading first.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select Symbol",
            available_symbols,
            index=0,
            key="symbol_selector",
        )

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Load data for selected symbol
    equity_df = load_equity_curve(selected_symbol)
    summary = load_summary(selected_symbol)

    if equity_df is None:
        st.error(f"No equity data found for {selected_symbol}")
        return

    # Calculate metrics
    metrics = get_latest_metrics(selected_symbol, equity_df, summary)

    # Display metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = f"{metrics['total_return_pct']:.2f}%"
        st.metric(
            "Total Return",
            f"${format_metric_value(metrics['total_return'])}",
            delta=delta,
            delta_color="normal" if metrics['total_return'] >= 0 else "inverse",
        )

    with col2:
        st.metric(
            "Current Value",
            f"${format_metric_value(metrics['final_value'])}",
        )

    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2f}%",
            delta_color="inverse",
        )

    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
        )

    st.markdown("---")

    # Equity curve chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Equity Curve")
        fig_equity = plot_equity_curve(equity_df, title=f"{selected_symbol} Portfolio Value")
        st.plotly_chart(fig_equity, use_container_width=True)

    with col2:
        st.subheader("Drawdown")
        fig_drawdown = plot_drawdown(equity_df, title=f"{selected_symbol} Drawdown")
        st.plotly_chart(fig_drawdown, use_container_width=True)

    st.markdown("---")

    # Returns distribution
    st.subheader("Returns Distribution")
    fig_returns = plot_returns_distribution(equity_df, title=f"{selected_symbol} Returns")
    st.plotly_chart(fig_returns, use_container_width=True)

    # Detailed metrics
    st.markdown("---")
    st.subheader("Detailed Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Points", metrics['data_points'])
        st.metric("Initial Capital", f"${format_metric_value(metrics['initial_capital'])}")

    with col2:
        st.metric("Volatility", f"{metrics['volatility']:.2f}%")
        if "total_trades" in metrics:
            st.metric("Total Trades", metrics['total_trades'])

    with col3:
        st.metric("Current Position", f"{metrics['current_position']:.4f}")
        if "win_rate" in metrics:
            st.metric("Win Rate", f"{metrics['win_rate']}%")

    # Log summary
    st.markdown("---")
    st.subheader("Recent Activity")

    log_summary = get_log_summary(selected_symbol)
    if log_summary['latest_entry']:
        st.info(f"**Latest Log Entry:** {log_summary['latest_entry']}")
    else:
        st.caption("No recent log entries")


def render_comparison_page(data: dict):
    """Render comparison page for multiple symbols."""
    st.title("🔍 Multi-Symbol Comparison")

    available_symbols = data["symbols"]

    if len(available_symbols) < 2:
        st.warning("Need at least 2 symbols with data for comparison.")
        return

    # Multi-select symbols
    default_symbols = available_symbols[:min(4, len(available_symbols))]

    selected_symbols = st.multiselect(
        "Select Symbols to Compare",
        available_symbols,
        default=default_symbols,
        key="compare_symbols",
    )

    if not selected_symbols or len(selected_symbols) < 2:
        st.info("Select at least 2 symbols to compare.")
        return

    # Load data for selected symbols
    equity_data = {}
    for symbol in selected_symbols:
        df = load_equity_curve(symbol)
        if df is not None:
            equity_data[symbol] = df

    if len(equity_data) < 2:
        st.error("Could not load data for selected symbols.")
        return

    # Normalize toggle
    normalize = st.checkbox("Normalize to 100%", value=True, key="normalize_compare")

    # Comparison chart
    st.subheader("Equity Curve Comparison")
    fig_compare = plot_comparison(
        equity_data,
        normalize=normalize,
        title="Symbol Comparison",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Performance comparison table
    st.markdown("---")
    st.subheader("Performance Metrics")

    metrics_data = []
    for symbol in selected_symbols:
        df = equity_data.get(symbol)
        if df is not None:
            summary = load_summary(symbol)
            metrics = get_latest_metrics(symbol, df, summary)
            metrics_data.append({
                "Symbol": symbol,
                "Total Return %": f"{metrics['total_return_pct']:.2f}",
                "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
                "Max Drawdown %": f"{metrics['max_drawdown']:.2f}",
                "Volatility %": f"{metrics['volatility']:.2f}",
                "Trades": metrics.get('total_trades', 0),
            })

    if metrics_data:
        st.dataframe(
            pd.DataFrame(metrics_data),
            use_container_width=True,
            hide_index=True,
        )

    # Correlation matrix
    if len(equity_data) >= 2:
        st.markdown("---")
        st.subheader("Correlation Matrix")

        corr_matrix = calculate_correlation(equity_data)
        if not corr_matrix.empty:
            fig_corr = plot_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)


def render_analytics_page(data: dict):
    """Render analytics page with detailed analysis."""
    st.title("📊 Trade Analytics")

    available_symbols = data["symbols"]

    if not available_symbols:
        st.warning("No data available for analytics.")
        return

    # Symbol selector
    selected_symbol = st.selectbox(
        "Select Symbol",
        available_symbols,
        index=0,
        key="analytics_symbol",
    )

    equity_df = load_equity_curve(selected_symbol)
    if equity_df is None:
        st.error(f"No data found for {selected_symbol}")
        return

    # Basic statistics
    st.subheader("Return Statistics")

    from graphwiz_trader.paper_trading.dashboard.metrics import calculate_returns

    returns = calculate_returns(equity_df) * 100  # Convert to percentage

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Return", f"{returns.mean():.3f}%")

    with col2:
        st.metric("Std Dev", f"{returns.std():.3f}%")

    with col3:
        st.metric("Min Return", f"{returns.min():.3f}%")

    with col4:
        st.metric("Max Return", f"{returns.max():.3f}%")

    st.markdown("---")

    # Distribution chart
    st.subheader("Return Distribution")
    fig_dist = plot_returns_distribution(equity_df, title=f"{selected_symbol} Return Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

    # Drawdown analysis
    st.markdown("---")
    st.subheader("Drawdown Analysis")

    fig_dd = plot_drawdown(equity_df, title=f"{selected_symbol} Drawdown Over Time")
    st.plotly_chart(fig_dd, use_container_width=True)


def render_market_data_page(data: dict):
    """Render market data page with live prices and candlestick charts."""
    st.title("📡 Market Data")

    available_symbols = data["config_symbols"]

    if not available_symbols:
        st.warning("No symbols configured.")
        return

    # Symbol selector
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select Symbol",
            available_symbols,
            index=0,
            key="market_symbol",
        )

    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=3,
            key="timeframe",
        )

    with col3:
        limit = st.selectbox(
            "Candles",
            [50, 100, 200, 500],
            index=1,
            key="candle_limit",
        )

    # Fetch candles
    with st.spinner(f"Fetching {selected_symbol} data..."):
        candles_df = fetch_recent_candles(selected_symbol, timeframe=timeframe, limit=limit)

    if candles_df is not None and not candles_df.empty:
        st.subheader(f"💹 {selected_symbol} - Price & RSI")

        # Display candlestick chart with RSI
        fig = plot_candlestick_with_rsi(candles_df, title=f"{selected_symbol} ({timeframe})")
        st.plotly_chart(fig, use_container_width=True)

        # Display 24h statistics
        st.markdown("---")
        st.subheader("📊 24h Statistics")

        stats = get_24h_stats(selected_symbol)

        if stats:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("24h High", f"${stats['high']:,.2f}")

            with col2:
                st.metric("24h Low", f"${stats['low']:,.2f}")

            with col3:
                st.metric("24h Volume", f"${stats['quote_volume']:,.0f}")

            with col4:
                change_color = "normal" if stats['change_pct'] >= 0 else "inverse"
                st.metric("24h Change", f"{stats['change_pct']:+.2f}%", delta_color=change_color)

    else:
        st.error(f"Failed to fetch data for {selected_symbol}")


def render_performance_ranking(data: dict):
    """Render performance ranking page."""
    st.title("🏆 Performance Ranking")

    available_symbols = data["symbols"]

    if not available_symbols:
        st.warning("No data available.")
        return

    # Load metrics for all symbols
    rankings = []

    for symbol in available_symbols:
        equity_df = load_equity_curve(symbol)
        summary = load_summary(symbol)

        if equity_df is not None:
            metrics = get_latest_metrics(symbol, equity_df, summary)
            rankings.append({
                "Symbol": symbol,
                "Total Return %": metrics['total_return_pct'],
                "Sharpe Ratio": metrics['sharpe_ratio'],
                "Max Drawdown %": metrics['max_drawdown'],
                "Volatility %": metrics['volatility'],
                "Trades": metrics.get('total_trades', 0),
                "Win Rate %": metrics.get('win_rate', 0),
                "Final Value": metrics['final_value'],
            })

    if not rankings:
        st.warning("No ranking data available.")
        return

    # Create DataFrame and sort by different metrics
    ranking_df = pd.DataFrame(rankings)

    st.subheader("Rank by Performance")

    sort_metric = st.selectbox(
        "Sort by",
        ["Total Return %", "Sharpe Ratio", "Max Drawdown %", "Win Rate %"],
        index=0,
        key="rank_metric",
    )

    ascending = sort_metric == "Max Drawdown %"  # Lower drawdown is better

    sorted_df = ranking_df.sort_values(by=sort_metric, ascending=ascending)

    # Add rank column
    sorted_df.insert(0, "Rank", range(1, len(sorted_df) + 1))

    # Display with styling
    st.dataframe(
        sorted_df,
        use_container_width=True,
        hide_index=True,
    )

    # Export button
    st.markdown("---")
    st.subheader("Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export Ranking as CSV", use_container_width=True):
            csv = sorted_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"performance_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


def render_settings_page(data: dict):
    """Render settings page."""
    st.title("⚙️ Settings")

    st.subheader("Dashboard Settings")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-Refresh", value=False, key="auto_refresh")
    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=30,
            key="refresh_interval",
        )
        st.info(f"Dashboard will refresh every {refresh_interval} seconds")
    else:
        st.caption("Auto-refresh is disabled. Click refresh button manually.")

    st.markdown("---")

    # Service configuration info
    st.subheader("Service Configuration")

    st.info("""
    **Service Management:**
    - Start/Stop: `python scripts/paper_trading_service.py start|stop`
    - Status: `python scripts/paper_trading_service.py status`
    - Add Symbol: `python scripts/paper_trading_service.py add SYMBOL --capital 10000`
    - Remove Symbol: `python scripts/paper_trading_service.py remove SYMBOL`
    """)

    st.markdown("---")

    # Data location
    st.subheader("Data Locations")

    st.json({
        "data_directory": str(Path("data/paper_trading").absolute()),
        "log_directory": str(Path("logs").absolute()),
        "config_file": str(Path("config/paper_trading.json").absolute()),
    })


def main():
    """Main application entry point."""
    # Load data
    data = load_data()

    # Sidebar navigation
    with st.sidebar:
        st.title("📈 GraphWiz Trader")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "📊 Overview",
                "📈 Symbol Detail",
                "🔍 Comparison",
                "📊 Analytics",
                "📡 Market Data",
                "🏆 Performance Ranking",
                "⚙️ Settings",
            ],
            key="page_navigation",
        )

        st.markdown("---")

        # Quick stats in sidebar
        st.caption("Quick Stats")
        running_count = len(data["service_status"])
        total_symbols = len(data["config_symbols"])

        st.metric("Running", running_count)
        st.metric("Configured", total_symbols)

        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        if st.button("🔄 Refresh"):
            st.rerun()

    # Render selected page
    if page.startswith("📊 Overview"):
        render_overview_page(data)
    elif page.startswith("📈 Symbol Detail"):
        render_symbol_detail_page(data)
    elif page.startswith("🔍 Comparison"):
        render_comparison_page(data)
    elif page.startswith("📊 Analytics"):
        render_analytics_page(data)
    elif page.startswith("📡 Market Data"):
        render_market_data_page(data)
    elif page.startswith("🏆 Performance Ranking"):
        render_performance_ranking(data)
    elif page.startswith("⚙️ Settings"):
        render_settings_page(data)


if __name__ == "__main__":
    main()
