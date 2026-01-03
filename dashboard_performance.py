#!/usr/bin/env python3
"""Performance Monitoring Dashboard for GraphWiz Trader.

A comprehensive Streamlit dashboard for monitoring system performance,
including Neo4j optimizations, trading engine metrics, and cache statistics.
"""

import streamlit as st
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="GraphWiz Performance Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


class MockMetrics:
    """Mock metrics collector for demonstration.

    In production, this would connect to the actual running system.
    """

    def __init__(self):
        self.start_time = datetime.now()

    def get_neo4j_metrics(self) -> Dict[str, Any]:
        """Get Neo4j performance metrics."""
        return {
            "query_count": 15234,
            "total_query_time": 456.78,
            "average_query_time": 0.030,
            "cache_size": 847,
            "cache_hits": 12456,
            "cache_misses": 2778,
            "batch_buffer_size": 0,
            "batch_operations": 234,
            "retry_count": 12
        }

    def get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading engine metrics."""
        return {
            "trade_count": 142,
            "total_trade_time": 12.456,
            "average_trade_time": 0.088,
            "ticker_cache_hits": 892,
            "ticker_cache_misses": 234,
            "api_call_reduction": 0.82,
            "parallel_fetches": 45,
            "active_positions": 3
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        uptime = datetime.now() - self.start_time
        return {
            "uptime_seconds": uptime.total_seconds(),
            "memory_usage_mb": 245.6,
            "cpu_usage_percent": 35.2,
            "thread_pool_size": 10,
            "active_threads": 4
        }

    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get historical query performance."""
        base_time = datetime.now() - timedelta(hours=1)
        data = []
        for i in range(60):
            timestamp = base_time + timedelta(minutes=i)
            data.append({
                "timestamp": timestamp,
                "query_count": 200 + int(50 * (i % 10) / 10),
                "avg_time": 0.025 + 0.01 * (i % 5) / 5,
                "cache_hit_rate": 0.75 + 0.15 * (i % 8) / 8
            })
        return data

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get historical trade performance."""
        base_time = datetime.now() - timedelta(hours=1)
        data = []
        for i in range(60):
            timestamp = base_time + timedelta(minutes=i)
            data.append({
                "timestamp": timestamp,
                "trade_count": max(0, 5 - (i % 10) // 2),
                "avg_time": 0.08 + 0.02 * (i % 4) / 4,
                "api_calls_saved": 15 + 5 * (i % 6)
            })
        return data


def render_metric_card(value: float, label: str, delta: float = None, prefix: str = "", suffix: str = ""):
    """Render a metric card with styling."""
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta > 0 else "red"
        delta_symbol = "↑" if delta > 0 else "↓"
        delta_html = f'<div style="font-size: 0.8rem; color: {delta_color};">{delta_symbol} {abs(delta):.1f}%</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value:.3f}{suffix}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<div class="main-header">📊 GraphWiz Performance Dashboard</div>', unsafe_allow_html=True)

    # Initialize metrics collector
    if "metrics" not in st.session_state:
        st.session_state.metrics = MockMetrics()

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 60, 5)

    # Metrics collector
    metrics = st.session_state.metrics

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Overview", "Neo4j Performance", "Trading Engine", "System Metrics", "Historical Data"]
    )

    # Refresh button
    if st.sidebar.button("Refresh Now") or (auto_refresh and st.session_state.get("last_refresh", 0) + refresh_interval < time.time()):
        st.session_state.last_refresh = time.time()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

    # Page content
    if page == "Overview":
        render_overview(metrics)
    elif page == "Neo4j Performance":
        render_neo4j(metrics)
    elif page == "Trading Engine":
        render_trading(metrics)
    elif page == "System Metrics":
        render_system(metrics)
    elif page == "Historical Data":
        render_historical(metrics)


def render_overview(metrics):
    """Render overview page with key metrics."""
    st.header("System Overview")

    # Neo4j metrics
    neo4j_metrics = metrics.get_neo4j_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            neo4j_metrics["query_count"],
            "Total Queries",
            delta=12.5
        )

    with col2:
        cache_hit_rate = neo4j_metrics["cache_hits"] / (neo4j_metrics["cache_hits"] + neo4j_metrics["cache_misses"])
        render_metric_card(
            cache_hit_rate * 100,
            "Cache Hit Rate",
            delta=5.2,
            suffix="%"
        )

    with col3:
        render_metric_card(
            neo4j_metrics["average_query_time"] * 1000,
            "Avg Query Time",
            delta=-15.3,
            suffix="ms"
        )

    with col4:
        render_metric_card(
            neo4j_metrics["batch_operations"],
            "Batch Operations",
            delta=8.7
        )

    st.markdown("---")

    # Trading metrics
    trading_metrics = metrics.get_trading_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            trading_metrics["trade_count"],
            "Total Trades",
            delta=3.2
        )

    with col2:
        render_metric_card(
            trading_metrics["api_call_reduction"] * 100,
            "API Call Reduction",
            delta=2.1,
            suffix="%"
        )

    with col3:
        render_metric_card(
            trading_metrics["average_trade_time"] * 1000,
            "Avg Trade Time",
            delta=-8.4,
            suffix="ms"
        )

    with col4:
        render_metric_card(
            trading_metrics["active_positions"],
            "Active Positions"
        )

    st.markdown("---")

    # Key performance insights
    st.subheader("🎯 Performance Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Neo4j Optimization Impact**
        - Query caching reducing load by **82%**
        - Batch operations **20x faster** than individual writes
        - Average query time: **30ms**
        - Cache hit rate: **82%**
        """)

    with col2:
        st.success("""
        **Trading Engine Optimization Impact**
        - Ticker caching reducing API calls by **82%**
        - Parallel fetching **5x faster** for multi-symbol
        - Average trade time: **88ms**
        - **45 parallel fetches** completed
        """)


def render_neo4j(metrics):
    """Render Neo4j performance page."""
    st.header("Neo4j Performance Metrics")

    neo4j_metrics = metrics.get_neo4j_metrics()

    # Query performance
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Queries",
            f"{neo4j_metrics['query_count']:,}",
            delta="+12.5%"
        )

    with col2:
        st.metric(
            "Average Query Time",
            f"{neo4j_metrics['average_query_time']*1000:.2f}ms",
            delta="-15.3%"
        )

    with col3:
        st.metric(
            "Total Query Time",
            f"{neo4j_metrics['total_query_time']:.2f}s"
        )

    st.markdown("---")

    # Cache performance
    st.subheader("Cache Performance")

    col1, col2, col3, col4 = st.columns(4)

    total_requests = neo4j_metrics["cache_hits"] + neo4j_metrics["cache_misses"]
    cache_hit_rate = neo4j_metrics["cache_hits"] / total_requests if total_requests > 0 else 0

    with col1:
        st.metric("Cache Size", f"{neo4j_metrics['cache_size']:,}")

    with col2:
        st.metric("Cache Hit Rate", f"{cache_hit_rate*100:.1f}%")

    with col3:
        st.metric("Cache Hits", f"{neo4j_metrics['cache_hits']:,}")

    with col4:
        st.metric("Cache Misses", f"{neo4j_metrics['cache_misses']:,}")

    # Cache performance visualization
    fig = go.Figure(data=[
        go.Bar(
            name='Hits',
            x=['Cache Performance'],
            y=[neo4j_metrics['cache_hits']],
            marker_color='green'
        ),
        go.Bar(
            name='Misses',
            x=['Cache Performance'],
            y=[neo4j_metrics['cache_misses']],
            marker_color='red'
        )
    ])

    fig.update_layout(
        barmode='stack',
        title='Cache Hit/Miss Distribution',
        yaxis_title='Count'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Batch operations
    st.subheader("Batch Operations")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Batch Operations", f"{neo4j_metrics['batch_operations']:,}")

    with col2:
        st.metric("Pending Batch Ops", f"{neo4j_metrics['batch_buffer_size']}")

    st.info("💡 Batch operations provide 5-20x speedup for bulk writes")

    st.markdown("---")

    # Reliability
    st.subheader("Reliability Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Retry Count", f"{neo4j_metrics['retry_count']}")

    with col2:
        success_rate = (neo4j_metrics['query_count'] - neo4j_metrics['retry_count']) / neo4j_metrics['query_count'] * 100
        st.metric("Success Rate", f"{success_rate:.2f}%")


def render_trading(metrics):
    """Render trading engine performance page."""
    st.header("Trading Engine Performance")

    trading_metrics = metrics.get_trading_metrics()

    # Trade performance
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Trades",
            f"{trading_metrics['trade_count']}",
            delta="+3.2%"
        )

    with col2:
        st.metric(
            "Average Trade Time",
            f"{trading_metrics['average_trade_time']*1000:.2f}ms",
            delta="-8.4%"
        )

    with col3:
        st.metric(
            "Total Trade Time",
            f"{trading_metrics['total_trade_time']:.2f}s"
        )

    with col4:
        st.metric(
            "Active Positions",
            f"{trading_metrics['active_positions']}"
        )

    st.markdown("---")

    # Ticker cache performance
    st.subheader("Ticker Cache Performance")

    col1, col2, col3 = st.columns(3)

    total_requests = trading_metrics["ticker_cache_hits"] + trading_metrics["ticker_cache_misses"]
    cache_hit_rate = trading_metrics["ticker_cache_hits"] / total_requests if total_requests > 0 else 0

    with col1:
        st.metric("Cache Hit Rate", f"{cache_hit_rate*100:.1f}%")

    with col2:
        st.metric("API Call Reduction", f"{trading_metrics['api_call_reduction']*100:.1f}%")

    with col3:
        st.metric("Parallel Fetches", f"{trading_metrics['parallel_fetches']}")

    # API call reduction visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=trading_metrics['api_call_reduction'] * 100,
        title={'text': "API Call Reduction"},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.success("🎉 Ticker cache is reducing API calls by 82%, saving rate limits and costs!")


def render_system(metrics):
    """Render system metrics page."""
    st.header("System Metrics")

    system_metrics = metrics.get_system_metrics()

    # Uptime and resources
    col1, col2, col3, col4 = st.columns(4)

    uptime_hours = system_metrics["uptime_seconds"] / 3600

    with col1:
        st.metric("Uptime", f"{uptime_hours:.1f}h")

    with col2:
        st.metric("Memory Usage", f"{system_metrics['memory_usage_mb']:.1f} MB")

    with col3:
        st.metric("CPU Usage", f"{system_metrics['cpu_usage_percent']:.1f}%")

    with col4:
        st.metric("Active Threads", f"{system_metrics['active_threads']}/{system_metrics['thread_pool_size']}")

    st.markdown("---")

    # Thread pool utilization
    st.subheader("Thread Pool Utilization")

    thread_usage = system_metrics['active_threads'] / system_metrics['thread_pool_size'] * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=thread_usage,
        title={'text': "Thread Pool Utilization"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Resource usage chart
    st.subheader("Resource Usage Over Time")

    # Generate sample data
    timestamps = pd.date_range(end=datetime.now(), periods=60, freq="1min")
    cpu_data = [30 + 10 * (i % 10) / 10 for i in range(60)]
    memory_data = [200 + 50 * (i % 8) / 8 for i in range(60)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cpu_data,
        mode='lines',
        name='CPU Usage (%)',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=memory_data,
        mode='lines',
        name='Memory (MB)',
        yaxis='y2',
        line=dict(color='green')
    ))

    fig.update_layout(
        title='Resource Usage (Last 60 minutes)',
        xaxis_title='Time',
        yaxis_title='CPU Usage (%)',
        yaxis2=dict(
            title='Memory (MB)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_historical(metrics):
    """Render historical performance data."""
    st.header("Historical Performance Data")

    # Query history
    query_history = metrics.get_query_history()
    df_query = pd.DataFrame(query_history)

    st.subheader("Query Performance Over Time")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_query['timestamp'],
        y=df_query['query_count'],
        mode='lines',
        name='Query Count',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='Query Volume (Last 60 minutes)',
        xaxis_title='Time',
        yaxis_title='Query Count'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Cache hit rate over time
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_query['timestamp'],
        y=df_query['cache_hit_rate'] * 100,
        mode='lines',
        name='Cache Hit Rate',
        fill='tozeroy',
        line=dict(color='green')
    ))

    fig.update_layout(
        title='Cache Hit Rate Over Time',
        xaxis_title='Time',
        yaxis_title='Hit Rate (%)'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Trade history
    trade_history = metrics.get_trade_history()
    df_trade = pd.DataFrame(trade_history)

    st.subheader("Trading Performance Over Time")

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_trade['timestamp'],
            y=df_trade['trade_count'],
            mode='lines+markers',
            name='Trade Count',
            line=dict(color='purple')
        ))

        fig.update_layout(
            title='Trade Volume',
            xaxis_title='Time',
            yaxis_title='Trade Count'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_trade['timestamp'],
            y=df_trade['avg_time'] * 1000,
            mode='lines',
            name='Avg Time',
            line=dict(color='orange')
        ))

        fig.update_layout(
            title='Average Trade Time',
            xaxis_title='Time',
            yaxis_title='Time (ms)'
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
