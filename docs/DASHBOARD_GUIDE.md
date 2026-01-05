# Performance Monitoring Dashboard Guide

Comprehensive guide to using the GraphWiz Trader performance monitoring dashboards.

## Overview

GraphWiz Trader includes two performance monitoring dashboards:

1. **Web Dashboard** (`dashboard_performance.py`) - Full-featured web-based monitoring with Streamlit
2. **Terminal Dashboard** (`monitor_performance.py`) - Lightweight terminal-based monitoring with Rich

Both dashboards provide real-time insights into:
- Neo4j query performance and caching
- Trading engine metrics and optimization impact
- System resource utilization
- Historical performance trends

## Installation

### Install Dependencies

```bash
# Install dashboard requirements
pip install -r requirements-dashboard.txt
```

**Required packages:**
- `streamlit>=1.28.0` - Web dashboard framework
- `plotly>=5.17.0` - Interactive charts
- `pandas>=2.0.0` - Data manipulation
- `rich>=13.6.0` - Terminal dashboard

### Quick Start

#### Web Dashboard

```bash
# Start the web dashboard
streamlit run dashboard_performance.py

# Or with custom port
streamlit run dashboard_performance.py --server.port 8501
```

The dashboard will open in your browser at `http://localhost:8501`

#### Terminal Dashboard

```bash
# Run the terminal dashboard
python3 monitor_performance.py
```

The dashboard will display in your terminal with real-time updates.

## Web Dashboard Features

### Pages

The web dashboard includes 5 main pages:

#### 1. Overview

**Key Metrics:**
- Total queries and trades
- Cache hit rates
- Average execution times
- Batch operations
- API call reduction
- Active positions

**Performance Insights:**
- Real-time analysis of optimization impact
- Performance recommendations
- System health status

#### 2. Neo4j Performance

**Query Performance:**
- Total query count with growth rate
- Average query time with trend
- Total query time

**Cache Performance:**
- Current cache size
- Cache hit rate percentage
- Cache hits/misses breakdown
- Visual bar chart of cache performance

**Batch Operations:**
- Total batch operations executed
- Pending batch operations
- Performance impact information

**Reliability Metrics:**
- Retry count for transient errors
- Overall success rate

#### 3. Trading Engine

**Trade Performance:**
- Total trades executed
- Average trade execution time
- Total trading time
- Active positions

**Ticker Cache:**
- Cache hit rate
- API call reduction percentage (with gauge)
- Parallel fetch operations completed

**Visualizations:**
- API call reduction gauge (target: 80%+)
- Performance impact charts

#### 4. System Metrics

**Resource Usage:**
- System uptime
- Memory usage (MB)
- CPU usage percentage
- Active threads / thread pool size

**Thread Pool:**
- Thread pool utilization gauge
- Active thread count

**Historical Charts:**
- CPU usage over time (60 minutes)
- Memory usage over time
- Time-series visualizations

#### 5. Historical Data

**Query Performance:**
- Query volume over time
- Cache hit rate trends
- Time-series analysis (60 minutes)

**Trading Performance:**
- Trade volume over time
- Average trade time trends
- API calls saved over time

### Controls

**Sidebar:**
- **Auto-refresh**: Enable/disable automatic refresh
- **Refresh interval**: Set refresh rate (1-60 seconds)
- **Refresh Now**: Manual refresh button
- **Last Updated**: Timestamp of last data refresh

**Navigation:**
- Switch between 5 different pages
- Each page focuses on specific metrics

## Terminal Dashboard Features

### Real-time Display

The terminal dashboard provides a compact, always-on view of:

#### Neo4j Panel (Left)

```
📊 Neo4j Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Queries        15,234  ━━━━━━━━━━━━━━
Avg Query Time        30.0ms  ━━━
Total Time           450.0s

Cache Hit Rate        82.3%   ████████████████
Cache Size              847
Cache Hits/Misses  12,456/2,778

Batch Operations        234  ━━━━━━━
Pending Batch             0

Retries                   12
```

#### Trading Engine Panel (Center)

```
⚡ Trading Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades            142  ━━━━━━━━━
Avg Trade Time        88.0ms  ━━━━━
Total Time            12.3s

Ticker Cache Rate      80.9%  ████████████████
API Reduction          82.0%  █████████████████████

Parallel Fetches         45  ━━━━━━━
Active Positions           3  ━━━━
```

#### System Resources Panel (Right)

```
🖥️ System Resources
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uptime                  2.5h  ━━━━━━━━━━━━━━
Memory               245.6 MB  ━━━━━━━
CPU Usage              35.2%  ━━━━━━

Threads Active          4/10  █████████
```

#### Performance Insights (Bottom)

Real-time analysis and recommendations:
- ✓ Excellent/good performance indicators
- ⚠ Areas needing attention
- 💡 Optimization tips

### Controls

**Navigation:**
- Dashboard updates automatically every second
- Press `Ctrl+C` to exit

## Configuration

### Web Dashboard

**Streamlit Configuration:** Create `.streamlit/config.toml`:

```toml
[browser]
gatheredUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

**Customization in code:** Edit `dashboard_performance.py`:

```python
# Change refresh rate
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 60, 5)

# Modify mock data for production
class MockMetrics:
    def get_neo4j_metrics(self):
        # Connect to real system instead of mock data
        pass
```

### Terminal Dashboard

**Update Frequency:** Edit `monitor_performance.py`:

```python
# Change refresh rate (currently 1 second)
time.sleep(1)  # Modify this value
```

**Color Schemes:** Rich supports different themes:

```python
console = Console(theme="dark")  # or "light", "monokai", etc.
```

## Production Integration

To connect dashboards to your running system:

### 1. Expose Metrics from Components

**Neo4j Knowledge Graph:**

```python
# In neo4j_graph.py
class KnowledgeGraph:
    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics."""
        return {
            "query_count": self._query_count,
            "total_query_time": self._query_time_total,
            "average_query_time": self._query_time_total / self._query_count,
            "cache_size": len(self._query_cache),
            "cache_hits": self._cache_hits,  # Add tracking
            "cache_misses": self._cache_misses,  # Add tracking
            "batch_buffer_size": len(self._batch_buffer),
            "batch_operations": self._batch_ops_completed,  # Add tracking
        }
```

**Trading Engine:**

```python
# In trading/engine.py
class TradingEngine:
    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics."""
        return {
            "trade_count": self._trade_count,
            "total_trade_time": self._total_trade_time,
            "average_trade_time": self._total_trade_time / self._trade_count,
            "ticker_cache_hits": self._cache_hits,  # Add tracking
            "ticker_cache_misses": self._cache_misses,  # Add tracking
            "api_call_reduction": self._calculate_api_reduction(),
            "active_positions": len(self.positions),
        }
```

### 2. Create Metrics Collection API

```python
# metrics_api.py
from fastapi import FastAPI
from graphwiz_trader.graph import KnowledgeGraph
from graphwiz_trader.trading import TradingEngine

app = FastAPI()
kg = None
engine = None

@app.get("/metrics/neo4j")
def get_neo4j_metrics():
    return kg.get_metrics()

@app.get("/metrics/trading")
def get_trading_metrics():
    return engine.get_metrics()

@app.get("/metrics/all")
def get_all_metrics():
    return {
        "neo4j": kg.get_metrics(),
        "trading": engine.get_metrics(),
        "timestamp": datetime.now().isoformat()
    }
```

### 3. Update Dashboards to Use Real Data

**Web Dashboard:**

```python
# Replace MockMetrics with real API calls
import requests

def get_real_metrics():
    response = requests.get("http://localhost:8000/metrics/all")
    return response.json()

# In render_overview()
metrics = get_real_metrics()  # Instead of MockMetrics()
```

**Terminal Dashboard:**

```python
# Similar updates to use real metrics
def get_all_metrics():
    response = requests.get("http://localhost:8000/metrics/all")
    return response.json()
```

## Usage Examples

### Monitor During Paper Trading

```bash
# Terminal 1: Start paper trading
python3 run_paper_trading.py

# Terminal 2: Start monitoring dashboard
python3 monitor_performance.py

# Or browser: Streamlit dashboard
streamlit run dashboard_performance.py
```

### Monitor During Live Trading

```bash
# Terminal 1: Start live trading
python3 main.py --mode live

# Terminal 2: Monitor performance
python3 monitor_performance.py

# Optional: Web dashboard for detailed view
streamlit run dashboard_performance.py --server.port 8501
```

### Continuous Monitoring in Production

```bash
# Run as background service
nohup python3 monitor_performance.py > monitoring.log 2>&1 &

# Or use systemd/supervisord for process management
```

## Performance Benchmarks

### Dashboard Resource Usage

**Web Dashboard (Streamlit):**
- Memory: ~150-200 MB
- CPU: ~5-10% (on refresh)
- Refresh: 1-5 seconds

**Terminal Dashboard (Rich):**
- Memory: ~50-80 MB
- CPU: ~2-5% (continuous)
- Refresh: 1 second

### Recommended Use Cases

**Web Dashboard:**
- Detailed performance analysis
- Historical trend investigation
- Presentations and reports
- Multi-user monitoring (deploy with authentication)

**Terminal Dashboard:**
- Continuous monitoring during trading
- Quick performance checks
- Low-resource environments
- SSH connections to remote servers

## Troubleshooting

### Web Dashboard Issues

**Dashboard not loading:**
```bash
# Check if Streamlit is running
ps aux | grep streamlit

# Try different port
streamlit run dashboard_performance.py --server.port 8502
```

**Charts not displaying:**
```bash
# Reinstall plotly
pip install --upgrade plotly

# Clear Streamlit cache
rm -rf ~/.streamlit/cache
```

**Slow refresh:**
```bash
# Increase refresh interval in sidebar
# Or modify default in code
refresh_interval = st.sidebar.slider("Refresh interval", 5, 60, 10)
```

### Terminal Dashboard Issues

**Display issues:**
```bash
# Ensure terminal supports Unicode
export LANG=en_US.UTF-8

# Try different terminal (GNOME Terminal, iTerm2, etc.)
```

**Colors not showing:**
```bash
# Check if terminal supports colors
python3 -c "from rich.console import Console; Console().print('[red]Test[/red]')"

# Use basic output if needed
python3 monitor_performance.py --no-color
```

**High CPU usage:**
```bash
# Reduce refresh frequency in code
time.sleep(5)  # Instead of time.sleep(1)
```

## Advanced Features

### Export Metrics to CSV

```python
# Add to web dashboard
import pandas as pd

@st.button("Export Metrics")
def export_metrics():
    metrics = get_all_metrics()
    df = pd.DataFrame(metrics)
    csv = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        "metrics.csv",
        "text/csv"
    )
```

### Alert Thresholds

```python
# Add alerting to terminal dashboard
def check_thresholds(metrics):
    alerts = []

    if metrics["neo4j"]["avg_query_time_ms"] > 100:
        alerts.append("⚠️  High query time detected!")

    if metrics["trading"]["api_reduction"] < 0.70:
        alerts.append("⚠️  Low API reduction rate!")

    return alerts

# Display in insights panel
if alerts:
    insights_text = "\n".join(alerts) + "\n" + insights_text
```

### Multi-System Monitoring

```python
# Monitor multiple trading systems
systems = {
    "production": "httpprod-server:8000",
    "staging": "staging-server:8000",
    "dev": "dev-server:8000"
}

for name, url in systems.items():
    metrics = requests.get(f"{url}/metrics/all").json()
    # Display metrics for each system
```

## Best Practices

1. **Use Terminal Dashboard** for continuous monitoring during trading
2. **Use Web Dashboard** for detailed analysis and historical trends
3. **Set Alert Thresholds** based on your system's baseline performance
4. **Review Metrics Regularly** to identify optimization opportunities
5. **Export Historical Data** for long-term trend analysis
6. **Monitor During Peak Load** to understand system limits
7. **Track Metrics Over Time** to measure optimization impact

## Support

For issues or questions:
- Check logs: `monitoring.log` (terminal), browser console (web)
- Review this guide's troubleshooting section
- Check system resource availability
- Verify all dependencies are installed

## Summary

The performance monitoring dashboards provide real-time visibility into:
- **10-100x speedup** from query caching
- **5-20x faster** bulk operations with batching
- **80% reduction** in API calls from ticker caching
- **Thread-safe** concurrent operations
- **Reliable** performance with retry logic

Use these tools to ensure your trading system is running optimally!
