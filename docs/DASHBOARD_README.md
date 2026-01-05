# Performance Monitoring Dashboards

Real-time performance monitoring for GraphWiz Trader optimization metrics.

## Quick Start

### Terminal Dashboard (Recommended)

Lightweight, real-time monitoring in your terminal:

```bash
# Install dependencies
pip install rich

# Run terminal dashboard
python3 monitor_performance.py
```

**Features:**
- ✅ Real-time metrics (1-second refresh)
- ✅ Low resource usage (~50MB RAM)
- ✅ Works over SSH
- ✅ No browser required
- ✅ Performance insights and recommendations

### Web Dashboard (Full-Featured)

Interactive web-based monitoring with charts:

```bash
# Install dependencies
pip install -r requirements-dashboard.txt

# Run web dashboard
streamlit run dashboard_performance.py
```

**Features:**
- 📊 Interactive charts and graphs
- 📈 Historical performance trends
- 🎯 Multiple pages (Overview, Neo4j, Trading, System, History)
- 🔄 Auto-refresh (configurable)
- 💡 Performance insights

## What's Monitored

### Neo4j Performance
- Query execution time and volume
- Cache hit rate (target: >80%)
- Batch operations
- Retry count

### Trading Engine
- Trade execution time
- Ticker cache effectiveness
- API call reduction (target: >80%)
- Parallel fetch operations

### System Resources
- CPU and memory usage
- Thread pool utilization
- System uptime

## Dashboard Screenshots

### Terminal Dashboard
```
📊 GraphWiz Performance Monitor | 2026-01-03 12:34:56

┌─────────────────────┬─────────────────────┬─────────────────────┐
│  📊 Neo4j           │  ⚡ Trading Engine   │  🖥️  System         │
│  Performance        │                     │  Resources          │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Total Queries 15,234│ Total Trades      142│ Uptime           2.5h│
│ Avg Query     30.0ms│ Avg Trade       88.0ms│ Memory        245.6MB│
│ Cache Hit      82.3%│ API Reduction    82.0%│ CPU Usage        35%│
│ Batch Ops        234│ Parallel Feths      45│ Threads          4/10│
└─────────────────────┴─────────────────────┴─────────────────────┘

💡 Performance Insights
✓ Excellent Neo4j cache performance (>80%)
✓ Fast query execution (<50ms)
✓ Outstanding API call reduction (>80%)
✓ Fast trade execution (<100ms)
```

## Documentation

See [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) for:
- Detailed installation instructions
- Configuration options
- Production integration
- Troubleshooting
- Advanced features

## Performance Impact

The dashboards help you monitor:

| Optimization | Impact | Dashboard Metrics |
|-------------|--------|-------------------|
| Query Caching | 10-100x faster | Cache hit rate, query time |
| Batch Writes | 5-20x faster | Batch ops count |
| Ticker Cache | 80% fewer API calls | API reduction % |
| Parallel Fetch | 5-10x faster | Parallel fetches |

## Requirements

**Terminal Dashboard:**
- Python 3.7+
- rich>=13.6.0

**Web Dashboard:**
- Python 3.7+
- streamlit>=1.28.0
- plotly>=5.17.0
- pandas>=2.0.0

## Usage Tips

1. **During Development**: Use terminal dashboard for continuous feedback
2. **For Analysis**: Use web dashboard for historical trends
3. **In Production**: Monitor regularly to catch performance regressions
4. **After Changes**: Compare metrics before/after optimizations

## License

Part of GraphWiz Trader - See main LICENSE file.
