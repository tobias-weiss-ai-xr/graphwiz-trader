# Paper Trading Dashboard

Interactive web-based dashboard for visualizing and analyzing paper trading results.

## Features

### Overview Page
- Service status monitoring
- Active symbol status table
- Quick actions for service management
- Real-time status updates

### Symbol Detail Page
- Interactive equity curve charts
- Drawdown visualization
- Performance metrics cards:
  - Total return (absolute & percentage)
  - Current portfolio value
  - Maximum drawdown
  - Sharpe ratio
  - Win rate (if trades executed)
- Return distribution histogram
- Detailed metrics breakdown
- Recent log entries

### Comparison Page
- Multi-symbol equity curve comparison
- Normalized performance view
- Side-by-side metrics table
- Correlation matrix heatmap

### Analytics Page
- Return statistics (mean, std dev, min, max)
- Return distribution analysis
- Drawdown over time visualization

### Settings Page
- Auto-refresh configuration
- Service management instructions
- Data location information

## Installation

### Prerequisites

The dashboard requires the following dependencies (automatically included):

```bash
pip install streamlit>=1.28.0 plotly>=5.17.0
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Dashboard

Run the dashboard launcher:

```bash
python scripts/run_dashboard.py
```

The dashboard will open in your browser at: **http://localhost:8501**

### Alternative Launch Methods

You can also run Streamlit directly:

```bash
streamlit run src/graphwiz_trader/paper_trading/dashboard.py
```

### Navigation

Use the sidebar to navigate between pages:
- 📊 **Overview** - Service status and summary
- 📈 **Symbol Detail** - Individual symbol analysis
- 🔍 **Comparison** - Multi-symbol comparison
- 📊 **Analytics** - Detailed return analysis
- ⚙️ **Settings** - Configuration and options

### Features by Page

#### Overview
- View running service instances
- Check uptime and resource usage
- Quick refresh button

#### Symbol Detail
1. Select a symbol from the dropdown
2. View equity curve chart
3. See drawdown visualization
4. Review performance metrics
5. Check recent log entries

#### Comparison
1. Select multiple symbols to compare
2. Toggle normalization (100% baseline)
3. View correlation matrix
4. Compare performance metrics

#### Analytics
1. Select symbol to analyze
2. View return distribution
3. See drawdown over time
4. Review statistics

#### Settings
1. Enable/disable auto-refresh
2. Set refresh interval
3. View service commands
4. Check data locations

## Data Requirements

The dashboard requires paper trading data to be present:

```
data/paper_trading/
├── BTC_USDT_equity_*.csv
├── BTC_USDT_summary_*.json
├── ETH_USDT_equity_*.csv
├── ETH_USDT_summary_*.json
└── ...
```

Generate data by running the paper trading service:

```bash
# Start the service
python scripts/paper_trading_service.py start

# Or run a single instance
python scripts/paper_trade.py --symbol BTC/USDT --capital 10000
```

## File Structure

```
src/graphwiz_trader/paper_trading/
├── dashboard.py                 # Main Streamlit app
└── dashboard/
    ├── __init__.py             # Module exports
    ├── data_loader.py          # Data loading functions
    ├── metrics.py              # Performance metrics calculations
    ├── charts.py               # Plotly chart generation
    └── service_monitor.py      # Service status monitoring

scripts/
└── run_dashboard.py            # Dashboard launcher
```

## Troubleshooting

### Dashboard Won't Start

**Problem**: Module import errors

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/graphwiz-trader

# Install dependencies
pip install -r requirements.txt
```

**Problem**: Port 8501 already in use

**Solution**:
```bash
# Use a different port
streamlit run src/graphwiz_trader/paper_trading/dashboard.py --server.port 8502
```

### No Data Available

**Problem**: "No data available" message

**Solution**:
1. Check service is running:
   ```bash
   python scripts/paper_trading_service.py status
   ```

2. Start service if needed:
   ```bash
   python scripts/paper_trading_service.py start
   ```

3. Wait for at least one check cycle (default: 1 hour)

### Charts Not Loading

**Problem**: Charts appear empty or show errors

**Solution**:
1. Check browser console for errors
2. Ensure data files exist and are valid
3. Try clicking refresh button
4. Check logs for errors:
   ```bash
   tail -f logs/*.log
   ```

### Service Status Not Updating

**Problem**: Dashboard shows stale status information

**Solution**:
1. Click refresh button in sidebar
2. Enable auto-refresh in Settings
3. Check that service processes are actually running:
   ```bash
   ps aux | grep paper_trade.py
   ```

## Advanced Usage

### Custom Refresh Interval

Enable auto-refresh in Settings and set your preferred interval (10-300 seconds).

### Running on Remote Server

To access dashboard from a remote machine:

```bash
streamlit run src/graphwiz_trader/paper_trading/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Then access via: `http://your-server-ip:8501`

### Exporting Charts

Charts are interactive using Plotly. To export:
1. Click the camera icon in the chart toolbar
2. Select download format (PNG, SVG, PDF)
3. Chart will be downloaded to your default download location

## Performance Tips

1. **Limit Symbols**: Comparison page can get slow with 5+ symbols
2. **Date Range**: For large datasets, consider adding date filters
3. **Auto-Refresh**: Disable auto-refresh if not needed to save resources
4. **Browser**: Use modern browser (Chrome, Firefox, Edge) for best performance

## Security Considerations

- Dashboard runs on localhost by default (not exposed externally)
- No authentication built-in (add nginx/traefik proxy for production)
- Read-only access to data files (cannot modify trading behavior)
- Service status monitoring requires process list access

## Future Enhancements

Planned features for future releases:

- [ ] Real-time WebSocket updates
- [ ] Alert system for significant events
- [ ] Built-in backtesting interface
- [ ] Export reports as PDF/Excel
- [ ] Mobile responsive design
- [ ] Historical parameter comparison
- [ ] Trade signal annotations on charts
- [ ] Custom date range selectors
- [ ] Portfolio aggregation view

## Support

For issues or questions:

1. Check logs in `logs/` directory
2. Review service documentation: `docs/SERVICE_SETUP.md`
3. Run diagnostic: `python scripts/diagnose_service.py`
4. Check GitHub issues: https://github.com/tobias-weiss-ai-xr/graphwiz-trader/issues
