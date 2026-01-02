# Grafana Dashboards

This directory contains Grafana dashboard configurations for graphwiz-trader.

## Importing Dashboards

1. Open Grafana
2. Go to Dashboards -> Import
3. Upload the JSON file or paste the contents
4. Select your Prometheus data source
5. Click Import

## Available Dashboards

### Main Dashboard (`graphwiz-main.json`)
- System metrics (CPU, memory, disk, network)
- Trading metrics (P&L, win rate, Sharpe ratio, drawdown)
- Exchange metrics (latency, connection status)
- Agent metrics (accuracy, confidence)
- Risk metrics (VaR, exposure, leverage)
- Portfolio metrics (value, returns)

### Alerts Dashboard (`graphwiz-alerts.json`)
- Active alert count
- Alert timeline
- Alerts by severity
- Alert statistics

## Auto-Generation

Dashboards can also be auto-generated using Python:

```python
from graphwiz_trader.monitoring.dashboard import DashboardConfig, export_dashboard_to_file

# Generate main dashboard
dashboard = DashboardConfig.get_default_dashboard()
export_dashboard_to_file(dashboard, 'dashboards/graphwiz-main.json')

# Generate alerts dashboard
alerts_dashboard = DashboardConfig.get_alerts_dashboard()
export_dashboard_to_file(alerts_dashboard, 'dashboards/graphwiz-alerts.json')
```

## Customization

Edit the dashboard JSON files directly or modify the Python generation code in `src/graphwiz_trader/monitoring/dashboard.py`.
