# Monitoring System - Quick Reference

## File Locations

```
monitoring/
├── __init__.py          - Main exports
├── metrics.py           - Prometheus metrics collector
├── alerting.py          - Multi-channel alerting
├── health.py            - Health checks & recovery
├── dashboard.py         - Grafana & WebSocket
└── monitor.py           - Main orchestrator
```

## Quick Start

```python
from graphwiz_trader.monitoring import create_monitor

# 1. Create monitor
monitor = create_monitor(config)

# 2. Set dependencies (optional)
monitor.set_dependencies(
    exchange_manager=exchanges,
    neo4j_graph=graph,
    trading_engine=engine
)

# 3. Start monitoring
await monitor.start()

# 4. Record events
monitor.record_trade({'symbol': 'BTC/USDT', 'pnl': 100})
monitor.record_agent_prediction('agent1', {'prediction': 1, 'confidence': 0.8})

# 5. Query status
metrics = monitor.get_metrics()
health = monitor.get_health_status()

# 6. Stop when done
await monitor.stop()
```

## Prometheus Metrics

All metrics exposed at `http://localhost:8000/metrics`

### System Metrics
- `graphwiz_system_cpu_usage_percent` - CPU usage
- `graphwiz_system_memory_usage_bytes` - Memory used
- `graphwiz_system_disk_usage_percent` - Disk usage per mount

### Trading Metrics
- `graphwiz_trading_total_trades` - Total trade counter
- `graphwiz_trading_pnl_usd` - Total P&L
- `graphwiz_trading_win_rate` - Win rate percentage
- `graphwiz_trading_sharpe_ratio` - Sharpe ratio
- `graphwiz_trading_current_drawdown` - Current drawdown

### Exchange Metrics
- `graphwiz_exchange_latency_seconds` - API latency histogram
- `graphwiz_exchange_fill_rate` - Order fill rate per exchange
- `graphwiz_exchange_error_rate` - Error rate per exchange

### Agent Metrics
- `graphwiz_agent_accuracy` - Prediction accuracy per agent
- `graphwiz_agent_confidence` - Average confidence per agent

### Risk Metrics
- `graphwiz_risk_var_95_usd` - Value at Risk (95%)
- `graphwiz_risk_exposure_usd` - Current exposure
- `graphwiz_risk_leverage` - Current leverage

## Alert Rules

Built-in rules automatically checked:

| Rule | Condition | Severity | Cooldown |
|------|-----------|----------|----------|
| high_cpu | CPU > 90% | WARNING | 10m |
| high_memory | Memory > 90% | WARNING | 10m |
| exchange_disconnect | Exchange offline | CRITICAL | 1m |
| high_latency | P99 latency > 5s | WARNING | 5m |
| large_loss | Drawdown > 10% | CRITICAL | 5m |
| rate_limit | Rate limit < 10 | WARNING | 5m |
| neo4j_disconnect | Neo4j offline | CRITICAL | 1m |
| risk_limit_breach | Leverage > 3x | EMERGENCY | 10s |

## Notification Channels

### Discord
```yaml
notifications:
  discord:
    webhook_url: "https://discord.com/api/webhooks/..."
```

### Slack
```yaml
notifications:
  slack:
    webhook_url: "https://hooks.slack.com/services/..."
```

### Email
```yaml
notifications:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    from: "alerts@example.com"
    to: ["trader@example.com"]
```

### Telegram
```yaml
notifications:
  telegram:
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
```

## WebSocket Updates

Connect to `ws://localhost:8765` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'metrics') {
    console.log('Metrics:', data.data);
  } else if (data.type === 'alert') {
    console.log('Alert:', data.data);
  } else if (data.type === 'health') {
    console.log('Health:', data.data);
  }
};
```

## Health Checks

8 built-in health checks:

1. `exchange_connectivity` (30s) - Tests exchange APIs
2. `neo4j_connectivity` (60s) - Tests database
3. `api_rate_limits` (10s) - Monitors rate limits
4. `system_resources` (60s) - CPU, memory, disk
5. `disk_space` (300s) - Long-term disk monitoring
6. `agent_health` (60s) - Agent status
7. `portfolio_health` (30s) - Risk metrics
8. `database_locks` (60s) - Neo4j transactions

## Manual Alerts

```python
from graphwiz_trader.monitoring import AlertSeverity, AlertChannel

monitor.create_manual_alert(
    title="Custom Alert",
    message="Something important happened",
    severity=AlertSeverity.WARNING,
    channels=["DISCORD", "SLACK"],
    metadata={'key': 'value'}
)
```

## Configuration File Structure

```yaml
monitoring:
  prometheus_port: 8000
  metrics_interval_seconds: 15
  health_check_interval_seconds: 30
  alert_check_interval_seconds: 10

notifications:
  discord:
    webhook_url: "..."
  slack:
    webhook_url: "..."

alerts:
  cooldown_warning: 300
  cooldown_critical: 60
  custom_rules:
    my_rule:
      condition: "system['cpu']['percent'] > 95"
      severity: "CRITICAL"
      channels: ["DISCORD"]
```

## Common Queries

### Get current metrics
```python
metrics = monitor.get_metrics()
print(metrics['trading']['total_pnl'])
print(metrics['system']['cpu']['percent'])
```

### Get health status
```python
health = monitor.get_health_status()
print(health['overall_status'])  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
```

### Get active alerts
```python
alerts = monitor.get_active_alerts()
for alert in alerts:
    print(f"{alert.severity}: {alert.title}")
```

### Query historical data
```python
# Last 24 hours of P&L
data = await monitor.query_prometheus(
    query='graphwiz_trading_pnl_usd',
    hours=24
)
```

## Troubleshooting

### Prometheus not receiving metrics
- Check port 8000 is open
- Verify `enable_prometheus: true` in config
- Test: `curl http://localhost:8000/metrics`

### Alerts not sending
- Check notification channel configs
- Verify webhook URLs/tokens are correct
- Check alert hasn't triggered cooldown
- Test with `AlertChannel.LOG`

### Health checks failing
- Verify dependencies set with `set_dependencies()`
- Check service connectivity (Neo4j, exchanges)
- Review health check logs
- Reset circuit breaker if needed

## Performance Tuning

### Reduce overhead
```yaml
monitoring:
  metrics_interval_seconds: 30    # Increase from 15
  health_check_interval_seconds: 60  # Increase from 30
```

### Reduce memory
```python
# In metrics.py, reduce history
if len(self.trade_history) > 500:  # Was 1000
    self.trade_history = self.trade_history[-500:]
```

### Reduce alert spam
```yaml
alerts:
  cooldown_warning: 600    # 10 minutes
  cooldown_critical: 120   # 2 minutes
```

## Dependencies

Install required packages:
```bash
pip install prometheus-client psutil aiohttp websockets
```

Optional for email:
```bash
pip install secure-smtplib
```

## Export Locations

- Prometheus metrics: `http://localhost:8000/metrics`
- WebSocket updates: `ws://localhost:8765`
- Grafana dashboards: `dashboards/*.json`
- Documentation: `docs/MONITORING.md`
- Configuration: `monitoring_config.example.yaml`

## Key Classes

| Class | Purpose | File |
|-------|---------|------|
| `TradingMonitor` | Main orchestrator | monitor.py |
| `MetricsCollector` | Prometheus metrics | metrics.py |
| `AlertManager` | Alert management | alerting.py |
| `HealthChecker` | Health checks | health.py |
| `GrafanaDashboard` | Dashboard generation | dashboard.py |
| `RealTimeMonitor` | WebSocket server | dashboard.py |

## Event Callbacks

```python
# On alert
async def on_alert(alert):
    print(f"Alert: {alert.title}")

monitor.on_alert = on_alert

# On health change
async def on_health_change(health_summary):
    print(f"Health: {health_summary['overall_status']}")

monitor.on_health_change = on_health_change
```
