# Monitoring and Alerting System

Comprehensive monitoring and alerting system for graphwiz-trader with real-time performance tracking, multi-channel notifications, and automated recovery actions.

## Architecture

The monitoring system consists of five main components:

1. **Metrics Collector** (`metrics.py`) - Prometheus-based metrics collection
2. **Alert Manager** (`alerting.py`) - Multi-channel alerting with deduplication
3. **Health Checker** (`health.py`) - System health checks with automated recovery
4. **Dashboard** (`dashboard.py`) - Grafana dashboards and real-time WebSocket updates
5. **Main Monitor** (`monitor.py`) - Orchestrates all monitoring components

## Features

### Metrics Collection

- **System Metrics**: CPU, memory, disk, network usage
- **Trading Metrics**: P&L, win rate, Sharpe ratio, drawdown
- **Exchange Metrics**: Latency, fill rate, error rate, rate limits
- **Agent Metrics**: Accuracy, confidence, prediction count
- **Risk Metrics**: VaR (95%, 99%), exposure, leverage, correlation
- **Portfolio Metrics**: Value, cash, positions, returns (1h, 24h, 7d)

All metrics are exported to Prometheus for visualization and alerting.

### Multi-Channel Alerting

Alerts can be sent to multiple channels simultaneously:

- **Discord** - Webhook integration with rich embeds
- **Slack** - Webhook integration with formatted messages
- **Email** - SMTP with HTML templates
- **Telegram** - Bot API integration
- **Webhook** - Custom webhook endpoints
- **Log** - Always available as fallback

### Alert Severity Levels

- **INFO** - Informational messages (cooldown: 10 minutes)
- **WARNING** - Warning conditions (cooldown: 5 minutes)
- **CRITICAL** - Critical failures (cooldown: 1 minute)
- **EMERGENCY** - Emergency situations (cooldown: 10 seconds)

### Built-in Alert Rules

1. **High CPU** - CPU usage > 90%
2. **High Memory** - Memory usage > 90%
3. **Exchange Disconnect** - Exchange connection lost
4. **High Latency** - Exchange P99 latency > 5s
5. **Large Loss** - Drawdown > 10%
6. **Rate Limit** - API rate limit < 10 requests
7. **Neo4j Disconnect** - Database connection lost
8. **Agent Failure** - Multiple agent failures
9. **Risk Limit Breach** - Leverage > 3x
10. **Disk Space** - Disk usage > 90%

### Health Checks

Automated health checks for:

- Exchange connectivity (every 30s)
- Neo4j connectivity (every 60s)
- API rate limits (every 10s)
- System resources (every 60s)
- Disk space (every 5 minutes)
- Agent health (every 60s)
- Portfolio health (every 30s)
- Database locks (every 60s)

### Automated Recovery Actions

When health checks fail, the system can automatically:

- **Reconnect** - Attempt to reconnect to services
- **Restart** - Restart failed agents/services
- **Scale Down** - Reduce trading activity
- **Pause Trading** - Halt all trading operations
- **Close Positions** - Emergency position closure
- **Clear Cache** - Clear system caches
- **Reset Rate Limits** - Reset rate limit counters
- **Notify Admin** - Send emergency notifications

### Circuit Breakers

Circuit breakers prevent cascade failures:

- Opens after 3 consecutive failures
- Automatic retry after 5 minutes
- Per-service circuit breaker tracking
- Manual reset capability

## Installation

Install required dependencies:

```bash
pip install prometheus-client psutil aiohttp websockets
```

For email notifications:
```bash
pip install secure-smtplib
```

## Configuration

Create a `monitoring.yaml` configuration file (see `monitoring_config.example.yaml`):

```yaml
monitoring:
  prometheus_port: 8000
  prometheus_url: "http://localhost:9090"
  enable_prometheus: true
  metrics_interval_seconds: 15
  health_check_interval_seconds: 30
  enable_realtime: true

notifications:
  discord:
    webhook_url: "https://discord.com/api/webhooks/YOUR_WEBHOOK"
  slack:
    webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK"
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    from: "trading@example.com"
    to:
      - "trader@example.com"
```

## Usage

### Basic Usage

```python
import asyncio
from graphwiz_trader.monitoring import create_monitor

async def main():
    # Load configuration
    config = {
        'monitoring': {
            'prometheus_port': 8000,
            'enable_prometheus': True
        }
    }

    # Create monitor
    monitor = create_monitor(config)

    # Set up dependencies (optional)
    monitor.set_dependencies(
        exchange_manager=exchange_mgr,
        neo4j_graph=graph,
        trading_engine=engine,
        agent_manager=agent_mgr
    )

    # Start monitoring
    await monitor.start()

    try:
        # Record trades
        monitor.record_trade({
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000,
            'quantity': 0.1,
            'pnl': 150.50,
            'success': True
        })

        # Record agent predictions
        monitor.record_agent_prediction(
            'momentum_agent',
            {
                'prediction': 1,
                'confidence': 0.85,
                'symbol': 'BTC/USDT'
            }
        )

        # Create manual alerts
        monitor.create_manual_alert(
            title="Custom Alert",
            message="Something happened",
            severity=AlertSeverity.WARNING
        )

        # Keep running
        await asyncio.sleep(3600)

    finally:
        await monitor.stop()

asyncio.run(main())
```

### Advanced Usage

#### Custom Alert Rules

```python
from graphwiz_trader.monitoring import AlertRule, AlertSeverity, AlertChannel

custom_rule = AlertRule(
    name="custom_drawdown",
    condition=lambda m: m.get('trading', {}).get('current_drawdown', 0) > 0.05,
    severity=AlertSeverity.WARNING,
    channels=[AlertChannel.DISCORD, AlertChannel.SLACK],
    title_template="Custom Drawdown Alert",
    message_template="Drawdown at {drawdown_percent}%",
    cooldown_seconds=600
)

monitor.alert_manager.custom_rules['custom_drawdown'] = custom_rule
```

#### Querying Historical Data

```python
# Query Prometheus for historical metrics
data = await monitor.query_prometheus(
    query='graphwiz_trading_pnl_usd',
    hours=24,
    resolution='1m'
)

# Get trading performance history
performance = await monitor.get_trading_performance_history(hours=24)
```

#### WebSocket Real-time Updates

```python
import websockets
import json

async def receive_updates():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data['type'] == 'metrics':
                print("Metrics update:", data['data'])
            elif data['type'] == 'alert':
                print("New alert:", data['data'])
            elif data['type'] == 'health':
                print("Health update:", data['data'])

asyncio.run(receive_updates())
```

## Grafana Integration

### Setting up Prometheus

1. Install Prometheus:
```bash
# Linux
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64
```

2. Configure `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'graphwiz-trader'
    static_configs:
      - targets: ['localhost:8000']
```

3. Start Prometheus:
```bash
./prometheus --config.file=prometheus.yml
```

### Importing Grafana Dashboards

1. Start Grafana (if not already running)
2. Go to Dashboards -> Import
3. Upload dashboard JSON from `dashboards/` directory
4. Select Prometheus as data source
5. Click Import

Alternatively, generate dashboards programmatically:

```python
from graphwiz_trader.monitoring.dashboard import (
    DashboardConfig,
    export_dashboard_to_file
)

# Generate main dashboard
dashboard = DashboardConfig.get_default_dashboard()
export_dashboard_to_file(dashboard, 'graphwiz-main.json')

# Generate alerts dashboard
alerts_dashboard = DashboardConfig.get_alerts_dashboard()
export_dashboard_to_file(alerts_dashboard, 'graphwiz-alerts.json')
```

## API Reference

### TradingMonitor

Main monitoring system class.

#### Methods

- `async start()` - Start monitoring system
- `async stop()` - Stop monitoring system
- `record_trade(trade: Dict)` - Record a trade
- `record_agent_prediction(agent_name: str, prediction: Dict, actual_outcome: Optional[float])` - Record prediction
- `create_manual_alert(title: str, message: str, severity: AlertSeverity, ...)` - Create manual alert
- `get_metrics() -> Dict` - Get current metrics
- `get_health_status() -> Dict` - Get health summary
- `get_active_alerts() -> List` - Get active alerts
- `get_alert_history(hours: int = 24) -> List` - Get alert history
- `async query_prometheus(query: str, hours: int = 24) -> Dict` - Query Prometheus

### MetricsCollector

Collects and exports metrics to Prometheus.

#### Methods

- `collect_system_metrics() -> Dict` - Collect system metrics
- `record_trade(trade: Dict)` - Record trade and update metrics
- `calculate_trading_metrics(lookback_hours: int = 24) -> Dict` - Calculate trading metrics
- `record_exchange_latency(exchange: str, latency: float)` - Record exchange latency
- `record_agent_prediction(agent_name: str, prediction: Dict, actual_outcome: Optional[float])` - Record prediction
- `update_portfolio_metrics(portfolio: Dict)` - Update portfolio metrics
- `calculate_risk_metrics(portfolio_value: float, positions: List[Dict]) -> Dict` - Calculate risk metrics
- `get_all_metrics() -> Dict` - Get all collected metrics

### AlertManager

Manages alerts and notifications.

#### Methods

- `check_alert(rule_name: str, metrics: Dict, context: Optional[Dict]) -> Optional[Alert]` - Check if alert should trigger
- `async send_alert(alert: Alert) -> bool` - Send alert to channels
- `resolve_alert(rule_name: str)` - Mark alert as resolved
- `get_active_alerts() -> List[Alert]` - Get active alerts
- `get_alert_history(hours: int = 24, severity: Optional[AlertSeverity]) -> List[Alert]` - Get alert history

### HealthChecker

Performs health checks and recovery actions.

#### Methods

- `async run_health_check(check_name: str) -> HealthResult` - Run specific health check
- `async run_all_health_checks() -> Dict[str, HealthResult]` - Run all health checks
- `get_health_summary() -> Dict` - Get health summary
- `reset_circuit_breaker(check_name: str)` - Reset circuit breaker

## Best Practices

1. **Set Appropriate Thresholds** - Customize alert thresholds based on your trading strategy and risk tolerance
2. **Use Multiple Channels** - Configure critical alerts to use multiple notification channels
3. **Monitor Rate Limits** - Keep an eye on API rate limits to avoid being throttled
4. **Regular Health Checks** - Health checks should run frequently enough to catch issues early
5. **Test Recovery Actions** - Test recovery actions in a safe environment before relying on them
6. **Review Alert History** - Regularly review alert history to identify patterns and improve configurations
7. **Dashboard Customization** - Customize Grafana dashboards to show the metrics most important to your strategy

## Troubleshooting

### Prometheus Not Receiving Metrics

- Check if Prometheus server is running
- Verify Prometheus configuration includes correct target (localhost:8000)
- Check firewall settings
- Verify `enable_prometheus: true` in configuration

### Alerts Not Being Sent

- Verify notification channel configurations (webhook URLs, tokens, etc.)
- Check alert cooldown periods
- Verify alert conditions are being met
- Check logs for error messages
- Test with AlertChannel.LOG to verify alerts are being generated

### Health Checks Failing

- Verify dependencies are set correctly with `set_dependencies()`
- Check service connectivity (Neo4j, exchanges)
- Review health check logs
- Check circuit breaker status with `get_circuit_breaker_status()`

### High Memory Usage

- Reduce metrics history retention
- Increase alert cooldown periods
- Reduce health check frequency
- Review and optimize Prometheus queries

## Performance Considerations

- **Metrics Collection**: Runs every 15 seconds by default, adjust based on needs
- **Health Checks**: Asynchronous execution prevents blocking
- **Alert Deduplication**: Prevents alert spam during sustained issues
- **Circuit Breakers**: Prevents cascade failures from repeated attempts
- **WebSocket**: Efficient real-time updates without polling

## Security

- Store sensitive credentials (API tokens, passwords) in environment variables or secure vaults
- Use HTTPS/TLS for webhook connections
- Rotate API tokens regularly
- Restrict WebSocket server to trusted networks in production
- Monitor alert logs for suspicious activity

## License

MIT License - See project root for details
