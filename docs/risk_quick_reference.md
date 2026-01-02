# Risk Management System - Quick Reference Guide

## File Structure

```
src/graphwiz_trader/risk/
├── __init__.py              # Public API exports
├── manager.py               # RiskManager orchestrator (723 lines)
├── calculators.py           # Risk calculations (764 lines)
├── limits.py                # Risk limits & validation (659 lines)
└── alerts.py                # Alerting system (673 lines)

tests/
└── test_risk_manager.py     # Comprehensive tests (598 lines)

docs/
├── risk_management.md                # Full documentation
└── risk_implementation_summary.md    # Implementation details

examples/
└── risk_management_demo.py           # Working examples
```

## Quick Start

### 1. Initialize Risk Manager

```python
from graphwiz_trader.risk import RiskManager, RiskLimitsConfig

config = RiskLimitsConfig(
    max_position_size=0.10,
    max_total_exposure=1.0,
    max_daily_loss_pct=0.05,
)

rm = RiskManager(account_balance=100000.0, limits_config=config)
```

### 2. Calculate Position Size

```python
from graphwiz_trader.risk import PositionSizingStrategy

result = rm.calculate_position_size(
    symbol="BTC",
    entry_price=50000.0,
    stop_loss_price=49000.0,
    strategy=PositionSizingStrategy.FIXED_FRACTIONAL,
)
# Returns: position_size, position_value, dollar_risk, etc.
```

### 3. Add Position

```python
position = rm.add_position(
    symbol="BTC",
    quantity=result['position_size'],
    entry_price=50000.0,
    side="long",
    sector="Crypto",
    stop_loss=49000.0,
    take_profit=56000.0,
)
```

### 4. Update Prices

```python
rm.update_position_price("BTC", 51000.0)
# Automatically checks stop-loss and take-profit
```

### 5. Close Position

```python
pnl = rm.close_position("BTC")
# Returns realized P&L
```

## Position Sizing Strategies

| Strategy | Description | Parameters |
|----------|-------------|------------|
| Fixed Fractional | Risk fixed % per trade | risk_per_trade |
| Kelly Criterion | Optimal sizing based on edge | win_rate, avg_win, avg_loss, kelly_fraction |
| Fixed Dollar | Risk fixed dollar amount | fixed_amount |
| Volatility Target | Scale by volatility | volatility, target_volatility |
| Risk Parity | Equal risk contribution | num_positions |

## Risk Limits

### Position-Level
- `max_position_size`: Max % per position (default: 10%)
- `max_position_size_abs`: Max absolute dollar amount
- `min_position_size`: Min position size

### Portfolio-Level
- `max_total_exposure`: Max total exposure (default: 100%)
- `max_long_exposure`: Max long exposure
- `max_short_exposure`: Max short exposure
- `max_gross_exposure`: Max gross exposure (long + short)
- `max_net_exposure`: Max net exposure (long - short)

### Risk Limits
- `max_daily_loss_pct`: Max daily loss % (default: 5%)
- `max_drawdown_pct`: Max drawdown % (default: 20%)
- `max_correlated_exposure`: Max correlated exposure (default: 30%)
- `correlation_threshold`: Correlation threshold (default: 0.70)

### Sector Limits
- `max_sector_exposure`: Max per sector (default: 40%)
- `max_asset_class_exposure`: Max per asset class (default: 60%)
- `max_single_asset_concentration`: Max single asset (default: 20%)

### Trading Limits
- `max_trades_per_day`: Max trades per day (default: 100)
- `max_turnover_pct`: Max daily turnover (default: 50%)
- `max_drawdown_duration_days`: Max days in drawdown (default: 90)

## Risk Metrics

### Portfolio Risk
- `portfolio_value`: Total portfolio value
- `var_95`: Value at Risk (95% confidence)
- `cvar_95`: Conditional VaR (Expected Shortfall)
- `portfolio_std`: Portfolio standard deviation
- `worst_case_loss`: Maximum historical loss

### Drawdown Metrics
- `max_drawdown`: Maximum drawdown %
- `max_drawdown_abs`: Maximum drawdown $
- `max_drawdown_duration`: Duration in days
- `current_drawdown`: Current drawdown %
- `avg_drawdown`: Average drawdown %

### Other Metrics
- `portfolio_beta`: Portfolio beta vs benchmark
- `sharpe_ratio`: Risk-adjusted return (using std dev)
- `sortino_ratio`: Downside risk-adjusted return

## Alert Types

| Type | Severity | Trigger |
|------|----------|---------|
| POSITION_SIZE_EXCEEDED | Warning | Position > max_position_size |
| EXPOSURE_LIMIT_EXCEEDED | Critical | Total exposure > limit |
| DAILY_LOSS_LIMIT | Critical | Daily loss > max_daily_loss |
| DRAWDOWN_WARNING | Critical | Drawdown > max_drawdown |
| CORRELATION_RISK | Warning | High correlated exposure |
| CONCENTRATION_RISK | Warning | Sector/asset concentration |
| VOLATILITY_SPIKE | Warning | Volatility threshold exceeded |
| TRADING_LIMIT_EXCEEDED | Warning | Trades > max_trades_per_day |
| MARGIN_CALL_WARNING | Emergency | Margin account warning |
| PORTFOLIO_REBALANCE_NEEDED | Info | Portfolio needs rebalancing |

## Notification Channels

```python
# Console (default, always enabled)
# No configuration needed

# Discord
from graphwiz_trader.risk import DiscordNotificationChannel
discord = DiscordNotificationChannel(webhook_url="your_webhook_url")
rm.alert_manager.add_channel(discord)

# Slack
from graphwiz_trader.risk import SlackNotificationChannel
slack = SlackNotificationChannel(webhook_url="your_webhook_url")
rm.alert_manager.add_channel(slack)

# Email
from graphwiz_trader.risk import EmailNotificationChannel
email = EmailNotificationChannel(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    sender_email="your_email@gmail.com",
    sender_password="your_password",
    recipient_emails=["recipient@example.com"],
)
rm.alert_manager.add_channel(email)
```

## Common Tasks

### Check Portfolio State

```python
state = rm.get_portfolio_state()
print(f"Total Value: ${state.total_value:,.2f}")
print(f"Cash: ${state.cash_balance:,.2f}")
print(f"Positions: {len(state.positions)}")
print(f"Daily P&L: ${state.daily_pnl:,.2f}")
```

### Calculate Portfolio Risk

```python
risk = rm.calculate_portfolio_risk(confidence_level=0.95)
print(f"VaR (95%): ${risk['var_95']:,.2f}")
print(f"CVaR: ${risk['cvar_95']:,.2f}")
```

### Get Correlation Matrix

```python
corr = rm.get_correlation_matrix()
print(f"BTC-ETH: {corr[('BTC', 'ETH')]:.3f}")
```

### Check Alerts

```python
alerts = rm.alert_manager.get_active_alerts()
for alert in alerts:
    print(f"{alert.severity.value}: {alert.message}")

# Get statistics
stats = rm.alert_manager.get_alert_statistics()
print(f"Total Alerts: {stats['total_alerts']}")
print(f"Active: {stats['active_alerts']}")
```

### Reset Daily Metrics

```python
rm.reset_daily_metrics()  # Call at start of trading day
```

## Knowledge Graph Integration

### Stored in Neo4j

**Nodes:**
- `Position`: Open positions
- `ClosedTrade`: Historical trades
- `PortfolioRisk`: Risk metrics
- `RiskAlert`: Alerts and violations

**Example Queries:**

```cypher
// Current positions
MATCH (p:Position)
RETURN p.symbol, p.quantity, p.entry_price, p.current_price

// Recent alerts
MATCH (a:RiskAlert)
WHERE a.timestamp > datetime() - duration('P1D')
RETURN a.alert_type, a.severity, a.message

// Portfolio risk history
MATCH (r:PortfolioRisk)
RETURN r.timestamp, r.var_95, r.cvar_95
ORDER BY r.timestamp DESC LIMIT 100

// Closed trades by symbol
MATCH (t:ClosedTrade {symbol: "BTC"})
RETURN t.entry_price, t.exit_price, t.pnl
ORDER BY t.close_time DESC
```

## Error Handling

### Position Rejected

```python
try:
    result = rm.calculate_position_size(...)
    position = rm.add_position(...)
except ValueError as e:
    print(f"Trade rejected: {e}")
    # Handle rejection (reduce size, skip trade, etc.)
```

### Limit Breach

```python
# Register custom handler
def handle_limit_breach(alert):
    if alert.severity == AlertSeverity.CRITICAL:
        # Stop trading
        # Close positions
        # Notify team
        pass

rm.alert_manager.register_handler(
    AlertType.EXPOSURE_LIMIT_EXCEEDED,
    handle_limit_breach,
)
```

## Testing

```bash
# Run all risk tests
pytest tests/test_risk_manager.py -v

# Run specific test
pytest tests/test_risk_manager.py::TestPositionSizing -v

# With coverage
pytest tests/test_risk_manager.py --cov=src/graphwiz_trader/risk
```

## Best Practices

1. **Always use position sizing** - Never skip size calculation
2. **Set stop-losses** - Define exit points before entry
3. **Use alerts** - Set up notifications for critical limits
4. **Monitor daily** - Check risk metrics every day
5. **Track metrics** - Store everything in knowledge graph
6. **Test changes** - Verify risk parameter changes in simulation
7. **Use hard limits** - Don't override risk checks
8. **Diversify** - Use correlation analysis to avoid concentration

## Configuration Templates

### Conservative (Low Risk)
```python
RiskLimitsConfig(
    max_position_size=0.05,
    max_total_exposure=0.80,
    max_daily_loss_pct=0.02,
    max_drawdown_pct=0.10,
    max_correlated_exposure=0.20,
)
```

### Balanced (Medium Risk)
```python
RiskLimitsConfig(
    max_position_size=0.10,
    max_total_exposure=1.0,
    max_daily_loss_pct=0.05,
    max_drawdown_pct=0.20,
    max_correlated_exposure=0.30,
)
```

### Aggressive (High Risk)
```python
RiskLimitsConfig(
    max_position_size=0.20,
    max_total_exposure=1.50,
    max_daily_loss_pct=0.10,
    max_drawdown_pct=0.30,
    max_correlated_exposure=0.40,
)
```

## Support

- Full documentation: `/docs/risk_management.md`
- Implementation details: `/docs/risk_implementation_summary.md`
- Working examples: `/examples/risk_management_demo.py`
- Test cases: `/tests/test_risk_manager.py`
