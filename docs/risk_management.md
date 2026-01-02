# Risk Management System

Comprehensive risk management framework for graphwiz-trader with position sizing, portfolio monitoring, correlation analysis, and exposure limits.

## Features

### 1. Position Sizing Strategies

**Multiple Strategies Supported:**
- **Fixed Fractional**: Risk fixed percentage of capital per trade
- **Kelly Criterion**: Optimal position sizing based on win rate and risk/reward
- **Fixed Dollar**: Risk fixed dollar amount per trade
- **Volatility Target**: Scale positions based on asset volatility
- **Risk Parity**: Equal risk contribution across positions

**Example:**
```python
from graphwiz_trader.risk import calculate_position_size, PositionSizingStrategy

result = calculate_position_size(
    account_balance=100000.0,
    entry_price=50000.0,
    stop_loss_price=49000.0,  # 2% stop loss
    risk_per_trade=0.02,      # 2% risk
    strategy=PositionSizingStrategy.FIXED_FRACTIONAL,
)

print(f"Position Size: {result['position_size']:.4f} BTC")
print(f"Position Value: ${result['position_value']:,.2f}")
print(f"Dollar Risk: ${result['dollar_risk']:,.2f}")
```

### 2. Portfolio Risk Monitoring

**Risk Metrics Calculated:**
- **VaR (Value at Risk)**: Maximum expected loss at confidence level
- **CVaR (Conditional VaR)**: Expected loss beyond VaR
- **Portfolio Standard Deviation**: Overall portfolio volatility
- **Worst Case Loss**: Maximum historical loss
- **Beta**: Portfolio sensitivity to benchmark
- **Sharpe/Sortino Ratios**: Risk-adjusted return metrics

**Example:**
```python
risk_metrics = risk_manager.calculate_portfolio_risk(
    confidence_level=0.95,
    method="historical",  # or 'parametric', 'monte_carlo'
)

print(f"Portfolio VaR (95%): ${risk_metrics['var_95']:,.2f}")
print(f"Portfolio CVaR: ${risk_metrics['cvar_95']:,.2f}")
print(f"Portfolio Std Dev: {risk_metrics['portfolio_std']:.4f}")
```

### 3. Correlation Analysis

**Capabilities:**
- Calculate correlation matrix for portfolio assets
- Identify highly correlated position clusters
- Limit exposure to correlated assets
- Support for Pearson, Spearman, and Kendall correlations

**Example:**
```python
correlation_matrix = risk_manager.get_correlation_matrix()

# Access correlation between two assets
correlation = correlation_matrix[('BTC', 'ETH')]
print(f"BTC-ETH Correlation: {correlation:.3f}")
```

### 4. Risk Limits and Validation

**Configurable Limits:**
- Maximum position size (percentage and absolute)
- Maximum total exposure (gross and net)
- Maximum daily loss limit
- Maximum correlated exposure
- Sector and asset class concentration limits
- Maximum drawdown limits
- Trading frequency limits
- Leverage limits

**Example:**
```python
from graphwiz_trader.risk import RiskLimitsConfig, RiskManager

config = RiskLimitsConfig(
    max_position_size=0.10,          # 10% max per position
    max_total_exposure=1.0,          # 100% max total exposure
    max_daily_loss_pct=0.05,         # 5% max daily loss
    max_correlated_exposure=0.30,    # 30% max in correlated assets
    max_sector_exposure=0.40,        # 40% max per sector
)

risk_manager = RiskManager(
    account_balance=100000.0,
    limits_config=config,
)
```

### 5. Stop-Loss and Take-Profit Calculator

**Features:**
- Percentage-based stops
- ATR (Average True Range) based stops
- Support/resistance level stops
- Trailing stop-loss calculation
- Risk/reward ratio optimization

**Example:**
```python
from graphwiz_trader.risk import StopLossCalculator

calculator = StopLossCalculator(
    default_stop_loss_pct=0.02,
    default_take_profit_pct=0.06,
    risk_reward_ratio=3.0,
)

# Calculate stop loss
stop_loss = calculator.calculate_stop_loss(
    entry_price=100.0,
    side="long",
    stop_loss_pct=0.02,
)

# Calculate take profit based on R:R ratio
take_profit = calculator.calculate_take_profit(
    entry_price=100.0,
    stop_loss_price=98.0,
    side="long",
)

# Calculate trailing stop
trailing_stop = calculator.calculate_trailing_stop(
    entry_price=100.0,
    current_price=110.0,
    side="long",
    trailing_distance_pct=0.03,
)
```

### 6. Risk Alerting System

**Alert Types:**
- Position size exceeded
- Exposure limit exceeded
- Daily loss limit
- Drawdown warning
- Correlation risk
- Concentration risk
- Volatility spike
- Trading limit exceeded

**Notification Channels:**
- Console logging (default)
- Discord webhooks
- Email notifications
- Slack webhooks

**Example:**
```python
from graphwiz_trader.risk import RiskAlertManager, AlertType, AlertSeverity

alert_manager = risk_manager.alert_manager

# Add Discord notifications
from graphwiz_trader.risk import DiscordNotificationChannel

discord_channel = DiscordNotificationChannel(
    webhook_url="your_webhook_url"
)
alert_manager.add_channel(discord_channel)

# Set custom thresholds
from graphwiz_trader.risk import AlertThreshold

alert_manager.set_threshold(
    AlertThreshold(
        metric_name="drawdown",
        warning_threshold=0.05,   # 5%
        critical_threshold=0.10,  # 10%
        emergency_threshold=0.15, # 15%
    )
)
```

## Usage Guide

### Basic Setup

```python
from graphwiz_trader.risk import RiskManager, RiskLimitsConfig

# Configure risk limits
config = RiskLimitsConfig(
    max_position_size=0.20,
    max_total_exposure=1.0,
    max_daily_loss_pct=0.05,
)

# Initialize risk manager
risk_manager = RiskManager(
    account_balance=100000.0,
    limits_config=config,
    knowledge_graph=your_kg_instance,  # Optional
)
```

### Adding Positions

```python
# Calculate position size first
size_result = risk_manager.calculate_position_size(
    symbol="BTC",
    entry_price=50000.0,
    stop_loss_price=49000.0,
    strategy=PositionSizingStrategy.FIXED_FRACTIONAL,
)

# Add position
position = risk_manager.add_position(
    symbol="BTC",
    quantity=size_result['position_size'],
    entry_price=50000.0,
    side="long",
    sector="Crypto",
    asset_class="Cryptocurrency",
    stop_loss=49000.0,
    take_profit=56000.0,
)
```

### Monitoring Portfolio

```python
# Get portfolio state
state = risk_manager.get_portfolio_state()
print(f"Total Value: ${state.total_value:,.2f}")
print(f"Number of Positions: {len(state.positions)}")
print(f"Daily P&L: ${state.daily_pnl:,.2f}")

# Calculate risk metrics
risk_metrics = risk_manager.calculate_portfolio_risk()
print(f"Portfolio VaR: ${risk_metrics['var_95']:,.2f}")

# Check drawdown
drawdown = risk_manager.calculate_max_drawdown()
print(f"Max Drawdown: {drawdown['max_drawdown']:.2%}")
print(f"Current Drawdown: {drawdown['current_drawdown']:.2%}")
```

### Updating Prices

```python
# Update position prices
risk_manager.update_position_price("BTC", 51000.0)

# Check if stop-loss or take-profit triggered
# (Automatically handled by update_position_price)
```

### Closing Positions

```python
# Close position at current price
pnl = risk_manager.close_position("BTC")

# Close position at specific price
pnl = risk_manager.close_position("ETH", exit_price=3200.0)
```

### Alert Management

```python
# Get active alerts
active_alerts = risk_manager.alert_manager.get_active_alerts()

# Get alert statistics
stats = risk_manager.alert_manager.get_alert_statistics()
print(f"Total Alerts: {stats['total_alerts']}")
print(f"Active Alerts: {stats['active_alerts']}")

# Acknowledge alert
risk_manager.alert_manager.acknowledge_alert(alert_index=0)

# Resolve alert
risk_manager.alert_manager.resolve_alert(alert_index=0)
```

## Knowledge Graph Integration

All risk metrics are automatically stored in the knowledge graph:

**Nodes Created:**
- `Position`: Current open positions
- `ClosedTrade`: Historical trades
- `PortfolioRisk`: Portfolio risk metrics
- `RiskAlert`: Risk alerts and violations

**Example Query:**
```cypher
// Get all positions with risk metrics
MATCH (p:Position)
RETURN p.symbol, p.quantity, p.entry_price, p.current_price, p.side

// Get recent risk alerts
MATCH (a:RiskAlert)
WHERE a.timestamp > datetime() - duration('P7D')
RETURN a.alert_type, a.severity, a.message
ORDER BY a.timestamp DESC

// Get portfolio risk history
MATCH (r:PortfolioRisk)
RETURN r.timestamp, r.var_95, r.cvar_95, r.portfolio_std
ORDER BY r.timestamp DESC
LIMIT 100
```

## Configuration

### Risk Limits Configuration

```python
config = RiskLimitsConfig(
    # Position limits
    max_position_size=0.10,              # Max 10% per position
    max_position_size_abs=None,          # Absolute dollar limit
    min_position_size=0.001,             # Min 0.1% per position

    # Portfolio limits
    max_total_exposure=1.0,              # Max 100% total exposure
    max_long_exposure=1.0,               # Max long exposure
    max_short_exposure=0.5,              # Max short exposure

    # Daily loss limits
    max_daily_loss_pct=0.05,             # Max 5% daily loss
    max_daily_loss_abs=None,             # Absolute dollar limit

    # Correlation limits
    max_correlated_exposure=0.30,        # Max 30% in correlated assets
    correlation_threshold=0.70,          # Correlation threshold

    # Sector limits
    max_sector_exposure=0.40,            # Max 40% per sector
    max_asset_class_exposure=0.60,       # Max 60% per asset class
    max_single_asset_concentration=0.20, # Max 20% in single asset

    # Drawdown limits
    max_drawdown_pct=0.20,               # Max 20% drawdown
    max_drawdown_duration_days=90,       # Max days in drawdown

    # Trading limits
    max_trades_per_day=100,              # Max trades per day
    max_turnover_pct=0.50,               # Max 50% daily turnover

    # Leverage limits
    max_gross_exposure=1.5,              # Max gross exposure
    max_net_exposure=1.0,                # Max net exposure
)
```

### Position Sizing Parameters

**Fixed Fractional:**
- `risk_per_trade`: Percentage of capital to risk (default: 0.02)

**Kelly Criterion:**
- `win_rate`: Historical win rate (default: 0.55)
- `avg_win`: Average win as multiple of risk (default: 1.5)
- `avg_loss`: Average loss as multiple of risk (default: 1.0)
- `kelly_fraction`: Fraction of Kelly to use (default: 0.5 for half-Kelly)

**Volatility Target:**
- `volatility`: Asset volatility (default: 0.02)
- `target_volatility`: Target portfolio volatility (default: 0.01)

**Risk Parity:**
- `num_positions`: Number of positions in portfolio (default: 10)

## Testing

Run the test suite:

```bash
pytest tests/test_risk_manager.py -v
```

Run the demonstration script:

```bash
python examples/risk_management_demo.py
```

## Best Practices

1. **Always Use Position Sizing**: Never enter a trade without calculating proper position size based on risk
2. **Set Stop Losses**: Always define exit points before entering trades
3. **Diversify**: Use correlation analysis to avoid concentration in correlated assets
4. **Monitor Drawdowns**: Track current drawdown and avoid overtrading during drawdowns
5. **Respect Limits**: Configure hard limits that cannot be exceeded
6. **Use Alerts**: Set up notifications for critical risk breaches
7. **Track Metrics**: Store all risk metrics in the knowledge graph for analysis
8. **Review Regularly**: Analyze risk metrics and adjust limits as needed

## Advanced Features

### Custom Risk Handlers

```python
def handle_drawdown_alert(alert):
    """Custom handler for drawdown alerts."""
    if alert.severity == AlertSeverity.CRITICAL:
        # Reduce position sizes
        # Stop new entries
        # Notify team
        pass

alert_manager.register_handler(
    AlertType.DRAWDOWN_WARNING,
    handle_drawdown_alert,
)
```

### Multi-Strategy Position Sizing

```python
# Calculate sizes with multiple strategies
strategies = [
    PositionSizingStrategy.FIXED_FRACTIONAL,
    PositionSizingStrategy.KELLY_CRITERION,
    PositionSizingStrategy.VOLATILITY_TARGET,
]

sizes = []
for strategy in strategies:
    result = risk_manager.calculate_position_size(
        symbol="BTC",
        entry_price=50000.0,
        stop_loss_price=49000.0,
        strategy=strategy,
    )
    sizes.append(result['position_size'])

# Use minimum or average of strategies
final_size = min(sizes)  # Conservative approach
```

## Performance Considerations

- Correlation matrix is cached for 15 minutes
- Price history is stored for all positions
- VaR calculations can use historical, parametric, or Monte Carlo methods
- For large portfolios, consider using parametric VaR for speed

## License

MIT License - See project root for details.
