# Trading Modes Implementation

This document describes the comprehensive trading mode system implemented for graphwiz-trader, including paper trading, simulated trading, and live trading modes with extensive safety checks and transition management.

## Overview

The trading mode system provides three distinct modes with progressive safety layers:

1. **Paper Trading Mode** (Default) - Simulated execution with real market data
2. **Simulated Trading Mode** - Historical data replay for backtesting
3. **Live Trading Mode** - Real trading with actual funds (requires explicit activation)

## Architecture

### Components

#### 1. Trading Mode Manager (`modes.py`)

**Location**: `/opt/git/graphwiz-trader/src/graphwiz_trader/trading/modes.py`

**Key Features**:
- `TradingMode` enum (PAPER, SIMULATED, LIVE)
- Mode switching with validation
- Emergency stop functionality
- Comprehensive logging of all mode changes
- Knowledge graph integration for audit trail

**Usage**:
```python
from graphwiz_trader.trading import TradingModeManager, TradingMode

# Initialize with knowledge graph
mode_manager = TradingModeManager(
    knowledge_graph=kg,
    config=trading_mode_config,
    approval_callback=approval_function
)

# Switch to live trading (requires approval)
await mode_manager.switch_mode(TradingMode.LIVE, reason="Production deployment")

# Emergency stop
await mode_manager.emergency_stop_mode(reason="Excessive losses detected")

# Check current mode
if mode_manager.is_live_trading():
    logger.warning("LIVE TRADING IS ACTIVE")
```

**Key Methods**:
- `get_current_mode()` - Returns current trading mode
- `is_paper_trading()` - Check if in paper mode
- `is_live_trading()` - Check if in live mode
- `can_execute_live_trades()` - Check if live trades can execute
- `switch_mode()` - Switch between trading modes
- `emergency_stop_mode()` - Activate emergency stop
- `clear_emergency_stop()` - Clear emergency stop status
- `get_mode_history()` - Get audit trail of mode changes

#### 2. Paper Trading Engine (`paper_trading.py`)

**Location**: `/opt/git/graphwiz-trader/src/graphwiz_trader/trading/paper_trading.py`

**Key Features**:
- Realistic order execution with slippage simulation
- Virtual portfolio management
- Performance tracking and analytics
- Real-time market data integration
- Readiness assessment for live trading

**Slippage Models**:
- `realistic` - Variable slippage based on order size and market conditions (default)
- `fixed` - Fixed slippage percentage
- `none` - No slippage (ideal execution)

**Usage**:
```python
from graphwiz_trader.trading import PaperTradingEngine

# Initialize with virtual balance
paper_engine = PaperTradingEngine(
    initial_balance={"USDT": 10000.0},
    knowledge_graph=kg,
    config=paper_trading_config
)

# Execute a paper trade
result = await paper_engine.execute_order(
    order=order,
    market_price=50000.0
)

# Get performance metrics
metrics = paper_engine.calculate_performance_metrics(current_prices)

# Check readiness for live trading
readiness = await paper_engine.check_readiness_for_live_trading()
if readiness["ready_for_live_trading"]:
    print("Ready to transition to live trading!")
```

**Performance Metrics**:
- Total trades, winning/losing trades
- Win rate percentage
- Total return percentage
- Realized and unrealized P&L
- Maximum drawdown
- Sharpe ratio
- Profit factor

#### 3. Safety Checks Module (`safety.py`)

**Location**: `/opt/git/graphwiz-trader/src/graphwiz_trader/trading/safety.py`

**Key Features**:
- Pre-trade validation checklist
- Position size limits
- Daily loss limits (hard and soft)
- Exchange balance checks
- API key validation
- Rate limit checks
- Circuit breakers for extreme market conditions
- Symbol whitelist/blacklist support

**Safety Checks**:
1. **Daily Limits**
   - Maximum daily trades
   - Daily loss limits (soft: 3%, hard: 5%)
   - Trade frequency limits

2. **Position Size**
   - Maximum position as percentage of portfolio
   - Maximum absolute position value
   - Portfolio concentration limits

3. **API Safety**
   - Rate limit enforcement (burst and sustained)
   - API key validation
   - Exchange connectivity checks

4. **Market Conditions**
   - Volatility thresholds
   - Liquidity requirements
   - Circuit breaker activation

5. **Balance Checks**
   - Minimum balance thresholds
   - Sufficient funds verification

**Usage**:
```python
from graphwiz_trader.trading import SafetyChecks

# Initialize safety checks
safety = SafetyChecks(
    knowledge_graph=kg,
    config=safety_config,
    violation_callback=handle_violation
)

# Pre-trade validation
passed, violations = await safety.pre_trade_validation(
    symbol="BTC/USDT",
    side="buy",
    amount=0.1,
    price=50000.0,
    portfolio_value=10000.0,
    current_positions={},
    exchange_name="binance"
)

if not passed:
    for violation in violations:
        logger.error("Safety violation: {}", violation.message)

# Check API keys
valid, message = await safety.validate_api_keys(exchange_config)

# Activate circuit breaker
await safety.activate_circuit_breaker(reason="Extreme market volatility")
```

**Violation Severity Levels**:
- `critical` - Trading must halt (e.g., daily loss limit exceeded)
- `warning` - Warning issued but trading continues
- `info` - Informational only

#### 4. Transition Manager (`transition.py`)

**Location**: `/opt/git/graphwiz-trader/src/graphwiz_trader/trading/transition.py`

**Key Features**:
- Paper trading validation requirements
- Gradual transition to live trading
- Performance monitoring during transition
- Automatic rollback triggers
- Knowledge graph tracking

**Transition Requirements**:
- Minimum 3 days of paper trading
- Minimum 100 trades executed
- Maximum drawdown < 10%
- Win rate > 55%
- Sharpe ratio > 1.5
- 2+ consecutive profitable days

**Gradual Transition Stages**:
1. Start with 10% of capital
2. After 3 days, increase to 25%
3. After 3 more days, increase to 50%
4. After 3 more days, increase to 100%

**Rollback Triggers**:
- Drawdown exceeds 5% during transition
- 5+ consecutive losses
- Circuit breaker activated
- Safety violations detected

**Usage**:
```python
from graphwiz_trader.trading import TransitionManager

# Initialize transition manager
transition_mgr = TransitionManager(
    paper_engine=paper_engine,
    safety_checks=safety,
    knowledge_graph=kg,
    config=transition_config,
    alert_callback=send_alert
)

# Validate paper trading readiness
validation = await transition_mgr.validate_paper_trading_readiness()
if validation["ready_for_transition"]:
    print("Ready for live trading!")
    print(validation["recommendations"])

# Start gradual transition
result = await transition_mgr.start_gradual_transition(initial_capital_pct=10.0)

# Increase capital allocation after validation
result = await transition_mgr.increase_capital_allocation()

# Check for rollback conditions
should_rollback, reason = await transition_mgr.check_rollback_conditions(live_performance)
if should_rollback:
    await transition_mgr.execute_rollback(reason)

# Get transition status
status = transition_mgr.get_transition_status()
```

## Configuration

### Trading Modes Configuration

**Location**: `/opt/git/graphwiz-trader/config/trading_modes.yaml`

**Key Sections**:

1. **Mode Settings**
   - Paper, simulated, and live trading configurations
   - Execution parameters (slippage, fees, delays)
   - Approval requirements

2. **Transition Requirements**
   - Minimum paper trading criteria
   - Gradual capital allocation steps
   - Monitoring intervals
   - Rollback thresholds

3. **Safety Checks**
   - Position size limits
   - Daily loss limits
   - API rate limits
   - Circuit breaker settings
   - Symbol restrictions

4. **Approval Workflow**
   - Required approval steps
   - Approval methods
   - Timeout settings

5. **Emergency Stop**
   - Auto-activation triggers
   - Notification settings
   - Clear confirmation requirements

6. **Audit Trail**
   - Logging settings
   - Knowledge graph sync
   - Retention policies

**Example Configuration**:
```yaml
modes:
  paper:
    enabled: true
    slippage_model: "realistic"
    base_slippage: 0.0005

  live:
    enabled: false  # Must be explicitly enabled
    requires_approval: true
    max_drawdown_pct: 10.0

transition_requirements:
  min_paper_days: 3
  min_trades: 100
  max_drawdown_pct: 10.0
  min_win_rate: 55.0
  min_sharpe_ratio: 1.5

safety:
  daily_loss_limit_hard_pct: 5.0
  max_position_size_pct: 30.0
  api_rate_limit: 1200
```

## Integration Example

### Complete Workflow

```python
from graphwiz_trader.trading import (
    TradingModeManager,
    PaperTradingEngine,
    SafetyChecks,
    TransitionManager
)
from graphwiz_trader.graph import KnowledgeGraph

# Initialize knowledge graph
kg = KnowledgeGraph(neo4j_config)
kg.connect()

# Initialize trading mode manager
mode_manager = TradingModeManager(
    knowledge_graph=kg,
    config=trading_mode_config,
    approval_callback=live_trading_approval
)

# Start in paper trading mode (default)
print(f"Current mode: {mode_manager.get_current_mode()}")  # PAPER

# Initialize paper trading engine
paper_engine = PaperTradingEngine(
    initial_balance={"USDT": 10000.0},
    knowledge_graph=kg,
    config=paper_config
)

# Initialize safety checks
safety = SafetyChecks(
    knowledge_graph=kg,
    config=safety_config,
    violation_callback=handle_safety_violation
)

# Initialize transition manager
transition_mgr = TransitionManager(
    paper_engine=paper_engine,
    safety_checks=safety,
    knowledge_graph=kg,
    config=transition_config
)

# Execute paper trades
order = create_order(...)
result = await paper_engine.execute_order(order, market_price)

# After sufficient paper trading, validate readiness
validation = await transition_mgr.validate_paper_trading_readiness()

if validation["ready_for_transition"]:
    # Start gradual transition
    await transition_mgr.start_gradual_transition()

    # Switch to live trading mode
    await mode_manager.switch_mode(
        TradingMode.LIVE,
        reason="Paper trading requirements met"
    )

    # Monitor and increase allocation over time
    await transition_mgr.increase_capital_allocation()

# Handle emergencies
if critical_condition:
    await mode_manager.emergency_stop_mode(reason="Critical losses")
    await transition_mgr.execute_rollback(reason="Emergency stop activated")
```

## Safety Features

### Default Safe Behavior

1. **Paper Trading by Default**
   - System starts in paper trading mode
   - Live trading requires explicit activation
   - Virtual portfolio used for testing

2. **Explicit Approvals**
   - Live trading requires approval callback
   - Confirmation dialogs for critical operations
   - Audit trail of all mode changes

3. **Multiple Safety Layers**
   - Pre-trade validation
   - Real-time monitoring
   - Automatic circuit breakers
   - Emergency stop capability

### Emergency Stop

The emergency stop immediately halts all trading operations:

```python
# Activate emergency stop
await mode_manager.emergency_stop_mode(reason="Detected anomaly")

# All trading operations will check:
if mode_manager.emergency_stop:
    raise Exception("Trading halted - emergency stop active")

# Clear emergency stop (requires explicit confirmation)
await mode_manager.clear_emergency_stop(reason="Issue resolved")
```

### Circuit Breaker

Automatic trading halt when extreme conditions detected:

- Daily loss limit exceeded (hard limit: 5%)
- Maximum drawdown exceeded (10%)
- Abnormal market conditions
- API failures
- Safety violations cascade

## Knowledge Graph Integration

All trading mode events are logged to Neo4j knowledge graph:

**Node Types**:
- `TradingModeChange` - Mode transitions
- `PaperTrade` - Paper trading executions
- `SafetyViolation` - Safety check failures
- `EmergencyStop` - Emergency stop events
- `CircuitBreaker` - Circuit breaker activations
- `ValidationResult` - Transition validations
- `TransitionStart` - Transition beginnings
- `CapitalIncrease` - Capital allocation changes
- `Rollback` - Rollback to paper trading

**Query Examples**:
```cypher
# Get all mode changes
MATCH (mc:TradingModeChange)
RETURN mc
ORDER BY mc.timestamp DESC

# Find safety violations
MATCH (sv:SafetyViolation)
WHERE sv.severity = 'critical'
RETURN sv

# Track paper trading performance
MATCH (pt:PaperTrade)
RETURN pt.symbol, COUNT(pt) as trade_count, AVG(pt.slippage_pct)
```

## Best Practices

### 1. Always Start with Paper Trading
- Test strategies thoroughly in paper mode
- Validate all safety checks
- Ensure consistent profitability

### 2. Use Gradual Transition
- Start with 10% capital allocation
- Monitor performance carefully
- Increase allocation gradually

### 3. Monitor Safety Violations
- Review all safety violations
- Address root causes
- Adjust risk parameters as needed

### 4. Keep Emergency Stop Ready
- Always have emergency stop configured
- Test emergency procedures
- Know how to quickly halt trading

### 5. Maintain Audit Trail
- Review mode change history
- Analyze paper trading performance
- Learn from safety violations

### 6. Regular Validation
- Re-validate paper trading requirements
- Check live trading performance
- Adjust transition parameters

## Security Considerations

1. **API Key Protection**
   - Never commit API keys to version control
   - Use environment variables or secure vaults
   - Validate API keys before live trading

2. **Access Control**
   - Require explicit approval for live trading
   - Implement multi-signature approvals for production
   - Log all approval decisions

3. **Risk Limits**
   - Set conservative position size limits
   - Use daily loss limits
   - Implement maximum drawdown protection

4. **Emergency Procedures**
   - Test emergency stop functionality
   - Have rollback procedures documented
   - Configure alert notifications

## Troubleshooting

### Common Issues

**Issue**: Cannot switch to live trading
- Check if `enabled: true` in config
- Verify approval callback is configured
- Ensure paper trading requirements met

**Issue**: Safety violations blocking trades
- Review violation details in logs
- Adjust position sizes or risk parameters
- Check daily limits and reset times

**Issue**: Automatic rollback triggered
- Review rollback reason in logs
- Analyze live trading performance
- Return to paper trading if needed

**Issue**: Circuit breaker activated
- Identify trigger condition
- Resolve underlying issue
- Clear circuit breaker explicitly

## Files Summary

| File | Description |
|------|-------------|
| `/opt/git/graphwiz-trader/src/graphwiz_trader/trading/modes.py` | Trading mode management |
| `/opt/git/graphwiz-trader/src/graphwiz_trader/trading/paper_trading.py` | Paper trading engine |
| `/opt/git/graphwiz-trader/src/graphwiz_trader/trading/safety.py` | Safety checks module |
| `/opt/git/graphwiz-trader/src/graphwiz_trader/trading/transition.py` | Transition manager |
| `/opt/git/graphwiz-trader/config/trading_modes.yaml` | Configuration file |

## Conclusion

This trading mode system provides a comprehensive, safe framework for progressing from paper trading to live trading with multiple layers of protection. The system ensures that strategies are thoroughly tested before real money is risked, and provides robust monitoring and safety mechanisms throughout the trading lifecycle.
