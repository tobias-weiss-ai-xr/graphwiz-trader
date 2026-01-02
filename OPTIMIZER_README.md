# Agent Looper Integration for GraphWiz Trader

This document describes the integration between agent-looper and graphwiz-trader for continuous autonomous optimization.

## Overview

The integration provides:

1. **SAIA Agent Integration**: Uses the SAIA agent from agent-looper for intelligent optimization decisions
2. **Multiple Optimization Types**: Strategy parameters, risk limits, agent weights, trading pairs, and indicators
3. **Safety First**: Paper trading validation, approval workflows, circuit breakers, and rollback capability
4. **Knowledge Graph Tracking**: All optimizations tracked in Neo4j knowledge graph
5. **Continuous Optimization**: Automated optimization loops with configurable frequencies

## Architecture

```
graphwiz-trader/
├── src/graphwiz_trader/optimizer/
│   ├── __init__.py
│   ├── looper_integration.py      # Main TradingOptimizer class
│   └── orchestrator.py             # OptimizationOrchestrator for coordination
├── config/
│   └── optimization_goals.yaml     # Optimization goals and constraints
└── examples/
    └── optimizer_demo.py           # Example usage

agent-looper/
└── src/projects/graphwiz-trader/
    ├── __init__.py
    ├── config.yaml                 # Project configuration
    └── goals.yaml                  # Goals in agent-looper format
```

## Components

### 1. TradingOptimizer (`looper_integration.py`)

The main optimizer class that integrates SAIA agent and provides optimization methods:

#### Key Features:
- **SAIA Agent Integration**: Uses `qwen3-coder-14b` model for intelligent optimization
- **Rollback Capability**: Automatic state snapshots for safe rollbacks
- **Constraint Validation**: Validates all optimizations against safety constraints
- **Multiple Optimization Types**:
  - `optimize_strategy_parameters()`: Tune strategy parameters
  - `optimize_risk_limits()`: Adjust risk management limits
  - `optimize_agent_weights()`: Optimize agent decision weights
  - `optimize_trading_pairs()`: Select optimal trading pairs
  - `optimize_indicators()`: Tune technical indicator parameters

#### Example Usage:

```python
from graphwiz_trader.optimizer import TradingOptimizer

# Initialize optimizer
optimizer = TradingOptimizer(
    project_path="/opt/git/graphwiz-trader",
    knowledge_graph=neo4j_graph,
    enable_auto_approve=False,  # Require manual approval
)

# Optimize strategy parameters
result = await optimizer.optimize_strategy_parameters(
    current_performance={
        "sharpe_ratio": 1.8,
        "max_drawdown": 0.08,
        "win_rate": 0.58,
    }
)

# Approve and apply
optimizer.approve_optimization(result.optimization_id)
optimizer.apply_optimization(result.optimization_id)

# Rollback if needed
optimizer.rollback_optimization(result.optimization_id)
```

### 2. OptimizationOrchestrator (`orchestrator.py`)

Coordinates multiple optimization loops with comprehensive safety:

#### Key Features:
- **Multiple Optimization Loops**: Runs different optimizations on schedules
- **Safety Checks**: Pre-optimization validation of system state
- **Paper Trading Validation**: All optimizations tested in paper trading first
- **Circuit Breaker**: Automatic pause on failures or safety violations
- **Approval Workflow**: Configurable approval for different optimization types
- **Performance Tracking**: Comprehensive tracking of all optimization results

#### Example Usage:

```python
from graphwiz_trader.optimizer import OptimizationOrchestrator

# Initialize orchestrator
orchestrator = OptimizationOrchestrator(
    project_path="/opt/git/graphwiz-trader",
    knowledge_graph=neo4j_graph,
    enable_auto_approve=False,
)

# Start continuous optimization
await orchestrator.start()

# Let it run...
await asyncio.sleep(3600)  # Run for 1 hour

# Check status
status = orchestrator.get_status()
print(f"Active: {status['active_optimizations']}")
print(f"Circuit Breaker: {status['circuit_breaker_state']}")

# Stop gracefully
await orchestrator.stop()

# Emergency stop if needed
await orchestrator.stop(emergency=True)
```

## Configuration

### Optimization Goals (`config/optimization_goals.yaml`)

Defines optimization targets:

```yaml
goals:
  - name: sharpe_ratio
    target_value: 2.5
    current_value: 1.5
    priority: high

  - name: max_drawdown
    target_value: 0.08  # 8%
    current_value: 0.15  # 15%
    priority: critical

constraints:
  risk_limits:
    max_position_size: 0.20  # 20%
    max_daily_loss: 0.05  # 5%
    min_liquidity_usd: 1000000  # $1M

  optimization:
    require_paper_trading: true
    paper_trading_duration_hours: 24
    require_approval: true
```

### Agent Looper Project Config (`agent-looper/src/projects/graphwiz-trader/config.yaml`)

Configures the agent-looper integration:

```yaml
project:
  name: "graphwiz-trader"
  path: "/opt/git/graphwiz-trader"

saia:
  model: "qwen3-coder-14b"
  temperature: 0.3

optimizations:
  strategy_parameters:
    enabled: true
    frequency: "daily"
    requires_approval: true

  risk_limits:
    enabled: true
    frequency: "weekly"
    requires_approval: true

paper_trading:
  enabled: true
  initial_capital: 10000
  validation_criteria:
    min_trades: 50
    min_duration_hours: 24
```

## Optimization Types

### 1. Strategy Parameter Optimization

Optimizes trading strategy parameters:
- Entry/exit thresholds
- Stop loss and take profit levels
- Position sizing multipliers
- Lookback periods
- Trend confirmation parameters

**Frequency**: Daily
**Approval Required**: Yes
**Paper Trading**: 24 hours, 50 trades minimum

### 2. Risk Limit Optimization

Adjusts risk management parameters:
- Maximum drawdown limits
- Daily loss limits
- Position size limits
- Correlation exposure limits
- Portfolio concentration limits

**Frequency**: Weekly
**Approval Required**: Yes
**Paper Trading**: 48 hours

### 3. Agent Weight Optimization

Optimizes agent decision weights:
- Technical analysis agent weight
- Sentiment analysis agent weight
- Risk management agent weight
- Portfolio management agent weight

**Frequency**: Daily
**Approval Required**: No (auto-approve small adjustments)
**Paper Trading**: 12 hours

### 4. Trading Pair Optimization

Selects optimal trading pairs:
- Evaluates liquidity and volatility
- Assesses correlation with existing pairs
- Checks spread and trading costs
- Ensures diversification

**Frequency**: Weekly
**Approval Required**: Yes
**Paper Trading**: 48 hours

### 5. Indicator Optimization

Tunes technical indicator parameters:
- RSI periods and thresholds
- MACD parameters
- Bollinger Band settings
- EMA periods
- VWAP parameters

**Frequency**: Monthly
**Approval Required**: Yes
**Paper Trading**: 72 hours

## Safety Features

### 1. Constraint Validation

All optimizations validated against:
- Maximum drawdown threshold (10%)
- Minimum Sharpe ratio (2.0)
- Minimum win rate (60%)
- Maximum position size (20%)
- Minimum liquidity ($1M daily volume)

### 2. Paper Trading Validation

Required before live deployment:
- Minimum 50 trades
- Minimum 24 hours duration
- Sharpe ratio > 1.5
- Win rate > 55%
- Profit factor > 1.5

### 3. Circuit Breaker

Automatic halt on:
- 5 consecutive optimization failures
- Drawdown exceeds 8%
- Daily loss exceeds 3%
- System failure detected

Cooldown period: 60 minutes

### 4. Rollback Capability

Automatic rollback on:
- Optimization application failure
- Performance degradation in paper trading
- Safety constraint violation

### 5. Approval Workflow

Multi-stage approval for critical changes:
1. Paper trading validation
2. Review and human approval
3. Staged rollout (10% -> 100%)
4. Full deployment with monitoring

## Running the Optimizer

### Basic Usage

```python
import asyncio
from graphwiz_trader.optimizer import OptimizationOrchestrator

async def main():
    # Initialize
    orchestrator = OptimizationOrchestrator(
        project_path="/opt/git/graphwiz-trader",
        enable_auto_approve=False,  # Require manual approval
    )

    # Start
    await orchestrator.start()

    # Run continuously
    while True:
        await asyncio.sleep(60)

        # Check status
        status = orchestrator.get_status()
        print(f"State: {status['state']}")
        print(f"Active: {status['active_optimizations']}")

asyncio.run(main())
```

### With Manual Approval

```python
# Get pending approvals
pending = orchestrator.optimizer.get_pending_optimizations()

for opt in pending:
    print(f"ID: {opt.optimization_id}")
    print(f"Type: {opt.optimization_type.value}")
    print(f"Expected improvement: {opt.expected_improvement:.2%}")
    print(f"Confidence: {opt.confidence_score:.2%}")
    print(f"Reasoning: {opt.reasoning[:200]}...")

    # Approve if satisfied
    if user_approves():
        orchestrator.approve_optimization(opt.optimization_id)
```

### Emergency Stop

```python
# Immediate halt of all operations
await orchestrator.stop(emergency=True)

# Later, reset circuit breaker and resume
orchestrator.reset_circuit_breaker()
await orchestrator.start()
```

## Knowledge Graph Integration

All optimizations tracked in Neo4j:

```cypher
// Find recent optimizations
MATCH (opt:Optimization)
WHERE opt.timestamp > datetime() - duration('P7D')
RETURN opt ORDER BY opt.timestamp DESC

// Find optimizations by type
MATCH (opt:Optimization {type: 'strategy_parameters'})
RETURN opt

// Find failed optimizations
MATCH (opt:Optimization {status: 'failed'})
RETURN opt

// Track optimization performance
MATCH (opt:Optimization)
WHERE opt.expected_improvement > 0.1
RETURN opt.type, avg(opt.expected_improvement)
```

## Monitoring

### Key Metrics

Track these metrics to ensure optimizer health:

1. **Optimization Success Rate**: Percentage of successful optimizations
2. **Paper Trading Pass Rate**: Percentage passing paper trading validation
3. **Average Improvement**: Mean expected improvement across all optimizations
4. **Circuit Breaker Trips**: Number of times circuit breaker activated
5. **Approval Rate**: Percentage of optimizations approved after review

### Logging

Optimizer logs to:
- File: `/opt/git/graphwiz-trader/logs/optimizer.log`
- Rotation: 100 MB per file
- Retention: 30 days

### Alerts

Configure alerts for:
- Optimization failures
- Circuit breaker trips
- Approval requests
- Performance degradation

## Best Practices

1. **Start with Paper Trading**: Always test optimizations in paper trading first
2. **Require Approval for Critical Changes**: Keep auto-approve disabled for production
3. **Monitor Circuit Breaker**: Investigate immediately when circuit breaker trips
4. **Review Optimization History**: Regularly review past optimizations for patterns
5. **Adjust Constraints**: Update constraints as market conditions change
6. **Keep Knowledge Graph Updated**: Ensure all optimization data is tracked
7. **Test Rollback Procedures**: Regularly test rollback functionality

## Troubleshooting

### Circuit Breaker Keeps Tripping

- Check safety constraints in config
- Review recent optimization failures
- Verify market data quality
- Check system logs for errors

### Optimizations Not Running

- Verify orchestrator is in RUNNING state
- Check if circuit breaker is OPEN
- Review optimization loop schedules
- Ensure paper trading is enabled

### Poor Optimization Results

- Review SAIA agent responses
- Check if constraints are too restrictive
- Verify input data quality
- Consider adjusting SAIA model or temperature

### Approval Timeout

- Check notification system configuration
- Verify approval workflow settings
- Review approval timeout duration
- Consider enable_auto_approve for non-critical optimizations

## Future Enhancements

- Multi-objective optimization (Pareto fronts)
- Reinforcement learning integration
- Market regime detection
- A/B testing framework
- Real-time optimization dashboards
- Automated parameter sweep optimization
- Ensemble optimization strategies
