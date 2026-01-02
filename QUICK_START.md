# Quick Start Guide: Agent Looper Integration

## Installation

The optimizer is already integrated into graphwiz-trader. No additional installation needed.

## Prerequisites

1. **SAIA API Keys**: Configure in `/opt/git/agent-looper/.saia-keys`
2. **Neo4j**: Optional, for tracking optimizations in knowledge graph
3. **Python 3.10+**: Required for async/await support

## Basic Usage

### 1. Run the Demo

```bash
cd /opt/git/graphwiz-trader
python3 examples/optimizer_demo.py
```

### 2. Use in Your Code

```python
import asyncio
from graphwiz_trader.optimizer import OptimizationOrchestrator

async def main():
    # Initialize orchestrator
    orchestrator = OptimizationOrchestrator(
        project_path="/opt/git/graphwiz-trader",
        enable_auto_approve=False,  # Safe: requires manual approval
    )

    # Start continuous optimization
    await orchestrator.start()

    # Let it run...
    await asyncio.sleep(3600)  # 1 hour

    # Check status
    status = orchestrator.get_status()
    print(f"Active: {status['active_optimizations']}")
    print(f"Circuit Breaker: {status['circuit_breaker_state']}")

    # Stop gracefully
    await orchestrator.stop()

asyncio.run(main())
```

### 3. Manual Optimization

```python
from graphwiz_trader.optimizer import TradingOptimizer

# Initialize
optimizer = TradingOptimizer(
    project_path="/opt/git/graphwiz-trader",
)

# Optimize strategy parameters
result = await optimizer.optimize_strategy_parameters(
    current_performance={
        "sharpe_ratio": 1.8,
        "max_drawdown": 0.08,
        "win_rate": 0.58,
    }
)

# Review results
print(f"Expected improvement: {result.expected_improvement:.2%}")
print(f"Confidence: {result.confidence_score:.2%}")
print(f"Reasoning: {result.reasoning}")

# Approve if satisfied
if input("Approve? (y/n): ") == "y":
    optimizer.approve_optimization(result.optimization_id)
    optimizer.apply_optimization(result.optimization_id)
    print("Applied successfully!")

    # Rollback if needed
    # optimizer.rollback_optimization(result.optimization_id)
```

## Configuration

### Edit Optimization Goals

```bash
nano /opt/git/graphwiz-trader/config/optimization_goals.yaml
```

Key settings:
- Target Sharpe ratio
- Maximum drawdown limit
- Win rate target
- Risk limits

### Edit Agent Looper Config

```bash
nano /opt/git/agent-looper/src/projects/graphwiz-trader/config.yaml
```

Key settings:
- SAIA model selection
- Paper trading duration
- Approval requirements
- Circuit breaker thresholds

## Key Concepts

### Optimization Types

1. **Strategy Parameters**: Tune entry/exit logic (daily)
2. **Risk Limits**: Adjust risk management (weekly)
3. **Agent Weights**: Balance agent decisions (daily)
4. **Trading Pairs**: Select optimal pairs (weekly)
5. **Indicators**: Tune technical indicators (monthly)

### Safety Features

- **Paper Trading**: All optimizations tested first
- **Approval Required**: Manual approval for critical changes
- **Circuit Breaker**: Auto-pause on failures
- **Rollback**: One-click revert if needed

### States

- **STOPPED**: Not running
- **RUNNING**: Actively optimizing
- **PAUSED**: Temporarily paused
- **EMERGENCY_STOP**: Forced halt

## Common Operations

### Check Status

```python
status = orchestrator.get_status()
print(status)
```

### View Optimization History

```python
history = orchestrator.get_optimization_history()
for opt in history:
    print(f"{opt['type']}: {opt['status']} ({opt['confidence']:.0%})")
```

### Approve Pending Optimizations

```python
pending = optimizer.get_pending_optimizations()
for opt in pending:
    print(f"{opt.optimization_id}: {opt.optimization_type.value}")
    # Approve if good
    optimizer.approve_optimization(opt.optimization_id)
```

### Emergency Stop

```python
await orchestrator.stop(emergency=True)
```

### Reset Circuit Breaker

```python
orchestrator.reset_circuit_breaker()
await orchestrator.resume()
```

## Troubleshooting

### Circuit Breaker Tripped

```python
# Check why
status = orchestrator.get_status()
print(f"Circuit breaker: {status['circuit_breaker_state']}")

# Reset after fixing issue
orchestrator.reset_circuit_breaker()
```

### Optimizations Not Running

```python
# Check state
status = orchestrator.get_status()
print(f"State: {status['state']}")

# Ensure running
if status['state'] != 'RUNNING':
    await orchestrator.start()
```

### Poor Results

- Check constraints in `config/optimization_goals.yaml`
- Verify input data quality
- Review SAIA agent responses
- Consider adjusting SAIA model or temperature

## File Locations

```
graphwiz-trader/
├── src/graphwiz_trader/optimizer/
│   ├── looper_integration.py    # TradingOptimizer class
│   └── orchestrator.py            # OptimizationOrchestrator class
├── config/
│   └── optimization_goals.yaml   # Goals and constraints
├── examples/
│   └── optimizer_demo.py         # Usage example
└── tests/
    └── test_optimizer.py         # Test suite

agent-looper/
└── src/projects/graphwiz-trader/
    ├── config.yaml               # Project config
    └── goals.yaml                # Agent-looper goals
```

## Next Steps

1. **Run the demo**: `python3 examples/optimizer_demo.py`
2. **Read the full docs**: `OPTIMIZER_README.md`
3. **Configure goals**: Edit `config/optimization_goals.yaml`
4. **Integrate with trading engine**: Connect to actual trading system
5. **Set up monitoring**: Configure alerts and notifications

## Safety Reminders

- Always start with paper trading enabled
- Keep auto-approve disabled for production
- Monitor circuit breaker events
- Test rollback procedures regularly
- Review optimizations before approving
- Keep knowledge graph enabled for audit trail

## Support

- Full documentation: `OPTIMIZER_README.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Examples: `examples/optimizer_demo.py`
- Tests: `tests/test_optimizer.py`
