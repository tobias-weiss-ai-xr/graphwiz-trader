# AI Trading Agents - Quick Reference

## Import Examples

```python
# Import the orchestrator
from graphwiz_trader.agents import AgentOrchestrator

# Import individual agents
from graphwiz_trader.agents import (
    TechnicalAnalysisAgent,
    SentimentAnalysisAgent,
    RiskManagementAgent,
    MomentumAgent,
    MeanReversionAgent
)

# Import decision engine components
from graphwiz_trader.agents import (
    DecisionEngine,
    ConsensusMethod,
    ConflictResolution,
    TradingSignal
)
```

## Basic Usage

### Initialize Orchestrator
```python
orchestrator = AgentOrchestrator(
    config=agent_config,
    knowledge_graph=kg
)
```

### Get Trading Decision
```python
decision = await orchestrator.get_decision(
    market_data={
        "symbol": "BTC/USDT",
        "close": 50000.0,
        "volume": 1000
    },
    indicators={
        "RSI": {"value": 65.0},
        "MACD": {"histogram": 5.0}
    },
    context={
        "portfolio": {"exposure": 0.3}
    }
)
```

### Access Decision Results
```python
signal = decision.signal              # BUY, SELL, or HOLD
confidence = decision.confidence      # 0.0 to 1.0
reasoning = decision.reasoning        # String explanation
agents = decision.participating_agents  # List of agent names
conflict = decision.conflict_score    # 0.0 (unanimous) to 1.0 (max conflict)
```

### Update Performance
```python
await orchestrator.update_performance(
    symbol="BTC/USDT",
    decision=decision,
    entry_price=50000.0,
    current_price=50500.0,
    position_size=0.1,
    action_taken="BUY"
)
```

## Agent Types

### TechnicalAnalysisAgent
**Analyzes**: RSI, MACD, Bollinger Bands, EMA
**Best for**: Technical analysis, price patterns

### SentimentAnalysisAgent
**Analyzes**: News, social media sentiment
**Best for**: Market sentiment, news-driven trading

### RiskManagementAgent
**Analyzes**: Volatility, exposure, drawdown
**Best for**: Risk control, position sizing
**Note**: Always keep enabled!

### MomentumAgent
**Analyzes**: ROC, ADX, volume momentum
**Best for**: Trend following, momentum strategies

### MeanReversionAgent
**Analyzes**: Z-score, BB position, Stochastic
**Best for**: Statistical arbitrage, mean reversion

## Consensus Methods

```python
ConsensusMethod.MAJORITY_VOTE        # Simple majority
ConsensusMethod.WEIGHTED_VOTE        # Weighted by performance (default)
ConsensusMethod.CONFIDENCE_WEIGHTED  # Weighted by confidence
ConsensusMethod.BEST_PERFORMER       # Follow best agent
ConsensusMethod.UNANIMOUS            # Only if all agree
```

## Conflict Resolution

```python
ConflictResolution.HIGH_CONFIDENCE_WINS  # Highest confidence (default)
ConflictResolution.BEST_PERFORMER_WINS   # Best performing agent
ConflictResolution.RISK_AVERSE           # Default to HOLD
ConflictResolution.MAJORITY_RULES        # Follow majority
ConflictResolution.MANUAL_REVIEW         # Flag for review
```

## Configuration Examples

### Minimal
```yaml
agents:
  technical:
    enabled: true
  risk:
    enabled: true
```

### Full
```yaml
agents:
  technical:
    enabled: true
    min_confidence: 0.6
    max_confidence: 0.95
    learning_rate: 0.1

  sentiment:
    enabled: true
    min_confidence: 0.6

  risk:
    enabled: true
    min_confidence: 0.7  # Higher threshold

  momentum:
    enabled: true

  mean_reversion:
    enabled: true

orchestrator:
  consensus_method: "weighted_vote"
  conflict_resolution: "high_confidence"
  min_confidence_threshold: 0.6
```

## Decision Result Structure

```python
{
    "signal": "BUY",
    "confidence": 0.75,
    "reasoning": "Technical Analysis: RSI neutral...",
    "participating_agents": ["technical", "momentum", "risk"],
    "agent_signals": {
        "technical": {...},
        "momentum": {...},
        "risk": {...}
    },
    "consensus_method": "weighted_vote",
    "conflict_score": 0.3,
    "timestamp": "2026-01-01T12:00:00"
}
```

## Common Tasks

### Get Current Agent Weights
```python
weights = orchestrator.get_agent_weights()
# {"technical": 1.05, "risk": 1.15, ...}
```

### Get Performance Report
```python
report = orchestrator.get_performance_report(days=7)
# Includes: total decisions, signal distribution, accuracy, etc.
```

### Get Decision Statistics
```python
stats = orchestrator.get_decision_statistics()
# Overall system statistics
```

### Reconfigure Agents
```python
await orchestrator.reconfigure_agents(new_config)
```

## Troubleshooting

### All agents return HOLD
→ Check indicator data quality
→ Lower `min_confidence` thresholds
→ Verify market data is current

### Low agent agreement
→ Review agent strategies
→ Adjust `consensus_method`
→ Check for conflicting configurations

### Poor performance
→ Review `get_agent_performance()`
→ Adjust `learning_rate`
→ Disable underperforming agents
→ Recalibrate confidence thresholds

## Best Practices

1. **Always enable Risk Management agent**
2. **Enable at least 3 agents total**
3. **Review performance weekly**
4. **Start with paper trading**
5. **Monitor conflict scores**
6. **Use WEIGHTED_VOTE for balanced decisions**
7. **Use RISK_AVERSE conflict resolution for new strategies**

## File Locations

- **Source**: `/opt/git/graphwiz-trader/src/graphwiz_trader/agents/`
- **Config Example**: `/opt/git/graphwiz-trader/config/agents.example.yaml`
- **Documentation**: `/opt/git/graphwiz-trader/docs/AGENTS.md`
- **Tests**: `/opt/git/graphwiz-trader/tests/test_agents_full.py`

## Key Classes

- **AgentOrchestrator**: Main coordinator
- **TradingAgent**: Base class for agents
- **DecisionEngine**: Aggregates signals
- **AgentDecision**: Decision data structure
- **AgentPerformance**: Performance metrics
- **TradingSignal**: BUY/SELL/HOLD enum

## Quick Decision Flow

```
Market Data → Agents (parallel) → Individual Decisions
                                        ↓
                              Decision Engine
                                        ↓
                    Consensus → Conflict Resolution
                                        ↓
                              Final Decision
```

## Performance Metrics Tracked

- **Total Decisions**: Number of decisions made
- **Accuracy**: Percentage of correct decisions
- **Profit Factor**: Total profit / total loss
- **Recent Accuracy**: Rolling accuracy (last 20)
- **Average Confidence**: Mean confidence score
