# AI Trading Agents System

## Overview

The GraphWiz Trader AI agents system provides a sophisticated multi-agent architecture for making trading decisions. Multiple specialized agents analyze market data from different perspectives and reach consensus through intelligent voting mechanisms.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                        │
│  - Coordinates all agents                                   │
│  - Manages agent weights based on performance              │
│  - Tracks agent performance over time                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├── Decision Engine
                              │   - Aggregates signals
                              │   - Implements consensus
                              │   - Resolves conflicts
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    ▼                         ▼                         ▼
┌─────────┐            ┌────────────┐           ┌──────────────┐
│Technical│            │ Sentiment  │           │    Risk      │
│ Analysis│            │  Analysis  │           │  Management  │
│         │            │            │           │              │
│ RSI     │            │ News       │           │ Volatility   │
│ MACD    │            │ Social     │           │ Exposure     │
│ BB      │            │ Overall    │           │ Drawdown     │
└─────────┘            └────────────┘           └──────────────┘

    ▼                         ▼                         ▼
┌─────────┐            ┌────────────┐           ┌──────────────┐
│Momentum │            │Mean        │           │    More...   │
│         │            │Reversion   │           │              │
│ ROC     │            │ Z-score    │           │ Custom      │
│ ADX     │            │ BB         │           │ Agents      │
│ Volume  │            │ Stochastic │           │              │
└─────────┘            └────────────┘           └──────────────┘
```

## Agent Types

### 1. TechnicalAnalysisAgent

Analyzes technical indicators to identify trading opportunities.

**Indicators Used:**
- RSI (Relative Strength Index) - Overbought/oversold conditions
- MACD (Moving Average Convergence Divergence) - Trend momentum
- Bollinger Bands - Volatility and price extremes
- EMA (Exponential Moving Average) - Trend direction

**Output:**
- Signal: BUY/SELL/HOLD
- Confidence: 0.0 - 1.0
- Reasoning: Detailed explanation

**Example:**
```python
decision = await technical_agent.analyze(
    market_data={"close": 50000.0, "volume": 1000},
    indicators={
        "RSI": {"value": 25.0},
        "MACD": {"histogram": 10.5},
        "BB": {"upper": 51000, "lower": 49000}
    }
)
# Output: BUY with high confidence (RSI oversold, MACD bullish)
```

### 2. SentimentAnalysisAgent

Analyzes market sentiment from news and social media.

**Data Sources:**
- News sentiment (financial news, press releases)
- Social media sentiment (Twitter, Reddit)
- Overall market sentiment score

**Indicators Used:**
- Sentiment score (-1.0 to 1.0)
- Sentiment volume
- Sentiment trend (rising/falling/stable)

**Example:**
```python
decision = await sentiment_agent.analyze(
    market_data={"close": 50000.0},
    indicators={},
    context={
        "news_sentiment": {"score": 0.6, "count": 10},
        "social_sentiment": {"score": 0.4, "volume": 500}
    }
)
# Output: BUY based on positive sentiment
```

### 3. RiskManagementAgent

Monitors risk factors and adjusts trading decisions accordingly.

**Risk Factors:**
- Volatility levels
- Portfolio exposure
- Drawdown levels
- Correlation between positions

**Output:**
- Signal: BUY/SELL/HOLD (prefers conservative)
- Position size recommendations
- Risk warnings

**Example:**
```python
decision = await risk_agent.analyze(
    market_data={"close": 50000.0},
    indicators={"volatility": {"value": 0.08}},
    context={
        "portfolio": {"exposure": 0.85, "drawdown": -0.05}
    }
)
# Output: HOLD due to high volatility and exposure
```

### 4. MomentumAgent

Follows trends using momentum indicators.

**Indicators Used:**
- ROC (Rate of Change) - Price momentum
- ADX (Average Directional Index) - Trend strength
- Volume momentum
- Price trend analysis

**Example:**
```python
decision = await momentum_agent.analyze(
    market_data={
        "close": 50000.0,
        "price_history": [49500, 49600, 49700, 49800, 50000]
    },
    indicators={
        "ROC": {"value": 3.5},
        "ADX": {"value": 28, "di_plus": 25, "di_minus": 20}
    }
)
# Output: BUY (strong positive momentum)
```

### 5. MeanReversionAgent

Identifies mean reversion opportunities for statistical arbitrage.

**Indicators Used:**
- Z-score - Statistical deviation from mean
- Bollinger Band position
- Stochastic oscillator
- Price vs SMA deviation

**Example:**
```python
decision = await mean_reversion_agent.analyze(
    market_data={"close": 52000.0},
    indicators={
        "zscore": {"value": 2.2},
        "BB": {"upper": 51500, "lower": 48500},
        "SMA": {"value": 50000}
    }
)
# Output: SELL (price significantly above mean)
```

## Decision Engine

The DecisionEngine aggregates signals from all agents using configurable consensus methods.

### Consensus Methods

1. **MAJORITY_VOTE** - Simple majority rule
2. **WEIGHTED_VOTE** - Weighted by agent performance
3. **CONFIDENCE_WEIGHTED** - Weighted by confidence levels
4. **BEST_PERFORMER** - Follow the best performing agent
5. **UNANIMOUS** - Only act if all agents agree

### Conflict Resolution

When agents disagree, the system uses:

1. **HIGH_CONFIDENCE_WINS** - Highest confidence decision wins
2. **BEST_PERFORMER_WINS** - Follow best performing agent
3. **RISK_AVERSE** - Default to HOLD in conflicts
4. **MAJORITY_RULES** - Follow majority opinion
5. **MANUAL_REVIEW** - Flag for manual review

### Conflict Score

The system calculates a conflict score (0.0 to 1.0):
- 0.0: All agents agree (unanimous)
- 0.5: Moderate disagreement
- 1.0: Maximum disagreement

## Agent Performance Tracking

Each agent tracks:

- **Total Decisions**: Number of decisions made
- **Accuracy**: Percentage of correct decisions
- **Profit Factor**: Total profit / total loss
- **Recent Accuracy**: Rolling accuracy over last N decisions
- **Average Confidence**: Average confidence score

### Weight Adjustment

Agent weights are automatically adjusted based on performance:

```python
# Weight is calculated from recent accuracy and profit factor
performance_score = recent_accuracy * min(profit_factor, 2.0) / 2.0
new_weight = old_weight * 0.9 + performance_score * 0.1

# Weights are clamped to reasonable range
final_weight = max(0.3, min(new_weight, 2.0))
```

## Configuration

### Basic Configuration

```yaml
agents:
  technical:
    enabled: true
    model: "gpt-4"
    min_confidence: 0.6
    max_confidence: 0.95

  sentiment:
    enabled: true
    model: "gpt-4"
    min_confidence: 0.6

  risk:
    enabled: true
    model: "gpt-4"
    min_confidence: 0.7  # Higher threshold for risk

  momentum:
    enabled: true

  mean_reversion:
    enabled: true
```

### Orchestrator Configuration

```yaml
orchestrator:
  consensus_method: "weighted_vote"
  conflict_resolution: "high_confidence"
  min_confidence_threshold: 0.6
```

## Usage Examples

### Basic Usage

```python
from graphwiz_trader.agents import AgentOrchestrator

# Initialize orchestrator
orchestrator = AgentOrchestrator(
    config=agent_config,
    knowledge_graph=kg
)

# Get trading decision
decision = await orchestrator.get_decision(
    market_data={
        "symbol": "BTC/USDT",
        "close": 50000.0,
        "volume": 1000
    },
    indicators={
        "RSI": {"value": 65.0},
        "MACD": {"histogram": 5.0},
        "BB": {"upper": 51000, "lower": 49000}
    },
    context={
        "portfolio": {"value": 10000, "exposure": 0.3}
    }
)

print(f"Signal: {decision.signal}")
print(f"Confidence: {decision.confidence}")
print(f"Reasoning: {decision.reasoning}")
```

### Performance Tracking

```python
# Update performance after trade
performance = await orchestrator.update_performance(
    symbol="BTC/USDT",
    decision=decision,
    entry_price=50000.0,
    current_price=50500.0,
    position_size=0.1,
    action_taken="BUY"
)

print(performance)
# {
#   "technical": {"accuracy": 0.65, "profit_factor": 1.2},
#   "sentiment": {"accuracy": 0.58, "profit_factor": 1.1},
#   ...
# }
```

### Get Agent Weights

```python
weights = orchestrator.get_agent_weights()
print(weights)
# {"technical": 1.05, "sentiment": 0.98, "risk": 1.15, ...}
```

### Generate Performance Report

```python
report = orchestrator.get_performance_report(days=7)
print(report)
# {
#   "period_days": 7,
#   "total_decisions": 150,
#   "signal_distribution": {"BUY": 60, "SELL": 40, "HOLD": 50},
#   "average_confidence": 0.72,
#   "agent_performance": {...}
# }
```

## Agent Decision Flow

1. **Market Data Input**
   - Price, volume, indicators
   - Portfolio state
   - Sentiment data

2. **Parallel Agent Analysis**
   - All agents run concurrently
   - Each generates independent decision
   - Includes confidence and reasoning

3. **Decision Engine Aggregation**
   - Collects all agent decisions
   - Applies consensus method
   - Resolves conflicts if needed

4. **Final Decision Output**
   - Signal (BUY/SELL/HOLD)
   - Overall confidence
   - Detailed reasoning
   - Conflict score

5. **Performance Update**
   - Track decision outcome
   - Update agent performance
   - Adjust agent weights

## Best Practices

1. **Agent Configuration**
   - Enable at least 3 agents for diversity
   - Always keep Risk Management agent enabled
   - Adjust confidence thresholds per agent

2. **Consensus Method**
   - Use WEIGHTED_VOTE for balanced decisions
   - Use UNANIMOUS for conservative trading
   - Use BEST_PERFORMER in trending markets

3. **Conflict Resolution**
   - Use RISK_AVERSE for new strategies
   - Use HIGH_CONFIDENCE_WINS for established systems
   - Use MANUAL_REVIEW for critical decisions

4. **Performance Monitoring**
   - Review agent performance weekly
   - Adjust weights manually if needed
   - Disable consistently poor-performing agents

5. **Testing**
   - Backtest with historical data
   - Paper trade before live trading
   - Monitor agent agreement levels

## Extending the System

### Custom Agent

```python
from graphwiz_trader.agents import TradingAgent, AgentDecision, TradingSignal

class MyCustomAgent(TradingAgent):
    async def analyze(self, market_data, indicators, context=None):
        # Your analysis logic here
        signal = TradingSignal.BUY
        confidence = 0.75
        reasoning = "Custom analysis logic"

        return AgentDecision(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            agent_name=self.name,
            metadata={"method": "custom"}
        )
```

### Custom Consensus Method

```python
from graphwiz_trader.agents.decision import DecisionEngine

class CustomDecisionEngine(DecisionEngine):
    async def make_decision(self, agent_decisions, agent_performances, context):
        # Your custom consensus logic here
        pass
```

## Troubleshooting

### Agents Always Return HOLD
- Check indicator data quality
- Adjust min_confidence thresholds
- Verify market data is current

### Low Agent Agreement
- Review agent configurations
- Check for conflicting strategies
- Adjust consensus method

### Poor Performance
- Review individual agent performance
- Adjust learning rates
- Consider disabling underperforming agents
- Recalibrate confidence thresholds

## API Reference

See inline documentation for detailed API reference:
- `AgentOrchestrator` - Main orchestrator class
- `TradingAgent` - Base agent class
- `DecisionEngine` - Decision aggregation engine
- `AgentDecision` - Decision data structure
- `AgentPerformance` - Performance metrics

## Contributing

To add new agents or improve existing ones:

1. Inherit from `TradingAgent`
2. Implement the `analyze()` method
3. Return `AgentDecision` with signal, confidence, and reasoning
4. Add configuration to config.yaml
5. Update this documentation

## License

Part of the GraphWiz Trader project. See main LICENSE file.
