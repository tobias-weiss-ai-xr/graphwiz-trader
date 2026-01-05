# Sentiment Analysis - Quick Start Guide

## What's New

Implemented comprehensive sentiment-based trading strategy with:

✅ **Knowledge Extractor** - Fetches sentiment from news, social media, and on-chain data
✅ **Sentiment Analyzer** - Crypto-specific sentiment analysis with lexicon
✅ **Trading Strategy** - Generates BUY/SELL/HOLD signals based on sentiment
✅ **Neo4j Integration** - Stores and queries historical sentiment data
✅ **Configuration** - Fully configurable via `config/sentiment.yaml`
✅ **Demo Script** - Test and explore with `test_sentiment_strategy.py`

## File Structure

```
src/graphwiz_trader/
├── sentiment/
│   ├── __init__.py              # Module exports
│   └── knowledge_extractor.py   # Data extraction & analysis
├── strategies/
│   └── sentiment_strategy.py    # Trading strategy implementation
└── graph/
    └── neo4j_graph.py           # Added sentiment node methods

config/
└── sentiment.yaml               # Sentiment configuration

docs/
└── SENTIMENT_ANALYSIS.md        # Full documentation

test_sentiment_strategy.py       # Demo script
```

## Quick Start

### 1. Install Dependencies

```bash
# Already installed if you have the venv set up
# If not:
pip install aiohttp pyyaml loguru
```

### 2. Configure

Edit `config/sentiment.yaml`:

```yaml
knowledge_extractor:
  enabled: true
  sources:
    news: true
    social: true
    onchain: true

sentiment_strategy:
  strong_buy_threshold: 0.6
  buy_threshold: 0.3
  sell_threshold: -0.3
  strong_sell_threshold: -0.6
```

### 3. Run Demo

```bash
# Full strategy demo
python test_sentiment_strategy.py --full

# Component demos
python test_sentiment_strategy.py --components
```

### 4. Use in Your Code

```python
import asyncio
from graphwiz_trader.sentiment import (
    KnowledgeExtractor,
    SentimentTradingStrategy
)

async def trade_with_sentiment():
    # Initialize
    extractor = KnowledgeExtractor(config)
    strategy = SentimentTradingStrategy(
        config=strategy_config,
        knowledge_extractor=extractor,
        knowledge_graph=kg  # Optional
    )

    # Update sentiment data
    await strategy.update_and_analyze(['BTC', 'ETH'])

    # Generate signal
    signal = strategy.generate_signal('BTC', 45000.0, 1000.0)

    if signal:
        print(f"Signal: {signal.signal.value}")
        print(f"Confidence: {signal.confidence:.2%}")

        # Execute trade based on signal...
    else:
        print("No signal generated")

asyncio.run(trade_with_sentiment())
```

## Signal Types

| Signal | Condition | Action |
|--------|-----------|--------|
| STRONG_BUY | Score ≥ 0.6 | Buy 1.5x position |
| BUY | Score ≥ 0.3 | Buy base position |
| HOLD | -0.3 < Score < 0.3 | No action |
| SELL | Score ≤ -0.3 | Sell base position |
| STRONG_SELL | Score ≤ -0.6 | Sell 1.5x position |

## Sentiment Sources

| Source | Weight | Description |
|--------|--------|-------------|
| News | 1.0 | Crypto news articles |
| Twitter | 0.7 | Social media posts |
| Reddit | 0.7 | Forum discussions |
| On-chain | 0.9 | Exchange flows, metrics |

## Key Features

### 1. Crypto-Specific Sentiment Analysis

```python
from graphwiz_trader.sentiment.knowledge_extractor import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze text
result = analyzer.analyze_text(
    "Bitcoin going to moon! 🚀 Institutional adoption accelerating!"
)

print(f"Score: {result['sentiment_score']:.3f}")  # 0.75 (bullish)
print(f"Confidence: {result['confidence']:.2f}")  # 0.85
print(f"Keywords: {result['keywords']}")  # ['institutional', 'adoption']
```

### 2. Time-Based Decay

Older sentiment data has less influence:

```python
# 1 hour old → 100% weight
# 3 hours old → 50% weight
# 6 hours old → 0% weight (configurable)
```

### 3. Momentum Analysis

Tracks sentiment velocity:

```python
momentum = strategy.calculate_sentiment_momentum('BTC', periods=3)

# Positive momentum = sentiment improving
# Negative momentum = sentiment worsening
```

### 4. Aggregate Metrics

```python
aggregate = extractor.calculate_aggregate_sentiment(sentiments)

print(f"Bullish: {aggregate['bullish_count']}")
print(f"Bearish: {aggregate['bearish_count']}")
print(f"Average Score: {aggregate['average_score']:.3f}")
```

## Neo4j Integration (Optional)

Store sentiment data for historical analysis:

```python
from graphwiz_trader.graph.neo4j_graph import KnowledgeGraph

# Connect
kg = KnowledgeGraph({'uri': 'bolt://localhost:7687', ...})
kg.connect()

# Store sentiment
await kg.create_sentiment_node(
    symbol='BTC',
    timestamp=datetime.now(),
    source='news',
    sentiment_score=0.7,
    confidence=0.8,
    volume=500,
    keywords=['adoption'],
    metadata={}
)

# Query sentiment
recent = kg.get_recent_sentiment('BTC', hours_back=24)
aggregate = kg.get_aggregate_sentiment('BTC', hours_back=24)
trends = kg.get_sentiment_trend('BTC', hours_back=24)
```

## Configuration Examples

### Conservative Strategy

```yaml
sentiment_strategy:
  strong_buy_threshold: 0.7    # Higher threshold
  buy_threshold: 0.4
  min_confidence: 0.7           # Higher confidence
  min_data_points: 10           # More data required
  position_multiplier: 1.0      # No position increase
```

### Aggressive Strategy

```yaml
sentiment_strategy:
  strong_buy_threshold: 0.4     # Lower threshold
  buy_threshold: 0.2
  min_confidence: 0.4           # Lower confidence
  min_data_points: 3            # Less data required
  position_multiplier: 2.0      # Double position for strong signals
```

## Troubleshooting

**Problem**: No signals generated
```python
# Check:
1. Are minimum data points met? (min_data_points: 5)
2. Is confidence high enough? (min_confidence: 0.5)
3. Is volume sufficient? (min_volume: 100)
```

**Problem**: Low confidence scores
```python
# Solution:
1. Decrease min_confidence in config
2. Increase update frequency
3. Add more sentiment sources
```

**Problem**: Too many false signals
```python
# Solution:
1. Increase strong_buy_threshold
2. Increase min_confidence
3. Increase min_data_points
4. Add momentum filter
```

## Next Steps

1. **Configure API Keys**: Add real data source credentials to `.env`
2. **Test with Demo**: Run `test_sentiment_strategy.py`
3. **Paper Trade**: Use with paper trading mode first
4. **Monitor**: Check confidence scores and signal quality
5. **Optimize**: Adjust thresholds based on performance

## Production Checklist

- [ ] Configure real API keys (Twitter, Reddit, News, etc.)
- [ ] Enable Neo4j for historical data
- [ ] Set up alerts for extreme sentiment
- [ ] Configure data retention policies
- [ ] Test with paper trading first
- [ ] Monitor for rate limiting
- [ ] Set up logging and monitoring
- [ ] Document trading decisions

## Resources

- **Full Documentation**: `docs/SENTIMENT_ANALYSIS.md`
- **Demo Script**: `test_sentiment_strategy.py`
- **Configuration**: `config/sentiment.yaml`
- **Neo4j Queries**: See `neo4j_graph.py` sentiment methods

## Support

For issues or questions:
1. Check `docs/SENTIMENT_ANALYSIS.md` for detailed documentation
2. Run demo script to verify installation
3. Check logs for error messages
4. Verify configuration in `config/sentiment.yaml`
