# Sentiment-Based Trading Strategy

## Overview

The sentiment-based trading strategy analyzes market sentiment from multiple sources to generate trading signals. It combines:

- **News Analysis**: Articles from crypto news outlets
- **Social Media**: Twitter, Reddit, Telegram, Discord
- **On-Chain Metrics**: Exchange flows, active addresses, HODLer behavior

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  News    │  │ Twitter  │  │ Reddit   │  │ On-chain │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼────────────┼────────────┼────────────┼────────────┘
        │            │            │            │
        └────────────┴────────────┴────────────┘
                             │
                    ┌────────▼────────┐
                    │ Knowledge       │
                    │ Extractor       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Sentiment       │
                    │ Analyzer        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Sentiment      │
                    │  Strategy       │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌────────▼────────┐
     │ Knowledge Graph │          │ Trading Signals │
     │ (Neo4j)         │          │                 │
     └─────────────────┘          └─────────────────┘
```

## Components

### 1. Knowledge Extractor (`knowledge_extractor.py`)

Fetches sentiment data from various sources and performs initial analysis.

**Key Classes:**
- `SentimentSource`: Enum of data sources (NEWS, TWITTER, REDDIT, etc.)
- `SentimentData`: Raw sentiment data from a source
- `AnalyzedSentiment`: Analyzed sentiment with score and confidence
- `SentimentAnalyzer`: Analyzes text sentiment using crypto-specific lexicon
- `KnowledgeExtractor`: Orchestrates data extraction from all sources

**Example Usage:**
```python
from graphwiz_trader.sentiment import KnowledgeExtractor

config = {
    'sources': {
        'news': True,
        'social': True,
        'onchain': True
    },
    'update_interval_seconds': 300
}

extractor = KnowledgeExtractor(config)

# Extract and analyze sentiment
sentiments = await extractor.extract_and_analyze(
    symbols=['BTC', 'ETH'],
    hours_back=24
)
```

### 2. Sentiment Strategy (`sentiment_strategy.py`)

Generates trading signals based on aggregated sentiment data.

**Key Classes:**
- `SentimentSignal`: Enum of signals (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- `SentimentSignalResult`: Complete signal with metadata
- `SentimentStrategyConfig`: Strategy configuration
- `SentimentTradingStrategy`: Main strategy implementation

**Example Usage:**
```python
from graphwiz_trader.strategies.sentiment_strategy import (
    SentimentTradingStrategy,
    SentimentStrategyConfig
)

# Create configuration
config = SentimentStrategyConfig(
    strong_buy_threshold=0.6,
    buy_threshold=0.3,
    min_confidence=0.5,
    min_data_points=5
)

# Initialize strategy
strategy = SentimentTradingStrategy(
    config=config,
    knowledge_extractor=extractor,
    knowledge_graph=kg
)

# Generate signal
signal = strategy.generate_signal(
    symbol='BTC',
    current_price=45000.0,
    balance=1000.0
)

print(f"Signal: {signal.signal.value}")
print(f"Confidence: {signal.confidence:.2%}")
print(f"Sentiment Score: {signal.sentiment_score:.3f}")
```

### 3. Knowledge Graph Integration (`neo4j_graph.py`)

Stores and retrieves sentiment data in Neo4j for historical analysis.

**Key Methods:**
- `create_sentiment_node()`: Store sentiment data point
- `get_recent_sentiment()`: Retrieve recent sentiment
- `get_aggregate_sentiment()`: Get aggregate metrics
- `get_sentiment_trend()`: Get sentiment over time

**Example Usage:**
```python
from graphwiz_trader.graph.neo4j_graph import KnowledgeGraph

# Initialize
kg_config = {
    'uri': 'bolt://localhost:7687',
    'username': 'neo4j',
    'password': 'your_password'
}

kg = KnowledgeGraph(kg_config)
kg.connect()

# Store sentiment
await kg.create_sentiment_node(
    symbol='BTC',
    timestamp=datetime.now(),
    source='news',
    sentiment_score=0.7,
    confidence=0.8,
    volume=500,
    keywords=['adoption', 'institutional'],
    metadata={}
)

# Get aggregate sentiment
aggregate = kg.get_aggregate_sentiment('BTC', hours_back=24)
print(f"Average sentiment: {aggregate['avg_sentiment']:.3f}")
print(f"Data points: {aggregate['data_points']}")
```

## Configuration

Edit `config/sentiment.yaml` to configure:

### Data Sources
```yaml
knowledge_extractor:
  sources:
    news: true
    social: true
    onchain: true
  update_interval_seconds: 300  # 5 minutes
```

### Signal Thresholds
```yaml
sentiment_strategy:
  strong_buy_threshold: 0.6
  buy_threshold: 0.3
  sell_threshold: -0.3
  strong_sell_threshold: -0.6
```

### Risk Management
```yaml
sentiment_strategy:
  min_confidence: 0.5
  min_data_points: 5
  min_volume: 100
  max_sentiment_position_pct: 0.35  # Max 35% of balance
```

## Signal Generation Process

1. **Data Collection**
   - Fetch news from configured sources
   - Scrape social media posts
   - Query on-chain metrics

2. **Analysis**
   - Analyze text sentiment using crypto lexicon
   - Calculate sentiment scores (-1 to 1)
   - Determine confidence levels (0 to 1)

3. **Aggregation**
   - Apply time-based decay (older data = less weight)
   - Calculate weighted averages by source reliability
   - Compute momentum (rate of change)

4. **Signal Generation**
   ```
   if adjusted_score >= 0.6 → STRONG_BUY
   elif adjusted_score >= 0.3 → BUY
   elif adjusted_score <= -0.6 → STRONG_SELL
   elif adjusted_score <= -0.3 → SELL
   else → HOLD
   ```

5. **Position Sizing**
   - Base position adjusted by signal strength
   - Confidence multiplier (0.5 to 1.0)
   - Momentum adjustment (±20%)
   - Maximum of 35% of balance

## Integration with Trading Engine

### Basic Integration

```python
from graphwiz_trader.trading.engine import TradingEngine
from graphwiz_trader.sentiment import (
    KnowledgeExtractor,
    SentimentTradingStrategy
)

# Initialize components
extractor = KnowledgeExtractor(config['knowledge_extractor'])
strategy = SentimentTradingStrategy(
    config=strategy_config,
    knowledge_extractor=extractor,
    knowledge_graph=kg
)

# Update sentiment data
await strategy.update_and_analyze(['BTC', 'ETH'])

# Generate signals and execute trades
for symbol in ['BTC', 'ETH']:
    # Get current price from exchange
    ticker = exchange.fetch_ticker(f'{symbol}/EUR')
    current_price = ticker['last']

    # Generate signal
    signal = strategy.generate_signal(
        symbol=symbol,
        current_price=current_price,
        balance=1000.0
    )

    if signal and signal.signal in [SentimentSignal.BUY, SentimentSignal.STRONG_BUY]:
        # Calculate position
        position_eur = strategy.calculate_position_size(
            signal,
            current_price,
            1000.0,
            250  # Base position in EUR
        )

        # Execute trade
        amount = position_eur / current_price
        order = trading_engine.execute_market_order(
            symbol=symbol,
            side='buy',
            amount=amount
        )

        logger.info(f"Executed {signal.signal.value} order: {amount:.6f} {symbol}")
```

### Combining with RSI Strategy

```python
# Get RSI signal
rsi_signal = get_rsi_signal(prices)

# Get sentiment signal
sentiment_signal = sentiment_strategy.generate_signal(...)

# Combine signals
if rsi_signal == 'BUY' and sentiment_signal.signal == SentimentSignal.BUY:
    final_action = 'STRONG_BUY'
    confidence = (rsi_confidence + sentiment_signal.confidence) / 2
elif rsi_signal == 'BUY' and sentiment_signal.signal == SentimentSignal.HOLD:
    final_action = 'BUY'
    confidence = rsi_confidence * 0.7
elif rsi_signal == 'BUY' and sentiment_signal.signal == SentimentSignal.SELL:
    final_action = 'HOLD'  # Conflicting signals
    confidence = 0.3
else:
    final_action = 'HOLD'
    confidence = 0.5
```

## API Integrations (Production)

To enable real data sources, add API credentials to `.env`:

```bash
# News APIs
NEWS_API_KEY=your_newsapi_key
CRYPTOPANIC_API_KEY=your_cryptopanic_key
MESSARI_API_KEY=your_messari_key

# Social Media
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# On-Chain Data
GLASSNODE_API_KEY=your_glassnode_key
COINMETRICS_API_KEY=your_coinmetrics_key
CRYPTOQUANT_API_KEY=your_cryptoquant_key
```

## Running the Demo

### Full Strategy Demo
```bash
python test_sentiment_strategy.py --full
```

This will:
1. Load configuration
2. Initialize knowledge extractor and strategy
3. Fetch and analyze sentiment data
4. Generate trading signals
5. Calculate position sizes
6. Store data in Neo4j (if configured)

### Component Demos
```bash
python test_sentiment_strategy.py --components
```

This will test individual components separately.

## Monitoring and Debugging

### Enable Logging
```python
from loguru import logger

logger.add("logs/sentiment_analysis.log", rotation="1 day")
logger.info("Starting sentiment analysis")
```

### View Sentiment History
```python
# Get sentiment from graph
recent = kg.get_recent_sentiment('BTC', hours_back=24)

# Get trends
trends = kg.get_sentiment_trend('BTC', hours_back=24, interval_hours=1)

for trend in trends:
    print(f"{trend['time_bucket']}: {trend['avg_sentiment']:.3f}")
```

### Check Signal Confidence
```python
# Low confidence - don't trade
if signal.confidence < 0.5:
    logger.warning(f"Low confidence for {symbol}: {signal.confidence}")

# Check data quality
if signal.metadata['data_points'] < 10:
    logger.warning(f"Insufficient data for {symbol}")
```

## Performance Considerations

1. **Update Frequency**: Default is 5 minutes, adjust based on needs
2. **Data Retention**: Sentiment data kept for 30 days by default
3. **Rate Limits**: Respect API rate limits (60 requests/minute default)
4. **Caching**: Use Neo4j query cache to reduce database load

## Security Best Practices

1. **API Keys**: Store in environment variables, never commit to git
2. **Rate Limiting**: Implement backoff strategies for API limits
3. **Data Validation**: Validate all external data before use
4. **Error Handling**: Handle network failures gracefully
5. **Logging**: Log all trading decisions for audit trail

## Troubleshooting

### No Signals Generated
- Check if minimum data points requirement is met
- Verify confidence threshold is not too high
- Ensure sentiment sources are enabled

### Low Confidence Scores
- Increase data collection frequency
- Add more sentiment sources
- Check sentiment analyzer lexicon

### Neo4j Connection Issues
- Verify Neo4j is running: `systemctl status neo4j`
- Check credentials in config
- Test connection: `bolt://localhost:7687`

### API Rate Limits
- Increase `update_interval_seconds`
- Reduce number of symbols tracked
- Implement request queuing

## Future Enhancements

- [ ] Machine learning sentiment analysis (BERT, RoBERTa)
- [ ] Real-time WebSocket streams for social media
- [ ] Multi-language sentiment analysis
- [ ] Custom sentiment lexicon builder
- [ ] Sentiment heatmaps and visualizations
- [ ] Backtesting framework for sentiment strategies
- [ ] Alert system integration
