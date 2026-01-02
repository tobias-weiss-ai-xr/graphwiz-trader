# Knowledge Graph Integration for graphwiz-trader

## Overview

This comprehensive knowledge graph integration provides advanced market data storage, analytics, and real-time streaming capabilities for graphwiz-trader using Neo4j 5.x.

## Architecture

### Components

1. **KnowledgeGraph** (`neo4j_graph.py`): Core graph database connector with market data methods
2. **GraphAnalytics** (`analytics.py`): Advanced analytics engine for market insights
3. **GraphDataManager** (`data_manager.py`): Batch ingestion and real-time streaming manager
4. **Data Models** (`models.py`): Comprehensive data models for all market entities

### Data Models

#### Asset Nodes
- AssetNode: Tradeable assets (crypto, stocks, forex, commodities)
- Properties: symbol, name, asset_type, base_currency, quote_currency, decimals, metadata

#### Exchange Nodes
- ExchangeNode: Exchange information with capabilities and fees
- Properties: name, display_name, maker_fee, taker_fee, supports_margin/futures/spot

#### Market Data Nodes
- OHLCVNode: Candlestick data with temporal indexing
- TradeNode: Individual trade executions
- OrderBookNode: Order book snapshots
- IndicatorNode: Technical indicator values (RSI, MACD, etc.)
- SignalNode: Trading signals from agents
- SentimentNode: Market sentiment data

#### Relationships
- CORRELATED_WITH: Asset correlations with coefficients
- ARBITRAGE_WITH: Cross-exchange arbitrage opportunities
- TRADED_ON: Asset availability on exchanges
- IMPACTS: Market impact relationships

## Features

### 1. Market Data Storage

#### OHLCV Data
```python
from graphwiz_trader.graph import KnowledgeGraph, OHLCVNode
from datetime import datetime

graph = KnowledgeGraph(config)
graph.connect()

# Single record
ohlcv = OHLCVNode(
    symbol="BTC/USD",
    exchange="binance",
    timestamp=datetime.utcnow(),
    timeframe="1h",
    open=45000.0,
    high=45500.0,
    low=44800.0,
    close=45200.0,
    volume=1234.56
)
graph.store_ohlcv(ohlcv)

# Batch storage (optimized with UNWIND)
graph.store_ohlcv_batch(ohlcv_list)
```

#### Trade Data
```python
from graphwiz_trader.graph import TradeNode

trade = TradeNode(
    trade_id="trade_123",
    symbol="BTC/USD",
    exchange="binance",
    timestamp=datetime.utcnow(),
    side="BUY",
    price=45000.0,
    quantity=0.5,
    amount=22500.0,
    fee=22.5
)
graph.store_trade(trade)

# Batch ingestion
from graphwiz_trader.graph import GraphDataManager

data_manager = GraphDataManager(graph)
data_manager.ingest_trade_data(trade_list, batch_size=1000)
```

#### Order Book Snapshots
```python
from graphwiz_trader.graph import OrderBookNode

orderbook = OrderBookNode(
    symbol="BTC/USD",
    exchange="binance",
    timestamp=datetime.utcnow(),
    bids=[[45000.0, 1.5], [44990.0, 2.0]],
    asks=[[45010.0, 1.2], [45020.0, 1.8]],
    bid_depth=3.5,
    ask_depth=3.0,
    spread=10.0,
    spread_percentage=0.022
)
graph.store_orderbook(orderbook)
```

### 2. Market Relationships

#### Correlation Analysis
```python
from graphwiz_trader.graph import GraphAnalytics, CorrelationRelationship

analytics = GraphAnalytics(graph)

# Calculate correlation matrix
result = analytics.calculate_correlation_matrix(
    symbols=["BTC/USD", "ETH/USD", "BNB/USD"],
    exchange="binance",
    window="24h",
    min_correlation=0.5
)

# Store correlation
correlation = CorrelationRelationship(
    symbol1="BTC/USD",
    symbol2="ETH/USD",
    correlation_coefficient=0.85,
    p_value=0.001,
    window="24h"
)
graph.create_correlation(correlation)
```

#### Arbitrage Detection
```python
# Detect cross-exchange arbitrage
opportunities = analytics.detect_arbitrage_opportunities(
    exchanges=["binance", "kraken", "coinbase"],
    symbols=["BTC/USD", "ETH/USD"],
    min_profit_percentage=0.5,
    include_fees=True
)

# Detect triangular arbitrage
tri_arb = analytics.detect_triangular_arbitrage(
    base_currency="USD",
    min_profit_percentage=0.1
)
```

### 3. Graph Analytics

#### Pathfinding
```python
# Find shortest path between assets
path = graph.find_shortest_path(
    from_symbol="BTC/USD",
    to_symbol="ETH/USD",
    max_depth=3
)
# Returns: [{"path_symbols": ["BTC/USD", "USD", "ETH/USD"], ...}]
```

#### Market Impact Analysis
```python
# Analyze impact of large trades
impact = graph.analyze_market_impact(
    symbol="BTC/USD",
    exchange="binance",
    volume_threshold=10000
)
```

#### Pattern Detection
```python
# Detect pump and dump patterns
pump_dump = analytics.detect_pump_and_dump(
    symbol="BTC/USD",
    exchange="binance",
    lookback_hours=24,
    volume_spike_threshold=3.0,
    price_change_threshold=20.0
)

# Detect accumulation/distribution
acc_dist = analytics.detect_accumulation_distribution(
    symbol="BTC/USD",
    exchange="binance",
    lookback_hours=48
)
```

### 4. Real-Time Streaming

```python
# Start streaming
data_manager.start_streaming(
    queue_size=10000,
    worker_count=2,
    batch_size=100,
    batch_timeout_ms=1000
)

# Stream data (non-blocking)
data_manager.stream_ohlcv(ohlcv_dict)
data_manager.stream_trade(trade_dict)
data_manager.stream_signal(signal_dict)

# Stop streaming (processes remaining items)
data_manager.stop_streaming()
```

### 5. Data Retention

```python
# Define retention policies (days)
policies = {
    "OHLCV": 30,
    "Trade": 7,
    "OrderBook": 1,
    "Indicator": 30
}

# Dry run to preview deletions
results = data_manager.cleanup_old_data(policies, dry_run=True)

# Execute cleanup
results = data_manager.cleanup_old_data(policies, dry_run=False)
```

## Schema Optimization

### Constraints
- `Asset.symbol`: Unique constraint
- `Exchange.name`: Unique constraint
- `Trade.trade_id`: Unique constraint
- `Signal.signal_id`: Unique constraint

### Indexes
- `OHLCV.timestamp`: Time-series query optimization
- `Trade.timestamp`: Trade history queries
- `Signal.timestamp`: Signal history queries
- `Indicator.timestamp`: Indicator queries
- `Sentiment.timestamp`: Sentiment analysis
- `Asset.asset_type`: Asset type filtering

## Performance Optimizations

### Batch Operations
All batch operations use Neo4j's UNWIND clause for efficient bulk inserts:

```cypher
UNWIND $batch AS row
MERGE (o:OHLCV {symbol: row.symbol, ...})
SET o.open = row.open, ...
```

### Temporal Patterns
Time-series data uses temporal indexing for fast range queries:

```cypher
MATCH (o:OHLCV)
WHERE o.timestamp >= datetime($start_time)
  AND o.timestamp <= datetime($end_time)
```

### Query Optimization
- Parameterized queries prevent query plan caching issues
- LIMIT clauses prevent large result sets
- Index hints for complex queries

## Integration with Trading Engine

### Storing Signals
```python
from graphwiz_trader.graph import SignalNode, SignalType

# Agent generates signal
signal = SignalNode(
    signal_id=f"signal_{agent_name}_{timestamp}",
    symbol="BTC/USD",
    exchange="binance",
    timestamp=datetime.utcnow(),
    signal_type=SignalType.BUY,
    agent_name="TechnicalAnalysisAgent",
    confidence=0.85,
    reason="RSI oversold + MACD crossover",
    target_price=47000.0,
    stop_loss=44500.0,
    take_profit=48500.0,
    indicators=["RSI", "MACD", "BB"]
)
graph.store_signal(signal)
```

### Querying Historical Performance
```python
# Get agent's historical signals
signals = graph.get_signals(
    symbol="BTC/USD",
    signal_type=SignalType.BUY,
    min_confidence=0.7,
    start_time=datetime.utcnow() - timedelta(days=30),
    limit=100
)

# Analyze signal performance
for signal in signals:
    # Compare signal price with subsequent price action
    pass
```

## Risk Manager Integration

### Market Impact Analysis
```python
# Before placing large order, analyze market impact
impact = graph.analyze_market_impact(
    symbol="BTC/USD",
    exchange="binance",
    volume_threshold=order_amount
)

# Adjust order size based on impact
if impact["avg_impact"] > threshold:
    # Reduce order size or split into multiple orders
    pass
```

### Sentiment Analysis
```python
# Get recent sentiment for asset
sentiment = graph.get_average_sentiment(
    symbol="BTC/USD",
    start_time=datetime.utcnow() - timedelta(hours=24),
    end_time=datetime.utcnow()
)

# Adjust position sizing based on sentiment
if sentiment["avg_score"] < -0.5:  # Strongly negative
    # Reduce position or implement tighter stops
    pass
```

## Usage Example

See `/opt/git/graphwiz-trader/examples/knowledge_graph_usage.py` for comprehensive examples.

## Requirements

- Neo4j 5.x
- Python 3.9+
- neo4j-python-driver
- numpy
- scipy
- scikit-learn

## Configuration

```python
config = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "your_password",
    "database": "neo4j"  # Optional, defaults to "neo4j"
}
```

## Best Practices

1. **Batch Operations**: Always use batch methods for bulk data ingestion
2. **Indexing**: Let the schema initialization create necessary indexes
3. **Time-Series Data**: Use appropriate timeframes for your use case
4. **Retention Policies**: Implement cleanup to prevent database bloat
5. **Streaming**: Use streaming for real-time data, batch for historical
6. **Correlation Updates**: Update correlations periodically (e.g., daily)
7. **Arbitrage Detection**: Run frequently but cache results

## Monitoring

### Storage Statistics
```python
stats = data_manager.get_storage_stats()
# Returns counts for all node types and relationships
```

### Query Performance
Enable Neo4j query logging to monitor slow queries:
```conf
db.logs.query.enabled=true
db.logs.query.threshold=1000ms
```

## Troubleshooting

### Connection Issues
- Verify Neo4j is running: `systemctl status neo4j`
- Check firewall rules for port 7687
- Verify credentials in config

### Performance Issues
- Check index usage with `EXPLAIN` and `PROFILE` in Neo4j browser
- Reduce batch sizes if memory constrained
- Implement retention policies for old data

### Memory Management
- Neo4j heap size: Set in `neo4j.conf`
- Consider partitioning old data to separate database

## Future Enhancements

1. **Machine Learning Integration**: Store model predictions and backtest results
2. **Social Network Analysis**: Track sentiment propagation across assets
3. **Time-Series Forecasting**: Integrate prediction models
4. **Advanced Pattern Recognition**: More sophisticated market patterns
5. **Real-Time Alerts**: Trigger actions based on graph patterns
