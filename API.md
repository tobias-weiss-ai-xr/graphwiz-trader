# API Reference

Complete API documentation for GraphWiz Trader.

## Table of Contents

1. [Trading Engine API](#trading-engine-api)
2. [Agent API](#agent-api)
3. [Risk Management API](#risk-management-api)
4. [Knowledge Graph API](#knowledge-graph-api)
5. [Backtesting API](#backtesting-api)
6. [Monitoring API](#monitoring-api)

---

## Trading Engine API

### TradingEngine

Main trading engine class that orchestrates the trading system.

#### Initialization

```python
from graphwiz_trader.trading.engine import TradingEngine

engine = TradingEngine(
    exchanges: Dict[str, Any],
    knowledge_graph: Neo4jKnowledgeGraph,
    risk_params: Dict[str, Any],
    config_path: Optional[str] = None
)
```

**Parameters:**
- `exchanges`: Dictionary of exchange instances (CCXT)
- `knowledge_graph`: Neo4j knowledge graph instance
- `risk_params`: Risk management parameters
- `config_path`: Optional path to configuration file

#### Methods

##### `async execute_trade_signal(market_data: Dict) -> Dict`

Execute a trade based on market signal.

**Parameters:**
- `market_data`: Dictionary containing market data
  - `symbol` (str): Trading pair symbol
  - `price` (float): Current price
  - `volume` (float): Trading volume
  - `timestamp` (str): ISO format timestamp

**Returns:**
```python
{
    "status": "success",  # or "rejected", "error"
    "action": "buy",      # "buy", "sell", "hold"
    "symbol": "BTC/USDT",
    "order_id": "12345",
    "price": 50000.0,
    "quantity": 0.1,
    "confidence": 0.85,
    "reason": "Strong bullish signal"
}
```

##### `async get_agent_decision(market_data: Dict) -> Dict`

Get trading decision from AI agents.

**Returns:**
```python
{
    "action": "buy",
    "confidence": 0.85,
    "reasoning": "Strong momentum with high volume",
    "position_size": 0.5,
    "stop_loss": 49000,
    "take_profit": 53000
}
```

##### `get_positions(symbol: Optional[str] = None) -> List[Dict]`

Get current open positions.

**Parameters:**
- `symbol`: Optional symbol filter

**Returns:**
```python
[{
    "symbol": "BTC/USDT",
    "side": "long",
    "entry_price": 50000,
    "current_price": 51000,
    "quantity": 0.5,
    "pnl": 500.0,
    "pnl_pct": 0.02
}]
```

##### `async close_position(symbol: str, reason: str) -> Dict`

Close a position.

**Parameters:**
- `symbol`: Symbol to close
- `reason`: Reason for closing position

**Returns:**
```python
{
    "status": "success",
    "symbol": "BTC/USDT",
    "exit_price": 51000,
    "pnl": 500.0,
    "pnl_pct": 0.02
}
```

---

## Agent API

### BaseAgent

Base class for all trading agents.

#### Creating Custom Agent

```python
from graphwiz_trader.agents import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)

    async def analyze(self, market_data: Dict) -> Dict:
        # Your analysis logic
        return {
            "action": "buy",
            "confidence": 0.8,
            "reasoning": "Custom logic"
        }
```

### TechnicalAgent

Technical analysis agent using indicators.

```python
from graphwiz_trader.agents.technical_agent import TechnicalAgent

agent = TechnicalAgent(
    model="gpt-4",
    temperature=0.3,
    indicators=["rsi", "macd", "sma_cross"]
)

decision = await agent.analyze({
    "symbol": "BTC/USDT",
    "price": 50000,
    "volume": 1000,
    "indicators": {
        "rsi": 65,
        "macd": 0.5,
        "sma_20": 49000,
        "sma_50": 48000
    }
})
```

### AgentOrchestrator

Orchestrates multiple agents for consensus decision.

```python
from graphwiz_trader.agents.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Add agents
orchestrator.add_agent("technical", technical_agent)
orchestrator.add_agent("sentiment", sentiment_agent)
orchestrator.add_agent("risk", risk_agent)

# Get consensus decision
decision = await orchestrator.get_consensus(market_data)
```

---

## Risk Management API

### RiskManager

Core risk management class.

#### Initialization

```python
from graphwiz_trader.risk.manager import RiskManager

risk_manager = RiskManager(
    account_balance=100000,
    risk_per_trade=0.02,
    max_drawdown_pct=0.10,
    max_position_size=5000
)
```

#### Methods

##### `calculate_position_size(entry_price: float, stop_loss: float, risk_per_trade: float) -> float`

Calculate optimal position size based on risk.

**Example:**
```python
size = risk_manager.calculate_position_size(
    entry_price=50000,
    stop_loss=49000,
    risk_per_trade=0.02
)
# Returns: 0.2 BTC (risking $200 on $1000 price drop)
```

##### `check_risk_limits(trade: Dict) -> Tuple[bool, str]`

Check if trade meets risk criteria.

**Returns:**
```python
(True, "Trade approved")  # or
(False, "Exceeds maximum position size")
```

##### `check_drawdown_limit(current_balance: float) -> bool`

Check if current drawdown exceeds limit.

```python
is_safe = risk_manager.check_drawdown_limit(85000)  # 15% drawdown
# Returns: False if max_drawdown_pct is 0.10
```

##### `adjust_trailing_stop(entry_price: float, current_stop: float, current_price: float, trail_distance: float) -> float`

Adjust trailing stop loss.

```python
new_stop = risk_manager.adjust_trailing_stop(
    entry_price=50000,
    current_stop=49000,
    current_price=51000,
    trail_distance=0.03
)
# Returns: 49470 (3% below current price)
```

---

## Knowledge Graph API

### Neo4jKnowledgeGraph

Interface to Neo4j knowledge graph.

#### Initialization

```python
from graphwiz_trader.knowledge_graph.neo4j_graph import Neo4jKnowledgeGraph

kg = Neo4jKnowledgeGraph(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)
await kg.connect()
```

#### Methods

##### `async query(cypher: str, params: Optional[Dict] = None) -> List[Dict]`

Execute Cypher query.

**Example:**
```python
results = await kg.query("""
    MATCH (a:Asset)-[r:CORRELATED_WITH]->(b:Asset)
    WHERE a.symbol = $symbol
    RETURN b.symbol, r.correlation_coefficient
    ORDER BY r.correlation_coefficient DESC
    LIMIT 10
""", params={"symbol": "BTC"})
```

##### `async store_market_context(data: Dict) -> None`

Store market context in graph.

**Parameters:**
```python
{
    "symbol": "BTC/USDT",
    "price": 50000,
    "volume": 1000,
    "timestamp": "2024-01-01T00:00:00Z",
    "indicators": {...}
}
```

##### `async find_similar_patterns(symbol: str, pattern_type: str, lookback_days: int) -> List[Dict]`

Find similar historical patterns.

**Returns:**
```python
[{
    "pattern": "double_top",
    "date": "2023-06-15",
    "success_rate": 0.75,
    "avg_gain": 0.10
}]
```

##### `async log_agent_decision(decision: Dict) -> None`

Log agent decision to graph.

```python
await kg.log_agent_decision({
    "timestamp": "2024-01-01T00:00:00Z",
    "agent": "technical",
    "action": "buy",
    "confidence": 0.85,
    "symbol": "BTC/USDT"
})
```

---

## Backtesting API

### BacktestEngine

Historical strategy testing engine.

#### Initialization

```python
from graphwiz_trader.backtesting import BacktestEngine

engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)
```

#### Methods

##### `run(data: pd.DataFrame, strategy: str, **params) -> Dict`

Run backtest with given strategy.

**Parameters:**
- `data`: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
- `strategy`: Strategy name (e.g., "sma_cross", "momentum")
- `**params`: Strategy-specific parameters

**Example:**
```python
results = engine.run(
    data=historical_data,
    strategy="sma_cross",
    short_window=20,
    long_window=50
)
```

**Returns:**
```python
{
    "total_return": 0.25,
    "annual_return": 0.30,
    "sharpe_ratio": 1.5,
    "max_drawdown": 0.12,
    "win_rate": 0.60,
    "profit_factor": 1.8,
    "total_trades": 150,
    "winning_trades": 90,
    "losing_trades": 60
}
```

##### `optimize(data: pd.DataFrame, strategy: str, param_grid: Dict) -> Dict`

Optimize strategy parameters.

```python
best_params = engine.optimize(
    data=historical_data,
    strategy="sma_cross",
    param_grid={
        "short_window": [10, 20, 30],
        "long_window": [50, 100, 200]
    }
)
```

---

## Monitoring API

### MetricsTracker

Track and report performance metrics.

#### Initialization

```python
from graphwiz_trader.monitoring.metrics import MetricsTracker

tracker = MetricsTracker()
```

#### Methods

##### `track_trade(trade: Dict) -> None`

Record a trade.

```python
tracker.track_trade({
    "symbol": "BTC/USDT",
    "side": "buy",
    "entry_price": 50000,
    "exit_price": 51000,
    "quantity": 0.5,
    "pnl": 500,
    "timestamp": "2024-01-01T00:00:00Z"
})
```

##### `get_performance_metrics(period: str = "all") -> Dict`

Get performance metrics.

**Parameters:**
- `period`: "day", "week", "month", "all"

**Returns:**
```python
{
    "total_pnl": 5000.0,
    "total_return": 0.05,
    "win_rate": 0.60,
    "sharpe_ratio": 1.5,
    "max_drawdown": 0.08,
    "total_trades": 50
}
```

##### `get_dashboard_data() -> Dict`

Get data for dashboard display.

```python
data = tracker.get_dashboard_data()
# Returns formatted data for Plotly Dash dashboard
```

---

## Trading Modes API

### TradingMode

Configuration for different trading modes.

#### Available Modes

```python
from graphwiz_trader.trading.modes import TradingMode

# Paper trading
mode = TradingMode.PAPER_TRADING

# Live trading
mode = TradingMode.LIVE_TRADING

# High-frequency trading
mode = TradingMode.HFT

# Swing trading
mode = TradingMode.SWING_TRADING
```

#### Configuration

```python
mode = TradingMode(
    name="custom",
    execution_type="simulated",  # or "real"
    max_position_size=5000,
    risk_per_trade=0.02,
    require_confirmation=True
)
```

---

## Error Handling

### Common Exceptions

```python
from graphwiz_trader.exceptions import (
    TradingError,
    RiskLimitExceeded,
    OrderExecutionError,
    AgentTimeout,
    KnowledgeGraphError
)

try:
    result = await engine.execute_trade_signal(market_data)
except RiskLimitExceeded as e:
    print(f"Trade rejected: {e.message}")
except OrderExecutionError as e:
    print(f"Order failed: {e.message}")
    # Implement retry logic
except AgentTimeout as e:
    print(f"Agent timeout: {e.agent_name}")
    # Use fallback logic
```

---

## WebSocket API

### Real-time Market Data

```python
from graphwiz_trader.trading.websocket import MarketDataFeed

feed = MarketDataFeed(exchange="binance")

@feed.on_candle
async def handle_candle(candle):
    print(f"New candle: {candle}")

@feed.on_ticker
async def handle_ticker(ticker):
    print(f"Ticker update: {ticker}")

await feed.subscribe("BTC/USDT", "1m")
await feed.start()
```

---

## Utility Functions

### Technical Indicators

```python
from graphwiz_trader.analysis.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_sma,
    calculate_ema,
    calculate_bollinger_bands
)

# Calculate indicators
rsi = calculate_rsi(prices, period=14)
macd, signal, histogram = calculate_macd(prices)
sma = calculate_sma(prices, period=20)
bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
```

### Data Utilities

```python
from graphwiz_trader.utils.data import (
    fetch_ohlcv,
    resample_data,
    calculate_returns
)

# Fetch historical data
data = await fetch_ohlcv(
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="1h",
    limit=1000
)

# Resample to different timeframe
daily_data = resample_data(data, "1d")

# Calculate returns
returns = calculate_returns(data["close"])
```

---

For more examples, see the `examples/` directory in the repository.
