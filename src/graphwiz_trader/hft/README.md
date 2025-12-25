# High-Frequency Trading (HFT) Module

Ultra-low latency trading capabilities for cryptocurrency markets with built-in risk management and knowledge graph integration.

## Features

### Market Data
- **Real-time WebSocket feeds** from multiple exchanges
- **Order book synchronization** with depth management
- **Price history tracking** for statistical analysis
- **Event-driven architecture** for minimal latency

### Strategies
- **Statistical Arbitrage**: Mean reversion using z-score analysis
- **Cross-Exchange Arbitrage**: Exploits price differences across exchanges
- **Extensible framework**: Easy to add custom strategies

### Risk Management
- Position size limits (per symbol and global)
- Exposure management
- Order rate limiting
- Circuit breaker for loss protection
- Real-time drawdown monitoring

### Execution
- Multi-exchange support (200+ via ccxt)
- Market and limit orders
- Latency tracking (millisecond precision)
- Simultaneous order placement for arbitrage

## Quick Start

```python
import asyncio
from graphwiz_trader.hft import (
    WebSocketMarketData,
    OrderBookManager,
    FastOrderExecutor,
    HFTRiskManager,
    StatisticalArbitrage,
)

async def main():
    # Initialize components
    exchanges_config = {
        "binance": {
            "enabled": True,
            "api_key": "your_key",
            "api_secret": "your_secret",
        }
    }

    # Market data
    market_data = WebSocketMarketData(exchanges_config)

    # Order book manager
    ob_manager = OrderBookManager(max_depth=20)

    # Risk manager
    risk_config = {
        "max_position_size": 1.0,
        "max_exposure": 10000,
        "circuit_breaker": -0.05,
    }
    risk_manager = HFTRiskManager(risk_config)

    # Strategy
    strategy_config = {"lookback": 100, "z_threshold": 2.0}
    strategy = StatisticalArbitrage(strategy_config, knowledge_graph=None)

    # Register callbacks
    async def on_orderbook(data):
        ob_manager.update(data["exchange"], data["symbol"], data)
        await strategy.on_orderbook_update(data)

    market_data.register_callback("orderbook", on_orderbook)

    # Start trading
    await market_data.start()
    await market_data.connect("binance", ["BTC/USDT", "ETH/USDT"])
    await strategy.start()

    # Run indefinitely
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
```

## Module Structure

```
hft/
├── __init__.py              # Module exports
├── market_data.py           # WebSocket market data feeds
├── orderbook.py             # Order book management & arbitrage detection
├── executor.py              # Fast order execution engine
├── risk.py                  # Risk management & position tracking
├── strategies/
│   ├── __init__.py          # Strategy exports
│   ├── base.py              # Abstract strategy base class
│   ├── stat_arb.py          # Statistical arbitrage strategy
│   └── cross_exchange_arb.py # Cross-exchange arbitrage
└── README.md                # This file
```

## Strategy Development

Create custom strategies by extending `HFTStrategy`:

```python
from graphwiz_trader.hft.strategies import HFTStrategy

class MyStrategy(HFTStrategy):
    async def on_market_data(self, data):
        # Process market ticks
        pass

    async def on_orderbook_update(self, orderbook):
        # Process order book updates
        pass

    async def generate_signal(self):
        # Generate trading signals
        return {
            "symbol": "BTC/USDT",
            "action": "buy",
            "confidence": 0.95,
        }
```

## Risk Management

The `HFTRiskManager` provides comprehensive risk controls:

```python
risk_manager = HFTRiskManager({
    "max_position_size": 1.0,           # Max position per symbol
    "max_exposure": 10000,              # Total capital at risk
    "max_orders_per_second": 10,        # Rate limit
    "circuit_breaker": -0.05,           # -5% loss triggers halt
    "max_drawdown_pct": 10.0,           # Max drawdown percentage
})

# Check order before execution
approved, reason = await risk_manager.check_order(order)
if approved:
    # Execute order
    pass
else:
    print(f"Order rejected: {reason}")

# Get risk metrics
metrics = risk_manager.get_risk_metrics()
print(f"Daily P&L: ${metrics['daily_pnl']:.2f}")
print(f"Exposure: {metrics['exposure_utilization_pct']:.1f}%")
```

## Order Execution

Fast order execution with latency tracking:

```python
executor = FastOrderExecutor(exchanges_config)

# Place single order
order = await executor.place_order(
    exchange="binance",
    symbol="BTC/USDT",
    side="buy",
    amount=0.1,
    order_type="market"
)
print(f"Order executed in {order['latency_ms']:.2f}ms")

# Place simultaneous orders (for arbitrage)
orders = [
    {"exchange": "binance", "symbol": "BTC/USDT", "side": "buy", "amount": 0.1},
    {"exchange": "okx", "symbol": "BTC/USDT", "side": "sell", "amount": 0.1},
]
results = await executor.place_simultaneous_orders(orders)

# Get average latency
avg_latency = executor.get_average_latency("binance")
print(f"Average latency: {avg_latency:.2f}ms")
```

## Order Book Analytics

Analyze order books for trading opportunities:

```python
ob_manager = OrderBookManager()

# Update order books
ob_manager.update("binance", "BTC/USDT", orderbook_data)
ob_manager.update("okx", "BTC/USDT", orderbook_data)

# Find arbitrage opportunities
opportunities = ob_manager.get_arbitrage_opportunities(
    "BTC/USDT",
    min_profit_bps=5.0  # Minimum 5 basis points profit
)

for opp in opportunities:
    print(f"Buy on {opp['buy_exchange']} at {opp['buy_price']}")
    print(f"Sell on {opp['sell_exchange']} at {opp['sell_price']}")
    print(f"Expected profit: {opp['profit_bps']:.2f} bps")

# Get order book metrics
book = ob_manager.get_book("binance", "BTC/USDT")
print(f"Spread: {book.spread_bps:.2f} bps")
print(f"Mid price: ${book.mid_price:.2f}")
print(f"Imbalance: {book.imbalance:.2f}")

# Calculate VWAP for large order
vwap = book.get_vwap("buy", quantity=1.0)
print(f"VWAP for 1.0 BTC: ${vwap:.2f}")
```

## Performance Monitoring

Track strategy performance in real-time:

```python
# Strategy automatically tracks performance
performance = strategy.get_performance()

print(f"Total trades: {performance['trades']}")
print(f"Win rate: {performance['win_rate']:.2%}")
print(f"Total P&L: ${performance['profit_loss']:.2f}")
print(f"Winning trades: {performance['winning_trades']}")
print(f"Losing trades: {performance['losing_trades']}")

# View detailed performance logs
await strategy.log_performance()
```

## Configuration

Configure via YAML:

```yaml
hft:
  enabled: true

  market_data:
    exchanges:
      binance:
        enabled: true
        symbols: ["BTC/USDT", "ETH/USDT"]
        websocket: true

  strategies:
    statistical_arbitrage:
      enabled: true
      lookback: 100
      z_threshold: 2.0
      max_position_size: 0.5

    cross_exchange_arbitrage:
      enabled: true
      min_profit_bps: 5
      max_position_size: 0.1
      fee_bps: 10

  risk:
    max_position_size: 1.0
    max_exposure: 10000
    max_orders_per_second: 10
    circuit_breaker_threshold: -0.05
```

## Testing

Run the test suite:

```bash
# Run all HFT tests
pytest tests/test_hft.py -v

# Run specific test class
pytest tests/test_hft.py::TestOrderBook -v

# Run with coverage
pytest tests/test_hft.py --cov=graphwiz_trader.hft
```

## Dependencies

- `ccxt` / `ccxt.pro` - Exchange connectivity
- `numpy` - Statistical calculations
- `loguru` - Logging
- `aiohttp` - Async HTTP client
- `asyncio` - Async programming

## Performance Tips

1. **Use WebSocket connections** instead of REST API for market data
2. **Enable order batching** for simultaneous execution
3. **Run on low-latency infrastructure** (colocation near exchanges)
4. **Monitor latency metrics** and optimize bottlenecks
5. **Use limit orders** when spread allows to capture rebates
6. **Implement proper error handling** and reconnection logic

## Known Limitations

- Requires `ccxt.pro` for WebSocket support (Pro version)
- Network latency depends on geographic proximity to exchanges
- Some exchanges have rate limits that may restrict HFT strategies
- Circuit breaker requires manual reset after triggering

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/tobias-weiss-ai-xr/graphwiz-trader/issues
- Documentation: See `docs/HFT_INTEGRATION_PLAN.md` for detailed architecture
- Tests: See `tests/test_hft.py` for usage examples

## License

MIT License - See LICENSE file for details
