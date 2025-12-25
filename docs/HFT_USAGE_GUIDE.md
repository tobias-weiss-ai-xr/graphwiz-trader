# HFT Module Usage Guide

This guide explains how to use the High-Frequency Trading (HFT) module in GraphWiz Trader.

## Quick Start

### 1. Enable HFT in Configuration

Edit your `config.yaml` file and enable the HFT module:

```yaml
hft:
  enabled: true  # Enable HFT module

  market_data:
    exchanges:
      binance:
        enabled: true
        symbols:
          - "BTC/USDT"
          - "ETH/USDT"
        websocket: true

  strategies:
    statistical_arbitrage:
      enabled: true
      lookback_period: 100
      z_score_threshold: 2.0
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

### 2. Initialize and Start HFT Engine

```python
import asyncio
from graphwiz_trader.hft import HFTEngine
from graphwiz_trader.graph.neo4j_graph import Neo4jGraph
from graphwiz_trader.utils.config import load_config

async def main():
    # Load configuration
    config = load_config("config/config.yaml")

    # Initialize knowledge graph
    kg = Neo4jGraph(config["neo4j"])
    await kg.connect()

    # Initialize HFT engine
    hft_config = config.get("hft", {})
    hft_engine = HFTEngine(hft_config, kg)

    # Start the engine
    await hft_engine.start()

    # Run for some time
    await asyncio.sleep(3600)  # Run for 1 hour

    # Stop the engine
    await hft_engine.stop()

    # Close knowledge graph connection
    await kg.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Monitoring Engine Status

```python
# Get current status
status = hft_engine.get_status()

print(f"Engine Running: {status['running']}")
print(f"Circuit Breaker: {status['circuit_breaker']}")

# Strategy performance
for strategy in status['strategies']:
    print(f"\nStrategy: {strategy['name']}")
    print(f"  Running: {strategy['running']}")
    print(f"  Trades: {strategy['performance']['trades']}")
    print(f"  Win Rate: {strategy['performance']['win_rate']:.2%}")
    print(f"  P&L: ${strategy['performance']['profit_loss']:.2f}")

# Risk metrics
risk = status['risk_metrics']
print(f"\nRisk Metrics:")
print(f"  Daily P&L: ${risk['daily_pnl']:.2f}")
print(f"  Max Drawdown: {risk['max_drawdown_pct']:.2f}%")
```

### Accessing Analytics

```python
# Get analytics summary for the last 7 days
analytics = await hft_engine.get_analytics_summary(days=7)

# Top performing strategies
for performer in analytics.get('top_performers', []):
    print(f"Strategy: {performer['strategy']}")
    print(f"  Total Trades: {performer['total_trades']}")
    print(f"  Total P&L: ${performer['total_pnl']:.2f}")
    print(f"  Win Rate: {performer['win_rate']:.2f}%")

# System performance
sys_perf = analytics.get('system_performance', {})
print(f"\nSystem Performance:")
print(f"  Avg CPU: {sys_perf.get('avg_cpu_percent', 0):.1f}%")
print(f"  Avg Memory: {sys_perf.get('avg_memory_percent', 0):.1f}%")
```

### Manual Order Execution

```python
# Execute a single order with risk checks
order = {
    'exchange': 'binance',
    'symbol': 'BTC/USDT',
    'side': 'buy',
    'amount': 0.01,
    'price': None,  # Market order
    'order_type': 'market'
}

result = await hft_engine.execute_order(order)

if result:
    print(f"Order executed: {result['order_id']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
else:
    print("Order rejected by risk manager")
```

### Simultaneous Orders (Arbitrage)

```python
# Execute multiple orders simultaneously
orders = [
    {
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': 0.01,
        'order_type': 'market'
    },
    {
        'exchange': 'okx',
        'symbol': 'BTC/USDT',
        'side': 'sell',
        'amount': 0.01,
        'order_type': 'market'
    }
]

results = await hft_engine.execute_simultaneous_orders(orders)

for result in results:
    if isinstance(result, dict):
        print(f"Order on {result['exchange']}: {result['status']}")
```

### Reset Circuit Breaker

```python
# Reset circuit breaker after manual review
hft_engine.reset_circuit_breaker()
print("Circuit breaker reset - trading resumed")
```

## Component Usage

### Using Individual Components

You can also use HFT components individually:

#### Order Book Manager

```python
from graphwiz_trader.hft import OrderBookManager

# Initialize
ob_manager = OrderBookManager()

# Update order book
orderbook_data = {
    'timestamp': 1234567890,
    'bids': [[50000, 1.5], [49999, 2.0]],
    'asks': [[50001, 1.0], [50002, 1.5]]
}

ob_manager.update('binance', 'BTC/USDT', orderbook_data)

# Find arbitrage opportunities
opportunities = ob_manager.get_arbitrage_opportunities('BTC/USDT', min_profit_bps=5)

for opp in opportunities:
    print(f"Buy on {opp['buy_exchange']} @ {opp['buy_price']}")
    print(f"Sell on {opp['sell_exchange']} @ {opp['sell_price']}")
    print(f"Profit: {opp['profit_bps']} bps")
```

#### Risk Manager

```python
from graphwiz_trader.hft import HFTRiskManager

# Initialize
risk_config = {
    'max_position_size': 1.0,
    'max_exposure': 10000,
    'max_orders_per_second': 10,
    'circuit_breaker_threshold': -0.05
}

risk_manager = HFTRiskManager(risk_config)

# Check order
order = {
    'symbol': 'BTC/USDT',
    'exchange': 'binance',
    'side': 'buy',
    'amount': 0.5
}

approved, reason = await risk_manager.check_order(order)

if approved:
    print("Order approved")
else:
    print(f"Order rejected: {reason}")
```

#### Performance Monitor

```python
from graphwiz_trader.hft import PerformanceMonitor

# Initialize
monitor = PerformanceMonitor(knowledge_graph, interval=5)

# Start monitoring
await monitor.start()

# Record latency
monitor.record_latency('order_execution', 8.5)  # 8.5ms
monitor.record_latency('market_data', 0.8)  # 0.8ms

# Get current metrics
metrics = monitor.get_current_metrics()
print(f"CPU: {metrics['cpu_percent']:.1f}%")
print(f"Memory: {metrics['memory_percent']:.1f}%")

# Get latency summary
latency = monitor.get_latency_metrics()
print(f"Avg Order Latency: {latency.get('avg_order_execution_latency_ms', 0):.2f}ms")

# Stop monitoring
await monitor.stop()
```

#### Analytics

```python
from graphwiz_trader.hft import HFTAnalytics

# Initialize
analytics = HFTAnalytics(knowledge_graph)

# Store a pattern
pattern = {
    'type': 'mean_reversion',
    'symbol': 'BTC/USDT',
    'exchange': 'binance',
    'success_rate': 0.65,
    'avg_profit_bps': 12.5,
    'occurrence_count': 42,
    'indicators': {'rsi': 30, 'macd': -50}
}

await analytics.store_pattern(pattern)

# Get strategy performance
perf = await analytics.get_strategy_performance('StatisticalArbitrage', days=7)
print(f"Total Trades: {perf['total_trades']}")
print(f"Total P&L: ${perf['total_pnl']:.2f}")
print(f"Win Rate: {perf['win_rate']:.2f}%")

# Get best patterns
best_patterns = await analytics.get_best_patterns('BTC/USDT', limit=10)
for pattern in best_patterns:
    print(f"Pattern: {pattern['type']} - {pattern['avg_profit_bps']} bps")
```

## Configuration Reference

### HFT Configuration Options

```yaml
hft:
  enabled: false                    # Enable/disable HFT module

  market_data:
    exchanges:
      <exchange_name>:
        enabled: true               # Enable exchange
        symbols: []                 # List of trading symbols
        websocket: true             # Use WebSocket feeds

  strategies:
    statistical_arbitrage:
      enabled: true
      lookback_period: 100         # Price history lookback
      z_score_threshold: 2.0       # Z-score signal threshold
      max_position_size: 0.5       # Max position as fraction

    cross_exchange_arbitrage:
      enabled: true
      min_profit_bps: 5            # Minimum profit (basis points)
      max_position_size: 0.1       # Max position size
      fee_bps: 10                  # Trading fees

  risk:
    max_position_size: 1.0         # Max position per symbol
    max_exposure: 10000            # Max total exposure (USDT)
    max_orders_per_second: 10      # Rate limit
    circuit_breaker_threshold: -0.05  # -5% daily loss
    max_drawdown_pct: 10.0         # Max drawdown percentage

  performance:
    max_order_latency_ms: 10       # Target order latency
    max_processing_latency_ms: 5   # Target processing latency
    enable_profiling: false        # Enable profiling
    monitoring_interval: 5         # Metrics interval (seconds)

  knowledge_graph:
    store_trades: true             # Store trades in Neo4j
    store_patterns: true           # Store patterns
    store_arbitrage_opportunities: true
    analyze_correlations: true
    retention_days: 30             # Data retention period
```

## Best Practices

1. **Start with Paper Trading**: Test strategies thoroughly before live trading
2. **Monitor Performance**: Regularly check latency metrics and system health
3. **Set Conservative Limits**: Start with low position sizes and exposure limits
4. **Use Circuit Breakers**: Always keep circuit breakers enabled
5. **Review Logs**: Check logs regularly for errors and warnings
6. **Backup Knowledge Graph**: Regularly backup your Neo4j database
7. **Test Failover**: Practice circuit breaker resets and error recovery
8. **Monitor Costs**: Track trading fees and ensure they don't exceed profits

## Troubleshooting

### High Latency
- Check network connection to exchanges
- Review system resource usage (CPU, memory)
- Consider colocation or VPS closer to exchanges

### Circuit Breaker Triggered
- Review trade history to identify losing trades
- Check strategy parameters and market conditions
- Reset circuit breaker after manual review

### Orders Rejected
- Check risk manager logs for rejection reason
- Verify position limits and exposure limits
- Ensure order rate is within limits

### WebSocket Connection Issues
- Verify exchange API credentials
- Check firewall and network settings
- Review exchange status and rate limits

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/graphwiz-trader/issues
- Documentation: See `docs/` folder
- HFT Integration Plan: `docs/HFT_INTEGRATION_PLAN.md`
- Implementation Status: `docs/HFT_IMPLEMENTATION_STATUS.md`
