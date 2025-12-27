# Modern Strategies Integration Guide

## Overview

This guide explains how to integrate modern trading strategies (Grid Trading, Smart DCA, AMM, Triangular Arbitrage) with the GraphWiz trading engine for both paper trading and live trading.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Quick Start](#quick-start)
3. [Strategy Adapter](#strategy-adapter)
4. [Paper Trading](#paper-trading)
5. [Live Trading](#live-trading)
6. [Examples](#examples)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Engine                          │
│  - Order execution                                         │
│  - Risk management                                         │
│  - Portfolio management                                    │
│  - Position tracking                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Modern Strategy Adapter                        │
│  - Unified interface                                       │
│  - Signal generation                                       │
│  - State management                                        │
│  - Trade conversion                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│Grid Trading │  │Smart DCA    │  │    AMM      │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Data Flow

1. **Market Data** → Strategy Adapter
2. **Signal Generation** → Trading Signals
3. **Order Conversion** → Trading Engine Format
4. **Execution** → Exchange/Portfolio
5. **State Update** → Strategy State

---

## Quick Start

### 1. Import Required Modules

```python
from graphwiz_trader.strategies import (
    GridTradingStrategy,
    GridTradingMode,
    SmartDCAStrategy,
    ModernStrategyAdapter,
    create_modern_strategy_adapter,
)
```

### 2. Create Strategy

```python
# Option 1: Direct instantiation
strategy = GridTradingStrategy(
    symbol='BTC/USDT',
    upper_price=55000,
    lower_price=45000,
    num_grids=10,
    grid_mode=GridTradingMode.GEOMETRIC,
    investment_amount=10000,
)

# Option 2: Factory function
strategy = create_modern_strategy(
    strategy_type="grid_trading",
    symbol='BTC/USDT',
    upper_price=55000,
    lower_price=45000,
    num_grids=10,
)
```

### 3. Create Adapter

```python
adapter = ModernStrategyAdapter(strategy)
# or
adapter = create_modern_strategy_adapter(
    strategy_type="grid_trading",
    symbol='BTC/USDT',
    upper_price=55000,
    lower_price=45000,
    num_grids=10,
)
```

### 4. Generate Signals

```python
signals = adapter.generate_trading_signals(
    current_price=50000,
    historical_data=market_data,
)
```

### 5. Execute Trades

```python
for order in signals['orders']:
    result = trading_engine.execute_trade(
        symbol=order['symbol'],
        side=order['side'],
        amount=order['amount'],
        price=order['price'],
        order_type=order['order_type'],
    )

    # Update strategy state
    adapter.execute_trade(result)
```

---

## Strategy Adapter

### What is the Adapter?

The `ModernStrategyAdapter` provides a **unified interface** for all modern strategies, making them compatible with the trading engine. It handles:

- ✅ Signal generation
- ✅ Order format conversion
- ✅ State management
- ✅ Performance tracking

### Key Methods

#### 1. `generate_trading_signals()`

Generate trading signals from the strategy.

```python
signals = adapter.generate_trading_signals(
    current_price=50000,
    historical_data=market_data,
    # Strategy-specific parameters
    current_inventory_a=10,  # For AMM
    current_inventory_b=30000,
    price_data=price_dict,  # For arbitrage
)
```

**Returns:**
```python
{
    "status": "success",
    "strategy": "grid_trading",
    "current_price": 50000,
    "orders": [...],
    "metadata": {...},
}
```

#### 2. `execute_trade()`

Update strategy state after trade execution.

```python
trade_result = {
    'status': 'executed',
    'symbol': 'BTC/USDT',
    'side': 'buy',
    'amount': 0.1,
    'price': 50000,
    'metadata': {...},
}

adapter.execute_trade(trade_result)
```

#### 3. `get_strategy_status()`

Get current strategy status and metrics.

```python
status = adapter.get_strategy_status(current_price=50000)
```

**For Smart DCA:**
```python
{
    'total_invested': 3500.00,
    'total_quantity': 0.066862,
    'avg_purchase_price': 52346.88,
    'current_value': 3677.39,
    'pnl': 177.39,
    'pnl_pct': 5.07,
}
```

**For AMM:**
```python
{
    'total_trades': 3,
    'total_fees': 0.01,
    'adverse_selection_rate': 0.00,
    'avg_price_impact': 0.0056,
}
```

---

## Paper Trading

### Setting Up Paper Trading

```python
from graphwiz_trader.strategies import ModernStrategyAdapter
from graphwiz_trader.trading.exchange import create_exchange

# Create strategy adapter
strategy = GridTradingStrategy(...)
adapter = ModernStrategyAdapter(strategy)

# Create paper trader
trader = ModernStrategyPaperTrader(
    strategy_adapter=adapter,
    exchange_name="binance",
    symbol="BTC/USDT",
    initial_capital=10000,
)
```

### Running Paper Trading

```python
# Single iteration
data = trader.fetch_latest_data(limit=100)
result = trader.generate_and_execute_signals(data)

print(f"Price: ${result['current_price']}")
print(f"Portfolio: ${result['portfolio_value']}")

# Continuous trading
trader.start(interval_seconds=3600, max_iterations=10)
```

### Performance Metrics

```python
metrics = trader.get_performance_metrics()

print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
```

---

## Live Trading

### Risk Management

Before live trading, ensure proper risk management:

```python
trading_config = {
    "max_position_size": 0.1,  # 10% of portfolio
    "max_open_positions": 5,
    "stop_loss_percent": 2.0,
    "take_profit_percent": 5.0,
    "min_trade_amount": 10,
}
```

### Live Trading Setup

```python
from graphwiz_trader.trading.engine import TradingEngine

# Initialize trading engine
engine = TradingEngine(
    trading_config=trading_config,
    exchanges_config={
        "binance": {
            "enabled": True,
            "api_key": "your_api_key",
            "api_secret": "your_api_secret",
            "sandbox": True,  # Start with testnet!
        }
    },
    knowledge_graph=kg,
    agent_orchestrator=agents,
)

# Start engine
engine.start()

# Create strategy adapter
adapter = create_modern_strategy_adapter(
    strategy_type="grid_trading",
    symbol='BTC/USDT',
    upper_price=55000,
    lower_price=45000,
    num_grids=10,
)

# Trading loop
while True:
    # Fetch market data
    market_data = fetch_market_data()

    # Generate signals
    signals = adapter.generate_trading_signals(
        current_price=market_data['price'],
        historical_data=market_data['history'],
    )

    # Execute orders
    for order in signals['orders']:
        result = engine.execute_trade(
            symbol=order['symbol'],
            side=order['side'],
            amount=order['amount'],
            price=order['price'],
            order_type=order['order_type'],
        )

        # Update strategy state
        adapter.execute_trade(result)

    time.sleep(60)  # Wait 1 minute
```

---

## Examples

### Example 1: Grid Trading

```python
from graphwiz_trader.strategies import GridTradingStrategy, GridTradingMode

# Create strategy
strategy = GridTradingStrategy(
    symbol='BTC/USDT',
    upper_price=55000,
    lower_price=45000,
    num_grids=10,
    grid_mode=GridTradingMode.GEOMETRIC,
    investment_amount=10000,
    dynamic_rebalancing=True,
)

adapter = ModernStrategyAdapter(strategy)

# Generate signals
signals = adapter.generate_trading_signals(
    current_price=50000,
    historical_data=market_data,
)

# Execute grid orders
for order in signals['orders']:
    print(f"{order['side'].upper()} {order['amount']:.6f} @ ${order['price']:.2f}")
```

### Example 2: Smart DCA

```python
from graphwiz_trader.strategies import SmartDCAStrategy

# Create strategy
strategy = SmartDCAStrategy(
    symbol='BTC/USDT',
    total_investment=10000,
    purchase_frequency='weekly',
    purchase_amount=500,
    volatility_adjustment=True,
    momentum_boost=0.5,
)

adapter = ModernStrategyAdapter(strategy)

# Weekly DCA loop
prices = [55000, 53000, 51000, 49000, 51000]

for price in prices:
    signals = adapter.generate_trading_signals(price)

    if signals['should_execute']:
        order = signals['order']
        print(f"Buy {order['amount']:.6f} BTC @ ${price:.2f}")
        print(f"Reason: {order['metadata']['reason']}")

        # Execute and update state
        trade_result = execute_trade(order)
        adapter.execute_trade(trade_result)

# Check portfolio
status = adapter.get_strategy_status(current_price=51000)
print(f"P&L: {status['pnl_pct']:.2f}%")
```

### Example 3: AMM

```python
from graphwiz_trader.strategies import AutomatedMarketMakingStrategy

# Create strategy
strategy = AutomatedMarketMakingStrategy(
    token_a='ETH',
    token_b='USDT',
    pool_price=3000,
    price_range=(2400, 3600),
    base_fee_rate=0.003,
    inventory_target_ratio=0.5,
)

adapter = ModernStrategyAdapter(strategy)

# Check if rebalance needed
signals = adapter.generate_trading_signals(
    current_price=3000,
    current_inventory_a=10,
    current_inventory_b=30000,
)

if signals['needs_rebalance']:
    print(f"Current ratio: {signals['current_ratio_a']:.2%}")
    print(f"Target ratio: {signals['target_ratio']:.2%}")

    for order in signals['orders']:
        print(f"Rebalance: {order['side']} {order['amount']}")
```

### Example 4: Triangular Arbitrage

```python
from graphwiz_trader.strategies import TriangularArbitrageStrategy

# Create strategy
strategy = TriangularArbitrageStrategy(
    exchanges=['binance', 'okx'],
    trading_pairs=['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
    min_profit_threshold=0.005,
    fee_rate=0.001,
)

adapter = ModernStrategyAdapter(strategy)

# Update prices and find opportunities
price_data = {
    'binance': {
        'BTC/USDT': 50000,
        'ETH/BTC': 0.060,
        'ETH/USDT': 3000,
    },
    'okx': {
        'BTC/USDT': 50100,
        'ETH/BTC': 0.061,
        'ETH/USDT': 2990,
    },
}

signals = adapter.generate_trading_signals(
    current_price=0,
    price_data=price_data,
)

if signals['opportunities_found'] > 0:
    opportunity = signals['orders'][0]['metadata']
    print(f"Found arbitrage on {opportunity['exchange']}")
    print(f"Profit: {opportunity['profit_pct']:.2%}")
    print(f"Path: {' → '.join(opportunity['path'])}")
```

---

## API Reference

### ModernStrategyAdapter

#### Constructor

```python
ModernStrategyAdapter(strategy)
```

**Parameters:**
- `strategy`: Strategy instance (GridTradingStrategy, SmartDCAStrategy, etc.)

#### Methods

##### `generate_trading_signals()`

```python
generate_trading_signals(
    current_price: float,
    historical_data: Optional[pd.DataFrame] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `current_price`: Current market price
- `historical_data`: Historical OHLCV data
- `**kwargs`: Strategy-specific parameters

**Returns:**
Dictionary with trading signals

##### `execute_trade()`

```python
execute_trade(trade_result: Dict[str, Any]) -> bool
```

**Parameters:**
- `trade_result`: Result from trading engine

**Returns:**
True if strategy updated successfully

##### `get_strategy_status()`

```python
get_strategy_status(current_price: float) -> Dict[str, Any]
```

**Parameters:**
- `current_price`: Current market price

**Returns:**
Strategy status dictionary

### Factory Functions

#### `create_modern_strategy_adapter()`

```python
create_modern_strategy_adapter(
    strategy_type: str,
    **kwargs
) -> ModernStrategyAdapter
```

**Parameters:**
- `strategy_type`: One of 'grid_trading', 'smart_dca', 'amm', 'triangular_arbitrage'
- `**kwargs`: Strategy-specific parameters

**Returns:**
ModernStrategyAdapter instance

---

## Best Practices

### 1. Start with Paper Trading

Always test strategies with paper trading before live trading:

```python
# ✅ Good: Start with paper trading
trader = ModernStrategyPaperTrader(adapter, initial_capital=10000)
trader.start(interval_seconds=3600, max_iterations=100)

# ❌ Bad: Jump straight to live trading
engine = TradingEngine(...)  # Without testing!
```

### 2. Use Proper Risk Management

```python
# ✅ Good: Conservative risk limits
trading_config = {
    "max_position_size": 0.05,  # 5% max
    "stop_loss_percent": 2.0,
    "take_profit_percent": 5.0,
}

# ❌ Bad: No risk limits
trading_config = {}  # Dangerous!
```

### 3. Monitor Strategy Performance

```python
# ✅ Good: Track metrics
status = adapter.get_strategy_status(current_price)
logger.info(f"Strategy P&L: {status.get('pnl_pct', 0):.2f}%")

if status.get('pnl_pct', 0) < -10:
    logger.warning("Strategy losing money! Consider pausing.")

# ❌ Bad: No monitoring
# Just let it run without checking
```

### 4. Handle Errors Gracefully

```python
# ✅ Good: Error handling
try:
    signals = adapter.generate_trading_signals(current_price, data)
    execute_orders(signals['orders'])
except Exception as e:
    logger.error(f"Strategy error: {e}")
    # Continue running, don't crash

# ❌ Bad: No error handling
signals = adapter.generate_trading_signals(current_price, data)
execute_orders(signals['orders'])  # Could crash!
```

### 5. Use Testnet Before Mainnet

```python
# ✅ Good: Start with testnet
exchanges_config = {
    "binance": {
        "sandbox": True,  # Testnet
        "api_key": testnet_key,
    }
}

# ❌ Bad: Start with mainnet
exchanges_config = {
    "binance": {
        "sandbox": False,  # Real money!
        "api_key": mainnet_key,
    }
}
```

### 6. Diversify Strategies

```python
# ✅ Good: Multiple strategies
strategies = [
    create_modern_strategy_adapter("grid_trading", symbol='BTC/USDT'),
    create_modern_strategy_adapter("smart_dca", symbol='ETH/USDT'),
    create_modern_strategy_adapter("amm", token_a='SOL', token_b='USDT'),
]

# ❌ Bad: All eggs in one basket
strategies = [
    create_modern_strategy_adapter("grid_trading", symbol='BTC/USDT'),
]  # Only one strategy
```

---

## Troubleshooting

### Problem: Strategy not generating signals

**Solution:** Check if historical data is provided when required:

```python
# For strategies with historical data analysis
signals = adapter.generate_trading_signals(
    current_price=50000,
    historical_data=market_data,  # Required for some strategies
)
```

### Problem: Orders not executing

**Solution:** Check order format and trading engine status:

```python
result = engine.execute_trade(
    symbol=order['symbol'],
    side=order['side'],
    amount=order['amount'],
    price=order['price'],
    order_type=order['order_type'],
)

print(f"Status: {result['status']}")
print(f"Reason: {result.get('message', 'N/A')}")
```

### Problem: Strategy state not updating

**Solution:** Ensure `execute_trade()` is called after each execution:

```python
result = engine.execute_trade(...)
adapter.execute_trade(result)  # Update state!
```

---

## Next Steps

1. **Run Examples:**
   ```bash
   python examples/modern_strategies_trading_integration.py
   python examples/modern_strategies_paper_trading.py
   ```

2. **Backtest with Historical Data:**
   - Collect historical price data
   - Run paper trading simulations
   - Compare different parameters

3. **Deploy to Paper Trading:**
   - Connect to live market data
   - Run strategy for 1-2 weeks
   - Monitor performance metrics

4. **Graduate to Live Trading:**
   - Start with small amounts
   - Use testnet first
   - Gradually increase position sizes

---

## Additional Resources

- **Documentation:** `docs/MODERN_STRATEGIES_DOCUMENTATION.md`
- **Examples:** `examples/modern_strategies_*.py`
- **Tests:** `tests/integration/test_modern_strategies.py`
- **Research:** `MODERN_STRATEGIES_COMPLETE.md` (with links to papers)
