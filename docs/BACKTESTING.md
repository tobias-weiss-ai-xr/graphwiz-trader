# Backtesting Guide

Complete guide to backtesting trading strategies with GraphWiz Trader.

## Quick Start

### 1. Run a Quick Backtest (Generated Data)

```bash
# Run all strategies and compare
python scripts/backtest_example.py --strategy all

# Run specific strategy
python scripts/backtest_example.py --strategy sma
python scripts/backtest_example.py --strategy rsi
python scripts/backtest_example.py --strategy custom

# Specify symbol and time period
python scripts/backtest_example.py --symbol ETH/USDT --days 30 --strategy sma
```

### 2. Fetch Real Historical Data

```bash
# Fetch 30 days of hourly BTC/USDT data from Binance
python scripts/fetch_data.py --exchange binance --symbol BTC/USDT --timeframe 1h --days 30 --save

# Fetch daily data for multiple symbols
python scripts/fetch_data.py --symbol ETH/USDT --timeframe 1d --days 90 --save
python scripts/fetch_data.py --symbol SOL/USDT --timeframe 1h --days 30 --save
```

## Built-in Strategies

### SMA Crossover Strategy

**Logic:** Buy when fast SMA crosses above slow SMA, sell when it crosses below.

```python
from graphwiz_trader.backtesting import SimpleMovingAverageStrategy

strategy = SimpleMovingAverageStrategy(fast_period=10, slow_period=30)
```

**Parameters:**
- `fast_period`: Short-term SMA period (default: 10)
- `slow_period`: Long-term SMA period (default: 30)

**Best for:** Trending markets

### RSI Mean Reversion Strategy

**Logic:** Buy when RSI is oversold (< 30), sell when overbought (> 70).

```python
from graphwiz_trader.backtesting import RSIMeanReversionStrategy

strategy = RSIMeanReversionStrategy(oversold=30, overbought=70)
```

**Parameters:**
- `oversold`: RSI level to buy (default: 30)
- `overbought`: RSI level to sell (default: 70)

**Best for:** Range-bound markets

## Custom Strategies

Create your own strategy by defining a function that takes a context dictionary and returns 'buy', 'sell', or 'hold':

```python
def my_strategy(context):
    """Custom strategy example."""
    indicators = context.get('technical_indicators', {})
    current_price = context.get('current_price', 0)

    # Get indicator values
    rsi = indicators.get('rsi', {}).get('value')
    macd = indicators.get('macd', {}).get('value')

    # Your trading logic
    if rsi and rsi < 30:
        return 'buy'
    elif rsi and rsi > 70:
        return 'sell'

    return 'hold'

# Run backtest
from graphwiz_trader.backtesting import BacktestEngine

engine = BacktestEngine(initial_capital=10000, commission=0.001)
result = engine.run_backtest(data, my_strategy, "BTC/USDT")
```

## Context Dictionary

Your strategy function receives a context dictionary with:

```python
{
    'symbol': 'BTC/USDT',
    'current_price': 50000.0,
    'timestamp': datetime(...),
    'data_points': 150,

    'technical_indicators': {
        'rsi': {'value': 65.5, 'signal': 'neutral'},
        'macd': {
            'value': 100.5,
            'signal': 98.2,
            'histogram': 2.3
        },
        'sma_fast': {'value': 50500.0},
        'sma_slow': {'value': 50000.0},
        # ... more indicators
    },

    'volume': 1500000.0,
    'avg_volume': 1200000.0,
}
```

## Backtest Results

The backtest returns a `BacktestResult` object with:

```python
result.symbol              # Trading symbol
result.start_date          # Backtest start date
result.end_date            # Backtest end date
result.initial_capital     # Starting capital
result.final_capital       # Ending capital
result.total_return        # Total profit/loss
result.total_return_pct    # Return percentage
result.trades              # List of Trade objects
result.win_rate            # Win rate (0-1)
result.sharpe_ratio        # Sharpe ratio
result.sortino_ratio       # Sortino ratio
result.max_drawdown        # Maximum drawdown %
result.metrics            # Additional metrics
```

### Example: Analyzing Results

```python
# Print summary
print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
print(f"Win Rate: {result.win_rate*100:.2f}%")

# Access trades
for trade in result.trades:
    print(f"{trade.action} {trade.quantity} @ {trade.price}")

# Get equity curve
equity_curve = result.metrics['equity_curve']
```

## Advanced Usage

### Parameter Optimization

Test different parameter combinations:

```python
from itertools import product

# Test different SMA periods
fast_periods = [5, 10, 15]
slow_periods = [20, 30, 40]

results = []
for fast, slow in product(fast_periods, slow_periods):
    strategy = SimpleMovingAverageStrategy(fast, slow)
    result = engine.run_backtest(data, strategy, "BTC/USDT")

    results.append({
        'fast': fast,
        'slow': slow,
        'return': result.total_return_pct,
        'sharpe': result.sharpe_ratio
    })

# Find best parameters
best = max(results, key=lambda x: x['return'])
print(f"Best: Fast={best['fast']}, Slow={best['slow']}, Return={best['return']:.2f}%")
```

### Multi-Symbol Backtesting

```python
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

all_results = {}
for symbol in symbols:
    data = fetch_data(symbol)
    result = engine.run_backtest(data, strategy, symbol)
    all_results[symbol] = result

# Compare results
for symbol, result in all_results.items():
    print(f"{symbol}: {result.total_return_pct:.2f}%")
```

### Walk-Forward Testing

Test strategy robustness over different time periods:

```python
# Split data into chunks
chunk_size = len(data) // 5
chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

results = []
for i, chunk in enumerate(chunks):
    result = engine.run_backtest(chunk, strategy, "BTC/USDT")
    results.append(result.total_return_pct)
    print(f"Period {i+1}: {result.total_return_pct:.2f}%")

print(f"Average return: {sum(results)/len(results):.2f}%")
print(f"Std deviation: {np.std(results):.2f}%")
```

## Performance Metrics

Understanding the metrics:

### Sharpe Ratio
- Measures risk-adjusted returns
- **> 1**: Good
- **> 2**: Very good
- **> 3**: Excellent
- **< 0**: Strategy loses money

### Sortino Ratio
- Similar to Sharpe but only penalizes downside risk
- Higher is better

### Max Drawdown
- Largest peak-to-trough decline
- Lower is better
- **< 10%**: Excellent
- **10-20%**: Good
- **> 20%**: High risk

### Win Rate
- Percentage of profitable trades
- Higher is better, but not everything
- A strategy with 40% win rate can be profitable if wins are larger than losses

## Best Practices

1. **Start Simple**: Begin with basic strategies before adding complexity
2. **Use Realistic Costs**: Include commission and slippage
3. **Out-of-Sample Testing**: Hold out some data for validation
4. **Avoid Overfitting**: Don't optimize too much on historical data
5. **Consider Market Conditions**: Strategies work differently in trending vs ranging markets
6. **Risk Management**: Always use position sizing and stop-losses
7. **Transaction Costs**: Frequent trading eats into profits

## Common Pitfalls

❌ **Don't:**
- Over-optimize parameters
- Ignore transaction costs
- Use too little historical data
- Test only in bull markets
- Assume future = past

✅ **Do:**
- Test on multiple time periods
- Include realistic costs
- Validate on out-of-sample data
- Monitor live performance
- Adjust strategy as markets change

## Next Steps

1. Run example backtests: `python scripts/backtest_example.py --strategy all`
2. Create your own strategy
3. Fetch real market data: `python scripts/fetch_data.py --save`
4. Optimize parameters
5. Paper trade before going live

## Troubleshooting

**Not enough trades?**
- Check if your strategy conditions are too restrictive
- Try longer time periods
- Adjust indicator thresholds

**Poor performance?**
- Verify indicator calculations
- Check for look-ahead bias
- Ensure transaction costs are realistic
- Try different market conditions

**Memory errors?**
- Reduce data size
- Process in chunks
- Use larger timeframes

For more examples, see `scripts/backtest_example.py`.
