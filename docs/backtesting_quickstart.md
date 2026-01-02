# Backtesting Framework - Quick Start Guide

## Installation

First, install the additional dependencies for backtesting:

```bash
# Install backtesting dependencies
pip install -r requirements-backtesting.txt

# Or install individually
pip install matplotlib plotly scipy kaleido
```

## Quick Start

### 1. Run the Example Script

The easiest way to get started is to run the example backtesting script:

```bash
cd /opt/git/graphwiz-trader
python3 examples/backtesting_example.py
```

This will:
- Load historical BTC/USDT data for the last 90 days
- Test 4 different strategies (Momentum, Mean Reversion, Grid Trading, DCA)
- Generate performance reports
- Create comparison visualizations

### 2. Run a Simple Backtest

Create a simple Python script:

```python
from datetime import datetime, timedelta
from graphwiz_trader.backtesting import (
    BacktestEngine,
    MomentumStrategy,
)

# Initialize engine
engine = BacktestEngine(
    config_path="config/backtesting.yaml",
)

# Create strategy
strategy = MomentumStrategy(
    lookback_period=20,
    momentum_threshold=0.02,
)

# Run backtest
result = engine.run_backtest(
    strategy=strategy,
    symbol="BTC/USDT",
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    timeframe="1h",
    initial_capital=10000.0,
)

# Print results
metrics = result["metrics"]
print(f"Total Return: {metrics.total_return * 100:.2f}%")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Max Drawdown: {metrics.max_drawdown * 100:.2f}%")

# Generate report
engine.generate_report("Momentum")
```

### 3. Test Multiple Strategies

```python
from graphwiz_trader.backtesting import (
    MomentumStrategy,
    MeanReversionStrategy,
    GridTradingStrategy,
    DCAStrategy,
)

strategies = [
    MomentumStrategy(lookback_period=20),
    MeanReversionStrategy(lookback_period=20),
    GridTradingStrategy(grid_levels=10),
    DCAStrategy(purchase_interval="1D"),
]

results = engine.run_multiple_backtests(
    strategies=strategies,
    symbol="BTC/USDT",
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now(),
    initial_capital=10000.0,
)

# Find best strategy
best = engine.get_best_strategy(metric="sharpe_ratio")
print(f"Best strategy: {best}")

# Generate comparison report
engine.generate_comparison_report()
```

### 4. Create Your Own Strategy

```python
from graphwiz_trader.backtesting.strategies import (
    BaseStrategy,
    Signal,
    Side,
)

class MyStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(name="MyStrategy", **kwargs)

    def generate_signals(self, data):
        signals = []

        # Calculate indicators
        data["sma_short"] = data["close"].rolling(10).mean()
        data["sma_long"] = data["close"].rolling(30).mean()

        # Generate signals
        for i in range(30, len(data)):
            if data["sma_short"].iloc[i] > data["sma_long"].iloc[i]:
                signals.append(Signal(
                    timestamp=data.index[i],
                    side=Side.BUY,
                    price=data["close"].iloc[i],
                    quantity=0.0,
                    reason="SMA crossover",
                    confidence=1.0,
                ))
            elif data["sma_short"].iloc[i] < data["sma_long"].iloc[i]:
                signals.append(Signal(
                    timestamp=data.index[i],
                    side=Side.SELL,
                    price=data["close"].iloc[i],
                    quantity=0.0,
                    reason="SMA crossover",
                    confidence=1.0,
                ))

        return signals

# Use your strategy
strategy = MyStrategy(initial_capital=10000)
engine.run_backtest(strategy, "BTC/USDT", start_date, end_date)
```

## Configuration

Edit `config/backtesting.yaml` to customize:

```yaml
# Exchange settings
exchange: binance
cache_dir: /opt/git/graphwiz-trader/data

# Trading costs
commission_rate: 0.001  # 0.1%
slippage_rate: 0.0005  # 0.05%

# Risk metrics
risk_free_rate: 0.02  # 2% annual

# Output
output_dir: /opt/git/graphwiz-trader/backtests
```

## Understanding the Output

### Performance Metrics

- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return (>1 is good, >2 is excellent)
- **Max Drawdown**: Largest peak-to-valley decline (lower is better)
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss (>1 is profitable)

### Reports

Reports are saved in the `backtests/` directory:
- `{strategy_name}_{timestamp}.html`: Individual strategy report
- `comparison_{timestamp}.html`: Strategy comparison

Open these HTML files in a web browser to view interactive charts.

## Common Use Cases

### 1. Optimize Strategy Parameters

```python
lookback_periods = [10, 15, 20, 25, 30]
results = {}

for period in lookback_periods:
    strategy = MomentumStrategy(lookback_period=period)
    result = engine.run_backtest(strategy, "BTC/USDT", start, end)
    results[period] = result["metrics"].sharpe_ratio

best_period = max(results, key=results.get)
print(f"Best lookback period: {best_period}")
```

### 2. Test Different Timeframes

```python
timeframes = ["15m", "1h", "4h", "1d"]

for tf in timeframes:
    result = engine.run_backtest(
        strategy=strategy,
        symbol="BTC/USDT",
        start_date=start,
        end_date=end,
        timeframe=tf,
    )
    print(f"{tf}: Sharpe = {result['metrics'].sharpe_ratio:.3f}")
```

### 3. Multiple Symbols

```python
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

for symbol in symbols:
    result = engine.run_backtest(
        strategy=strategy,
        symbol=symbol,
        start_date=start,
        end_date=end,
    )
    print(f"{symbol}: Return = {result['metrics'].total_return * 100:.2f}%")
```

## Tips

1. **Start Simple**: Begin with basic strategies before adding complexity
2. **Use Caching**: Enable data caching to speed up repeated backtests
3. **Validate Data**: Always check data quality before backtesting
4. **Realistic Costs**: Include commission and slippage in your tests
5. **Multiple Periods**: Test different time periods to ensure robustness
6. **Compare to Benchmark**: Compare against buy-and-hold strategy

## Troubleshooting

### No signals generated

- Adjust strategy parameters (thresholds, lookback periods)
- Check if data range is sufficient
- Verify symbol and timeframe are valid

### Poor performance

- Test different parameter combinations
- Try different timeframes
- Ensure realistic transaction costs
- Consider market conditions during test period

### Data fetching errors

- Check internet connection
- Verify exchange API is accessible
- Try different timeframes or symbols
- Check CCXT exchange support

## Next Steps

- Read full documentation: `docs/backtesting.md`
- Review example strategies in `src/graphwiz_trader/backtesting/strategies.py`
- Run tests: `pytest tests/test_backtesting.py`
- Create custom strategies for your trading ideas
