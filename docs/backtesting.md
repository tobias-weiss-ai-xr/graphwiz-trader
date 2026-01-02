# Backtesting Framework Documentation

## Overview

The graphwiz-trader backtesting framework provides comprehensive tools for testing trading strategies against historical market data. It includes realistic cost modeling, detailed performance analysis, and interactive visualizations.

## Architecture

```
graphwiz_trader/backtesting/
├── __init__.py           # Module exports
├── data.py              # Historical data management
├── strategies.py        # Trading strategy implementations
├── analysis.py          # Performance analysis and metrics
└── engine.py            # Main backtesting engine
```

## Features

### Data Management (`data.py`)

**DataManager Class:**
- Fetch historical OHLCV data from CCXT exchanges
- Local caching for performance optimization
- Multiple timeframe support (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
- Data validation and cleaning (outliers, missing values, consistency checks)
- Data resampling to different timeframes

**Key Methods:**
```python
data_manager = DataManager(
    cache_dir="/path/to/cache",
    exchange_name="binance",
    enable_cache=True,
)

# Fetch data
df = data_manager.fetch_ohlcv(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now(),
)

# Resample data
df_daily = data_manager.resample(df, "1d")
```

### Trading Strategies (`strategies.py`)

#### BaseStrategy

Abstract base class for all strategies.

**Attributes:**
- `name`: Strategy name
- `initial_capital`: Starting capital
- `commission_rate`: Commission per trade (as fraction, e.g., 0.001 = 0.1%)
- `slippage_rate`: Expected slippage per trade (as fraction)

**Methods:**
- `generate_signals(data)`: Generate trading signals
- `execute_trade(signal, current_position)`: Execute a trade signal

#### MomentumStrategy

Buys when price shows strong positive momentum, sells when momentum reverses.

**Parameters:**
- `lookback_period`: Period to calculate momentum (default: 20)
- `momentum_threshold`: Minimum momentum to trigger trade (default: 0.02 = 2%)

```python
strategy = MomentumStrategy(
    lookback_period=20,
    momentum_threshold=0.02,
    initial_capital=10000,
    commission_rate=0.001,
)
```

#### MeanReversionStrategy

Buys when price is below mean (oversold), sells when above mean (overbought).

**Parameters:**
- `lookback_period`: Period for moving average/std dev (default: 20)
- `std_dev_threshold`: Std dev threshold for signals (default: 2.0)

```python
strategy = MeanReversionStrategy(
    lookback_period=20,
    std_dev_threshold=2.0,
    initial_capital=10000,
)
```

#### GridTradingStrategy

Places buy/sell orders at regular intervals around current price.

**Parameters:**
- `grid_levels`: Number of grid levels (default: 10)
- `grid_spacing`: Percentage spacing between levels (default: 0.01 = 1%)

```python
strategy = GridTradingStrategy(
    grid_levels=10,
    grid_spacing=0.01,
    initial_capital=10000,
)
```

#### DCAStrategy

Dollar Cost Averaging - buys fixed amount at regular intervals.

**Parameters:**
- `purchase_interval`: Time between purchases (e.g., "1D", "1W")
- `purchase_amount`: Fixed amount to buy each interval

```python
strategy = DCAStrategy(
    purchase_interval="1D",
    purchase_amount=1000.0,
    initial_capital=10000,
)
```

### Performance Analysis (`analysis.py`)

**PerformanceAnalyzer Class:**

**Key Metrics:**
- **Total Return**: Overall portfolio return
- **Annualized Return**: Return annualized over backtest period
- **Volatility**: Standard deviation of returns (annualized)
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Calmar Ratio**: Annualized return / max drawdown
- **Omega Ratio**: Probability-weighted ratio of gains to losses

**Methods:**
```python
analyzer = PerformanceAnalyzer(risk_free_rate=0.02)

# Calculate all metrics
metrics = analyzer.calculate_all_metrics(
    equity_curve=equity_series,
    trades=trades_list,
)

# Generate plots
fig = analyzer.plot_equity_curve(equity_curve)
fig = analyzer.plot_drawdown(equity_curve)
fig = analyzer.plot_returns_distribution(returns)

# Compare strategies
comparison_fig = analyzer.compare_strategies(
    results={"Strategy1": metrics1, "Strategy2": metrics2},
)

# Generate HTML report
analyzer.generate_report(
    metrics=metrics,
    equity_curve=equity_curve,
    returns=returns,
    output_path="report.html",
)
```

### Backtesting Engine (`engine.py`)

**BacktestEngine Class:**

Main coordinator for running backtests.

**Initialization:**
```python
engine = BacktestEngine(
    config_path="config/backtesting.yaml",
    output_dir="backtests/",
)
```

**Running Backtests:**

Single strategy:
```python
result = engine.run_backtest(
    strategy=strategy,
    symbol="BTC/USDT",
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now(),
    timeframe="1h",
    initial_capital=10000.0,
)
```

Multiple strategies:
```python
results = engine.run_multiple_backtests(
    strategies=[strategy1, strategy2, strategy3],
    symbol="BTC/USDT",
    start_date=start_date,
    end_date=end_date,
    timeframe="1h",
    initial_capital=10000.0,
)
```

**Generating Reports:**
```python
# Individual strategy report
engine.generate_report("Momentum")

# Comparison report for all strategies
engine.generate_comparison_report()

# Save results to YAML
engine.save_results()
```

## Configuration

Create a YAML configuration file (`config/backtesting.yaml`):

```yaml
# Exchange settings
exchange: binance
enable_cache: true
cache_dir: /opt/git/graphwiz-trader/data

# Risk-free rate for Sharpe/Sortino
risk_free_rate: 0.02  # 2% annual

# Default backtest settings
initial_capital: 10000.0
default_timeframe: 1h

# Commission and slippage
commission_rate: 0.001  # 0.1% per trade
slippage_rate: 0.0005  # 0.05% per trade

# Output settings
output_dir: /opt/git/graphwiz-trader/backtests
save_equity_curves: true
save_trade_history: true

# Strategy parameters
strategies:
  momentum:
    lookback_period: 20
    momentum_threshold: 0.02

  mean_reversion:
    lookback_period: 20
    std_dev_threshold: 2.0

  grid_trading:
    grid_levels: 10
    grid_spacing: 0.01

  dca:
    purchase_interval: 1D
    purchase_amount: 1000.0
```

## Example Usage

See `examples/backtesting_example.py` for a complete example.

```python
from datetime import datetime, timedelta
from graphwiz_trader.backtesting import (
    BacktestEngine,
    MomentumStrategy,
    MeanReversionStrategy,
)

# Initialize engine
engine = BacktestEngine(
    config_path="config/backtesting.yaml",
)

# Create strategies
strategies = [
    MomentumStrategy(lookback_period=20, momentum_threshold=0.02),
    MeanReversionStrategy(lookback_period=20, std_dev_threshold=2.0),
]

# Run backtests
results = engine.run_multiple_backtests(
    strategies=strategies,
    symbol="BTC/USDT",
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now(),
    timeframe="1h",
    initial_capital=10000.0,
)

# Display results
for name, result in results.items():
    metrics = result["metrics"]
    print(f"{name}:")
    print(f"  Total Return: {metrics.total_return * 100:.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {metrics.max_drawdown * 100:.2f}%")

# Generate reports
engine.generate_comparison_report()
engine.save_results()
```

## Creating Custom Strategies

To create a custom strategy, inherit from `BaseStrategy`:

```python
from graphwiz_trader.backtesting.strategies import BaseStrategy, Signal, Side

class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1, param2, **kwargs):
        super().__init__(name="MyCustomStrategy", **kwargs)
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, data):
        signals = []

        # Your signal generation logic here
        for i in range(len(data)):
            # Generate signal based on your strategy
            if your_buy_condition:
                signals.append(Signal(
                    timestamp=data.index[i],
                    side=Side.BUY,
                    price=data["close"].iloc[i],
                    quantity=0.0,  # Calculated automatically
                    reason="Buy signal",
                    confidence=1.0,
                ))
            elif your_sell_condition:
                signals.append(Signal(
                    timestamp=data.index[i],
                    side=Side.SELL,
                    price=data["close"].iloc[i],
                    quantity=0.0,
                    reason="Sell signal",
                    confidence=1.0,
                ))

        return signals
```

## Performance Considerations

### Vectorized Operations

The framework uses pandas vectorized operations for performance:
- Indicator calculations use pandas rolling operations
- Signal generation is vectorized where possible
- Batch execution of signals

### Caching

Data is cached locally to avoid repeated API calls:
- Cache files stored in `cache_dir`
- Automatic cache validation
- Configurable cache expiration

### Memory Management

For large datasets:
- Process data in chunks
- Use appropriate timeframes (e.g., 1h instead of 1m)
- Enable caching to reduce memory usage

## Testing

Run the test suite:

```bash
pytest tests/test_backtesting.py -v
```

Tests cover:
- Strategy signal generation
- Performance metrics calculation
- Trade execution
- Data validation
- Report generation

## Output

### Reports

HTML reports include:
- Performance metrics summary
- Interactive equity curve
- Drawdown visualization
- Returns distribution
- Trade analysis

### Saved Files

Results are saved to the output directory:
- `backtests/{strategy_name}_{timestamp}.html`: Individual strategy reports
- `backtests/comparison_{timestamp}.html`: Strategy comparison
- `backtests/backtest_results_{timestamp}.yaml`: Raw results data

## Best Practices

1. **Data Quality**: Always validate data before backtesting
2. **Realistic Costs**: Include commission and slippage
3. **Out-of-Sample Testing**: Reserve data for validation
4. **Multiple Timeframes**: Test across different timeframes
5. **Parameter Sensitivity**: Test parameter robustness
6. **Benchmark Comparison**: Compare against buy-and-hold
7. **Risk Management**: Monitor drawdown and volatility

## Limitations

- Historical data may not represent future conditions
- Exchange API limits for data fetching
- Simplified order execution (no partial fills)
- No position sizing based on risk
- Limited to CCXT-supported exchanges

## Future Enhancements

- [ ] Walk-forward optimization
- [ ] Multi-asset portfolio backtesting
- [ ] Advanced order types (stop-loss, take-profit)
- [ ] Position sizing based on volatility
- [ ] Monte Carlo simulation
- [ ] Parameter optimization framework
- [ ] Live trading integration
