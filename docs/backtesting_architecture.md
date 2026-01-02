# Backtesting Framework Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Backtesting Framework                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                   BacktestEngine                        │    │
│  │  - Orchestrates backtests                              │    │
│  │  - Manages results and reports                          │    │
│  │  - Configuration handling                               │    │
│  └────────┬──────────────────────────────────┬────────────┘    │
│           │                                  │                   │
│           ▼                                  ▼                   │
│  ┌────────────────┐              ┌────────────────────┐        │
│  │  DataManager   │              │ PerformanceAnalyzer│        │
│  │  - Fetch data  │              │  - Calculate metrics│       │
│  │  - Cache data  │              │  - Generate plots   │       │
│  │  - Validation  │              │  - Create reports   │       │
│  └───────┬────────┘              └──────────┬─────────┘        │
│          │                                    │                   │
│          ▼                                    │                   │
│  ┌────────────────┐                          │                   │
│  │    CCXT        │                          │                   │
│  │  Exchanges     │                          │                   │
│  └────────────────┘                          │                   │
│                                              │                   │
│                                              ▼                   │
│                                   ┌────────────────────┐        │
│                                   │  Report Generator  │        │
│                                   │  - HTML reports     │        │
│                                   │  - Interactive charts│      │
│                                   └────────────────────┘        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Trading Strategies                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐                                             │
│  │  BaseStrategy  │ ◄────────── Abstract Base Class             │
│  │  - generate_signals()                                         │
│  │  - execute_trade()                                            │
│  │  - apply_costs()                                              │
│  └────────┬───────┘                                             │
│           │                                                       │
│           ├─────────────────┬───────────────────┬──────────────┐ │
│           ▼                 ▼                   ▼              ▼ │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  ┌─────────┐ │
│  │ Momentum    │  │ MeanReversion│  │GridTrading│  │   DCA   │ │
│  │ Strategy    │  │  Strategy    │  │ Strategy  │  │Strategy │ │
│  └─────────────┘  └──────────────┘  └───────────┘  └─────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. Configuration
   ┌─────────────┐
   │    YAML     │
   │ Config File │
   └──────┬──────┘
          │
          ▼

2. Data Loading
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ DataManager │ ──▶ │   CCXT API  │ ──▶ │ OHLCV Data  │
   └─────────────┘     └─────────────┘     └──────┬──────┘
                                                   │
                                                   ▼
                                           ┌─────────────┐
                                           │    Cache    │
                                           └─────────────┘

3. Signal Generation
   ┌─────────────┐     ┌─────────────┐
   │  Strategy   │ ──▶ │   Signals   │
   │             │     │   (List)    │
   └─────────────┘     └──────┬──────┘
                              │
                              ▼

4. Signal Execution
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │   Signals   │ ──▶ │   Engine    │ ──▶ │   Trades    │
   └─────────────┘     └─────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                         ┌─────────────┐
                                         │Equity Curve │
                                         └─────────────┘

5. Performance Analysis
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │Equity Curve │ ──▶ │  Analyzer   │ ──▶ │   Metrics   │
   │   + Trades  │     │             │     │  (15+ ops)  │
   └─────────────┘     └─────────────┘     └──────┬──────┘
                                                      │
                                                      ▼
                                              ┌─────────────┐
                                              │   Reports   │
                                              │   (HTML)    │
                                              └─────────────┘
```

## Class Relationships

```
BacktestEngine
    │
    ├── uses ──▶ DataManager
    │             │
    │             ├── uses ──▶ ccxt.Exchange
    │             └── creates ──▶ pd.DataFrame
    │
    ├── uses ──▶ BaseStrategy (abstract)
    │                 │
    │                 ├── extends ──▶ MomentumStrategy
    │                 ├── extends ──▶ MeanReversionStrategy
    │                 ├── extends ──▶ GridTradingStrategy
    │                 └── extends ──▶ DCAStrategy
    │
    ├── uses ──▶ PerformanceAnalyzer
    │                 │
    │                 ├── creates ──▶ PerformanceMetrics
    │                 ├── creates ──▶ Signal
    │                 └── creates ──▶ Trade
    │
    └── creates ──▶ Reports (HTML)

Data Structures:
    ├── Signal: timestamp, side, price, quantity, reason, confidence
    ├── Trade: entry_time, exit_time, prices, quantity, pnl, reason
    └── PerformanceMetrics: 15+ metric fields
```

## Module Dependencies

```
backtesting/
├── __init__.py
│   └── exports: BacktestEngine, Strategies, Analyzer, DataManager
│
├── data.py
│   ├── imports: ccxt, pandas, numpy, loguru
│   └── exports: DataManager
│
├── strategies.py
│   ├── imports: abc, dataclasses, enum, pandas, numpy, loguru
│   └── exports: BaseStrategy, 4 concrete strategies, Signal, Trade, Side
│
├── analysis.py
│   ├── imports: pandas, numpy, scipy, matplotlib, plotly, loguru
│   └── exports: PerformanceAnalyzer, PerformanceMetrics
│
└── engine.py
    ├── imports: yaml, pandas, loguru
    │             from .data import DataManager
    │             from .strategies import BaseStrategy, Signal, Side, Trade
    │             from .analysis import PerformanceAnalyzer, PerformanceMetrics
    └── exports: BacktestEngine
```

## Key Design Patterns

### 1. Strategy Pattern
- `BaseStrategy` defines interface
- Concrete strategies implement `generate_signals()`
- Easy to add new strategies

### 2. Template Method Pattern
- `BaseStrategy.execute_trade()` defines execution flow
- Subclasses customize signal generation

### 3. Facade Pattern
- `BacktestEngine` provides simple interface
- Hides complexity of data, strategies, analysis

### 4. Dataclass Pattern
- `Signal`, `Trade`, `PerformanceMetrics` use dataclasses
- Clean, immutable data structures

### 5. Factory Pattern
- `DataManager.fetch_ohlcv()` creates DataFrames
- Strategy constructors create configured instances

## Performance Optimizations

1. **Vectorized Operations**
   - Pandas rolling calculations
   - NumPy array operations
   - Avoid loops where possible

2. **Caching**
   - Local data cache
   - Avoid repeated API calls
   - Configurable expiration

3. **Lazy Evaluation**
   - Signals generated on demand
   - Reports generated when requested

4. **Memory Management**
   - Process data in chunks
   - Clean up intermediate results
   - Efficient data structures

## Extension Points

### Adding New Strategies

```python
class MyStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(name="MyStrategy", **kwargs)
        # Custom initialization

    def generate_signals(self, data):
        # Custom signal logic
        return signals
```

### Adding New Metrics

```python
def calculate_custom_metric(self, returns):
    # Custom calculation
    return metric_value
```

### Adding New Visualizations

```python
def plot_custom_chart(self, data):
    fig = go.Figure()
    # Custom plotting
    return fig
```

### Adding New Data Sources

```python
class CustomDataManager(DataManager):
    def fetch_ohlcv(self, symbol, timeframe, ...):
        # Custom data fetching
        return data
```

## Testing Strategy

```
tests/test_backtesting.py
├── TestStrategies
│   ├── test_base_strategy_initialization
│   ├── test_apply_costs_buy
│   ├── test_apply_costs_sell
│   ├── test_momentum_strategy_signals
│   ├── test_mean_reversion_strategy_signals
│   ├── test_grid_trading_strategy_signals
│   └── test_dca_strategy_signals
│
├── TestPerformanceAnalyzer
│   ├── test_calculate_returns
│   ├── test_calculate_volatility
│   ├── test_calculate_sharpe_ratio
│   ├── test_calculate_max_drawdown
│   └── test_calculate_all_metrics
│
├── TestBacktestEngine
│   ├── test_engine_initialization
│   ├── test_execute_signals
│   ├── test_run_backtest
│   └── test_generate_report
│
└── TestDataManager
    ├── test_validate_and_clean_data
    └── test_resample
```

## Configuration Hierarchy

```
1. Code defaults (lowest priority)
2. YAML configuration file
3. Constructor parameters (highest priority)
```

## Error Handling Strategy

1. **Data Fetching**: Retry with exponential backoff
2. **Validation**: Clean or skip invalid data
3. **Execution**: Log warnings, continue processing
4. **Analysis**: Handle edge cases (e.g., division by zero)
5. **Reporting**: Graceful degradation if plots fail

## Security Considerations

1. **API Keys**: Loaded from environment variables
2. **Cache Files**: Permission-restricted
3. **User Input**: Validated and sanitized
4. **File Operations**: Safe path handling
5. **Network**: Rate limiting via CCXT

## Future Enhancements

1. Walk-forward optimization
2. Multi-asset portfolios
3. Advanced order types
4. Risk-based position sizing
5. Monte Carlo simulation
6. Machine learning integration
7. Real-time paper trading
8. Parameter optimization framework
