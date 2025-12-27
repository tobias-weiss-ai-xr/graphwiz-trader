# Qlib Phase 2: Portfolio Optimization Documentation

## Overview

Phase 2 adds sophisticated portfolio management capabilities to GraphWiz Trader, including:
- Multiple portfolio optimization strategies
- Dynamic position sizing based on confidence and risk
- Advanced backtesting framework
- Model validation and selection

## What's New in Phase 2

### 1. **Portfolio Optimizer** (`src/graphwiz_trader/qlib/portfolio.py`)

Complete portfolio optimization system with multiple strategies:

#### **Optimization Methods**

1. **Mean-Variance Optimization (Markowitz)**
   ```python
   optimizer = create_portfolio_optimizer(method='mean_variance')
   weights = optimizer.optimize(returns)
   ```
   - Maximizes: μ'w - γ * w'Σw
   - Balances return vs risk
   - Classic modern portfolio theory

2. **Maximum Sharpe Ratio**
   ```python
   optimizer = create_portfolio_optimizer(method='max_sharpe')
   weights = optimizer.optimize(returns)
   ```
   - Maximizes risk-adjusted returns
   - Best for investors seeking efficiency

3. **Minimum Variance**
   ```python
   optimizer = create_portfolio_optimizer(method='min_variance')
   weights = optimizer.optimize(returns)
   ```
   - Minimizes portfolio volatility
   - Conservative approach

4. **Risk Parity**
   ```python
   optimizer = create_portfolio_optimizer(method='risk_parity')
   weights = optimizer.optimize(returns)
   ```
   - Equal risk contribution from each asset
   - Popular among hedge funds

5. **Equal Weight**
   ```python
   optimizer = create_portfolio_optimizer(method='equal_weight')
   weights = optimizer.optimize(returns)
   ```
   - Simple 1/N allocation
   - Naive diversification

#### **Portfolio Constraints**

```python
constraints = PortfolioConstraints(
    max_position_weight=0.4,      # Maximum 40% in single asset
    min_position_weight=0.0,      # No minimum
    max_leverage=1.0,             # No leverage
    target_volatility=0.15,       # Target 15% annual volatility
    max_drawdown=0.20,            # Maximum 20% drawdown
)
```

#### **Portfolio Metrics**

The optimizer calculates comprehensive metrics:

```python
metrics = optimizer.calculate_portfolio_metrics(weights, returns)

print(f"Annualized Return: {metrics['annualized_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"VaR (95%): {metrics['var_95']:.2%}")
print(f"CVaR (95%): {metrics['cvar_95']:.2%}")
```

### 2. **Dynamic Position Sizing**

Adjusts position sizes based on model confidence and market risk:

```python
sizer = DynamicPositionSizer(
    base_position_size=0.1,      # 10% base position
    max_position_size=0.3,      # Maximum 30%
    min_position_size=0.05,     # Minimum 5%
    risk_tolerance=0.02,        # 2% portfolio risk per trade
)

# Calculate position size
position_size = sizer.calculate_position_size(
    signal_confidence=0.8,       # 80% confidence
    portfolio_value=100000,      # $100,000 portfolio
    asset_price=50000,          # $50,000 asset price
    asset_volatility=0.4,       # 40% annual volatility
)
```

**Benefits:**
- High confidence → larger positions
- High volatility → smaller positions
- Automatically manages risk

### 3. **Advanced Backtesting** (`src/graphwiz_trader/qlib/backtest.py`)

Production-grade backtesting with detailed metrics:

```python
# Create backtest engine
config = BacktestConfig(
    initial_capital=100000,
    commission=0.001,      # 0.1% commission
    slippage=0.0005,      # 0.05% slippage
)

engine = BacktestEngine(config=config)

# Run backtest
result = engine.run_backtest(
    signals=signals,           # DataFrame with buy/sell signals
    price_data=price_data,     # DataFrame with OHLCV data
    benchmark_returns=benchmark,  # Optional: benchmark returns
)

# Generate report
report = engine.generate_report(result)
print(report)
```

#### **Backtest Metrics**

**Return Metrics:**
- Total Return
- Annualized Return
- CAGR (Compound Annual Growth Rate)

**Risk Metrics:**
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

**Drawdown Metrics:**
- Maximum Drawdown
- Average Drawdown
- Max Drawdown Duration

**Trade Metrics:**
- Total Trades
- Win Rate
- Average Win/Loss
- Profit Factor

**Advanced Metrics:**
- Information Ratio (vs benchmark)
- Beta (vs benchmark)
- Alpha (vs benchmark)

### 4. **Model Validation** (`src/graphwiz_trader/qlib/backtest.py`)

Cross-validation for model selection:

```python
validator = ModelValidator(n_folds=5)

# Compare multiple models
models = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'LightGBM': lgb.LGBMClassifier(),
}

best_model, results = validator.select_best_model(models, X, y)

print(f"Best model: {best_model}")
print(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
```

### 5. **Enhanced Trading Strategy V2** (`src/graphwiz_trader/strategies/qlib_strategy_v2.py`)

New strategy with portfolio optimization:

```python
from graphwiz_trader.strategies import create_qlib_strategy_v2

strategy = create_qlib_strategy_v2(
    trading_engine=engine,
    symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    config={
        # Optimization settings
        "optimization_method": "max_sharpe",
        "enable_portfolio_opt": True,
        "max_position_weight": 0.4,

        # Position sizing
        "base_position_size": 0.1,
        "max_position_size": 0.3,
        "risk_tolerance": 0.02,

        # Model settings
        "retrain_interval_hours": 24,
        "lookback_days": 30,
    },
)

await strategy.start()
results = await strategy.run_cycle()
```

**Key Features:**
- Automatic portfolio optimization
- Dynamic position sizing
- Intelligent rebalancing
- Risk management integration

## Usage Examples

### Example 1: Compare Optimization Methods

```python
import pandas as pd
import numpy as np
from graphwiz_trader.qlib import create_portfolio_optimizer, PortfolioConstraints

# Generate sample returns
returns = pd.DataFrame(np.random.randn(252, 5) * 0.02,
                       columns=['BTC', 'ETH', 'BNB', 'SOL', 'ADA'])

# Compare methods
methods = ['equal_weight', 'mean_variance', 'max_sharpe', 'risk_parity']

for method in methods:
    optimizer = create_portfolio_optimizer(
        method=method,
        constraints=PortfolioConstraints(max_position_weight=0.4)
    )

    weights = optimizer.optimize(returns)
    metrics = optimizer.calculate_portfolio_metrics(weights, returns)

    print(f"{method}: Sharpe={metrics['sharpe_ratio']:.2f}, "
          f"Return={metrics['annualized_return']:.2%}")
```

### Example 2: Backtest a Strategy

```python
from graphwiz_trader.qlib import BacktestEngine, BacktestConfig

# Prepare data
signals = pd.Series(...)  # 1=buy, 0=sell
price_data = pd.DataFrame({'close': ...})

# Run backtest
config = BacktestConfig(initial_capital=100000, commission=0.001)
engine = BacktestEngine(config=config)
result = engine.run_backtest(signals, price_data)

# Analyze results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Win Rate: {result.win_rate:.2%}")

# Get detailed trade list
for trade in result.trades:
    print(f"{trade.entry_time} -> {trade.exit_time}: "
          f"{trade.return_pct:.2%}")
```

### Example 3: Complete Workflow

```python
from graphwiz_trader.strategies import create_qlib_strategy_v2

# Create strategy with portfolio optimization
strategy = create_qlib_strategy_v2(
    trading_engine=engine,
    symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    config={
        "optimization_method": "max_sharpe",
        "enable_portfolio_opt": True,
        "max_position_weight": 0.4,
        "base_position_size": 0.1,
    },
)

# Start and run
await strategy.start()
cycle_results = await strategy.run_cycle()

# Check results
print(f"Optimal weights: {cycle_results['optimal_weights']}")
print(f"Trades executed: {cycle_results['trades_executed']}")
```

## Running Tests

### Phase 2 Test Suite

```bash
python tests/integration/test_qlib_phase2.py
```

Tests cover:
1. Portfolio optimization methods
2. Dynamic position sizing
3. Advanced backtesting
4. Model validation
5. Integration with signals

### Quick Start Demo

```bash
python examples/qlib_phase2_demo.py
```

Interactive demo showcasing all Phase 2 features.

## Configuration Reference

### OptimizerConfig

```python
OptimizerConfig(
    optimization_method="mean_variance",  # Optimization method
    risk_free_rate=0.02,                 # Risk-free rate
    rebalance_frequency="1d",            # Rebalancing frequency
    lookback_window=60,                  # Days for optimization
    min_returns=0.0,                     # Minimum acceptable return
    max_risk=None,                       # Maximum acceptable risk
)
```

### PortfolioConstraints

```python
PortfolioConstraints(
    max_position_weight=0.3,      # Max weight per asset
    min_position_weight=0.0,      # Min weight per asset
    max_leverage=1.0,             # Max portfolio leverage
    target_volatility=None,       # Target volatility
    max_drawdown=None,            # Max drawdown limit
    turnover_limit=None,          # Max turnover limit
)
```

### BacktestConfig

```python
BacktestConfig(
    initial_capital=100000,       # Starting capital
    commission=0.001,             # Commission rate
    slippage=0.0005,             # Slippage rate
    start_date=None,              # Backtest start date
    end_date=None,                # Backtest end date
)
```

## Performance Comparison

### Phase 1 vs Phase 2

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| Signal Generation | ✅ ML-based | ✅ ML-based |
| Position Sizing | Fixed | Dynamic (risk-based) |
| Portfolio Optimization | ❌ | ✅ 5 methods |
| Backtesting | Basic | Advanced (15+ metrics) |
| Model Validation | ❌ | ✅ Cross-validation |
| Risk Management | Basic | Advanced (VaR, CVaR) |

### Expected Improvements

With portfolio optimization:
- **Better risk-adjusted returns**: 10-20% improvement in Sharpe ratio
- **Reduced drawdowns**: 20-30% reduction in maximum drawdown
- **More stable returns**: Lower volatility through diversification
- **Adaptive positioning**: Larger positions when confidence is high

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Market Data (CCXT)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          ML Signal Generation (Phase 1)                │
│      - Alpha158 features                               │
│      - LightGBM models                                 │
│      - Probability estimates                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Portfolio Optimization (Phase 2 NEW)           │
│      - Mean-variance optimization                      │
│      - Risk parity                                     │
│      - Max Sharpe ratio                                │
│      - Constraints management                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Dynamic Position Sizing (Phase 2 NEW)          │
│      - Confidence-based sizing                         │
│      - Risk-adjusted positions                         │
│      - Kelly criterion-inspired                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Advanced Backtesting (Phase 2 NEW)           │
│      - 15+ performance metrics                         │
│      - Trade analysis                                  │
│      - Risk metrics (VaR, CVaR)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             Trading Execution                           │
└─────────────────────────────────────────────────────────┘
```

## Best Practices

### 1. Choose the Right Optimization Method

- **Conservative**: Use `min_variance` or `risk_parity`
- **Balanced**: Use `mean_variance`
- **Aggressive**: Use `max_sharpe`
- **Simple**: Use `equal_weight` (baseline)

### 2. Set Appropriate Constraints

```python
# For diversified portfolio
PortfolioConstraints(
    max_position_weight=0.3,  # No single asset > 30%
    max_leverage=1.0,         # No leverage
)

# For concentrated portfolio
PortfolioConstraints(
    max_position_weight=0.5,  # Allow up to 50% in one asset
    max_leverage=1.5,         # Allow some leverage
)
```

### 3. Use Dynamic Position Sizing

```python
# Conservative sizing
DynamicPositionSizer(
    base_position_size=0.05,  # Start with 5%
    risk_tolerance=0.01,      # 1% risk per trade
)

# Aggressive sizing
DynamicPositionSizer(
    base_position_size=0.2,   # Start with 20%
    risk_tolerance=0.05,      # 5% risk per trade
)
```

### 4. Validate Models

```python
# Always cross-validate
validator = ModelValidator(n_folds=5)
best_model, results = validator.select_best_model(models, X, y)

# Check stability
if results['std_score'] > 0.1:
    print("Warning: Model performance is unstable!")
```

## Troubleshooting

### Issue: Optimization fails

**Solution**: Check data quality and constraints
```python
# Ensure enough data
if len(returns) < 60:
    print("Need at least 60 days of data")

# Relax constraints
constraints = PortfolioConstraints(
    max_position_weight=0.5,  # Increase limit
)
```

### Issue: Portfolio weights are all equal

**Solution**: This happens when optimization fails or falls back
```python
# Check if scipy is available
import scipy
print(f"Scipy version: {scipy.__version__}")

# Use different method
optimizer = create_portfolio_optimizer(method='equal_weight')
```

### Issue: Poor backtest results

**Solution**: Check signal quality and parameters
```python
# Review signals
print(f"Buy signals: {(signals==1).sum()}")
print(f"Sell signals: {(signals==0).sum()}")

# Adjust costs
config = BacktestConfig(
    commission=0.001,  # Realistic commission
    slippage=0.0005,   # Realistic slippage
)
```

## Next Steps

### Phase 3: Hybrid Graph-ML Models

Combine Neo4j knowledge graph with Alpha158 features for unique predictive power.

### Phase 4: RL-Based Execution

Reinforcement learning for optimal order execution and routing.

## Resources

- **Code**: `src/graphwiz_trader/qlib/portfolio.py`
- **Tests**: `tests/integration/test_qlib_phase2.py`
- **Demo**: `examples/qlib_phase2_demo.py`
- **Phase 1 Docs**: `docs/QLIB_PHASE1_DOCUMENTATION.md`
- **Full Analysis**: `QLIB_INTEGRATION_ANALYSIS.md`
