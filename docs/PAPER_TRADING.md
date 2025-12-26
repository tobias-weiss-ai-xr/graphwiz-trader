# Paper Trading Guide

Paper trading allows you to test trading strategies without risking real money. It simulates live trading by:
- Fetching real market data from exchanges
- Generating trading signals using your strategy
- Executing virtual trades
- Tracking performance metrics
- **NOT executing real trades**

## Quick Start

### Single Run Test
```bash
# Run once and see current status
python scripts/paper_trade.py --symbol BTC/USDT --capital 10000
```

### Continuous Paper Trading
```bash
# Run continuously (check every hour)
python scripts/paper_trade.py --symbol BTC/USDT --continuous

# Run with custom interval (e.g., every 30 minutes)
python scripts/paper_trade.py --symbol BTC/USDT --continuous --interval 1800

# Run for specific number of iterations
python scripts/paper_trade.py --symbol BTC/USDT --iterations 24
```

### Custom Strategy Parameters
```bash
# Custom RSI levels
python scripts/paper_trade.py --symbol BTC/USDT --oversold 20 --overbought 70

# Custom starting capital
python scripts/paper_trade.py --symbol BTC/USDT --capital 50000

# Custom commission rate
python scripts/paper_trade.py --symbol BTC/USDT --commission 0.002  # 0.2%
```

## Recommended Configuration

Based on backtesting results, use these parameters:

```bash
python scripts/paper_trade.py \
    --symbol BTC/USDT \
    --capital 10000 \
    --oversold 25 \
    --overbought 65 \
    --continuous \
    --interval 3600  # Check every hour
```

## Output Files

Results are saved to `data/paper_trading/`:

1. **Trades CSV**: `BTC_USDT_trades_YYYYMMDD_HHMMSS.csv`
   - All executed trades with timestamps, prices, quantities
   - Buy/sell actions with costs and proceeds
   - P&L for sell trades

2. **Equity Curve CSV**: `BTC_USDT_equity_YYYYMMDD_HHMMSS.csv`
   - Portfolio value over time
   - Capital and position tracking
   - Useful for performance visualization

3. **Summary JSON**: `BTC_USDT_summary_YYYYMMDD_HHMMSS.json`
   - Performance metrics (return, win rate, drawdown)
   - Trade statistics
   - Current position

## Interpreting Results

### Performance Summary

```
================================================================================
PAPER TRADING PERFORMANCE SUMMARY
================================================================================
Initial Capital:  $10,000.00
Final Value:      $10,500.00
Total Return:     $500.00 (+5.00%)
--------------------------------------------------------------------------------
Total Trades:     6
  Buy Trades:     3
  Sell Trades:    3
Win Rate:         100%
Winning Trades:   3/3
Max Drawdown:     2.45%
================================================================================
```

### Key Metrics

- **Total Return**: Overall profit/loss percentage
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Current Position**: Current holdings (if any)

### When to Go Live

Consider live trading only after:

✅ **Paper trading for 2-4 weeks**
- Strategy performs consistently
- Positive returns over multiple periods
- Win rate matches backtest expectations

✅ **Risk management verified**
- Max drawdown within acceptable limits
- No unexpected losses
- Position sizing working correctly

✅ **Market conditions tested**
- Tested in different market conditions
- Strategy adapts to volatility
- No overfitting to specific period

⚠️ **Never go live if:**
- Paper trading shows losses
- Drawdown exceeds 5-10%
- Win rate below 50%
- Strategy doesn't match backtest results

## Advanced Usage

### Multiple Symbols
```bash
# Run paper trading for multiple symbols
python scripts/paper_trade.py --symbol BTC/USDT --continuous &
python scripts/paper_trade.py --symbol ETH/USDT --continuous &
python scripts/paper_trade.py --symbol SOL/USDT --continuous &
```

### Custom Exchange
```bash
# Use different exchange
python scripts/paper_trade.py --exchange coinbase --symbol BTC/USD
```

### Monitoring Performance

Load equity curve in Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load equity curve
df = pd.read_csv("data/paper_trading/BTC_USDT_equity_*.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Plot portfolio value
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["total_value"])
plt.title("Paper Trading Portfolio Value")
plt.xlabel("Time")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.show()
```

## Troubleshooting

### No Trades Executed
- Check if RSI is in buy/sell zone (must be < 25 or > 65)
- Verify sufficient data is being fetched
- Check logs for errors: `tail -f logs/paper_trading.log`

### Connection Errors
- Verify internet connection
- Check exchange API status
- Try different exchange if Binance is down

### Signal Generated But No Trade
- Check if capital is available (for buys)
- Check if position exists (for sells)
- Review trade execution logs

## Next Steps

After successful paper trading:

1. **Review results**: Analyze equity curve, win rate, drawdown
2. **Adjust parameters**: If performance differs from backtests
3. **Test longer**: Run for 2-4 weeks minimum
4. **Consider live trading**: Only if paper trading is profitable
5. **Start small**: Begin with 1-2% of portfolio per trade

See [LIVE_TRADING.md](LIVE_TRADING.md) for live trading setup.
