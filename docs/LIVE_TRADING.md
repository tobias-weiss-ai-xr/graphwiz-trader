# Live Trading Guide

⚠️ **CRITICAL WARNING**: This guide covers REAL trading with REAL money. You can lose money. Always test thoroughly with paper trading first.

## Prerequisites

Before attempting live trading, ensure you have:

✅ **Completed Paper Trading**
- 2-4 weeks of profitable paper trading
- Consistent performance across different market conditions
- Win rate matches backtest expectations (60%+)
- Max drawdown within acceptable limits (< 5%)

✅ **Account Setup**
- Exchange account created and verified
- API keys generated with appropriate permissions
- 2FA enabled on exchange account
- Funds deposited (start small!)

✅ **Risk Management**
- Understood all safety limits
- Set maximum loss limits you can afford
- Emergency stop procedures planned
- Position sizing rules established

## Quick Start

### 1. Set API Credentials

```bash
export EXCHANGE_API_KEY="your_api_key_here"
export EXCHANGE_API_SECRET="your_api_secret_here"
```

Or use a `.env` file:
```bash
echo "EXCHANGE_API_KEY=your_api_key_here" > .env
echo "EXCHANGE_API_SECRET=your_api_secret_here" >> .env
```

### 2. Test Run (No Trades)

```bash
# Run once to test connection and setup
python scripts/live_trade.py \
    --symbol BTC/USDT \
    --max-position 50 \
    --max-daily-loss 100 \
    --test
```

This will:
- Connect to exchange
- Fetch current prices
- Generate signals
- **NOT execute trades** (test mode)

### 3. Start Live Trading

```bash
# Start with conservative limits
python scripts/live_trade.py \
    --symbol BTC/USDT \
    --max-position 100 \
    --max-position-pct 0.02 \
    --max-daily-loss 200 \
    --max-daily-trades 5 \
    --interval 3600
```

## Safety Limits

### Position Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-position` | $100 | Maximum $ per position |
| `--max-position-pct` | 0.02 | Max 2% of portfolio per trade |
| `--max-total-exposure` | 0.10 | Max 10% total exposure |

### Daily Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-daily-loss` | $200 | Stop if daily loss exceeds $200 |
| `--max-daily-loss-pct` | 0.05 | Stop if daily loss exceeds 5% |
| `--max-daily-trades` | 5 | Maximum 5 trades per day |

### Risk Management

**Automatic Stop Loss**: -2% from entry price
**Automatic Take Profit**: +5% from entry price
**Emergency Shutdown**: Closes all positions immediately

## Recommended Starting Configuration

### Phase 1: Testing (First Week)

```bash
python scripts/live_trade.py \
    --symbol BTC/USDT \
    --max-position 50 \
    --max-position-pct 0.01 \
    --max-daily-loss 100 \
    --max-daily-trades 3 \
    --interval 3600
```

**Goal**: Test system, ensure no bugs, verify trade execution
**Expected**: 1-3 trades, small profits or losses

### Phase 2: Small Scale (Second Week)

If Phase 1 successful:
```bash
python scripts/live_trade.py \
    --symbol BTC/USDT \
    --max-position 100 \
    --max-position-pct 0.02 \
    --max-daily-loss 200 \
    --max-daily-trades 5 \
    --interval 3600
```

**Goal**: Build track record, refine strategy
**Expected**: 3-10 trades, profitable overall

### Phase 3: Full Scale (After 2+ Weeks)

Only if Phase 2 is consistently profitable:
```bash
python scripts/live_trade.py \
    --symbol BTC/USDT \
    --max-position 500 \
    --max-position-pct 0.05 \
    --max-daily-loss 500 \
    --max-daily-trades 10 \
    --interval 3600
```

## Monitoring

### Real-Time Monitoring

The system logs everything:
```
2025-12-26 10:00:00 | WARNING | 🚨 LIVE TRADING MODE - REAL MONEY WILL BE USED
2025-12-26 10:00:01 | INFO | Price: $87,054.84, Signal: buy, Action: buy
2025-12-26 10:00:05 | SUCCESS | ✅ BUY ORDER EXECUTED
```

### Check Status

View saved results:
```bash
cat data/live_trading/BTC_USDT_summary_*.json
```

### Emergency Stop

If something goes wrong:

1. **Ctrl+C**: Graceful shutdown, closes positions on next iteration
2. **Kill process**: Immediate stop, positions remain open
3. **Manual intervention**: Log into exchange and manually close positions

## Risk Management Rules

### ✅ DO

- Start with minimum amounts ($50-100 per trade)
- Use stop-losses on every trade
- Monitor positions daily
- Keep detailed logs
- Diversify across assets
- Review performance weekly

### ❌ DON'T

- Risk more than 2% per trade
- Ignore daily loss limits
- Trade without stop-losses
- Increase size after losses
- Trade during high volatility
- Use money you can't afford to lose

## Performance Metrics

Track these metrics weekly:

| Metric | Target | Warning |
|--------|--------|---------|
| Win Rate | > 60% | < 50% |
| Avg Win/Loss Ratio | > 1.5 | < 1.0 |
| Max Drawdown | < 5% | > 10% |
| Daily Loss | < $200 | > $500 |
| Total Return | > 2%/month | < 0% |

### When to Stop Trading

🛑 **Stop immediately if:**
- Daily loss limit hit
- Max drawdown exceeded
- Exchange connection issues
- Unusual market conditions
- System bugs or errors

🛑 **Re-evaluate if:**
- Losing for 3+ days straight
- Win rate below 50%
- Strategy not performing as backtested
- Market regime change detected

## Troubleshooting

### "Insufficient Funds" Error

**Cause**: Not enough balance in quote currency

**Solution**:
```bash
# Check balance
python -c "from ccxt import binance; e = binance({'apiKey': '$EXCHANGE_API_KEY', 'secret': '$EXCHANGE_API_SECRET'}); print(e.fetch_balance())"
```

### Order Rejected

**Cause**: Exchange filters (min/max order size, price precision)

**Solution**:
- Check exchange trading rules
- Increase trade size to meet minimums
- Verify symbol is correct

### Connection Lost

**Cause**: Network issues or exchange downtime

**Solution**:
- Check internet connection
- Verify exchange status page
- Restart script when connection restored

## Security Best Practices

1. **API Keys**
   - Never share API keys
   - Use read-only keys for testing
   - Restrict IP addresses if possible
   - Rotate keys monthly

2. **Exchange Account**
   - Enable 2FA
   - Use withdrawal whitelist
   - Regular security audits
   - Separate email for trading

3. **System**
   - Keep software updated
   - Use firewall
   - Monitor logs for suspicious activity
   - Backup configuration files

## Next Steps

After successful live trading:

1. **Scale gradually**: Increase position sizes slowly
2. **Add more symbols**: Diversify across assets
3. **Advanced strategies**: Implement multi-strategy approach
4. **Automation**: Set up monitoring and alerts

See [STRATEGY_ENHANCEMENTS.md](STRATEGY_ENHANCEMENTS.md) for advanced strategies.
