# GoEmotions Live Trading Deployment Guide

**Status**: ✅ Ready for Deployment
**Date**: 2026-01-04
**Strategy**: GoEmotions + Technical Analysis (Multi-Factor)
**Exchange**: Kraken (MiCA licensed for Germany)

---

## ⚠️ CRITICAL WARNINGS

**THIS IS REAL MONEY TRADING**

- You will execute REAL trades with REAL money
- You can lose REAL funds if the strategy performs poorly
- Start with MINIMAL amounts (€300-500)
- Test thoroughly with paper trading first
- Monitor closely for the first week
- Understand that past performance doesn't guarantee future results

**By proceeding, you acknowledge:**
1. You have read and understand the risks
2. You are starting with conservative amounts (€300-500)
3. You will monitor trades closely
4. You accept full responsibility for any losses

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Completed 72-hour paper trading validation (optional but recommended)
- [ ] Reviewed all trading signals and performance
- [ ] Funded Kraken account with EUR (€500-1000 recommended)
- [ ] Generated Kraken API keys with NO withdrawal permissions
- [ ] Enabled IP whitelisting in Kraken settings
- [ ] Backed up your .env file

### Configuration
- [ ] .env configured with KRAKEN_API_KEY and KRAKEN_API_SECRET
- [ ] Reviewed config/germany_live_custom.yaml
- [ ] Set MAX_POSITION_EUR to 300-500 (conservative)
- [ ] Set MAX_DAILY_LOSS_EUR to 50-100
- [ ] Set MAX_DAILY_TRADES to 2-3

### Testing
- [ ] Tested Kraken API connection successfully
- [ ] Run test mode: `python scripts/live_trade_goemotions.py --test`
- [ ] Verified account balance is accessible
- [ ] Confirmed no withdrawal permissions on API key

### Deployment
- [ ] Built Docker image: `./manage_live_trading.sh build`
- [ ] Reviewed all safety parameters
- [ ] Understood emergency stop procedures
- [ ] Prepared monitoring setup

---

## 🚀 Deployment Steps

### Step 1: Verify Configuration

```bash
# Check .env has credentials
grep KRAKEN_API_KEY .env
grep KRAKEN_API_SECRET .env

# Test Kraken connection
python scripts/live_trade_goemotions.py --test
```

**Expected Output**:
```
✅ Credentials loaded
Fetching balance...
✅ Kraken API Connection Successful!
Available EUR: €XXX.XX
```

### Step 2: Review Safety Limits

Edit `.env` if needed:
```bash
nano .env
```

**Conservative Settings (Recommended)**:
```bash
MAX_POSITION_EUR=300       # Start small
MAX_DAILY_LOSS_EUR=50      # Stop losing after €50
MAX_DAILY_TRADES=2         # Max 2 trades per day
REQUIRE_CONFIRMATION=true  # Manual approval for trades
```

### Step 3: Build Docker Image

```bash
./manage_live_trading.sh build
```

**Expected Output**:
```
Building Docker image...
✓ Image built successfully
```

### Step 4: Test Run (Manual Mode)

```bash
# Single symbol, manual confirmation
python scripts/live_trade_goemotions.py \
  --symbols BTC/EUR \
  --max-position 300 \
  --test
```

### Step 5: Start Live Trading

**Option A: Multi-Symbol (Recommended for diversification)**
```bash
./manage_live_trading.sh start \
  --symbols "BTC/EUR ETH/EUR SOL/EUR" \
  --max-position 300
```

**Option B: Single Symbol (Conservative)**
```bash
python scripts/live_trade_goemotions.py \
  --symbols BTC/EUR \
  --max-position 300 \
  --max-daily-loss 50 \
  --interval 3600
```

**Expected Output**:
```
⚠️  WARNING: LIVE TRADING - REAL MONEY WILL BE USED
================================================================================
Symbols:         BTC/EUR, ETH/EUR, SOL/EUR
Strategy:        GoEmotions + Technical Analysis
Max Position:    €300.00
Max Daily Loss:  €50.00
Max Daily Trades: 2
Confirmation:    ON
================================================================================

Analyzing BTC/EUR...
  Price: €78,000.00
  RSI: 65.0
  24h: +1.2%
  Technical: HOLD
  Emotion: HOLD (neutral, intensity: 0.30)
  Final: HOLD (confidence: 50.00%)
  → No trade (confidence too low or HOLD signal)
```

---

## 📊 Monitoring

### Real-Time Monitoring

```bash
# View live logs
docker logs -f graphwiz-live-trading

# Or use management script
./manage_live_trading.sh logs
```

### Check Status

```bash
# Container status
./manage_live_trading.sh status

# Or docker directly
docker ps --filter name=graphwiz-live-trading
```

### Use Monitoring Dashboard

```bash
python monitor_live_trading.py --mode watch
```

---

## 🛑 Emergency Stop

### Immediate Stop (Recommended)

```bash
# Stop container gracefully
./manage_live_trading.sh stop

# Or kill immediately
docker stop graphwiz-live-trading
```

### Cancel All Orders

```bash
# SSH into container
./manage_live_trading.sh shell

# Inside container, run Python
python3 << 'PY'
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()
exchange = ccxt.kraken({
    'apiKey': os.getenv('KRAKEN_API_KEY'),
    'secret': os.getenv('KRAKEN_API_SECRET'),
})

# Cancel all open orders
exchange.cancel_all_orders('BTC/EUR')
print('All orders cancelled')
PY
```

---

## 📈 Performance Tracking

### Daily Review (First Week)

```bash
# Check today's trades
cat logs/live_trading/live_trading_*.csv | grep "$(date +%Y-%m-%d)"

# View performance
python monitor_live_trading.py --mode status
```

### Weekly Metrics to Track

1. **Total Return**: (Current Value - Starting Capital) / Starting Capital
2. **Win Rate**: Winning Trades / Total Trades
3. **Max Drawdown**: Largest peak-to-trough decline
4. **Sharpe Ratio**: Risk-adjusted returns
5. **Average Trade Size**: Average position size

### Success Criteria

**Week 1**:
- [ ] No emergency stops triggered
- [ ] Daily loss limits respected
- [ ] Max 2 trades per day
- [ ] Win rate ≥ 40%
- [ ] Max drawdown ≤ 10%

**Month 1**:
- [ ] Positive or break-even return
- [ ] Win rate ≥ 45%
- [ ] Max drawdown ≤ 15%
- [ ] No regulatory issues

---

## 🔧 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs graphwiz-live-trading

# Common issues:
# 1. Missing credentials → Verify .env file
# 2. Invalid API key → Check Kraken settings
# 3. Insufficient funds → Deposit EUR to Kraken
```

### No Trades Executing

```bash
# Check logs for signals
docker logs graphwiz-live-trading | grep "Final:"

# Possible reasons:
# 1. Confidence too low (< 70%)
# 2. RSI in neutral zone (30-70)
# 3. Emotion intensity not extreme enough
# 4. Daily trade limit reached
```

### Connection Errors

```bash
# Test connection manually
python scripts/live_trade_goemotions.py --test

# If failed:
# 1. Check internet connection
# 2. Verify Kraken API status
# 3. Check IP whitelist settings
# 4. Rotate API keys if needed
```

---

## 📁 Files Created

```
logs/live_trading/
├── live_trading_YYYYMMDD_HHMMSS.log      # Main log
├── live_trading_YYYYMMDD_HHMMSS.csv      # Trade history
└── equity_YYYYMMDD_HHMMSS.csv            # Equity curve

data/
└── live_trading_performance.json         # Performance metrics
```

---

## 🔄 Scaling Strategy

### Week 1-2: Conservative Phase
- Max position: €300
- Daily loss limit: €50
- Max daily trades: 2
- Symbols: 1-2 (BTC/EUR, ETH/EUR)
- Manual confirmation: ON

### Week 3-4: Growth Phase (if profitable)
- Max position: €500
- Daily loss limit: €75
- Max daily trades: 3
- Symbols: 3-4 (add SOL/EUR)
- Manual confirmation: ON

### Month 2+: Expansion Phase (if consistent profit)
- Max position: €1000
- Daily loss limit: €150
- Max daily trades: 4
- Symbols: 5-7 (multi-asset)
- Manual confirmation: Consider OFF (with caution)

---

## 📞 Support & Resources

### Documentation
- Paper Trading Guide: `PAPER_TRADING_DOCKER.md`
- Live Trading Plan: `/home/weiss/.claude/plans/curried-watching-pine.md`
- API Documentation: https://docs.kraken.com/rest/

### Monitoring Scripts
- `monitor_live_trading.py` - Real-time monitoring
- `./manage_live_trading.sh status` - Quick status check

### Important Reminders
1. **Start Small**: €300-500 initial capital
2. **Monitor Closely**: Check twice daily first week
3. **Use Stop Losses**: Set tight stop losses (1.5-2%)
4. **Keep Logs**: Review trade history regularly
5. **Stay Informed**: Monitor market conditions
6. **Don't Panic**: Markets fluctuate, stick to strategy
7. **Know When to Stop**: If losing 3+ days consecutively

---

## ✅ Deployment Complete Checklist

- [ ] API credentials verified
- [ ] Kraken connection tested
- [ ] Docker image built
- [ ] Safety limits configured
- [ ] First trade executed successfully
- [ ] Monitoring dashboard running
- [ ] Emergency procedures documented
- [ ] Daily review schedule set

---

## 📊 Expected Behavior

### What to Expect:

**First 24 Hours**:
- 0-4 trades (depending on market conditions)
- Mostly HOLD signals (confidence < 70%)
- System learning market patterns

**First Week**:
- 5-15 trades total
- Mostly on BTC/EUR and ETH/EUR
- Occasional trades on other symbols when oversold/overbought

**Market Conditions**:
- **Bull Market**: More SELL signals (take profits)
- **Bear Market**: More BUY signals (buy dips)
- **Sideways**: Mostly HOLD signals

### When System Trades:

**BUY Signals Triggered When**:
- RSI < 30 (oversold)
- Emotions show fear/anxiety (intensity > 75%)
- Both technical and emotion agree (confidence > 70%)

**SELL Signals Triggered When**:
- RSI > 70 (overbought)
- Emotions show euphoria/excitement (intensity > 75%)
- Both technical and emotion agree (confidence > 70%)

---

**Status**: ✅ READY FOR DEPLOYMENT

**Next Steps**:
1. Run `python scripts/live_trade_goemotions.py --test`
2. If successful, run `./manage_live_trading.sh start`
3. Monitor with `docker logs -f graphwiz-live-trading`
4. Review trades twice daily

**Good Luck! 🚀**
