# Live Trading Quick Start Guide

**Get started in 5 minutes** ⚡

---

## 🎯 Prerequisites

✅ Kraken account funded with EUR (€500+ recommended)
✅ Kraken API keys generated (NO withdrawal permissions!)
✅ Paper trading completed (recommended)
✅ .env file configured with credentials

---

## ⚡ Quick Start (3 Steps)

### Step 1: Test Connection (30 seconds)

```bash
python scripts/live_trade_goemotions.py --test
```

**Expected**: `✅ Kraken API Connection Successful!`

### Step 2: Build Docker Image (2 minutes)

```bash
./manage_live_trading.sh build
```

**Expected**: `✓ Image built successfully`

### Step 3: Start Live Trading (2 minutes)

```bash
# Conservative start - BTC only, €300 max position
python scripts/live_trade_goemotions.py \
  --symbols BTC/EUR \
  --max-position 300 \
  --max-daily-loss 50 \
  --interval 3600
```

**First Trade**: You'll be prompted to confirm:
```
⚠️  Execute BUY order? (yes/no): 
```

Type `yes` to execute, `no` to skip.

---

## 📊 Monitor Your Trades

### View Live Logs
```bash
# In another terminal
docker logs -f graphwiz-live-trading
```

### Check Status Anytime
```bash
./manage_live_trading.sh status
```

### Stop Trading
```bash
# Graceful stop
./manage_live_trading.sh stop

# Emergency stop
docker stop graphwiz-live-trading
```

---

## 🎛️ Configuration Examples

### Ultra-Conservative (Recommended)
```bash
python scripts/live_trade_goemotions.py \
  --symbols BTC/EUR \
  --max-position 300 \
  --max-daily-loss 50 \
  --max-daily-trades 2 \
  --interval 3600
```

### Moderate (After 1 week of profit)
```bash
python scripts/live_trade_goemotions.py \
  --symbols "BTC/EUR ETH/EUR" \
  --max-position 500 \
  --max-daily-loss 75 \
  --max-daily-trades 3 \
  --interval 3600
```

### Aggressive (NOT recommended - only after 1 month profit)
```bash
python scripts/live_trade_goemotions.py \
  --symbols "BTC/EUR ETH/EUR SOL/EUR XRP/EUR" \
  --max-position 1000 \
  --max-daily-loss 150 \
  --max-daily-trades 4 \
  --interval 1800
```

---

## 🔍 What to Expect

### First Hour
- System analyzes market every 60 minutes
- Mostly HOLD signals (confidence too low)
- 0-1 trades likely

### First Day
- 1-3 trades maximum (if conditions right)
- Manual confirmation required for each
- System learns market patterns

### First Week
- 5-15 trades total
- Monitor closely
- Review logs daily

---

## 🛑 Emergency Commands

### Stop Everything
```bash
docker stop graphwiz-live-trading
```

### Check What's Happening
```bash
docker logs --tail 100 graphwiz-live-trading
```

### View All Trades
```bash
cat logs/live_trading/live_trading_*.csv
```

---

## ⚠️ Important Safety Rules

1. **START SMALL** - €300 max position to begin
2. **USE CONFIRMATION** - Keep manual prompts ON
3. **MONITOR DAILY** - Check logs at least twice daily
4. **SET LIMITS** - Never exceed max daily loss
5. **STOP IF LOSING** - Pause after 3 consecutive losing days
6. **KEEP RECORDS** - Save all trade logs

---

## 📈 Success Indicators

✅ **Good Signs**:
- Positive or break-even after week 1
- Win rate ≥ 40%
- Max drawdown ≤ 10%
- No emergency stops

⚠️ **Warning Signs**:
- 3+ consecutive losing days
- Daily loss limits hit frequently
- Win rate < 30%
- Large drawdowns (>15%)

🛑 **Stop Trading If**:
- You're uncomfortable with the risk
- You don't understand a trade
- Technical issues occur
- Market conditions are extreme

---

## 📞 Need Help?

### Documentation
- Full Guide: `LIVE_TRADING_DEPLOYMENT.md`
- Paper Trading: `PAPER_TRADING_DOCKER.md`
- Plan: `/home/weiss/.claude/plans/curried-watching-pine.md`

### Scripts
- Test Connection: `python scripts/live_trade_goemotions.py --test`
- Monitor: `python monitor_live_trading.py --mode watch`
- Status: `./manage_live_trading.sh status`

### Troubleshooting
```bash
# Check logs
docker logs graphwiz-live-trading

# Restart container
./manage_live_trading.sh restart

# Check API keys
grep KRAKEN .env
```

---

## ✅ Ready to Start?

1. ✅ Tested connection? `python scripts/live_trade_goemotions.py --test`
2. ✅ Built image? `./manage_live_trading.sh build`
3. ✅ Reviewed safety limits? Check .env file
4. ✅ Understood the risks? Read warnings above
5. ✅ Prepared to monitor? Have logs open in another terminal

**If all YES, run:**
```bash
python scripts/live_trade_goemotions.py --symbols BTC/EUR --max-position 300
```

**Good luck! 🚀**

Remember: Start small, monitor closely, and stay safe!
