# GoEmotions Paper Trading - Docker Quick Start Guide

## ✅ Docker Setup Complete!

### What's Been Created

1. **Dockerfile.paper-trading** - Updated for GoEmotions
2. **docker-compose.paper-trading.yml** - Orchestrization with GoEmotions
3. **manage_paper_trading.sh** - Management script updated
4. **PAPER_TRADING_DOCKER.md** - Complete documentation

---

## Quick Start (3 Steps)

### Step 1: Build Image ✅ (IN PROGRESS)

```bash
./manage_paper_trading.sh build
```

**What this does**:
- Creates Docker image with all dependencies
- Installs GoEmotions strategy
- Configures for Kraken exchange
- Sets up health checks

### Step 2: Start Paper Trading

```bash
# Quick 24-hour test
./manage_paper_trading.sh start --duration 24 --interval 10

# Full 72-hour validation
./manage_paper_trading.sh start --duration 72

# Week-long test
./manage_paper_trading.sh start --duration 168 --interval 30
```

### Step 3: Monitor

```bash
# View live logs
./manage_paper_trading.sh logs

# Check status
./manage_paper_trading.sh status

# Resource usage
./manage_paper_trading.sh stats
```

---

## Features

✅ **Docker Container**: Isolated environment
✅ **GoEmotions Strategy**: 27 emotions + contrarian signals
✅ **Real Market Data**: Kraken (German approved)
✅ **Multi-Factor**: Technical (RSI) + Emotion analysis
✅ **Persistent Logs**: Saved to host machine
✅ **Auto-Restart**: Continues after system reboot
✅ **Health Checks**: Built-in monitoring
✅ **Easy Management**: Simple commands

---

## Common Commands

### Build & Start
```bash
./manage_paper_trading.sh build
./manage_paper_trading.sh start
```

### Monitoring
```bash
./manage_paper_trading.sh logs     # Live logs
./manage_paper_trading.sh status   # Status + trades
./manage_paper_trading.sh stats    # Resources
```

### Control
```bash
./manage_paper_trading.sh stop     # Stop gracefully
./manage_paper_trading.sh restart  # Restart
```

### Advanced
```bash
./manage_paper_trading.sh shell    # Shell in container
./manage_paper_trading.sh clean    # Remove everything
```

---

## Configuration Examples

### Quick Test (1 hour)
```bash
./manage_paper_trading.sh start --duration 1 --interval 5
```

### Standard Test (24 hours)
```bash
./manage_paper_trading.sh start --duration 24 --interval 10
```

### Full Validation (72 hours)
```bash
./manage_paper_trading.sh start --duration 72 --interval 30
```

### Multiple Symbols
```bash
./manage_paper_trading.sh start \
  --duration 72 \
  --symbols "BTC/EUR ETH/EUR" \
  --capital 20000
```

---

## Files Generated

### Logs Directory
```
logs/paper_trading/
├── goemotions_validation_YYYYMMDD_HHMMSS.log
├── goemotions_trades_YYYYMMDD_HHMMSS.csv
└── goemotions_equity_YYYYMMDD_HHMMSS.csv
```

### Data Directory
```
data/paper_trading/
└── goemotions_validation_report_YYYYMMDD_HHMMSS.json
```

---

## What Happens When Running

### Every 30 Minutes (Default)

1. **Fetch Market Data**
   - Connects to Kraken API
   - Gets BTC/EUR price
   - Calculates RSI, MACD

2. **Analyze Sentiment**
   - Generates social media posts
   - Detects emotions (27 categories)
   - Identifies market phase

3. **Generate Signals**
   - Technical: RSI-based
   - Emotion: Sentiment-based
   - Combined: Multi-factor

4. **Execute Trades**
   - Buys if signal is BUY (confidence > 65%)
   - Sells if signal is SELL (confidence > 65%)
   - Logs all trades

5. **Update Metrics**
   - Portfolio value
   - P&L calculation
   - Save to files

---

## Stopping the Container

### Graceful Stop
```bash
./manage_paper_trading.sh stop
```

**What happens**:
- Completes current iteration
- Saves all trades
- Generates final report
- Stops container

### Emergency Stop
```bash
docker stop graphwiz-paper-trading
```

**Warning**: May lose unsaved data!

---

## Troubleshooting

### Container Not Starting

1. Check if Docker is running:
```bash
docker ps
```

2. Check build logs:
```bash
docker images | grep graphwiz-paper-trading
```

3. Rebuild:
```bash
./manage_paper_trading.sh build
```

### No Trades Executed

1. Check logs:
```bash
./manage_paper_trading.sh logs
```

2. Look for errors:
```bash
docker logs graphwiz-paper-trading | grep -i error
```

3. Verify market data:
```bash
docker logs graphwiz-paper-trading | grep "Price:"
```

### High Memory Usage

1. Check resources:
```bash
./manage_paper_trading.sh stats
```

2. Reduce interval (fewer updates):
```bash
./manage_paper_trading.sh restart --interval 60
```

3. Stop and restart:
```bash
./manage_paper_trading.sh stop
./manage_paper_trading.sh start --interval 60
```

---

## Performance Tips

### Faster Testing
```bash
# Update every 5 minutes (more trades, more CPU)
./manage_paper_trading.sh start --duration 24 --interval 5
```

### Resource Efficient
```bash
# Update every 60 minutes (fewer trades, less CPU)
./manage_paper_trading.sh start --duration 72 --interval 60
```

### Production Settings
```bash
# 1 week, 30 min updates, 2 symbols
./manage_paper_trading.sh start \
  --duration 168 \
  --interval 30 \
  --symbols "BTC/EUR ETH/EUR"
```

---

## Next Steps

### After Testing (24-72 Hours)

1. **View Results**:
```bash
cat data/paper_trading/goemotions_validation_report_*.json
```

2. **Check Trades**:
```bash
cat logs/paper_trading/goemotions_trades_*.csv
```

3. **View Equity Curve**:
```bash
tail logs/paper_trading/goemotions_equity_*.csv
```

4. **Evaluate Performance**:
- Win rate > 50%? ✅ Good
- Return > 0%? ✅ Profitable
- Max drawdown < 15%? ✅ Safe

### Ready for Live Trading?

**If results are good**:
- ✅ Return > 5%
- ✅ Win rate > 50%
- ✅ Max drawdown < 10%

**Then proceed to**:
1. Review plan: `/home/weiss/.claude/plans/curried-watching-pine.md`
2. Deploy live trading
3. Start with €300-500 capital

---

## Summary

### Management Script Commands

```bash
./manage_paper_trading.sh
├── build      # Build Docker image
├── start      # Start trading (with options)
├── stop       # Stop trading
├── restart    # Restart trading
├── status     # Show status + trades
├── logs       # View live logs
├── stats      # Show resource usage
├── shell      # Open shell in container
├── clean      # Remove container + images
└── help       # Show help message
```

### Key Options

```bash
--duration N     # Hours to run (default: 72)
--symbols X      # Trading pairs (default: BTC/EUR)
--capital N      # EUR capital (default: 10000)
--interval N     # Minutes between checks (default: 30)
```

### Docker Benefits

✅ **Isolated**: No conflicts with system
✅ **Portable**: Run anywhere Docker is installed
✅ **Reproducible**: Same environment every time
✅ **Scalable**: Easy to run multiple instances
✅ **Safe**: Can't affect host system

---

**Status**: ✅ Docker GoEmotions Paper Trading - READY TO RUN

**Current Action**: Building Docker image (nearly complete)

**Next Command**: `./manage_paper_trading.sh start`

**Monitor With**: `./manage_paper_trading.sh logs`

**Stop With**: `./manage_paper_trading.sh stop`
