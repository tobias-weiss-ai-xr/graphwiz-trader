# Extended Paper Trading Validation - Quick Start Guide

## Overview

This guide will help you run a comprehensive 24-72 hour paper trading validation to test GraphWiz Trader thoroughly before live trading.

---

## ✅ Prerequisites Check

### 1. Dependencies

```bash
# Check if required packages are installed
pip install loguru ccxt pandas numpy

# Or install all project dependencies
pip install -r requirements.txt
```

### 2. Neo4j Database

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Should show Neo4j container running
# If not, start it:
docker-compose up -d neo4j
```

### 3. Create Required Directories

```bash
# Directories are created automatically, but you can verify:
mkdir -p logs/paper_trading
mkdir -p data/paper_trading
```

---

## 🚀 Quick Start (3 Options)

### Option 1: Quick 24-Hour Test

**Best for:** Initial validation, testing the system

```bash
# Run for 24 hours with default settings
python run_extended_paper_trading.py --duration 24
```

### Option 2: Recommended 72-Hour Validation

**Best for:** Comprehensive testing before live trading

```bash
# Run for 72 hours (recommended)
python run_extended_paper_trading.py --duration 72
```

### Option 3: Custom Configuration

**Best for:** Testing specific strategies or symbols

```bash
# Custom duration (48 hours)
python run_extended_paper_trading.py --duration 48

# Custom symbols
python run_extended_paper_trading.py --duration 72 \
  --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT

# Custom capital ($50,000)
python run_extended_paper_trading.py --duration 72 --capital 50000

# Custom update interval (every 15 minutes)
python run_extended_paper_trading.py --duration 72 --interval 15
```

---

## 🔄 Running in Background

### Using nohup (Recommended)

```bash
# Start in background
nohup python run_extended_paper_trading.py --duration 72 > paper_trading.log 2>&1 &

# Save the PID
echo $! > paper_trading.pid

# Check if it's running
ps -p $(cat paper_trading.pid)

# Stop it
kill $(cat paper_trading.pid)
```

### Using screen

```bash
# Start screen session
screen -S paper_trading

# Run validation
python run_extended_paper_trading.py --duration 72

# Detach: Ctrl+A, then D
# Reattach: screen -r paper_trading
```

### Using tmux

```bash
# Start tmux session
tmux new -s paper_trading

# Run validation
python run_extended_paper_trading.py --duration 72

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t paper_trading
```

---

## 📊 Monitoring the Validation

### Real-Time Monitoring

```bash
# In another terminal, monitor the logs
tail -f logs/paper_trading/validation_*.log

# Or use the monitoring script
python monitor_paper_trading.py

# Show latest report
python monitor_paper_trading.py --report

# Show last 50 log lines
python monitor_paper_trading.py --tail 50

# Show equity curve
python monitor_paper_trading.py --equity
```

### Check Status

```bash
# View latest validation report
cat data/paper_trading/validation_report_*.json | jq .

# Or use Python
python -c "import json; print(json.dumps(json.load(open('data/paper_trading/validation_report_*.json')), indent=2))"
```

### View Trade History

```bash
# View trades CSV
cat logs/paper_trading/trades_*.csv | column -t -s,

# Or open in spreadsheet software
libreoffice logs/paper_trading/trades_*.csv
```

---

## 📈 What to Expect

### During Validation

The system will:

1. **Every 30 minutes** (configurable):
   - Fetch market data for each symbol
   - Calculate technical indicators (RSI)
   - Generate trading signals (BUY/SELL/HOLD)
   - Execute virtual trades if signals are strong
   - Update portfolio value

2. **Log continuously**:
   - Trade executions with reasons
   - Portfolio performance
   - Win/loss tracking
   - Return and drawdown metrics

3. **Generate reports**:
   - Real-time status updates
   - Equity curve logging
   - Final validation report

### Typical Results

**For a 72-hour validation:**

- **Trades**: 10-50 trades (depending on market volatility)
- **Expected Return**: -5% to +15% (varies by market conditions)
- **Target Metrics**:
  - ✅ **Win Rate** > 50%
  - ✅ **Max Drawdown** < 20%
  - ✅ **Sharpe Ratio** > 1.0
  - ✅ **Positive Return** after 72 hours

---

## 📋 Validation Checklist

Use this checklist to determine if ready for live trading:

### ✅ Technical Validation

- [ ] System ran for full duration (24-72 hours)
- [ ] No crashes or errors
- [ ] All trades logged correctly
- [ ] Neo4j integration working
- [ ] Portfolio tracking accurate

### ✅ Performance Validation

- [ ] **Total Return**: Positive (or minimal loss < 2%)
- [ ] **Win Rate**: > 50%
- [ ] **Max Drawdown**: < 20%
- [ ] **Sharpe Ratio**: > 1.0
- [ ] **Profit Factor**: > 1.5

### ✅ Risk Validation

- [ ] Daily loss limits respected
- [ ] Position sizes appropriate
- [ ] No over-leveraging
- [ ] Stop losses effective

### ✅ Operational Validation

- [ ] Logs comprehensive and readable
- [ ] Reports generated correctly
- [ ] Can stop and restart safely
- [ ] Monitoring works as expected

---

## 🎯 Interpreting Results

### Status Messages

At the end of validation, you'll see one of these statuses:

#### ✅ **EXCELLENT - Ready for live trading**

**Criteria:**
- Total Return > 5%
- Max Drawdown < 10%

**Action:** Consider starting live trading with small capital

#### ✅ **GOOD - Consider live trading with caution**

**Criteria:**
- Total Return > 0%
- Max Drawdown < 15%

**Action:** Run another 72-hour validation or proceed with very small capital

#### ⚠️ **MODERATE - Continue validation**

**Criteria:**
- Mixed performance
- Some concerning metrics

**Action:**
- Run for another 72 hours
- Adjust strategy parameters
- Review losing trades

#### ❌ **POOR - Not ready for live trading**

**Criteria:**
- Total Return < 0%
- High drawdown > 20%

**Action:**
- Do NOT proceed to live trading
- Review and adjust strategy
- Fix any bugs or issues
- Re-run validation

---

## 🔧 Troubleshooting

### No Trades Executed

**Problem:** RSI not triggering buy/sell signals

**Solution:**
```bash
# Lower RSI thresholds to be more sensitive
# Edit the script and change these values:
# if rsi < 30:  # Change to 35 or 40
# if rsi > 70:  # Change to 65 or 60

# Or run during more volatile market periods
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'loguru'`

**Solution:**
```bash
# Install dependencies
pip install loguru ccxt pandas numpy

# Or install all requirements
pip install -r requirements.txt
```

### Neo4j Connection Issues

**Problem:** Can't connect to Neo4j

**Solution:**
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Start Neo4j
docker-compose up -d neo4j

# Verify connection
curl http://localhost:7474
```

### Validation Stops Unexpectedly

**Problem:** Script crashes or stops

**Solution:**
```bash
# Check the error log
tail -100 logs/paper_trading/validation_*.log

# Run in foreground first to see errors
python run_extended_paper_trading.py --duration 1
```

---

## 📁 Files Generated

After running validation, you'll have:

```
logs/paper_trading/
├── validation_20260102_120000.log      # Main log
├── trades_20260102_120000.csv          # All trades
└── equity_20260102_120000.csv          # Portfolio value over time

data/paper_trading/
└── validation_report_20260102_120000.json  # Final report
```

---

## 🎓 Next Steps

### After Successful Validation

1. **Review Results**
   ```bash
   python monitor_paper_trading.py --report
   ```

2. **Analyze Trades**
   ```bash
   # Open trades CSV in spreadsheet
   libreoffice logs/paper_trading/trades_*.csv
   ```

3. **Decide on Live Trading**
   - If EXCELLENT: Proceed to live trading
   - If GOOD: Consider or run more validation
   - If MODERATE/POOR: Adjust and re-run

4. **Gradual Transition**
   - Start with 10% of target capital
   - Monitor closely for 1 week
   - Increase to 25% if performing well
   - Continue gradual increase to 100%

---

## 📞 Support

### Getting Help

1. **Check Logs First**
   ```bash
   tail -100 logs/paper_trading/validation_*.log
   ```

2. **Review Configuration**
   - Check `config/paper_trading.yaml`
   - Verify Neo4j connection settings

3. **Documentation**
   - `docs/PAPER_TRADING.md` - Comprehensive guide
   - `TRADING_MODES_README.md` - Architecture details

---

## 🎉 Summary

**You're now ready to run extended paper trading validation!**

**Recommended Command:**
```bash
# Run 72-hour validation in background
nohup python run_extended_paper_trading.py --duration 72 > paper_trading.log 2>&1 &

# Monitor progress
tail -f logs/paper_trading/validation_*.log
```

**Key Metrics to Watch:**
- ✅ Total Return (target: > 5%)
- ✅ Win Rate (target: > 50%)
- ✅ Max Drawdown (target: < 20%)
- ✅ Sharpe Ratio (target: > 1.0)

**After Validation:**
```bash
# View final report
python monitor_paper_trading.py --report
```

Good luck with your validation! 🚀
