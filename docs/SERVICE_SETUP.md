# Paper Trading Service Setup Guide

Complete guide for running paper trading as a background service.

## Quick Start

### 1. Start the Service

```bash
# Start all configured symbols
python scripts/paper_trading_service.py start

# Check status
python scripts/paper_trading_service.py status
```

### 2. View Logs

```bash
# View all logs
python scripts/paper_trading_service.py logs

# View specific symbol logs
python scripts/paper_trading_service.py logs --symbol BTC/USDT

# View last 100 lines
python scripts/paper_trading_service.py logs --tail 100
```

### 3. Stop the Service

```bash
python scripts/paper_trading_service.py stop
```

## Configuration

### Add New Symbols

```bash
# Add DOGE/USDT with custom settings
python scripts/paper_trading_service.py add DOGE/USDT \
    --capital 5000 \
    --oversold 20 \
    --overbought 80 \
    --restart
```

### Remove Symbols

```bash
python scripts/paper_trading_service.py remove DOGE/USDT --restart
```

### Enable/Disable Symbols

```bash
# Disable a symbol (keep config but don't run)
python scripts/paper_trading_service.py disable SOL/USDT

# Re-enable
python scripts/paper_trading_service.py enable SOL/USDT
```

## Advanced Setup

### Systemd Service (Linux)

1. **Edit service file**:
```bash
nano graphwiz-paper-trading.service
```

2. **Update paths**:
```ini
User=your_username
WorkingDirectory=/path/to/graphwiz-trader
Environment="PATH=/path/to/graphwiz-trader/venv/bin"
ExecStart=/path/to/graphwiz-trader/venv/bin/python scripts/paper_trading_service.py start
ExecStop=/path/to/graphwiz-trader/venv/bin/python scripts/paper_trading_service.py stop
```

3. **Install service**:
```bash
# Copy to systemd
sudo cp graphwiz-paper-trading.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable graphwiz-paper-trading

# Start service
sudo systemctl start graphwiz-paper-trading

# Check status
sudo systemctl status graphwiz-paper-trading
```

4. **Control service**:
```bash
# Start
sudo systemctl start graphwiz-paper-trading

# Stop
sudo systemctl stop graphwiz-paper-trading

# Restart
sudo systemctl restart graphwiz-paper-trading

# View logs
sudo journalctl -u graphwiz-paper-trading -f
```

### Docker Service (Alternative)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  paper-trading:
    build: .
    command: python scripts/paper_trading_service.py start
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Service Management

### Status Commands

```bash
# Check if running
python scripts/paper_trading_service.py status

# View all logs
python scripts/paper_trading_service.py logs

# View specific symbol
python scripts/paper_trading_service.py logs --symbol BTC/USDT
```

### Control Commands

```bash
# Start
python scripts/paper_trading_service.py start

# Stop
python scripts/paper_trading_service.py stop

# Restart
python scripts/paper_trading_service.py restart
```

## Configuration File

Configuration is stored in `config/paper_trading.json`:

```json
{
  "BTC/USDT": {
    "symbol": "BTC/USDT",
    "capital": 10000,
    "oversold": 25,
    "overbought": 65,
    "interval": 3600,
    "enabled": true
  },
  "ETH/USDT": {
    "symbol": "ETH/USDT",
    "capital": 10000,
    "oversold": 25,
    "overbought": 65,
    "interval": 3600,
    "enabled": true
  }
}
```

## Log Files

Logs are stored in `logs/` directory:

```
logs/
├── BTC_USDT.log        # BTC/USDT paper trading logs
├── ETH_USDT.log        # ETH/USDT paper trading logs
├── SOL_USDT.log        # SOL/USDT paper trading logs
└── service.log         # Service manager logs
```

## Monitoring

### Check Active Processes

```bash
# Find all paper trading processes
ps aux | grep paper_trade

# Check with service tool
python scripts/paper_trading_service.py status
```

### View Results

Results are saved to `data/paper_trading/`:

```bash
# List all result files
ls -la data/paper_trading/

# View latest results
cat data/paper_trading/BTC_USDT_summary_*.json | python -m json.tool

# View equity curve
cat data/paper_trading/BTC_USDT_equity_*.csv
```

## Troubleshooting

### Service Won't Start

1. Check configuration:
```bash
cat config/paper_trading.json
```

2. Check logs:
```bash
python scripts/paper_trading_service.py logs
```

3. Verify dependencies:
```bash
pip install -r requirements.txt
```

### High Memory Usage

Reduce number of symbols or increase interval:

```bash
# Disable some symbols
python scripts/paper_trading_service.py disable SOL/USDT
python scripts/paper_trading_service.py disable DOGE/USDT

# Increase check interval (edit config file)
# interval: 7200  # Check every 2 hours instead of 1
```

### Stale Processes

Kill all paper trading processes:

```bash
pkill -f paper_trade.py

# Or use service stop
python scripts/paper_trading_service.py stop
```

## Best Practices

1. **Start Small**: Begin with 2-3 symbols
2. **Monitor Regularly**: Check status and logs daily
3. **Review Results**: Analyze equity curves weekly
4. **Adjust Parameters**: Optimize based on performance
5. **Use Systemd**: For production, use systemd service
6. **Log Rotation**: Set up logrotate for log files

## Production Deployment

For production use:

1. **Use systemd service** (see above)
2. **Set up log rotation**:
```bash
sudo nano /etc/logrotate.d/graphwiz-paper-trading
```

3. **Monitor with alerts**:
```bash
# Add cron job to check status
0 */6 * * * /path/to/graphwiz-trader/scripts/paper_trading_service.py status
```

4. **Backup results**:
```bash
# Add to cron
0 2 * * * cp -r data/paper_trading backups/paper_trading_$(date +\%Y\%m\%d)
```
