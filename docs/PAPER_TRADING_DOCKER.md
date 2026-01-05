# GoEmotions Paper Trading - Docker Guide

## Quick Start

### 1. Build the Docker Image

```bash
./manage_paper_trading.sh build
```

### 2. Start Paper Trading (Default: 72 hours)

```bash
./manage_paper_trading.sh start
```

### 3. View Live Logs

```bash
./manage_paper_trading.sh logs
```

### 4. Check Status

```bash
./manage_paper_trading.sh status
```

### 5. Stop Trading

```bash
./manage_paper_trading.sh stop
```

---

## Management Commands

### build
Build the GoEmotions paper trading Docker image.

```bash
./manage_paper_trading.sh build
```

**Output**: Creates `graphwiz-paper-trading:latest` image

---

### start
Start the paper trading service.

**Usage**:
```bash
./manage_paper_trading.sh start [options]
```

**Options**:
- `--duration` <hours> - Trading duration (default: 72)
- `--symbols` <pairs> - Trading pairs (default: BTC/EUR)
- `--capital` <euros> - Initial capital (default: 10000)
- `--interval` <minutes> - Update interval (default: 30)

**Examples**:
```bash
# Quick 24-hour test
./manage_paper_trading.sh start --duration 24 --interval 10

# Full week validation
./manage_paper_trading.sh start --duration 168 --interval 30

# Multiple symbols
./manage_paper_trading.sh start --duration 72 --symbols "BTC/EUR ETH/EUR"

# Custom capital
./manage_paper_trading.sh start --capital 50000 --duration 24
```

---

### stop
Stop the paper trading service gracefully.

```bash
./manage_paper_trading.sh stop
```

**What happens**:
- Container stops after completing current iteration
- All trades saved to logs
- Final report generated

---

### restart
Restart the paper trading service.

```bash
./manage_paper_trading.sh restart [options]
```

**Accepts same options as `start`**

---

### status
Show service status and recent activity.

```bash
./manage_paper_trading.sh status
```

**Output includes**:
- Container running status
- Resource usage (CPU, memory)
- Recent trades (last 10)
- Current portfolio value
- Performance metrics

**Example output**:
```
✓ Container is RUNNING

NAMES                      STATUS          PORTS
graphwiz-paper-trading    Up 2 hours      -

Resource Usage:
NAME                      CPU %     MEM USAGE / LIMIT
graphwiz-paper-trading    2.5%     512MiB / 2GiB

Recent Activity:
✅ BUY BTC/EUR: 0.0123 @ €75,000 (€922.50)
📊 Portfolio: €10,250.00 (+2.5%)
```

---

### logs
View live logs from the container.

```bash
./manage_paper_trading.sh logs
```

**Features**:
- Follow mode (tail -f)
- Color-coded output
- Filters out internal logs

**Press Ctrl+C to exit** (container keeps running)

---

### stats
Show detailed resource usage statistics.

```bash
./manage_paper_trading.sh stats
```

**Output includes**:
- CPU percentage
- Memory usage
- Network I/O
- Block I/O

---

### shell
Open a bash shell inside the container.

```bash
./manage_paper_trading.sh shell
```

**Useful for**:
- Debugging
- Inspecting logs
- Running custom commands
- Checking files

**Example**:
```bash
./manage_paper_trading.sh shell

# Inside container:
root@container:/app# ls -la logs/paper_trading/
root@container:/app# cat logs/paper_trading/goemotions_trades_*.csv
root@container:/app# python --version
root@container:/app# exit
```

---

### clean
Remove container and images.

```bash
./manage_paper_trading.sh clean
```

**Warning**: This will:
- Stop the container
- Remove the container
- Remove the Docker image
- Delete volumes (with -v flag)

**Use with caution!**

---

## Configuration

### Environment Variables

Edit `docker-compose.paper-trading.yml`:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
  - TZ=Europe/Berlin            # Your timezone
  - DURATION=72                  # Default duration in hours
  - SYMBOLS=BTC/EUR             # Default trading pairs
  - CAPITAL=10000               # Default capital in EUR
  - INTERVAL=30                 # Default interval in minutes
```

### Resource Limits

Edit `docker-compose.paper-trading.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'          # Max CPU cores
      memory: 2G         # Max memory
    reservations:
      cpus: '0.5'        # Min CPU cores
      memory: 512M       # Min memory
```

### Logging Configuration

Edit `docker-compose.paper-trading.yml`:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"      # Max log file size
    max-file: "3"        # Number of log files to keep
```

---

## Volumes and Persistence

### Logs Directory

**Host**: `./logs/paper_trading/`
**Container**: `/app/logs/paper_trading/`

**Files created**:
```
logs/paper_trading/
├── goemotions_validation_YYYYMMDD_HHMMSS.log    # Main log
├── goemotions_trades_YYYYMMDD_HHMMSS.csv       # Trade history
└── goemotions_equity_YYYYMMDD_HHMMSS.csv       # Equity curve
```

### Data Directory

**Host**: `./data/paper_trading/`
**Container**: `/app/data/paper_trading/`

**Files created**:
```
data/paper_trading/
└── goemotions_validation_report_YYYYMMDD_HHMMSS.json  # Final report
```

### Accessing Logs

**From host**:
```bash
# View latest log
tail -f logs/paper_trading/goemotions_validation_*.log

# View trades
cat logs/paper_trading/goemotions_trades_*.csv

# View equity
tail logs/paper_trading/goemotions_equity_*.csv
```

**From container**:
```bash
./manage_paper_trading.sh shell
cd /app/logs/paper_trading
ls -la
```

---

## Running in Background

### Start and Detach

```bash
# Start in background
./manage_paper_trading.sh start

# Container runs in background automatically
# No need for nohup or screen
```

### Monitor Background Process

```bash
# Check status
./manage_paper_trading.sh status

# View logs
./manage_paper_trading.sh logs

# Resource usage
./manage_paper_trading.sh stats
```

### Auto-Restart

Container is configured with:
```yaml
restart: unless-stopped
```

**Behavior**:
- Automatically restarts on crash
- Automatically restarts on system reboot
- Does NOT restart if manually stopped

---

## Docker Compose Direct Usage

You can also use docker-compose directly:

```bash
# Build
docker-compose -f docker-compose.paper-trading.yml build

# Start (with custom env)
DURATION=24 SYMBOLS="BTC/EUR" INTERVAL=10 \
  docker-compose -f docker-compose.paper-trading.yml up -d

# View logs
docker-compose -f docker-compose.paper-trading.yml logs -f

# Stop
docker-compose -f docker-compose.paper-trading.yml down

# Restart
docker-compose -f docker-compose.paper-trading.yml restart
```

---

## Troubleshooting

### Container Won't Start

**Problem**: Container exits immediately

**Solutions**:
1. Check logs: `docker logs graphwiz-paper-trading`
2. Check if port is already in use: `docker ps -a`
3. Verify Docker image: `docker images | grep graphwiz-paper-trading`
4. Rebuild image: `./manage_paper_trading.sh build`

### No Trades Executed

**Problem**: Container running but no trades

**Solutions**:
1. Check logs: `./manage_paper_trading.sh logs`
2. Verify market data fetching: Look for "Price:" in logs
3. Lower confidence threshold in code
4. Check if Kraken API is accessible

### High Memory Usage

**Problem**: Container using too much memory

**Solutions**:
1. Check stats: `./manage_paper_trading.sh stats`
2. Reduce memory limit in docker-compose.yml
3. Increase update interval (less frequent polling)
4. Restart container: `./manage_paper_trading.sh restart`

### Logs Not Appearing

**Problem**: Can't see log files

**Solutions**:
1. Check volume mounting: `docker inspect graphwiz-paper-trading | grep Mounts`
2. Verify directory permissions: `ls -la logs/paper_trading/`
3. Check inside container: `./manage_paper_trading.sh shell` then `ls /app/logs/paper_trading/`

### Container Crashes Repeatedly

**Problem**: Container keeps restarting

**Solutions**:
1. Check crash reason: `docker logs graphwiz-paper-trading --tail 100`
2. Check health status: `docker inspect graphwiz-paper-trading | grep Health`
3. Verify Python code syntax
4. Test without Docker: `python run_extended_paper_trading_goemotions.py --duration 1`

---

## Performance Tuning

### Update Interval

**Faster (more trades, more CPU)**:
```bash
./manage_paper_trading.sh start --interval 5
```

**Slower (fewer trades, less CPU)**:
```bash
./manage_paper_trading.sh start --interval 60
```

**Recommended**:
- Testing: 5-10 minutes
- Production: 30-60 minutes

### Resource Allocation

**For heavy testing** (multiple symbols):
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

**For minimal usage**:
```yaml
deploy:
  resources:
    limits:
      cpus: '1'
      memory: 1G
```

### Logging Level

**Less verbose** (production):
```yaml
environment:
  - LOG_LEVEL=WARNING
```

**More verbose** (debugging):
```yaml
environment:
  - LOG_LEVEL=DEBUG
```

---

## Multi-Container Setups

### Run Multiple Instances

**Different symbols**:
```bash
# Terminal 1: BTC/EUR
DURATION=72 SYMBOLS="BTC/EUR" \
  docker-compose -f docker-compose.paper-trading.yml up -d

# Terminal 2: ETH/EUR
DURATION=72 SYMBOLS="ETH/EUR" CONTAINER_NAME=graphwiz-paper-trading-eth \
  docker-compose -f docker-compose.paper-trading.yml up -d
```

**Different configurations**:
```bash
# Instance 1: Aggressive (5 min interval)
./manage_paper_trading.sh start --interval 5

# Instance 2: Conservative (60 min interval)
# (Need to create separate compose file for this)
```

---

## Monitoring and Alerts

### Health Checks

Container has built-in health check:
```yaml
HEALTHCHECK --interval=30s --timeout=10s --retries=3
```

**Check health**:
```bash
docker inspect graphwiz-paper-trading | grep -A 5 Health
```

### Log Monitoring

**Watch for errors**:
```bash
./manage_paper_trading.sh logs | grep -i error
```

**Watch for trades**:
```bash
./manage_paper_trading.sh logs | grep "Trade #"
```

**Watch for signals**:
```bash
./manage_paper_trading.sh logs | grep "Final:"
```

### Performance Metrics

**Export metrics** (from final report):
```bash
# After validation completes
cat data/paper_trading/goemotions_validation_report_*.json | jq .
```

---

## Backup and Restore

### Backup Logs

```bash
# Create backup
tar -czf paper_trading_backup_$(date +%Y%m%d).tar.gz \
    logs/paper_trading/ \
    data/paper_trading/

# List backups
ls -lh paper_trading_backup_*.tar.gz
```

### Restore Logs

```bash
# Extract backup
tar -xzf paper_trading_backup_20260104.tar.gz

# Verify
ls logs/paper_trading/
```

---

## Updating

### Update Code

1. Stop container:
```bash
./manage_paper_trading.sh stop
```

2. Pull latest code (if using git):
```bash
git pull
```

3. Rebuild image:
```bash
./manage_paper_trading.sh build
```

4. Start container:
```bash
./manage_paper_trading.sh start
```

### Rollback

If new version has issues:

1. Checkout previous version:
```bash
git log --oneline
git checkout <previous-commit>
```

2. Rebuild:
```bash
./manage_paper_trading.sh build
```

3. Start:
```bash
./manage_paper_trading.sh start
```

---

## Production Deployment

### Recommended Settings

```yaml
# docker-compose.paper-trading.yml
environment:
  - LOG_LEVEL=INFO
  - TZ=Europe/Berlin
  - DURATION=168          # 1 week
  - SYMBOLS=BTC/EUR ETH/EUR
  - CAPITAL=10000
  - INTERVAL=30          # 30 minutes

deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '0.5'
      memory: 512M

restart: unless-stopped
```

### Pre-Deployment Checklist

- [ ] Tested locally for 24 hours
- [ ] Reviewed configuration
- [ ] Set up log rotation
- [ ] Configured resource limits
- [ ] Tested monitoring (`status`, `logs` commands)
- [ ] Prepared backup strategy
- [ ] Documented custom settings

### Post-Deployment

1. Monitor initial logs: `./manage_paper_trading.sh logs`
2. Check status after 1 hour: `./manage_paper_trading.sh status`
3. Verify trades being executed
4. Monitor resource usage: `./manage_paper_trading.sh stats`

---

## Summary

### Quick Reference

```bash
# Build and start
./manage_paper_trading.sh build
./manage_paper_trading.sh start

# Monitor
./manage_paper_trading.sh logs
./manage_paper_trading.sh status
./manage_paper_trading.sh stats

# Stop
./manage_paper_trading.sh stop

# Custom run
./manage_paper_trading.sh start --duration 24 --symbols "BTC/EUR ETH/EUR" --capital 50000

# Clean
./manage_paper_trading.sh clean
```

### Key Features

✅ **Isolated Environment**: Docker container for clean execution
✅ **Easy Management**: Simple commands for all operations
✅ **Persistent Storage**: Logs and data saved to host
✅ **Resource Limits**: CPU and memory constrained
✅ **Auto-Restart**: Container restarts on failure
✅ **Health Checks**: Built-in health monitoring
✅ **GoEmotions Strategy**: Real-time sentiment analysis
✅ **Kraken Integration**: German-approved exchange
✅ **Multi-Factor Signals**: Technical + Emotion analysis

---

**Status**: ✅ Docker Paper Trading - FULLY CONFIGURED

**Next Step**: Run `./manage_paper_trading.sh build` then `./manage_paper_trading.sh start`
