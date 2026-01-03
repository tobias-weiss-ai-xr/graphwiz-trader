# Paper Trading Container Service

Run GraphWiz paper trading as a Docker container service with easy management.

## Quick Start

### 1. Build the Image

```bash
./manage_paper_trading.sh build
```

### 2. Start the Service

```bash
# Start with default settings (7 days, 30-min intervals)
./manage_paper_trading.sh start

# Start with custom settings
./manage_paper_trading.sh start --duration 24 --interval 5
```

### 3. Monitor the Service

```bash
# Check status
./manage_paper_trading.sh status

# View logs
./manage_paper_trading.sh logs

# View resource usage
./manage_paper_trading.sh stats
```

## Management Commands

| Command | Description |
|---------|-------------|
| `build` | Build Docker image |
| `start` | Start paper trading service |
| `stop` | Stop paper trading service |
| `restart` | Restart service |
| `status` | Show service status and recent logs |
| `logs` | Follow service logs (live) |
| `stats` | Show resource usage statistics |
| `shell` | Open shell inside container |
| `clean` | Remove containers and images |
| `help` | Show help message |

## Configuration

### Duration & Intervals

```bash
# Run for 24 hours with 5-minute analysis intervals
./manage_paper_trading.sh start --duration 24 --interval 5

# Run for 1 week with 30-minute intervals (default)
./manage_paper_trading.sh start --duration 168 --interval 30

# Run for 72 hours with 15-minute intervals
./manage_paper_trading.sh start --duration 72 --interval 15
```

### Strategy Configuration

Edit `run_extended_paper_trading.py` to modify trading strategy:

```python
# Current: AGGRESSIVE strategy
# - Buy when RSI < 42 (was 30)
# - Sell when RSI > 58 (was 70)
# - Position size: 25% of portfolio
# - Sell 75% of position on exit
```

## Docker Compose

### Direct Docker Compose Commands

```bash
# Start service
docker-compose -f docker-compose.paper-trading.yml up -d

# Stop service
docker-compose -f docker-compose.paper-trading.yml down

# View logs
docker-compose -f docker-compose.paper-trading.yml logs -f

# Restart service
docker-compose -f docker-compose.paper-trading.yml restart
```

### Environment Variables

Edit `docker-compose.paper-trading.yml`:

```yaml
environment:
  - LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
  - TZ=UTC                 # Timezone
  - PYTHONUNBUFFERED=1      # Python output
```

## Resource Limits

Default limits in `docker-compose.paper-trading.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '0.5'
      memory: 512M
```

Adjust based on your system.

## Volumes & Persistence

### Logs Directory

```
./logs/paper_trading/        # Host
↓
/app/logs/paper_trading/     # Container
```

Logs persist on the host even if container is removed.

### Log Files

- `validation_YYYYMMDD_HHMMSS.log` - Detailed log
- `validation_stdout.log` - Standard output
- `equity_YYYYMMDD_HHMMSS.csv` - Equity curve data

## Monitoring

### Check Service Health

```bash
# Using management script
./manage_paper_trading.sh status

# Using docker
docker ps -f name=graphwiz-paper-trading

# Container health
docker inspect graphwiz-paper-trading | grep -A 5 Health
```

### View Live Logs

```bash
# Follow all logs
./manage_paper_trading.sh logs

# Or with docker
docker logs -f graphwiz-paper-trading

# Last 100 lines
docker logs --tail 100 graphwiz-paper-trading
```

### Resource Monitoring

```bash
# Live stats
./manage_paper_trading.sh stats

# Or with docker
docker stats graphwiz-paper-trading
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs graphwiz-paper-trading

# Check if port is already in use
docker ps -a

# Remove old container
docker rm -f graphwiz-paper-trading

# Restart
./manage_paper_trading.sh start
```

### No Trades Executing

Check the strategy settings:

```bash
# View logs for signals
./manage_paper_trading.sh logs | grep "Signal:"

# Open shell to check configuration
./manage_paper_trading.sh shell
cat run_extended_paper_trading.py | grep -A 20 "_analyze_signals"
```

### High Memory Usage

```bash
# Check resource usage
./manage_paper_trading.sh stats

# Reduce memory limit in docker-compose.paper-trading.yml
# Then restart
./manage_paper_trading.sh restart
```

### View Inside Container

```bash
# Open shell
./manage_paper_trading.sh shell

# Inside container:
ls -la logs/paper_trading/
tail -50 logs/paper_trading/validation_stdout.log
ps aux
```

## Advanced Usage

### Custom Docker Build

```bash
# Build with custom tag
docker build -f Dockerfile.paper-trading -t graphwiz-paper-trading:v2 .

# Run custom container
docker run -d \
  --name graphwiz-paper-trading \
  -v $(pwd)/logs/paper_trading:/app/logs/paper_trading \
  -e LOG_LEVEL=DEBUG \
  graphwiz-paper-trading:v2 \
  python run_extended_paper_trading.py --duration 48 --interval 10
```

### Multiple Instances

```bash
# Instance 1
docker run -d \
  --name graphwiz-paper-trading-1 \
  -v $(pwd)/logs/paper_trading:/app/logs/paper_trading \
  graphwiz-paper-trading \
  python run_extended_paper_trading.py --duration 24 --interval 5

# Instance 2 (different strategy/symbols)
docker run -d \
  --name graphwiz-paper-trading-2 \
  -v $(pwd)/logs/paper_trading:/app/logs/paper_trading \
  graphwiz-paper-trading \
  python run_extended_paper_trading.py --duration 24 --interval 10
```

### Integration with Monitoring

Add to existing monitoring stack:

```yaml
# In your docker-compose.yml
services:
  paper-trading:
    image: graphwiz-paper-trading:latest
    # ... configuration ...

  prometheus:
    image: prom/prometheus
    # ... configuration ...

  grafana:
    image: grafana/grafana
    # ... configuration ...
```

## Production Deployment

### Systemd Service

Create `/etc/systemd/system/graphwiz-paper-trading.service`:

```ini
[Unit]
Description=GraphWiz Paper Trading Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/git/graphwiz-trader
ExecStart=/opt/git/graphwiz-trader/manage_paper_trading.sh start
ExecStop=/opt/git/graphwiz-trader/manage_paper_trading.sh stop
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable graphwiz-paper-trading
sudo systemctl start graphwiz-paper-trading
sudo systemctl status graphwiz-paper-trading
```

### Auto-Restart Policy

In `docker-compose.paper-trading.yml`:

```yaml
services:
  paper-trading:
    restart: unless-stopped  # Always restart except manual stop
    # or
    restart: always           # Always restart
    # or
    restart: on-failure       # Restart only on failure
```

### Log Rotation

Docker logs are configured to rotate:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"   # Max file size 10MB
    max-file: "3"      # Keep 3 files
```

Application logs in `logs/paper_trading/` can be rotated with logrotate:

```bash
# /etc/logrotate.d/graphwiz-paper-trading
/opt/git/graphwiz-trader/logs/paper_trading/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

## Security

### Run as Non-Root

Add to `Dockerfile.paper-trading`:

```dockerfile
RUN useradd -m -u 1000 trader
USER trader
```

### Resource Limits

Already configured in `docker-compose.paper-trading.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
```

### Network Isolation

```yaml
# Add to docker-compose
networks:
  default:
    driver: bridge
    internal: true  # No external network access
```

## Performance Tuning

### Optimization Tips

1. **Reduce Interval** - More frequent analysis = more trades
2. **Adjust Position Size** - Edit `run_extended_paper_trading.py`
3. **Resource Limits** - Increase CPU/memory for faster execution
4. **Log Level** - Set to WARNING to reduce I/O

### Benchmarking

```bash
# Monitor performance
docker stats graphwiz-paper-trading --no-stream

# Check execution time
grep "Iteration" logs/paper_trading/validation_stdout.log
```

## Support

For issues:
1. Check logs: `./manage_paper_trading.sh logs`
2. Check status: `./manage_paper_trading.sh status`
3. Review this documentation
4. Check Docker: `docker ps -a`

## Summary

The containerized paper trading service provides:
- ✅ Easy deployment and management
- ✅ Resource isolation and limits
- ✅ Automatic restart on failure
- ✅ Persistent log storage
- ✅ Simple monitoring and debugging
- ✅ Production-ready configuration

Use the management script for all operations!
