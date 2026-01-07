# Live Trading Deployment with New Configuration

## Prerequisites

1. Updated Docker image with new configuration support
2. Configuration file: `config/goemotions_trading.yaml`
3. Environment variables in `.env` file

## Deployment Steps

### 1. Update .env File

Add or update these environment variables in `/opt/git/graphwiz-trader/.env`:

```bash
# Trading Configuration
MAX_POSITION_EUR=250
MAX_DAILY_TRADES=25
MAX_DAILY_LOSS_EUR=75
UPDATE_INTERVAL_SECONDS=30

# GoEmotions Configuration
GOEMOTIONS_MIN_DATA_POINTS=3
GOEMOTIONS_MAX_POSITION_PCT=0.25
```

### 2. Rebuild Docker Image

```bash
cd /opt/git/graphwiz-trader
docker build -f Dockerfile.live-trading -t graphwiz-live-trading:latest .
```

### 3. Deploy with Docker Compose

```bash
cd /opt/git/graphwiz-trader
docker-compose -f docker-compose.live-trading.yml up -d --build
```

Or deploy with systemd:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable graphwiz-live-trading
sudo systemctl start graphwiz-live-trading
```

### 4. Verify Deployment

```bash
# Check container status
docker ps --filter "name=graphwiz-live-trading"

# Check logs
docker logs -f graphwiz-live-trading

# Check health
curl http://localhost:8080/health
```

## Configuration Changes

### Before (Hardcoded Parameters)

```bash
ExecStart=/usr/bin/docker run --rm \
  --name graphwiz-live-trading \
  --env-file /opt/git/graphwiz-trader/.env \
  graphwiz-live-trading:latest \
  python scripts/live_trade_goemotions.py \
    --symbols BTC/EUR ETH/EUR SOL/EUR ... \
    --max-position 250 \
    --max-daily-loss 75 \
    --max-daily-trades 15 \
    --interval 3600 \
    --no-confirm
```

### After (Config File + Environment Variables)

```bash
ExecStart=/usr/bin/docker run --rm \
  --name graphwiz-live-trading \
  --env-file /opt/git/graphwiz-trader/.env \
  -e MAX_POSITION_EUR=250 \
  -e MAX_DAILY_TRADES=25 \
  -e MAX_DAILY_LOSS_EUR=75 \
  -e UPDATE_INTERVAL_SECONDS=30 \
  -v /opt/git/graphwiz-trader/config:/app/config:ro \
  graphwiz-live-trading:latest \
  python scripts/live_trade_goemotions_new.py \
    --config config/goemotions_trading.yaml
```

## Key Benefits

1. **Centralized Configuration**: All parameters in `config/goemotions_trading.yaml`
2. **Environment Variable Overrides**: Easy changes without code changes
3. **Parameter Changes Applied**:
   - Confidence threshold: 0.70 → 0.65
   - Max daily trades: 15 → 25
   - Trading interval: 3600s → 30s
4. **Type Safety**: Configuration validation with defaults
5. **Hot Reloading**: Can reload config without restart

## Monitoring

### Logs

```bash
# Follow container logs
docker logs -f graphwiz-live-trading

# Check systemd logs
sudo journalctl -u graphwiz-live-trading -f

# Check trading logs
tail -f /opt/git/graphwiz-trader/logs/live_trading/live_trading_service.log
```

### Health Checks

```bash
# Container health
docker inspect --format='{{.State.Health.Status}}' graphwiz-live-trading

# HTTP health endpoint
curl http://localhost:8080/health

# Metrics endpoint
curl http://localhost:9090/metrics
```

## Troubleshooting

### Container Not Starting

```bash
# Check logs
docker logs graphwiz-live-trading

# Check configuration
python3 -c "
from graphwiz_trader.goemotions_config import load_config
config = load_config('config/goemotions_trading.yaml')
print(f'Config loaded: {config.trading_mode}')
"
```

### Configuration Errors

```bash
# Validate YAML
python3 -c "
import yaml
with open('config/goemotions_trading.yaml') as f:
    config = yaml.safe_load(f)
    print('Valid YAML')
"
```

### Environment Variables Not Set

```bash
# Check .env file
cat /opt/git/graphwiz-trader/.env | grep -E "MAX_|UPDATE_|GOEMOTIONS_"

# Test environment variables
docker run --rm \
  --env-file /opt/git/graphwiz-trader/.env \
  -e MAX_POSITION_EUR=250 \
  python:3.10-slim \
  sh -c 'echo MAX_POSITION_EUR=$MAX_POSITION_EUR'
```

## Rollback

If issues occur, rollback to previous configuration:

```bash
# Stop service
sudo systemctl stop graphwiz-live-trading

# Revert to old script
sudo systemctl edit graphwiz-live-trading
# Change live_trade_goemotions_new.py back to live_trade_goemotions.py

# Restart
sudo systemctl start graphwiz-live-trading
```

## Maintenance

### Update Configuration

1. Edit `config/goemotions_trading.yaml`
2. Restart service (or reload config in application)
3. Verify changes in logs

### Update Environment Variables

1. Edit `.env` file
2. Restart container
3. Verify new values applied

```bash
# Quick restart with env changes
docker-compose -f docker-compose.live-trading.yml restart
```
