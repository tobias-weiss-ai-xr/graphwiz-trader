# Configuration Refactoring Guide

## Overview

This document describes the configuration refactoring that centralizes trading parameters in YAML configuration files.

## New Configuration Files

### 1. `config/goemotions_trading.yaml`

Main configuration file for GoEmotions-based trading with environment variable support.

#### Key Features:
- Environment variable substitution: `${VAR:-default}`
- Separate configurations for live and paper trading
- Comprehensive GoEmotions settings
- Technical indicator parameters
- Risk management limits

#### Configuration Sections:

**Trading Mode:**
```yaml
trading_mode:
  type: live  # or paper
```

**Symbols:**
```yaml
symbols:
  live:
    - BTC/EUR
    - ETH/EUR
    - SOL/EUR
  paper:
    - BTC/EUR
    - ETH/EUR
```

**GoEmotions Settings:**
```yaml
goemotions:
  enabled: true
  strategy:
    extreme_euphoria_threshold: 0.75
    extreme_fear_threshold: 0.75
    use_contrarian_signals: true
    min_data_points: 3
```

**Live Trading Parameters:**
```yaml
live_trading:
  position_management:
    max_position_eur: 250
  trade_limits:
    max_daily_trades: 25
  execution:
    interval_seconds: 30
  confidence_threshold: 0.65
```

### 2. `src/graphwiz_trader/goemotions_config.py`

Configuration loader module providing:

- `GoEmotionsConfig`: Main configuration manager class
- `load_config()`: Convenience function to load config
- Environment variable substitution
- Dot-notation access: `config.get('live_trading.position_management.max_position_eur')`

## Usage

### Loading Configuration

```python
from graphwiz_trader.goemotions_config import load_config

# Load configuration
config = load_config('config/goemotions_trading.yaml')

# Access configuration
print(f"Trading mode: {config.trading_mode}")
print(f"Symbols: {config.symbols}")
print(f"Max position: €{config.get('live_trading.position_management.max_position_eur')}")
```

### Environment Variables

Set environment variables in `.env` file:

```bash
# Trading mode
TRADING_MODE=live

# Live trading
MAX_POSITION_EUR=250
MAX_DAILY_TRADES=25
UPDATE_INTERVAL_SECONDS=30
MAX_DAILY_LOSS_EUR=75

# Paper trading
PAPER_CAPITAL_EUR=10000
PAPER_DURATION_HOURS=72
PAPER_UPDATE_INTERVAL_SECONDS=30

# GoEmotions
GOEMOTIONS_MIN_DATA_POINTS=3
GOEMOTIONS_MAX_POSITION_PCT=0.25
```

## Changes Made

### Configuration Parameters Moved to Config File:

**Live Trading (`scripts/live_trade_goemotions.py`):**
- Max position EUR: `live_trading.position_management.max_position_eur`
- Max daily trades: `live_trading.trade_limits.max_daily_trades`
- Update interval: `live_trading.execution.interval_seconds`
- Confidence threshold: `live_trading.confidence_threshold`

**Paper Trading (`run_extended_paper_trading_goemotions.py`):**
- Capital: `paper_trading.capital.initial_eur`
- Update interval: `paper_trading.execution.interval_seconds`
- Duration: `paper_trading.duration.hours`

## Migration Guide

### Old Approach (Hardcoded):
```python
class GoEmotionsLiveTrader:
    def __init__(
        self,
        max_position_eur: float = 250.0,
        max_daily_trades: int = 25,
        update_interval_seconds: int = 30
    ):
        self.max_position = max_position_eur
        self.max_daily_trades = max_daily_trades
        self.update_interval = update_interval_seconds
```

### New Approach (Config-Driven):
```python
from graphwiz_trader.goemotions_config import load_config

class GoEmotionsLiveTrader:
    def __init__(self, config_path: str):
        config = load_config(config_path)
        
        live_config = config.get_live_trading_config()
        execution_config = live_config.get('execution', {})
        limits_config = live_config.get('trade_limits', {})
        position_config = live_config.get('position_management', {})
        
        self.max_position = position_config.get('max_position_eur', 250)
        self.max_daily_trades = limits_config.get('max_daily_trades', 25)
        self.update_interval = execution_config.get('interval_seconds', 30)
```

## Benefits

1. **Centralized Configuration**: All parameters in one place
2. **Environment Variable Support**: Easy to change settings without code changes
3. **Type Safety**: Configuration validation with defaults
4. **Hot Reloading**: Reload config without restart
5. **Separation of Concerns**: Config separate from business logic
6. **Easy Testing**: Use different configs for dev/test/prod

## Testing Configuration

```bash
# Test configuration loading
python3 -c "
from graphwiz_trader.goemotions_config import load_config
config = load_config('config/goemotions_trading.yaml')
print(f'Trading mode: {config.trading_mode}')
print(f'Max position: €{config.get(\"live_trading.position_management.max_position_eur\")}')
"
```

## Docker Integration

Update Docker Compose to mount configuration:

```yaml
services:
  live-trading:
    volumes:
      - ./config/goemotions_trading.yaml:/app/config/goemotions_trading.yaml:ro
      - .env:/app/.env:ro
    environment:
      - TRADING_MODE=live
      - MAX_POSITION_EUR=250
      - MAX_DAILY_TRADES=25
```

## Future Enhancements

- Add config validation schema
- Support for multiple environment configs (dev/stage/prod)
- Web UI for config management
- Config versioning and rollback
- Config diff and comparison tools
