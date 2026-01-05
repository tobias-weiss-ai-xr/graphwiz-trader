# GraphWiz Trader

**Automated trading system powered by knowledge graphs**

## Overview

GraphWiz Trader is an intelligent automated trading system that leverages knowledge graph technology to make informed trading decisions. By integrating market data, technical indicators, and fundamental analysis into a Neo4j knowledge graph, the system can uncover complex relationships and patterns that traditional trading systems might miss.

## Architecture

```
graphwiz-trader/
├── src/
│   └── graphwiz_trader/
│       ├── agents/          # AI agents for trading decisions
│       ├── analysis/        # Technical and fundamental analysis
│       ├── graph/           # Neo4j knowledge graph integration
│       ├── trading/         # Trading execution and order management
│       └── utils/           # Utility functions
├── config/                  # Configuration files
├── data/                    # Market data storage
├── logs/                    # Application logs
├── backtests/               # Backtesting results
└── tests/                   # Test suite
```

## Features

- **Knowledge Graph Integration**: Uses Neo4j to store and query relationships between assets, markets, and economic indicators
- **Multi-Agent System**: AI agents specialized in different aspects of trading (technical analysis, sentiment analysis, risk management)
- **Real-time Data Processing**: Streams and processes market data in real-time
- **Backtesting Engine**: Test strategies against historical data
- **Risk Management**: Built-in position sizing and stop-loss mechanisms
- **Multi-Exchange Support**: Trade across multiple exchanges simultaneously

## Tech Stack

- **Python 3.10+**
- **Neo4j**: Knowledge graph database
- **LangChain**: AI agent framework
- **pandas**: Data analysis
- **CCXT**: Cryptocurrency exchange integration
- **Docker**: Containerization

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Neo4j database (local or remote)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/tobias-weiss-ai-xr/graphwiz-trader.git
cd graphwiz-trader
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the system:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

5. Run the system:
```bash
python -m graphwiz_trader.main
```

### Docker Setup

```bash
docker-compose up -d
```

## Configuration

Edit `config/config.yaml` to configure:

- Neo4j connection settings
- Exchange API keys
- Trading parameters
- Risk management settings
- Agent configurations

## Usage

### Basic Trading

```python
from graphwiz_trader import GraphWizTrader

# Initialize trader
trader = GraphWizTrader(config_path='config/config.yaml')

# Start trading
trader.start()
```

### Backtesting

```python
from graphwiz_trader.backtesting import BacktestEngine

engine = BacktestEngine(config_path='config/config.yaml')
results = engine.run(
    strategy='momentum',
    start_date='2024-01-01',
    end_date='2024-12-31'
)
print(results.summary())
```

### Knowledge Graph Queries

```python
from graphwiz_trader.graph import KnowledgeGraph

kg = KnowledgeGraph()
kg.connect()

# Find correlated assets
correlations = kg.query("""
    MATCH (a:Asset)-[:CORRELATED_WITH]->(b:Asset)
    WHERE a.symbol = 'BTC'
    RETURN b.symbol, b.correlation_coefficient
    ORDER BY b.correlation_coefficient DESC
""")
```

## Trading System Features

### 1. Paper Trading (No Real Money)

Test strategies without risking real money:

```bash
# Run once
python scripts/paper_trade.py --symbol BTC/USDT --capital 10000

# Run continuously
python scripts/paper_trade.py --symbol BTC/USDT --continuous --interval 3600

# Custom parameters
python scripts/paper_trade.py --symbol BTC/USDT --oversold 25 --overbought 65
```

**Features**:
- ✅ Real market data from exchanges
- ✅ Virtual portfolio tracking
- ✅ Trade history and performance metrics
- ✅ NO real money at risk

**Documentation**: See [docs/PAPER_TRADING.md](docs/PAPER_TRADING.md)

### 2. Live Trading (Real Money)

⚠️ **WARNING**: Executes REAL trades with REAL money. Always paper trade first!

```bash
# Set API credentials
export EXCHANGE_API_KEY="your_api_key"
export EXCHANGE_API_SECRET="your_api_secret"

# Test run (no trades)
python scripts/live_trade.py --symbol BTC/USDT --test

# Start live trading
python scripts/live_trade.py \
    --symbol BTC/USDT \
    --max-position 100 \
    --max-daily-loss 200 \
    --max-daily-trades 5
```

**Safety Features**:
- ✅ Position size limits
- ✅ Daily loss limits
- ✅ Maximum trade counts
- ✅ Automatic stop-loss (2%)
- ✅ Automatic take-profit (5%)
- ✅ Emergency shutdown

**Documentation**: See [docs/LIVE_TRADING.md](docs/LIVE_TRADING.md)

### 3. Advanced Strategies

**Adaptive RSI Strategy** - Adjusts parameters based on market conditions:
```python
from graphwiz_trader.strategies import AdaptiveRSIStrategy

strategy = AdaptiveRSIStrategy(
    base_oversold=30,
    base_overbought=70,
    adaptation_strength=0.5
)
```

**Market Regime Detection**:
```python
from graphwiz_trader.strategies import RegimeDetector

detector = RegimeDetector()
regime_info = detector.detect(market_data)
# Returns: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, etc.
```

**Documentation**: See [docs/STRATEGY_ENHANCEMENTS.md](docs/STRATEGY_ENHANCEMENTS.md)

### 4. Monitoring & Alerts

**Real-time monitoring** with notifications to Telegram/Discord:

```bash
# Set up Telegram alerts
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Or Discord alerts
export DISCORD_WEBHOOK_URL="your_webhook_url"
```

**Alerts include**:
- ✅ Trade executions
- ✅ Position closed (with P&L)
- ✅ Daily summaries
- ✅ System errors
- ✅ Critical events

**Documentation**: See [docs/MONITORING.md](docs/MONITORING.md)

## Backtesting Results & Recommendations

### Performance Summary (Real Market Data)

Based on comprehensive backtesting with real historical data from Binance (Nov 2024 - Dec 2024):

**Best Performing Strategy**: RSI Mean Reversion
| Metric | Value |
|--------|-------|
| Return | +0.50% |
| Win Rate | 100% (3/3 trades) |
| Max Drawdown | 2.45% |
| Period | 90 days |
| Timeframe | Daily (1d) |

**Key Findings**:
- **Less is More**: RSI's 3 trades outperformed SMA's 37-92 trades
- **Transaction Costs**: High-frequency trading loses 1-3% to fees
- **Timeframe**: Daily signals show best quality, lowest noise
- **Market Conditions**: Current sideways/choppy market unfavorable for trend-following

### Recommended Strategy Configuration

```python
from graphwiz_trader.backtesting import RSIMeanReversionStrategy

strategy = RSIMeanReversionStrategy(
    oversold=25,      # Buy when RSI < 25 (extreme fear)
    overbought=65,    # Sell when RSI > 65 (moderate greed)
    period=14         # Standard RSI period
)
```

**Parameters**:
- **Timeframe**: 1d (daily candles)
- **Stop Loss**: 2%
- **Take Profit**: 5%
- **Max Position Size**: 1-2% of portfolio
- **Expected Trades**: 3-5 per quarter (conservative)

**Usage Example**:
```bash
# Fetch real data
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1d --days 90 --save

# Run backtest
python scripts/backtest_real_data.py --symbol BTC/USDT --strategies rsi

# Optimize parameters
python scripts/optimize_params.py --file data/BTC_USDT_1d_20251226.csv --strategy rsi

# View results summary
python scripts/summarize_results.py
```

### Important Warnings

⚠️ **Past performance ≠ future results**
- 90 days is a short sample period
- Market conditions change constantly
- Always use stop-losses
- Start with paper trading
- Never risk more than 1-2% per trade
- Diversify across multiple assets

### Strategy Comparison

| Strategy | Return | Win Rate | Trades | Drawdown | Verdict |
|----------|--------|----------|--------|----------|---------|
| RSI (25/65) | +0.50% | 100% | 3 | 2.45% | ✅ Recommended |
| RSI (30/70) | -0.17% to +0.21% | 50-100% | 2-6 | 0.14-5.63% | ⚠️ Moderate |
| SMA (5/20) | 0% | 0% | 37 | 0% | ❌ No signals |
| SMA (10/30) | -1.85% to -3.63% | 7-51% | 78-92 | 3.38-3.63% | ❌ Over-trading |

## Project Status

This project is currently in early development. Features are being added incrementally.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Contact

- GitHub: [@tobias-weiss-ai-xr](https://github.com/tobias-weiss-ai-xr)
- Project: [GraphWiz Trader](https://github.com/tobias-weiss-ai-xr/graphwiz-trader)

## Acknowledgments

- [Neo4j](https://neo4j.com/) - Graph database platform
- [LangChain](https://langchain.com/) - AI agent framework
- [CCXT](https://ccxt.com/) - Exchange integration library
