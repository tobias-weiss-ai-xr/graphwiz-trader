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
