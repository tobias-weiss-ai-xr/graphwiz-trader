"""Main entry point for GraphWiz Trader."""

import argparse
import sys
from pathlib import Path
from loguru import logger

from graphwiz_trader.graph import KnowledgeGraph
from graphwiz_trader.trading import TradingEngine
from graphwiz_trader.agents import AgentOrchestrator
from graphwiz_trader.utils.config import load_config


class GraphWizTrader:
    """Main trading system class."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the GraphWiz Trader system.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.kg = None
        self.trading_engine = None
        self.agent_orchestrator = None
        self._running = False

    def start(self) -> None:
        """Start the trading system."""
        logger.info("Starting GraphWiz Trader v{}", self.config.get("version", "0.1.0"))

        # Initialize knowledge graph
        logger.info("Connecting to knowledge graph...")
        self.kg = KnowledgeGraph(self.config.get("neo4j", {}))
        self.kg.connect()

        # Initialize agent orchestrator
        logger.info("Initializing AI agents...")
        self.agent_orchestrator = AgentOrchestrator(
            self.config.get("agents", {}),
            self.kg
        )

        # Initialize trading engine
        logger.info("Initializing trading engine...")
        self.trading_engine = TradingEngine(
            self.config.get("trading", {}),
            self.config.get("exchanges", {}),
            self.kg,
            self.agent_orchestrator
        )

        # Start trading
        self._running = True
        self.trading_engine.start()

        logger.info("GraphWiz Trader started successfully")

    def stop(self) -> None:
        """Stop the trading system."""
        logger.info("Stopping GraphWiz Trader...")
        self._running = False

        if self.trading_engine:
            self.trading_engine.stop()

        if self.kg:
            self.kg.disconnect()

        logger.info("GraphWiz Trader stopped")

    def is_running(self) -> bool:
        """Check if the system is running.

        Returns:
            True if system is running, False otherwise
        """
        return self._running


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="GraphWiz Trader - Automated trading with knowledge graphs"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="GraphWiz Trader 0.1.0"
    )

    args = parser.parse_args()

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Configuration file not found: {}", config_path)
        logger.info("Copy config/config.example.yaml to config/config.yaml and configure it")
        return 1

    try:
        trader = GraphWizTrader(str(config_path))
        trader.start()

        # Keep running until interrupted
        while trader.is_running():
            try:
                import time
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                break

        trader.stop()
        return 0

    except Exception as e:
        logger.exception("Fatal error: {}", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
