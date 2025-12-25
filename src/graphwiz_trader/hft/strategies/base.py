"""
HFT Strategy Base Classes.

Provides abstract base classes for high-frequency trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger


class HFTStrategy(ABC):
    """Base class for HFT strategies."""

    def __init__(self, config: Dict[str, Any], knowledge_graph: Any) -> None:
        """
        Initialize HFT strategy.

        Args:
            config: Strategy configuration
            knowledge_graph: Knowledge graph instance for pattern storage
        """
        self.config = config
        self.kg = knowledge_graph
        self.running = False
        self.performance = {
            "trades": 0,
            "profit_loss": 0.0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
        }
        self.strategy_name = self.__class__.__name__

    @abstractmethod
    async def on_market_data(self, data: Dict[str, Any]) -> None:
        """
        Handle incoming market data.

        Args:
            data: Market data update
        """
        pass

    @abstractmethod
    async def on_orderbook_update(self, orderbook: Dict[str, Any]) -> None:
        """
        Handle order book update.

        Args:
            orderbook: Order book data
        """
        pass

    @abstractmethod
    async def generate_signal(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal.

        Returns:
            Trading signal dictionary or None
        """
        pass

    async def start(self) -> None:
        """Start strategy."""
        self.running = True
        logger.info(f"Starting {self.strategy_name}")

    async def stop(self) -> None:
        """Stop strategy."""
        self.running = False
        logger.info(f"Stopped {self.strategy_name}")
        await self.log_performance()

    async def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Log trade to knowledge graph and update performance.

        Args:
            trade: Trade information
        """
        self.performance["trades"] += 1
        pnl = trade.get("pnl", 0.0)
        self.performance["profit_loss"] += pnl

        if pnl > 0:
            self.performance["winning_trades"] += 1
            self.performance["total_profit"] += pnl
        elif pnl < 0:
            self.performance["losing_trades"] += 1
            self.performance["total_loss"] += abs(pnl)

        # Calculate win rate
        if self.performance["trades"] > 0:
            self.performance["win_rate"] = (
                self.performance["winning_trades"] / self.performance["trades"]
            )

        # Store in Neo4j for analysis
        if self.kg:
            try:
                await self.kg.write(
                    """
                    CREATE (t:Trade {
                        strategy: $strategy,
                        symbol: $symbol,
                        side: $side,
                        price: $price,
                        quantity: $quantity,
                        pnl: $pnl,
                        timestamp: datetime()
                    })
                    """,
                    strategy=self.strategy_name,
                    symbol=trade.get("symbol"),
                    side=trade.get("side"),
                    price=trade.get("price"),
                    quantity=trade.get("quantity"),
                    pnl=pnl,
                )
            except Exception as e:
                logger.error(f"Failed to log trade to knowledge graph: {e}")

    async def log_performance(self) -> None:
        """Log current performance metrics."""
        logger.info(f"{self.strategy_name} Performance:")
        logger.info(f"  Total Trades: {self.performance['trades']}")
        logger.info(f"  Win Rate: {self.performance['win_rate']:.2%}")
        logger.info(f"  Total P&L: ${self.performance['profit_loss']:.2f}")
        logger.info(f"  Winning Trades: {self.performance['winning_trades']}")
        logger.info(f"  Losing Trades: {self.performance['losing_trades']}")

        if self.performance["winning_trades"] > 0:
            avg_win = (
                self.performance["total_profit"] / self.performance["winning_trades"]
            )
            logger.info(f"  Average Win: ${avg_win:.2f}")

        if self.performance["losing_trades"] > 0:
            avg_loss = (
                self.performance["total_loss"] / self.performance["losing_trades"]
            )
            logger.info(f"  Average Loss: ${avg_loss:.2f}")

    def get_performance(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Performance dictionary
        """
        return self.performance.copy()

    async def reset_performance(self) -> None:
        """Reset performance metrics."""
        self.performance = {
            "trades": 0,
            "profit_loss": 0.0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
        }
        logger.info(f"Performance metrics reset for {self.strategy_name}")
