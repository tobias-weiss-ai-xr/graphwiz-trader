"""
HFT Engine.

Main orchestrator for high-frequency trading operations.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from graphwiz_trader.hft.analytics import HFTAnalytics
from graphwiz_trader.hft.executor import FastOrderExecutor
from graphwiz_trader.hft.market_data import WebSocketMarketData
from graphwiz_trader.hft.monitoring import PerformanceMonitor
from graphwiz_trader.hft.orderbook import OrderBookManager
from graphwiz_trader.hft.risk import HFTRiskManager
from graphwiz_trader.hft.strategies import (
    CrossExchangeArbitrage,
    HFTStrategy,
    StatisticalArbitrage,
)


class HFTEngine:
    """
    High-Frequency Trading Engine.

    Orchestrates all HFT components including market data, strategies,
    order execution, risk management, and analytics.
    """

    def __init__(self, config: Dict[str, Any], knowledge_graph: Any) -> None:
        """
        Initialize HFT engine.

        Args:
            config: HFT configuration dictionary
            knowledge_graph: Neo4j knowledge graph instance
        """
        self.config = config
        self.kg = knowledge_graph
        self.running = False

        # Initialize components
        self.orderbook_manager = OrderBookManager()
        self.risk_manager = HFTRiskManager(config.get("risk", {}))
        self.executor = FastOrderExecutor(config.get("exchanges", {}))
        self.market_data = WebSocketMarketData(config.get("exchanges", {}))
        self.analytics = HFTAnalytics(knowledge_graph)

        # Performance monitoring
        monitoring_interval = config.get("performance", {}).get("monitoring_interval", 5)
        self.monitor = PerformanceMonitor(knowledge_graph, monitoring_interval)

        # Strategies
        self.strategies: List[HFTStrategy] = []
        self._initialize_strategies()

        # Register market data callbacks
        self.market_data.register_callback("ticker", self._on_ticker)
        self.market_data.register_callback("orderbook", self._on_orderbook)

    def _initialize_strategies(self) -> None:
        """Initialize trading strategies based on configuration."""
        strategies_config = self.config.get("strategies", {})

        # Statistical Arbitrage
        if strategies_config.get("statistical_arbitrage", {}).get("enabled", False):
            stat_arb = StatisticalArbitrage(strategies_config["statistical_arbitrage"], self.kg)
            self.strategies.append(stat_arb)
            logger.info("Initialized Statistical Arbitrage strategy")

        # Cross-Exchange Arbitrage
        if strategies_config.get("cross_exchange_arbitrage", {}).get("enabled", False):
            cross_arb = CrossExchangeArbitrage(
                strategies_config["cross_exchange_arbitrage"],
                self.kg,
                self.orderbook_manager,
            )
            self.strategies.append(cross_arb)
            logger.info("Initialized Cross-Exchange Arbitrage strategy")

    async def start(self) -> None:
        """Start the HFT engine."""
        if self.running:
            logger.warning("HFT engine is already running")
            return

        logger.info("Starting HFT engine...")
        self.running = True

        # Start performance monitoring
        await self.monitor.start()

        # Start strategies
        for strategy in self.strategies:
            await strategy.start()

        # Start market data feeds
        await self.market_data.start()

        logger.info("HFT engine started successfully")

    async def stop(self) -> None:
        """Stop the HFT engine."""
        if not self.running:
            logger.warning("HFT engine is not running")
            return

        logger.info("Stopping HFT engine...")
        self.running = False

        # Stop market data feeds
        await self.market_data.stop()

        # Stop strategies
        for strategy in self.strategies:
            await strategy.stop()

        # Stop performance monitoring
        await self.monitor.stop()

        # Close executor connections
        await self.executor.close()

        logger.info("HFT engine stopped successfully")

    async def _on_ticker(self, data: Dict[str, Any]) -> None:
        """
        Handle ticker data updates.

        Args:
            data: Ticker data
        """
        try:
            # Record latency
            import time

            start_time = time.time()

            # Forward to strategies
            for strategy in self.strategies:
                if strategy.running:
                    await strategy.on_market_data(data)

            # Record processing latency
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_latency("market_data", latency_ms)

        except Exception as e:
            logger.error(f"Error processing ticker data: {e}")

    async def _on_orderbook(self, data: Dict[str, Any]) -> None:
        """
        Handle order book updates.

        Args:
            data: Order book data
        """
        try:
            import time

            start_time = time.time()

            # Update order book manager
            exchange = data.get("exchange")
            symbol = data.get("symbol")

            if exchange and symbol:
                self.orderbook_manager.update(exchange, symbol, data)

            # Forward to strategies
            for strategy in self.strategies:
                if strategy.running:
                    await strategy.on_orderbook_update(data)

            # Record processing latency
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_latency("market_data", latency_ms)

        except Exception as e:
            logger.error(f"Error processing order book update: {e}")

    async def execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a trading order with risk checks.

        Args:
            order: Order details

        Returns:
            Execution result or None if rejected
        """
        try:
            import time

            start_time = time.time()

            # Check risk limits
            approved, reason = await self.risk_manager.check_order(order)

            if not approved:
                logger.warning(f"Order rejected: {reason}")
                return None

            # Execute order
            result = await self.executor.place_order(
                exchange=order["exchange"],
                symbol=order["symbol"],
                side=order["side"],
                amount=order["amount"],
                price=order.get("price"),
                order_type=order.get("order_type", "market"),
            )

            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_latency("order_execution", latency_ms)

            # Update risk manager
            if result and result.get("status") == "filled":
                fill_data = {
                    "symbol": result["symbol"],
                    "exchange": result["exchange"],
                    "side": result["side"],
                    "filled": result["amount"],
                    "pnl": 0.0,  # Will be calculated later
                }
                self.risk_manager.update_position(fill_data)

            return result

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return None

    async def execute_simultaneous_orders(
        self, orders: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Execute multiple orders simultaneously (for arbitrage).

        Args:
            orders: List of order details

        Returns:
            List of execution results
        """
        try:
            import time

            start_time = time.time()

            # Check all orders against risk limits
            approved_orders = []
            for order in orders:
                approved, reason = await self.risk_manager.check_order(order)
                if approved:
                    approved_orders.append(order)
                else:
                    logger.warning(f"Order rejected: {reason}")

            if not approved_orders:
                return []

            # Execute all approved orders
            results = await self.executor.place_simultaneous_orders(approved_orders)

            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_latency("order_execution", latency_ms)

            # Update risk manager for successful fills
            for result in results:
                if isinstance(result, dict) and result.get("status") == "filled":
                    fill_data = {
                        "symbol": result["symbol"],
                        "exchange": result["exchange"],
                        "side": result["side"],
                        "filled": result["amount"],
                        "pnl": 0.0,
                    }
                    self.risk_manager.update_position(fill_data)

            return results

        except Exception as e:
            logger.error(f"Error executing simultaneous orders: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """
        Get engine status.

        Returns:
            Status dictionary
        """
        return {
            "running": self.running,
            "strategies": [
                {
                    "name": s.strategy_name,
                    "running": s.running,
                    "performance": s.get_performance(),
                }
                for s in self.strategies
            ],
            "risk_metrics": self.risk_manager.get_risk_metrics(),
            "performance_metrics": self.monitor.get_current_metrics(),
            "circuit_breaker": self.risk_manager.circuit_breaker_tripped,
        }

    async def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics summary.

        Args:
            days: Look back period in days

        Returns:
            Analytics summary
        """
        summary = {}

        # Get strategy performance
        for strategy in self.strategies:
            perf = await self.analytics.get_strategy_performance(strategy.strategy_name, days)
            if perf:
                summary[strategy.strategy_name] = perf

        # Get top performers
        top_performers = await self.analytics.get_top_performers(days=days)
        summary["top_performers"] = top_performers

        # Get performance summary
        summary["system_performance"] = self.monitor.get_performance_summary()

        return summary

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker (manual intervention)."""
        self.risk_manager.reset_circuit_breaker()
        logger.info("Circuit breaker reset via HFT engine")
