"""
Cross-Exchange Arbitrage Strategy.

Exploits price differences across multiple exchanges.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from graphwiz_trader.hft.orderbook import OrderBookManager
from graphwiz_trader.hft.strategies.base import HFTStrategy


class CrossExchangeArbitrage(HFTStrategy):
    """Cross-exchange arbitrage strategy."""

    def __init__(
        self,
        config: Dict[str, Any],
        knowledge_graph: Any,
        orderbook_manager: OrderBookManager,
    ) -> None:
        """
        Initialize cross-exchange arbitrage strategy.

        Args:
            config: Strategy configuration
            knowledge_graph: Knowledge graph instance
            orderbook_manager: Order book manager instance
        """
        super().__init__(config, knowledge_graph)
        self.orderbook_manager = orderbook_manager
        self.min_profit_bps = config.get("min_profit_bps", 5.0)
        self.max_position_size = config.get("max_position_size", 0.1)
        self.fee_bps = config.get("fee_bps", 10.0)  # Trading fees in basis points
        self.active_arbitrages: Dict[str, Dict[str, Any]] = {}

    async def on_market_data(self, data: Dict[str, Any]) -> None:
        """
        Handle incoming market data (not primary data source for this strategy).

        Args:
            data: Market data update
        """
        # This strategy primarily uses order book updates
        pass

    async def on_orderbook_update(self, orderbook: Dict[str, Any]) -> None:
        """
        Check for arbitrage opportunities on each order book update.

        Args:
            orderbook: Order book data
        """
        if not self.running:
            return

        symbol = orderbook.get("symbol")
        if not symbol:
            return

        # Find arbitrage opportunities
        opportunities = self.orderbook_manager.get_arbitrage_opportunities(
            symbol, self.min_profit_bps
        )

        for opp in opportunities:
            await self._evaluate_and_execute_arbitrage(opp)

    async def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Generate arbitrage signal for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Arbitrage signal or None
        """
        opportunities = self.orderbook_manager.get_arbitrage_opportunities(
            symbol, self.min_profit_bps
        )

        if opportunities:
            # Return the best opportunity (highest profit)
            best_opp = max(opportunities, key=lambda x: x["profit_bps"])
            return best_opp

        return None

    async def _evaluate_and_execute_arbitrage(self, opportunity: Dict[str, Any]) -> None:
        """
        Evaluate and execute arbitrage opportunity.

        Args:
            opportunity: Arbitrage opportunity data
        """
        symbol = opportunity["symbol"]
        buy_exchange = opportunity["buy_exchange"]
        sell_exchange = opportunity["sell_exchange"]
        profit_bps = opportunity["profit_bps"]

        # Check if we already have an active arbitrage for this pair
        arb_key = f"{symbol}:{buy_exchange}:{sell_exchange}"
        if arb_key in self.active_arbitrages:
            return

        # Get available liquidity
        buy_book = self.orderbook_manager.get_book(buy_exchange, symbol)
        sell_book = self.orderbook_manager.get_book(sell_exchange, symbol)

        if not buy_book or not sell_book:
            return

        # Calculate maximum executable size
        buy_liquidity = buy_book.get_liquidity("buy", depth=5)
        sell_liquidity = sell_book.get_liquidity("sell", depth=5)
        max_size = min(buy_liquidity, sell_liquidity, self.max_position_size)

        if max_size <= 0:
            return

        # Adjust profit for fees
        net_profit_bps = profit_bps - self.fee_bps

        if net_profit_bps < self.min_profit_bps:
            logger.debug(
                f"Arbitrage opportunity below threshold after fees: "
                f"{net_profit_bps:.2f} bps (min: {self.min_profit_bps:.2f} bps)"
            )
            return

        # Calculate expected profit
        buy_price = opportunity["buy_price"]
        sell_price = opportunity["sell_price"]
        expected_profit = (sell_price - buy_price) * max_size
        expected_profit_bps = net_profit_bps

        logger.info(
            f"Arbitrage opportunity: Buy {symbol} on {buy_exchange} at {buy_price:.2f}, "
            f"Sell on {sell_exchange} at {sell_price:.2f}, "
            f"Size: {max_size:.4f}, Expected profit: ${expected_profit:.2f} "
            f"({expected_profit_bps:.2f} bps)"
        )

        # Execute arbitrage (simulation)
        await self._execute_simultaneous_trades(
            {
                "symbol": symbol,
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "quantity": max_size,
                "expected_profit_bps": expected_profit_bps,
                "expected_profit": expected_profit,
            }
        )

    async def _execute_simultaneous_trades(self, arbitrage: Dict[str, Any]) -> None:
        """
        Execute both legs of arbitrage simultaneously.

        Args:
            arbitrage: Arbitrage trade details
        """
        symbol = arbitrage["symbol"]
        buy_exchange = arbitrage["buy_exchange"]
        sell_exchange = arbitrage["sell_exchange"]
        quantity = arbitrage["quantity"]
        expected_profit = arbitrage["expected_profit"]

        # Mark arbitrage as active
        arb_key = f"{symbol}:{buy_exchange}:{sell_exchange}"
        self.active_arbitrages[arb_key] = arbitrage

        # In real implementation, this would:
        # 1. Place buy order on buy_exchange
        # 2. Place sell order on sell_exchange (simultaneously)
        # 3. Monitor fills
        # 4. Handle partial fills or failed orders

        # For simulation, assume successful execution
        logger.info(
            f"Executed arbitrage: "
            f"Buy {quantity:.4f} {symbol} on {buy_exchange}, "
            f"Sell {quantity:.4f} {symbol} on {sell_exchange}"
        )

        # Log the trade
        await self.log_trade(
            {
                "symbol": symbol,
                "side": "arbitrage",
                "price": arbitrage["sell_price"],
                "quantity": quantity,
                "pnl": expected_profit,
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": arbitrage["buy_price"],
                "sell_price": arbitrage["sell_price"],
            }
        )

        # Remove from active arbitrages
        self.active_arbitrages.pop(arb_key, None)

    def get_active_arbitrages(self) -> Dict[str, Dict[str, Any]]:
        """
        Get currently active arbitrage positions.

        Returns:
            Dictionary of active arbitrages
        """
        return self.active_arbitrages.copy()

    async def cancel_all_arbitrages(self) -> None:
        """Cancel all active arbitrage positions."""
        arb_keys = list(self.active_arbitrages.keys())
        for arb_key in arb_keys:
            arbitrage = self.active_arbitrages[arb_key]
            logger.info(f"Cancelling arbitrage: {arb_key}")
            # In real implementation, would cancel orders
            self.active_arbitrages.pop(arb_key, None)

    async def get_arbitrage_statistics(self) -> Dict[str, Any]:
        """
        Get arbitrage performance statistics.

        Returns:
            Statistics dictionary
        """
        performance = self.get_performance()

        stats = {
            "strategy": self.strategy_name,
            "total_arbitrages": performance["trades"],
            "successful_arbitrages": performance["winning_trades"],
            "failed_arbitrages": performance["losing_trades"],
            "success_rate": performance["win_rate"],
            "total_profit": performance["profit_loss"],
            "active_arbitrages": len(self.active_arbitrages),
        }

        if performance["winning_trades"] > 0:
            stats["average_profit"] = (
                performance["total_profit"] / performance["winning_trades"]
            )

        if performance["losing_trades"] > 0:
            stats["average_loss"] = (
                performance["total_loss"] / performance["losing_trades"]
            )

        return stats
