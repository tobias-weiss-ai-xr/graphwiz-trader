"""
Statistical Arbitrage Strategy.

Implements mean reversion strategy based on statistical analysis.
"""

from collections import defaultdict, deque
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from graphwiz_trader.hft.strategies.base import HFTStrategy


class StatisticalArbitrage(HFTStrategy):
    """Statistical arbitrage using mean reversion."""

    def __init__(self, config: Dict[str, Any], knowledge_graph: Any) -> None:
        """
        Initialize statistical arbitrage strategy.

        Args:
            config: Strategy configuration
            knowledge_graph: Knowledge graph instance
        """
        super().__init__(config, knowledge_graph)
        self.lookback_period = config.get("lookback", 100)
        self.z_score_threshold = config.get("z_threshold", 2.0)
        self.price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.lookback_period)
        )
        self.positions: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}

    async def on_market_data(self, data: Dict[str, Any]) -> None:
        """
        Process market tick data.

        Args:
            data: Market data update
        """
        symbol = data.get("symbol")
        if not symbol:
            return

        # Calculate mid price
        bid = data.get("bid")
        ask = data.get("ask")

        if bid and ask:
            price = (bid + ask) / 2
        elif data.get("last"):
            price = data["last"]
        else:
            return

        # Store price in history
        self.price_history[symbol].append(
            {
                "price": price,
                "timestamp": data.get("timestamp"),
            }
        )

        # Check for mean reversion signal
        if self.running:
            signal = await self.generate_signal(symbol)
            if signal:
                await self._execute_signal(signal)

    async def on_orderbook_update(self, orderbook: Dict[str, Any]) -> None:
        """
        Handle order book updates (not used in this strategy).

        Args:
            orderbook: Order book data
        """
        # Statistical arbitrage primarily uses ticker/price data
        # Order book updates can be optionally processed here
        pass

    async def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Generate mean reversion signal.

        Args:
            symbol: Trading symbol

        Returns:
            Trading signal or None
        """
        if len(self.price_history[symbol]) < self.lookback_period:
            return None

        prices = [p["price"] for p in self.price_history[symbol]]
        prices_array = np.array(prices)

        # Calculate z-score
        mean = np.mean(prices_array)
        std = np.std(prices_array)
        current_price = prices[-1]

        if std == 0:
            return None

        z_score = (current_price - mean) / std

        # Check current position
        current_position = self.positions.get(symbol, 0.0)

        # Generate signal based on z-score
        if z_score > self.z_score_threshold and current_position >= 0:
            # Price too high, sell signal
            return {
                "symbol": symbol,
                "action": "sell",
                "reason": "mean_reversion",
                "z_score": z_score,
                "current_price": current_price,
                "mean_price": mean,
                "std": std,
                "confidence": min(abs(z_score) / self.z_score_threshold, 1.0),
            }
        elif z_score < -self.z_score_threshold and current_position <= 0:
            # Price too low, buy signal
            return {
                "symbol": symbol,
                "action": "buy",
                "reason": "mean_reversion",
                "z_score": z_score,
                "current_price": current_price,
                "mean_price": mean,
                "std": std,
                "confidence": min(abs(z_score) / self.z_score_threshold, 1.0),
            }
        elif abs(z_score) < 0.5 and current_position != 0:
            # Price returned to mean, close position
            return {
                "symbol": symbol,
                "action": "close",
                "reason": "mean_reversion_close",
                "z_score": z_score,
                "current_price": current_price,
                "mean_price": mean,
                "std": std,
                "confidence": 1.0,
            }

        return None

    async def _execute_signal(self, signal: Dict[str, Any]) -> None:
        """
        Execute trading signal (placeholder for actual execution).

        Args:
            signal: Trading signal
        """
        symbol = signal["symbol"]
        action = signal["action"]
        current_price = signal["current_price"]

        logger.info(
            f"Signal generated - {action.upper()} {symbol} at {current_price:.2f} "
            f"(z-score: {signal['z_score']:.2f})"
        )

        # Update positions (simulation)
        if action == "buy":
            self.positions[symbol] = self.positions.get(symbol, 0.0) + 1.0
            self.entry_prices[symbol] = current_price
            logger.info(f"Opened long position in {symbol} at {current_price:.2f}")

        elif action == "sell":
            self.positions[symbol] = self.positions.get(symbol, 0.0) - 1.0
            self.entry_prices[symbol] = current_price
            logger.info(f"Opened short position in {symbol} at {current_price:.2f}")

        elif action == "close":
            # Close position and calculate P&L
            position = self.positions.get(symbol, 0.0)
            if position == 0:
                return

            entry_price = self.entry_prices.get(symbol, current_price)
            if position > 0:
                # Close long
                pnl = (current_price - entry_price) * abs(position)
            else:
                # Close short
                pnl = (entry_price - current_price) * abs(position)

            # Log trade
            await self.log_trade(
                {
                    "symbol": symbol,
                    "side": "long" if position > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "price": current_price,
                    "quantity": abs(position),
                    "pnl": pnl,
                }
            )

            # Reset position
            self.positions[symbol] = 0.0
            self.entry_prices.pop(symbol, None)
            logger.info(f"Closed position in {symbol} at {current_price:.2f}, P&L: ${pnl:.2f}")

    def get_current_positions(self) -> Dict[str, float]:
        """
        Get current positions.

        Returns:
            Dictionary of symbol to position size
        """
        return self.positions.copy()

    async def close_all_positions(self) -> None:
        """Close all open positions."""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            if self.positions[symbol] != 0:
                # Generate close signal
                prices = [p["price"] for p in self.price_history[symbol]]
                if prices:
                    current_price = prices[-1]
                    signal = {
                        "symbol": symbol,
                        "action": "close",
                        "reason": "force_close",
                        "z_score": 0.0,
                        "current_price": current_price,
                        "mean_price": current_price,
                        "std": 0.0,
                        "confidence": 1.0,
                    }
                    await self._execute_signal(signal)
