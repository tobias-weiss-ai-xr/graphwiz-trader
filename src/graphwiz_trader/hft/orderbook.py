"""
Order Book Management Module.

Manages order books from multiple exchanges for arbitrage detection.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class OrderBookManager:
    """Manages multiple exchange order books for arbitrage."""

    def __init__(self, max_depth: int = 20) -> None:
        """
        Initialize order book manager.

        Args:
            max_depth: Maximum depth of order book to maintain
        """
        self.books: Dict[str, Dict[str, "OrderBook"]] = defaultdict(dict)
        self.max_depth = max_depth
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def update(self, exchange: str, symbol: str, orderbook: Dict[str, Any]) -> None:
        """
        Update local order book.

        Args:
            exchange: Exchange identifier
            symbol: Trading symbol
            orderbook: Order book data with 'bids', 'asks', 'timestamp'
        """
        if exchange not in self.books[symbol]:
            self.books[symbol][exchange] = OrderBook(symbol, self.max_depth)

        self.books[symbol][exchange].update(orderbook)

        # Store price history
        if orderbook.get("bids") and orderbook.get("asks"):
            self.price_history[f"{exchange}:{symbol}"].append(
                {
                    "timestamp": orderbook.get("timestamp"),
                    "bid": orderbook["bids"][0][0] if orderbook["bids"] else 0,
                    "ask": orderbook["asks"][0][0] if orderbook["asks"] else 0,
                }
            )

    def get_arbitrage_opportunities(
        self, symbol: str, min_profit_bps: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Find cross-exchange arbitrage opportunities.

        Args:
            symbol: Trading symbol
            min_profit_bps: Minimum profit in basis points

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        if len(self.books.get(symbol, {})) < 2:
            return opportunities

        exchanges = list(self.books[symbol].keys())
        for i, exchange1 in enumerate(exchanges):
            for exchange2 in exchanges[i + 1 :]:
                book1 = self.books[symbol][exchange1]
                book2 = self.books[symbol][exchange2]

                if not book1.best_bid or not book2.best_ask:
                    continue

                # Buy on exchange2, sell on exchange1
                profit1 = (book1.best_bid - book2.best_ask) / book2.best_ask * 10000

                if profit1 >= min_profit_bps:
                    opportunities.append(
                        {
                            "symbol": symbol,
                            "buy_exchange": exchange2,
                            "sell_exchange": exchange1,
                            "buy_price": book2.best_ask,
                            "sell_price": book1.best_bid,
                            "profit_bps": profit1,
                            "type": "cross_exchange",
                            "timestamp": book1.last_update_timestamp,
                        }
                    )

                # Buy on exchange1, sell on exchange2
                if not book2.best_bid or not book1.best_ask:
                    continue

                profit2 = (book2.best_bid - book1.best_ask) / book1.best_ask * 10000

                if profit2 >= min_profit_bps:
                    opportunities.append(
                        {
                            "symbol": symbol,
                            "buy_exchange": exchange1,
                            "sell_exchange": exchange2,
                            "buy_price": book1.best_ask,
                            "sell_price": book2.best_bid,
                            "profit_bps": profit2,
                            "type": "cross_exchange",
                            "timestamp": book2.last_update_timestamp,
                        }
                    )

        return opportunities

    def get_triangular_arbitrage(
        self, base_currency: str, quote_currency: str = "USDT"
    ) -> List[Dict[str, Any]]:
        """
        Find triangular arbitrage opportunities.

        Example: BTC -> ETH -> USDT -> BTC

        Args:
            base_currency: Base currency (e.g., 'BTC')
            quote_currency: Quote currency (e.g., 'USDT')

        Returns:
            List of triangular arbitrage opportunities
        """
        opportunities = []

        # This is a complex calculation requiring multiple trading pairs
        # Implementation would need to:
        # 1. Find all available trading pairs with base_currency
        # 2. Calculate potential profit paths
        # 3. Account for trading fees
        # 4. Consider liquidity at each step

        logger.info(
            f"Triangular arbitrage detection for {base_currency}/{quote_currency} not yet implemented"
        )

        return opportunities

    def get_book(self, exchange: str, symbol: str) -> Optional["OrderBook"]:
        """
        Get order book for specific exchange and symbol.

        Args:
            exchange: Exchange identifier
            symbol: Trading symbol

        Returns:
            OrderBook instance or None if not found
        """
        return self.books.get(symbol, {}).get(exchange)

    def get_all_books(self, symbol: str) -> Dict[str, "OrderBook"]:
        """
        Get all order books for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary of exchange to OrderBook
        """
        return self.books.get(symbol, {})

    def get_price_history(self, exchange: str, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get price history for exchange and symbol.

        Args:
            exchange: Exchange identifier
            symbol: Trading symbol
            limit: Number of historical records to return

        Returns:
            List of price history records
        """
        key = f"{exchange}:{symbol}"
        history = list(self.price_history.get(key, []))
        return history[-limit:] if limit else history


class OrderBook:
    """Single exchange order book."""

    def __init__(self, symbol: str, max_depth: int = 20) -> None:
        """
        Initialize order book.

        Args:
            symbol: Trading symbol
            max_depth: Maximum depth to maintain
        """
        self.symbol = symbol
        self.max_depth = max_depth
        self.bids: List[Tuple[float, float]] = []  # (price, volume)
        self.asks: List[Tuple[float, float]] = []
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.last_update_timestamp: Optional[int] = None

    def update(self, orderbook: Dict[str, Any]) -> None:
        """
        Update order book with new data.

        Args:
            orderbook: Order book data with 'bids', 'asks', 'timestamp'
        """
        # Sort and limit bids (descending by price)
        raw_bids = orderbook.get("bids", [])
        self.bids = sorted(raw_bids[: self.max_depth], key=lambda x: x[0], reverse=True)

        # Sort and limit asks (ascending by price)
        raw_asks = orderbook.get("asks", [])
        self.asks = sorted(raw_asks[: self.max_depth], key=lambda x: x[0])

        # Update best prices
        if self.bids:
            self.best_bid = float(self.bids[0][0])
        else:
            self.best_bid = None

        if self.asks:
            self.best_ask = float(self.asks[0][0])
        else:
            self.best_ask = None

        self.last_update_timestamp = orderbook.get("timestamp")

    @property
    def spread(self) -> Optional[float]:
        """
        Get bid-ask spread.

        Returns:
            Spread as decimal or None if prices unavailable
        """
        if self.best_bid and self.best_ask:
            return (self.best_ask - self.best_bid) / self.best_ask
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """
        Get spread in basis points.

        Returns:
            Spread in basis points or None if prices unavailable
        """
        spread = self.spread
        return spread * 10000 if spread else None

    @property
    def mid_price(self) -> Optional[float]:
        """
        Get mid price.

        Returns:
            Mid price or None if prices unavailable
        """
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def imbalance(self) -> float:
        """
        Get order book imbalance.

        Positive values indicate more buying pressure (bid-heavy),
        negative values indicate more selling pressure (ask-heavy).

        Returns:
            Imbalance ratio between -1 and 1
        """
        bid_volume = sum(float(v) for _, v in self.bids[:10])
        ask_volume = sum(float(v) for _, v in self.asks[:10])

        if bid_volume + ask_volume == 0:
            return 0.0

        return (bid_volume - ask_volume) / (bid_volume + ask_volume)

    def get_vwap(self, side: str, quantity: float) -> Optional[float]:
        """
        Calculate volume-weighted average price for a given quantity.

        Args:
            side: 'buy' or 'sell'
            quantity: Quantity to calculate VWAP for

        Returns:
            VWAP or None if insufficient liquidity
        """
        levels = self.asks if side == "buy" else self.bids
        if not levels:
            return None

        total_cost = 0.0
        filled_quantity = 0.0

        for price, volume in levels:
            if filled_quantity >= quantity:
                break

            fill_qty = min(float(volume), quantity - filled_quantity)
            total_cost += float(price) * fill_qty
            filled_quantity += fill_qty

        if filled_quantity < quantity:
            # Insufficient liquidity
            return None

        return total_cost / filled_quantity if filled_quantity > 0 else None

    def get_liquidity(self, side: str, depth: int = 10) -> float:
        """
        Get total liquidity at a given depth.

        Args:
            side: 'buy' or 'sell'
            depth: Number of levels to consider

        Returns:
            Total volume available
        """
        levels = self.asks if side == "buy" else self.bids
        return sum(float(v) for _, v in levels[:depth])

    def __repr__(self) -> str:
        """String representation of order book."""
        return (
            f"OrderBook(symbol={self.symbol}, "
            f"best_bid={self.best_bid}, "
            f"best_ask={self.best_ask}, "
            f"spread_bps={self.spread_bps:.2f if self.spread_bps else None})"
        )
