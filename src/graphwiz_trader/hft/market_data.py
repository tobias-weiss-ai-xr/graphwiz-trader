"""
WebSocket Market Data Module.

Provides real-time market data feeds from cryptocurrency exchanges.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

try:
    import ccxt.pro as ccxtpro
except ImportError:
    logger.warning("ccxt.pro not available, WebSocket features will be limited")
    ccxtpro = None


class WebSocketMarketData:
    """Real-time market data via WebSocket."""

    def __init__(self, exchanges: Dict[str, Any]) -> None:
        """
        Initialize WebSocket market data handler.

        Args:
            exchanges: Dictionary of exchange configurations
        """
        self.exchanges = exchanges
        self.callbacks: Dict[str, Callable] = {}
        self.running = False
        self.exchange_instances: Dict[str, Any] = {}
        self.tasks: List[asyncio.Task] = []

    async def connect(self, exchange_id: str, symbols: List[str]) -> None:
        """
        Connect to exchange WebSocket.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'okx')
            symbols: List of trading symbols to subscribe to
        """
        if not ccxtpro:
            raise RuntimeError("ccxt.pro is required for WebSocket connections")

        exchange_config = self.exchanges.get(exchange_id)
        if not exchange_config:
            raise ValueError(f"Exchange {exchange_id} not configured")

        try:
            # Initialize exchange with Pro support
            exchange_class = getattr(ccxtpro, exchange_id)
            exchange = exchange_class(
                {
                    "apiKey": exchange_config.get("api_key"),
                    "secret": exchange_config.get("api_secret"),
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )

            self.exchange_instances[exchange_id] = exchange

            # Load markets
            await exchange.load_markets()

            # Start streaming tasks for each symbol
            for symbol in symbols:
                if symbol in exchange.markets:
                    # Start ticker stream
                    ticker_task = asyncio.create_task(
                        self._stream_ticker(exchange_id, symbol)
                    )
                    self.tasks.append(ticker_task)

                    # Start order book stream
                    orderbook_task = asyncio.create_task(
                        self._stream_orderbook(exchange_id, symbol)
                    )
                    self.tasks.append(orderbook_task)

                    logger.info(f"Subscribed to {symbol} on {exchange_id}")
                else:
                    logger.warning(f"Symbol {symbol} not found on {exchange_id}")

        except Exception as e:
            logger.error(f"Failed to connect to {exchange_id}: {e}")
            raise

    async def _stream_ticker(self, exchange_id: str, symbol: str) -> None:
        """
        Stream ticker updates.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading symbol
        """
        exchange = self.exchange_instances.get(exchange_id)
        if not exchange:
            return

        while self.running:
            try:
                ticker = await exchange.watch_ticker(symbol)
                await self._process_ticker(exchange_id, symbol, ticker)
            except Exception as e:
                logger.error(f"Error streaming ticker for {symbol} on {exchange_id}: {e}")
                await asyncio.sleep(1)

    async def _stream_orderbook(self, exchange_id: str, symbol: str) -> None:
        """
        Stream order book updates.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading symbol
        """
        exchange = self.exchange_instances.get(exchange_id)
        if not exchange:
            return

        while self.running:
            try:
                orderbook = await exchange.watch_order_book(symbol, limit=20)
                await self._process_orderbook(exchange_id, symbol, orderbook)
            except Exception as e:
                logger.error(f"Error streaming orderbook for {symbol} on {exchange_id}: {e}")
                await asyncio.sleep(1)

    async def _process_ticker(
        self, exchange_id: str, symbol: str, ticker: Dict[str, Any]
    ) -> None:
        """
        Process ticker update.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading symbol
            ticker: Ticker data
        """
        callback = self.callbacks.get("ticker")
        if callback:
            try:
                await callback(
                    {
                        "exchange": exchange_id,
                        "symbol": symbol,
                        "last": ticker.get("last"),
                        "bid": ticker.get("bid"),
                        "ask": ticker.get("ask"),
                        "volume": ticker.get("baseVolume"),
                        "timestamp": ticker.get("timestamp"),
                    }
                )
            except Exception as e:
                logger.error(f"Error in ticker callback: {e}")

    async def _process_orderbook(
        self, exchange_id: str, symbol: str, orderbook: Dict[str, Any]
    ) -> None:
        """
        Process order book update.

        Args:
            exchange_id: Exchange identifier
            symbol: Trading symbol
            orderbook: Order book data
        """
        # Calculate bid-ask spread
        bid = orderbook["bids"][0][0] if orderbook.get("bids") else None
        ask = orderbook["asks"][0][0] if orderbook.get("asks") else None

        if bid and ask:
            spread = (ask - bid) / ask
            spread_bps = spread * 10000

            # Publish to callback
            callback = self.callbacks.get("orderbook")
            if callback:
                try:
                    await callback(
                        {
                            "exchange": exchange_id,
                            "symbol": symbol,
                            "bid": bid,
                            "ask": ask,
                            "spread_bps": spread_bps,
                            "timestamp": orderbook.get("timestamp"),
                            "bid_volume": orderbook["bids"][0][1] if orderbook.get("bids") else 0,
                            "ask_volume": orderbook["asks"][0][1] if orderbook.get("asks") else 0,
                            "bids": orderbook.get("bids", []),
                            "asks": orderbook.get("asks", []),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error in orderbook callback: {e}")

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for events.

        Args:
            event: Event type ('ticker', 'orderbook', etc.)
            callback: Async callback function
        """
        self.callbacks[event] = callback
        logger.info(f"Registered callback for {event} events")

    async def start(self) -> None:
        """Start the market data feed."""
        self.running = True
        logger.info("Market data feed started")

    async def stop(self) -> None:
        """Stop the market data feed and cleanup."""
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

        # Close all exchange connections
        for exchange_id, exchange in self.exchange_instances.items():
            try:
                await exchange.close()
                logger.info(f"Closed connection to {exchange_id}")
            except Exception as e:
                logger.error(f"Error closing {exchange_id}: {e}")

        self.exchange_instances.clear()
        logger.info("Market data feed stopped")
