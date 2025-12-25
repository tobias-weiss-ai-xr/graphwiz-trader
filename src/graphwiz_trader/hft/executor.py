"""
Fast Order Execution Module.

Provides ultra-low latency order execution across multiple exchanges.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import ccxt.pro as ccxtpro
except ImportError:
    logger.warning("ccxt.pro not available, using standard ccxt")
    import ccxt as ccxtpro


class FastOrderExecutor:
    """Ultra-low latency order execution."""

    def __init__(self, exchanges: Dict[str, Any]) -> None:
        """
        Initialize fast order executor.

        Args:
            exchanges: Dictionary of exchange configurations
        """
        self.exchanges: Dict[str, Any] = {}
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {}
        self.order_history: List[Dict[str, Any]] = []

        for exchange_id, config in exchanges.items():
            if config.get("enabled"):
                try:
                    exchange_class = getattr(ccxtpro, exchange_id)
                    self.exchanges[exchange_id] = exchange_class(
                        {
                            "apiKey": config.get("api_key"),
                            "secret": config.get("api_secret"),
                            "enableRateLimit": True,
                            "options": {
                                "defaultType": config.get("default_type", "spot"),
                            },
                        }
                    )
                    # Rate limiter to prevent API bans
                    self.rate_limiters[exchange_id] = asyncio.Semaphore(
                        config.get("rate_limit", 10)
                    )
                    logger.info(f"Initialized executor for {exchange_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize {exchange_id}: {e}")

    async def place_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Place order with minimal latency.

        Args:
            exchange: Exchange identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price (required for limit orders)
            order_type: 'market' or 'limit'
            params: Additional exchange-specific parameters

        Returns:
            Order result dictionary

        Raises:
            ValueError: If exchange not available
            Exception: If order fails
        """
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange {exchange} not available")

        rate_limiter = self.rate_limiters[exchange]
        exc = self.exchanges[exchange]

        async with rate_limiter:
            start_time = time.perf_counter()

            try:
                if order_type == "market":
                    order = await exc.create_market_order(
                        symbol, side, amount, params=params
                    )
                elif order_type == "limit":
                    if price is None:
                        raise ValueError("Price required for limit orders")
                    order = await exc.create_limit_order(
                        symbol, side, amount, price, params=params
                    )
                else:
                    raise ValueError(f"Unsupported order type: {order_type}")

                latency_ms = (time.perf_counter() - start_time) * 1000

                result = {
                    "order_id": order["id"],
                    "exchange": exchange,
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "price": price or order.get("price"),
                    "type": order_type,
                    "status": order["status"],
                    "latency_ms": latency_ms,
                    "timestamp": order.get("timestamp"),
                    "filled": order.get("filled", 0),
                    "remaining": order.get("remaining", amount),
                    "cost": order.get("cost", 0),
                }

                logger.info(
                    f"Order placed in {latency_ms:.2f}ms: {side.upper()} {amount} {symbol} "
                    f"on {exchange} (ID: {order['id']})"
                )

                # Store in history
                self.order_history.append(result)

                return result

            except Exception as e:
                logger.error(f"Order failed on {exchange}: {e}")
                raise

    async def place_simultaneous_orders(
        self, orders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Place multiple orders simultaneously for arbitrage.

        Args:
            orders: List of order dictionaries

        Returns:
            List of order results (or exceptions)
        """
        tasks = [
            self.place_order(
                order["exchange"],
                order["symbol"],
                order["side"],
                order["amount"],
                order.get("price"),
                order.get("type", "market"),
                order.get("params"),
            )
            for order in orders
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def cancel_order(
        self, exchange: str, order_id: str, symbol: str
    ) -> bool:
        """
        Cancel order immediately.

        Args:
            exchange: Exchange identifier
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            True if cancelled successfully, False otherwise
        """
        exc = self.exchanges.get(exchange)
        if not exc:
            logger.error(f"Exchange {exchange} not available")
            return False

        try:
            await exc.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id} on {exchange}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on {exchange}: {e}")
            return False

    async def cancel_all_orders(
        self, exchange: str, symbol: Optional[str] = None
    ) -> int:
        """
        Cancel all open orders.

        Args:
            exchange: Exchange identifier
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        exc = self.exchanges.get(exchange)
        if not exc:
            logger.error(f"Exchange {exchange} not available")
            return 0

        try:
            open_orders = await exc.fetch_open_orders(symbol)
            cancelled_count = 0

            for order in open_orders:
                try:
                    await exc.cancel_order(order["id"], order["symbol"])
                    cancelled_count += 1
                except Exception as e:
                    logger.error(f"Failed to cancel order {order['id']}: {e}")

            logger.info(
                f"Cancelled {cancelled_count}/{len(open_orders)} orders on {exchange}"
            )
            return cancelled_count

        except Exception as e:
            logger.error(f"Failed to cancel orders on {exchange}: {e}")
            return 0

    async def get_open_orders(
        self, exchange: str, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Args:
            exchange: Exchange identifier
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        exc = self.exchanges.get(exchange)
        if not exc:
            logger.error(f"Exchange {exchange} not available")
            return []

        try:
            orders = await exc.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch open orders from {exchange}: {e}")
            return []

    async def get_order_status(
        self, exchange: str, order_id: str, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get order status.

        Args:
            exchange: Exchange identifier
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            Order status dictionary or None
        """
        exc = self.exchanges.get(exchange)
        if not exc:
            logger.error(f"Exchange {exchange} not available")
            return None

        try:
            order = await exc.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id} from {exchange}: {e}")
            return None

    async def get_balance(
        self, exchange: str, currency: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get account balance.

        Args:
            exchange: Exchange identifier
            currency: Optional currency filter

        Returns:
            Balance dictionary
        """
        exc = self.exchanges.get(exchange)
        if not exc:
            logger.error(f"Exchange {exchange} not available")
            return {}

        try:
            balance = await exc.fetch_balance()
            if currency:
                return {
                    "currency": currency,
                    "free": balance.get("free", {}).get(currency, 0),
                    "used": balance.get("used", {}).get(currency, 0),
                    "total": balance.get("total", {}).get(currency, 0),
                }
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch balance from {exchange}: {e}")
            return {}

    def get_order_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get order execution history.

        Args:
            limit: Optional limit on number of orders to return

        Returns:
            List of order records
        """
        if limit:
            return self.order_history[-limit:]
        return self.order_history.copy()

    def get_average_latency(self, exchange: Optional[str] = None) -> float:
        """
        Get average order execution latency.

        Args:
            exchange: Optional exchange filter

        Returns:
            Average latency in milliseconds
        """
        orders = self.order_history
        if exchange:
            orders = [o for o in orders if o["exchange"] == exchange]

        if not orders:
            return 0.0

        latencies = [o["latency_ms"] for o in orders if "latency_ms" in o]
        return sum(latencies) / len(latencies) if latencies else 0.0

    async def close(self) -> None:
        """Close all exchange connections."""
        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Closed connection to {exchange_id}")
            except Exception as e:
                logger.error(f"Error closing {exchange_id}: {e}")

        self.exchanges.clear()
