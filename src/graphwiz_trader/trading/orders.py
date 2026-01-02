"""Order management system for trading operations."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from loguru import logger


class OrderType(Enum):
    """Order types supported by the trading system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order lifecycle statuses."""
    PENDING = "pending"           # Order created, not yet submitted to exchange
    OPEN = "open"                 # Order submitted and active on exchange
    PARTIALLY_FILLED = "partially_filled"  # Order partially filled
    FILLED = "filled"             # Order completely filled
    CANCELLED = "cancelled"       # Order cancelled
    REJECTED = "rejected"         # Order rejected by exchange
    EXPIRED = "expired"           # Order expired
    FAILED = "failed"             # Order failed due to system error


class OrderValidationError(Exception):
    """Raised when order validation fails."""
    pass


class OrderExecutionError(Exception):
    """Raised when order execution fails."""
    pass


class Order:
    """Represents a single trading order."""

    def __init__(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str],
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        exchange: Optional[str] = None,
        client_order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize an order.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            side: Order side (buy or sell)
            order_type: Type of order (market, limit, etc.)
            amount: Order amount in base currency
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            exchange: Exchange to execute on
            client_order_id: Custom client order ID
            metadata: Additional order metadata
        """
        # Normalize enums
        if isinstance(side, str):
            side = OrderSide(side.lower())
        if isinstance(order_type, str):
            order_type = OrderType(order_type.lower())

        self.order_id = client_order_id or str(uuid4())
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.amount = Decimal(str(amount))
        self.price = Decimal(str(price)) if price is not None else None
        self.stop_price = Decimal(str(stop_price)) if stop_price is not None else None
        self.exchange = exchange
        self.status = OrderStatus.PENDING
        self.filled_amount = Decimal("0")
        self.avg_fill_price: Optional[Decimal] = None
        self.fees: Dict[str, Decimal] = {}
        self.timestamp = datetime.now(timezone.utc)
        self.updated_timestamp = self.timestamp
        self.exchange_order_id: Optional[str] = None
        self.metadata = metadata or {}
        self.error_message: Optional[str] = None
        self.retry_count = 0
        self.max_retries = 3

        # Validation will be done by OrderManager

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "amount": float(self.amount),
            "price": float(self.price) if self.price else None,
            "stop_price": float(self.stop_price) if self.stop_price else None,
            "exchange": self.exchange,
            "status": self.status.value,
            "filled_amount": float(self.filled_amount),
            "avg_fill_price": float(self.avg_fill_price) if self.avg_fill_price else None,
            "fees": {k: float(v) for k, v in self.fees.items()},
            "timestamp": self.timestamp.isoformat(),
            "updated_timestamp": self.updated_timestamp.isoformat(),
            "exchange_order_id": self.exchange_order_id,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "retry_count": self.retry_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """Create order from dictionary representation."""
        order = cls(
            symbol=data["symbol"],
            side=data["side"],
            order_type=data["order_type"],
            amount=data["amount"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            exchange=data.get("exchange"),
            client_order_id=data["order_id"],
            metadata=data.get("metadata", {})
        )
        order.status = OrderStatus(data["status"])
        order.filled_amount = Decimal(str(data["filled_amount"]))
        order.avg_fill_price = Decimal(str(data["avg_fill_price"])) if data.get("avg_fill_price") else None
        order.fees = {k: Decimal(str(v)) for k, v in data.get("fees", {}).items()}
        order.timestamp = datetime.fromisoformat(data["timestamp"])
        order.updated_timestamp = datetime.fromisoformat(data["updated_timestamp"])
        order.exchange_order_id = data.get("exchange_order_id")
        order.error_message = data.get("error_message")
        order.retry_count = data.get("retry_count", 0)
        return order

    def __repr__(self) -> str:
        return (f"Order(id={self.order_id}, symbol={self.symbol}, side={self.side.value}, "
                f"type={self.order_type.value}, amount={self.amount}, status={self.status.value})")


class OrderManager:
    """Manages order creation, validation, modification, and cancellation."""

    def __init__(
        self,
        min_order_amount: float = 0.001,
        max_order_amount: float = 1000000,
        max_price_deviation: float = 0.5,  # 50% max deviation from current price
        price_precision: int = 8,
        amount_precision: int = 8
    ):
        """Initialize OrderManager.

        Args:
            min_order_amount: Minimum order amount allowed
            max_order_amount: Maximum order amount allowed
            max_price_deviation: Maximum price deviation from market (as percentage)
            price_precision: Decimal places for price
            amount_precision: Decimal places for amount
        """
        self.min_order_amount = Decimal(str(min_order_amount))
        self.max_order_amount = Decimal(str(max_order_amount))
        self.max_price_deviation = Decimal(str(max_price_deviation))
        self.price_precision = price_precision
        self.amount_precision = amount_precision
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

    def create_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str],
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        exchange: Optional[str] = None,
        validate: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """Create a new order.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy or sell)
            order_type: Type of order
            amount: Order amount
            price: Limit price (if applicable)
            stop_price: Stop price (if applicable)
            exchange: Exchange to execute on
            validate: Whether to validate the order
            metadata: Additional metadata

        Returns:
            Created Order object

        Raises:
            OrderValidationError: If validation fails
        """
        try:
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
                stop_price=stop_price,
                exchange=exchange,
                metadata=metadata
            )

            if validate:
                self.validate_order(order)

            self.orders[order.order_id] = order
            logger.info(
                "Created order: {} {} {} {} @ {}",
                order.side.value, order.amount, order.symbol,
                order.order_type.value,
                order.price if order.price else "market"
            )

            return order

        except (InvalidOperation, ValueError) as e:
            logger.error("Failed to create order: {}", str(e))
            raise OrderValidationError(f"Invalid order parameters: {str(e)}")

    def validate_order(
        self,
        order: Order,
        current_price: Optional[float] = None
    ) -> bool:
        """Validate an order.

        Args:
            order: Order to validate
            current_price: Current market price (for price deviation check)

        Returns:
            True if valid

        Raises:
            OrderValidationError: If validation fails
        """
        # Check amount
        if order.amount <= 0:
            raise OrderValidationError(f"Order amount must be positive: {order.amount}")

        if order.amount < self.min_order_amount:
            raise OrderValidationError(
                f"Order amount {order.amount} below minimum {self.min_order_amount}"
            )

        if order.amount > self.max_order_amount:
            raise OrderValidationError(
                f"Order amount {order.amount} exceeds maximum {self.max_order_amount}"
            )

        # Check price requirements based on order type
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT,
                                OrderType.TAKE_PROFIT_LIMIT]:
            if order.price is None or order.price <= 0:
                raise OrderValidationError(
                    f"{order.order_type.value} orders require a valid price"
                )

        if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                raise OrderValidationError(
                    f"{order.order_type.value} orders require a valid stop price"
                )

        if order.order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                raise OrderValidationError(
                    f"{order.order_type.value} orders require a valid take profit price"
                )

        # Check price deviation if current price provided
        if current_price is not None and order.price is not None:
            current_price_dec = Decimal(str(current_price))
            deviation = abs(order.price - current_price_dec) / current_price_dec

            if deviation > self.max_price_deviation:
                raise OrderValidationError(
                    f"Order price deviation {deviation:.2%} exceeds maximum "
                    f"{self.max_price_deviation:.2%}"
                )

        # Validate symbol format
        if not self._is_valid_symbol(order.symbol):
            raise OrderValidationError(f"Invalid symbol format: {order.symbol}")

        logger.debug("Order validation passed: {}", order.order_id)
        return True

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Validate trading pair symbol format."""
        # Basic check for symbol format (e.g., BTC/USDT, ETH-BTC)
        parts = symbol.replace("/", "-").split("-")
        return len(parts) == 2 and all(len(p) > 0 for p in parts)

    def modify_order(
        self,
        order_id: str,
        amount: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Modify an existing order.

        Args:
            order_id: Order ID to modify
            amount: New amount
            price: New price
            stop_price: New stop price

        Returns:
            Modified order

        Raises:
            OrderValidationError: If order cannot be modified
        """
        if order_id not in self.orders:
            raise OrderValidationError(f"Order not found: {order_id}")

        order = self.orders[order_id]

        # Only pending orders can be modified
        if order.status != OrderStatus.PENDING:
            raise OrderValidationError(
                f"Cannot modify order with status {order.status.value}"
            )

        # Update fields
        if amount is not None:
            order.amount = Decimal(str(amount))

        if price is not None:
            order.price = Decimal(str(price))

        if stop_price is not None:
            order.stop_price = Decimal(str(stop_price))

        order.updated_timestamp = datetime.now(timezone.utc)

        # Re-validate
        self.validate_order(order)

        logger.info("Modified order: {}", order_id)
        return order

    async def cancel_order(
        self,
        order_id: str,
        exchange_order_id: Optional[str] = None
    ) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel
            exchange_order_id: Exchange order ID (if already submitted)

        Returns:
            True if cancelled successfully

        Raises:
            OrderExecutionError: If cancellation fails
        """
        if order_id not in self.orders:
            logger.warning("Cannot cancel non-existent order: {}", order_id)
            return False

        order = self.orders[order_id]

        # Can only cancel open or pending orders
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED,
                           OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            logger.warning(
                "Cannot cancel order with status: {}", order.status.value
            )
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_timestamp = datetime.now(timezone.utc)

        logger.info("Cancelled order: {}", order_id)
        return True

    def update_order_status(
        self,
        order_id: str,
        status: Union[OrderStatus, str],
        filled_amount: Optional[float] = None,
        avg_fill_price: Optional[float] = None,
        fees: Optional[Dict[str, float]] = None,
        exchange_order_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Update order status from exchange response.

        Args:
            order_id: Order ID to update
            status: New order status
            filled_amount: Amount filled so far
            avg_fill_price: Average fill price
            fees: Fees paid
            exchange_order_id: Exchange order ID
            error_message: Error message if failed
        """
        if order_id not in self.orders:
            logger.error("Cannot update non-existent order: {}", order_id)
            return

        order = self.orders[order_id]

        if isinstance(status, str):
            status = OrderStatus(status.lower())

        order.status = status
        order.updated_timestamp = datetime.now(timezone.utc)

        if filled_amount is not None:
            order.filled_amount = Decimal(str(filled_amount))

        if avg_fill_price is not None:
            order.avg_fill_price = Decimal(str(avg_fill_price))

        if fees:
            order.fees = {k: Decimal(str(v)) for k, v in fees.items()}

        if exchange_order_id:
            order.exchange_order_id = exchange_order_id

        if error_message:
            order.error_message = error_message

        logger.debug(
            "Updated order {}: status={}, filled={}/{}",
            order_id, status.value, order.filled_amount, order.amount
        )

        # Move to history if terminal state
        if status in [OrderStatus.FILLED, OrderStatus.CANCELLED,
                     OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED]:
            self._archive_order(order_id)

    def _archive_order(self, order_id: str) -> None:
        """Archive order to history."""
        if order_id in self.orders:
            order = self.orders.pop(order_id)
            self.order_history.append(order)
            logger.debug("Archived order: {}", order_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all active orders for a symbol."""
        return [o for o in self.orders.values() if o.symbol == symbol]

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with a specific status."""
        return [o for o in self.orders.values() if o.status == status]

    def get_active_orders(self) -> List[Order]:
        """Get all active orders (pending, open, partially filled)."""
        active_statuses = [
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED
        ]
        return [o for o in self.orders.values() if o.status in active_statuses]

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of orders to return

        Returns:
            List of historical orders
        """
        history = self.order_history

        if symbol:
            history = [o for o in history if o.symbol == symbol]

        # Sort by timestamp descending
        history = sorted(history, key=lambda x: x.timestamp, reverse=True)

        return history[:limit]

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all active orders.

        Args:
            symbol: Only cancel orders for this symbol (optional)

        Returns:
            Number of orders cancelled
        """
        active_orders = self.get_active_orders()

        if symbol:
            active_orders = [o for o in active_orders if o.symbol == symbol]

        cancelled_count = 0
        for order in active_orders:
            if asyncio.iscoroutinefunction(self.cancel_order):
                # If called from async context
                asyncio.create_task(self.cancel_order(order.order_id))
            else:
                # Sync fallback
                self.cancel_order(order.order_id)
            cancelled_count += 1

        logger.info("Cancelled {} orders", cancelled_count)
        return cancelled_count

    def calculate_total_fees(self, order_id: str) -> Decimal:
        """Calculate total fees for an order."""
        order = self.get_order(order_id)
        if not order:
            order = next((o for o in self.order_history if o.order_id == order_id), None)

        if order:
            return sum(order.fees.values()) if order.fees else Decimal("0")

        return Decimal("0")

    def get_statistics(self) -> Dict[str, Any]:
        """Get order statistics.

        Returns:
            Dictionary with order statistics
        """
        total_orders = len(self.orders) + len(self.order_history)

        status_counts = {}
        for status in OrderStatus:
            count = len([o for o in self.order_history if o.status == status])
            status_counts[status.value] = count

        active_count = len(self.get_active_orders())

        return {
            "total_orders": total_orders,
            "active_orders": active_count,
            "archived_orders": len(self.order_history),
            "status_breakdown": status_counts
        }
