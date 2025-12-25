"""Trading engine for order execution."""

import ccxt
import asyncio
from decimal import Decimal
from datetime import datetime
from loguru import logger
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str
    amount: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = ""


@dataclass
class Order:
    """Represents a trade order."""
    id: str
    symbol: str
    side: str
    amount: float
    price: float
    order_type: str = "market"
    status: str = "pending"
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = ""
    exchange_order_id: Optional[str] = None


class TradingEngine:
    """Main trading engine for executing trades."""

    def __init__(
        self,
        trading_config: Dict[str, Any],
        exchanges_config: Dict[str, Any],
        knowledge_graph,
        agent_orchestrator
    ):
        """Initialize trading engine.

        Args:
            trading_config: Trading configuration
            exchanges_config: Exchange configurations
            knowledge_graph: Knowledge graph instance
            agent_orchestrator: Agent orchestrator instance
        """
        self.config = trading_config
        self.exchanges_config = exchanges_config
        self.kg = knowledge_graph
        self.agents = agent_orchestrator
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self._running = False
        self._order_counter = 0

    def start(self) -> None:
        """Start the trading engine."""
        logger.info("Starting trading engine...")
        self._initialize_exchanges()
        self._running = True
        logger.info("Trading engine started")

    def stop(self) -> None:
        """Stop the trading engine."""
        logger.info("Stopping trading engine...")
        self._running = False
        self._close_exchanges()
        logger.info("Trading engine stopped")

    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        for exchange_name, config in self.exchanges_config.items():
            if not config.get("enabled", False):
                continue

            try:
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    "apiKey": config.get("api_key"),
                    "secret": config.get("api_secret"),
                    "sandbox": config.get("sandbox", False),
                    "enableRateLimit": True,
                })

                # Test connection
                if config.get("test_mode", True):
                    exchange.set_sandbox_mode(True)

                self.exchanges[exchange_name] = exchange
                logger.info("Initialized exchange: {}", exchange_name)

            except Exception as e:
                logger.error("Failed to initialize exchange {}: {}", exchange_name, e)

    def _close_exchanges(self) -> None:
        """Close exchange connections."""
        for name, exchange in self.exchanges.items():
            try:
                exchange.close()
                logger.info("Closed exchange: {}", name)
            except Exception as e:
                logger.warning("Error closing exchange {}: {}", name, e)

    def execute_trade(
        self,
        symbol: str,
        side: str,
        amount: float,
        exchange_name: str = "binance",
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute a trade.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            side: Trade side ("buy" or "sell")
            amount: Amount to trade
            exchange_name: Exchange to use (default: "binance")
            order_type: Order type ("market" or "limit")
            price: Price for limit orders (required for limit orders)

        Returns:
            Trade result dictionary with order details
        """
        try:
            # Validate inputs
            if side not in ["buy", "sell"]:
                return {"status": "error", "message": f"Invalid side: {side}"}

            if amount <= 0:
                return {"status": "error", "message": f"Invalid amount: {amount}"}

            if order_type == "limit" and price is None:
                return {"status": "error", "message": "Price required for limit orders"}

            # Check if exchange is available
            if exchange_name not in self.exchanges:
                return {"status": "error", "message": f"Exchange {exchange_name} not initialized"}

            exchange = self.exchanges[exchange_name]

            # Risk management checks
            risk_check = self._check_risk_limits(symbol, side, amount, exchange_name)
            if risk_check["allowed"] is False:
                logger.warning("Trade rejected by risk management: {}", risk_check["reason"])
                return {"status": "rejected", "reason": risk_check["reason"]}

            # Get current market price
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker["last"]

            # Execute order on exchange
            logger.info("Executing {} order: {} {} @ {} on {}", side, amount, symbol, current_price, exchange_name)

            if order_type == "market":
                order = exchange.create_market_order(symbol, side, amount)
            else:
                order = exchange.create_limit_order(symbol, side, amount, price)

            # Create order record
            self._order_counter += 1
            order_record = Order(
                id=f"ORD-{self._order_counter}",
                symbol=symbol,
                side=side,
                amount=amount,
                price=order.get("average") or order.get("price") or current_price,
                order_type=order_type,
                status=order["status"],
                exchange=exchange_name,
                exchange_order_id=str(order.get("id", ""))
            )
            self.orders.append(order_record)

            # Update or create position
            self._update_position(order_record, current_price)

            # Store in knowledge graph
            self._store_trade_in_kg(order_record, order)

            logger.info("Order executed successfully: {} - {}", order_record.id, order["status"])

            return {
                "status": "executed",
                "order_id": order_record.id,
                "exchange_order_id": order.get("id"),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": order_record.price,
                "exchange": exchange_name,
                "timestamp": order_record.timestamp.isoformat()
            }

        except ccxt.InsufficientFunds as e:
            logger.error("Insufficient funds for trade: {}", e)
            return {"status": "error", "message": "Insufficient funds"}
        except ccxt.NetworkError as e:
            logger.error("Network error during trade: {}", e)
            return {"status": "error", "message": "Network error"}
        except Exception as e:
            logger.error("Error executing trade: {}", e)
            return {"status": "error", "message": str(e)}

    def _check_risk_limits(
        self,
        symbol: str,
        side: str,
        amount: float,
        exchange_name: str
    ) -> Dict[str, Any]:
        """Check if trade passes risk management rules.

        Args:
            symbol: Trading pair symbol
            side: Trade side
            amount: Amount to trade
            exchange_name: Exchange name

        Returns:
            Dictionary with 'allowed' boolean and 'reason' if not allowed
        """
        # Check minimum trade amount
        min_trade_amount = self.config.get("min_trade_amount", 10)
        try:
            ticker = self.exchanges[exchange_name].fetch_ticker(symbol)
            estimated_value = amount * ticker["last"]
            if estimated_value < min_trade_amount:
                return {
                    "allowed": False,
                    "reason": f"Trade value ${estimated_value:.2f} below minimum ${min_trade_amount}"
                }
        except Exception:
            pass

        # Check maximum open positions
        max_open_positions = self.config.get("max_open_positions", 5)
        if len(self.positions) >= max_open_positions:
            # Allow closing positions but not opening new ones
            position_key = f"{symbol}_{exchange_name}"
            if position_key not in self.positions or self.positions[position_key].side != side:
                return {
                    "allowed": False,
                    "reason": f"Maximum open positions ({max_open_positions}) reached"
                }

        # Check position size
        max_position_size = self.config.get("max_position_size", 0.1)
        position_key = f"{symbol}_{exchange_name}"
        if position_key in self.positions:
            current_position = self.positions[position_key]
            new_total = current_position.amount + amount if side == current_position.side else abs(current_position.amount - amount)
            if new_total * current_position.entry_price > max_position_size * 100000:  # Assume $100k portfolio
                return {
                    "allowed": False,
                    "reason": f"Position size exceeds maximum {max_position_size * 100}% of portfolio"
                }

        return {"allowed": True}

    def _update_position(self, order: Order, current_price: float) -> None:
        """Update or create position based on order.

        Args:
            order: Order that was executed
            current_price: Current market price
        """
        position_key = f"{order.symbol}_{order.exchange}"

        if position_key in self.positions:
            position = self.positions[position_key]

            # Update existing position
            if position.side == order.side:
                # Adding to position (average price calculation)
                total_value = (position.amount * position.entry_price) + (order.amount * order.price)
                total_amount = position.amount + order.amount
                position.amount = total_amount
                position.entry_price = total_value / total_amount if total_amount > 0 else 0
            else:
                # Reducing or closing position
                position.amount -= order.amount
                if position.amount <= 0:
                    # Position closed
                    del self.positions[position_key]
                    logger.info("Position closed: {}", position_key)
                    return

            position.current_price = current_price
        else:
            # Open new position
            stop_loss_pct = self.config.get("stop_loss_percent", 2.0) / 100
            take_profit_pct = self.config.get("take_profit_percent", 5.0) / 100

            self.positions[position_key] = Position(
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                entry_price=order.price,
                current_price=current_price,
                stop_loss=order.price * (1 - stop_loss_pct) if order.side == "buy" else order.price * (1 + stop_loss_pct),
                take_profit=order.price * (1 + take_profit_pct) if order.side == "buy" else order.price * (1 - take_profit_pct),
                exchange=order.exchange
            )
            logger.info("New position opened: {} - {} {} @ {}", position_key, order.side, order.amount, order.price)

    def _store_trade_in_kg(self, order: Order, exchange_order: Dict[str, Any]) -> None:
        """Store trade information in knowledge graph.

        Args:
            order: Order record
            exchange_order: Raw exchange order data
        """
        try:
            if self.kg is None:
                return

            # Create trade node in Neo4j
            cypher = """
            MERGE (t:Trade {id: $order_id})
            SET t.symbol = $symbol,
                t.side = $side,
                t.amount = $amount,
                t.price = $price,
                t.order_type = $order_type,
                t.status = $status,
                t.exchange = $exchange,
                t.exchange_order_id = $exchange_order_id,
                t.timestamp = datetime($timestamp)
            """

            self.kg.write(cypher, {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "amount": order.amount,
                "price": order.price,
                "order_type": order.order_type,
                "status": order.status,
                "exchange": order.exchange,
                "exchange_order_id": order.exchange_order_id,
                "timestamp": order.timestamp.isoformat()
            })

        except Exception as e:
            logger.warning("Failed to store trade in knowledge graph: {}", e)

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions.

        Returns:
            List of position dictionaries
        """
        return [
            {
                "symbol": pos.symbol,
                "side": pos.side,
                "amount": pos.amount,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "pnl": (pos.current_price - pos.entry_price) * pos.amount if pos.side == "buy" else (pos.entry_price - pos.current_price) * pos.amount,
                "pnl_percent": ((pos.current_price - pos.entry_price) / pos.entry_price * 100) if pos.side == "buy" else ((pos.entry_price - pos.current_price) / pos.entry_price * 100),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "exchange": pos.exchange,
                "timestamp": pos.timestamp.isoformat()
            }
            for pos in self.positions.values()
        ]

    def get_orders(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history.

        Args:
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        recent_orders = self.orders[-limit:] if len(self.orders) > limit else self.orders
        return [
            {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "amount": order.amount,
                "price": order.price,
                "order_type": order.order_type,
                "status": order.status,
                "exchange": order.exchange,
                "exchange_order_id": order.exchange_order_id,
                "timestamp": order.timestamp.isoformat()
            }
            for order in recent_orders
        ]

    def check_stop_loss_and_take_profit(self) -> List[Dict[str, Any]]:
        """Check all positions for stop loss and take profit triggers.

        Returns:
            List of triggered actions
        """
        triggered_actions = []

        for position_key, position in list(self.positions.items()):
            try:
                exchange = self.exchanges.get(position.exchange)
                if not exchange:
                    continue

                ticker = exchange.fetch_ticker(position.symbol)
                current_price = ticker["last"]
                position.current_price = current_price

                should_close = False
                reason = ""

                if position.side == "buy":
                    if position.stop_loss and current_price <= position.stop_loss:
                        should_close = True
                        reason = "stop_loss"
                    elif position.take_profit and current_price >= position.take_profit:
                        should_close = True
                        reason = "take_profit"
                else:  # sell
                    if position.stop_loss and current_price >= position.stop_loss:
                        should_close = True
                        reason = "stop_loss"
                    elif position.take_profit and current_price <= position.take_profit:
                        should_close = True
                        reason = "take_profit"

                if should_close:
                    # Close position
                    close_side = "sell" if position.side == "buy" else "buy"
                    result = self.execute_trade(
                        position.symbol,
                        close_side,
                        position.amount,
                        position.exchange
                    )

                    triggered_actions.append({
                        "position": position_key,
                        "action": "close",
                        "reason": reason,
                        "result": result
                    })

                    logger.info("Closed position {} due to {}: {}", position_key, reason, result.get("order_id"))

            except Exception as e:
                logger.error("Error checking position {}: {}", position_key, e)

        return triggered_actions
