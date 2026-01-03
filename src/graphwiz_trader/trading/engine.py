"""Trading engine for order execution with optimizations."""

import ccxt
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from loguru import logger
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import threading

# Import alert system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from graphwiz_trader.alerts import AlertManager
from graphwiz_trader.alerts.config import CONSOLE_ONLY


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
        agent_orchestrator,
        alert_manager: Optional[AlertManager] = None
    ):
        """Initialize trading engine.

        Args:
            trading_config: Trading configuration
            exchanges_config: Exchange configurations
            knowledge_graph: Knowledge graph instance
            agent_orchestrator: Agent orchestrator instance
            alert_manager: Optional alert manager instance
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
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self._last_summary_date = None

        # Performance optimizations
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="trading_engine")
        self._ticker_cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        self._ticker_cache_ttl = 1.0  # 1 second cache for tickers
        self._cache_lock = threading.Lock()
        self._trade_count = 0
        self._total_trade_time = 0.0
        self._metrics_lock = threading.Lock()

        # Initialize alert manager
        if alert_manager is None:
            logger.info("Initializing alert manager with console output only...")
            self.alert_manager = AlertManager(CONSOLE_ONLY)
        else:
            self.alert_manager = alert_manager

        logger.info("✅ Alert Manager initialized in TradingEngine")

    def start(self) -> None:
        """Start the trading engine."""
        logger.info("Starting trading engine...")
        self._initialize_exchanges()
        self._running = True
        logger.info("✅ Trading engine started with optimized performance")

    def stop(self) -> None:
        """Stop the trading engine and cleanup resources."""
        logger.info("Stopping trading engine...")
        self._running = False
        self._close_exchanges()

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        # Log performance metrics
        self._log_performance_metrics()

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
                self.alert_manager.exchange_disconnected(
                    exchange=exchange_name,
                    error=str(e)
                )

    def _close_exchanges(self) -> None:
        """Close exchange connections."""
        for name, exchange in self.exchanges.items():
            try:
                exchange.close()
                logger.info("Closed exchange: {}", name)
            except Exception as e:
                logger.warning("Error closing exchange {}: {}", name, e)

    def _get_ticker(self, exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
        """Get ticker with caching to reduce API calls.

        Args:
            exchange: Exchange instance
            symbol: Trading pair symbol

        Returns:
            Ticker dictionary
        """
        cache_key = f"{exchange.id}:{symbol}"
        current_time = time.time()

        # Check cache
        with self._cache_lock:
            if cache_key in self._ticker_cache:
                ticker, timestamp = self._ticker_cache[cache_key]
                if current_time - timestamp < self._ticker_cache_ttl:
                    logger.debug("Ticker cache hit for {}", cache_key)
                    return ticker

        # Fetch from exchange
        ticker = exchange.fetch_ticker(symbol)

        # Update cache
        with self._cache_lock:
            self._ticker_cache[cache_key] = (ticker, current_time)

        return ticker

    def fetch_tickers_parallel(self, symbols: List[str], exchange_name: str = "binance") -> Dict[str, Dict[str, Any]]:
        """Fetch multiple tickers in parallel using thread pool.

        Args:
            symbols: List of trading pair symbols
            exchange_name: Exchange to use

        Returns:
            Dictionary mapping symbols to tickers
        """
        if exchange_name not in self.exchanges:
            logger.error("Exchange {} not initialized", exchange_name)
            return {}

        exchange = self.exchanges[exchange_name]
        tickers = {}

        # Use thread pool for parallel fetching
        futures = []
        for symbol in symbols:
            future = self._executor.submit(self._get_ticker, exchange, symbol)
            futures.append((symbol, future))

        # Collect results
        for symbol, future in futures:
            try:
                tickers[symbol] = future.result(timeout=5)
            except Exception as e:
                logger.warning("Failed to fetch ticker for {}: {}", symbol, e)
                tickers[symbol] = {}

        return tickers

    def _log_performance_metrics(self) -> None:
        """Log performance metrics."""
        with self._metrics_lock:
            avg_trade_time = (self._total_trade_time / self._trade_count
                            if self._trade_count > 0 else 0)

            logger.info(
                "📊 Trading Engine Performance: {} trades, {:.3f}s total, {:.3f}s avg/trade",
                self._trade_count,
                self._total_trade_time,
                avg_trade_time
            )

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
                self.alert_manager.position_size_warning(
                    exchange=exchange_name,
                    symbol=symbol,
                    current_size=amount,
                    max_size=self.config.get("max_position_size", 0.1)
                )
                return {"status": "rejected", "reason": risk_check["reason"]}

            # Get current market price (with caching)
            ticker = self._get_ticker(exchange, symbol)
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

            # Send trade execution alert
            self.alert_manager.trade_executed(
                exchange=exchange_name,
                symbol=symbol,
                side=side,
                amount=amount,
                price=order_record.price,
                order_id=order_record.id
            )

            # Update daily statistics
            self.daily_trades += 1

            # Update performance metrics
            with self._metrics_lock:
                self._trade_count += 1

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
            self.alert_manager.trade_failed(
                exchange=exchange_name,
                symbol=symbol,
                side=side,
                amount=amount,
                error=str(e)
            )
            return {"status": "error", "message": "Insufficient funds"}
        except ccxt.NetworkError as e:
            logger.error("Network error during trade: {}", e)
            self.alert_manager.exchange_disconnected(
                exchange=exchange_name,
                error=str(e)
            )
            return {"status": "error", "message": "Network error"}
        except Exception as e:
            logger.error("Error executing trade: {}", e)
            self.alert_manager.system_error(
                component="TradingEngine",
                error=f"Trade execution failed: {str(e)}"
            )
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
            ticker = self._get_ticker(self.exchanges[exchange_name], symbol)
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
                # Calculate realized P&L for the closed portion
                if position.side == "buy":
                    realized_pnl = (order.price - position.entry_price) * order.amount
                else:
                    realized_pnl = (position.entry_price - order.price) * order.amount

                position.amount -= order.amount
                if position.amount <= 0:
                    # Position fully closed
                    del self.positions[position_key]
                    logger.info("Position closed: {}", position_key)

                    # Update daily P&L with realized profit/loss
                    self.update_position_pnl(position_key, realized_pnl)
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

                ticker = self._get_ticker(exchange, position.symbol)
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
                    # Send alert before closing
                    if reason == "stop_loss":
                        loss_pct = abs((current_price - position.entry_price) / position.entry_price * 100)
                        self.alert_manager.stop_loss_hit(
                            exchange=position.exchange,
                            symbol=position.symbol,
                            entry_price=position.entry_price,
                            current_price=current_price,
                            loss_pct=loss_pct
                        )
                    elif reason == "take_profit":
                        profit_pct = (current_price - position.entry_price) / position.entry_price * 100
                        self.alert_manager.profit_target(
                            exchange=position.exchange,
                            symbol=position.symbol,
                            entry_price=position.entry_price,
                            current_price=current_price,
                            profit_pct=profit_pct
                        )

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

    def check_daily_loss_limit(self) -> None:
        """Check if daily loss limit has been reached and alert if needed."""
        daily_loss_limit = self.config.get("daily_loss_limit", 150)

        # Calculate total unrealized P&L
        total_unrealized_pnl = 0.0
        for position in self.positions.values():
            if position.side == "buy":
                total_unrealized_pnl += (position.current_price - position.entry_price) * position.amount
            else:
                total_unrealized_pnl += (position.entry_price - position.current_price) * position.amount

        total_pnl = self.daily_pnl + total_unrealized_pnl

        # Check if we've hit the daily loss limit
        if total_pnl <= -daily_loss_limit:
            self.alert_manager.daily_loss_limit(
                current_loss=abs(total_pnl),
                limit=daily_loss_limit,
                exchange=list(self.exchanges.keys())[0] if self.exchanges else "Unknown"
            )

    def send_daily_summary(self) -> None:
        """Send daily trading summary alert."""
        today = datetime.now().date()

        # Only send once per day
        if self._last_summary_date == today:
            return

        # Calculate current P&L from open positions
        total_unrealized_pnl = 0.0
        for position in self.positions.values():
            if position.side == "buy":
                total_unrealized_pnl += (position.current_price - position.entry_price) * position.amount
            else:
                total_unrealized_pnl += (position.entry_price - position.current_price) * position.amount

        total_pnl = self.daily_pnl + total_unrealized_pnl

        # Send daily summary alert
        self.alert_manager.daily_summary(
            date=today.isoformat(),
            trades=self.daily_trades,
            pnl=total_pnl,
            positions=len(self.positions)
        )

        self._last_summary_date = today

        # Reset daily statistics (but keep position tracking)
        self.daily_pnl = total_unrealized_pnl  # Carry over unrealized P&L
        self.daily_trades = 0

    def update_position_pnl(self, position_key: str, realized_pnl: float) -> None:
        """Update daily P&L when a position is closed.

        Args:
            position_key: Position identifier
            realized_pnl: Realized profit/loss from closed position
        """
        self.daily_pnl += realized_pnl

        # Check daily loss limit after each position close
        self.check_daily_loss_limit()

