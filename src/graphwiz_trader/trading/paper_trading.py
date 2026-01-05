"""Paper trading engine with simulated order execution and virtual portfolio."""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import numpy as np

from loguru import logger

from graphwiz_trader.trading.orders import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
)
from graphwiz_trader.trading.portfolio import PortfolioManager


class PaperTradingEngine:
    """Simulated trading engine for paper trading with realistic execution."""

    def __init__(
        self,
        initial_balance: Dict[str, float],
        knowledge_graph,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize paper trading engine.

        Args:
            initial_balance: Initial virtual balance
            knowledge_graph: Knowledge graph instance
            config: Paper trading configuration
        """
        self.config = config or {}

        # Virtual portfolio
        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            risk_per_trade=self.config.get("risk_per_trade", 0.02),
            max_position_size=self.config.get("max_position_size", 0.3),
            max_portfolio_risk=self.config.get("max_portfolio_risk", 0.1),
            stop_loss_pct=self.config.get("stop_loss_pct", 0.05),
            take_profit_pct=self.config.get("take_profit_pct", 0.15),
        )

        self.kg = knowledge_graph

        # Slippage model
        self.slippage_model = self.config.get("slippage_model", "realistic")
        self.base_slippage = Decimal(str(self.config.get("base_slippage", 0.0005)))  # 0.05%

        # Fee structure
        self.maker_fee = Decimal(str(self.config.get("maker_fee", 0.001)))  # 0.1%
        self.taker_fee = Decimal(str(self.config.get("taker_fee", 0.001)))  # 0.1%

        # Execution delay (simulated latency)
        self.execution_delay = self.config.get("execution_delay", 0.1)  # seconds

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

        # Performance tracking
        self.trade_count = 0
        self.start_time = datetime.now(timezone.utc)

        # Market data callback
        self.market_data_callback: Optional[Callable] = None

        logger.info("Paper trading engine initialized with balance: {}", initial_balance)

    async def execute_order(self, order: Order, market_price: float) -> Dict[str, Any]:
        """Execute an order in paper trading mode.

        Args:
            order: Order to execute
            market_price: Current market price

        Returns:
            Execution result
        """
        try:
            # Simulate execution delay
            await asyncio.sleep(self.execution_delay)

            # Calculate realistic fill price with slippage
            fill_price = await self._calculate_fill_price(order, market_price)

            # Calculate fees
            fee = await self._calculate_fees(order, fill_price)

            # Fill order
            order.status = OrderStatus.FILLED
            order.filled_amount = order.amount
            order.avg_fill_price = Decimal(str(fill_price))
            order.updated_timestamp = datetime.now(timezone.utc)

            # Set fee
            fee_currency = order.symbol.split("/")[1]  # Quote currency
            order.fees[fee_currency] = Decimal(str(fee))

            # Update portfolio
            await self._update_portfolio(order, fill_price, fee)

            # Track order
            self.orders[order.order_id] = order
            self.order_history.append(order)
            self.trade_count += 1

            # Create execution result
            result = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "amount": float(order.amount),
                "filled_amount": float(order.filled_amount),
                "requested_price": float(order.price) if order.price else market_price,
                "fill_price": fill_price,
                "market_price": market_price,
                "slippage_pct": ((fill_price - market_price) / market_price) * 100,
                "fees": {fee_currency: float(fee)},
                "status": "filled",
                "timestamp": order.updated_timestamp.isoformat(),
                "execution_time": self.execution_delay,
                "paper_trading": True,
            }

            # Log to knowledge graph
            await self._log_paper_trade(order, result)

            logger.info(
                "Paper trade executed: {} {} {} @ {} (slippage: {:.4}%)",
                order.side.value,
                order.amount,
                order.symbol,
                fill_price,
                result["slippage_pct"],
            )

            return result

        except Exception as e:
            logger.error("Paper trade execution failed: {}", e)
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            raise

    async def _calculate_fill_price(self, order: Order, market_price: float) -> float:
        """Calculate realistic fill price with slippage.

        Args:
            order: Order being executed
            market_price: Current market price

        Returns:
            Fill price with slippage
        """
        market_price_dec = Decimal(str(market_price))

        # Different slippage models
        if self.slippage_model == "realistic":
            # Variable slippage based on order size and randomness
            slippage = await self._calculate_realistic_slippage(order, market_price_dec)
        elif self.slippage_model == "fixed":
            slippage = self.base_slippage
        elif self.slippage_model == "none":
            slippage = Decimal("0")
        else:
            slippage = self.base_slippage

        # Apply slippage based on order side
        if order.side == OrderSide.BUY:
            # Buyers pay more (worse price)
            fill_price = market_price_dec * (1 + slippage)
        else:
            # Sellers receive less (worse price)
            fill_price = market_price_dec * (1 - slippage)

        return float(fill_price)

    async def _calculate_realistic_slippage(self, order: Order, market_price: Decimal) -> Decimal:
        """Calculate realistic variable slippage.

        Args:
            order: Order being executed
            market_price: Current market price

        Returns:
            Slippage as decimal (e.g., 0.001 = 0.1%)
        """
        # Base slippage
        slippage = self.base_slippage

        # Add random component (market impact simulation)
        # Use normal distribution with mean=0, std=0.0002
        random_component = Decimal(str(random.gauss(0, 0.0002)))
        slippage += max(random_component, Decimal("0"))  # Only positive slippage

        # Size-based slippage (larger orders have more slippage)
        order_value = order.amount * market_price
        size_multiplier = min(Decimal(str(order_value)) / Decimal("10000"), Decimal("1.0"))
        slippage += self.base_slippage * size_multiplier

        # Order type adjustment
        if order.order_type == OrderType.MARKET:
            # Market orders have higher slippage
            slippage *= Decimal("1.5")
        elif order.order_type == OrderType.LIMIT:
            # Limit orders may have lower slippage if near market
            if order.price:
                price_diff = abs(order.price - market_price) / market_price
                if price_diff < Decimal("0.001"):  # Within 0.1% of market
                    slippage *= Decimal("0.5")

        return min(slippage, Decimal("0.005"))  # Cap at 0.5%

    async def _calculate_fees(self, order: Order, fill_price: float) -> Decimal:
        """Calculate trading fees.

        Args:
            order: Order being executed
            fill_price: Price at which order was filled

        Returns:
            Fee amount in quote currency
        """
        # Calculate trade value
        trade_value = Decimal(str(order.amount * fill_price))

        # Use taker fee for market orders, maker for limit
        if order.order_type == OrderType.MARKET:
            fee_rate = self.taker_fee
        else:
            fee_rate = self.maker_fee

        return trade_value * fee_rate

    async def _update_portfolio(self, order: Order, fill_price: float, fee: float) -> None:
        """Update virtual portfolio with filled order.

        Args:
            order: Filled order
            fill_price: Fill price
            fee: Fee paid
        """
        try:
            # Update position
            self.portfolio.update_position(
                symbol=order.symbol,
                side=order.side.value,
                amount=float(order.filled_amount),
                price=fill_price,
                fee=fee,
            )

            logger.debug(
                "Virtual portfolio updated: {} {} {} @ {}",
                order.side.value,
                order.filled_amount,
                order.symbol,
                fill_price,
            )

        except Exception as e:
            logger.error("Failed to update virtual portfolio: {}", e)
            raise

    async def _log_paper_trade(self, order: Order, result: Dict[str, Any]) -> None:
        """Log paper trade to knowledge graph.

        Args:
            order: Order that was executed
            result: Execution result
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (pt:PaperTrade {
                        trade_id: $trade_id,
                        timestamp: datetime($timestamp),
                        symbol: $symbol,
                        side: $side,
                        order_type: $order_type,
                        amount: $amount,
                        fill_price: $fill_price,
                        market_price: $market_price,
                        slippage_pct: $slippage_pct,
                        fee: $fee,
                        portfolio_value: $portfolio_value
                    })
                    RETURN pt
                    """,
                    trade_id=order.order_id,
                    timestamp=result["timestamp"],
                    symbol=order.symbol,
                    side=order.side.value,
                    order_type=order.order_type.value,
                    amount=float(order.amount),
                    fill_price=result["fill_price"],
                    market_price=result["market_price"],
                    slippage_pct=result["slippage_pct"],
                    fee=float(result["fees"].get("USDT", 0)),
                    portfolio_value=float(self.portfolio.get_total_portfolio_value()),
                )
                logger.debug("Logged paper trade to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log paper trade to graph: {}", e)

    def get_virtual_portfolio(self) -> PortfolioManager:
        """Get virtual portfolio.

        Returns:
            Portfolio manager instance
        """
        return self.portfolio

    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get paper trading portfolio statistics.

        Returns:
            Portfolio statistics
        """
        stats = self.portfolio.get_portfolio_statistics()

        # Add paper trading specific stats
        stats.update(
            {
                "paper_trading": True,
                "trade_count": self.trade_count,
                "start_time": self.start_time.isoformat(),
                "runtime_hours": (datetime.now(timezone.utc) - self.start_time).total_seconds()
                / 3600,
            }
        )

        return stats

    def calculate_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.

        Args:
            current_prices: Current market prices

        Returns:
            Performance metrics dictionary
        """
        # Calculate unrealized P&L
        total_unrealized = self.portfolio.calculate_unrealized_pnl(current_prices)

        # Get portfolio value
        total_value = self.portfolio.get_total_portfolio_value(current_prices)
        initial_value = sum(self.portfolio.initial_balances.values())

        # Calculate returns
        total_return = (total_value - initial_value) / initial_value if initial_value > 0 else 0

        # Win rate
        win_rate = (
            self.portfolio.winning_trades / self.portfolio.total_trades
            if self.portfolio.total_trades > 0
            else 0
        )

        # Average win/loss
        avg_win = Decimal("0")
        avg_loss = Decimal("0")

        total_won = Decimal("0")
        total_lost = Decimal("0")

        for position in self.portfolio.positions.values():
            if position.realized_pnl > 0:
                total_won += position.realized_pnl
            elif position.realized_pnl < 0:
                total_lost += abs(position.realized_pnl)

        if self.portfolio.winning_trades > 0:
            avg_win = total_won / self.portfolio.winning_trades
        if self.portfolio.losing_trades > 0:
            avg_loss = total_lost / self.portfolio.losing_trades

        # Profit factor
        profit_factor = total_won / total_lost if total_lost > 0 else Decimal("0")

        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(current_prices)

        # Sharpe ratio approximation (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio(current_prices)

        return {
            "total_trades": self.portfolio.total_trades,
            "winning_trades": self.portfolio.winning_trades,
            "losing_trades": self.portfolio.losing_trades,
            "win_rate": float(win_rate),
            "total_return_pct": float(total_return * 100),
            "total_realized_pnl": float(self.portfolio.total_realized_pnl),
            "total_unrealized_pnl": float(total_unrealized),
            "total_value": float(total_value),
            "initial_value": float(initial_value),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_drawdown * 100),
            "sharpe_ratio": float(sharpe_ratio),
            "total_fees_paid": float(self.portfolio.total_fees_paid),
        }

    def _calculate_max_drawdown(self, current_prices: Dict[str, float]) -> Decimal:
        """Calculate maximum drawdown.

        Args:
            current_prices: Current market prices

        Returns:
            Maximum drawdown as decimal (e.g., 0.10 = 10%)
        """
        # Simplified max drawdown calculation
        # In production, you'd track peak values over time
        current_value = self.portfolio.get_total_portfolio_value(current_prices)
        initial_value = sum(self.portfolio.initial_balances.values())

        if current_value < initial_value:
            drawdown = (initial_value - current_value) / initial_value
            return drawdown

        return Decimal("0")

    def _calculate_sharpe_ratio(self, current_prices: Dict[str, float]) -> float:
        """Calculate Sharpe ratio (simplified approximation).

        Args:
            current_prices: Current market prices

        Returns:
            Sharpe ratio
        """
        # Simplified Sharpe ratio calculation
        # In production, use proper time-series returns
        try:
            total_return_pct = self.portfolio.total_realized_pnl / sum(
                self.portfolio.initial_balances.values()
            )

            # Assume 5% annual risk-free rate and scale by runtime
            runtime_days = max(1, (datetime.now(timezone.utc) - self.start_time).days)
            annual_return = total_return_pct * (365 / runtime_days)
            risk_free_rate = 0.05

            # Approximate volatility (simplified)
            # In production, calculate actual standard deviation of returns
            volatility = 0.15  # Assume 15% annual volatility

            sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            return float(sharpe)

        except Exception as e:
            logger.warning("Failed to calculate Sharpe ratio: {}", e)
            return 0.0

    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get paper trading order history.

        Args:
            limit: Maximum number of orders

        Returns:
            List of orders
        """
        orders = self.order_history[-limit:]
        return [order.to_dict() for order in orders]

    def reset_portfolio(self, new_balance: Optional[Dict[str, float]] = None) -> None:
        """Reset virtual portfolio to initial state.

        Args:
            new_balance: New initial balance (uses original if None)
        """
        if new_balance is None:
            new_balance = {k: float(v) for k, v in self.portfolio.initial_balances.items()}

        self.portfolio = PortfolioManager(
            initial_balance=new_balance,
            risk_per_trade=self.portfolio.risk_per_trade,
            max_position_size=self.portfolio.max_position_size,
            max_portfolio_risk=self.portfolio.max_portfolio_risk,
            stop_loss_pct=self.portfolio.stop_loss_pct,
            take_profit_pct=self.portfolio.take_profit_pct,
        )

        self.orders.clear()
        self.order_history.clear()
        self.trade_count = 0
        self.start_time = datetime.now(timezone.utc)

        logger.info("Paper trading portfolio reset with balance: {}", new_balance)

    async def check_readiness_for_live_trading(self) -> Dict[str, Any]:
        """Check if paper trading performance meets live trading requirements.

        Returns:
            Readiness assessment dictionary
        """
        # Get current prices (would need market data integration)
        # For now, use placeholder
        current_prices = {}

        metrics = self.calculate_performance_metrics(current_prices)

        # Define requirements (from config)
        requirements = {
            "min_days": self.config.get("min_days", 3),
            "min_trades": self.config.get("min_trades", 100),
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 10.0),
            "min_win_rate": self.config.get("min_win_rate", 55.0),
            "min_sharpe_ratio": self.config.get("min_sharpe_ratio", 1.5),
        }

        # Calculate runtime days
        runtime_days = (datetime.now(timezone.utc) - self.start_time).days

        # Check each requirement
        checks = {
            "min_days": {
                "required": requirements["min_days"],
                "actual": runtime_days,
                "passed": runtime_days >= requirements["min_days"],
            },
            "min_trades": {
                "required": requirements["min_trades"],
                "actual": metrics["total_trades"],
                "passed": metrics["total_trades"] >= requirements["min_trades"],
            },
            "max_drawdown": {
                "required": f"<={requirements['max_drawdown_pct']}%",
                "actual": f"{metrics['max_drawdown_pct']:.2f}%",
                "passed": metrics["max_drawdown_pct"] <= requirements["max_drawdown_pct"],
            },
            "win_rate": {
                "required": f">={requirements['min_win_rate']}%",
                "actual": f"{metrics['win_rate']*100:.2f}%",
                "passed": metrics["win_rate"] * 100 >= requirements["min_win_rate"],
            },
            "sharpe_ratio": {
                "required": f">={requirements['min_sharpe_ratio']}",
                "actual": f"{metrics['sharpe_ratio']:.2f}",
                "passed": metrics["sharpe_ratio"] >= requirements["min_sharpe_ratio"],
            },
        }

        # Overall passed status
        all_passed = all(check["passed"] for check in checks.values())

        # Generate recommendations
        recommendations = []
        if not checks["min_days"]["passed"]:
            recommendations.append(
                f"Continue paper trading for {requirements['min_days'] - runtime_days} more days"
            )
        if not checks["min_trades"]["passed"]:
            recommendations.append(
                f"Execute {requirements['min_trades'] - metrics['total_trades']} more trades"
            )
        if not checks["max_drawdown"]["passed"]:
            recommendations.append("Reduce drawdown before live trading")
        if not checks["win_rate"]["passed"]:
            recommendations.append("Improve win rate before live trading")
        if not checks["sharpe_ratio"]["passed"]:
            recommendations.append("Improve risk-adjusted returns before live trading")

        return {
            "ready_for_live_trading": all_passed,
            "checks": checks,
            "recommendations": recommendations,
            "current_metrics": metrics,
        }

    def __repr__(self) -> str:
        return (
            f"PaperTradingEngine("
            f"trades={self.trade_count}, "
            f"portfolio_value={self.portfolio.get_total_portfolio_value()})"
        )
