"""
Adapter for integrating modern strategies with the trading engine.

This module provides a unified interface for running modern strategies
(Grid Trading, Smart DCA, AMM, Triangular Arbitrage) with the trading engine.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from loguru import logger

from .modern_strategies import (
    GridTradingStrategy,
    GridTradingMode,
    SmartDCAStrategy,
    AutomatedMarketMakingStrategy,
    TriangularArbitrageStrategy,
    create_modern_strategy,
)


class ModernStrategyAdapter:
    """Adapter for running modern strategies with trading engine."""

    def __init__(
        self,
        strategy: Union[
            GridTradingStrategy,
            SmartDCAStrategy,
            AutomatedMarketMakingStrategy,
            TriangularArbitrageStrategy,
        ],
    ):
        """Initialize adapter with a modern strategy.

        Args:
            strategy: Modern strategy instance
        """
        self.strategy = strategy
        self.strategy_type = strategy.__class__.__name__
        logger.info(f"Initialized ModernStrategyAdapter for {self.strategy_type}")

    def generate_trading_signals(
        self, current_price: float, historical_data: Optional[pd.DataFrame] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate trading signals based on strategy type.

        Args:
            current_price: Current market price
            historical_data: Historical price data
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dictionary with trading signals and metadata
        """
        try:
            if isinstance(self.strategy, GridTradingStrategy):
                return self._generate_grid_signals(current_price, historical_data)

            elif isinstance(self.strategy, SmartDCAStrategy):
                return self._generate_dca_signals(current_price, historical_data)

            elif isinstance(self.strategy, AutomatedMarketMakingStrategy):
                return self._generate_amm_signals(current_price, **kwargs)

            elif isinstance(self.strategy, TriangularArbitrageStrategy):
                return self._generate_arbitrage_signals(**kwargs)

            else:
                return {
                    "status": "error",
                    "message": f"Unknown strategy type: {self.strategy_type}",
                }

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {"status": "error", "message": str(e)}

    def _generate_grid_signals(
        self, current_price: float, historical_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate signals for grid trading strategy.

        Args:
            current_price: Current market price
            historical_data: Historical price data

        Returns:
            Signals dictionary with orders to place
        """
        signals = self.strategy.generate_signals(current_price, historical_data)

        # Convert to trading engine format
        orders = []
        for order in signals.get("orders_to_place", []):
            orders.append(
                {
                    "symbol": self.strategy.symbol,
                    "side": order["side"],
                    "amount": order["quantity"],
                    "price": order["price"],
                    "order_type": "limit",  # Grid orders are always limit orders
                    "strategy": "grid_trading",
                    "metadata": {"grid_level": order.get("grid_level"), "reason": "grid_placement"},
                }
            )

        return {
            "status": "success",
            "strategy": "grid_trading",
            "current_price": signals["current_price"],
            "grid_levels": signals["grid_levels"],
            "orders": orders,
            "rebalance_needed": signals.get("rebalance_needed", False),
            "trailing_profit_active": signals.get("trailing_profit_active", False),
            "trailing_sell_price": signals.get("trailing_sell_price"),
        }

    def _generate_dca_signals(
        self, current_price: float, historical_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate signals for smart DCA strategy.

        Args:
            current_price: Current market price
            historical_data: Historical price data

        Returns:
            Signals dictionary with purchase order
        """
        purchase = self.strategy.calculate_next_purchase(current_price, historical_data)

        # Convert to trading engine format
        order = {
            "symbol": self.strategy.symbol,
            "side": purchase["action"],
            "amount": purchase["quantity"],
            "price": purchase["price"],
            "order_type": "market",  # DCA orders are typically market orders
            "strategy": "smart_dca",
            "metadata": {
                "reason": purchase["reason"],
                "amount_usd": purchase["amount"],
                "volatility_adjusted": purchase.get("volatility_adjusted", False),
                "momentum_boost": purchase.get("momentum_boost", False),
            },
        }

        return {
            "status": "success",
            "strategy": "smart_dca",
            "order": order,
            "should_execute": purchase["action"] == "buy",
        }

    def _generate_amm_signals(self, current_price: float, **kwargs) -> Dict[str, Any]:
        """Generate signals for AMM strategy.

        Args:
            current_price: Current market price
            **kwargs: Additional parameters (current_inventory_a, current_inventory_b)

        Returns:
            Signals dictionary with rebalancing actions
        """
        current_inventory_a = kwargs.get("current_inventory_a", 0)
        current_inventory_b = kwargs.get("current_inventory_b", 0)

        recommendations = self.strategy.calculate_optimal_positions(
            current_inventory_a=current_inventory_a,
            current_inventory_b=current_inventory_b,
            current_price=current_price,
        )

        # Convert rebalancing actions to orders
        orders = []
        if recommendations["needs_rebalance"] and recommendations["actions"]:
            for action in recommendations["actions"]:
                if "buy" in action.lower():
                    orders.append(
                        {
                            "symbol": f"{self.strategy.token_a}/{self.strategy.token_b}",
                            "side": "buy",
                            "amount": kwargs.get("trade_amount", 1.0),  # Default trade amount
                            "price": current_price,
                            "order_type": "market",
                            "strategy": "amm",
                            "metadata": {
                                "reason": "inventory_rebalance",
                                "current_ratio_a": recommendations["current_ratio_a"],
                                "target_ratio": recommendations["target_ratio"],
                            },
                        }
                    )

        return {
            "status": "success",
            "strategy": "amm",
            "current_ratio_a": recommendations["current_ratio_a"],
            "target_ratio": recommendations["target_ratio"],
            "needs_rebalance": recommendations["needs_rebalance"],
            "orders": orders,
        }

    def _generate_arbitrage_signals(self, **kwargs) -> Dict[str, Any]:
        """Generate signals for triangular arbitrage strategy.

        Args:
            **kwargs: Price data dictionary

        Returns:
            Signals dictionary with arbitrage opportunities
        """
        # Update prices if provided
        price_data = kwargs.get("price_data")
        if price_data:
            self.strategy.update_prices(price_data)

        # Find opportunities
        opportunities = self.strategy.find_arbitrage_opportunities()

        # Convert opportunities to orders
        orders = []
        if opportunities:
            best_opportunity = opportunities[0]
            orders.append(
                {
                    "strategy": "triangular_arbitrage",
                    "opportunity": best_opportunity,
                    "metadata": {
                        "exchange": best_opportunity["exchange"],
                        "path": best_opportunity["path"],
                        "profit_pct": best_opportunity["profit_pct"],
                        "estimated_profit": best_opportunity["estimated_profit"],
                    },
                }
            )

        return {
            "status": "success",
            "strategy": "triangular_arbitrage",
            "opportunities_found": len(opportunities),
            "orders": orders,
            "should_execute": len(opportunities) > 0,
        }

    def execute_trade(self, trade_result: Dict[str, Any]) -> bool:
        """Update strategy state after trade execution.

        Args:
            trade_result: Result from trading engine

        Returns:
            True if strategy updated successfully
        """
        try:
            if isinstance(self.strategy, SmartDCAStrategy):
                # Record DCA purchase
                if trade_result.get("status") == "executed":
                    purchase_data = {
                        "action": trade_result["side"],
                        "amount": trade_result.get("metadata", {}).get("amount_usd", 0),
                        "price": trade_result["price"],
                        "quantity": trade_result["amount"],
                    }
                    # symbol is optional for execute_purchase
                    if "symbol" in trade_result:
                        purchase_data["symbol"] = trade_result["symbol"]
                    self.strategy.execute_purchase(purchase_data)
                    logger.info(
                        f"Recorded DCA purchase: {trade_result['amount']} @ ${trade_result['price']}"
                    )

            elif isinstance(self.strategy, AutomatedMarketMakingStrategy):
                # Simulate trade for AMM
                self.strategy.simulate_trade(
                    {
                        "side": trade_result["side"],
                        "amount": trade_result["amount"],
                        "price": trade_result["price"],
                    }
                )
                logger.info(f"Recorded AMM trade: {trade_result['side']} {trade_result['amount']}")

            elif isinstance(self.strategy, TriangularArbitrageStrategy):
                # Record arbitrage execution
                opportunity = trade_result.get("metadata", {}).get("opportunity")
                if opportunity:
                    result = self.strategy.execute_arbitrage(
                        opportunity, trade_size=trade_result.get("trade_size", 1000)
                    )
                    logger.info(f"Recorded arbitrage execution: {result}")

            return True

        except Exception as e:
            logger.error(f"Error updating strategy state: {e}")
            return False

    def get_strategy_status(self, current_price: float) -> Dict[str, Any]:
        """Get current strategy status and metrics.

        Args:
            current_price: Current market price

        Returns:
            Strategy status dictionary
        """
        try:
            if isinstance(self.strategy, SmartDCAStrategy):
                return self.strategy.get_portfolio_status(current_price)

            elif isinstance(self.strategy, AutomatedMarketMakingStrategy):
                return self.strategy.get_pool_metrics()

            elif isinstance(self.strategy, GridTradingStrategy):
                return {
                    "strategy_type": "grid_trading",
                    "symbol": self.strategy.symbol,
                    "grid_levels": len(self.strategy.grid_levels),
                    "upper_price": self.strategy.upper_price,
                    "lower_price": self.strategy.lower_price,
                    "current_price": current_price,
                }

            elif isinstance(self.strategy, TriangularArbitrageStrategy):
                return {
                    "strategy_type": "triangular_arbitrage",
                    "exchanges": self.strategy.exchanges,
                    "trading_pairs": self.strategy.trading_pairs,
                }

            else:
                return {"strategy_type": self.strategy_type}

        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            return {"error": str(e)}


def create_modern_strategy_adapter(strategy_type: str, **kwargs) -> ModernStrategyAdapter:
    """Factory function to create modern strategy adapter.

    Args:
        strategy_type: Type of strategy ('grid_trading', 'smart_dca', 'amm', 'triangular_arbitrage')
        **kwargs: Strategy-specific parameters

    Returns:
        ModernStrategyAdapter instance

    Examples:
        >>> adapter = create_modern_strategy_adapter(
        ...     strategy_type='grid_trading',
        ...     symbol='BTC/USDT',
        ...     upper_price=55000,
        ...     lower_price=45000,
        ...     num_grids=10
        ... )
    """
    strategy = create_modern_strategy(strategy_type, **kwargs)
    return ModernStrategyAdapter(strategy)
