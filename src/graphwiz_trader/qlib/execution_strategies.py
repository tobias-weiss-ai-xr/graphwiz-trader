"""Smart execution strategies for optimal trade execution.

This module implements various execution strategies:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Implementation Shortfall
- POV (Percentage of Volume)
- RL-based execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum

from .rl_execution import (
    ExecutionEnvironment,
    TWAPExecutor,
    SmartOrderRouter,
    ExecutionAnalyzer,
    OrderBook,
    ExecutionState,
)


class ExecutionStrategy(Enum):
    """Execution strategy types."""
    MARKET = "market"  # Immediate market order
    LIMIT = "limit"  # Limit order
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    POV = "pov"  # Percentage of Volume
    SHORTFALL = "shortfall"  # Implementation Shortfall
    RL = "rl"  # Reinforcement Learning


@dataclass
class ExecutionPlan:
    """Execution plan for an order."""
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: float
    strategy: ExecutionStrategy
    start_time: datetime
    end_time: Optional[datetime]
    slices: List[Dict[str, Any]]  # Execution slices
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class OptimalExecutionEngine:
    """
    Main execution engine using optimal strategies.

    Selects and executes the best strategy based on:
    - Order size
    - Market conditions
    - Time constraints
    - Risk tolerance
    """

    def __init__(
        self,
        default_strategy: ExecutionStrategy = ExecutionStrategy.TWAP,
        risk_tolerance: str = 'medium',  # 'low', 'medium', 'high'
    ):
        """
        Initialize execution engine.

        Args:
            default_strategy: Default execution strategy
            risk_tolerance: Risk tolerance level
        """
        self.default_strategy = default_strategy
        self.risk_tolerance = risk_tolerance

        # Initialize strategy executors
        self.twap_executor = TWAPExecutor(num_slices=10)
        self.order_router = SmartOrderRouter(
            exchanges=['binance', 'okx'],
            fee_schedule={'binance': 0.001, 'okx': 0.001}
        )

        logger.info(f"Optimal execution engine initialized: {default_strategy.value}")

    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_data: pd.DataFrame,
        strategy: Optional[ExecutionStrategy] = None,
        time_horizon: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """
        Create optimal execution plan.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Total quantity to execute
            market_data: Historical market data
            strategy: Execution strategy (uses default if None)
            time_horizon: Time horizon in minutes
            params: Additional strategy parameters

        Returns:
            Execution plan
        """
        strategy = strategy or self.default_strategy

        # Calculate time horizon if not provided
        if time_horizon is None:
            # Default: Execute over 1 hour
            time_horizon = 60

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=time_horizon)

        # Generate slices based on strategy
        if strategy == ExecutionStrategy.TWAP:
            slices = self._create_twap_slices(
                quantity,
                time_horizon,
                params or {},
            )
        elif strategy == ExecutionStrategy.VWAP:
            slices = self._create_vwap_slices(
                quantity,
                market_data,
                time_horizon,
                params or {},
            )
        elif strategy == ExecutionStrategy.MARKET:
            slices = self._create_market_slice(
                quantity,
                params or {},
            )
        elif strategy == ExecutionStrategy.POV:
            slices = self._create_pov_slices(
                quantity,
                market_data,
                time_horizon,
                params or {},
            )
        else:
            # Default to TWAP
            slices = self._create_twap_slices(
                quantity,
                time_horizon,
                {},
            )

        plan = ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            strategy=strategy,
            start_time=start_time,
            end_time=end_time,
            slices=slices,
            params=params or {},
        )

        logger.info(f"Created {strategy.value} execution plan: {len(slices)} slices")

        return plan

    def _create_twap_slices(
        self,
        quantity: float,
        time_horizon: int,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create TWAP execution slices."""
        num_slices = params.get('num_slices', 10)
        slice_quantity = quantity / num_slices

        slices = []
        for i in range(num_slices):
            slice_time = timedelta(minutes=time_horizon * (i + 1) / num_slices)

            slices.append({
                'slice_id': i + 1,
                'quantity': slice_quantity,
                'execution_time': slice_time,
                'type': 'market',
                'strategy': 'twap',
            })

        return slices

    def _create_vwap_slices(
        self,
        quantity: float,
        market_data: pd.DataFrame,
        time_horizon: int,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create VWAP execution slices based on historical volume patterns."""
        num_slices = params.get('num_slices', 10)

        # Use historical volume distribution
        # For simplicity, use equal slices (should use actual volume profile in production)
        slice_quantity = quantity / num_slices

        slices = []
        for i in range(num_slices):
            slice_time = timedelta(minutes=time_horizon * (i + 1) / num_slices)

            slices.append({
                'slice_id': i + 1,
                'quantity': slice_quantity,
                'execution_time': slice_time,
                'type': 'market',
                'strategy': 'vwap',
            })

        logger.info(f"Created VWAP slices (using equal distribution)")
        return slices

    def _create_market_slice(
        self,
        quantity: float,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create immediate market execution."""
        slices = [{
            'slice_id': 1,
            'quantity': quantity,
            'execution_time': timedelta(0),
            'type': 'market',
            'strategy': 'market',
        }]

        return slices

    def _create_pov_slices(
        self,
        quantity: float,
        market_data: pd.DataFrame,
        time_horizon: int,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create POV (Percentage of Volume) execution slices."""
        participation_rate = params.get('participation_rate', 0.1)  # 10% of volume

        # Estimate total market volume over horizon
        avg_volume = market_data['volume'].mean()
        total_market_volume = avg_volume * (time_horizon / 60)  # Rough estimate

        # Calculate execution schedule
        slices = []
        num_intervals = params.get('num_intervals', 10)

        for i in range(num_intervals):
            interval_volume = total_market_volume / num_intervals
            slice_quantity = interval_volume * participation_rate

            # Don't exceed remaining quantity
            remaining_quantity = quantity - sum(s['quantity'] for s in slices)
            slice_quantity = min(slice_quantity, remaining_quantity)

            slice_time = timedelta(minutes=time_horizon * (i + 1) / num_intervals)

            slices.append({
                'slice_id': i + 1,
                'quantity': slice_quantity,
                'execution_time': slice_time,
                'type': 'market',
                'strategy': 'pov',
                'participation_rate': participation_rate,
            })

        return slices

    def execute_plan(
        self,
        plan: ExecutionPlan,
        execute_func: callable,
    ) -> Dict[str, Any]:
        """
        Execute the plan.

        Args:
            plan: Execution plan
            execute_func: Function to execute individual slices
                             Should have signature: func(symbol, side, quantity) -> result

        Returns:
            Execution results
        """
        logger.info(f"Executing {plan.strategy.value} plan for {plan.symbol}")

        execution_results = []
        total_executed = 0.0
        total_cost = 0.0
        execution_prices = []

        for slice_plan in plan.slices:
            try:
                logger.info(f"Executing slice {slice_plan['slice_id']}: {slice_plan['quantity']:.4f}")

                # Execute slice
                result = execute_func(
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=slice_plan['quantity'],
                )

                if result and result.get('status') == 'executed':
                    execution_results.append(result)
                    total_executed += slice_plan['quantity']
                    execution_prices.append(result.get('price', 0))

                    logger.info(f"✓ Slice {slice_plan['slice_id']} executed: {result.get('order_id')}")
                else:
                    logger.warning(f"✗ Slice {slice_plan['slice_id']} failed")

            except Exception as e:
                logger.error(f"Error executing slice {slice_plan['slice_id']}: {e}")

        # Calculate execution metrics
        completion_rate = total_executed / plan.total_quantity if plan.total_quantity > 0 else 0
        avg_price = np.mean(execution_prices) if execution_prices else 0

        results = {
            'strategy': plan.strategy.value,
            'total_quantity': plan.total_quantity,
            'executed_quantity': total_executed,
            'completion_rate': completion_rate,
            'avg_execution_price': avg_price,
            'num_slices': len(plan.slices),
            'successful_slices': len(execution_results),
            'execution_results': execution_results,
        }

        logger.info(f"Execution complete: {completion_rate:.1%} filled at avg price {avg_price:.2f}")

        return results


class SlippageMinimizer:
    """
    Minimize slippage through intelligent execution.

    Techniques:
    - Order splitting
    - Smart timing
    - Venue selection
    - Limit orders when appropriate
    """

    def __init__(
        self,
        max_slippage_threshold: float = 0.5,  # 0.5%
        order_size_threshold: float = 0.1,  # As % of market volume
    ):
        """
        Initialize slippage minimizer.

        Args:
            max_slippage_threshold: Maximum acceptable slippage (%)
            order_size_threshold: Order size threshold (as % of volume)
        """
        self.max_slippage_threshold = max_slippage_threshold
        self.order_size_threshold = order_size_threshold

    def estimate_slippage(
        self,
        quantity: float,
        market_volume: float,
        current_spread: float,
        volatility: float,
    ) -> float:
        """
        Estimate execution slippage.

        Args:
            quantity: Order quantity
            market_volume: Current market volume
            current_spread: Current bid-ask spread
            volatility: Current volatility

        Returns:
            Estimated slippage (%)
        """
        # Order size impact
        size_ratio = quantity / market_volume
        size_impact = size_ratio * 0.1  # 0.1% impact per 100% of volume

        # Spread impact
        spread_impact = current_spread / 2  # Pay half spread on average

        # Volatility impact
        volatility_impact = volatility * 0.5

        # Total estimated slippage
        estimated_slippage = (
            size_impact + spread_impact + volatility_impact
        ) * 100  # Convert to percentage

        return estimated_slippage

    def recommend_strategy(
        self,
        quantity: float,
        market_volume: float,
        current_spread: float,
        volatility: float,
        urgency: str = 'medium',  # 'low', 'medium', 'high'
    ) -> ExecutionStrategy:
        """
        Recommend optimal execution strategy.

        Args:
            quantity: Order quantity
            market_volume: Current market volume
            current_spread: Current bid-ask spread
            volatility: Current volatility
            urgency: Execution urgency

        Returns:
            Recommended execution strategy
        """
        # Estimate slippage for immediate execution
        estimated_slippage = self.estimate_slippage(
            quantity, market_volume, current_spread, volatility
        )

        # Order size ratio
        size_ratio = quantity / market_volume

        # Decision logic
        if urgency == 'high':
            # High urgency: execute quickly
            if size_ratio > self.order_size_threshold:
                return ExecutionStrategy.TWAP
            else:
                return ExecutionStrategy.MARKET

        elif urgency == 'low':
            # Low urgency: minimize cost
            if estimated_slippage > self.max_slippage_threshold:
                return ExecutionStrategy.TWAP
            else:
                return ExecutionStrategy.LIMIT

        else:  # medium urgency
            # Medium urgency: balance speed and cost
            if size_ratio > self.order_size_threshold * 2:
                return ExecutionStrategy.VWAP
            elif estimated_slippage > self.max_slippage_threshold:
                return ExecutionStrategy.TWAP
            else:
                return ExecutionStrategy.MARKET

    def calculate_optimal_slice_size(
        self,
        total_quantity: float,
        market_volume: float,
        volatility: float,
        target_slippage: float = 0.1,
    ) -> float:
        """
        Calculate optimal slice size to meet slippage target.

        Args:
            total_quantity: Total order quantity
            market_volume: Market volume
            volatility: Current volatility
            target_slippage: Target slippage (%)

        Returns:
            Optimal slice size
        """
        # Simple model: slice size inversely proportional to volatility
        # Higher volatility → smaller slices
        base_slice = total_quantity * 0.1  # Start with 10% slices

        volatility_adjustment = max(0.1, 1.0 - volatility)
        optimal_slice = base_slice * volatility_adjustment

        # Ensure we don't exceed target slippage per slice
        max_slice = (target_slippage / 100) * market_volume
        optimal_slice = min(optimal_slice, max_slice)

        return optimal_slice


def create_optimal_execution_engine(
    default_strategy: ExecutionStrategy = ExecutionStrategy.TWAP,
    risk_tolerance: str = 'medium',
) -> OptimalExecutionEngine:
    """
    Convenience function to create execution engine.

    Args:
        default_strategy: Default execution strategy
        risk_tolerance: Risk tolerance level

    Returns:
        OptimalExecutionEngine instance
    """
    return OptimalExecutionEngine(
        default_strategy=default_strategy,
        risk_tolerance=risk_tolerance,
    )
