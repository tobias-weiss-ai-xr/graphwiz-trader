"""Reinforcement Learning-based order execution using Qlib methodology.

This module implements RL agents for optimal trade execution:
- PPO (Proximal Policy Optimization) agent
- Smart order routing
- Slippage reduction
- Execution quality optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        logger.warning("Gym/Gymnasium not available. RL execution will be limited.")

from .config import QlibConfig


class ExecutionAction(Enum):
    """Possible execution actions."""
    WAIT = 0
    BUY_MARKET_SMALL = 1
    BUY_MARKET_MEDIUM = 2
    BUY_MARKET_LARGE = 3
    BUY_LIMIT_ABOVE = 4
    SELL_MARKET_SMALL = 5
    SELL_MARKET_MEDIUM = 6
    SELL_MARKET_LARGE = 7
    SELL_LIMIT_BELOW = 8


@dataclass
class OrderBook:
    """Order book state."""
    bids: pd.DataFrame = field(default_factory=pd.DataFrame)  # [price, volume]
    asks: pd.DataFrame = field(default_factory=pd.DataFrame)  # [price, volume]
    timestamp: datetime = field(default_factory=datetime.now)

    def get_best_bid(self) -> float:
        """Get best bid price."""
        if len(self.bids) > 0:
            return self.bids.iloc[0]['price']
        return 0.0

    def get_best_ask(self) -> float:
        """Get best ask price."""
        if len(self.asks) > 0:
            return self.asks.iloc[0]['price']
        return 0.0

    def get_spread(self) -> float:
        """Get bid-ask spread."""
        return self.get_best_ask() - self.get_best_bid()

    def get_mid_price(self) -> float:
        """Get mid price."""
        return (self.get_best_bid() + self.get_best_ask()) / 2


@dataclass
class ExecutionState:
    """Current execution state."""
    target_quantity: float  # Total quantity to execute
    executed_quantity: float = 0.0  # Quantity executed so far
    avg_execution_price: float = 0.0  # Average execution price
    remaining_quantity: float = 0.0  # Remaining to execute
    elapsed_time: float = 0.0  # Elapsed execution time
    slippage: float = 0.0  # Current slippage
    market_impact: float = 0.0  # Market impact


class ExecutionEnvironment(gym.Env if GYM_AVAILABLE else object):
    """
    RL environment for optimal order execution.

    Simulates order execution with:
    - Realistic order book dynamics
    - Market impact modeling
    - Slippage calculation
    - Time pressure
    """

    def __init__(
        self,
        order_book_history: pd.DataFrame,
        target_quantity: float,
        time_horizon: int = 100,  # Number of steps
        side: str = 'buy',
    ):
        """
        Initialize execution environment.

        Args:
            order_book_history: Historical order book data
            target_quantity: Total quantity to execute
            time_horizon: Execution time horizon (steps)
            side: Order side ('buy' or 'sell')
        """
        if not GYM_AVAILABLE:
            logger.warning("Gym not available. Execution environment will be limited.")

        self.order_book_history = order_book_history
        self.target_quantity = target_quantity
        self.time_horizon = time_horizon
        self.side = side

        # State tracking
        self.current_step = 0
        self.execution_state = ExecutionState(
            target_quantity=target_quantity,
            remaining_quantity=target_quantity,
        )

        # Action space: 9 possible actions
        if GYM_AVAILABLE:
            self.action_space = spaces.Discrete(9)

        # Observation space: [remaining_ratio, time_ratio, price_momentum, volatility, spread, depth_ratio]
        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
                dtype=np.float32,
            )

        # Track execution for reward calculation
        self.execution_prices = []
        self.execution_volumes = []

        logger.info(f"Execution environment initialized: {side} {target_quantity} units over {time_horizon} steps")

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.execution_state = ExecutionState(
            target_quantity=self.target_quantity,
            remaining_quantity=self.target_quantity,
        )
        self.execution_prices = []
        self.execution_volumes = []

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= len(self.order_book_history):
            return np.zeros(6)

        current_book = self.order_book_history.iloc[self.current_step]

        # Calculate features
        remaining_ratio = self.execution_state.remaining_quantity / self.target_quantity
        time_ratio = self.current_step / self.time_horizon

        # Price momentum (recent price change)
        window = min(10, self.current_step + 1)
        recent_prices = self.order_book_history['mid_price'].iloc[
            max(0, self.current_step - window):self.current_step + 1
        ]
        price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] if len(recent_prices) > 1 else 0

        # Volatility
        volatility = recent_prices.std() / recent_prices.mean() if len(recent_prices) > 1 else 0

        # Spread
        spread = current_book['spread'] if 'spread' in current_book else 0

        # Depth ratio (bid/ask depth imbalance)
        depth_ratio = current_book.get('depth_ratio', 1.0)

        observation = np.array([
            remaining_ratio,
            time_ratio,
            price_momentum,
            volatility,
            spread,
            depth_ratio,
        ], dtype=np.float32)

        return observation

    def step(self, action: int):
        """
        Execute one step.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.current_step >= len(self.order_book_history):
            return self._get_observation(), 0, True, {}

        # Get current market state
        current_state = self.order_book_history.iloc[self.current_step]
        current_price = current_state['mid_price']

        # Execute action
        execution_quantity, execution_price = self._execute_action(action, current_state)

        # Update execution state
        if execution_quantity > 0:
            self.execution_prices.append(execution_price)
            self.execution_volumes.append(execution_quantity)

            self.execution_state.executed_quantity += execution_quantity
            self.execution_state.remaining_quantity = max(
                0,
                self.target_quantity - self.execution_state.executed_quantity
            )

            # Update average execution price
            total_value = sum(p * v for p, v in zip(self.execution_prices, self.execution_volumes))
            total_volume = sum(self.execution_volumes)
            self.execution_state.avg_execution_price = total_value / total_volume if total_volume > 0 else 0

        # Calculate reward
        reward = self._calculate_reward(execution_quantity, execution_price, current_price)

        # Check if done
        done = (
            self.execution_state.remaining_quantity <= 0.01 or
            self.current_step >= self.time_horizon - 1
        )

        self.current_step += 1
        self.execution_state.elapsed_time = self.current_step

        info = {
            'executed_quantity': self.execution_state.executed_quantity,
            'remaining_quantity': self.execution_state.remaining_quantity,
            'avg_price': self.execution_state.avg_execution_price,
            'step': self.current_step,
        }

        return self._get_observation(), reward, done, False, info

    def _execute_action(self, action: int, market_state: pd.Series) -> Tuple[float, float]:
        """Execute action and return (quantity, price)."""
        action_enum = ExecutionAction(action)

        # Get current prices
        bid_price = market_state.get('best_bid', market_state['mid_price'])
        ask_price = market_state.get('best_ask', market_state['mid_price'])

        # Determine quantity based on action
        if action_enum == ExecutionAction.WAIT:
            return 0.0, 0.0

        elif action_enum in [ExecutionAction.BUY_MARKET_SMALL, ExecutionAction.SELL_MARKET_SMALL]:
            quantity = min(self.execution_state.remaining_quantity, self.target_quantity * 0.1)
        elif action_enum in [ExecutionAction.BUY_MARKET_MEDIUM, ExecutionAction.SELL_MARKET_MEDIUM]:
            quantity = min(self.execution_state.remaining_quantity, self.target_quantity * 0.25)
        elif action_enum in [ExecutionAction.BUY_MARKET_LARGE, ExecutionAction.SELL_MARKET_LARGE]:
            quantity = min(self.execution_state.remaining_quantity, self.target_quantity * 0.5)
        else:
            quantity = min(self.execution_state.remaining_quantity, self.target_quantity * 0.1)

        # Determine execution price based on action
        if action_enum == ExecutionAction.BUY_MARKET_SMALL:
            price = ask_price
        elif action_enum == ExecutionAction.BUY_MARKET_MEDIUM:
            price = ask_price * 1.0005  # Slight price impact
        elif action_enum == ExecutionAction.BUY_MARKET_LARGE:
            price = ask_price * 1.001  # More price impact
        elif action_enum == ExecutionAction.BUY_LIMIT_ABOVE:
            price = bid_price * 0.9995  # Limit order
        elif action_enum == ExecutionAction.SELL_MARKET_SMALL:
            price = bid_price
        elif action_enum == ExecutionAction.SELL_MARKET_MEDIUM:
            price = bid_price * 0.9995
        elif action_enum == ExecutionAction.SELL_MARKET_LARGE:
            price = bid_price * 0.999
        elif action_enum == ExecutionAction.SELL_LIMIT_BELOW:
            price = ask_price * 1.0005
        else:
            price = market_state['mid_price']

        return quantity, price

    def _calculate_reward(self, quantity: float, execution_price: float, market_price: float) -> float:
        """Calculate reward for execution."""
        if quantity == 0:
            # Small penalty for waiting when time is limited
            time_pressure = self.current_step / self.time_horizon
            return -0.01 * time_pressure

        # Reward based on execution quality
        if self.side == 'buy':
            # For buys, lower price is better
            price_advantage = (market_price - execution_price) / market_price
        else:
            # For sells, higher price is better
            price_advantage = (execution_price - market_price) / market_price

        # Reward: price improvement + completion bonus
        completion_ratio = quantity / self.target_quantity
        reward = (price_advantage * 100) + (completion_ratio * 10)

        return reward


class TWAPExecutor:
    """
    Time-Weighted Average Price execution strategy.

    Splits orders evenly across time to minimize market impact.
    """

    def __init__(
        self,
        num_slices: int = 10,
        time_interval: str = '1h',
    ):
        """
        Initialize TWAP executor.

        Args:
            num_slices: Number of time slices
            time_interval: Time between slices
        """
        self.num_slices = num_slices
        self.time_interval = time_interval

        logger.info(f"TWAP executor initialized: {num_slices} slices, {time_interval} interval")

    def generate_schedule(
        self,
        total_quantity: float,
        start_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Generate TWAP execution schedule.

        Args:
            total_quantity: Total quantity to execute
            start_time: Execution start time

        Returns:
            List of execution orders
        """
        slice_quantity = total_quantity / self.num_slices
        schedule = []

        for i in range(self.num_slices):
            execution_time = start_time + timedelta(hours=i+1)

            schedule.append({
                'slice': i + 1,
                'quantity': slice_quantity,
                'execution_time': execution_time,
                'type': 'market',
            })

        logger.info(f"Generated TWAP schedule: {len(schedule)} slices of {slice_quantity:.4f} each")

        return schedule


class SmartOrderRouter:
    """
    Smart order routing across multiple exchanges.

    Minimizes execution costs by routing orders to optimal venues.
    """

    def __init__(
        self,
        exchanges: List[str],
        fee_schedule: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize order router.

        Args:
            exchanges: List of available exchanges
            fee_schedule: Exchange fees (default: 0.1% for all)
        """
        self.exchanges = exchanges

        if fee_schedule is None:
            self.fee_schedule = {exc: 0.001 for exc in exchanges}
        else:
            self.fee_schedule = fee_schedule

        logger.info(f"Smart order router initialized for {len(exchanges)} exchanges")

    def find_best_execution(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_books: Dict[str, OrderBook],
    ) -> Tuple[str, float, float]:
        """
        Find best execution venue.

        Args:
            symbol: Trading symbol
            quantity: Quantity to execute
            side: 'buy' or 'sell'
            order_books: Order books by exchange

        Returns:
            Tuple of (exchange, price, total_cost)
        """
        best_exchange = None
        best_total_cost = float('inf')
        best_price = None

        for exchange in self.exchanges:
            if exchange not in order_books:
                continue

            order_book = order_books[exchange]
            fee = self.fee_schedule[exchange]

            # Get execution price
            if side == 'buy':
                price = order_book.get_best_ask()
            else:
                price = order_book.get_best_bid()

            # Calculate total cost (price * quantity + fees)
            execution_cost = price * quantity
            fee_cost = execution_cost * fee
            total_cost = execution_cost + fee_cost

            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_exchange = exchange
                best_price = price

        if best_exchange is None:
            logger.warning("No valid exchange found for execution")
            return None, 0.0, 0.0

        logger.debug(f"Best execution: {exchange} @ {best_price:.2f}")

        return best_exchange, best_price, best_total_cost


class ExecutionAnalyzer:
    """
    Analyze execution quality and calculate metrics.
    """

    @staticmethod
    def calculate_slippage(
        execution_price: float,
        benchmark_price: float,
        side: str,
    ) -> float:
        """
        Calculate execution slippage.

        Args:
            execution_price: Actual execution price
            benchmark_price: Benchmark price (e.g., mid price at order start)
            side: 'buy' or 'sell'

        Returns:
            Slippage as percentage
        """
        if side == 'buy':
            slippage = (execution_price - benchmark_price) / benchmark_price
        else:
            slippage = (benchmark_price - execution_price) / benchmark_price

        return slippage * 100  # Convert to percentage

    @staticmethod
    def calculate_market_impact(
        execution_prices: List[float],
        execution_volumes: List[float],
        arrival_price: float,
    ) -> float:
        """
        Calculate market impact of execution.

        Args:
            execution_prices: List of execution prices
            execution_volumes: List of execution volumes
            arrival_price: Price at order arrival

        Returns:
            Market impact as percentage
        """
        if not execution_prices:
            return 0.0

        # VWAP
        total_value = sum(p * v for p, v in zip(execution_prices, execution_volumes))
        total_volume = sum(execution_volumes)
        vwap = total_value / total_volume if total_volume > 0 else arrival_price

        # Market impact
        impact = (vwap - arrival_price) / arrival_price * 100

        return impact

    @staticmethod
    def analyze_execution_quality(
        execution_state: ExecutionState,
        benchmark_price: float,
        arrival_price: float,
    ) -> Dict[str, float]:
        """
        Comprehensive execution quality analysis.

        Args:
            execution_state: Execution state
            benchmark_price: Benchmark price
            arrival_price: Arrival price

        Returns:
            Dictionary of execution metrics
        """
        metrics = {}

        # Completion rate
        metrics['completion_rate'] = (
            execution_state.executed_quantity / execution_state.target_quantity
        )

        # Average execution price
        metrics['avg_execution_price'] = execution_state.avg_execution_price

        # Slippage vs benchmark
        # Assume buy side for now
        metrics['slippage_benchmark'] = ExecutionAnalyzer.calculate_slippage(
            execution_state.avg_execution_price,
            benchmark_price,
            'buy',
        )

        # Market impact
        metrics['market_impact'] = ExecutionAnalyzer.calculate_market_impact(
            [],  # Would need actual execution history
            [],
            arrival_price,
        )

        # Execution time
        metrics['execution_time'] = execution_state.elapsed_time

        return metrics


def create_execution_environment(
    order_book_history: pd.DataFrame,
    target_quantity: float,
    time_horizon: int = 100,
    side: str = 'buy',
) -> ExecutionEnvironment:
    """
    Convenience function to create execution environment.

    Args:
        order_book_history: Historical order book data
        target_quantity: Quantity to execute
        time_horizon: Execution horizon
        side: Order side

    Returns:
        ExecutionEnvironment instance
    """
    return ExecutionEnvironment(
        order_book_history=order_book_history,
        target_quantity=target_quantity,
        time_horizon=time_horizon,
        side=side,
    )
