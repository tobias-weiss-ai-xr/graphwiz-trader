"""
Trading strategies for backtesting.

This module provides various trading strategies:
- Base strategy interface
- Momentum strategy
- Mean reversion strategy
- Grid trading strategy
- Dollar Cost Averaging (DCA) strategy
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from loguru import logger


class Side(Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Signal:
    """Trading signal."""

    timestamp: datetime
    side: Side
    price: float
    quantity: float
    reason: str
    confidence: float = 1.0


@dataclass
class Trade:
    """Completed trade."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: Side
    pnl: float
    pnl_percentage: float
    reason: str


class BaseStrategy(ABC):
    """
    Base class for trading strategies.

    All strategies should inherit from this class and implement
    the generate_signals method.
    """

    def __init__(
        self,
        name: str,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,  # 0.05%
    ):
        """
        Initialize base strategy.

        Args:
            name: Strategy name
            initial_capital: Starting capital
            commission_rate: Commission rate per trade (as fraction)
            slippage_rate: Expected slippage per trade (as fraction)
        """
        self.name = name
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # State variables
        self.current_capital = initial_capital
        self.position = 0.0  # Current position size
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from historical data.

        Args:
            data: OHLCV DataFrame with price data

        Returns:
            List of Signal objects
        """
        pass

    def apply_costs(self, price: float, side: Side) -> float:
        """
        Apply commission and slippage to execution price.

        Args:
            price: Original price
            side: Order side

        Returns:
            Adjusted price after costs
        """
        # Apply slippage
        if side == Side.BUY:
            execution_price = price * (1 + self.slippage_rate)
        else:
            execution_price = price * (1 - self.slippage_rate)

        # Apply commission (separate from execution price)
        return execution_price

    def calculate_position_size(
        self,
        price: float,
        signal: Signal,
    ) -> float:
        """
        Calculate position size based on signal and capital.

        Args:
            price: Current price
            signal: Trading signal

        Returns:
            Position size (quantity)
        """
        # Default: use signal quantity, or allocate 100% of capital
        if signal.quantity > 0:
            return signal.quantity

        # Calculate max position based on available capital
        available_capital = self.current_capital * signal.confidence
        quantity = available_capital / price

        return quantity

    def execute_trade(
        self,
        signal: Signal,
        current_position: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a trade signal.

        Args:
            signal: Trading signal
            current_position: Current position size

        Returns:
            Trade execution details or None
        """
        execution_price = self.apply_costs(signal.price, signal.side)
        quantity = self.calculate_position_size(execution_price, signal)

        # Calculate commission
        commission = execution_price * quantity * self.commission_rate

        # Calculate total cost/proceeds
        if signal.side == Side.BUY:
            total_cost = execution_price * quantity + commission
            if total_cost > self.current_capital:
                logger.warning(
                    f"Insufficient capital for buy: need {total_cost:.2f}, "
                    f"have {self.current_capital:.2f}"
                )
                return None

            self.current_capital -= total_cost
            new_position = current_position + quantity

        else:  # SELL
            total_proceeds = execution_price * quantity - commission
            if quantity > current_position:
                logger.warning(
                    f"Insufficient position for sell: trying to sell {quantity:.4f}, "
                    f"have {current_position:.4f}"
                )
                return None

            self.current_capital += total_proceeds
            new_position = current_position - quantity

        return {
            "timestamp": signal.timestamp,
            "side": signal.side,
            "price": execution_price,
            "quantity": quantity,
            "commission": commission,
            "position": new_position,
            "capital": self.current_capital,
            "reason": signal.reason,
        }


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.

    Buys when price momentum is strong (positive returns),
    sells when momentum weakens or reverses.
    """

    def __init__(self, lookback_period: int = 20, momentum_threshold: float = 0.02, **kwargs):
        """
        Initialize momentum strategy.

        Args:
            lookback_period: Period to calculate momentum
            momentum_threshold: Minimum momentum to trigger trade
            **kwargs: Additional arguments for BaseStrategy
        """
        super().__init__(name="Momentum", **kwargs)
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate momentum-based signals."""
        signals = []

        # Calculate momentum (rate of change)
        data["momentum"] = data["close"].pct_change(self.lookback_period)

        # Calculate volatility for position sizing
        data["volatility"] = data["close"].pct_change().rolling(20).std()

        # Generate signals
        for i in range(self.lookback_period + 1, len(data)):
            current_time = data.index[i]
            momentum = data["momentum"].iloc[i]
            volatility = data["volatility"].iloc[i]
            price = data["close"].iloc[i]

            # Entry signals
            if momentum > self.momentum_threshold:
                # Strong upward momentum - buy
                signals.append(
                    Signal(
                        timestamp=current_time,
                        side=Side.BUY,
                        price=price,
                        quantity=0.0,  # Will be calculated
                        reason=f"Strong momentum: {momentum:.4f}",
                        confidence=min(1.0, abs(momentum) / self.momentum_threshold * 0.5),
                    )
                )

            elif momentum < -self.momentum_threshold:
                # Strong downward momentum - sell (if we have position)
                signals.append(
                    Signal(
                        timestamp=current_time,
                        side=Side.SELL,
                        price=price,
                        quantity=0.0,
                        reason=f"Negative momentum: {momentum:.4f}",
                        confidence=min(1.0, abs(momentum) / self.momentum_threshold * 0.5),
                    )
                )

        logger.info(f"Generated {len(signals)} momentum signals")
        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.

    Buys when price is below mean (oversold),
    sells when price is above mean (overbought).
    """

    def __init__(self, lookback_period: int = 20, std_dev_threshold: float = 2.0, **kwargs):
        """
        Initialize mean reversion strategy.

        Args:
            lookback_period: Period for moving average and std dev
            std_dev_threshold: Number of std devs from mean for signals
            **kwargs: Additional arguments for BaseStrategy
        """
        super().__init__(name="MeanReversion", **kwargs)
        self.lookback_period = lookback_period
        self.std_dev_threshold = std_dev_threshold

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate mean reversion signals."""
        signals = []

        # Calculate mean and standard deviation
        data["mean"] = data["close"].rolling(self.lookback_period).mean()
        data["std"] = data["close"].rolling(self.lookback_period).std()
        data["z_score"] = (data["close"] - data["mean"]) / data["std"]

        # Generate signals
        for i in range(self.lookback_period + 1, len(data)):
            current_time = data.index[i]
            z_score = data["z_score"].iloc[i]
            price = data["close"].iloc[i]

            # Entry signals
            if z_score < -self.std_dev_threshold:
                # Price significantly below mean - oversold, buy
                signals.append(
                    Signal(
                        timestamp=current_time,
                        side=Side.BUY,
                        price=price,
                        quantity=0.0,
                        reason=f"Oversold: z-score {z_score:.2f}",
                        confidence=min(1.0, abs(z_score) / self.std_dev_threshold * 0.5),
                    )
                )

            elif z_score > self.std_dev_threshold:
                # Price significantly above mean - overbought, sell
                signals.append(
                    Signal(
                        timestamp=current_time,
                        side=Side.SELL,
                        price=price,
                        quantity=0.0,
                        reason=f"Overbought: z-score {z_score:.2f}",
                        confidence=min(1.0, abs(z_score) / self.std_dev_threshold * 0.5),
                    )
                )

        logger.info(f"Generated {len(signals)} mean reversion signals")
        return signals


class GridTradingStrategy(BaseStrategy):
    """
    Grid trading strategy.

    Places buy orders at regular intervals below current price
    and sell orders at regular intervals above current price.
    """

    def __init__(self, grid_levels: int = 10, grid_spacing: float = 0.01, **kwargs):  # 1%
        """
        Initialize grid trading strategy.

        Args:
            grid_levels: Number of grid levels
            grid_spacing: Percentage spacing between grid levels
            **kwargs: Additional arguments for BaseStrategy
        """
        super().__init__(name="GridTrading", **kwargs)
        self.grid_levels = grid_levels
        self.grid_spacing = grid_spacing

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate grid trading signals."""
        signals = []

        # Calculate initial price range
        first_price = data["close"].iloc[0]
        price_min = first_price * (1 - self.grid_levels * self.grid_spacing / 2)
        price_max = first_price * (1 + self.grid_levels * self.grid_spacing / 2)

        # Create grid levels
        buy_levels = np.linspace(price_min, first_price, self.grid_levels // 2)
        sell_levels = np.linspace(first_price, price_max, self.grid_levels // 2)

        # Track which levels have been triggered
        buy_levels_triggered = set()
        sell_levels_triggered = set()

        for i in range(len(data)):
            current_time = data.index[i]
            price = data["close"].iloc[i]

            # Check buy levels
            for idx, level in enumerate(buy_levels):
                if idx not in buy_levels_triggered and price <= level:
                    signals.append(
                        Signal(
                            timestamp=current_time,
                            side=Side.BUY,
                            price=level,
                            quantity=0.0,
                            reason=f"Grid buy level {idx}: {level:.2f}",
                            confidence=1.0 / self.grid_levels,
                        )
                    )
                    buy_levels_triggered.add(idx)

            # Check sell levels
            for idx, level in enumerate(sell_levels):
                if idx not in sell_levels_triggered and price >= level:
                    signals.append(
                        Signal(
                            timestamp=current_time,
                            side=Side.SELL,
                            price=level,
                            quantity=0.0,
                            reason=f"Grid sell level {idx}: {level:.2f}",
                            confidence=1.0 / self.grid_levels,
                        )
                    )
                    sell_levels_triggered.add(idx)

        logger.info(f"Generated {len(signals)} grid trading signals")
        return signals


class DCAStrategy(BaseStrategy):
    """
    Dollar Cost Averaging (DCA) strategy.

    Buys fixed amount at regular intervals regardless of price.
    Reduces timing risk and builds position over time.
    """

    def __init__(
        self,
        purchase_interval: str = "1D",  # Pandas frequency string
        purchase_amount: float = 1000.0,
        **kwargs,
    ):
        """
        Initialize DCA strategy.

        Args:
            purchase_interval: Time between purchases (e.g., '1D', '1W')
            purchase_amount: Fixed amount to buy each interval
            **kwargs: Additional arguments for BaseStrategy
        """
        super().__init__(name="DCA", **kwargs)
        self.purchase_interval = purchase_interval
        self.purchase_amount = purchase_amount

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate DCA signals."""
        signals = []

        # Resample data to purchase interval
        resampled = data.resample(self.purchase_interval).last()
        resampled = resampled.dropna()

        last_purchase_time = None

        for i in range(len(resampled)):
            current_time = resampled.index[i]
            price = resampled["close"].iloc[i]

            # Skip if too close to last purchase
            if last_purchase_time and (
                current_time - last_purchase_time < pd.Timedelta(self.purchase_interval)
            ):
                continue

            # Calculate quantity based on fixed purchase amount
            quantity = self.purchase_amount / price

            signals.append(
                Signal(
                    timestamp=current_time,
                    side=Side.BUY,
                    price=price,
                    quantity=quantity,
                    reason=f"DCA purchase: {self.purchase_amount:.2f}",
                    confidence=1.0,
                )
            )

            last_purchase_time = current_time

        logger.info(f"Generated {len(signals)} DCA signals")
        return signals

    def calculate_position_size(
        self,
        price: float,
        signal: Signal,
    ) -> float:
        """Use fixed purchase amount for position sizing."""
        # DCA always uses fixed amount
        return signal.quantity
