"""Backtesting engine for strategy testing."""

from loguru import logger
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import numpy as np
import pandas as pd

from graphwiz_trader.analysis import TechnicalAnalysis


@dataclass
class Trade:
    """Represents a trade in backtest."""
    timestamp: datetime
    symbol: str
    action: str  # "buy" or "sell"
    price: float
    quantity: float
    value: float
    reason: str = ""


@dataclass
class BacktestResult:
    """Results of a backtest."""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    trades: List[Trade] = field(default_factory=list)
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.trades:
            winning_trades = [t for t in self.trades if self._is_winning_trade(t)]
            self.win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0

    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade was profitable."""
        # Simplified: buy trades are profitable if price increased, sell if decreased
        # In real implementation, would match buy/sell pairs
        return trade.action == "buy"  # Placeholder


class BacktestEngine:
    """Backtesting engine for trading strategies."""

    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital for backtest
            commission: Commission rate per trade (default 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.technical_analysis = TechnicalAnalysis()

    def run_backtest(
        self,
        historical_data: List[Dict[str, Any]],
        strategy: Callable[[Dict[str, Any]], str],
        symbol: str = "BTC/USDT"
    ) -> BacktestResult:
        """Run a backtest on historical data.

        Args:
            historical_data: List of OHLCV data points
            strategy: Strategy function that takes context and returns "buy", "sell", or "hold"
            symbol: Trading symbol

        Returns:
            BacktestResult with performance metrics
        """
        if not historical_data:
            logger.error("No historical data provided")
            return self._create_empty_result(symbol)

        logger.info("Starting backtest for {} with {} data points", symbol, len(historical_data))

        # Initialize backtest state
        capital = self.initial_capital
        position = 0.0  # Amount of asset held
        trades = []
        equity_curve = [capital]

        # Track entry price for position
        entry_price = 0.0
        position_side = None  # "long" or "short"

        # Process each data point
        for i, data_point in enumerate(historical_data):
            current_price = data_point.get("close", data_point.get("price", 0))
            timestamp = data_point.get("timestamp", datetime.now())

            if current_price <= 0:
                continue

            # Build context for strategy
            context = self._build_context(historical_data[:i+1], current_price, symbol)

            # Get strategy decision
            try:
                signal = strategy(context)
            except Exception as e:
                logger.warning("Strategy error at index {}: {}", i, e)
                signal = "hold"

            # Execute trades based on signal
            trade = self._execute_strategy(
                signal, current_price, capital, position, entry_price, position_side, timestamp, symbol
            )

            if trade:
                trades.append(trade)
                # Update state
                if trade.action == "buy":
                    # Calculate cost with commission
                    cost = trade.quantity * trade.price * (1 + self.commission)

                    # Only buy if we have enough capital
                    if cost > capital:
                        # Reduce quantity to fit available capital
                        trade.quantity = (capital / (1 + self.commission)) / trade.price
                        cost = capital
                        trade.value = trade.quantity * trade.price

                    capital -= cost
                    position += trade.quantity

                    if position_side != "long":
                        entry_price = trade.price
                        position_side = "long"

                elif trade.action == "sell":
                    if position_side == "long" and position > 0:
                        # Sell the entire position or the specified quantity
                        sell_quantity = min(trade.quantity, position)
                        proceeds = sell_quantity * trade.price * (1 - self.commission)

                        capital += proceeds
                        position -= sell_quantity

                        if position <= 0:
                            position = 0
                            position_side = None

            # Calculate current equity
            if position > 0:
                current_equity = capital + (position * current_price)
            else:
                current_equity = capital
            equity_curve.append(current_equity)

        # Calculate final results
        final_capital = equity_curve[-1] if equity_curve else capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Calculate metrics
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        sortino_ratio = self._calculate_sortino_ratio(equity_curve)

        result = BacktestResult(
            symbol=symbol,
            start_date=historical_data[0].get("timestamp", datetime.now()),
            end_date=historical_data[-1].get("timestamp", datetime.now()),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            trades=trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            metrics={
                "equity_curve": equity_curve,
                "trade_count": len(trades),
                "avg_trade_value": np.mean([t.value for t in trades]) if trades else 0,
                "final_position": position
            }
        )

        logger.info("Backtest complete: Return {:.2f}%, {} trades, Sharpe: {:.2f}",
                   total_return_pct, len(trades), sharpe_ratio)

        return result

    def _build_context(self, historical_data: List[Dict[str, Any]], current_price: float, symbol: str) -> Dict[str, Any]:
        """Build context for strategy decision.

        Args:
            historical_data: Historical price data
            current_price: Current market price
            symbol: Trading symbol

        Returns:
            Context dictionary
        """
        # Extract prices for technical analysis
        prices = [d.get("close", d.get("price", 0)) for d in historical_data]
        volumes = [d.get("volume", 0) for d in historical_data]
        highs = [d.get("high", d.get("close", 0)) for d in historical_data]
        lows = [d.get("low", d.get("close", 0)) for d in historical_data]
        closes = [d.get("close", d.get("price", 0)) for d in historical_data]

        # Perform technical analysis
        analysis = self.technical_analysis.analyze(
            prices=prices,
            volumes=volumes if any(volumes) else None,
            high=highs if any(highs) else None,
            low=lows if any(lows) else None,
            close=closes if any(closes) else None
        )

        return {
            "symbol": symbol,
            "current_price": current_price,
            "technical_indicators": analysis,
            "data_points": len(historical_data)
        }

    def _execute_strategy(
        self,
        signal: str,
        price: float,
        capital: float,
        position: float,
        entry_price: float,
        position_side: Optional[str],
        timestamp: datetime,
        symbol: str
    ) -> Optional[Trade]:
        """Execute strategy signal.

        Args:
            signal: Strategy signal ("buy", "sell", "hold")
            price: Current price
            capital: Available capital
            position: Current position size
            entry_price: Average entry price
            position_side: "long" or "short"
            timestamp: Trade timestamp
            symbol: Trading symbol

        Returns:
            Trade object or None
        """
        if signal == "hold":
            return None

        # Position sizing (simplified: use 10% of capital per trade)
        trade_size = capital * 0.1
        quantity = trade_size / price

        if signal == "buy":
            return Trade(
                timestamp=timestamp,
                symbol=symbol,
                action="buy",
                price=price,
                quantity=quantity,
                value=trade_size,
                reason="Strategy buy signal"
            )
        elif signal == "sell":
            # Sell entire position if we have one
            sell_quantity = position if position > 0 else quantity
            return Trade(
                timestamp=timestamp,
                symbol=symbol,
                action="sell",
                price=price,
                quantity=sell_quantity,
                value=sell_quantity * price,
                reason="Strategy sell signal"
            )

        return None

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown.

        Args:
            equity_curve: List of equity values over time

        Returns:
            Maximum drawdown as percentage
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd * 100  # Return as percentage

    def _calculate_sharpe_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.

        Args:
            equity_curve: List of equity values over time
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(ret)

        if not returns:
            return 0.0

        # Calculate average return and std dev
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize (assuming daily data)
        sharpe = (avg_return * 252 - risk_free_rate) / (std_return * np.sqrt(252))
        return sharpe

    def _calculate_sortino_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside risk only).

        Args:
            equity_curve: List of equity values over time
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sortino ratio
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(ret)

        if not returns:
            return 0.0

        # Separate negative returns
        negative_returns = [r for r in returns if r < 0]

        avg_return = np.mean(returns)

        if not negative_returns:
            return 0.0  # No downside risk

        # Calculate downside deviation
        downside_dev = np.std(negative_returns)

        if downside_dev == 0:
            return 0.0

        # Annualize
        sortino = (avg_return * 252 - risk_free_rate) / (downside_dev * np.sqrt(252))
        return sortino

    def _create_empty_result(self, symbol: str) -> BacktestResult:
        """Create empty result for failed backtest.

        Args:
            symbol: Trading symbol

        Returns:
            Empty BacktestResult
        """
        return BacktestResult(
            symbol=symbol,
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0.0,
            total_return_pct=0.0,
            trades=[]
        )


class SimpleMovingAverageStrategy:
    """Simple moving average crossover strategy for backtesting."""

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        """Initialize SMA strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period

    def __call__(self, context: Dict[str, Any]) -> str:
        """Generate trading signal.

        Args:
            context: Trading context with technical indicators

        Returns:
            "buy", "sell", or "hold"
        """
        indicators = context.get("technical_indicators", {})
        ema = indicators.get("ema", {})

        ema_20 = ema.get("latest_20")
        ema_50 = ema.get("latest_50")

        if ema_20 is None or ema_50 is None:
            return "hold"

        current_price = context.get("current_price", 0)

        # Buy signal: fast EMA crosses above slow EMA
        if ema_20 > ema_50 and current_price > ema_20:
            return "buy"

        # Sell signal: fast EMA crosses below slow EMA
        if ema_20 < ema_50 and current_price < ema_20:
            return "sell"

        return "hold"


class RSIMeanReversionStrategy:
    """RSI mean reversion strategy."""

    def __init__(self, oversold: float = 30, overbought: float = 70):
        """Initialize RSI strategy.

        Args:
            oversold: RSI level for oversold (buy signal)
            overbought: RSI level for overbought (sell signal)
        """
        self.oversold = oversold
        self.overbought = overbought

    def __call__(self, context: Dict[str, Any]) -> str:
        """Generate trading signal based on RSI.

        Args:
            context: Trading context

        Returns:
            "buy", "sell", or "hold"
        """
        indicators = context.get("technical_indicators", {})
        rsi = indicators.get("rsi", {})
        rsi_value = rsi.get("latest")

        if rsi_value is None:
            return "hold"

        # Buy when oversold
        if rsi_value <= self.oversold:
            return "buy"

        # Sell when overbought
        if rsi_value >= self.overbought:
            return "sell"

        return "hold"

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """Generate trading signal from price data.

        This is a convenience method for paper trading and live trading.
        It calculates RSI from the DataFrame and returns a signal.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            "buy", "sell", or None
        """
        try:
            # Calculate RSI using existing TechnicalIndicators class
            from ..analysis import TechnicalIndicators

            prices = data["close"].tolist()
            rsi_result = TechnicalIndicators.rsi(prices, period=14)

            if not rsi_result.values or rsi_result.values[-1] is None:
                return None

            rsi_value = rsi_result.values[-1]

            # Buy when oversold
            if rsi_value <= self.oversold:
                return "buy"

            # Sell when overbought
            if rsi_value >= self.overbought:
                return "sell"

            return None

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
