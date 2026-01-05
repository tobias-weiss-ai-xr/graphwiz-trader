"""Enhanced Qlib-based trading strategy with portfolio optimization.

This strategy integrates:
- ML-based signal generation
- Portfolio optimization
- Dynamic position sizing
- Advanced risk management
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

try:
    from graphwiz_trader.qlib import (
        QlibConfig,
        QlibDataAdapter,
        QlibSignalGenerator,
        PortfolioOptimizer,
        DynamicPositionSizer,
        PortfolioConstraints,
        OptimizerConfig,
        BacktestEngine,
        BacktestConfig,
    )

    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("Qlib not available. QlibStrategyV2 will not function.")

from graphwiz_trader.trading.engine import TradingEngine


class QlibStrategyV2:
    """
    Enhanced trading strategy with portfolio optimization.

    This strategy uses Qlib's ML models for signal generation and
    portfolio optimization for position management.
    """

    def __init__(
        self,
        trading_engine: TradingEngine,
        symbols: List[str],
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize Qlib strategy V2.

        Args:
            trading_engine: Trading engine instance
            symbols: List of symbols to trade
            config: Strategy configuration
            model_path: Path to pre-trained model
        """
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib not available. Install Qlib dependencies.")

        self.trading_engine = trading_engine
        self.symbols = symbols
        self.config = config or {}
        self.model_path = model_path

        # Initialize Qlib components
        self.qlib_config = QlibConfig(
            provider=self.config.get("qlib_provider", "ccxt"),
            region=self.config.get("qlib_region", "crypto"),
            freq=self.config.get("qlib_freq", "1h"),
        )

        self.data_adapter = QlibDataAdapter(
            exchange_id=self.config.get("exchange", "binance"),
            config=self.qlib_config,
        )

        self.signal_generator = QlibSignalGenerator(
            config=self.qlib_config,
            model_path=model_path,
        )

        # Portfolio optimization
        opt_config = OptimizerConfig(
            optimization_method=self.config.get("optimization_method", "mean_variance"),
            risk_free_rate=self.config.get("risk_free_rate", 0.02),
            rebalance_frequency=self.config.get("rebalance_frequency", "1d"),
            lookback_window=self.config.get("lookback_window", 60),
        )

        constraints = PortfolioConstraints(
            max_position_weight=self.config.get("max_position_weight", 0.4),
            min_position_weight=self.config.get("min_position_weight", 0.0),
            max_leverage=self.config.get("max_leverage", 1.0),
            target_volatility=self.config.get("target_volatility"),
            max_drawdown=self.config.get("max_drawdown"),
        )

        self.portfolio_optimizer = PortfolioOptimizer(
            config=opt_config,
            constraints=constraints,
        )

        # Dynamic position sizer
        self.position_sizer = DynamicPositionSizer(
            base_position_size=self.config.get("base_position_size", 0.1),
            max_position_size=self.config.get("max_position_size", 0.3),
            min_position_size=self.config.get("min_position_size", 0.05),
            risk_tolerance=self.config.get("risk_tolerance", 0.02),
        )

        # Strategy settings
        self.signal_threshold = self.config.get("signal_threshold", 0.6)
        self.retrain_interval_hours = self.config.get("retrain_interval_hours", 24)
        self.lookback_days = self.config.get("lookback_days", 30)
        self.enable_portfolio_opt = self.config.get("enable_portfolio_opt", True)

        # State tracking
        self.last_retrain_time = None
        self.last_rebalance_time = None
        self.current_weights = pd.Series([0.0] * len(symbols), index=symbols)
        self._running = False

        # Portfolio value tracking
        self.portfolio_value = self.config.get("initial_capital", 100000.0)

        logger.info(f"QlibStrategyV2 initialized for symbols: {symbols}")
        logger.info(
            f"Portfolio optimization: {'enabled' if self.enable_portfolio_opt else 'disabled'}"
        )

    async def start(self) -> None:
        """Start the strategy."""
        logger.info("Starting QlibStrategyV2...")
        await self.data_adapter.initialize()

        # Load or train model
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading pre-trained model from {self.model_path}")
            self.signal_generator.load_model(self.model_path)
        else:
            logger.info("No pre-trained model found. Training initial model...")
            await self.train_models()

        self._running = True
        logger.info("QlibStrategyV2 started")

    async def stop(self) -> None:
        """Stop the strategy."""
        logger.info("Stopping QlibStrategyV2...")
        self._running = False
        await self.data_adapter.close()
        logger.info("QlibStrategyV2 stopped")

    async def train_models(self) -> Dict[str, Any]:
        """Train ML models for all symbols."""
        logger.info("Training models for all symbols...")

        training_results = {}

        for symbol in self.symbols:
            try:
                logger.info(f"Fetching training data for {symbol}...")

                # Fetch historical data
                start_date = datetime.now() - timedelta(days=self.lookback_days)
                df = await self.data_adapter.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=datetime.now(),
                    timeframe=self.qlib_config.freq,
                )

                if len(df) < 100:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} candles")
                    continue

                # Train model
                results = self.signal_generator.train(
                    df=df,
                    symbol=symbol,
                    validation_split=0.2,
                )

                training_results[symbol] = results

                logger.info(
                    f"Training complete for {symbol}: "
                    f"Train Acc={results['train_accuracy']:.4f}, "
                    f"Val Acc={results['val_accuracy']:.4f}"
                )

            except Exception as e:
                logger.error(f"Error training model for {symbol}: {e}")
                continue

        # Save model
        if self.model_path:
            self.signal_generator.save_model(self.model_path)
            logger.info(f"Model saved to {self.model_path}")

        self.last_retrain_time = datetime.now()

        return training_results

    async def generate_signals(self) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals for all symbols."""
        if not self._running:
            logger.warning("Strategy not running. Cannot generate signals.")
            return {}

        signals = {}

        for symbol in self.symbols:
            try:
                # Get latest data
                df = await self.data_adapter.get_latest_data(
                    symbol=symbol,
                    timeframe=self.qlib_config.freq,
                )

                if len(df) < 100:
                    logger.warning(f"Insufficient data for signal generation on {symbol}")
                    continue

                # Generate prediction
                prediction = self.signal_generator.predict_latest(
                    df=df,
                    symbol=symbol,
                    threshold=self.signal_threshold,
                )

                signals[symbol] = prediction

                logger.info(
                    f"Generated signal for {symbol}: "
                    f"{prediction['signal']} (prob={prediction['probability']:.4f}, "
                    f"confidence={prediction['confidence']})"
                )

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue

        return signals

    async def optimize_portfolio(
        self,
        signals: Dict[str, Dict[str, Any]],
    ) -> pd.Series:
        """
        Optimize portfolio weights based on signals and market data.

        Args:
            signals: Dictionary of signals by symbol

        Returns:
            Optimal weights for each symbol
        """
        if not self.enable_portfolio_opt:
            # Use equal weights if optimization disabled
            n_symbols = len(self.symbols)
            return pd.Series(1.0 / n_symbols, index=self.symbols)

        logger.info("Optimizing portfolio...")

        try:
            # Fetch historical returns for all symbols
            returns_dict = {}

            for symbol in self.symbols:
                try:
                    start_date = datetime.now() - timedelta(
                        days=self.portfolio_optimizer.config.lookback_window
                    )
                    df = await self.data_adapter.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=datetime.now(),
                        timeframe=self.qlib_config.freq,
                    )

                    # Calculate returns
                    returns = df["close"].pct_change().dropna()
                    returns_dict[symbol] = returns

                except Exception as e:
                    logger.error(f"Error fetching returns for {symbol}: {e}")
                    continue

            if not returns_dict:
                logger.warning("No returns data available. Using equal weights.")
                n_symbols = len(self.symbols)
                return pd.Series(1.0 / n_symbols, index=self.symbols)

            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_dict)

            # Calculate expected returns based on signals
            expected_returns = pd.Series(index=self.symbols, dtype=float)

            for symbol in self.symbols:
                if symbol in signals:
                    # Use signal probability as expected return
                    # Convert probability to expected return
                    prob = signals[symbol]["probability"]
                    # Map 0-1 probability to -10% to +10% expected return
                    expected_returns[symbol] = (prob - 0.5) * 0.2
                else:
                    expected_returns[symbol] = 0.0

            # Optimize portfolio
            optimal_weights = self.portfolio_optimizer.optimize(
                returns=returns_df,
                expected_returns=expected_returns,
                current_weights=self.current_weights,
            )

            logger.info(f"Optimal weights: {optimal_weights.to_dict()}")

            return optimal_weights

        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            logger.info("Falling back to equal weights")
            n_symbols = len(self.symbols)
            return pd.Series(1.0 / n_symbols, index=self.symbols)

    async def execute_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        optimal_weights: pd.Series,
    ) -> List[Dict[str, Any]]:
        """
        Execute trades based on signals and optimal weights.

        Args:
            signals: Dictionary of signals
            optimal_weights: Optimal portfolio weights

        Returns:
            List of execution results
        """
        results = []

        # Get current positions
        current_positions = self.trading_engine.get_positions()
        position_values = {}

        # Calculate current position values
        for pos in current_positions:
            symbol_key = pos["symbol"].replace("/", "")
            position_values[symbol_key] = pos.get("amount", 0) * pos.get(
                "current_price", pos.get("entry_price", 1)
            )

        # Calculate target position values
        target_values = {}
        for symbol, weight in optimal_weights.items():
            target_values[symbol] = weight * self.portfolio_value

        # Execute trades to reach target weights
        for symbol in self.symbols:
            try:
                signal = signals.get(symbol, {})
                signal_type = signal.get("signal", "HOLD")

                # Only trade if signal is BUY
                if signal_type != "BUY":
                    # If we have a position and signal is not BUY, consider closing
                    current_value = position_values.get(symbol, 0)
                    target_value = target_values.get(symbol, 0)

                    if current_value > 0 and target_value == 0:
                        # Close position
                        result = await self._close_position(symbol)
                        results.append(result)
                    continue

                # Calculate current and target positions
                current_value = position_values.get(symbol, 0)
                target_value = target_values.get(symbol, 0)

                # Calculate trade value
                trade_value = target_value - current_value

                if abs(trade_value) < self.portfolio_value * 0.01:  # Less than 1% change
                    logger.debug(f"Position for {symbol} already optimal. Skipping.")
                    continue

                if trade_value > 0:
                    # Buy
                    result = await self._execute_buy(symbol, trade_value, signal)
                    results.append(result)
                else:
                    # Sell
                    result = await self._execute_sell(symbol, abs(trade_value), signal)
                    results.append(result)

            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
                results.append({"symbol": symbol, "status": "error", "error": str(e)})

        # Update current weights
        self.current_weights = optimal_weights
        self.last_rebalance_time = datetime.now()

        return results

    async def _execute_buy(
        self,
        symbol: str,
        trade_value: float,
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a buy order."""
        try:
            exchange = self.config.get("exchange", "binance")

            # Get current price
            if self.trading_engine.exchanges:
                ticker = self.trading_engine.exchanges[exchange].fetch_ticker(symbol)
                current_price = ticker["last"]
            else:
                logger.error(f"Exchange {exchange} not available")
                return {"symbol": symbol, "status": "error", "error": "Exchange not available"}

            # Calculate quantity
            quantity = trade_value / current_price

            logger.info(
                f"Executing BUY for {symbol}: ${trade_value:.2f} ({quantity:.4f} @ ${current_price:.2f})"
            )

            result = self.trading_engine.execute_trade(
                symbol=symbol,
                side="buy",
                amount=quantity,
                exchange_name=exchange,
                order_type="market",
            )

            return result

        except Exception as e:
            logger.error(f"Error executing buy for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    async def _execute_sell(
        self,
        symbol: str,
        trade_value: float,
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a sell order."""
        try:
            exchange = self.config.get("exchange", "binance")

            # Get current position
            positions = self.trading_engine.get_positions()
            symbol_positions = [
                pos for pos in positions if symbol in pos["symbol"] and pos["side"] == "buy"
            ]

            if not symbol_positions:
                logger.info(f"No long position to sell for {symbol}")
                return {"symbol": symbol, "status": "skipped", "reason": "No position"}

            # Get current price
            if self.trading_engine.exchanges:
                ticker = self.trading_engine.exchanges[exchange].fetch_ticker(symbol)
                current_price = ticker["last"]
            else:
                return {"symbol": symbol, "status": "error", "error": "Exchange not available"}

            # Calculate quantity to sell
            quantity = trade_value / current_price

            # Don't sell more than we have
            max_quantity = symbol_positions[0]["amount"]
            quantity = min(quantity, max_quantity)

            logger.info(
                f"Executing SELL for {symbol}: ${trade_value:.2f} ({quantity:.4f} @ ${current_price:.2f})"
            )

            result = self.trading_engine.execute_trade(
                symbol=symbol,
                side="sell",
                amount=quantity,
                exchange_name=exchange,
                order_type="market",
            )

            return result

        except Exception as e:
            logger.error(f"Error executing sell for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    async def _close_position(self, symbol: str) -> Dict[str, Any]:
        """Close entire position for a symbol."""
        try:
            exchange = self.config.get("exchange", "binance")

            # Get current position
            positions = self.trading_engine.get_positions()
            symbol_positions = [
                pos for pos in positions if symbol in pos["symbol"] and pos["side"] == "buy"
            ]

            if not symbol_positions:
                return {"symbol": symbol, "status": "skipped", "reason": "No position"}

            quantity = symbol_positions[0]["amount"]

            logger.info(f"Closing position for {symbol}: {quantity}")

            result = self.trading_engine.execute_trade(
                symbol=symbol,
                side="sell",
                amount=quantity,
                exchange_name=exchange,
                order_type="market",
            )

            return result

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    async def run_cycle(self) -> Dict[str, Any]:
        """Run one complete strategy cycle."""
        logger.info("Running QlibStrategyV2 cycle...")

        # Check if we need to retrain
        should_retrain = False
        if self.last_retrain_time is None:
            should_retrain = True
        else:
            hours_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since_retrain >= self.retrain_interval_hours:
                should_retrain = True

        if should_retrain:
            logger.info("Retraining models...")
            await self.train_models()

        # Generate signals
        signals = await self.generate_signals()

        # Optimize portfolio
        optimal_weights = await self.optimize_portfolio(signals)

        # Execute trades
        execution_results = await self.execute_signals(signals, optimal_weights)

        cycle_results = {
            "timestamp": datetime.now().isoformat(),
            "signals_generated": len(signals),
            "portfolio_optimization": self.enable_portfolio_opt,
            "optimal_weights": optimal_weights.to_dict(),
            "trades_executed": len(execution_results),
            "executions": execution_results,
        }

        logger.info(f"Cycle complete: {len(signals)} signals, {len(execution_results)} trades")

        return cycle_results

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy state."""
        return {
            "strategy_type": "qlib_ml_v2",
            "version": "2.0",
            "symbols": self.symbols,
            "running": self._running,
            "signal_threshold": self.signal_threshold,
            "portfolio_optimization_enabled": self.enable_portfolio_opt,
            "optimization_method": self.portfolio_optimizer.config.optimization_method,
            "current_weights": self.current_weights.to_dict(),
            "last_retrain_time": (
                self.last_retrain_time.isoformat() if self.last_retrain_time else None
            ),
            "last_rebalance_time": (
                self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
            ),
            "model_loaded": self.signal_generator.model is not None,
            "config": self.config,
        }


def create_qlib_strategy_v2(
    trading_engine: TradingEngine,
    symbols: List[str],
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[Path] = None,
) -> QlibStrategyV2:
    """
    Convenience function to create Qlib Strategy V2.

    Args:
        trading_engine: Trading engine instance
        symbols: List of symbols to trade
        config: Strategy configuration
        model_path: Path to pre-trained model

    Returns:
        QlibStrategyV2 instance
    """
    return QlibStrategyV2(
        trading_engine=trading_engine,
        symbols=symbols,
        config=config,
        model_path=model_path,
    )
