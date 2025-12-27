"""Qlib-based trading strategy using ML signals."""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

try:
    from graphwiz_trader.qlib import (
        QlibConfig,
        QlibDataAdapter,
        QlibSignalGenerator,
    )
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("Qlib not available. QlibStrategy will not function.")

from graphwiz_trader.trading.engine import TradingEngine


class QlibStrategy:
    """
    Trading strategy that uses Qlib's ML-based signals.

    This strategy integrates Qlib's machine learning models to generate
    trading signals based on Alpha158 features and combines them with
    the GraphWiz knowledge graph for enhanced decision making.
    """

    def __init__(
        self,
        trading_engine: TradingEngine,
        symbols: List[str],
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize Qlib strategy.

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

        # Strategy settings
        self.signal_threshold = self.config.get("signal_threshold", 0.6)
        self.position_size = self.config.get("position_size", 0.1)
        self.max_positions = self.config.get("max_positions", 3)
        self.retrain_interval_hours = self.config.get("retrain_interval_hours", 24)
        self.lookback_days = self.config.get("lookback_days", 30)

        # State tracking
        self.last_retrain_time = None
        self.last_signal_time = {}
        self._running = False

        logger.info(f"QlibStrategy initialized for symbols: {symbols}")

    async def start(self) -> None:
        """Start the strategy."""
        logger.info("Starting QlibStrategy...")
        await self.data_adapter.initialize()

        # Load or train model
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading pre-trained model from {self.model_path}")
            self.signal_generator.load_model(self.model_path)
        else:
            logger.info("No pre-trained model found. Training initial model...")
            await self.train_model()

        self._running = True
        logger.info("QlibStrategy started")

    async def stop(self) -> None:
        """Stop the strategy."""
        logger.info("Stopping QlibStrategy...")
        self._running = False
        await self.data_adapter.close()
        logger.info("QlibStrategy stopped")

    async def train_model(self) -> Dict[str, Any]:
        """
        Train the ML model on historical data.

        Returns:
            Training results dictionary
        """
        logger.info("Training Qlib model...")

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

                # Train model for this symbol
                results = self.signal_generator.train(
                    df=df,
                    symbol=symbol,
                    validation_split=0.2,
                )

                training_results[symbol] = results

                logger.info(f"Training complete for {symbol}: "
                           f"Train Acc={results['train_accuracy']:.4f}, "
                           f"Val Acc={results['val_accuracy']:.4f}")

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
        """
        Generate trading signals for all tracked symbols.

        Returns:
            Dictionary mapping symbols to signal information
        """
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

                logger.info(f"Generated signal for {symbol}: "
                           f"{prediction['signal']} (prob={prediction['probability']:.4f}, "
                           f"confidence={prediction['confidence']})")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue

        return signals

    async def execute_signals(self, signals: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute trades based on generated signals.

        Args:
            signals: Dictionary of signals by symbol

        Returns:
            List of execution results
        """
        results = []

        # Check current positions
        current_positions = self.trading_engine.get_positions()
        current_symbols = {pos['symbol'] for pos in current_positions}

        for symbol, signal in signals.items():
            try:
                # Get current positions for this symbol
                symbol_positions = [
                    pos for pos in current_positions
                    if symbol in pos['symbol']
                ]

                # Execute based on signal
                if signal['signal'] == 'BUY':
                    # Only buy if we don't already have a position
                    if not symbol_positions:
                        # Check max positions limit
                        if len(current_positions) >= self.max_positions:
                            logger.info(f"Max positions reached. Skipping {symbol}")
                            continue

                        result = await self._execute_buy(symbol, signal)
                        results.append(result)

                elif signal['signal'] in ['HOLD', 'SELL']:
                    # Check if we should sell existing positions
                    if symbol_positions:
                        # Sell if signal is weak
                        if signal['probability'] < 0.3:
                            result = await self._execute_sell(symbol, signal)
                            results.append(result)

            except Exception as e:
                logger.error(f"Error executing signal for {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'status': 'error',
                    'error': str(e)
                })

        return results

    async def _execute_buy(
        self,
        symbol: str,
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a buy order based on signal."""
        try:
            exchange = self.config.get("exchange", "binance")

            # Calculate position size
            # For now, use a fixed percentage of portfolio
            # In production, this would use Kelly criterion or similar
            amount = self._calculate_position_size(symbol)

            logger.info(f"Executing BUY for {symbol}: {signal}")

            result = self.trading_engine.execute_trade(
                symbol=symbol,
                side="buy",
                amount=amount,
                exchange_name=exchange,
                order_type="market",
            )

            self.last_signal_time[symbol] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Error executing buy for {symbol}: {e}")
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}

    async def _execute_sell(
        self,
        symbol: str,
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a sell order based on signal."""
        try:
            exchange = self.config.get("exchange", "binance")

            # Get current position
            positions = self.trading_engine.get_positions()
            symbol_positions = [
                pos for pos in positions
                if symbol in pos['symbol'] and pos['side'] == 'buy'
            ]

            if not symbol_positions:
                logger.info(f"No long position to close for {symbol}")
                return {'symbol': symbol, 'status': 'skipped', 'reason': 'No position'}

            # Sell the entire position
            amount = symbol_positions[0]['amount']

            logger.info(f"Executing SELL for {symbol}: {signal}")

            result = self.trading_engine.execute_trade(
                symbol=symbol,
                side="sell",
                amount=amount,
                exchange_name=exchange,
                order_type="market",
            )

            self.last_signal_time[symbol] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Error executing sell for {symbol}: {e}")
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}

    def _calculate_position_size(self, symbol: str) -> float:
        """
        Calculate position size based on confidence and portfolio.

        Args:
            symbol: Trading symbol

        Returns:
            Position size in base currency
        """
        # For now, use a simple fixed position size
        # In production, this would consider:
        # - Portfolio value
        # - Signal confidence
        # - Risk management rules
        # - Volatility

        # This is a simplified calculation
        # Assume we want to trade $1000 worth of the asset
        return 1000.0

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Run one complete strategy cycle.

        This includes:
        1. Check if model needs retraining
        2. Generate signals
        3. Execute trades

        Returns:
            Dictionary with cycle results
        """
        logger.info("Running QlibStrategy cycle...")

        # Check if we need to retrain
        should_retrain = False
        if self.last_retrain_time is None:
            should_retrain = True
        else:
            hours_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since_retrain >= self.retrain_interval_hours:
                should_retrain = True

        if should_retrain:
            logger.info("Retraining model...")
            await self.train_model()

        # Generate signals
        signals = await self.generate_signals()

        # Execute signals
        execution_results = await self.execute_signals(signals)

        cycle_results = {
            'timestamp': datetime.now().isoformat(),
            'signals_generated': len(signals),
            'trades_executed': len(execution_results),
            'signals': signals,
            'executions': execution_results,
        }

        logger.info(f"Cycle complete: {len(signals)} signals, {len(execution_results)} trades")

        return cycle_results

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the strategy state.

        Returns:
            Dictionary with strategy information
        """
        return {
            'strategy_type': 'qlib_ml',
            'symbols': self.symbols,
            'running': self._running,
            'signal_threshold': self.signal_threshold,
            'position_size': self.position_size,
            'max_positions': self.max_positions,
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'model_loaded': self.signal_generator.model is not None,
            'config': self.config,
        }


def create_qlib_strategy(
    trading_engine: TradingEngine,
    symbols: List[str],
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[Path] = None,
) -> QlibStrategy:
    """
    Convenience function to create a Qlib strategy.

    Args:
        trading_engine: Trading engine instance
        symbols: List of symbols to trade
        config: Strategy configuration
        model_path: Path to pre-trained model

    Returns:
        QlibStrategy instance
    """
    return QlibStrategy(
        trading_engine=trading_engine,
        symbols=symbols,
        config=config,
        model_path=model_path,
    )
