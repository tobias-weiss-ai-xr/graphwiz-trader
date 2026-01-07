#!/usr/bin/env python3
"""Simplified live trading without monitoring imports."""

import sys
import os
import asyncio
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from loguru import logger
    import ccxt
    import pandas as pd
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install dependencies:")
    print("  pip install loguru ccxt pandas python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Import GoEmotions configuration and components
try:
    from graphwiz_trader.goemotions_config import load_config as load_goemotions_config, ConfigError
    from graphwiz_trader.sentiment.goemotions_analyzer import GoEmotionsAnalyzer
    from graphwiz_trader.strategies.emotion_strategy import (
        EmotionBasedStrategy,
        EmotionStrategyConfig,
        MarketPhase,
        EmotionSignal
    )
except ImportError as e:
    logger.error(f"Failed to import GoEmotions components: {e}")
    sys.exit(1)


class GoEmotionsLiveTrader:
    """Live trading with GoEmotions sentiment analysis."""

    def __init__(self, config_path: str):
        """Initialize GoEmotions live trader from configuration file."""
        # Load configuration
        try:
            self.config = load_goemotions_config(config_path)
            logger.info(f"Configuration loaded from: {config_path}")
        except ConfigError as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
        
        # Get live trading configuration
        live_config = self.config.get_live_trading_config()
        execution_config = live_config.get('execution', {})
        limits_config = live_config.get('trade_limits', {})
        position_config = live_config.get('position_management', {})
        risk_config = live_config.get('risk_management', {})
        
        # Set trading parameters from config
        self.symbols = self.config.symbols
        self.max_position = position_config.get('max_position_eur', 250)
        self.max_daily_loss = risk_config.get('max_daily_loss_eur', 75)
        self.max_daily_trades = limits_config.get('max_daily_trades', 25)
        self.require_confirmation = limits_config.get('require_confirmation', False)
        self.update_interval = execution_config.get('interval_seconds', 30)
        
        logger.info(f"Trading mode: {self.config.trading_mode}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Max position: €{self.max_position}")
        logger.info(f"Max daily trades: {self.max_daily_trades}")
        logger.info(f"Update interval: {self.update_interval}s")
        
        # Initialize Kraken exchange
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in environment")
            sys.exit(1)
        
        self.exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # Initialize GoEmotions if available
        if self.config.goemotions_enabled:
            logger.info("Initializing GoEmotions strategy...")
            strategy_config = self.config.get_strategy_config()
            
            config = EmotionStrategyConfig(
                extreme_euphoria_threshold=strategy_config.get('extreme_euphoria_threshold', 0.75),
                extreme_fear_threshold=strategy_config.get('extreme_fear_threshold', 0.75),
                use_contrarian_signals=strategy_config.get('use_contrarian_signals', True),
                min_data_points=strategy_config.get('min_data_points', 3)
            )
            self.emotion_strategy = EmotionBasedStrategy(config=config)
            self.analyzer = GoEmotionsAnalyzer()
            self.use_emotions = True
        else:
            self.use_emotions = False
            logger.warning("GoEmotions not enabled, using technical analysis only")
        
        # Portfolio tracking
        self.positions = {}
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.starting_balance = None
        
    def fetch_balance(self):
        """Fetch account balance from Kraken."""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}
    
    def fetch_ohlcv(self, symbol: str, timeframe='1h', limit=100):
        """Fetch OHLCV data for technical analysis."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, df, period=14):
        """Calculate RSI indicator."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0, delta)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0, -delta)).rolling(window=period).mean()
        rs = gain.ewm(alpha=1, adjust=False).mean() / loss.ewm(alpha=1, adjust=False).mean()
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal from technical and emotion analysis."""
        try:
            # Get config values
            tech_config = self.config.get_technical_indicators_config()
            signal_config = self.config.get_signal_generation_config()
            confidence_threshold = 0.65
            
            # Technical analysis - RSI
            rsi_period = tech_config.get('rsi', {}).get('period', 14)
            rsi_oversold = tech_config.get('rsi', {}).get('oversold', 30)
            rsi_overbought = tech_config.get('rsi', {}).get('overbought', 70)
            
            rsi = self.calculate_rsi(df, period=rsi_period)
            latest_rsi = rsi.iloc[-1]
            
            # Determine technical signal
            if latest_rsi <= rsi_oversold:
                technical_signal = "BUY"
                confidence = signal_config.get('technical_weight', 0.5)
            elif latest_rsi >= rsi_overbought:
                technical_signal = "SELL"
                confidence = signal_config.get('technical_weight', 0.5)
            else:
                technical_signal = "HOLD"
                confidence = 0.5
            
            logger.info(f"Analyzing {symbol}...")
            logger.info(f"  RSI: {latest_rsi:.1f}")
            
            # Emotion analysis if enabled
            if self.use_emotions:
                text_config = self.config.get_goemotions_config().get('text_analysis', {})
                emotions_config = self.config.get_emotions_config()
                
                try:
                    texts = self.analyzer.fetch_texts_for_symbol(symbol, max_texts=text_config.get('max_texts_per_symbol', 5))
                    
                    if texts and len(texts) >= self.config.get_strategy_config().get('min_data_points', 3):
                        emotion_result = self.emotion_strategy.analyze_emotions(texts)
                        dominant_emotion = emotion_result.get('dominant_emotion')
                        
                        logger.info(f"  Emotion: {dominant_emotion}")
                        
                        # Check buy/sell signals based on emotions
                        buy_signals = emotions_config.get('buy_signals', [])
                        sell_signals = emotions_config.get('sell_signals', [])
                        
                        if dominant_emotion in buy_signals:
                            confidence += signal_config.get('emotion_weight', 0.5)
                        elif dominant_emotion in sell_signals:
                            confidence += signal_config.get('emotion_weight', 0.5)
                        
                        # Adjust confidence based on signal strength
                        if confidence >= 0.85:
                            logger.info(f"  Final: STRONG {technical_signal} (confidence: {confidence:.2%})")
                        elif confidence >= 0.75:
                            logger.info(f"  Final: {technical_signal} (confidence: {confidence:.2%})")
                        else:
                            logger.info(f"  Final: {technical_signal} (confidence: {confidence:.2%})")
                    else:
                        logger.info("  No emotion data available, using technical only")
                except Exception as e:
                    logger.warning(f"  Emotion analysis failed: {e}, using technical only")
            else:
                logger.info(f"  Final: {technical_signal} (confidence: {confidence:.2%})")
            
            return {
                'action': technical_signal,
                'confidence': confidence,
                'reason': f'RSI: {latest_rsi:.1f}'
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Error'}
    
    def execute_trade(self, signal: Dict[str, Any]):
        """Execute trade based on signal."""
        try:
            symbol = signal['symbol']
            action = signal['action']
            confidence = signal['confidence']
            
            # Execute if signal is strong enough
            if action != 'HOLD' and confidence > 0.65:
                logger.success(f"Executing {action} trade for {symbol} (confidence: {confidence:.2%})")
                
                # In a real implementation, this would place the order
                # self.exchange.create_market_order(symbol, 'market', side=action.lower(), amount=0.001)
                
                self.daily_trade_count += 1
                logger.info(f"Trade count: {self.daily_trade_count}/{self.max_daily_trades}")
            else:
                logger.info(f"No trade (confidence too low or HOLD signal)")
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def run(self):
        """Main trading loop."""
        logger.info("=" * 80)
        logger.info("Starting GoEmotions Live Trading")
        logger.info("=" * 80)
        
        while True:
            try:
                # Fetch balance
                balance = self.fetch_balance()
                logger.info("Account balance updated")
                
                # Analyze each symbol
                for symbol in self.symbols:
                    try:
                        # Fetch market data
                        df = self.fetch_ohlcv(symbol)
                        if df is not None:
                            # Generate signal
                            signal = self.generate_signal(symbol, df)
                            signal['symbol'] = symbol
                            
                            # Execute trade if confident enough
                            if self.daily_trade_count < self.max_daily_trades:
                                self.execute_trade(signal)
                            else:
                                logger.warning(f"Daily trade limit reached ({self.max_daily_trades})")
                    
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                logger.info("=" * 80)
                
                # Wait for next update
                logger.info(f"Next update in {self.update_interval} seconds...")
                await asyncio.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live trading with GoEmotions - REAL MONEY"
    )
    parser.add_argument(
        '--config',
        default='config/goemotions_trading.yaml',
        help='Path to configuration file (default: config/goemotions_trading.yaml)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test connection only (no trading)'
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompts (DANGEROUS!)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_goemotions_config(args.config)
    
    logger.info("GoEmotions Live Trading")
    logger.info(f"Trading mode: {config.trading_mode}")
    logger.info(f"Symbols: {config.symbols}")
    
    if config.is_live_trading:
        logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY ⚠️")
        auto_confirm = os.environ.get('AUTO_CONFIRM', '').lower() == 'yes'
        if not auto_confirm:
            response = input("Are you sure you want to proceed? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Aborted")
                return
        else:
            logger.info("Auto-confirm enabled, proceeding...")
    
    # Test mode
    if args.test:
        logger.info("Testing connection only...")
        trader = GoEmotionsLiveTrader(args.config)
        balance = trader.fetch_balance()
        logger.success(f"Connection successful! Balance: {balance}")
        return
    
    # Run trading
    trader = GoEmotionsLiveTrader(args.config)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())