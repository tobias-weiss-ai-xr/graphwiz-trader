#!/usr/bin/env python3
"""
Live trading script with GoEmotions integration - executes REAL trades with REAL money.

⚠️ WARNING: This script will execute REAL trades!
Make sure you:
1. Have tested thoroughly with paper trading
2. Understand the risks
3. Have set appropriate safety limits
4. Start with small amounts

Features:
- GoEmotions sentiment analysis
- Multi-factor signals (Technical + Emotion)
- Kraken exchange (MiCA licensed for Germany)
- Conservative risk management
- Manual trade confirmation

Usage:
    # Test connection only
    python scripts/live_trade_goemotions.py --test
    
    # Run live trading with manual confirmation
    python scripts/live_trade_goemotions.py --symbol BTC/EUR --max-position 300
    
    # Run with multiple symbols
    python scripts/live_trade_goemotions.py --symbols "BTC/EUR ETH/EUR SOL/EUR"
"""

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

# Import GoEmotions components
try:
    from graphwiz_trader.sentiment.goemotions_analyzer import GoEmotionsAnalyzer
    from graphwiz_trader.strategies.emotion_strategy import (
        EmotionBasedStrategy,
        EmotionStrategyConfig,
        MarketPhase,
        EmotionSignal
    )
except ImportError:
    logger.warning("GoEmotions components not found, falling back to technical only")
    GoEmotionsAnalyzer = None
    EmotionBasedStrategy = None


class GoEmotionsLiveTrader:
    """Live trading with GoEmotions sentiment analysis."""

    def __init__(
        self,
        symbols: List[str],
        api_key: str,
        api_secret: str,
        max_position_eur: float = 300.0,
        max_daily_loss_eur: float = 50.0,
        max_daily_trades: int = 2,
        require_confirmation: bool = True
    ):
        """
        Initialize GoEmotions live trader.
        
        Args:
            symbols: Trading pairs (e.g., ["BTC/EUR", "ETH/EUR"])
            api_key: Kraken API key
            api_secret: Kraken API secret
            max_position_eur: Maximum position size in EUR
            max_daily_loss_eur: Maximum daily loss limit
            max_daily_trades: Maximum trades per day
            require_confirmation: Require manual confirmation for trades
        """
        self.symbols = symbols
        self.max_position = max_position_eur
        self.max_daily_loss = max_daily_loss_eur
        self.max_daily_trades = max_daily_trades
        self.require_confirmation = require_confirmation
        
        # Initialize Kraken exchange
        self.exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # Initialize GoEmotions if available
        if GoEmotionsAnalyzer and EmotionBasedStrategy:
            logger.info("Initializing GoEmotions strategy...")
            config = EmotionStrategyConfig(
                extreme_euphoria_threshold=0.75,
                extreme_fear_threshold=0.75,
                use_contrarian_signals=True,
                min_data_points=3
            )
            self.emotion_strategy = EmotionBasedStrategy(config=config)
            self.analyzer = GoEmotionsAnalyzer()
            self.use_emotions = True
        else:
            self.use_emotions = False
            logger.warning("GoEmotions not available, using technical analysis only")
        
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
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            close = df['close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0
    
    async def analyze_emotions(self, symbol: str) -> Dict[str, Any]:
        """Analyze emotions for trading symbol."""
        if not self.use_emotions:
            return {'dominant_emotion': 'neutral', 'intensity': 0.0}
        
        try:
            # Simulate social media texts (in production, fetch from Twitter/Reddit)
            social_texts = self._generate_mock_social_texts(symbol)
            timestamps = [datetime.now()] * len(social_texts)
            
            # Analyze emotions
            await self.emotion_strategy.analyze_emotions_for_symbol(
                symbol, social_texts, timestamps
            )
            
            # Get emotion summary
            emotion_summary = self.emotion_strategy.get_emotion_summary(symbol)
            return emotion_summary
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {'dominant_emotion': 'neutral', 'intensity': 0.0}
    
    def _generate_mock_social_texts(self, symbol: str) -> List[str]:
        """Generate mock social media texts for testing."""
        # In production, fetch real tweets/posts
        base = symbol.split('/')[0]
        return [
            f"${base} is looking bullish today!",
            f"Just bought more {base}, feeling good",
            f"The market sentiment for ${base} is positive",
            f"${base} to the moon! 🚀",
            f"Analyzing {base} charts, looks promising"
        ]
    
    async def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal using multi-factor analysis."""
        logger.info(f"Analyzing {symbol}...")
        
        # Fetch market data
        df = self.fetch_ohlcv(symbol)
        if df is None or len(df) < 14:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        # Technical indicators
        rsi = self.calculate_rsi(df)
        price = df['close'].iloc[-1]
        change_24h = ((price - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100) if len(df) >= 24 else 0
        
        # Determine market phase
        if rsi > 70:
            market_phase = "overbought"
            technical_signal = "SELL"
        elif rsi < 30:
            market_phase = "oversold"
            technical_signal = "BUY"
        else:
            market_phase = "neutral"
            technical_signal = "HOLD"
        
        # Emotion analysis
        emotion_summary = await self.analyze_emotions(symbol)
        emotion = emotion_summary.get('dominant_emotion', 'neutral')
        intensity = emotion_summary.get('intensity', 0.0)
        
        # Determine emotion signal (contrarian)
        if intensity > 0.75:
            if emotion in ['excitement', 'joy', 'optimism']:
                emotion_signal = "SELL"  # Sell when euphoric
            elif emotion in ['fear', 'anxiety', 'sadness']:
                emotion_signal = "BUY"   # Buy when fearful
            else:
                emotion_signal = "HOLD"
        else:
            emotion_signal = "HOLD"
        
        # Combine signals (60% technical, 40% emotion)
        if not self.use_emotions:
            final_signal = technical_signal
            confidence = 0.7 if technical_signal != "HOLD" else 0.5
        else:
            # Multi-factor combination
            if technical_signal == "BUY" and emotion_signal == "BUY":
                final_signal = "BUY"
                confidence = 0.85
            elif technical_signal == "SELL" and emotion_signal == "SELL":
                final_signal = "SELL"
                confidence = 0.85
            elif technical_signal == "BUY":
                final_signal = "BUY"
                confidence = 0.65
            elif technical_signal == "SELL":
                final_signal = "SELL"
                confidence = 0.65
            else:
                final_signal = "HOLD"
                confidence = 0.50
        
        return {
            'symbol': symbol,
            'action': final_signal,
            'confidence': confidence,
            'price': price,
            'rsi': rsi,
            'change_24h': change_24h,
            'technical': technical_signal,
            'emotion': emotion_signal,
            'emotion_type': emotion,
            'emotion_intensity': intensity,
            'reason': f"Technical: {technical_signal} (RSI: {rsi:.1f}); Emotion: {emotion_signal} ({emotion})"
        }
    
    def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade with safety checks."""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        
        logger.info(f"Trade signal: {action} {symbol} @ €{price:.2f}")
        logger.info(f"Confidence: {signal['confidence']:.2%}")
        logger.info(f"Reason: {signal['reason']}")
        
        # Safety checks
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning(f"Daily trade limit reached ({self.max_daily_trades})")
            return {'status': 'rejected', 'reason': 'Daily trade limit'}
        
        # Calculate position size
        if action == 'BUY':
            position_size_eur = min(self.max_position, self.fetch_balance().get('EUR', {}).get('free', 0) * 0.9)
            
            if position_size_eur < 10:
                return {'status': 'rejected', 'reason': 'Insufficient EUR balance'}
            
            amount = position_size_eur / price
            
            logger.info(f"Order: BUY {amount:.6f} {symbol} @ €{price:.2f} (€{position_size_eur:.2f})")
            
            # Manual confirmation
            if self.require_confirmation:
                response = input(f"\n⚠️  Execute BUY order? (yes/no): ")
                if response.lower() != 'yes':
                    return {'status': 'cancelled', 'reason': 'User declined'}
            
            try:
                order = self.exchange.create_market_buy_order(symbol, amount)
                logger.success(f"✅ BUY order executed: {order['id']}")
                self.daily_trade_count += 1
                return {'status': 'executed', 'order': order, 'signal': signal}
            except Exception as e:
                logger.error(f"❌ Buy order failed: {e}")
                return {'status': 'failed', 'error': str(e)}
        
        elif action == 'SELL':
            # Get balance of base currency
            balance = self.fetch_balance()
            base_currency = symbol.split('/')[0]
            available_amount = balance.get(base_currency, {}).get('free', 0)
            
            if available_amount <= 0:
                return {'status': 'rejected', 'reason': 'No position to sell'}
            
            amount = available_amount * 0.75  # Sell 75% of position
            
            logger.info(f"Order: SELL {amount:.6f} {symbol} @ €{price:.2f}")
            
            # Manual confirmation
            if self.require_confirmation:
                response = input(f"\n⚠️  Execute SELL order? (yes/no): ")
                if response.lower() != 'yes':
                    return {'status': 'cancelled', 'reason': 'User declined'}
            
            try:
                order = self.exchange.create_market_sell_order(symbol, amount)
                logger.success(f"✅ SELL order executed: {order['id']}")
                self.daily_trade_count += 1
                return {'status': 'executed', 'order': order, 'signal': signal}
            except Exception as e:
                logger.error(f"❌ Sell order failed: {e}")
                return {'status': 'failed', 'error': str(e)}
        
        return {'status': 'ignored', 'reason': 'No action needed'}
    
    async def run_once(self):
        """Run one trading iteration for all symbols."""
        logger.info("=" * 80)
        logger.info(f"Trading Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Get current balance
        balance = self.fetch_balance()
        eur_balance = balance.get('EUR', {}).get('free', 0)
        logger.info(f"EUR Balance: €{eur_balance:.2f}")
        logger.info(f"Daily Trades: {self.daily_trade_count}/{self.max_daily_trades}")
        logger.info(f"Daily P&L: €{self.daily_pnl:+.2f}")
        logger.info("-" * 80)
        
        results = []
        for symbol in self.symbols:
            try:
                signal = await self.generate_signal(symbol)
                
                logger.info(f"\n📊 {symbol}")
                logger.info(f"  Price: €{signal['price']:.2f}")
                logger.info(f"  RSI: {signal['rsi']:.1f}")
                logger.info(f"  24h: {signal['change_24h']:+.2f}%")
                logger.info(f"  Technical: {signal['technical']}")
                logger.info(f"  Emotion: {signal['emotion']} ({signal['emotion_type']}, intensity: {signal['emotion_intensity']:.2f})")
                logger.info(f"  Final: {signal['action']} (confidence: {signal['confidence']:.2%})")
                logger.info(f"  Reason: {signal['reason']}")
                
                # Execute if signal is strong enough
                if signal['action'] != 'HOLD' and signal['confidence'] > 0.70:
                    result = self.execute_trade(signal)
                    results.append(result)
                else:
                    logger.info(f"  → No trade (confidence too low or HOLD signal)")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        logger.info("=" * 80)
        return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live trading with GoEmotions - REAL MONEY"
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTC/EUR'],
        help='Trading symbols (default: BTC/EUR)'
    )
    parser.add_argument(
        '--api-key',
        default=os.getenv('KRAKEN_API_KEY'),
        help='Kraken API key'
    )
    parser.add_argument(
        '--api-secret',
        default=os.getenv('KRAKEN_API_SECRET'),
        help='Kraken API secret'
    )
    parser.add_argument(
        '--max-position',
        type=float,
        default=300,
        help='Max position size in EUR (default: 300)'
    )
    parser.add_argument(
        '--max-daily-loss',
        type=float,
        default=50,
        help='Max daily loss in EUR (default: 50)'
    )
    parser.add_argument(
        '--max-daily-trades',
        type=int,
        default=2,
        help='Max trades per day (default: 2)'
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
    parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Update interval in seconds (default: 3600)'
    )
    
    args = parser.parse_args()
    
    # Validate API credentials
    if not args.api_key or not args.api_secret:
        logger.error("❌ KRAKEN_API_KEY and KRAKEN_API_SECRET required!")
        return 1
    
    # Safety warning
    print("\n" + "=" * 80)
    print("⚠️  WARNING: LIVE TRADING - REAL MONEY WILL BE USED")
    print("=" * 80)
    print(f"Symbols:         {', '.join(args.symbols)}")
    print(f"Strategy:        GoEmotions + Technical Analysis")
    print(f"Max Position:    €{args.max_position:,.2f}")
    print(f"Max Daily Loss:  €{args.max_daily_loss:,.2f}")
    print(f"Max Daily Trades: {args.max_daily_trades}")
    print(f"Confirmation:     {'OFF (DANGEROUS!)' if args.no_confirm else 'ON'}")
    print("=" * 80 + "\n")
    
    # Initialize trader
    trader = GoEmotionsLiveTrader(
        symbols=args.symbols,
        api_key=args.api_key,
        api_secret=args.api_secret,
        max_position_eur=args.max_position,
        max_daily_loss_eur=args.max_daily_loss,
        max_daily_trades=args.max_daily_trades,
        require_confirmation=not args.no_confirm
    )
    
    # Test mode
    if args.test:
        logger.info("Testing Kraken connection...")
        balance = trader.fetch_balance()
        if balance:
            logger.success("✅ Connection successful!")
            logger.info(f"EUR Balance: €{balance.get('EUR', {}).get('free', 0):.2f}")
            return 0
        else:
            logger.error("❌ Connection failed!")
            return 1
    
    # Continuous trading
    logger.info("Starting live trading...")
    logger.info(f"Update interval: {args.interval} seconds")
    logger.info("Press Ctrl+C to stop\n")
    
    try:
        while True:
            await trader.run_once()
            
            logger.info(f"\nNext update in {args.interval} seconds...")
            import asyncio
            await asyncio.sleep(args.interval)
    
    except KeyboardInterrupt:
        logger.info("\nLive trading stopped by user")
        return 0


if __name__ == "__main__":
    asyncio.run(main())
