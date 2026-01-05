#!/usr/bin/env python3
"""
GraphWiz Trader - Extended Paper Trading with GoEmotions Strategy

This script runs a comprehensive 24-72 hour paper trading validation
that combines technical analysis (RSI) with sentiment analysis (GoEmotions)
using real market data from Kraken (MiCA licensed for Germany).

Features:
- Real-time market data from Kraken (BTC/EUR, ETH/EUR)
- GoEmotions-based sentiment analysis
- Technical indicators (RSI, MACD)
- Multi-factor signal generation
- Extended run time (24-72 hours)
- Performance tracking & metrics
- Trade logging & equity curves
- German exchange compliance (Kraken)

Usage:
    # Run for 24 hours
    python run_extended_paper_trading_goemotions.py --duration 24

    # Run for 72 hours (recommended)
    python run_extended_paper_trading_goemotions.py --duration 72

    # Custom symbols
    python run_extended_paper_trading_goemotions.py --duration 48 --symbols BTC/EUR ETH/EUR

    # Run in background
    nohup python run_extended_paper_trading_goemotions.py --duration 72 > goemotions_paper_trading.log 2>&1 &
"""

import sys
import os
import asyncio
import signal
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from loguru import logger
    import ccxt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install dependencies:")
    print("  pip install loguru ccxt pandas numpy")
    sys.exit(1)

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
    # Fallback to direct imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src" / "graphwiz_trader"))
    from sentiment.goemotions_analyzer import GoEmotionsAnalyzer
    from strategies.emotion_strategy import (
        EmotionBasedStrategy,
        EmotionStrategyConfig,
        MarketPhase,
        EmotionSignal
    )


class GoEmotionsPaperTrader:
    """
    Extended paper trading with GoEmotions strategy and real market data.

    Combines technical analysis (RSI, MACD) with sentiment analysis (GoEmotions)
    for intelligent trading decisions using Kraken exchange data.
    """

    def __init__(
        self,
        duration_hours: int = 24,
        symbols: List[str] = None,
        initial_capital_eur: float = 10000.0,
        update_interval_minutes: int = 30
    ):
        """
        Initialize the GoEmotions paper trader.

        Args:
            duration_hours: How long to run (default: 24 hours)
            symbols: Trading pairs (default: BTC/EUR, ETH/EUR)
            initial_capital_eur: Starting capital in EUR
            update_interval_minutes: How often to check markets (default: 30 min)
        """
        self.duration_hours = duration_hours
        self.symbols = symbols or ["BTC/EUR", "ETH/EUR"]
        self.initial_capital = initial_capital_eur
        self.update_interval = update_interval_minutes * 60  # Convert to seconds

        # State
        self.running = False
        self.start_time = None
        self.stop_time = None

        # Portfolio tracking
        self.portfolio = {
            "EUR": initial_capital_eur,
        }
        for symbol in self.symbols:
            base = symbol.split('/')[0]
            self.portfolio[base] = 0.0

        # Trade tracking
        self.trades = []
        self.equity_curve = []
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0
        }

        # Initialize GoEmotions strategy
        config = EmotionStrategyConfig(
            extreme_euphoria_threshold=0.75,
            extreme_fear_threshold=0.75,
            use_contrarian_signals=True,
            min_data_points=5,
            max_emotion_position_pct=0.25  # Max 25% of balance per trade
        )
        self.emotion_strategy = EmotionBasedStrategy(config=config)
        self.emotion_analyzer = GoEmotionsAnalyzer()

        # Exchange (Kraken - German approved)
        self.exchange = self._setup_exchange()

        # Setup logging
        self._setup_logging()

        # Simulated social media data (in production, fetch from Reddit/Twitter APIs)
        self._init_simulated_social_media()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_exchange(self) -> ccxt.Exchange:
        """Setup Kraken exchange for paper trading."""
        exchange = ccxt.kraken({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            },
        })
        logger.info(f"Exchange initialized: {exchange.name} (MiCA licensed for Germany)")
        return exchange

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path("logs/paper_trading")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main log file
        log_file = log_dir / f"goemotions_validation_{timestamp}.log"
        logger.add(
            str(log_file),
            rotation="100 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

        # Trade log
        trade_log = log_dir / f"goemotions_trades_{timestamp}.csv"
        self.trade_log_file = str(trade_log)

        # Equity curve log
        equity_log = log_dir / f"goemotions_equity_{timestamp}.csv"
        self.equity_log_file = str(equity_log)

        logger.info(f"Logging initialized: {log_file}")
        logger.info(f"Trade log: {self.trade_log_file}")
        logger.info(f"Equity log: {self.equity_log_file}")

    def _init_simulated_social_media(self):
        """Initialize simulated social media posts by market phase."""
        self.social_media_templates = {
            'capitulation': [
                "I've lost everything. My life savings are gone. This is pure hell.",
                "Bitcoin is dead. Going to zero. Ruined my life. Why did I believe?",
                "Complete despair. Panic selling. This is the worst thing ever.",
                "Scam, Ponzi, bubble has burst. Getting out while I can.",
                "Devastated. Heartbroken. My portfolio is destroyed.",
            ],
            'accumulation': [
                "Building my position slowly. Bitcoin has great technology and potential.",
                "Great buying opportunity here. Long-term believer.",
                "Accumulating while cheap. Patience is key.",
                "Undervalued at these levels. Smart money buying.",
                "The future looks bright. Time to be greedy when others are fearful.",
            ],
            'markup': [
                "Bitcoin breaking out! Excitement building! 🚀",
                "Joy to see green! Technicals looking great! 📈",
                "Community enthusiasm is high! Good momentum! ⚡",
                "Finally recovering! This is the start of something big!",
                "Bullish pattern forming. Getting excited!",
            ],
            'distribution': [
                "🚀🚀🚀 TO THE MOON! Bitcoin is unstoppable! Going to infinity!",
                "Lambo shopping spree! We're all gonna be rich! 💎🙌",
                "This is the future! Easy money printing! Never selling!",
                "Parabolic gains! This never goes down! To the moon! 🔥",
                "100x incoming! We're all rich! Diamond hands! 🤑",
            ],
            'markdown': [
                "Bitcoin is dropping again. Getting nervous about my position.",
                "This support isn't holding. Concerned about more downside.",
                "Why is it crashing? Is this the end?",
                "Looking weak. Thinking about selling.",
                "Disappointed by the lack of recovery.",
            ]
        }

    def _get_social_media_sentiment(self, market_phase: str, count: int = 5) -> List[str]:
        """Get simulated social media posts based on market phase."""
        if market_phase in self.social_media_templates:
            templates = self.social_media_templates[market_phase]
            return random.sample(templates, min(count, len(templates)))
        return []

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch current market data from Kraken."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate indicators
            rsi = self._calculate_rsi(df['close'], 14)
            macd, signal_line = self._calculate_macd(df['close'])

            return {
                'symbol': symbol,
                'price': ticker['last'],
                'volume': ticker['baseVolume'],
                'change_24h': ticker['percentage'],
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal_line,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except Exception:
            return 50.0

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
        try:
            exp_fast = prices.ewm(span=fast).mean()
            exp_slow = prices.ewm(span=slow).mean()
            macd = exp_fast - exp_slow
            signal_line = macd.ewm(span=signal).mean()
            return macd.iloc[-1], signal_line.iloc[-1]
        except Exception:
            return 0.0, 0.0

    def _determine_market_phase(self, rsi: float, change_24h: float) -> str:
        """Determine market phase based on technical indicators."""
        if rsi < 30 and change_24h < -5:
            return 'capitulation'
        elif rsi < 40 and change_24h < 0:
            return 'accumulation'
        elif rsi > 70:
            return 'distribution'
        elif rsi > 60 and change_24h > 0:
            return 'markup'
        elif rsi < 50 and change_24h < 0:
            return 'markdown'
        else:
            return 'markup'  # Default

    async def _analyze_signals(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate multi-factor trading signals.

        Combines:
        1. Technical analysis (RSI, MACD)
        2. Sentiment analysis (GoEmotions)
        3. Market phase identification
        """
        rsi = market_data.get('rsi', 50)
        change_24h = market_data.get('change_24h', 0)
        price = market_data.get('price', 0)

        # Determine market phase
        market_phase = self._determine_market_phase(rsi, change_24h)

        # Get social media sentiment for this phase
        social_texts = self._get_social_media_sentiment(market_phase, count=5)

        logger.info(f"    Market Phase: {market_phase.upper()}")
        logger.info(f"    Social Media: Analyzing {len(social_texts)} posts...")

        # Analyze emotions from social media
        emotions_detected = []
        for text in social_texts[:3]:  # Show first 3
            profile = self.emotion_analyzer.detect_emotions(text)
            emotions_detected.append(profile)
            preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(f"      \"{preview}\" → {profile.dominant_emotion.value} "
                       f"(intensity: {profile.emotional_intensity:.2f})")

        # Store emotions in strategy
        await self.emotion_strategy.analyze_emotions_for_symbol(
            symbol,
            social_texts,
            [datetime.now()] * len(social_texts)
        )

        # Get emotion summary
        emotion_summary = self.emotion_strategy.get_market_emotion_summary(symbol)

        logger.info(f"    Dominant Emotion: {emotion_summary['dominant_emotion']}")
        logger.info(f"    Emotional Intensity: {emotion_summary['emotional_intensity']:.2f}")
        logger.info(f"    Trading Bias: {emotion_summary['trading_bias']}")

        # Generate emotion-based signal
        emotion_signal = self.emotion_strategy.generate_signal(
            symbol,
            price,
            self.portfolio["EUR"]
        )

        # Combine technical and emotion signals
        technical_signal = self._get_technical_signal(rsi, change_24h)

        # Multi-factor decision
        final_signal = self._combine_signals(
            technical_signal,
            emotion_signal,
            rsi,
            change_24h,
            price,
            emotion_summary
        )

        return final_signal

    def _get_technical_signal(self, rsi: float, change_24h: float) -> str:
        """Get signal from technical analysis."""
        if rsi < 35:
            return "STRONG_BUY"
        elif rsi < 45:
            return "BUY"
        elif rsi > 65:
            return "STRONG_SELL"
        elif rsi > 55:
            return "SELL"
        else:
            return "HOLD"

    def _combine_signals(
        self,
        technical_signal: str,
        emotion_signal: Any,
        rsi: float,
        change_24h: float,
        price: float,
        emotion_summary: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Combine technical and emotion signals into final decision."""
        emotion_action = emotion_signal.signal.value if emotion_signal else "hold"

        # Signal priority matrix
        signal_strength = {
            "STRONG_BUY": 2,
            "BUY": 1,
            "HOLD": 0,
            "SELL": -1,
            "STRONG_SELL": -2
        }

        tech_strength = signal_strength.get(technical_signal, 0)
        emotion_strength = signal_strength.get(emotion_action, 0)

        # Combined signal (weighted average: 60% technical, 40% emotion)
        combined_strength = (tech_strength * 0.6) + (emotion_strength * 0.4)

        # Determine final action
        if combined_strength >= 1.5:
            final_action = "STRONG_BUY"
            confidence = 0.85
        elif combined_strength >= 0.5:
            final_action = "BUY"
            confidence = 0.75
        elif combined_strength <= -1.5:
            final_action = "STRONG_SELL"
            confidence = 0.85
        elif combined_strength <= -0.5:
            final_action = "SELL"
            confidence = 0.75
        else:
            final_action = "HOLD"
            confidence = 0.50

        # Build reasoning
        reasoning = []
        reasoning.append(f"Technical: {technical_signal} (RSI: {rsi:.1f})")

        dominant_emotion = emotion_summary['dominant_emotion'] if emotion_summary else 'N/A'
        reasoning.append(f"Emotion: {emotion_action.upper()} ({dominant_emotion})")

        if emotion_signal and emotion_signal.contrarian_indicator:
            reasoning.append("⚠️  CONTRARIAN: Extreme emotion detected")

        return {
            'action': final_action,
            'confidence': confidence,
            'reasoning': reasoning,
            'technical_signal': technical_signal,
            'emotion_signal': emotion_action,
            'price': price,
            'contrarian': emotion_signal.contrarian_indicator if emotion_signal else False
        }

    def _execute_trade(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a virtual trade."""
        action = signal['action']
        price = signal['price']
        base_currency = symbol.split('/')[0]

        # Simulate commission and slippage
        commission_rate = 0.0016  # Kraken taker fee
        slippage_rate = 0.0005

        if action in ["BUY", "STRONG_BUY"] and self.portfolio["EUR"] > 100:
            # Position sizing based on signal strength
            if action == "STRONG_BUY":
                position_pct = 0.30  # 30% for strong buy
            else:
                position_pct = 0.20  # 20% for regular buy

            # Apply contrarian boost if applicable
            if signal.get('contrarian'):
                position_pct *= 1.3  # +30% for contrarian signals

            # Cap at max position
            position_pct = min(position_pct, 0.30)

            amount_eur = self.portfolio["EUR"] * position_pct
            amount = amount_eur / price

            # Apply commission and slippage
            actual_price = price * (1 + slippage_rate)
            commission = amount_eur * commission_rate
            total_cost = amount_eur + commission

            if total_cost <= self.portfolio["EUR"]:
                self.portfolio["EUR"] -= total_cost
                self.portfolio[base_currency] += amount

                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'price': actual_price,
                    'amount': amount,
                    'total_eur': total_cost,
                    'commission': commission,
                    'reasoning': '; '.join(signal['reasoning']),
                    'confidence': signal['confidence'],
                    'contrarian': signal.get('contrarian', False)
                }

                self.trades.append(trade)
                contrarian_note = " [CONTRARIAN]" if signal.get('contrarian') else ""
                logger.success(f"    ✅ {action} {symbol}: {amount:.6f} @ €{actual_price:,.2f} "
                             f"(€{amount_eur:.2f}){contrarian_note}")

                return trade

        elif action in ["SELL", "STRONG_SELL"] and self.portfolio[base_currency] > 0:
            # Sell percentage based on signal strength
            sell_pct = 0.75 if action == "STRONG_SELL" else 0.50
            amount = self.portfolio[base_currency] * sell_pct

            # Apply commission and slippage
            actual_price = price * (1 - slippage_rate)
            proceeds = amount * actual_price
            commission = proceeds * commission_rate
            net_proceeds = proceeds - commission

            # Calculate P&L
            avg_price = self._calculate_average_price(symbol, base_currency)
            pnl = (actual_price - avg_price) * amount
            pnl_pct = ((actual_price - avg_price) / avg_price) * 100

            self.portfolio[base_currency] -= amount
            self.portfolio["EUR"] += net_proceeds

            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'price': actual_price,
                'amount': amount,
                'total_eur': net_proceeds,
                'commission': commission,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reasoning': '; '.join(signal['reasoning']),
                'confidence': signal['confidence'],
                'contrarian': signal.get('contrarian', False)
            }

            self.trades.append(trade)

            if pnl > 0:
                self.metrics['winning_trades'] += 1
                logger.success(f"    ✅ SELL {symbol}: {amount:.6f} @ €{actual_price:,.2f} | "
                             f"P&L: +€{pnl:.2f} (+{pnl_pct:.2f}%)")
            else:
                self.metrics['losing_trades'] += 1
                logger.warning(f"    ⚠️  SELL {symbol}: {amount:.6f} @ €{actual_price:,.2f} | "
                             f"P&L: -€{abs(pnl):.2f} ({pnl_pct:.2f}%)")

            return trade

        return None

    def _calculate_average_price(self, symbol: str, base_currency: str) -> float:
        """Calculate average entry price for a position."""
        symbol_trades = [t for t in self.trades if t['symbol'] == symbol and t['action'] in ['BUY', 'STRONG_BUY']]
        if not symbol_trades:
            return 0.0

        total_amount = sum(t['amount'] for t in symbol_trades)
        total_cost = sum(t['total_eur'] for t in symbol_trades)
        return total_cost / total_amount if total_amount > 0 else 0.0

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value in EUR."""
        total = self.portfolio["EUR"]

        for symbol in self.symbols:
            base_currency = symbol.split('/')[0]
            if self.portfolio[base_currency] > 0:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    total += self.portfolio[base_currency] * ticker['last']
                except Exception:
                    pass

        return total

    def _update_metrics(self):
        """Update performance metrics."""
        current_value = self._calculate_portfolio_value()

        # Total return
        self.metrics['total_return'] = ((current_value - self.initial_capital) / self.initial_capital) * 100

        # Win rate
        total_closed = self.metrics['winning_trades'] + self.metrics['losing_trades']
        if total_closed > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / total_closed) * 100

        # Max drawdown
        if len(self.equity_curve) > 1:
            peak = max(e['value'] for e in self.equity_curve)
            if current_value < peak:
                self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], ((peak - current_value) / peak) * 100)

        # Sharpe ratio
        if len(self.equity_curve) > 10:
            returns = [self.equity_curve[i]['value'] - self.equity_curve[i-1]['value']
                      for i in range(1, len(self.equity_curve))]
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    self.metrics['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252)

    def _log_status(self):
        """Log current status."""
        current_value = self._calculate_portfolio_value()
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        remaining_hours = max(0, self.duration_hours - elapsed_hours)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info(f"Elapsed:    {elapsed_hours:.1f}h / {self.duration_hours}h "
                   f"({elapsed_hours/self.duration_hours*100:.1f}%)")
        logger.info(f"Remaining:  {remaining_hours:.1f} hours")
        logger.info("")
        logger.info(f"Portfolio:")
        logger.info(f"  Value:     €{current_value:,.2f}")
        logger.info(f"  Return:    {self.metrics['total_return']:+.2f}%")
        logger.info(f"  Drawdown:  {self.metrics['max_drawdown']:.2f}%")
        logger.info("")
        logger.info(f"Trading:")
        logger.info(f"  Total:     {len(self.trades)} trades")
        logger.info(f"  Wins:      {self.metrics['winning_trades']} ({self.metrics['win_rate']:.1f}%)")
        logger.info(f"  Losses:    {self.metrics['losing_trades']}")
        logger.info("")
        logger.info(f"Metrics:")
        logger.info(f"  Sharpe:    {self.metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Win Rate:  {self.metrics['win_rate']:.1f}%")
        logger.info("=" * 80)
        logger.info("")

        # Log to equity curve
        equity_record = {
            'timestamp': datetime.now().isoformat(),
            'value': current_value,
            'return_pct': self.metrics['total_return'],
            'trades_count': len(self.trades)
        }
        self.equity_curve.append(equity_record)

        # Save to file
        self._save_logs()

    def _save_logs(self):
        """Save logs to files."""
        # Save trades to CSV
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            df_trades.to_csv(self.trade_log_file, index=False)

        # Save equity curve to CSV
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            df_equity.to_csv(self.equity_log_file, index=False)

    def _save_final_report(self):
        """Save final validation report."""
        current_value = self._calculate_portfolio_value()
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        report = {
            'validation_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.stop_time.isoformat() if self.stop_time else None,
                'duration_hours': elapsed_hours,
                'target_duration_hours': self.duration_hours,
                'completion_pct': (elapsed_hours / self.duration_hours) * 100
            },
            'portfolio': {
                'initial_capital_eur': self.initial_capital,
                'final_value_eur': current_value,
                'total_return_eur': current_value - self.initial_capital,
                'total_return_pct': self.metrics['total_return']
            },
            'trading': {
                'total_trades': len(self.trades),
                'winning_trades': self.metrics['winning_trades'],
                'losing_trades': self.metrics['losing_trades'],
                'win_rate_pct': self.metrics['win_rate']
            },
            'metrics': {
                'max_drawdown_pct': self.metrics['max_drawdown'],
                'sharpe_ratio': self.metrics['sharpe_ratio'],
                'final_portfolio': self.portfolio
            },
            'status': self._get_status()
        }

        # Save report
        report_dir = Path("data/paper_trading")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"goemotions_validation_report_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.success(f"Final report saved: {report_file}")
        logger.success(f"View: cat {report_file}")

        return report_file

    def _get_status(self) -> str:
        """Get validation status based on metrics."""
        if self.metrics['total_return'] > 5 and self.metrics['max_drawdown'] < 10:
            return "EXCELLENT - Ready for live trading"
        elif self.metrics['total_return'] > 0 and self.metrics['max_drawdown'] < 15:
            return "GOOD - Consider live trading with caution"
        elif self.metrics['total_return'] < 0:
            return "POOR - Not ready for live trading"
        else:
            return "MODERATE - Continue validation"

    async def run(self):
        """Run the extended paper trading validation."""
        self.start_time = datetime.now()
        self.running = True

        logger.success("")
        logger.success("=" * 80)
        logger.success("EXTENDED PAPER TRADING - GOEMOTIONS STRATEGY")
        logger.success("=" * 80)
        logger.success(f"Start Time:    {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.success(f"Duration:      {self.duration_hours} hours")
        logger.success(f"Symbols:       {', '.join(self.symbols)}")
        logger.success(f"Capital:       €{self.initial_capital:,.2f}")
        logger.success(f"Update Interval: {self.update_interval / 60:.0f} minutes")
        logger.success(f"Exchange:      Kraken (MiCA licensed Germany)")
        logger.success(f"Strategy:      GoEmotions + Technical Analysis")
        logger.success("=" * 80)
        logger.success("")

        iteration = 0

        try:
            while self.running:
                iteration += 1
                elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600

                # Check if we've reached the target duration
                if elapsed_hours >= self.duration_hours:
                    logger.success(f"Target duration reached: {elapsed_hours:.1f} hours")
                    break

                logger.info(f"[Iteration {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Analyze each symbol
                for symbol in self.symbols:
                    logger.info(f"  📊 {symbol}")

                    # Fetch market data
                    market_data = self._get_market_data(symbol)
                    if market_data:
                        logger.info(f"    Price: €{market_data['price']:,.2f} | "
                                   f"RSI: {market_data['rsi']:.1f} | "
                                   f"24h: {market_data['change_24h']:+.2f}%")

                        # Generate multi-factor signal
                        signal = await self._analyze_signals(symbol, market_data)

                        logger.info(f"    Technical: {signal['technical_signal']}")
                        logger.info(f"    Emotion:   {signal['emotion_signal']}")
                        logger.info(f"    Final:     {signal['action']} "
                                   f"(confidence: {signal['confidence']:.2f})")
                        logger.info(f"    Reasoning: {'; '.join(signal['reasoning'])}")

                        # Execute trade if signal is strong enough
                        if signal['action'] != 'HOLD' and signal['confidence'] > 0.65:
                            self._execute_trade(symbol, signal)
                        else:
                            logger.info(f"    📊 No trade (signal: {signal['action']}, "
                                       f"confidence: {signal['confidence']:.2f})")

                    await asyncio.sleep(2)  # Small delay between symbols

                # Update metrics and log status
                self._update_metrics()
                self._log_status()

                # Calculate time until next update
                logger.info(f"Next update in {self.update_interval / 60:.0f} minutes...")
                logger.info("")

                # Wait for next iteration
                await asyncio.sleep(self.update_interval)

        except Exception as e:
            logger.exception(f"Fatal error during validation: {e}")
        finally:
            self.stop_time = datetime.now()
            self.running = False
            self._save_final_report()

            logger.success("")
            logger.success("=" * 80)
            logger.success("VALIDATION COMPLETE")
            logger.success("=" * 80)
            logger.success(f"Total Duration: {elapsed_hours:.1f} hours")
            logger.success(f"Total Trades: {len(self.trades)}")
            logger.success(f"Final Return: {self.metrics['total_return']:+.2f}%")
            logger.success(f"Status: {self._get_status()}")
            logger.success("=" * 80)
            logger.success("")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run extended paper trading with GoEmotions strategy"
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=24,
        help='Duration in hours (default: 24)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTC/EUR', 'ETH/EUR'],
        help='Trading symbols (default: BTC/EUR ETH/EUR)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital in EUR (default: 10000)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Update interval in minutes (default: 30)'
    )

    args = parser.parse_args()

    trader = GoEmotionsPaperTrader(
        duration_hours=args.duration,
        symbols=args.symbols,
        initial_capital_eur=args.capital,
        update_interval_minutes=args.interval
    )

    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
