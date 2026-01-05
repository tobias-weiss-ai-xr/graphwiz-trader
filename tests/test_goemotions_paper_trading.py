#!/usr/bin/env python3
"""Test GoEmotions strategy with paper trading simulation.

This script demonstrates the GoEmotions-based trading strategy in a
realistic paper trading environment with simulated market data and
social media sentiment.
"""

import sys
import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graphwiz_trader.sentiment.goemotions_analyzer import GoEmotionsAnalyzer
from graphwiz_trader.strategies.emotion_strategy import (
    EmotionBasedStrategy,
    EmotionStrategyConfig,
    MarketPhase,
    EmotionSignal
)


class PaperTrader:
    """Simulated paper trading engine with GoEmotions strategy."""

    def __init__(self, initial_balance_eur=10000):
        self.balance_eur = initial_balance_eur
        self.btc_holdings = 0.0
        self.trades = []
        self.start_time = datetime.now()

        # Initialize GoEmotions strategy
        config = EmotionStrategyConfig(
            extreme_euphoria_threshold=0.75,
            extreme_fear_threshold=0.75,
            use_contrarian_signals=True,
            min_data_points=3,  # Lower for demo
            max_emotion_position_pct=0.30  # Max 30% of balance
        )

        self.strategy = EmotionBasedStrategy(config=config)
        self.analyzer = GoEmotionsAnalyzer()

        logger.info("GoEmotions Paper Trading Initialized")
        logger.info(f"Initial Balance: €{self.balance_eur:,.2f}")
        logger.info(f"Strategy: Contrarian signals enabled")
        logger.info(f"Max Position: 30% of balance")
        logger.info("")

    def simulate_market_data(self, iteration):
        """Simulate realistic BTC price movement."""
        # Base price around €45,000 with realistic volatility
        base_price = 45000.0

        # Add trend based on iteration
        trend = 0
        if iteration < 5:
            trend = -500  # Downtrend (fear building)
        elif iteration < 10:
            trend = -1000  # Capitulation
        elif iteration < 15:
            trend = 500  # Recovery
        elif iteration < 20:
            trend = 1000  # Uptrend
        else:
            trend = 2000  # Euphoria

        # Add noise
        noise = random.uniform(-500, 500)

        return base_price + trend + noise

    def generate_social_media_texts(self, price, iteration):
        """Generate realistic social media texts based on market state."""

        # Market phase determination based on price and iteration
        if iteration < 5:
            # Early markdown - concern building
            texts = [
                "Bitcoin is dropping again. Getting nervous about my position.",
                "This support isn't holding. Concerned about more downside.",
                "Why is it crashing? Is this the end?",
                "Looking weak. Thinking about selling.",
                "Disappointed by the lack of recovery."
            ]
        elif iteration < 10:
            # Capitulation - extreme fear
            texts = [
                "I've lost everything. My life savings are gone. This is pure hell.",
                "Bitcoin is dead. Going to zero. Ruined my life. Why did I believe?",
                "Complete despair. Panic selling. This is the worst thing ever.",
                "Scam, Ponzi, bubble has burst. Getting out while I can.",
                "Devastated. Heartbroken. My portfolio is destroyed."
            ]
        elif iteration < 15:
            # Accumulation - hope returning
            texts = [
                "Building my position slowly. Bitcoin has great technology and potential.",
                "Great buying opportunity here. Long-term believer.",
                "Accumulating while cheap. Patience is key.",
                "Undervalued at these levels. Smart money buying.",
                "The future looks bright. Time to be greedy when others are fearful."
            ]
        elif iteration < 20:
            # Markup - excitement
            texts = [
                "Bitcoin breaking out! Excitement building! 🚀",
                "Joy to see green! Technicals looking great! 📈",
                "Community enthusiasm is high! Good momentum! ⚡",
                "Finally recovering! This is the start of something big!",
                "Bullish pattern forming. Getting excited!"
            ]
        else:
            # Distribution - euphoria
            texts = [
                "🚀🚀🚀 TO THE MOON! Bitcoin is unstoppable! Going to infinity!",
                "Lambo shopping spree! We're all gonna be rich! 💎🙌",
                "This is the future! Easy money printing! Never selling!",
                "Parabolic gains! This never goes down! To the moon! 🔥",
                "100x incoming! We're all rich! Diamond hands! 🤑"
            ]

        return texts

    async def run_iteration(self, iteration):
        """Run one paper trading iteration."""
        timestamp = datetime.now()
        current_price = self.simulate_market_data(iteration)

        logger.info(f"\n{'='*80}")
        logger.info(f"Iteration {iteration} | {timestamp.strftime('%H:%M:%S')}")
        logger.info(f"BTC Price: €{current_price:,.2f}")
        logger.info(f"Portfolio: €{self.balance_eur:,.2f} + {self.btc_holdings:.4f} BTC")
        logger.info(f"{'='*80}")

        # Generate social media texts for this market state
        texts = self.generate_social_media_texts(current_price, iteration)
        logger.info(f"\n📱 Social Media Activity ({len(texts)} posts):")

        # Analyze emotions from texts
        emotions_detected = []
        for i, text in enumerate(texts[:3], 1):  # Show first 3
            profile = self.analyzer.detect_emotions(text)
            emotions_detected.append(profile)

            # Show preview of text
            preview = text[:60] + "..." if len(text) > 60 else text
            logger.info(f"  {i}. \"{preview}\"")
            logger.info(f"     → {profile.dominant_emotion.value} "
                       f"(intensity: {profile.emotional_intensity:.2f})")

        # Store emotions in strategy
        await self.strategy.analyze_emotions_for_symbol(
            'BTC',
            texts,
            [timestamp] * len(texts)
        )

        # Get market emotion summary
        summary = self.strategy.get_market_emotion_summary('BTC')

        logger.info(f"\n📊 Market Emotion Summary:")
        logger.info(f"  Data Points: {summary['data_points']}")
        logger.info(f"  Dominant Emotion: {summary['dominant_emotion']}")
        logger.info(f"  Intensity: {summary['emotional_intensity']:.2f}")
        logger.info(f"  Trading Bias: {summary['trading_bias']}")
        logger.info(f"  Market Phase: {summary['market_phase']}")

        # Generate trading signal
        signal_result = self.strategy.generate_signal(
            'BTC',
            current_price,
            self.balance_eur
        )

        if signal_result:
            logger.info(f"\n🎯 Trading Signal:")
            logger.info(f"  Signal: {signal_result.signal.value}")
            logger.info(f"  Confidence: {signal_result.confidence:.2f}")
            logger.info(f"  Market Phase: {signal_result.market_phase.value}")

            if signal_result.contrarian_indicator:
                logger.warning(f"  ⚠️  CONTRARIAN INDICATOR: "
                             f"Extreme emotion suggests reversal!")

            logger.info(f"  Reasoning:")
            for reason in signal_result.reasoning:
                logger.info(f"    • {reason}")

            # Calculate position size
            position_size = self.strategy.calculate_position_size(
                signal_result,
                current_price,
                self.balance_eur,
                250  # Base position
            )

            logger.info(f"\n💰 Position Sizing:")
            logger.info(f"  Base Position: €250.00")
            logger.info(f"  Signal Multiplier: {signal_result.signal.value}")

            if signal_result.signal == EmotionSignal.BUY:
                # Check if we have enough balance
                if position_size <= self.balance_eur:
                    btc_amount = position_size / current_price
                    self.balance_eur -= position_size
                    self.btc_holdings += btc_amount

                    trade = {
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': current_price,
                        'amount_eur': position_size,
                        'btc_amount': btc_amount,
                        'signal': signal_result.signal.value,
                        'emotion': summary['dominant_emotion'],
                        'phase': summary['market_phase'],
                        'contrarian': signal_result.contrarian_indicator
                    }
                    self.trades.append(trade)

                    logger.success(f"  ✅ EXECUTED: BUY {btc_amount:.4f} BTC "
                                 f"@ €{current_price:,.2f} (€{position_size:.2f})")
                else:
                    logger.warning(f"  ⚠️  INSUFFICIENT FUNDS: Need €{position_size:.2f}, "
                                 f"have €{self.balance_eur:.2f}")

            elif signal_result.signal == EmotionSignal.SELL:
                # Check if we have BTC to sell
                if self.btc_holdings > 0:
                    sell_amount = min(self.btc_holdings, position_size / current_price)
                    sell_value = sell_amount * current_price

                    self.btc_holdings -= sell_amount
                    self.balance_eur += sell_value

                    trade = {
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': current_price,
                        'amount_eur': sell_value,
                        'btc_amount': sell_amount,
                        'signal': signal_result.signal.value,
                        'emotion': summary['dominant_emotion'],
                        'phase': summary['market_phase'],
                        'contrarian': signal_result.contrarian_indicator
                    }
                    self.trades.append(trade)

                    logger.success(f"  ✅ EXECUTED: SELL {sell_amount:.4f} BTC "
                                 f"@ €{current_price:,.2f} (€{sell_value:.2f})")
                else:
                    logger.warning(f"  ⚠️  NO BTC HOLDINGS: Signal to sell but no BTC to sell")

            elif signal_result.signal == EmotionSignal.STRONG_BUY:
                # Larger position for strong buy
                if position_size <= self.balance_eur:
                    btc_amount = position_size / current_price
                    self.balance_eur -= position_size
                    self.btc_holdings += btc_amount

                    trade = {
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'action': 'STRONG_BUY',
                        'price': current_price,
                        'amount_eur': position_size,
                        'btc_amount': btc_amount,
                        'signal': signal_result.signal.value,
                        'emotion': summary['dominant_emotion'],
                        'phase': summary['market_phase'],
                        'contrarian': signal_result.contrarian_indicator
                    }
                    self.trades.append(trade)

                    logger.success(f"  ✅ EXECUTED: STRONG BUY {btc_amount:.4f} BTC "
                                 f"@ €{current_price:,.2f} (€{position_size:.2f})")
                else:
                    logger.warning(f"  ⚠️  INSUFFICIENT FUNDS: Need €{position_size:.2f}, "
                                 f"have €{self.balance_eur:.2f}")

            else:  # HOLD, STRONG_SELL
                logger.info(f"  📊 NO TRADE: Signal is {signal_result.signal.value}")

        else:
            logger.info(f"\n📊 No signal generated (insufficient data or confidence)")

        # Calculate portfolio value
        portfolio_value = self.balance_eur + (self.btc_holdings * current_price)
        logger.info(f"\n💼 Portfolio Value: €{portfolio_value:,.2f}")

        return portfolio_value

    def print_summary(self):
        """Print trading summary."""
        final_price = self.simulate_market_data(25)  # Approximate final price
        final_value = self.balance_eur + (self.btc_holdings * final_price)
        initial_value = 10000.0
        pnl = final_value - initial_value
        pnl_pct = (pnl / initial_value) * 100

        logger.info("\n\n")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 20 + "GoEmotions Paper Trading Summary" + " " * 24 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("")
        logger.info(f"Initial Balance:      €{initial_value:,.2f}")
        logger.info(f"Final Balance:        €{self.balance_eur:,.2f}")
        logger.info(f"BTC Holdings:         {self.btc_holdings:.4f} BTC")
        logger.info(f"Final Portfolio:      €{final_value:,.2f}")
        logger.info("")
        logger.info(f"Total P&L:            €{pnl:+,.2f} ({pnl_pct:+.2f}%)")
        logger.info("")
        logger.info(f"Total Trades:         {len(self.trades)}")
        logger.info("")

        if self.trades:
            logger.info("Trade History:")
            logger.info("-" * 80)

            # Group by action
            buys = [t for t in self.trades if t['action'] in ['BUY', 'STRONG_BUY']]
            sells = [t for t in self.trades if t['action'] == 'SELL']

            logger.info(f"BUY Signals: {len(buys)}")
            for trade in buys:
                contrarian = " [CONTRARIAN]" if trade['contrarian'] else ""
                logger.info(f"  Iteration {trade['iteration']}: "
                           f"{trade['action']} {trade['btc_amount']:.4f} BTC "
                           f"@ €{trade['price']:,.2f} "
                           f"({trade['phase']}){contrarian}")

            logger.info(f"\nSELL Signals: {len(sells)}")
            for trade in sells:
                contrarian = " [CONTRARIAN]" if trade['contrarian'] else ""
                logger.info(f"  Iteration {trade['iteration']}: "
                           f"{trade['action']} {trade['btc_amount']:.4f} BTC "
                           f"@ €{trade['price']:,.2f} "
                           f"({trade['phase']}){contrarian}")

            # Analyze performance
            logger.info("\n" + "=" * 80)
            logger.info("Strategy Analysis:")
            logger.info("=" * 80)

            contrarian_trades = [t for t in self.trades if t['contrarian']]

            logger.info(f"Contrarian Signals: {len(contrarian_trades)}")
            logger.info(f"Buy Phases: {len([t for t in self.trades if t['phase'] in ['accumulation', 'capitulation']])}")
            logger.info(f"Sell Phases: {len([t for t in self.trades if t['phase'] == 'distribution'])}")

            logger.info("\n" + "=" * 80)
            logger.info("Key Insights:")
            logger.info("=" * 80)

            if pnl > 0:
                logger.success(f"✅ Strategy Generated Profit: €{pnl:+,.2f}")
            else:
                logger.warning(f"⚠️  Strategy Generated Loss: €{pnl:+,.2f}")

            if contrarian_trades:
                logger.success(f"✅ Contrarian Signals Worked: {len(contrarian_trades)} "
                             f"trades against the crowd")
            else:
                logger.info(f"📊 No extreme emotions detected for contrarian trades")

            logger.info("")


async def main():
    """Main entry point."""
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 15 + "GoEmotions Strategy + Paper Trading Test" + " " * 19 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")

    # Initialize paper trader
    trader = PaperTrader(initial_balance_eur=10000)

    # Run 25 iterations covering full market cycle
    # 1-5: Markdown (concern)
    # 6-10: Capitulation (extreme fear)
    # 11-15: Accumulation (hope)
    # 16-20: Markup (excitement)
    # 21-25: Distribution (euphoria)

    logger.info("Starting paper trading simulation (25 iterations)")
    logger.info("Market Phases:")
    logger.info("  • Iterations 1-5: Markdown (concern building)")
    logger.info("  • Iterations 6-10: Capitulation (extreme fear)")
    logger.info("  • Iterations 11-15: Accumulation (hope returning)")
    logger.info("  • Iterations 16-20: Markup (excitement)")
    logger.info("  • Iterations 21-25: Distribution (euphoria)")
    logger.info("")

    for iteration in range(1, 26):
        await trader.run_iteration(iteration)

        # Small delay between iterations
        await asyncio.sleep(0.5)

    # Print summary
    trader.print_summary()

    logger.info("=" * 80)
    logger.info("✅ GoEmotions Paper Trading Test Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
