#!/usr/bin/env python3
"""Demo script for advanced GoEmotions-based trading strategy.

Demonstrates sophisticated emotion-based trading using 27 fine-grained emotions
from the GoEmotions dataset and behavioral finance principles.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from graphwiz_trader.sentiment.goemotions_analyzer import (
    GoEmotionsAnalyzer,
    GoEmotion,
    EmotionGroup
)
from graphwiz_trader.strategies.emotion_strategy import (
    EmotionBasedStrategy,
    EmotionStrategyFactory,
    MarketPhase,
    EmotionSignal,
    EmotionStrategyConfig
)


def demo_goemotions_analyzer():
    """Demonstrate GoEmotions emotion detection."""
    print("=" * 80)
    print("GoEmotions Analyzer Demo")
    print("=" * 80)

    analyzer = GoEmotionsAnalyzer()

    # Test texts representing different market emotional states
    test_cases = [
        {
            'name': 'Extreme Euphoria (Market Top)',
            'text': "🚀🚀🚀 TO THE MOON! Bitcoin is unstoppable! Going to infinity! "
                   "Lambo shopping spree! We're all gonna be rich! HODL diamond hands! "
                   "This is the future! Easy money printing! 💎🙌🔥"
        },
        {
            'name': 'Capitulation (Market Bottom)',
            'text': "I've lost everything. My life savings are gone. This is pure hell. "
                   "Scam, Ponzi, bubble has burst. Bitcoin is going to zero. "
                   "I'm ruined, devastated, heartbroken. Why did I believe? 😭💀📉"
        },
        {
            'name': 'Accumulation (Hope)',
            'text': "Building my position slowly. Bitcoin has great technology and potential. "
                   "Long-term believer here. Patience is key. Accumulating while cheap. "
                   "Mainnet upgrades coming. The future looks bright. 🌱💪"
        },
        {
            'name': 'FOMO/Greed',
            "text": "FOMO is real! I need to buy NOW before I miss out! Everyone's making "
                   "easy money! I'm all in, borrowing on margin, can't miss this! "
                   "10x my money! Where's the lambo?! 🤑💸🎰"
        },
        {
            'name': 'Panic/Confusion',
            'text': "WTF is happening?! Why is it crashing?! Someone explain please! "
                   "I'm panicking here! What do I do?! Help! Is this the end?! "
                   "Emergency!! Market chaos!! 😵🚨❓"
        },
        {
            'name': 'Neutral Analysis',
            'text': "Bitcoin is currently trading at €45,000 with a volume of 2.5K BTC. "
                   "RSI indicates 55. Support at €44,000, resistance at €46,500. "
                   "Just sharing the technical analysis for the community."
        }
    ]

    print("\n" + "=" * 80)
    print("Analyzing Market Emotions Using GoEmotions (27 Categories)")
    print("=" * 80)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"Test Case {i}: {test['name']}")
        print(f"{'─' * 80}")
        print(f"\nText: \"{test['text'][:100]}...\"")

        # Analyze emotions
        profile = analyzer.detect_emotions(test['text'])

        print(f"\n🎭 Emotion Analysis:")
        print(f"   Dominant Emotion: {profile.dominant_emotion.value}")
        print(f"   Emotional Intensity: {profile.emotional_intensity:.2f}/1.0")
        print(f"   Emotional Volatility: {profile.emotional_volatility:.3f}")
        print(f"   Trading Bias: {profile.trading_bias.upper()}")

        print(f"\n📊 Top 5 Emotions:")
        sorted_emotions = sorted(
            profile.emotions.items(),
            key=lambda x: x[1].score,
            reverse=True
        )[:5]

        for emotion, score in sorted_emotions:
            print(f"   {emotion.value:15s} → {score.score:.3f} (conf: {score.confidence:.2f})")

        print(f"\n📈 Emotion Group Distribution:")
        for group, score in sorted(profile.group_distribution.items(),
                                   key=lambda x: x[1],
                                   reverse=True):
            print(f"   {group.value:15s} → {score:.3f}")


def demo_emotion_strategy():
    """Demonstrate emotion-based trading strategy."""
    print("\n\n" + "=" * 80)
    print("Emotion-Based Trading Strategy Demo")
    print("=" * 80)

    # Initialize strategy
    from graphwiz_trader.strategies.emotion_strategy import EmotionStrategyConfig

    config = EmotionStrategyConfig(
        extreme_euphoria_threshold=0.7,
        extreme_fear_threshold=0.7,
        use_contrarian_signals=True,
        min_data_points=3  # Lower for demo
    )

    strategy = EmotionBasedStrategy(config=config)

    # Simulate market emotion evolution
    market_scenarios = [
        {
            'phase': 'Accumulation (Hope)',
            'texts': [
                "Slowly accumulating Bitcoin. Long-term potential here. 🌱",
                "Great technology, building position. Patience pays. 💪",
                "Undervalued at these levels. Smart money buying. 📊"
            ],
            'expected_signal': EmotionSignal.BUY
        },
        {
            'phase': 'Markup (Excitement)',
            'texts': [
                "Bitcoin breaking out! Excitement building! 🚀",
                "Joy to see green! Technicals looking great! 📈",
                "Community enthusiasm is high! Good momentum! ⚡"
            ],
            'expected_signal': EmotionSignal.HOLD
        },
        {
            'phase': 'Distribution (Euphoria)',
            'texts': [
                "TO THE MOON! 🌙 Going to infinity! Easy money! 💎",
                "Lambo time! We're all rich! Never selling! 🤑",
                "Parabolic gains! This never goes down! 🚀🔥"
            ],
            'expected_signal': EmotionSignal.SELL
        },
        {
            'phase': 'Capitulation (Extreme Fear)',
            'texts': [
                "I've lost everything. Devastated. My savings gone. 😭",
                "Bitcoin is dead. Going to zero. Ruined my life. 💀",
                "Complete despair. Panic selling. This is hell. 📉"
            ],
            'expected_signal': EmotionSignal.BUY  # Contrarian!
        }
    ]

    print("\nSimulating Market Emotional Cycle...\n")

    for scenario in market_scenarios:
        print(f"\n{'=' * 80}")
        print(f"Market Phase: {scenario['phase']}")
        print(f"{'=' * 80}")

        # Analyze texts
        import asyncio
        for text in scenario['texts']:
            profile = strategy.analyzer.detect_emotions(text)
            print(f"\n  Text: \"{text[:60]}...\"")
            print(f"  → {profile.dominant_emotion.value} "
                  f"(intensity: {profile.emotional_intensity:.2f})")

        # Generate signal (simulate having enough data)
        # For demo, manually create profile
        if 'Accumulation' in scenario['phase']:
            profile = strategy.analyzer.detect_emotions(scenario['texts'][0])
        elif 'Markup' in scenario['phase']:
            profile = strategy.analyzer.detect_emotions(scenario['texts'][0])
        elif 'Distribution' in scenario['phase']:
            profile = strategy.analyzer.detect_emotions(scenario['texts'][0])
        else:  # Capitulation
            profile = strategy.analyzer.detect_emotions(scenario['texts'][0])

        # Identify market phase
        market_phase = strategy.identify_market_phase(profile)

        print(f"\n  🎯 Market Phase: {market_phase.value}")
        print(f"  📊 Trading Bias: {profile.trading_bias}")

        # Determine expected signal
        if market_phase in [MarketPhase.CAPITULATION, MarketPhase.ACCUMULATION]:
            action = "BUY (Contrarian)"
        elif market_phase == MarketPhase.DISTRIBUTION:
            action = "SELL"
        else:
            action = "HOLD"

        print(f"  ⚡ Action: {action}")

        # Check if contrarian
        contrarian = strategy._is_contrarian_indicator(profile, market_phase)
        if contrarian:
            print(f"  🔄 CONTRARIAN INDICATOR: Extreme emotion suggests reversal!")


def demo_trading_signals():
    """Demonstrate complete trading signal generation."""
    print("\n\n" + "=" * 80)
    print("Complete Trading Signal Generation")
    print("=" * 80)

    strategy = EmotionBasedStrategy(
        config=EmotionStrategyConfig(min_data_points=3)
    )

    # Simulate emotion history for BTC
    symbol = 'BTC'
    current_price = 45000.0
    balance = 1000.0

    # Create simulated emotion timeline
    emotions_timeline = [
        {
            'time': datetime.now() - timedelta(hours=6),
            'texts': [
                "Bitcoin crashing! I'm scared! What's happening?!",
                "Panic selling! This is a disaster!"
            ]
        },
        {
            'time': datetime.now() - timedelta(hours=4),
            'texts': [
                "Still falling... I'm nervous about my position.",
                "Fear is real. Thinking about selling."
            ]
        },
        {
            'time': datetime.now() - timedelta(hours=2),
            'texts': [
                "Market looks weak. Sad to see my portfolio down.",
                "Disappointed by the lack of recovery."
            ]
        },
        {
            'time': datetime.now() - timedelta(hours=1),
            'texts': [
                "Finally stabilizing. Hope returning slowly.",
                "Looking to accumulate at these levels."
            ]
        },
        {
            'time': datetime.now(),
            'texts': [
                "Optimistic about recovery! Long-term bullish!",
                "Great buying opportunity for patient investors!"
            ]
        }
    ]

    print(f"\n📊 Simulating emotion evolution for {symbol}...\n")

    # Feed emotion data
    for data in emotions_timeline:
        print(f"  [{data['time'].strftime('%H:%M')}] ", end='')
        profile = strategy.analyzer.detect_emotions(data['texts'][0])
        print(f"{profile.dominant_emotion.value} (intensity: {profile.emotional_intensity:.2f})")

        # Store in strategy history
        import asyncio
        asyncio.run(strategy.analyze_emotions_for_symbol(
            symbol, data['texts'], [data['time']]
        ))

    # Generate signal
    print(f"\n🎯 Generating trading signal for {symbol}...")

    # Get summary
    summary = strategy.get_market_emotion_summary(symbol)

    print(f"\n  Market Emotion Summary:")
    print(f"    Data Points: {summary['data_points']}")
    print(f"    Dominant Emotion: {summary['dominant_emotion']}")
    print(f"    Intensity: {summary['emotional_intensity']:.2f}")
    print(f"    Trading Bias: {summary['trading_bias']}")
    print(f"    Market Phase: {summary['market_phase']}")

    # Generate actual signal (mock for demo)
    print(f"\n  📈 Trading Signal:")
    print(f"    Current Price: €{current_price:,.2f}")
    print(f"    Balance: €{balance:,.2f}")

    # Determine signal based on market phase
    phase = MarketPhase(summary['market_phase'])

    if phase == MarketPhase.CAPITULATION:
        signal = "STRONG_BUY"
        position = 300
    elif phase == MarketPhase.ACCUMULATION:
        signal = "BUY"
        position = 250
    elif phase == MarketPhase.DISTRIBUTION:
        signal = "SELL"
        position = 150
    else:
        signal = "HOLD"
        position = 0

    print(f"    Signal: {signal}")
    print(f"    Recommended Position: €{position:.2f}")


def main():
    """Run all demos."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "GoEmotions-Based Trading Strategy Demo" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")

    # Demo 1: Emotion detection
    demo_goemotions_analyzer()

    # Demo 2: Trading strategy
    demo_emotion_strategy()

    # Demo 3: Complete signal generation
    demo_trading_signals()

    print("\n\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
The GoEmotions-based strategy provides:

1. Fine-Grained Emotion Detection
   • 27 emotion categories from Google Research
   • Crypto-specific lexicons for each emotion
   • Emoji pattern recognition

2. Market Psychology Analysis
   • Identifies market phase (Accumulation → Distribution → Markdown)
   • Tracks emotional intensity and velocity
   • Detects contrarian opportunities

3. Advanced Trading Signals
   • Contrarian indicators (buy when extreme fear)
   • Emotion velocity (rate of change)
   • Multi-factor scoring

4. Behavioral Finance Principles
   • Fear & Greed Index concepts
   • Dow Theory market phases
   • Mean reversion signals

Sources:
- GoEmotions Dataset: https://aclanthology.org/2020.acl-main.372/
- Fear & Greed Index: https://www.investopedia.com/terms/f/fear-and-greed-index.asp
- MIT Research: https://web.mit.edu/Alo/www/Papers/AERPub.pdf

✅ Demo Complete!
""")

    print("=" * 80)


if __name__ == "__main__":
    main()
