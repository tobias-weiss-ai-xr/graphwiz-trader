#!/usr/bin/env python3
"""Demo script for sentiment-based trading strategy.

This script demonstrates how to use the knowledge extractor and sentiment
trading strategy to analyze market sentiment and generate trading signals.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from graphwiz_trader.sentiment import (
    KnowledgeExtractor,
    SentimentTradingStrategy,
    SentimentStrategyFactory,
    SentimentSignal
)
from graphwiz_trader.graph.neo4j_graph import KnowledgeGraph


async def demo_sentiment_analysis():
    """Demonstrate sentiment analysis functionality."""

    print("=" * 80)
    print("Sentiment-Based Trading Strategy Demo")
    print("=" * 80)

    # 1. Load configuration
    print("\n[1] Loading configuration...")
    import yaml
    with open('config/sentiment.yaml', 'r') as f:
        config = yaml.safe_load(f)

    extractor_config = config['knowledge_extractor']
    strategy_config = config['sentiment_strategy']
    symbols = config['symbols'][:3]  # Use first 3 symbols for demo

    print(f"   ✓ Tracking symbols: {symbols}")
    print(f"   ✓ Data sources: {[s for s, enabled in extractor_config['sources'].items() if enabled]}")

    # 2. Initialize Knowledge Graph (optional - set to None to skip)
    print("\n[2] Initializing Knowledge Graph...")
    kg = None

    # Uncomment to use Neo4j:
    # kg_config = {
    #     'uri': 'bolt://localhost:7687',
    #     'username': 'neo4j',
    #     'password': 'your_password'
    # }
    # kg = KnowledgeGraph(kg_config)
    # kg.connect()
    # print("   ✓ Connected to Neo4j")

    print("   ⚠️  Neo4j connection skipped (set up credentials to enable)")

    # 3. Initialize Knowledge Extractor
    print("\n[3] Initializing Knowledge Extractor...")
    extractor = KnowledgeExtractor(extractor_config)
    print("   ✓ Knowledge extractor ready")

    # 4. Initialize Sentiment Strategy
    print("\n[4] Initializing Sentiment Trading Strategy...")
    strategy = SentimentTradingStrategy(
        config=SentimentStrategyFactory.create_from_config(
            config,
            extractor,
            kg
        ).config,
        knowledge_extractor=extractor,
        knowledge_graph=kg
    )
    print("   ✓ Strategy initialized")

    # 5. Extract and analyze sentiment
    print("\n[5] Extracting and analyzing sentiment data...")

    for symbol in symbols:
        print(f"\n   Analyzing {symbol}...")

        # Fetch sentiment data
        await strategy.analyze_sentiment_for_symbol(symbol, hours_back=24)

        # Get aggregate metrics
        if symbol in strategy.sentiment_history:
            sentiments = [s for ts, s in strategy.sentiment_history[symbol]]
            aggregate = extractor.calculate_aggregate_sentiment(sentiments)

            print(f"   📊 {symbol} Sentiment Summary:")
            print(f"      - Average Score: {aggregate['average_score']:.3f}")
            print(f"      - Weighted Score: {aggregate['weighted_score']:.3f}")
            print(f"      - Total Volume: {aggregate['total_volume']}")
            print(f"      - Data Points: {aggregate['data_points']}")
            print(f"      - Bullish: {aggregate['bullish_count']} | "
                  f"Bearish: {aggregate['bearish_count']} | "
                  f"Neutral: {aggregate['neutral_count']}")

    # 6. Generate trading signals
    print("\n[6] Generating trading signals...")

    current_prices = {
        'BTC': 45000.0,
        'ETH': 2500.0,
        'SOL': 100.0
    }
    balance = 1000.0

    signals = {}
    for symbol in symbols:
        price = current_prices.get(symbol, 0)
        signal_result = strategy.generate_signal(symbol, price, balance)

        if signal_result:
            signals[symbol] = signal_result

            # Calculate position size
            base_position = 250  # EUR
            position_size = strategy.calculate_position_size(
                signal_result,
                price,
                balance,
                base_position
            )

            print(f"\n   🎯 {symbol} Trading Signal:")
            print(f"      - Signal: {signal_result.signal.value.upper()}")
            print(f"      - Confidence: {signal_result.confidence:.2%}")
            print(f"      - Sentiment Score: {signal_result.sentiment_score:.3f}")
            print(f"      - Momentum: {signal_result.sentiment_momentum:+.3f}")
            print(f"      - Volume Trend: {signal_result.volume_trend}")
            print(f"      - Key Drivers: {', '.join(signal_result.key_drivers[:3])}")
            print(f"      - Recommended Position: €{position_size:.2f}")

            # Show action
            if signal_result.signal in [SentimentSignal.BUY, SentimentSignal.STRONG_BUY]:
                action = "BUY"
                amount = position_size / price
                print(f"      → Action: {action} {amount:.6f} {symbol} @ €{price:.2f}")
            elif signal_result.signal in [SentimentSignal.SELL, SentimentSignal.STRONG_SELL]:
                action = "SELL"
                amount = position_size / price
                print(f"      → Action: {action} {amount:.6f} {symbol} @ €{price:.2f}")
            else:
                print(f"      → Action: HOLD")
        else:
            print(f"\n   ⚠️  {symbol}: No signal generated (insufficient data)")

    # 7. Store sentiment in knowledge graph (if enabled)
    if kg:
        print("\n[7] Storing sentiment data in Knowledge Graph...")
        for symbol in symbols:
            if symbol in strategy.sentiment_history:
                sentiments = [
                    s for ts, s in strategy.sentiment_history[symbol]
                    if ts > datetime.now() - timedelta(hours=24)
                ]
                await strategy.store_sentiment_in_graph(symbol, sentiments)
                print(f"   ✓ Stored {len(sentiments)} sentiment points for {symbol}")

    # 8. Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    if signals:
        buy_signals = sum(1 for s in signals.values()
                         if s.signal in [SentimentSignal.BUY, SentimentSignal.STRONG_BUY])
        sell_signals = sum(1 for s in signals.values()
                          if s.signal in [SentimentSignal.SELL, SentimentSignal.STRONG_SELL])
        hold_signals = sum(1 for s in signals.values()
                          if s.signal == SentimentSignal.HOLD)

        print(f"Total Signals Generated: {len(signals)}")
        print(f"  - BUY signals: {buy_signals}")
        print(f"  - SELL signals: {sell_signals}")
        print(f"  - HOLD signals: {hold_signals}")

        # Recommendation
        if buy_signals > sell_signals:
            print("\n📈 Market sentiment: BULLISH")
        elif sell_signals > buy_signals:
            print("\n📉 Market sentiment: BEARISH")
        else:
            print("\n➡️  Market sentiment: NEUTRAL")

    print("\n✅ Demo completed successfully!")
    print("=" * 80)

    # Cleanup
    if kg:
        kg.disconnect()


async def demo_individual_components():
    """Demonstrate individual components separately."""

    print("\n" + "=" * 80)
    print("Individual Component Demos")
    print("=" * 80)

    # 1. Sentiment Analyzer
    print("\n[1] Testing Sentiment Analyzer...")

    from graphwiz_trader.sentiment.knowledge_extractor import SentimentAnalyzer

    analyzer = SentimentAnalyzer()

    test_texts = [
        "Bitcoin is going to the moon! 🚀 Institutional adoption accelerating!",
        "Concerns about crypto regulations, market looks bearish",
        "Ethereum gas fees decreasing, scaling solutions working well"
    ]

    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\n   Text: \"{text[:60]}...\"")
        print(f"   Score: {result['sentiment_score']:.3f} | "
              f"Confidence: {result['confidence']:.2f}")
        print(f"   Bullish terms: {result['bullish_terms']} | "
              f"Bearish terms: {result['bearish_terms']}")
        print(f"   Keywords: {', '.join(result['keywords'][:3])}")

    # 2. Knowledge Extractor
    print("\n\n[2] Testing Knowledge Extractor...")

    extractor = KnowledgeExtractor({'sources': {'news': True, 'social': False}})

    sentiment_data = await extractor.extract_and_analyze(['BTC'], hours_back=24)

    for symbol, sentiments in sentiment_data.items():
        print(f"\n   {symbol}: {len(sentiments)} sentiment data points")
        for s in sentiments[:3]:  # Show first 3
            print(f"      - {s.source.value}: {s.sentiment_score:.3f} "
                  f"(confidence: {s.confidence:.2f})")

    print("\n✅ Component demos completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentiment-based trading strategy demo")
    parser.add_argument("--components", action="store_true",
                       help="Run individual component demos")
    parser.add_argument("--full", action="store_true",
                       help="Run full strategy demo")

    args = parser.parse_args()

    if args.components:
        asyncio.run(demo_individual_components())
    elif args.full:
        asyncio.run(demo_sentiment_analysis())
    else:
        print("Running full demo by default...")
        asyncio.run(demo_sentiment_analysis())
