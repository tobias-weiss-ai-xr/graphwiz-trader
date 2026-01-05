"""Sentiment analysis module for trading.

This module provides sentiment analysis capabilities for cryptocurrency trading,
including knowledge extraction from various sources and sentiment-based trading strategies.
"""

from graphwiz_trader.sentiment.knowledge_extractor import (
    KnowledgeExtractor,
    SentimentAnalyzer,
    SentimentSource,
    SentimentData,
    AnalyzedSentiment,
)

from graphwiz_trader.strategies.sentiment_strategy import (
    SentimentTradingStrategy,
    SentimentStrategyFactory,
    SentimentSignal,
    SentimentSignalResult,
    SentimentStrategyConfig,
)

# GoEmotions-based emotion analysis
from graphwiz_trader.sentiment.goemotions_analyzer import (
    GoEmotionsAnalyzer,
    GoEmotion,
    EmotionGroup,
    EmotionProfile,
    EmotionScore,
)

from graphwiz_trader.strategies.emotion_strategy import (
    EmotionBasedStrategy,
    EmotionStrategyFactory,
    MarketPhase,
    EmotionSignal as GoEmotionSignal,
    EmotionSignalResult as GoEmotionSignalResult,
    EmotionStrategyConfig,
)

__all__ = [
    # Knowledge Extraction
    "KnowledgeExtractor",
    "SentimentAnalyzer",
    "SentimentSource",
    "SentimentData",
    "AnalyzedSentiment",
    # Trading Strategy (Basic)
    "SentimentTradingStrategy",
    "SentimentStrategyFactory",
    "SentimentSignal",
    "SentimentSignalResult",
    "SentimentStrategyConfig",
    # GoEmotions Analysis
    "GoEmotionsAnalyzer",
    "GoEmotion",
    "EmotionGroup",
    "EmotionProfile",
    "EmotionScore",
    # Emotion-Based Strategy (Advanced)
    "EmotionBasedStrategy",
    "EmotionStrategyFactory",
    "MarketPhase",
    "GoEmotionSignal",
    "GoEmotionSignalResult",
    "EmotionStrategyConfig",
]
