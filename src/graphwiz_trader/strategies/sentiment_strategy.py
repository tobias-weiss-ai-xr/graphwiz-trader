"""Sentiment-based trading strategy.

This module implements trading strategies based on market sentiment analysis
from news, social media, and on-chain data.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum

from graphwiz_trader.sentiment.knowledge_extractor import (
    KnowledgeExtractor,
    SentimentSource,
    AnalyzedSentiment,
)


class SentimentSignal(Enum):
    """Sentiment-based trading signals."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class SentimentSignalResult:
    """Result of sentiment signal generation."""

    signal: SentimentSignal
    confidence: float
    sentiment_score: float
    sentiment_momentum: float  # Rate of change of sentiment
    volume_trend: str  # "increasing", "decreasing", "stable"
    key_drivers: List[str]  # Main factors driving the signal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "sentiment_score": self.sentiment_score,
            "sentiment_momentum": self.sentiment_momentum,
            "volume_trend": self.volume_trend,
            "key_drivers": self.key_drivers,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SentimentStrategyConfig:
    """Configuration for sentiment-based strategy."""

    # Signal thresholds
    strong_buy_threshold: float = 0.6  # Sentiment score for strong buy
    buy_threshold: float = 0.3  # Sentiment score for buy
    sell_threshold: float = -0.3  # Sentiment score for sell
    strong_sell_threshold: float = -0.6  # Sentiment score for strong sell

    # Minimum requirements
    min_confidence: float = 0.5  # Minimum confidence to trade
    min_data_points: int = 5  # Minimum sentiment data points required
    min_volume: int = 100  # Minimum mention volume

    # Momentum settings
    momentum_periods: int = 3  # Periods for momentum calculation
    momentum_weight: float = 0.3  # Weight of momentum in decision

    # Source weights
    news_weight: float = 1.0
    social_weight: float = 0.7
    onchain_weight: float = 0.9

    # Risk management
    position_multiplier: float = 1.5  # Multiply base position by this for strong signals
    max_sentiment_position_pct: float = 0.35  # Max 35% of balance for sentiment trades

    # Timing
    sentiment_decay_hours: int = 6  # Sentiment weight decays after this many hours


class SentimentTradingStrategy:
    """Trading strategy based on market sentiment analysis."""

    def __init__(
        self,
        config: SentimentStrategyConfig,
        knowledge_extractor: KnowledgeExtractor,
        knowledge_graph,
    ):
        """Initialize sentiment trading strategy.

        Args:
            config: Strategy configuration
            knowledge_extractor: Knowledge extractor instance
            knowledge_graph: Neo4j knowledge graph instance
        """
        self.config = config
        self.extractor = knowledge_extractor
        self.kg = knowledge_graph

        # Historical sentiment data for trend analysis
        self.sentiment_history: Dict[str, List[Tuple[datetime, AnalyzedSentiment]]] = {}

        logger.info("Sentiment-based trading strategy initialized")

    async def analyze_sentiment_for_symbol(
        self, symbol: str, hours_back: int = 24
    ) -> List[AnalyzedSentiment]:
        """Analyze sentiment for a specific symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC')
            hours_back: Hours of data to analyze

        Returns:
            List of analyzed sentiment data
        """
        # Fetch from knowledge extractor
        sentiment_data = await self.extractor.extract_and_analyze([symbol], hours_back)

        # Store in history
        sentiments = sentiment_data.get(symbol, [])

        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []

        self.sentiment_history[symbol].extend([(s.timestamp, s) for s in sentiments])

        # Clean old data (keep 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.sentiment_history[symbol] = [
            (ts, s) for ts, s in self.sentiment_history[symbol] if ts > cutoff
        ]

        return sentiments

    def calculate_sentiment_momentum(self, symbol: str, periods: int = 3) -> float:
        """Calculate sentiment momentum (rate of change).

        Args:
            symbol: Trading symbol
            periods: Number of periods to compare

        Returns:
            Momentum score (positive = improving sentiment)
        """
        if symbol not in self.sentiment_history:
            return 0.0

        history = self.sentiment_history[symbol]
        if len(history) < periods * 2:
            return 0.0

        # Sort by timestamp
        history = sorted(history, key=lambda x: x[0])

        # Calculate average sentiment for recent vs older periods
        recent = history[-periods:]
        older = history[-(periods * 2) : -periods]

        if not recent or not older:
            return 0.0

        recent_avg = sum(s.sentiment_score for _, s in recent) / len(recent)
        older_avg = sum(s.sentiment_score for _, s in older) / len(older)

        momentum = recent_avg - older_avg
        return momentum

    def calculate_volume_trend(self, symbol: str) -> Tuple[str, float]:
        """Calculate volume trend for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (trend_direction, change_percentage)
        """
        if symbol not in self.sentiment_history:
            return "stable", 0.0

        history = sorted(self.sentiment_history[symbol], key=lambda x: x[0])

        if len(history) < 4:
            return "stable", 0.0

        # Split into two halves
        mid = len(history) // 2
        older = history[:mid]
        recent = history[mid:]

        older_volume = sum(s.volume for _, s in older)
        recent_volume = sum(s.volume for _, s in recent)

        if older_volume == 0:
            return "increasing", 100.0

        change_pct = ((recent_volume - older_volume) / older_volume) * 100

        if change_pct > 20:
            return "increasing", change_pct
        elif change_pct < -20:
            return "decreasing", change_pct
        else:
            return "stable", change_pct

    def apply_time_decay(self, sentiments: List[AnalyzedSentiment]) -> List[AnalyzedSentiment]:
        """Apply time-based decay to sentiment weights.

        Args:
            sentiments: List of sentiments

        Returns:
            List with decay-adjusted confidence scores
        """
        now = datetime.now()
        decay_hours = self.config.sentiment_decay_hours

        result = []
        for sentiment in sentiments:
            hours_old = (now - sentiment.timestamp).total_seconds() / 3600

            # Exponential decay
            decay_factor = max(0.1, 1.0 - (hours_old / decay_hours))

            # Create adjusted sentiment
            adjusted = AnalyzedSentiment(
                symbol=sentiment.symbol,
                source=sentiment.source,
                sentiment_score=sentiment.sentiment_score,
                confidence=sentiment.confidence * decay_factor,
                volume=sentiment.volume,
                keywords=sentiment.keywords,
                timestamp=sentiment.timestamp,
                metadata={
                    **sentiment.metadata,
                    "decay_factor": decay_factor,
                    "hours_old": hours_old,
                },
            )
            result.append(adjusted)

        return result

    def generate_signal(
        self, symbol: str, current_price: float, balance: float
    ) -> Optional[SentimentSignalResult]:
        """Generate trading signal based on sentiment analysis.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            balance: Available balance

        Returns:
            Signal result or None if insufficient data
        """
        logger.info(f"Generating sentiment signal for {symbol}")

        # Get recent sentiment data
        sentiments = []
        if symbol in self.sentiment_history:
            cutoff = datetime.now() - timedelta(hours=24)
            sentiments = [s for ts, s in self.sentiment_history[symbol] if ts > cutoff]

        # Check minimum data requirements
        if len(sentiments) < self.config.min_data_points:
            logger.warning(
                f"Insufficient sentiment data for {symbol}: {len(sentiments)} < {self.config.min_data_points}"
            )
            return None

        # Apply time decay
        decayed_sentiments = self.apply_time_decay(sentiments)

        # Calculate aggregate sentiment
        total_volume = sum(s.volume for s in decayed_sentiments)
        if total_volume < self.config.min_volume:
            logger.warning(
                f"Insufficient volume for {symbol}: {total_volume} < {self.config.min_volume}"
            )
            return None

        # Calculate weighted sentiment score
        weighted_sum = 0
        total_weight = 0
        source_counts = {
            SentimentSource.NEWS: 0,
            SentimentSource.TWITTER: 0,
            SentimentSource.REDDIT: 0,
            SentimentSource.GLASSNODE: 0,
        }

        for s in decayed_sentiments:
            # Source weights
            weights = {
                SentimentSource.NEWS: self.config.news_weight,
                SentimentSource.TWITTER: self.config.social_weight,
                SentimentSource.REDDIT: self.config.social_weight,
                SentimentSource.GLASSNODE: self.config.onchain_weight,
            }

            weight = weights.get(s.source, 0.5) * s.confidence
            weighted_sum += s.sentiment_score * weight
            total_weight += weight

            if s.source in source_counts:
                source_counts[s.source] += 1

        sentiment_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Calculate momentum
        momentum = self.calculate_sentiment_momentum(symbol, self.config.momentum_periods)

        # Calculate volume trend
        volume_trend, volume_change = self.calculate_volume_trend(symbol)

        # Adjust sentiment score based on momentum
        adjusted_score = sentiment_score + (momentum * self.config.momentum_weight)

        # Clamp to valid range
        adjusted_score = max(-1.0, min(1.0, adjusted_score))

        # Generate signal
        if adjusted_score >= self.config.strong_buy_threshold:
            signal = SentimentSignal.STRONG_BUY
        elif adjusted_score >= self.config.buy_threshold:
            signal = SentimentSignal.BUY
        elif adjusted_score <= self.config.strong_sell_threshold:
            signal = SentimentSignal.STRONG_SELL
        elif adjusted_score <= self.config.sell_threshold:
            signal = SentimentSignal.SELL
        else:
            signal = SentimentSignal.HOLD

        # Calculate confidence (based on data quality and consistency)
        avg_confidence = sum(s.confidence for s in decayed_sentiments) / len(decayed_sentiments)

        # Boost confidence for strong momentum
        if abs(momentum) > 0.2:
            avg_confidence = min(0.95, avg_confidence + 0.1)

        # Identify key drivers
        all_keywords = []
        for s in decayed_sentiments:
            all_keywords.extend(s.keywords)

        # Get top keywords
        from collections import Counter

        keyword_counts = Counter(all_keywords)
        key_drivers = [k for k, v in keyword_counts.most_common(5)]

        # Create signal result
        result = SentimentSignalResult(
            signal=signal,
            confidence=avg_confidence,
            sentiment_score=sentiment_score,
            sentiment_momentum=momentum,
            volume_trend=volume_trend,
            key_drivers=key_drivers,
            timestamp=datetime.now(),
            metadata={
                "adjusted_score": adjusted_score,
                "volume_change": volume_change,
                "source_counts": {s.value: c for s, c in source_counts.items()},
                "data_points": len(decayed_sentiments),
                "total_volume": total_volume,
                "current_price": current_price,
                "symbol": symbol,
            },
        )

        logger.info(
            f"{symbol} Signal: {signal.value} | "
            f"Score: {sentiment_score:.3f} | "
            f"Confidence: {avg_confidence:.2f} | "
            f"Momentum: {momentum:+.3f}"
        )

        return result

    def calculate_position_size(
        self,
        signal: SentimentSignalResult,
        current_price: float,
        balance: float,
        base_position_eur: float,
    ) -> float:
        """Calculate position size based on sentiment signal strength.

        Args:
            signal: Sentiment signal result
            current_price: Current price
            balance: Available balance
            base_position_eur: Base position size in EUR

        Returns:
            Position size in EUR
        """
        # Start with base position
        position_eur = base_position_eur

        # Adjust based on signal strength
        if signal.signal == SentimentSignal.STRONG_BUY:
            position_eur *= self.config.position_multiplier
        elif signal.signal == SentimentSignal.STRONG_SELL:
            position_eur *= self.config.position_multiplier
        elif signal.signal == SentimentSignal.HOLD:
            position_eur = 0

        # Adjust based on confidence
        if signal.signal != SentimentSignal.HOLD:
            confidence_multiplier = 0.5 + (signal.confidence * 0.5)
            position_eur *= confidence_multiplier

        # Adjust based on momentum (positive momentum = increase position)
        if signal.sentiment_momentum > 0.1:
            position_eur *= 1.2
        elif signal.sentiment_momentum < -0.1:
            position_eur *= 0.8

        # Respect maximum
        max_position = balance * self.config.max_sentiment_position_pct
        position_eur = min(position_eur, max_position)

        logger.info(
            f"Position calculation: €{position_eur:.2f} "
            f"(base: €{base_position_eur:.2f}, max: €{max_position:.2f})"
        )

        return position_eur

    async def store_sentiment_in_graph(
        self, symbol: str, sentiments: List[AnalyzedSentiment]
    ) -> None:
        """Store sentiment data in Neo4j knowledge graph.

        Args:
            symbol: Trading symbol
            sentiments: List of analyzed sentiments
        """
        if not self.kg:
            logger.warning("No knowledge graph instance available")
            return

        try:
            # Store each sentiment data point
            for sentiment in sentiments:
                await self.kg.create_sentiment_node(
                    symbol=sentiment.symbol,
                    timestamp=sentiment.timestamp,
                    source=sentiment.source.value,
                    sentiment_score=sentiment.sentiment_score,
                    confidence=sentiment.confidence,
                    volume=sentiment.volume,
                    keywords=sentiment.keywords,
                    metadata=sentiment.metadata,
                )

            logger.info(f"Stored {len(sentiments)} sentiment points in knowledge graph")

        except Exception as e:
            logger.error(f"Failed to store sentiment in graph: {e}")

    async def update_and_analyze(self, symbols: List[str]) -> Dict[str, SentimentSignalResult]:
        """Update sentiment data and generate signals for multiple symbols.

        Args:
            symbols: List of symbols to analyze

        Returns:
            Dictionary mapping symbols to their signals
        """
        logger.info(f"Updating and analyzing sentiment for {symbols}")

        results = {}

        for symbol in symbols:
            # Fetch and analyze sentiment
            await self.analyze_sentiment_for_symbol(symbol)

            # Store in knowledge graph
            if symbol in self.sentiment_history:
                recent = [
                    s
                    for ts, s in self.sentiment_history[symbol]
                    if ts > datetime.now() - timedelta(hours=24)
                ]
                await self.store_sentiment_in_graph(symbol, recent)

            # Note: Signal generation requires current price, done separately
            results[symbol] = None  # Placeholder, actual signal in generate_signal()

        return results


class SentimentStrategyFactory:
    """Factory for creating sentiment-based strategies."""

    @staticmethod
    def create_from_config(
        config: Dict[str, Any], knowledge_extractor: KnowledgeExtractor, knowledge_graph
    ) -> SentimentTradingStrategy:
        """Create sentiment strategy from configuration.

        Args:
            config: Configuration dictionary
            knowledge_extractor: Knowledge extractor instance
            knowledge_graph: Neo4j knowledge graph instance

        Returns:
            Configured sentiment trading strategy
        """
        # Extract strategy config
        strategy_config = config.get("sentiment_strategy", {})

        sentiment_config = SentimentStrategyConfig(
            strong_buy_threshold=strategy_config.get("strong_buy_threshold", 0.6),
            buy_threshold=strategy_config.get("buy_threshold", 0.3),
            sell_threshold=strategy_config.get("sell_threshold", -0.3),
            strong_sell_threshold=strategy_config.get("strong_sell_threshold", -0.6),
            min_confidence=strategy_config.get("min_confidence", 0.5),
            min_data_points=strategy_config.get("min_data_points", 5),
            min_volume=strategy_config.get("min_volume", 100),
            momentum_periods=strategy_config.get("momentum_periods", 3),
            momentum_weight=strategy_config.get("momentum_weight", 0.3),
            news_weight=strategy_config.get("news_weight", 1.0),
            social_weight=strategy_config.get("social_weight", 0.7),
            onchain_weight=strategy_config.get("onchain_weight", 0.9),
            position_multiplier=strategy_config.get("position_multiplier", 1.5),
            max_sentiment_position_pct=strategy_config.get("max_sentiment_position_pct", 0.35),
            sentiment_decay_hours=strategy_config.get("sentiment_decay_hours", 6),
        )

        return SentimentTradingStrategy(
            config=sentiment_config,
            knowledge_extractor=knowledge_extractor,
            knowledge_graph=knowledge_graph,
        )
