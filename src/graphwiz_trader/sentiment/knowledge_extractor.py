"""Knowledge extractor for fetching and analyzing market sentiment data.

This module provides functionality to extract sentiment data from various sources
including news, social media, and alternative data sources.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass, field
import re
from enum import Enum


class SentimentSource(Enum):
    """Sentiment data sources."""

    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    GLASSNODE = "glassnode"
    COINMETRICS = "coinmetrics"
    CUSTOM = "custom"


@dataclass
class SentimentData:
    """Raw sentiment data from a source."""

    source: SentimentSource
    symbol: str
    content: str
    timestamp: datetime
    url: Optional[str] = None
    author: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzedSentiment:
    """Analyzed sentiment with score and confidence."""

    symbol: str
    source: SentimentSource
    sentiment_score: float  # -1 (very negative) to 1 (very positive)
    confidence: float  # 0 to 1
    volume: int  # Number of mentions/posts
    keywords: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SentimentAnalyzer:
    """Analyzes text sentiment using various techniques."""

    def __init__(self):
        """Initialize sentiment analyzer."""
        # Crypto-specific sentiment lexicon
        self.bullish_terms = [
            "bullish",
            "moon",
            "pump",
            "rocket",
            "gem",
            "alpha",
            "breakout",
            "rally",
            "surge",
            "soar",
            "gain",
            "profit",
            "buy",
            "accumulate",
            "hodl",
            "diamond hands",
            "upgrade",
            "adoption",
            "partnership",
            "launch",
            "mainnet",
            "scaling",
        ]

        self.bearish_terms = [
            "bearish",
            "dump",
            "crash",
            "collapse",
            "bubble",
            "scam",
            "hack",
            "exploit",
            "ban",
            "regulation",
            "sec",
            "fud",
            "sell",
            "dumping",
            "correction",
            "dip",
            "bear market",
            "downgrade",
            "concern",
            "risk",
            "loss",
            "decline",
            "fall",
        ]

        self.crypto_symbols = {
            "BTC": ["bitcoin", "btc", "xbt"],
            "ETH": ["ethereum", "eth", "ether"],
            "SOL": ["solana", "sol"],
            "ADA": ["cardano", "ada"],
            "DOT": ["polkadot", "dot"],
            "MATIC": ["polygon", "matic"],
            "AVAX": ["avalanche", "avax"],
            "LINK": ["chainlink", "link"],
            "UNI": ["uniswap", "uni"],
            "AAVE": ["aave"],
        }

    def analyze_text(self, text: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze
            symbol: Optional crypto symbol to look for

        Returns:
            Dictionary with sentiment analysis results
        """
        text_lower = text.lower()

        # Count bullish and bearish terms
        bullish_count = sum(1 for term in self.bullish_terms if term in text_lower)
        bearish_count = sum(1 for term in self.bearish_terms if term in text_lower)

        # Calculate sentiment score (-1 to 1)
        total = bullish_count + bearish_count
        if total == 0:
            sentiment_score = 0.0
            confidence = 0.1
        else:
            sentiment_score = (bullish_count - bearish_count) / total
            # Higher confidence with more sentiment terms
            confidence = min(0.9, 0.3 + (total * 0.1))

        # Detect symbol mentions
        detected_symbols = set()
        if symbol:
            detected_symbols.add(symbol)

        for sym, variants in self.crypto_symbols.items():
            if any(variant in text_lower for variant in variants):
                detected_symbols.add(sym)

        # Extract keywords (simple approach: words with 5+ chars that appear)
        words = re.findall(r"\b[a-z]{5,}\b", text_lower)
        word_freq = {}
        for word in words:
            if word not in ["bitcoin", "ethereum", "crypto", "trading"]:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top 5 keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [k[0] for k in keywords]

        # Calculate volume impact (estimated based on text length and engagement)
        volume_estimate = len(text.split()) + (len(keywords) * 10)

        return {
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "detected_symbols": list(detected_symbols),
            "keywords": keywords,
            "volume_estimate": volume_estimate,
            "bullish_terms": bullish_count,
            "bearish_terms": bearish_count,
        }


class KnowledgeExtractor:
    """Extracts knowledge and sentiment from various sources."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize knowledge extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analyzer = SentimentAnalyzer()
        self.sources_enabled = config.get("sources", {})
        self.update_interval = config.get("update_interval_seconds", 300)
        self._running = False

        logger.info(
            f"Knowledge extractor initialized with sources: {list(self.sources_enabled.keys())}"
        )

    async def fetch_news_sentiment(
        self, symbols: List[str], hours_back: int = 24
    ) -> List[SentimentData]:
        """Fetch sentiment from news sources.

        Args:
            symbols: List of symbols to fetch news for
            hours_back: Number of hours to look back

        Returns:
            List of sentiment data points
        """
        # Simulated news data - in production, integrate with:
        # - CryptoCompare API
        # - CryptoPanic API
        # - NewsAPI
        # - Messari API

        simulated_news = [
            {
                "symbol": "BTC",
                "content": "Bitcoin breaks above $45000 as institutional adoption accelerates",
                "timestamp": datetime.now() - timedelta(hours=2),
            },
            {
                "symbol": "ETH",
                "content": "Ethereum scaling solutions show promise as gas fees decrease",
                "timestamp": datetime.now() - timedelta(hours=4),
            },
            {
                "symbol": "BTC",
                "content": "Concerns about regulatory crackdown on crypto exchanges",
                "timestamp": datetime.now() - timedelta(hours=6),
            },
        ]

        sentiment_data = []
        for news in simulated_news:
            if news["symbol"] in symbols:
                sentiment_data.append(
                    SentimentData(
                        source=SentimentSource.NEWS,
                        symbol=news["symbol"],
                        content=news["content"],
                        timestamp=news["timestamp"],
                        url=f"https://example.com/news/{news['symbol'].lower()}",
                    )
                )

        logger.info(f"Fetched {len(sentiment_data)} news items")
        return sentiment_data

    async def fetch_social_sentiment(
        self, symbols: List[str], hours_back: int = 24
    ) -> List[SentimentData]:
        """Fetch sentiment from social media sources.

        Args:
            symbols: List of symbols
            hours_back: Number of hours to look back

        Returns:
            List of sentiment data points
        """
        # Simulated social data - in production, integrate with:
        # - Twitter API v2
        # - Reddit API
        # - Telegram Bot API
        # - Discord API
        # - LunarCrush API

        simulated_posts = [
            {
                "symbol": "BTC",
                "content": "$BTC to the moon! 🚀 Accumulating before the halving",
                "source": SentimentSource.TWITTER,
                "timestamp": datetime.now() - timedelta(minutes=30),
            },
            {
                "symbol": "ETH",
                "content": "Bearish divergence on ETH 4h chart, be careful",
                "source": SentimentSource.TWITTER,
                "timestamp": datetime.now() - timedelta(minutes=15),
            },
            {
                "symbol": "BTC",
                "content": "Just bought more BTC at support, diamond hands!",
                "source": SentimentSource.REDDIT,
                "timestamp": datetime.now() - timedelta(hours=1),
            },
        ]

        sentiment_data = []
        for post in simulated_posts:
            if post["symbol"] in symbols:
                sentiment_data.append(
                    SentimentData(
                        source=post["source"],
                        symbol=post["symbol"],
                        content=post["content"],
                        timestamp=post["timestamp"],
                        author="crypto_trader",
                    )
                )

        logger.info(f"Fetched {len(sentiment_data)} social posts")
        return sentiment_data

    async def fetch_onchain_metrics(self, symbols: List[str]) -> List[SentimentData]:
        """Fetch on-chain metrics as sentiment indicators.

        Args:
            symbols: List of symbols

        Returns:
            List of sentiment data based on on-chain data
        """
        # Simulated on-chain data - in production, integrate with:
        # - Glassnode API
        # - Coin Metrics API
        # - CryptoQuant API
        # - IntoTheBlock

        # On-chain metrics to consider:
        # - Active addresses
        # - Exchange inflows/outflows
        # - HODLer behavior
        # - MVRV ratio
        # - NVT ratio

        sentiment_data = []

        # Example: High exchange inflows = bearish (potential selling)
        sentiment_data.append(
            SentimentData(
                source=SentimentSource.GLASSNODE,
                symbol="BTC",
                content="BTC exchange inflows spike to 30-day high - potential sell pressure",
                timestamp=datetime.now(),
                metrics={
                    "exchange_inflow": 1500,  # BTC
                    "active_addresses": 850000,
                    "mvrv_ratio": 2.1,
                },
            )
        )

        logger.info(f"Fetched {len(sentiment_data)} on-chain metrics")
        return sentiment_data

    def analyze_sentiment_batch(self, data_points: List[SentimentData]) -> List[AnalyzedSentiment]:
        """Analyze a batch of sentiment data points.

        Args:
            data_points: Raw sentiment data

        Returns:
            List of analyzed sentiment
        """
        analyzed = []

        for data in data_points:
            # Analyze the text content
            analysis = self.analyzer.analyze_text(data.content, data.symbol)

            # Calculate weighted sentiment score based on source reliability
            source_weights = {
                SentimentSource.NEWS: 1.0,
                SentimentSource.TWITTER: 0.7,
                SentimentSource.REDDIT: 0.6,
                SentimentSource.GLASSNODE: 0.9,
                SentimentSource.COINMETRICS: 0.9,
            }

            weight = source_weights.get(data.source, 0.5)
            weighted_score = analysis["sentiment_score"] * weight

            analyzed.append(
                AnalyzedSentiment(
                    symbol=data.symbol,
                    source=data.source,
                    sentiment_score=weighted_score,
                    confidence=analysis["confidence"],
                    volume=analysis["volume_estimate"],
                    keywords=analysis["keywords"],
                    timestamp=data.timestamp,
                    metadata={
                        "url": data.url,
                        "author": data.author,
                        "raw_score": analysis["sentiment_score"],
                        "source_weight": weight,
                        "bullish_terms": analysis["bullish_terms"],
                        "bearish_terms": analysis["bearish_terms"],
                    },
                )
            )

        return analyzed

    async def extract_and_analyze(
        self, symbols: List[str], hours_back: int = 24
    ) -> Dict[str, List[AnalyzedSentiment]]:
        """Extract and analyze sentiment from all enabled sources.

        Args:
            symbols: List of symbols to analyze
            hours_back: Hours to look back

        Returns:
            Dictionary mapping symbols to their analyzed sentiment
        """
        logger.info(f"Starting sentiment extraction for {symbols} ({hours_back}h back)")

        all_data = []

        # Fetch from all enabled sources
        if self.sources_enabled.get("news", False):
            news_data = await self.fetch_news_sentiment(symbols, hours_back)
            all_data.extend(news_data)

        if self.sources_enabled.get("social", False):
            social_data = await self.fetch_social_sentiment(symbols, hours_back)
            all_data.extend(social_data)

        if self.sources_enabled.get("onchain", False):
            onchain_data = await self.fetch_onchain_metrics(symbols)
            all_data.extend(onchain_data)

        # Analyze all data
        analyzed = self.analyze_sentiment_batch(all_data)

        # Group by symbol
        by_symbol = {sym: [] for sym in symbols}
        for sentiment in analyzed:
            if sentiment.symbol in by_symbol:
                by_symbol[sentiment.symbol].append(sentiment)

        # Log summary
        for sym, sentiments in by_symbol.items():
            if sentiments:
                avg_score = sum(s.sentiment_score for s in sentiments) / len(sentiments)
                logger.info(f"{sym}: {len(sentiments)} data points, avg sentiment: {avg_score:.3f}")

        return by_symbol

    def calculate_aggregate_sentiment(
        self, sentiments: List[AnalyzedSentiment]
    ) -> Dict[str, float]:
        """Calculate aggregate sentiment metrics.

        Args:
            sentiments: List of analyzed sentiments

        Returns:
            Dictionary with aggregate metrics
        """
        if not sentiments:
            return {
                "average_score": 0.0,
                "weighted_score": 0.0,
                "total_volume": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "confidence": 0.0,
            }

        # Calculate weighted average (by volume and confidence)
        total_weight = 0
        weighted_sum = 0

        for s in sentiments:
            weight = s.volume * s.confidence
            weighted_sum += s.sentiment_score * weight
            total_weight += weight

        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0

        # Simple average
        simple_avg = sum(s.sentiment_score for s in sentiments) / len(sentiments)

        # Count bullish/bearish/neutral
        bullish = sum(1 for s in sentiments if s.sentiment_score > 0.2)
        bearish = sum(1 for s in sentiments if s.sentiment_score < -0.2)
        neutral = len(sentiments) - bullish - bearish

        # Average confidence
        avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)

        return {
            "average_score": simple_avg,
            "weighted_score": weighted_avg,
            "total_volume": sum(s.volume for s in sentiments),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "confidence": avg_confidence,
            "data_points": len(sentiments),
        }
