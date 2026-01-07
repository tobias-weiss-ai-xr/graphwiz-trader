"""GoEmotions-based fine-grained emotion analysis for trading.

This module implements sophisticated emotion detection using the GoEmotions
taxonomy (27 emotion categories) and maps emotions to trading signals.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np


# GoEmotions 27 emotion categories
class GoEmotion(Enum):
    """GoEmotions 27 emotion categories from Google Research."""

    ADMIRATION = "admiration"
    AMUSEMENT = "amusement"
    ANGER = "anger"
    ANNOYANCE = "annoyance"
    APPROVAL = "approval"
    CARING = "caring"
    CONFUSION = "confusion"
    CURIOSITY = "curiosity"
    DESIRE = "desire"
    DISAPPOINTMENT = "disappointment"
    DISAPPROVAL = "disapproval"
    DISGUST = "disgust"
    EMBARRASSMENT = "embarrassment"
    EXCITEMENT = "excitement"
    FEAR = "fear"
    GRATITUDE = "gratitude"
    GRIEF = "grief"
    JOY = "joy"
    LOVE = "love"
    NERVOUSNESS = "nervousness"
    OPTIMISM = "optimism"
    PRIDE = "pride"
    REALIZATION = "realization"
    RELIEF = "relief"
    REMORSE = "remorse"
    SADNESS = "sadness"
    SURPRISE = "surprise"


# Emotion groups for trading
class EmotionGroup(Enum):
    """Groups emotions by trading implications."""

    EUPHORIA = "euphoria"  # joy, excitement, love, pride, amusement
    FEAR = "fear"  # fear, nervousness, grief, sadness
    GREED = "greed"  # desire, excitement, optimism
    PANIC = "panic"  # fear, confusion, nervousness
    DISGUST = "disgust"  # disgust, disapproval, disappointment, anger
    HOPE = "hope"  # optimism, desire, curiosity, gratitude
    NEUTRAL = "neutral"  # realization, surprise, confusion, curiosity
    AGGRESSION = "aggression"  # anger, annoyance, disgust


@dataclass
class EmotionScore:
    """Individual emotion detection result."""

    emotion: GoEmotion
    score: float  # 0 to 1
    confidence: float  # 0 to 1
    keywords: List[str] = field(default_factory=list)
    group: Optional[EmotionGroup] = None


@dataclass
class EmotionProfile:
    """Complete emotional profile of a text/market state."""

    emotions: Dict[GoEmotion, EmotionScore]
    dominant_emotion: GoEmotion
    emotional_intensity: float  # 0 to 1
    emotional_volatility: float  # Standard deviation of emotions
    group_distribution: Dict[EmotionGroup, float]
    trading_bias: str  # bullish, bearish, neutral
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "dominant_emotion": self.dominant_emotion.value,
            "intensity": self.emotional_intensity,
            "volatility": self.emotional_volatility,
            "trading_bias": self.trading_bias,
            "groups": {g.value: s for g, s in self.group_distribution.items()},
            "top_emotions": [
                {"emotion": e.value, "score": s.score}
                for e, s in sorted(self.emotions.items(), key=lambda x: x[1].score, reverse=True)[
                    :5
                ]
            ],
        }


class GoEmotionsAnalyzer:
    """Analyzes text using GoEmotions taxonomy for trading insights."""

    def __init__(self):
        """Initialize GoEmotions analyzer with crypto-specific lexicons."""
        self._build_emotion_lexicons()

    def _build_emotion_lexicons(self) -> None:
        """Build crypto-specific emotion lexicons.

        Maps emotion categories to crypto trading terminology.
        """

        # EUPHORIA: Manic phase, potential top
        self.euphoria_keywords = {
            # Direct emotion words
            "moon",
            "lambo",
            "wen",
            "diamond",
            "hands",
            "stonks",
            "rocket",
            "to",
            "mars",
            "infinity",
            "parabolic",
            "explosion",
            "gang",
            # Extreme positive
            "unstoppable",
            "guaranteed",
            "easy",
            "money",
            "printing",
            "free",
            "wealth",
            "rich",
            "millionaire",
            "early",
            "adopt",
            # Hype indicators
            "biggest",
            "greatest",
            "best",
            "huge",
            "massive",
            "giant",
            "revolutionary",
            "game",
            "changer",
            "future",
            "world",
            # Community excitement
            "fam",
            "community",
            "together",
            "believe",
            "hodl",
            "never",
            # Emoji patterns
            "🚀",
            "🌙",
            "💎",
            "🙌",
            "💰",
            "🤑",
            "🔥",
            "⚡",
            "🎉",
        }

        # FEAR: Capitulation, potential bottom
        self.fear_keywords = {
            # Direct emotion words
            "scared",
            "afraid",
            "terrified",
            "panic",
            "nightmare",
            "hell",
            "crash",
            "collapse",
            "dead",
            "buried",
            "rekt",
            "ruined",
            # Despair indicators
            "sell",
            "everything",
            "exit",
            "scam",
            "ponzi",
            "bubble",
            "burst",
            "zero",
            "worthless",
            "gone",
            "lost",
            "savings",
            # Market fear
            "dump",
            "dumping",
            "capitulation",
            "bloodbath",
            "massacre",
            "plummet",
            "freefall",
            "collapse",
            "implode",
            "doom",
            # Regulatory/institutional fear
            "ban",
            "illegal",
            "sec",
            "regulation",
            "shut",
            "down",
            "arrest",
            "jail",
            "investigation",
            "lawsuit",
            "fraud",
            # Emoji patterns
            "😱",
            "😰",
            "😨",
            "💀",
            "🪦",
            "📉",
            "🔻",
        }

        # GREED: FOMO, speculative behavior
        self.greed_keywords = {
            # Desire language
            "want",
            "need",
            "must",
            "have",
            "fomo",
            "missing",
            "out",
            # Quick profit seeking
            "quick",
            "fast",
            "easy",
            "instant",
            "overnight",
            "10x",
            "100x",
            # Speculation
            "betting",
            "all",
            "in",
            "leverage",
            "margin",
            "borrow",
            "loan",
            "life",
            "savings",
            "house",
            "car",
            "sell",
            "everything",
            # Chasing pumps
            "pump",
            "dump",
            "ride",
            "wave",
            "catch",
            "falling",
            "knife",
            # Emoji patterns
            "🤑",
            "🤤",
            "💸",
            "🎰",
            "🎲",
        }

        # PANIC: Confusion + fear, high volatility
        self.panic_keywords = {
            # Confusion
            "what",
            "happening",
            "dont",
            "know",
            "confused",
            "wtf",
            "help",
            # Urgency
            "now",
            "urgent",
            "immediately",
            "emergency",
            "quick",
            "fast",
            # Market chaos
            "why",
            "dropping",
            "crashing",
            "explain",
            "someone",
            "tell",
            # Desperation
            "please",
            "help",
            "understand",
            "losing",
            "everything",
            # Emoji patterns
            "😵",
            "😵‍💫",
            "🤯",
            "❓",
            "‼️",
            "🚨",
        }

        # DISGUST: Negative sentiment, potential selling pressure
        self.disgust_keywords = {
            # Direct disgust
            "disgusting",
            "terrible",
            "horrible",
            "awful",
            "hate",
            "trash",
            # Betrayal
            "liar",
            "scammer",
            "cheated",
            "stole",
            "rug",
            "pull",
            "exit",
            # Disappointment
            "promised",
            "delivered",
            "failed",
            "let",
            "down",
            "waste",
            # Technical rejection
            "bad",
            "tech",
            "slow",
            "expensive",
            "useless",
            "shitcoin",
            # Emoji patterns
            "🤮",
            "👎",
            "💩",
            "🗑️",
            "🚫",
        }

        # HOPE: Recovery phase, accumulation
        self.hope_keywords = {
            # Optimism
            "hope",
            "optimistic",
            "believe",
            "future",
            "potential",
            "promising",
            # Recovery indicators
            "building",
            "accumulating",
            "dip",
            "buy",
            "opportunity",
            "undervalued",
            "cheap",
            "sale",
            "discount",
            # Long-term vision
            "long",
            "term",
            "hold",
            "years",
            "patience",
            "wait",
            "timing",
            # Development
            "partnership",
            "adoption",
            "mainnet",
            "launch",
            "upgrade",
            "improvement",
            "progress",
            "roadmap",
            # Emoji patterns
            "🌱",
            "📈",
            "💪",
            "🙏",
            "✨",
            "🌟",
        }

        # NEUTRAL: Informational, analytical
        self.neutral_keywords = {
            # Information sharing
            "update",
            "announcement",
            "news",
            "report",
            "data",
            "analysis",
            # Questions
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "explain",
            # Technical discussion
            "chart",
            "pattern",
            "support",
            "resistance",
            "indicator",
            "rsi",
            # Price information
            "price",
            "trading",
            "at",
            "currently",
            "volume",
            "market",
            # Neutral markers
            "just",
            "sharing",
            "fyi",
            "info",
            "note",
            "reminder",
        }

        # AGGRESSION: Hostility, conflict
        self.aggression_keywords = {
            # Direct aggression
            "stupid",
            "idiot",
            "moron",
            "wrong",
            "bad",
            "hate",
            "dumb",
            # Argumentative
            "idiots",
            "losing",
            "money",
            "deserve",
            "warned",
            "told",
            # Hostility toward projects
            "scam",
            "shitcoin",
            "garbage",
            "trash",
            "worthless",
            "ponzi",
            # Emoji patterns
            "😡",
            "🤬",
            "👊",
            "⚔️",
            "💢",
        }

        # Compile all lexicons
        self.all_keywords = {
            "euphoria": self.euphoria_keywords,
            "fear": self.fear_keywords,
            "greed": self.greed_keywords,
            "panic": self.panic_keywords,
            "disgust": self.disgust_keywords,
            "hope": self.hope_keywords,
            "neutral": self.neutral_keywords,
            "aggression": self.aggression_keywords,
        }

    def detect_emotions(self, text: str, threshold: float = 0.1) -> EmotionProfile:
        """Detect emotions in text using GoEmotions taxonomy.

        Args:
            text: Text to analyze
            threshold: Minimum score threshold (0-1)

        Returns:
            Complete emotion profile
        """
        text_lower = text.lower()

        # Initialize emotion scores
        emotion_raw_scores = {
            GoEmotion.ADMIRATION: 0.0,
            GoEmotion.AMUSEMENT: 0.0,
            GoEmotion.ANGER: 0.0,
            GoEmotion.ANNOYANCE: 0.0,
            GoEmotion.APPROVAL: 0.0,
            GoEmotion.CARING: 0.0,
            GoEmotion.CONFUSION: 0.0,
            GoEmotion.CURIOSITY: 0.0,
            GoEmotion.DESIRE: 0.0,
            GoEmotion.DISAPPOINTMENT: 0.0,
            GoEmotion.DISAPPROVAL: 0.0,
            GoEmotion.DISGUST: 0.0,
            GoEmotion.EMBARRASSMENT: 0.0,
            GoEmotion.EXCITEMENT: 0.0,
            GoEmotion.FEAR: 0.0,
            GoEmotion.GRATITUDE: 0.0,
            GoEmotion.GRIEF: 0.0,
            GoEmotion.JOY: 0.0,
            GoEmotion.LOVE: 0.0,
            GoEmotion.NERVOUSNESS: 0.0,
            GoEmotion.OPTIMISM: 0.0,
            GoEmotion.PRIDE: 0.0,
            GoEmotion.REALIZATION: 0.0,
            GoEmotion.RELIEF: 0.0,
            GoEmotion.REMORSE: 0.0,
            GoEmotion.SADNESS: 0.0,
            GoEmotion.SURPRISE: 0.0,
        }

        # Score based on group keywords
        group_scores = {}

        for group, keywords in self.all_keywords.items():
            score = 0
            matched_keywords = []

            # Count keyword matches
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)

            # Normalize by text length
            if score > 0:
                group_scores[group] = score / max(1, len(text_lower.split()) * 0.1)
            else:
                group_scores[group] = 0.0

            # Map to specific GoEmotions
            if group == "euphoria":
                emotion_raw_scores[GoEmotion.JOY] = max(
                    emotion_raw_scores[GoEmotion.JOY], group_scores[group] * 0.9
                )
                emotion_raw_scores[GoEmotion.EXCITEMENT] = max(
                    emotion_raw_scores[GoEmotion.EXCITEMENT], group_scores[group] * 0.95
                )
                emotion_raw_scores[GoEmotion.AMUSEMENT] = max(
                    emotion_raw_scores[GoEmotion.AMUSEMENT], group_scores[group] * 0.7
                )
                emotion_raw_scores[GoEmotion.PRIDE] = max(
                    emotion_raw_scores[GoEmotion.PRIDE], group_scores[group] * 0.6
                )
                emotion_raw_scores[GoEmotion.ADMIRATION] = max(
                    emotion_raw_scores[GoEmotion.ADMIRATION], group_scores[group] * 0.5
                )

            elif group == "fear":
                emotion_raw_scores[GoEmotion.FEAR] = max(
                    emotion_raw_scores[GoEmotion.FEAR], group_scores[group] * 0.95
                )
                emotion_raw_scores[GoEmotion.NERVOUSNESS] = max(
                    emotion_raw_scores[GoEmotion.NERVOUSNESS], group_scores[group] * 0.9
                )
                emotion_raw_scores[GoEmotion.GRIEF] = max(
                    emotion_raw_scores[GoEmotion.GRIEF], group_scores[group] * 0.8
                )
                emotion_raw_scores[GoEmotion.SADNESS] = max(
                    emotion_raw_scores[GoEmotion.SADNESS], group_scores[group] * 0.7
                )

            elif group == "greed":
                emotion_raw_scores[GoEmotion.DESIRE] = max(
                    emotion_raw_scores[GoEmotion.DESIRE], group_scores[group] * 0.95
                )
                emotion_raw_scores[GoEmotion.EXCITEMENT] = max(
                    emotion_raw_scores[GoEmotion.EXCITEMENT], group_scores[group] * 0.7
                )
                emotion_raw_scores[GoEmotion.OPTIMISM] = max(
                    emotion_raw_scores[GoEmotion.OPTIMISM], group_scores[group] * 0.6
                )

            elif group == "panic":
                emotion_raw_scores[GoEmotion.CONFUSION] = max(
                    emotion_raw_scores[GoEmotion.CONFUSION], group_scores[group] * 0.9
                )
                emotion_raw_scores[GoEmotion.FEAR] = max(
                    emotion_raw_scores[GoEmotion.FEAR], group_scores[group] * 0.8
                )
                emotion_raw_scores[GoEmotion.NERVOUSNESS] = max(
                    emotion_raw_scores[GoEmotion.NERVOUSNESS], group_scores[group] * 0.7
                )
                emotion_raw_scores[GoEmotion.SURPRISE] = max(
                    emotion_raw_scores[GoEmotion.SURPRISE], group_scores[group] * 0.6
                )

            elif group == "disgust":
                emotion_raw_scores[GoEmotion.DISGUST] = max(
                    emotion_raw_scores[GoEmotion.DISGUST], group_scores[group] * 0.95
                )
                emotion_raw_scores[GoEmotion.DISAPPROVAL] = max(
                    emotion_raw_scores[GoEmotion.DISAPPROVAL], group_scores[group] * 0.9
                )
                emotion_raw_scores[GoEmotion.DISAPPOINTMENT] = max(
                    emotion_raw_scores[GoEmotion.DISAPPOINTMENT], group_scores[group] * 0.8
                )
                emotion_raw_scores[GoEmotion.ANNOYANCE] = max(
                    emotion_raw_scores[GoEmotion.ANNOYANCE], group_scores[group] * 0.7
                )
                emotion_raw_scores[GoEmotion.ANGER] = max(
                    emotion_raw_scores[GoEmotion.ANGER], group_scores[group] * 0.6
                )

            elif group == "hope":
                emotion_raw_scores[GoEmotion.OPTIMISM] = max(
                    emotion_raw_scores[GoEmotion.OPTIMISM], group_scores[group] * 0.95
                )
                emotion_raw_scores[GoEmotion.GRATITUDE] = max(
                    emotion_raw_scores[GoEmotion.GRATITUDE], group_scores[group] * 0.7
                )
                emotion_raw_scores[GoEmotion.CURIOSITY] = max(
                    emotion_raw_scores[GoEmotion.CURIOSITY], group_scores[group] * 0.6
                )
                emotion_raw_scores[GoEmotion.DESIRE] = max(
                    emotion_raw_scores[GoEmotion.DESIRE], group_scores[group] * 0.5
                )
                emotion_raw_scores[GoEmotion.RELIEF] = max(
                    emotion_raw_scores[GoEmotion.RELIEF], group_scores[group] * 0.4
                )

            elif group == "neutral":
                emotion_raw_scores[GoEmotion.REALIZATION] = max(
                    emotion_raw_scores[GoEmotion.REALIZATION], group_scores[group] * 0.8
                )
                emotion_raw_scores[GoEmotion.CURIOSITY] = max(
                    emotion_raw_scores[GoEmotion.CURIOSITY], group_scores[group] * 0.6
                )
                emotion_raw_scores[GoEmotion.SURPRISE] = max(
                    emotion_raw_scores[GoEmotion.SURPRISE], group_scores[group] * 0.4
                )

            elif group == "aggression":
                emotion_raw_scores[GoEmotion.ANGER] = max(
                    emotion_raw_scores[GoEmotion.ANGER], group_scores[group] * 0.95
                )
                emotion_raw_scores[GoEmotion.ANNOYANCE] = max(
                    emotion_raw_scores[GoEmotion.ANNOYANCE], group_scores[group] * 0.9
                )
                emotion_raw_scores[GoEmotion.DISAPPROVAL] = max(
                    emotion_raw_scores[GoEmotion.DISAPPROVAL], group_scores[group] * 0.7
                )

        # Normalize scores to 0-1
        max_score = max(emotion_raw_scores.values()) if emotion_raw_scores.values() else 1
        if max_score > 0:
            emotion_raw_scores = {e: min(1.0, s / max_score) for e, s in emotion_raw_scores.items()}

        # Filter by threshold
        emotion_scores = {
            e: EmotionScore(
                emotion=e,
                score=s,
                confidence=min(0.95, 0.3 + s * 0.65),
                keywords=[],
                group=self._get_emotion_group(e),
            )
            for e, s in emotion_raw_scores.items()
            if s >= threshold
        }

        # Calculate metrics
        dominant_emotion = (
            max(emotion_scores.items(), key=lambda x: x[1].score)[0]
            if emotion_scores
            else GoEmotion.REALIZATION
        )

        emotional_intensity = max(s.score for s in emotion_scores.values()) if emotion_scores else 0

        emotion_values = [s.score for s in emotion_scores.values()]
        emotional_volatility = np.std(emotion_values) if len(emotion_values) > 1 else 0

        # Calculate group distribution
        group_distribution = {}
        for emotion_score in emotion_scores.values():
            group = emotion_score.group
            if group:
                group_distribution[group] = max(
                    group_distribution.get(group, 0), emotion_score.score
                )

        # Determine trading bias
        trading_bias = self._calculate_trading_bias(group_distribution, emotional_intensity)

        return EmotionProfile(
            emotions=emotion_scores,
            dominant_emotion=dominant_emotion,
            emotional_intensity=emotional_intensity,
            emotional_volatility=emotional_volatility,
            group_distribution=group_distribution,
            trading_bias=trading_bias,
        )

    def _get_emotion_group(self, emotion: GoEmotion) -> Optional[EmotionGroup]:
        """Map GoEmotion to EmotionGroup."""
        mapping = {
            GoEmotion.JOY: EmotionGroup.EUPHORIA,
            GoEmotion.EXCITEMENT: EmotionGroup.EUPHORIA,
            GoEmotion.AMUSEMENT: EmotionGroup.EUPHORIA,
            GoEmotion.PRIDE: EmotionGroup.EUPHORIA,
            GoEmotion.LOVE: EmotionGroup.EUPHORIA,
            GoEmotion.FEAR: EmotionGroup.FEAR,
            GoEmotion.NERVOUSNESS: EmotionGroup.FEAR,
            GoEmotion.GRIEF: EmotionGroup.FEAR,
            GoEmotion.SADNESS: EmotionGroup.FEAR,
            GoEmotion.DESIRE: EmotionGroup.GREED,
            GoEmotion.EXCITEMENT: EmotionGroup.GREED,
            GoEmotion.OPTIMISM: EmotionGroup.HOPE,
            GoEmotion.CONFUSION: EmotionGroup.PANIC,
            GoEmotion.SURPRISE: EmotionGroup.NEUTRAL,
            GoEmotion.DISGUST: EmotionGroup.DISGUST,
            GoEmotion.DISAPPROVAL: EmotionGroup.DISGUST,
            GoEmotion.DISAPPOINTMENT: EmotionGroup.DISGUST,
            GoEmotion.OPTIMISM: EmotionGroup.HOPE,
            GoEmotion.GRATITUDE: EmotionGroup.HOPE,
            GoEmotion.CURIOSITY: EmotionGroup.HOPE,
            GoEmotion.REALIZATION: EmotionGroup.NEUTRAL,
            GoEmotion.ANGER: EmotionGroup.AGGRESSION,
            GoEmotion.ANNOYANCE: EmotionGroup.AGGRESSION,
            GoEmotion.ADMIRATION: EmotionGroup.EUPHORIA,
            GoEmotion.APPROVAL: EmotionGroup.EUPHORIA,
            GoEmotion.CARING: EmotionGroup.HOPE,
            GoEmotion.EMBARRASSMENT: EmotionGroup.NEUTRAL,
            GoEmotion.RELIEF: EmotionGroup.HOPE,
            GoEmotion.REMORSE: EmotionGroup.DISGUST,
        }
        return mapping.get(emotion)

    def _calculate_trading_bias(
        self, group_distribution: Dict[EmotionGroup, float], intensity: float
    ) -> str:
        """Calculate trading bias from emotion groups.

        Args:
            group_distribution: Distribution of emotion groups
            intensity: Emotional intensity (0-1)

        Returns:
            Trading bias: bullish, bearish, or neutral
        """
        bullish_score = (
            group_distribution.get(EmotionGroup.EUPHORIA, 0) * 1.0
            + group_distribution.get(EmotionGroup.HOPE, 0) * 0.7
            + group_distribution.get(EmotionGroup.GREED, 0) * 0.5
        )

        bearish_score = (
            group_distribution.get(EmotionGroup.FEAR, 0) * 1.0
            + group_distribution.get(EmotionGroup.PANIC, 0) * 0.9
            + group_distribution.get(EmotionGroup.DISGUST, 0) * 0.8
            + group_distribution.get(EmotionGroup.AGGRESSION, 0) * 0.6
        )

        if bullish_score > bearish_score * 1.3:
            return "bullish"
        elif bearish_score > bullish_score * 1.3:
            return "bearish"
        else:
            return "neutral"

    def fetch_texts_for_symbol(self, symbol: str, max_texts: int = 5) -> List[str]:
        """Fetch sample texts for sentiment analysis of a symbol.
        
        For demo purposes, returns mock texts that would typically come from:
        - Twitter/X feeds
        - Reddit discussions  
        - News headlines
        - Trading forums
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/EUR')
            max_texts: Maximum number of texts to return
            
        Returns:
            List of text samples for emotion analysis
        """
        # Mock texts based on common crypto sentiment patterns
        mock_texts = {
            'BTC/EUR': [
                "Bitcoin is breaking resistance! 🚀 To the moon everyone! HODL strong!",
                "Nervous about the current market, might sell my BTC soon",
                "BTC accumulation phase, buying the dip while prices are low",
                "This crypto crash is brutal, lost so much money on Bitcoin",
                "Bitcoin technical analysis shows strong support at current levels"
            ],
            'ETH/EUR': [
                "Ethereum upgrade is coming! ETH will explode 💰",
                "ETH gas fees are ridiculous, this can't continue",
                "Smart contract adoption on Ethereum is growing steadily",
                "Concerned about Ethereum's competition from other L1s",
                "Ethereum staking rewards look attractive for long-term holds"
            ]
        }
        
        # Get texts for symbol or return generic crypto texts
        texts = mock_texts.get(symbol, [
            "Crypto market is showing interesting patterns today",
            "Technical indicators suggest a potential breakout soon", 
            "Market volatility is high, trading with caution",
            " blockchain technology continues to evolve rapidly",
            "DeFi protocols are seeing increased usage and adoption"
        ])
        
        # Return limited number of texts
        return texts[:max_texts]

    def batch_analyze(self, texts: List[str]) -> List[EmotionProfile]:
        """Analyze multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of emotion profiles
        """
        return [self.detect_emotions(text) for text in texts]
