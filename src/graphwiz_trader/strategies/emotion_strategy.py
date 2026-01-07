"""Advanced emotion-based trading strategy using GoEmotions taxonomy.

Implements sophisticated trading signals based on fine-grained emotion detection
and market psychology principles from behavioral finance.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from graphwiz_trader.sentiment.goemotions_analyzer import (
    GoEmotionsAnalyzer,
    EmotionProfile,
    GoEmotion,
    EmotionGroup,
)


class MarketPhase(Enum):
    """Market psychological phases."""

    ACCUMULATION = "accumulation"  # Hope, optimism (BUY)
    MARKUP = "markup"  # Excitement, joy (HOLD/BUY)
    DISTRIBUTION = "distribution"  # Euphoria, greed (SELL)
    MARKDOWN = "markdown"  # Fear, panic (WAIT/HOLD)
    CAPITULATION = "capitulation"  # Despair, grief (BUY - contrarian)


class EmotionSignal(Enum):
    """Trading signals based on emotion analysis."""

    STRONG_BUY = "strong_buy"  # Capitulation, extreme fear
    BUY = "buy"  # Accumulation, hope
    HOLD = "hold"  # Neutral, mixed emotions
    SELL = "sell"  # Distribution, euphoria
    STRONG_SELL = "strong_sell"  # Mania, extreme greed


@dataclass
class EmotionSignalResult:
    """Result of emotion-based signal generation."""

    signal: EmotionSignal
    confidence: float
    market_phase: MarketPhase
    dominant_emotion: GoEmotion
    emotional_intensity: float  # 0 to 1
    contrarian_indicator: bool  # True if extreme emotion suggests reversal
    emotion_velocity: float  # Rate of emotion change
    reasoning: List[str]  # Human-readable explanation
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "market_phase": self.market_phase.value,
            "dominant_emotion": self.dominant_emotion.value,
            "intensity": self.emotional_intensity,
            "contrarian": self.contrarian_indicator,
            "velocity": self.emotion_velocity,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EmotionStrategyConfig:
    """Configuration for emotion-based strategy."""

    # Intensity thresholds (0-1)
    extreme_euphoria_threshold: float = 0.8  # Mania zone
    extreme_fear_threshold: float = 0.8  # Panic zone

    # Signal generation
    capitulation_buy_threshold: float = 0.7  # Buy when fear > 0.7
    euphoria_sell_threshold: float = 0.7  # Sell when euphoria > 0.7

    # Contrarian indicators
    use_contrarian_signals: bool = True  # Go against extreme emotions
    contrarian_confidence_boost: float = 0.15  # Boost confidence for contrarian trades

    # Emotion velocity
    velocity_periods: int = 3  # Periods for velocity calc
    velocity_threshold: float = 0.3  # Significant change threshold

    # Volume requirements
    min_data_points: int = 10
    min_confidence: float = 0.5

    # Position sizing
    contrarian_multiplier: float = 1.3  # Increase size for contrarian trades
    max_emotion_position_pct: float = 0.30  # Max 30% for emotion trades


class EmotionBasedStrategy:
    """Advanced trading strategy based on GoEmotions analysis."""

    def __init__(self, config: EmotionStrategyConfig, knowledge_graph=None):
        """Initialize emotion-based strategy.

        Args:
            config: Strategy configuration
            knowledge_graph: Optional Neo4j instance
        """
        self.config = config
        self.kg = knowledge_graph
        self.analyzer = GoEmotionsAnalyzer()

        # Historical emotion data for trend analysis
        self.emotion_history: Dict[str, List[Tuple[datetime, EmotionProfile]]] = {}

        logger.info("Emotion-based trading strategy initialized with GoEmotions")

    def analyze_emotions(
        self, texts: List[str]
    ) -> List[EmotionProfile]:
        """Analyze emotions for texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of emotion profiles
        """
        profiles = []
        for text in texts:
            profile = self.analyzer.detect_emotions(text)
            profiles.append(profile)

# Return emotion profiles for processing
        logger.info(
            f"Analyzed {len(profiles)} texts, "
            f"dominant emotion: {profiles[0].dominant_emotion.value if profiles else 'N/A'}"
        )

        return profiles

    def calculate_emotion_velocity(self, symbol: str, periods: int = 3) -> float:
        """Calculate rate of change in emotional intensity.

        Args:
            symbol: Trading symbol
            periods: Number of periods to compare

        Returns:
            Velocity score (negative = worsening sentiment, positive = improving)
        """
        if symbol not in self.emotion_history:
            return 0.0

        history = sorted(self.emotion_history[symbol], key=lambda x: x[0])

        if len(history) < periods * 2:
            return 0.0

        # Calculate average intensity for recent vs older periods
        recent = history[-periods:]
        older = history[-(periods * 2) : -periods]

        recent_intensity = np.mean([p.emotional_intensity for _, p in recent])
        older_intensity = np.mean([p.emotional_intensity for _, p in older])

        velocity = recent_intensity - older_intensity

        # Also consider emotion group shifts
        recent_groups = {}
        for _, p in recent:
            if p.dominant_emotion:
                group = self.analyzer._get_emotion_group(p.dominant_emotion)
                if group:
                    recent_groups[group] = recent_groups.get(group, 0) + 1

        older_groups = {}
        for _, p in older:
            if p.dominant_emotion:
                group = self.analyzer._get_emotion_group(p.dominant_emotion)
                if group:
                    older_groups[group] = older_groups.get(group, 0) + 1

        # Calculate shift score (positive = shift to positive emotions)
        positive_groups = {EmotionGroup.EUPHORIA, EmotionGroup.HOPE}
        negative_groups = {
            EmotionGroup.FEAR,
            EmotionGroup.PANIC,
            EmotionGroup.DISGUST,
            EmotionGroup.AGGRESSION,
        }

        recent_positive = sum(recent_groups.get(g, 0) for g in positive_groups)
        recent_negative = sum(recent_groups.get(g, 0) for g in negative_groups)

        older_positive = sum(older_groups.get(g, 0) for g in positive_groups)
        older_negative = sum(older_groups.get(g, 0) for g in negative_groups)

        group_shift = (recent_positive - recent_negative) - (older_positive - older_negative)

        # Combine intensity and group shift
        combined_velocity = (velocity + group_shift / periods) / 2

        return combined_velocity

    def identify_market_phase(self, profile: EmotionProfile) -> MarketPhase:
        """Identify current market psychological phase.

        Based on emotion profile and Dow Theory market phases.

        Args:
            profile: Current emotion profile

        Returns:
            Market phase
        """
        dominant = profile.dominant_emotion
        intensity = profile.emotional_intensity

        # CAPITULATION: Extreme fear, grief
        if dominant in [GoEmotion.FEAR, GoEmotion.GRIEF, GoEmotion.SADNESS]:
            if intensity > 0.7:
                return MarketPhase.CAPITULATION
            return MarketPhase.MARKDOWN

        # ACCUMULATION: Hope, optimism after fear
        if dominant in [GoEmotion.OPTIMISM, GoEmotion.GRATITUDE]:
            if intensity < 0.5:
                return MarketPhase.ACCUMULATION

        # MARKUP: Joy, excitement building
        if dominant in [GoEmotion.JOY, GoEmotion.EXCITEMENT, GoEmotion.ADMIRATION]:
            if intensity < 0.7:
                return MarketPhase.MARKUP
            return MarketPhase.DISTRIBUTION

        # DISTRIBUTION: Euphoria, greed, desire
        if dominant in [GoEmotion.PRIDE, GoEmotion.DESIRE]:
            if intensity > 0.6:
                return MarketPhase.DISTRIBUTION

        # PANIC/FEAR transition
        if dominant in [GoEmotion.NERVOUSNESS, GoEmotion.CONFUSION, GoEmotion.SURPRISE]:
            return MarketPhase.MARKDOWN

        # Default to accumulation/neutral
        return MarketPhase.ACCUMULATION

    def generate_signal(
        self, symbol: str, current_price: float, balance: float
    ) -> Optional[EmotionSignalResult]:
        """Generate trading signal based on emotion analysis.

        Implements contrarian strategies based on behavioral finance principles.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            balance: Available balance

        Returns:
            Signal result or None if insufficient data
        """
        logger.info(f"Generating emotion-based signal for {symbol}")

        # Get recent emotion profiles
        if symbol not in self.emotion_history:
            logger.warning(f"No emotion history for {symbol}")
            return None

        cutoff = datetime.now() - timedelta(hours=24)
        recent_profiles = [p for ts, p in self.emotion_history[symbol] if ts > cutoff]

        if len(recent_profiles) < self.config.min_data_points:
            logger.warning(f"Insufficient emotion data for {symbol}: {len(recent_profiles)}")
            return None

        # Get latest profile
        latest_profile = recent_profiles[-1]

        # Identify market phase
        market_phase = self.identify_market_phase(latest_profile)

        # Calculate emotion velocity
        velocity = self.calculate_emotion_velocity(symbol, self.config.velocity_periods)

        # Determine signal based on phase and emotion
        signal, confidence, reasoning = self._calculate_signal_from_phase(
            market_phase, latest_profile, velocity
        )

        # Check for contrarian indicator
        contrarian = self._is_contrarian_indicator(latest_profile, market_phase)

        if contrarian and self.config.use_contrarian_signals:
            # Flip signal for extreme emotions
            if signal == EmotionSignal.SELL and latest_profile.emotional_intensity > 0.8:
                signal = EmotionSignal.BUY
                confidence = min(0.95, confidence + self.config.contrarian_confidence_boost)
                reasoning.append("CONTRARIAN: Extreme euphoria suggests potential top")
            elif signal == EmotionSignal.BUY and latest_profile.emotional_intensity > 0.8:
                signal = EmotionSignal.HOLD
                reasoning.append("CAUTION: Extreme emotion, waiting for confirmation")

        result = EmotionSignalResult(
            signal=signal,
            confidence=confidence,
            market_phase=market_phase,
            dominant_emotion=latest_profile.dominant_emotion,
            emotional_intensity=latest_profile.emotional_intensity,
            contrarian_indicator=contrarian,
            emotion_velocity=velocity,
            reasoning=reasoning,
        )

        logger.info(
            f"{symbol} Signal: {signal.value} | "
            f"Phase: {market_phase.value} | "
            f"Intensity: {latest_profile.emotional_intensity:.2f} | "
            f"Contrarian: {contrarian}"
        )

        return result

    def _calculate_signal_from_phase(
        self, phase: MarketPhase, profile: EmotionProfile, velocity: float
    ) -> Tuple[EmotionSignal, float, List[str]]:
        """Calculate signal from market phase and emotion profile.

        Args:
            phase: Market phase
            profile: Emotion profile
            velocity: Emotion velocity

        Returns:
            Tuple of (signal, confidence, reasoning)
        """
        reasoning = []

        # CAPITULATION: Strong buy signal (contrarian)
        if phase == MarketPhase.CAPITULATION:
            reasoning.append("Market capitulation detected (extreme fear)")
            reasoning.append("Historically strong buying opportunity")

            if profile.emotional_intensity > 0.8:
                return EmotionSignal.STRONG_BUY, 0.85, reasoning
            else:
                return EmotionSignal.BUY, 0.75, reasoning

        # ACCUMULATION: Buy signal
        elif phase == MarketPhase.ACCUMULATION:
            reasoning.append("Early accumulation phase (hope returning)")

            if velocity > 0:  # Improving sentiment
                reasoning.append("Positive emotion velocity")
                return EmotionSignal.BUY, 0.70, reasoning
            else:
                return EmotionSignal.BUY, 0.60, reasoning

        # MARKUP: Hold or early sell
        elif phase == MarketPhase.MARKUP:
            reasoning.append("Markup phase (enthusiasm building)")

            # If velocity is very high and intensity is rising, consider taking profits
            if velocity > 0.4 and profile.emotional_intensity > 0.6:
                reasoning.append("Rapid emotion increase suggests late stage")
                return EmotionSignal.HOLD, 0.55, reasoning
            else:
                return EmotionSignal.HOLD, 0.50, reasoning

        # DISTRIBUTION: Sell signal
        elif phase == MarketPhase.DISTRIBUTION:
            reasoning.append("Distribution phase (euphoria/greed)")

            if profile.emotional_intensity > 0.7:
                reasoning.append("High emotional intensity suggests market top")
                return EmotionSignal.SELL, 0.75, reasoning
            else:
                return EmotionSignal.SELL, 0.65, reasoning

        # MARKDOWN: Wait or prepare
        else:  # MARKDOWN or default
            reasoning.append("Markdown phase (fear/panic)")

            if velocity < -0.3:  # Rapidly worsening
                reasoning.append("Rapid emotion decline - wait for capitulation")
                return EmotionSignal.HOLD, 0.50, reasoning
            elif profile.emotional_intensity > 0.6:
                # Near capitulation
                reasoning.append("High fear - approaching contrarian buy zone")
                return EmotionSignal.BUY, 0.60, reasoning
            else:
                return EmotionSignal.HOLD, 0.45, reasoning

    def _is_contrarian_indicator(self, profile: EmotionProfile, phase: MarketPhase) -> bool:
        """Check if extreme emotion suggests contrarian trade.

        Based on Fear & Greed Index principles.

        Args:
            profile: Emotion profile
            phase: Market phase

        Returns:
            True if contrarian indicator
        """
        # Extreme euphoria at market top
        if (
            phase in [MarketPhase.DISTRIBUTION, MarketPhase.MARKUP]
            and profile.emotional_intensity > self.config.extreme_euphoria_threshold
        ):
            return True

        # Extreme fear at market bottom
        if (
            phase == MarketPhase.CAPITULATION
            and profile.emotional_intensity > self.config.extreme_fear_threshold
        ):
            return True

        # High emotional volatility (uncertainty = opportunity)
        if profile.emotional_volatility > 0.4:
            return True

        return False

    def calculate_position_size(
        self,
        signal: EmotionSignalResult,
        current_price: float,
        balance: float,
        base_position_eur: float,
    ) -> float:
        """Calculate position size based on emotion signal.

        Args:
            signal: Emotion signal result
            current_price: Current price
            balance: Available balance
            base_position_eur: Base position in EUR

        Returns:
            Position size in EUR
        """
        position_eur = base_position_eur

        # Adjust for signal strength
        if signal.signal == EmotionSignal.STRONG_BUY:
            position_eur *= 1.5
        elif signal.signal == EmotionSignal.BUY:
            position_eur *= 1.0
        elif signal.signal == EmotionSignal.HOLD:
            position_eur *= 0
        elif signal.signal == EmotionSignal.SELL:
            position_eur *= 0.5  # Reduce position
        elif signal.signal == EmotionSignal.STRONG_SELL:
            position_eur *= 0  # Exit completely

        # Contrarian multiplier
        if signal.contrarian_indicator and signal.signal in [
            EmotionSignal.BUY,
            EmotionSignal.STRONG_BUY,
        ]:
            position_eur *= self.config.contrarian_multiplier

        # Confidence adjustment
        if signal.signal != EmotionSignal.HOLD:
            confidence_multiplier = 0.5 + (signal.confidence * 0.5)
            position_eur *= confidence_multiplier

        # Respect maximum
        max_position = balance * self.config.max_emotion_position_pct
        position_eur = min(position_eur, max_position)

        logger.info(
            f"Position calculation: €{position_eur:.2f} "
            f"(base: €{base_position_eur:.2f}, contrarian: {signal.contrarian_indicator})"
        )

        return position_eur

    def get_market_emotion_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive emotion summary for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with emotion summary
        """
        if symbol not in self.emotion_history:
            return {"symbol": symbol, "data_points": 0, "message": "No emotion data available"}

        cutoff = datetime.now() - timedelta(hours=24)
        recent = [p for ts, p in self.emotion_history[symbol] if ts > cutoff]

        if not recent:
            return {"symbol": symbol, "data_points": 0, "message": "No recent emotion data"}

        # Aggregate emotions
        emotion_counts = {}
        intensity_values = []
        group_distribution = {}

        for profile in recent:
            if profile.dominant_emotion:
                emotion = profile.dominant_emotion.value
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            intensity_values.append(profile.emotional_intensity)

            for group, score in profile.group_distribution.items():
                group_distribution[group.value] = group_distribution.get(group.value, 0) + score

        # Get latest
        latest = recent[-1]

        return {
            "symbol": symbol,
            "data_points": len(recent),
            "dominant_emotion": latest.dominant_emotion.value,
            "emotional_intensity": latest.emotional_intensity,
            "emotional_volatility": latest.emotional_volatility,
            "trading_bias": latest.trading_bias,
            "market_phase": self.identify_market_phase(latest).value,
            "average_intensity": np.mean(intensity_values),
            "top_emotions": sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "emotion_distribution": group_distribution,
            "timestamp": latest.timestamp.isoformat(),
        }

    # Alias for backward compatibility
    get_emotion_summary = get_market_emotion_summary

    def analyze_emotions_for_symbol(self, symbol: str, texts: List[str], timestamps: List[datetime]) -> None:
        """Analyze emotions for a specific symbol.

        Args:
            symbol: Trading symbol
            texts: List of texts to analyze
            timestamps: List of timestamps for each text
        """
        if symbol not in self.emotion_history:
            self.emotion_history[symbol] = []

        # Analyze each text
        profiles = []
        for text in texts:
            profile = self.analyzer.detect_emotions(text)
            profiles.append(profile)

        # Store in history with timestamps
        self.emotion_history[symbol].extend(
            [(ts, profile) for ts, profile in zip(timestamps, profiles)]
        )

        # Clean old data (keep 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        self.emotion_history[symbol] = [
            (ts, p) for ts, p in self.emotion_history[symbol] if ts > cutoff
        ]

        logger.info(
            f"{symbol}: Analyzed {len(profiles)} texts, "
            f"dominant emotion: {profiles[0].dominant_emotion.value if profiles else 'N/A'}"
        )


class EmotionStrategyFactory:
    """Factory for creating emotion-based strategies."""

    @staticmethod
    def create_from_config(config: Dict[str, Any], knowledge_graph=None) -> EmotionBasedStrategy:
        """Create emotion strategy from configuration.

        Args:
            config: Configuration dictionary
            knowledge_graph: Optional Neo4j instance

        Returns:
            Configured emotion-based strategy
        """
        strategy_config = EmotionStrategyConfig(
            extreme_euphoria_threshold=config.get("extreme_euphoria_threshold", 0.8),
            extreme_fear_threshold=config.get("extreme_fear_threshold", 0.8),
            capitulation_buy_threshold=config.get("capitulation_buy_threshold", 0.7),
            euphoria_sell_threshold=config.get("euphoria_sell_threshold", 0.7),
            use_contrarian_signals=config.get("use_contrarian_signals", True),
            contrarian_confidence_boost=config.get("contrarian_confidence_boost", 0.15),
            velocity_periods=config.get("velocity_periods", 3),
            velocity_threshold=config.get("velocity_threshold", 0.3),
            min_data_points=config.get("min_data_points", 10),
            min_confidence=config.get("min_confidence", 0.5),
            contrarian_multiplier=config.get("contrarian_multiplier", 1.3),
            max_emotion_position_pct=config.get("max_emotion_position_pct", 0.30),
        )

        return EmotionBasedStrategy(config=strategy_config, knowledge_graph=knowledge_graph)
