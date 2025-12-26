"""
Adaptive RSI strategy that adjusts parameters based on market regime.

This strategy automatically adapts its RSI levels based on:
- Market trend (up/down/ranging)
- Volatility (high/low)
- ADX (trend strength)
"""

import pandas as pd
from typing import Optional, Dict, Any

from ..backtesting import RSIMeanReversionStrategy
from .regime_detection import RegimeDetector, MarketRegime


class AdaptiveRSIStrategy:
    """Adaptive RSI strategy with regime-based parameter adjustment."""

    def __init__(
        self,
        base_oversold: float = 30,
        base_overbought: float = 70,
        adaptation_strength: float = 0.5,
    ):
        """Initialize adaptive RSI strategy.

        Args:
            base_oversold: Base oversold level
            base_overbought: Base overbought level
            adaptation_strength: How much to adjust (0-1)
        """
        self.base_oversold = base_oversold
        self.base_overbought = base_overbought
        self.adaptation_strength = adaptation_strength

        self.regime_detector = RegimeDetector()

        # Current parameters
        self.current_oversold = base_oversold
        self.current_overbought = base_overbought
        self.current_regime = None

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """Generate trading signal with adaptive parameters.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            "buy", "sell", or None
        """
        # Detect market regime
        regime_info = self.regime_detector.detect(data)
        self.current_regime = regime_info.regime

        # Adapt parameters based on regime
        self._adapt_parameters(regime_info)

        # Calculate RSI with adapted parameters
        from ..analysis import TechnicalIndicators

        prices = data["close"].tolist()
        rsi_result = TechnicalIndicators.rsi(prices, period=14)

        if not rsi_result.values or rsi_result.values[-1] is None:
            return None

        rsi_value = rsi_result.values[-1]

        # Generate signal using adapted parameters
        if rsi_value <= self.current_oversold:
            return "buy"

        if rsi_value >= self.current_overbought:
            return "sell"

        return None

    def _adapt_parameters(self, regime_info):
        """Adapt RSI parameters based on market regime.

        Args:
            regime_info: RegimeInfo from detector
        """
        regime = regime_info.regime
        confidence = regime_info.confidence

        # Only adapt if we're confident
        if confidence < 0.5:
            return

        if regime == MarketRegime.TRENDING_UP:
            # In uptrend: lower oversold (catch dips), higher overbought (let profits run)
            target_oversold = 20
            target_overbought = 80

        elif regime == MarketRegime.TRENDING_DOWN:
            # In downtrend: avoid buying, sell quickly
            target_oversold = 35  # Less aggressive buying
            target_overbought = 60  # More aggressive selling

        elif regime == MarketRegime.RANGING:
            # In ranging: standard mean reversion
            target_oversold = 25
            target_overbought = 75

        elif regime == MarketRegime.VOLATILE:
            # In volatile: wider bands
            target_oversold = 20
            target_overbought = 80

        elif regime == MarketRegime.LOW_VOLATILITY:
            # In low volatility: narrower bands
            target_oversold = 35
            target_overbought = 65

        else:
            target_oversold = self.base_oversold
            target_overbought = self.base_overbought

        # Adapt gradually
        self.current_oversold = self._blend(
            self.current_oversold, target_oversold, self.adaptation_strength * confidence
        )
        self.current_overbought = self._blend(
            self.current_overbought, target_overbought, self.adaptation_strength * confidence
        )

    def _blend(self, current: float, target: float, strength: float) -> float:
        """Blend current and target values.

        Args:
            current: Current value
            target: Target value
            strength: Blending strength (0-1)

        Returns:
            Blended value
        """
        return current + (target - current) * strength

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current adapted parameters.

        Returns:
            Dictionary with current parameters
        """
        return {
            "oversold": self.current_oversold,
            "overbought": self.current_overbought,
            "regime": self.current_regime.value if self.current_regime else None,
        }

    def get_recommendations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get trading recommendations based on regime.

        Args:
            data: Market data DataFrame

        Returns:
            Recommendations dictionary
        """
        regime_info = self.regime_detector.detect(data)

        recommendations = {
            "regime": regime_info.regime.value,
            "confidence": regime_info.confidence,
            "trend_strength": regime_info.trend_strength,
            "volatility": regime_info.volatility,
            "current_parameters": self.get_current_parameters(),
            "recommendations": regime_info.recommendations,
        }

        return recommendations
