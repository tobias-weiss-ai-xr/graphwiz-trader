"""
Market regime detection.

Identifies different market conditions (trending, ranging, volatile)
to adapt strategy parameters accordingly.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class MarketRegime(Enum):
    """Market regime types."""

    TRENDING_UP = "trending_up"  # Strong uptrend
    TRENDING_DOWN = "trending_down"  # Strong downtrend
    RANGING = "ranging"  # Sideways/ranging market
    VOLATILE = "volatile"  # High volatility
    LOW_VOLATILITY = "low_volatility"  # Low volatility


@dataclass
class RegimeInfo:
    """Information about current market regime."""

    regime: MarketRegime
    confidence: float  # 0-1, how confident we are
    trend_strength: float  # Trend strength (-1 to 1)
    volatility: float  # Current volatility
    adx: float  # Average directional index
    recommendations: Dict[str, Any]


class RegimeDetector:
    """Detect market regime using technical indicators."""

    def __init__(
        self,
        trend_period: int = 20,
        volatility_period: int = 20,
        adx_period: int = 14,
    ):
        """Initialize regime detector.

        Args:
            trend_period: Period for trend detection
            volatility_period: Period for volatility calculation
            adx_period: Period for ADX calculation
        """
        self.trend_period = trend_period
        self.volatility_period = volatility_period
        self.adx_period = adx_period

    def detect(self, data: pd.DataFrame) -> RegimeInfo:
        """Detect current market regime.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            RegimeInfo with current regime and recommendations
        """
        if len(data) < self.trend_period:
            return RegimeInfo(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                trend_strength=0.0,
                volatility=0.0,
                adx=0.0,
                recommendations={"action": "hold", "reason": "Insufficient data"},
            )

        # Calculate indicators
        closes = data["close"]
        highs = data["high"]
        lows = data["low"]

        # Trend detection (using moving averages and slope)
        sma_short = closes.rolling(window=10).mean()
        sma_long = closes.rolling(window=self.trend_period).mean()
        trend_slope = (sma_short.iloc[-1] - sma_long.iloc[-1]) / closes.iloc[-1]

        # ADX (trend strength)
        adx = self._calculate_adx(data, self.adx_period)
        adx_value = adx.iloc[-1] if len(adx) > 0 else 0

        # Volatility (using ATR)
        atr = self._calculate_atr(data, 14)
        volatility = (atr.iloc[-1] / closes.iloc[-1]) * 100 if len(atr) > 0 else 0

        # Price range (for ranging detection)
        recent_highs = highs.rolling(window=self.trend_period).max()
        recent_lows = lows.rolling(window=self.trend_period).min()
        price_range = (recent_highs.iloc[-1] - recent_lows.iloc[-1]) / closes.iloc[-1]

        # Determine regime
        regime, confidence = self._classify_regime(
            trend_slope, adx_value, volatility, price_range
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(regime, confidence, adx_value)

        return RegimeInfo(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_slope,
            volatility=volatility,
            adx=adx_value,
            recommendations=recommendations,
        )

    def _classify_regime(
        self,
        trend_slope: float,
        adx: float,
        volatility: float,
        price_range: float,
    ) -> tuple[MarketRegime, float]:
        """Classify market regime.

        Args:
            trend_slope: Trend slope (-1 to 1)
            adx: ADX value
            volatility: Volatility percentage
            price_range: Price range percentage

        Returns:
            (regime, confidence) tuple
        """
        confidence = 0.0

        # High volatility regime
        if volatility > 5:
            return MarketRegime.VOLATILE, min(1.0, volatility / 10)

        # Low volatility regime
        if volatility < 1:
            return MarketRegime.LOW_VOLATILITY, 0.8

        # Trending regimes (ADX > 25 indicates trending market)
        if adx > 25:
            if trend_slope > 0.02:  # Strong uptrend
                confidence = min(1.0, (adx - 25) / 25)
                return MarketRegime.TRENDING_UP, confidence
            elif trend_slope < -0.02:  # Strong downtrend
                confidence = min(1.0, (adx - 25) / 25)
                return MarketRegime.TRENDING_DOWN, confidence

        # Default: ranging market
        if price_range < 0.05:  # Tight range
            return MarketRegime.RANGING, 0.8
        else:
            return MarketRegime.RANGING, 0.5

    def _generate_recommendations(
        self, regime: MarketRegime, confidence: float, adx: float
    ) -> Dict[str, Any]:
        """Generate trading recommendations based on regime.

        Args:
            regime: Current market regime
            confidence: Detection confidence
            adx: ADX value

        Returns:
            Recommendation dictionary
        """
        if confidence < 0.5:
            return {
                "action": "hold",
                "reason": "Low confidence in regime detection",
            }

        if regime == MarketRegime.TRENDING_UP:
            return {
                "action": "buy_or_hold",
                "strategy": "trend_following",
                "reason": "Strong uptrend detected",
                "stop_loss": "2%",
                "take_profit": "8%",
            }

        elif regime == MarketRegime.TRENDING_DOWN:
            return {
                "action": "sell_or_hold",
                "strategy": "avoid_trading",
                "reason": "Downtrend detected, avoid long positions",
            }

        elif regime == MarketRegime.RANGING:
            return {
                "action": "mean_reversion",
                "strategy": "rsi_mean_reversion",
                "reason": "Ranging market, mean reversion works best",
                "oversold": 25,
                "overbought": 75,
            }

        elif regime == MarketRegime.VOLATILE:
            return {
                "action": "reduce_size",
                "strategy": "cautious",
                "reason": "High volatility, reduce position sizes",
                "position_multiplier": 0.5,
            }

        elif regime == MarketRegime.LOW_VOLATILITY:
            return {
                "action": "wait_for_breakout",
                "strategy": "breakout",
                "reason": "Low volatility, wait for breakout",
            }

        return {"action": "hold", "reason": "Unknown regime"}

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX).

        Args:
            data: DataFrame with OHLCV
            period: Calculation period

        Returns:
            ADX values
        """
        highs = data["high"]
        lows = data["low"]
        closes = data["close"]

        # True Range
        tr1 = highs - lows
        tr2 = abs(highs - closes.shift())
        tr3 = abs(lows - closes.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional movements
        dm_plus = (highs - highs.shift()).apply(lambda x: x if x > 0 else 0)
        dm_minus = (lows.shift() - lows).apply(lambda x: x if x > 0 else 0)

        # Smoothed
        atr = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()

        # DX
        di_plus = (dm_plus_smooth / atr) * 100
        di_minus = (dm_minus_smooth / atr) * 100
        dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100

        # ADX
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR).

        Args:
            data: DataFrame with OHLCV
            period: Calculation period

        Returns:
            ATR values
        """
        highs = data["high"]
        lows = data["low"]
        closes = data["close"]

        tr1 = highs - lows
        tr2 = abs(highs - closes.shift())
        tr3 = abs(lows - closes.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(window=period).mean()
