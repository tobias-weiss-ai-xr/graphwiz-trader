"""Specialized trading agents for different market analysis strategies."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TradingSignal(Enum):
    """Trading signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class AgentDecision:
    """Decision output from a trading agent.

    Attributes:
        signal: Trading signal (BUY/SELL/HOLD)
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Human-readable explanation for the decision
        metadata: Additional agent-specific information
        timestamp: When the decision was made
        agent_name: Name of the agent making the decision
    """

    signal: TradingSignal
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
        }


@dataclass
class AgentPerformance:
    """Performance metrics for an agent.

    Attributes:
        total_decisions: Total number of decisions made
        correct_decisions: Number of correct decisions
        accuracy: Overall accuracy (correct / total)
        profit_factor: Total profit / total loss
        average_confidence: Average confidence score
        recent_performance: Performance over last N decisions
        last_updated: Timestamp of last update
    """

    total_decisions: int = 0
    correct_decisions: int = 0
    accuracy: float = 0.0
    profit_factor: float = 1.0
    average_confidence: float = 0.0
    recent_performance: List[bool] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update(self, was_correct: bool, confidence: float, profit_loss: float) -> None:
        """Update performance metrics.

        Args:
            was_correct: Whether the decision was correct
            confidence: Confidence score of the decision
            profit_loss: Profit or loss from the decision
        """
        self.total_decisions += 1
        if was_correct:
            self.correct_decisions += 1

        self.accuracy = (
            self.correct_decisions / self.total_decisions if self.total_decisions > 0 else 0.0
        )

        # Update average confidence
        if self.total_decisions == 1:
            self.average_confidence = confidence
        else:
            self.average_confidence = (
                self.average_confidence * (self.total_decisions - 1) + confidence
            ) / self.total_decisions

        # Update profit factor (simplified - use cumulative PnL ratio)
        if profit_loss > 0:
            self.profit_factor = min(self.profit_factor * 1.01, 5.0)  # Cap at 5.0
        elif profit_loss < 0:
            self.profit_factor = max(self.profit_factor * 0.99, 0.2)  # Floor at 0.2

        # Track recent performance (last 100 decisions)
        self.recent_performance.append(was_correct)
        if len(self.recent_performance) > 100:
            self.recent_performance.pop(0)

        self.last_updated = datetime.utcnow()

    def get_recent_accuracy(self, window: int = 20) -> float:
        """Get accuracy over recent window.

        Args:
            window: Number of recent decisions to consider

        Returns:
            Recent accuracy rate
        """
        if not self.recent_performance:
            return 0.0

        recent = self.recent_performance[-window:]
        return sum(recent) / len(recent)


class TradingAgent(ABC):
    """Base class for trading agents.

    All specialized trading agents should inherit from this class and implement
    the analyze method.
    """

    def __init__(self, name: str, config: Dict[str, Any], knowledge_graph: Optional[Any] = None):
        """Initialize trading agent.

        Args:
            name: Agent name
            config: Agent configuration dictionary
            knowledge_graph: Optional knowledge graph instance
        """
        self.name = name
        self.config = config
        self.kg = knowledge_graph
        self.enabled = config.get("enabled", True)
        self.performance = AgentPerformance()

        # Configuration parameters
        self.min_confidence = config.get("min_confidence", 0.6)
        self.max_confidence = config.get("max_confidence", 0.95)
        self.learning_rate = config.get("learning_rate", 0.1)

        logger.info(f"Initialized {self.name} agent (enabled={self.enabled})")

    @abstractmethod
    async def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """Analyze market data and generate trading signal.

        Args:
            market_data: Current market data (price, volume, etc.)
            indicators: Technical indicators (RSI, MACD, etc.)
            context: Additional context information

        Returns:
            AgentDecision with signal, confidence, and reasoning
        """
        pass

    def _calculate_confidence(
        self, signal_strength: float, volatility_adjustment: float = 0.0
    ) -> float:
        """Calculate confidence score with adjustments.

        Args:
            signal_strength: Base signal strength (0-1)
            volatility_adjustment: Adjustment based on volatility

        Returns:
            Adjusted confidence score
        """
        confidence = signal_strength - volatility_adjustment

        # Apply performance adjustment
        recent_acc = self.performance.get_recent_accuracy()
        if recent_acc > 0.6:
            confidence *= 1 + self.learning_rate * (recent_acc - 0.5)
        elif recent_acc < 0.4:
            confidence *= 1 - self.learning_rate * (0.5 - recent_acc)

        # Clamp to configured range
        return max(self.min_confidence, min(self.max_confidence, confidence))

    async def update_performance(
        self, decision: AgentDecision, was_correct: bool, profit_loss: float
    ) -> None:
        """Update agent performance metrics.

        Args:
            decision: The decision that was made
            was_correct: Whether the decision was correct
            profit_loss: Profit or loss from the decision
        """
        self.performance.update(was_correct, decision.confidence, profit_loss)
        logger.debug(
            f"Updated {self.name} performance: accuracy={self.performance.accuracy:.3f}, "
            f"profit_factor={self.performance.profit_factor:.3f}"
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance.

        Returns:
            Performance metrics dictionary
        """
        return {
            "agent_name": self.name,
            "total_decisions": self.performance.total_decisions,
            "accuracy": self.performance.accuracy,
            "recent_accuracy": self.performance.get_recent_accuracy(),
            "profit_factor": self.performance.profit_factor,
            "average_confidence": self.performance.average_confidence,
            "last_updated": self.performance.last_updated.isoformat(),
        }


class TechnicalAnalysisAgent(TradingAgent):
    """Technical analysis agent using RSI, MACD, and Bollinger Bands."""

    async def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """Analyze technical indicators and generate signal.

        Args:
            market_data: Market price and volume data
            indicators: Technical indicators (RSI, MACD, BB, etc.)
            context: Additional context

        Returns:
            Trading decision based on technical analysis
        """
        signals = []
        reasoning_parts = []

        # RSI Analysis
        rsi = indicators.get("RSI", {})
        if rsi:
            rsi_value = rsi.get("value", 50)
            if rsi_value < 30:
                signals.append(("BUY", 0.7, f"RSI oversold at {rsi_value:.1f}"))
            elif rsi_value > 70:
                signals.append(("SELL", 0.7, f"RSI overbought at {rsi_value:.1f}"))
            elif rsi_value < 40:
                signals.append(("BUY", 0.4, f"RSI approaching oversold at {rsi_value:.1f}"))
            elif rsi_value > 60:
                signals.append(("SELL", 0.4, f"RSI approaching overbought at {rsi_value:.1f}"))

        # MACD Analysis
        macd = indicators.get("MACD", {})
        if macd:
            macd_line = macd.get("macd", 0)
            signal_line = macd.get("signal", 0)
            histogram = macd.get("histogram", 0)

            if histogram > 0 and macd_line > signal_line:
                signals.append(("BUY", 0.6, "MACD bullish crossover"))
            elif histogram < 0 and macd_line < signal_line:
                signals.append(("SELL", 0.6, "MACD bearish crossover"))
            elif histogram > 0:
                signals.append(("BUY", 0.3, "MACD histogram positive"))
            elif histogram < 0:
                signals.append(("SELL", 0.3, "MACD histogram negative"))

        # Bollinger Bands Analysis
        bb = indicators.get("BB", {})
        if bb:
            price = market_data.get("close", 0)
            upper_band = bb.get("upper", 0)
            lower_band = bb.get("lower", 0)
            middle_band = bb.get("middle", 0)

            if upper_band > 0 and lower_band > 0:
                bb_position = (price - lower_band) / (upper_band - lower_band)

                if bb_position < 0.1:
                    signals.append(
                        ("BUY", 0.8, f"Price near lower Bollinger Band ({bb_position:.2f})")
                    )
                elif bb_position > 0.9:
                    signals.append(
                        ("SELL", 0.8, f"Price near upper Bollinger Band ({bb_position:.2f})")
                    )
                elif bb_position < 0.3:
                    signals.append(("BUY", 0.4, f"Price in lower BB region ({bb_position:.2f})"))
                elif bb_position > 0.7:
                    signals.append(("SELL", 0.4, f"Price in upper BB region ({bb_position:.2f})"))

        # EMA Analysis
        ema = indicators.get("EMA", {})
        if ema:
            price = market_data.get("close", 0)
            ema_short = ema.get("short", 0)
            ema_long = ema.get("long", 0)

            if ema_short > ema_long and price > ema_short:
                signals.append(("BUY", 0.5, "Price above short EMA, bullish trend"))
            elif ema_short < ema_long and price < ema_short:
                signals.append(("SELL", 0.5, "Price below short EMA, bearish trend"))

        # Aggregate signals
        if not signals:
            return AgentDecision(
                signal=TradingSignal.HOLD,
                confidence=0.5,
                reasoning="No clear technical signals",
                agent_name=self.name,
                metadata={"method": "technical_analysis"},
            )

        # Weight voting by confidence
        buy_weight = sum(conf for sig, conf, _ in signals if sig == "BUY")
        sell_weight = sum(conf for sig, conf, _ in signals if sig == "SELL")

        if buy_weight > sell_weight * 1.2:
            signal = TradingSignal.BUY
            confidence = self._calculate_confidence(buy_weight / (buy_weight + sell_weight))
        elif sell_weight > buy_weight * 1.2:
            signal = TradingSignal.SELL
            confidence = self._calculate_confidence(sell_weight / (buy_weight + sell_weight))
        else:
            signal = TradingSignal.HOLD
            confidence = 0.5

        # Build reasoning
        top_signals = sorted(signals, key=lambda x: x[1], reverse=True)[:3]
        reasoning = f"Technical Analysis: " + "; ".join([s[2] for s in top_signals])

        return AgentDecision(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            agent_name=self.name,
            metadata={
                "method": "technical_analysis",
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "signal_count": len(signals),
            },
        )


class SentimentAnalysisAgent(TradingAgent):
    """Sentiment analysis agent using news and social media data."""

    async def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """Analyze market sentiment and generate signal.

        Args:
            market_data: Market price and volume data
            indicators: Sentiment indicators (news, social media)
            context: Additional context

        Returns:
            Trading decision based on sentiment analysis
        """
        context = context or {}

        # Get sentiment data
        news_sentiment = context.get("news_sentiment", {})
        social_sentiment = context.get("social_sentiment", {})
        overall_sentiment = context.get("overall_sentiment", 0.0)

        signals = []
        reasoning_parts = []

        # News sentiment
        if news_sentiment:
            sentiment_score = news_sentiment.get("score", 0.0)
            sentiment_count = news_sentiment.get("count", 0)

            if sentiment_score > 0.3 and sentiment_count >= 5:
                signals.append(("BUY", 0.7, f"Positive news sentiment ({sentiment_score:.2f})"))
            elif sentiment_score < -0.3 and sentiment_count >= 5:
                signals.append(("SELL", 0.7, f"Negative news sentiment ({sentiment_score:.2f})"))
            elif sentiment_score > 0.1:
                signals.append(
                    ("BUY", 0.4, f"Mildly positive news sentiment ({sentiment_score:.2f})")
                )
            elif sentiment_score < -0.1:
                signals.append(
                    ("SELL", 0.4, f"Mildly negative news sentiment ({sentiment_score:.2f})")
                )

        # Social media sentiment
        if social_sentiment:
            sentiment_score = social_sentiment.get("score", 0.0)
            volume = social_sentiment.get("volume", 0)
            trend = social_sentiment.get("trend", "neutral")

            if volume > 100:  # High volume
                if sentiment_score > 0.4 and trend == "rising":
                    signals.append(
                        ("BUY", 0.6, f"Strong positive social sentiment ({sentiment_score:.2f})")
                    )
                elif sentiment_score < -0.4 and trend == "falling":
                    signals.append(
                        ("SELL", 0.6, f"Strong negative social sentiment ({sentiment_score:.2f})")
                    )

        # Overall sentiment
        if abs(overall_sentiment) > 0.2:
            if overall_sentiment > 0.5:
                signals.append(
                    ("BUY", 0.5, f"Strong overall bullish sentiment ({overall_sentiment:.2f})")
                )
            elif overall_sentiment < -0.5:
                signals.append(
                    ("SELL", 0.5, f"Strong overall bearish sentiment ({overall_sentiment:.2f})")
                )

        # If no sentiment data available, hold
        if not signals:
            return AgentDecision(
                signal=TradingSignal.HOLD,
                confidence=0.5,
                reasoning="Insufficient sentiment data for analysis",
                agent_name=self.name,
                metadata={"method": "sentiment_analysis"},
            )

        # Aggregate signals
        buy_weight = sum(conf for sig, conf, _ in signals if sig == "BUY")
        sell_weight = sum(conf for sig, conf, _ in signals if sig == "SELL")

        if buy_weight > sell_weight * 1.3:
            signal = TradingSignal.BUY
            confidence = self._calculate_confidence(buy_weight / (buy_weight + sell_weight), 0.1)
        elif sell_weight > buy_weight * 1.3:
            signal = TradingSignal.SELL
            confidence = self._calculate_confidence(sell_weight / (buy_weight + sell_weight), 0.1)
        else:
            signal = TradingSignal.HOLD
            confidence = 0.5

        top_signals = sorted(signals, key=lambda x: x[1], reverse=True)[:2]
        reasoning = f"Sentiment Analysis: " + "; ".join([s[2] for s in top_signals])

        return AgentDecision(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            agent_name=self.name,
            metadata={
                "method": "sentiment_analysis",
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
                "overall_sentiment": overall_sentiment,
            },
        )


class RiskManagementAgent(TradingAgent):
    """Risk management agent for position sizing and risk assessment."""

    async def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """Analyze risk factors and generate signal.

        Args:
            market_data: Market price and volume data
            indicators: Risk indicators (volatility, drawdown, etc.)
            context: Additional context including portfolio state

        Returns:
            Risk-adjusted trading decision
        """
        context = context or {}
        portfolio = context.get("portfolio", {})

        # Get risk parameters
        volatility = indicators.get("volatility", {}).get("value", 0.0)
        max_position_size = self.config.get("max_position_size", 0.1)
        current_exposure = portfolio.get("exposure", 0.0)
        portfolio_value = portfolio.get("value", 10000)

        signals = []
        reasoning_parts = []

        # Volatility check
        if volatility > 0.05:  # High volatility (>5%)
            signals.append(
                ("HOLD", 0.8, f"High volatility ({volatility:.2%}), reduce position sizes")
            )
        elif volatility < 0.01:  # Low volatility (<1%)
            signals.append(("BUY", 0.3, "Low volatility, good entry opportunity"))

        # Exposure check
        if current_exposure > 0.8:  # Too much exposure
            signals.append(
                (
                    "SELL",
                    0.7,
                    f"High portfolio exposure ({current_exposure:.1%}), consider reducing",
                )
            )

        # Drawdown check
        drawdown = portfolio.get("drawdown", 0.0)
        if drawdown < -0.1:  # More than 10% drawdown
            signals.append(("HOLD", 0.9, f"Significant drawdown ({drawdown:.1%}), reduce risk"))
        elif drawdown < -0.05:  # 5-10% drawdown
            signals.append(("HOLD", 0.6, f"Moderate drawdown ({drawdown:.1%}), be cautious"))

        # Correlation check (if available)
        correlations = context.get("correlations", {})
        high_corr_count = sum(1 for corr in correlations.values() if abs(corr) > 0.7)
        if high_corr_count > 2:
            signals.append(
                ("HOLD", 0.5, f"High correlation ({high_corr_count} pairs), diversification risk")
            )

        # If no strong risk signals, allow trading with reduced size
        if not signals:
            return AgentDecision(
                signal=TradingSignal.HOLD,
                confidence=0.5,
                reasoning="Risk parameters within acceptable range",
                agent_name=self.name,
                metadata={
                    "method": "risk_management",
                    "volatility": volatility,
                    "recommended_position_size": max_position_size,
                },
            )

        # Aggregate risk signals
        hold_weight = sum(conf for sig, conf, _ in signals if sig == "HOLD")
        sell_weight = sum(conf for sig, conf, _ in signals if sig == "SELL")
        buy_weight = sum(conf for sig, conf, _ in signals if sig == "BUY")

        # Risk management prefers conservative actions
        if hold_weight > 0.6:
            signal = TradingSignal.HOLD
            confidence = min(hold_weight, 0.9)
        elif sell_weight > buy_weight:
            signal = TradingSignal.SELL
            confidence = min(sell_weight, 0.8)
        else:
            signal = TradingSignal.BUY
            confidence = min(buy_weight * 0.7, 0.6)  # Reduce buy confidence

        top_signals = sorted(signals, key=lambda x: x[1], reverse=True)[:2]
        reasoning = f"Risk Analysis: " + "; ".join([s[2] for s in top_signals])

        return AgentDecision(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            agent_name=self.name,
            metadata={
                "method": "risk_management",
                "volatility": volatility,
                "exposure": current_exposure,
                "drawdown": drawdown,
                "recommended_position_size": max_position_size * (1 - hold_weight),
            },
        )


class MomentumAgent(TradingAgent):
    """Momentum agent for trend-following strategies."""

    async def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """Analyze momentum and generate signal.

        Args:
            market_data: Market price and volume data
            indicators: Momentum indicators (ROC, momentum, etc.)
            context: Additional context

        Returns:
            Momentum-based trading decision
        """
        signals = []
        reasoning_parts = []

        # Rate of Change (ROC)
        roc = indicators.get("ROC", {})
        if roc:
            roc_value = roc.get("value", 0)
            if roc_value > 3:  # Strong positive momentum
                signals.append(("BUY", 0.8, f"Strong positive momentum ({roc_value:.1f}%)"))
            elif roc_value < -3:  # Strong negative momentum
                signals.append(("SELL", 0.8, f"Strong negative momentum ({roc_value:.1f}%)"))
            elif roc_value > 1:
                signals.append(("BUY", 0.5, f"Positive momentum ({roc_value:.1f}%)"))
            elif roc_value < -1:
                signals.append(("SELL", 0.5, f"Negative momentum ({roc_value:.1f}%)"))

        # Price momentum
        price_data = market_data.get("price_history", [])
        if len(price_data) >= 10:
            recent_prices = price_data[-10:]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100

            if price_change > 2:
                signals.append(
                    ("BUY", 0.6, f"Price trending up ({price_change:.1f}% over 10 periods)")
                )
            elif price_change < -2:
                signals.append(
                    ("SELL", 0.6, f"Price trending down ({price_change:.1f}% over 10 periods)")
                )

        # Volume momentum
        volume = market_data.get("volume", 0)
        volume_ma = indicators.get("volume_ma", {})
        if volume_ma and volume > 0:
            vol_ratio = volume / volume_ma.get("value", volume)
            if vol_ratio > 1.5:
                signals.append(("BUY", 0.4, f"High volume ({vol_ratio:.1f}x average)"))

        # ADX (Average Directional Index) for trend strength
        adx = indicators.get("ADX", {})
        if adx:
            adx_value = adx.get("value", 0)
            di_plus = adx.get("di_plus", 0)
            di_minus = adx.get("di_minus", 0)

            if adx_value > 25:  # Strong trend
                if di_plus > di_minus:
                    signals.append(
                        ("BUY", 0.7, f"Strong uptrend (ADX={adx_value:.1f}, DI+={di_plus:.1f})")
                    )
                else:
                    signals.append(
                        ("SELL", 0.7, f"Strong downtrend (ADX={adx_value:.1f}, DI-={di_minus:.1f})")
                    )
            elif adx_value < 20:  # Weak trend
                signals.append(
                    ("HOLD", 0.5, f"Weak trend (ADX={adx_value:.1f}), avoid momentum trades")
                )

        # If no clear momentum signals
        if not signals:
            return AgentDecision(
                signal=TradingSignal.HOLD,
                confidence=0.5,
                reasoning="No clear momentum signals",
                agent_name=self.name,
                metadata={"method": "momentum"},
            )

        # Aggregate signals
        buy_weight = sum(conf for sig, conf, _ in signals if sig == "BUY")
        sell_weight = sum(conf for sig, conf, _ in signals if sig == "SELL")

        if buy_weight > sell_weight * 1.2:
            signal = TradingSignal.BUY
            confidence = self._calculate_confidence(buy_weight / (buy_weight + sell_weight), 0.05)
        elif sell_weight > buy_weight * 1.2:
            signal = TradingSignal.SELL
            confidence = self._calculate_confidence(sell_weight / (buy_weight + sell_weight), 0.05)
        else:
            signal = TradingSignal.HOLD
            confidence = 0.5

        top_signals = sorted(signals, key=lambda x: x[1], reverse=True)[:2]
        reasoning = f"Momentum Analysis: " + "; ".join([s[2] for s in top_signals])

        return AgentDecision(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            agent_name=self.name,
            metadata={"method": "momentum", "buy_weight": buy_weight, "sell_weight": sell_weight},
        )


class MeanReversionAgent(TradingAgent):
    """Mean reversion agent for statistical arbitrage strategies."""

    async def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """Analyze mean reversion opportunities and generate signal.

        Args:
            market_data: Market price and volume data
            indicators: Statistical indicators (z-score, etc.)
            context: Additional context

        Returns:
            Mean reversion trading decision
        """
        signals = []
        reasoning_parts = []

        # Z-score analysis
        zscore = indicators.get("zscore", {})
        if zscore:
            z_value = zscore.get("value", 0)
            if z_value > 2:  # More than 2 standard deviations above mean
                signals.append(
                    ("SELL", 0.8, f"Price significantly overbought (z-score={z_value:.2f})")
                )
            elif z_value < -2:  # More than 2 standard deviations below mean
                signals.append(
                    ("BUY", 0.8, f"Price significantly oversold (z-score={z_value:.2f})")
                )
            elif z_value > 1.5:
                signals.append(("SELL", 0.5, f"Price above mean (z-score={z_value:.2f})"))
            elif z_value < -1.5:
                signals.append(("BUY", 0.5, f"Price below mean (z-score={z_value:.2f})"))

        # Bollinger Bands for mean reversion
        bb = indicators.get("BB", {})
        if bb:
            price = market_data.get("close", 0)
            upper_band = bb.get("upper", 0)
            lower_band = bb.get("lower", 0)

            if upper_band > 0 and lower_band > 0:
                bb_width = (upper_band - lower_band) / price

                if bb_width > 0.05:  # Wide bands - good for mean reversion
                    bb_position = (price - lower_band) / (upper_band - lower_band)

                    if bb_position > 0.9:
                        signals.append(
                            (
                                "SELL",
                                0.7,
                                f"Price at upper BB, expect reversion ({bb_position:.2f})",
                            )
                        )
                    elif bb_position < 0.1:
                        signals.append(
                            ("BUY", 0.7, f"Price at lower BB, expect reversion ({bb_position:.2f})")
                        )

        # Stochastic oscillator
        stoch = indicators.get("Stochastic", {})
        if stoch:
            k_value = stoch.get("k", 50)
            d_value = stoch.get("d", 50)

            if k_value > 80 and d_value > 80:
                signals.append(
                    ("SELL", 0.6, f"Stochastic overbought (K={k_value:.1f}, D={d_value:.1f})")
                )
            elif k_value < 20 and d_value < 20:
                signals.append(
                    ("BUY", 0.6, f"Stochastic oversold (K={k_value:.1f}, D={d_value:.1f})")
                )

        # Price vs moving average
        sma = indicators.get("SMA", {})
        if sma:
            price = market_data.get("close", 0)
            sma_value = sma.get("value", price)
            deviation = (price - sma_value) / sma_value

            if deviation > 0.03:  # More than 3% above SMA
                signals.append(("SELL", 0.5, f"Price {deviation:.1%} above SMA, expect reversion"))
            elif deviation < -0.03:  # More than 3% below SMA
                signals.append(("BUY", 0.5, f"Price {deviation:.1%} below SMA, expect reversion"))

        # If no mean reversion signals
        if not signals:
            return AgentDecision(
                signal=TradingSignal.HOLD,
                confidence=0.5,
                reasoning="No mean reversion opportunities detected",
                agent_name=self.name,
                metadata={"method": "mean_reversion"},
            )

        # Aggregate signals
        buy_weight = sum(conf for sig, conf, _ in signals if sig == "BUY")
        sell_weight = sum(conf for sig, conf, _ in signals if sig == "SELL")

        if buy_weight > sell_weight * 1.2:
            signal = TradingSignal.BUY
            confidence = self._calculate_confidence(buy_weight / (buy_weight + sell_weight), 0.08)
        elif sell_weight > buy_weight * 1.2:
            signal = TradingSignal.SELL
            confidence = self._calculate_confidence(sell_weight / (buy_weight + sell_weight), 0.08)
        else:
            signal = TradingSignal.HOLD
            confidence = 0.5

        top_signals = sorted(signals, key=lambda x: x[1], reverse=True)[:2]
        reasoning = f"Mean Reversion: " + "; ".join([s[2] for s in top_signals])

        return AgentDecision(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            agent_name=self.name,
            metadata={
                "method": "mean_reversion",
                "buy_weight": buy_weight,
                "sell_weight": sell_weight,
            },
        )
