"""Technical indicators for market analysis."""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class IndicatorResult:
    """Result of an indicator calculation."""

    name: str
    values: List[float]
    signals: List[str] = None  # "buy", "sell", "neutral"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.signals is None:
            self.signals = []
        if self.metadata is None:
            self.metadata = {}


class TechnicalIndicators:
    """Technical analysis indicators."""

    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average.

        Args:
            prices: List of prices
            period: Period for SMA

        Returns:
            List of SMA values (None for insufficient data)
        """
        if len(prices) < period:
            return [None] * len(prices)

        sma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                sma_values.append(None)
            else:
                avg = sum(prices[i - period + 1 : i + 1]) / period
                sma_values.append(avg)

        return sma_values

    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average.

        Args:
            prices: List of prices
            period: Period for EMA

        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return [None] * len(prices)

        multiplier = 2 / (period + 1)
        ema_values = [None] * (period - 1)

        # Start with SMA for first value
        initial_sma = sum(prices[:period]) / period
        ema_values.append(initial_sma)

        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = (prices[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)

        return ema_values

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> IndicatorResult:
        """Calculate Relative Strength Index.

        Args:
            prices: List of prices
            period: Period for RSI (default 14)

        Returns:
            IndicatorResult with RSI values and signals
        """
        if len(prices) < period + 1:
            return IndicatorResult(
                name="RSI",
                values=[None] * len(prices),
                signals=["neutral"] * len(prices),
                metadata={"period": period},
            )

        # Calculate price changes
        price_changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(change, 0) for change in price_changes]
        losses = [abs(min(change, 0)) for change in price_changes]

        # Calculate average gains and losses
        avg_gains = []
        avg_losses = []

        # First average
        avg_gains.append(sum(gains[:period]) / period)
        avg_losses.append(sum(losses[:period]) / period)

        # Subsequent averages using Wilder's smoothing
        for i in range(period, len(gains)):
            avg_gain = (avg_gains[-1] * (period - 1) + gains[i]) / period
            avg_loss = (avg_losses[-1] * (period - 1) + losses[i]) / period
            avg_gains.append(avg_gain)
            avg_losses.append(avg_loss)

        # Calculate RSI
        rsi_values = [None] * period
        signals = ["neutral"] * period

        for i in range(len(avg_gains)):
            if avg_losses[i] == 0:
                rsi = 100.0
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

            # Generate signals
            if rsi >= 70:
                signals.append("sell")  # Overbought
            elif rsi <= 30:
                signals.append("buy")  # Oversold
            else:
                signals.append("neutral")

        return IndicatorResult(
            name="RSI", values=rsi_values, signals=signals, metadata={"period": period}
        )

    @staticmethod
    def macd(
        prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> IndicatorResult:
        """Calculate Moving Average Convergence Divergence.

        Args:
            prices: List of prices
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)

        Returns:
            IndicatorResult with MACD line, signal line, and histogram
        """
        if len(prices) < slow_period + signal_period:
            return IndicatorResult(
                name="MACD",
                values=[None] * len(prices),
                metadata={
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period,
                },
            )

        # Calculate EMAs
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)

        # Calculate MACD line
        macd_line = []
        for i in range(len(prices)):
            if fast_ema[i] is None or slow_ema[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(fast_ema[i] - slow_ema[i])

        # Calculate signal line (EMA of MACD)
        valid_macd = [m for m in macd_line if m is not None]
        signal_ema = TechnicalIndicators.ema(valid_macd, signal_period)

        # Align signal line with original data
        signal_line = []
        macd_idx = 0
        for i in range(len(prices)):
            if macd_line[i] is None:
                signal_line.append(None)
            elif macd_idx < len(signal_ema) - len(valid_macd):
                signal_line.append(None)
                macd_idx += 1
            else:
                sig_ema_idx = macd_idx - (len(signal_ema) - len(valid_macd))
                if sig_ema_idx >= 0 and sig_ema_idx < len(signal_ema):
                    signal_line.append(signal_ema[sig_ema_idx])
                else:
                    signal_line.append(None)
                macd_idx += 1

        # Calculate histogram
        histogram = []
        signals = []

        for i in range(len(prices)):
            if macd_line[i] is None or signal_line[i] is None:
                histogram.append(None)
                signals.append("neutral")
            else:
                hist = macd_line[i] - signal_line[i]
                histogram.append(hist)

                # Generate signals based on histogram crossover
                if i > 0 and histogram[i - 1] is not None:
                    if histogram[i - 1] < 0 and histogram[i] > 0:
                        signals.append("buy")  # Bullish crossover
                    elif histogram[i - 1] > 0 and histogram[i] < 0:
                        signals.append("sell")  # Bearish crossover
                    else:
                        signals.append("neutral")
                else:
                    signals.append("neutral")

        # Store values as dict for each point
        values = [
            {"macd": macd_line[i], "signal": signal_line[i], "histogram": histogram[i]}
            for i in range(len(prices))
        ]

        return IndicatorResult(
            name="MACD",
            values=values,
            signals=signals,
            metadata={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
            },
        )

    @staticmethod
    def bollinger_bands(
        prices: List[float], period: int = 20, std_dev: float = 2.0
    ) -> IndicatorResult:
        """Calculate Bollinger Bands.

        Args:
            prices: List of prices
            period: Period for moving average (default 20)
            std_dev: Standard deviation multiplier (default 2.0)

        Returns:
            IndicatorResult with upper, middle, and lower bands
        """
        if len(prices) < period:
            return IndicatorResult(
                name="BB",
                values=[None] * len(prices),
                signals=["neutral"] * len(prices),
                metadata={"period": period, "std_dev": std_dev},
            )

        # Calculate middle band (SMA)
        sma = TechnicalIndicators.sma(prices, period)

        # Calculate upper and lower bands
        upper_band = []
        lower_band = []
        signals = []

        for i in range(len(prices)):
            if i < period - 1:
                upper_band.append(None)
                lower_band.append(None)
                signals.append("neutral")
            else:
                window = prices[i - period + 1 : i + 1]
                std = np.std(window)
                middle = sma[i]

                upper = middle + (std_dev * std)
                lower = middle - (std_dev * std)

                upper_band.append(upper)
                lower_band.append(lower)

                # Generate signals based on price position
                price = prices[i]
                if price <= lower:
                    signals.append("buy")  # Price at lower band (oversold)
                elif price >= upper:
                    signals.append("sell")  # Price at upper band (overbought)
                else:
                    signals.append("neutral")

        # Store values as dict
        values = [
            {"upper": upper_band[i], "middle": sma[i], "lower": lower_band[i]}
            for i in range(len(prices))
        ]

        return IndicatorResult(
            name="BB",
            values=values,
            signals=signals,
            metadata={"period": period, "std_dev": std_dev},
        )

    @staticmethod
    def vwap(prices: List[float], volumes: List[float]) -> List[float]:
        """Calculate Volume Weighted Average Price.

        Args:
            prices: List of prices
            volumes: List of volumes

        Returns:
            List of VWAP values (cumulative from start)
        """
        if len(prices) != len(volumes):
            logger.error("Prices and volumes must have same length")
            return [None] * len(prices)

        if len(prices) == 0:
            return []

        vwap_values = []
        cumulative_pv = 0.0  # Price * Volume
        cumulative_volume = 0.0

        for i in range(len(prices)):
            cumulative_pv += prices[i] * volumes[i]
            cumulative_volume += volumes[i]

            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
                vwap_values.append(vwap)
            else:
                vwap_values.append(None)

        return vwap_values

    @staticmethod
    def atr(
        high: List[float], low: List[float], close: List[float], period: int = 14
    ) -> List[float]:
        """Calculate Average True Range.

        Args:
            high: List of high prices
            low: List of low prices
            close: List of close prices
            period: Period for ATR (default 14)

        Returns:
            List of ATR values
        """
        if len(high) != len(low) or len(high) != len(close):
            logger.error("High, low, and close must have same length")
            return [None] * len(high)

        if len(high) < period + 1:
            return [None] * len(high)

        # Calculate True Range
        tr_values = []
        for i in range(len(high)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i - 1])
                tr3 = abs(low[i] - close[i - 1])
                tr = max(tr1, tr2, tr3)
            tr_values.append(tr)

        # Calculate ATR using RMA method
        atr_values = [None] * period
        atr_values.append(sum(tr_values[: period + 1]) / (period + 1))

        for i in range(period + 1, len(tr_values)):
            atr = (atr_values[-1] * period + tr_values[i]) / (period + 1)
            atr_values.append(atr)

        return atr_values


class TechnicalAnalysis:
    """Main technical analysis class combining all indicators."""

    def __init__(self):
        """Initialize technical analysis engine."""
        self.indicators = TechnicalIndicators()

    def analyze(
        self,
        prices: List[float],
        volumes: List[float] = None,
        high: List[float] = None,
        low: List[float] = None,
        close: List[float] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive technical analysis.

        Args:
            prices: List of typical prices (or close if using OHLC)
            volumes: List of volumes (optional, for VWAP)
            high: List of high prices (optional, for ATR)
            low: List of low prices (optional, for ATR)
            close: List of close prices (optional, for ATR)

        Returns:
            Dictionary with all indicator results and overall signal
        """
        results = {}

        # RSI
        rsi_result = self.indicators.rsi(prices)
        results["rsi"] = {
            "values": rsi_result.values,
            "latest": rsi_result.values[-1] if rsi_result.values else None,
            "signal": rsi_result.signals[-1] if rsi_result.signals else "neutral",
        }

        # MACD
        macd_result = self.indicators.macd(prices)
        results["macd"] = {
            "values": macd_result.values,
            "latest": macd_result.values[-1] if macd_result.values else None,
            "signal": macd_result.signals[-1] if macd_result.signals else "neutral",
        }

        # Bollinger Bands
        bb_result = self.indicators.bollinger_bands(prices)
        results["bollinger_bands"] = {
            "values": bb_result.values,
            "latest": bb_result.values[-1] if bb_result.values else None,
            "signal": bb_result.signals[-1] if bb_result.signals else "neutral",
        }

        # EMA
        ema_20 = self.indicators.ema(prices, 20)
        ema_50 = self.indicators.ema(prices, 50)
        results["ema"] = {
            "ema_20": ema_20,
            "ema_50": ema_50,
            "latest_20": ema_20[-1] if ema_20 else None,
            "latest_50": ema_50[-1] if ema_50 else None,
        }

        # VWAP (if volumes provided)
        if volumes:
            vwap_values = self.indicators.vwap(prices, volumes)
            results["vwap"] = {
                "values": vwap_values,
                "latest": vwap_values[-1] if vwap_values else None,
            }

        # ATR (if OHLC provided)
        if high and low and close:
            atr_values = self.indicators.atr(high, low, close)
            results["atr"] = {
                "values": atr_values,
                "latest": atr_values[-1] if atr_values else None,
            }

        # Calculate overall signal
        results["overall_signal"] = self._calculate_overall_signal(results)

        return results

    def _calculate_overall_signal(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall trading signal from all indicators.

        Args:
            results: Dictionary with all indicator results

        Returns:
            Dictionary with overall signal and confidence
        """
        buy_signals = 0
        sell_signals = 0
        total_signals = 0

        # Count signals from each indicator
        for indicator in ["rsi", "macd", "bollinger_bands"]:
            if indicator in results:
                signal = results[indicator].get("signal", "neutral")
                if signal == "buy":
                    buy_signals += 1
                elif signal == "sell":
                    sell_signals += 1
                total_signals += 1

        # EMA crossover signal
        if "ema" in results:
            ema_20 = results["ema"].get("latest_20")
            ema_50 = results["ema"].get("latest_50")
            if ema_20 is not None and ema_50 is not None:
                total_signals += 1
                if ema_20 > ema_50:
                    buy_signals += 1
                elif ema_20 < ema_50:
                    sell_signals += 1

        # Calculate confidence and final signal
        if total_signals == 0:
            return {"signal": "neutral", "confidence": 0.0}

        signal_strength = buy_signals - sell_signals
        confidence = abs(signal_strength) / total_signals

        if signal_strength >= 2:
            final_signal = "strong_buy"
        elif signal_strength == 1:
            final_signal = "buy"
        elif signal_strength <= -2:
            final_signal = "strong_sell"
        elif signal_strength == -1:
            final_signal = "sell"
        else:
            final_signal = "neutral"

        return {
            "signal": final_signal,
            "confidence": round(confidence, 2),
            "buy_count": buy_signals,
            "sell_count": sell_signals,
            "total_indicators": total_signals,
        }
