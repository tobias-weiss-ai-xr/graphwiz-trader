"""Advanced trading strategies based on 2025 research.

This module implements cutting-edge strategies from latest academic and industry research:

1. Advanced Mean Reversion - Based on Stoic.ai and OKX research
2. Pairs Trading with PCA - Based on 2025 statistical arbitrage research
3. Momentum with Volatility Filtering - Based on systematic crypto trading research
4. Multi-Factor Strategy - Based on multi-factor ML research

References:
- https://stoic.ai/blog/mean-reversion-trading-how-i-profit-from-crypto-market-overreactions/
- https://www.gate.io/crypto-wiki/article/a-complete-guide-to-statistical-arbitrage-strategies-in-cryptocurrency-trading-20251208
- https://medium.com/@briplotnik/systematic-crypto-trading-strategies-momentum-mean-reversion-volatility-filtering-8d7da06d60ed
- https://dl.acm.org/doi/10.1145/3766918.3766922
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
from enum import Enum

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    SCIKIT_AVAILABLE = True
except ImportError:
    SCIKIT_AVAILABLE = False
    logger.warning("Scikit-learn not available. Some strategies will be limited.")

from ..trading.engine import TradingEngine


class MeanReversionType(Enum):
    """Types of mean reversion strategies."""

    BOLLINGER = "bollinger"  # Price deviation from moving average
    RSI = "rsi"  # RSI-based reversion
    ZSCORE = "zscore"  # Statistical z-score
    ENVELOPE = "envelope"  # Moving average envelope
    MULTI = "multi"  # Combined signals


class AdvancedMeanReversionStrategy:
    """
    Advanced mean reversion strategy based on 2025 research.

    Sources:
    - https://stoic.ai/blog/mean-reversion-trading-how-i-profit-from-crypto-market-overreactions/
    - https://www.okx.com/zhhans-eu/learn/mean-reversion-strategies-crypto-futures
    - https://www.robuxio.com/algorithmic-crypto-trading-v-mean-reversion/
    """

    def __init__(
        self,
        reversion_type: MeanReversionType = MeanReversionType.MULTI,
        entry_threshold: float = 2.0,  # Standard deviations
        exit_threshold: float = 0.5,
        lookback_period: int = 20,
        volatility_filter: bool = True,
        volatility_threshold: float = 0.08,  # 8% volatility threshold
    ):
        """
        Initialize advanced mean reversion strategy.

        Args:
            reversion_type: Type of mean reversion to use
            entry_threshold: Threshold for entering position (std devs)
            exit_threshold: Threshold for exiting position (std devs)
            lookback_period: Period for indicators
            volatility_filter: Whether to filter high volatility
            volatility_threshold: Volatility threshold above which to pause trading
        """
        self.reversion_type = reversion_type
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback_period = lookback_period
        self.volatility_filter = volatility_filter
        self.volatility_threshold = volatility_threshold

        logger.info(f"Advanced Mean Reversal Strategy initialized: {reversion_type.value}")

    def generate_signals(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate mean reversion signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=df.index)

        # Calculate indicators based on type
        if self.reversion_type == MeanReversionType.BOLLINGER:
            signals = self._bollinger_reversion(df)
        elif self.reversion_type == MeanReversionType.RSI:
            signals = self._rsi_reversion(df)
        elif self.reversion_type == MeanReversionType.ZSCORE:
            signals = self._zscore_reversion(df)
        elif self.reversion_type == MeanReversionType.ENVELOPE:
            signals = self._envelope_reversion(df)
        else:  # MULTI
            signals = self._multi_reversion(df)

        # Apply volatility filter if enabled
        if self.volatility_filter:
            volatility = df["close"].pct_change().rolling(self.lookback_period).std()
            volatility_mask = volatility < self.volatility_threshold
            signals["signal"] = signals["signal"] & volatility_mask

        return signals

    def _bollinger_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands mean reversion."""
        signals = pd.DataFrame(index=df.index)

        # Calculate Bollinger Bands
        sma = df["close"].rolling(window=self.lookback_period).mean()
        std = df["close"].rolling(window=self.lookback_period).std()

        upper_band = sma + (self.entry_threshold * std)
        lower_band = sma - (self.entry_threshold * std)
        exit_band = sma + (self.exit_threshold * std)

        # Generate signals
        # Buy when price is below lower band (oversold)
        signals["signal"] = (df["close"] < lower_band).astype(int)

        # Exit when price crosses back to mean
        signals["exit_signal"] = (df["close"] > sma).astype(int)

        # Store metrics
        signals["upper_band"] = upper_band
        signals["lower_band"] = lower_band
        signals["sma"] = sma
        signals["position_size"] = (
            lower_band - df["close"]
        ) / lower_band  # More oversold = larger position

        return signals

    def _rsi_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI-based mean reversion."""
        signals = pd.DataFrame(index=df.index)

        # Calculate RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.lookback_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.lookback_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals["rsi"] = rsi

        # Generate signals
        # Buy when RSI < 30 (oversold)
        signals["signal"] = (rsi < 30).astype(int)

        # Exit when RSI > 50
        signals["exit_signal"] = (rsi > 50).astype(int)

        # Position size based on RSI level
        signals["position_size"] = (30 - rsi) / 30  # Lower RSI = larger position
        signals["position_size"] = signals["position_size"].clip(0, 1)

        return signals

    def _zscore_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score based mean reversion."""
        signals = pd.DataFrame(index=df.index)

        # Calculate z-score
        mean = df["close"].rolling(window=self.lookback_period).mean()
        std = df["close"].rolling(window=self.lookback_period).std()
        zscore = (df["close"] - mean) / std

        signals["zscore"] = zscore

        # Generate signals
        # Buy when z-score < -entry_threshold (very oversold)
        signals["signal"] = (zscore < -self.entry_threshold).astype(int)

        # Exit when z-score crosses 0
        signals["exit_signal"] = (zscore > 0).astype(int)

        # Position size based on z-score magnitude
        signals["position_size"] = (-zscore.abs() / self.entry_threshold).clip(0, 1)

        return signals

    def _envelope_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving average envelope reversion."""
        signals = pd.DataFrame(index=df.index)

        # Calculate envelopes (using high/low moving averages)
        ma_high = df["high"].rolling(window=self.lookback_period).mean()
        ma_low = df["low"].rolling(window=self.lookback_period).mean()
        ma_mid = (ma_high + ma_low) / 2

        # Envelope percentage
        envelope_pct = 0.02  # 2% envelope

        upper_envelope = ma_mid * (1 + envelope_pct)
        lower_envelope = ma_mid * (1 - envelope_pct)

        signals["upper_envelope"] = upper_envelope
        signals["lower_envelope"] = lower_envelope

        # Generate signals
        signals["signal"] = (df["close"] < lower_envelope).astype(int)
        signals["exit_signal"] = (df["close"] > ma_mid).astype(int)

        return signals

    def _multi_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combined mean reversion signals."""
        # Get signals from each method
        bollinger = self._bollinger_reversion(df)
        rsi = self._rsi_reversion(df)
        zscore = self._zscore_reversion(df)

        signals = pd.DataFrame(index=df.index)

        # Combine signals (voting mechanism)
        signals["bollinger_signal"] = bollinger["signal"]
        signals["rsi_signal"] = rsi["signal"]
        signals["zscore_signal"] = zscore["signal"]

        # Buy when at least 2 indicators agree
        signal_sum = bollinger["signal"] + rsi["signal"] + zscore["signal"]
        signals["signal"] = (signal_sum >= 2).astype(int)

        # Exit when price returns to mean
        sma = df["close"].rolling(window=self.lookback_period).mean()
        signals["exit_signal"] = (df["close"] > sma).astype(int)

        # Weighted position size (average of individual sizes)
        signals["position_size"] = (
            bollinger["position_size"] + rsi["position_size"] + zscore["position_size"]
        ) / 3

        return signals


class PairsTradingStrategy:
    """
    Statistical arbitrage pairs trading strategy based on 2025 research.

    Uses PCA for cointegration detection and pair selection.

    Sources:
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5263475
    - https://www.gate.io/crypto-wiki/article/a-complete-guide-to-statistical-arbitrage-strategies-in-cryptocurrency-trading-20251208
    - https://www.wundertrading.com/journal/en/learn/article/crypto-pairs-trading-strategy
    """

    def __init__(
        self,
        lookback_period: int = 30,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.0,
        n_components: int = 5,
    ):
        """
        Initialize pairs trading strategy.

        Args:
            lookback_period: Period for calculating statistics
            entry_zscore: Z-score threshold for opening position
            exit_zscore: Z-score threshold for closing position
            n_components: Number of PCA components for pair selection
        """
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.n_components = n_components

        if not SCIKIT_AVAILABLE:
            logger.warning("Scikit-learn not available. Pairs trading will be limited.")

        logger.info(f"Pairs Trading Strategy initialized: {n_components} PCA components")

    def select_pairs(
        self,
        price_data: Dict[str, pd.Series],
    ) -> List[Tuple[str, str, float]]:
        """
        Select best trading pairs using PCA.

        Args:
            price_data: Dictionary of price series by symbol

        Returns:
            List of (symbol1, symbol2, cointegration_score) tuples
        """
        if not SCIKIT_AVAILABLE:
            logger.error("Scikit-learn required for pair selection")
            return []

        # Create price matrix
        prices_df = pd.DataFrame(price_data)

        # Calculate returns
        returns_df = prices_df.pct_change().dropna()

        # Normalize returns
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_df)

        # Apply PCA
        pca = PCA(n_components=self.n_components)
        pca.fit(scaled_returns)

        # Find pairs based on PCA loadings
        pairs = []
        for component in range(self.n_components):
            # Get symbols with highest absolute loadings for this component
            loadings = abs(pca.components_[component])
            top_indices = loadings.argsort()[-4:]  # Top 4 symbols

            for i in range(len(top_indices)):
                for j in range(i + 1, len(top_indices)):
                    symbol1 = prices_df.columns[top_indices[i]]
                    symbol2 = prices_df.columns[top_indices[j]]

                    # Calculate correlation
                    corr = returns_df[symbol1].corr(returns_df[symbol2])

                    if abs(corr) > 0.7:  # High correlation
                        pairs.append((symbol1, symbol2, abs(corr)))

        # Sort by correlation and return top pairs
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

        logger.info(f"Selected {len(pairs)} trading pairs")
        for pair in pairs[:5]:
            logger.info(f"  {pair[0]} - {pair[1]}: {pair[2]:.4f}")

        return pairs

    def generate_signals(
        self,
        pair: Tuple[str, str],
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate trading signals for a pair.

        Args:
            pair: Tuple of (symbol1, symbol2)
            price_data: DataFrame with both symbols' prices

        Returns:
            DataFrame with signals and position sizing
        """
        symbol1, symbol2 = pair

        signals = pd.DataFrame(index=price_data.index)

        # Calculate spread (price1 - hedge_ratio * price2)
        hedge_ratio = self._calculate_hedge_ratio(price_data[symbol1], price_data[symbol2])
        spread = price_data[symbol1] - (hedge_ratio * price_data[symbol2])

        # Calculate z-score of spread
        spread_mean = spread.rolling(window=self.lookback_period).mean()
        spread_std = spread.rolling(window=self.lookback_period).std()
        spread_zscore = (spread - spread_mean) / spread_std

        signals["spread"] = spread
        signals["spread_zscore"] = spread_zscore
        signals["hedge_ratio"] = hedge_ratio

        # Generate signals
        # Long the spread when it's low (short when high)
        signals["signal"] = (spread_zscore < -self.entry_zscore).astype(int)  # Long spread
        signals["exit_signal"] = (abs(spread_zscore) < self.exit_zscore).astype(int)

        # Position sizing based on z-score confidence
        signals["position_size"] = (abs(spread_zscore) / self.entry_zscore).clip(0, 2)

        return signals

    def _calculate_hedge_ratio(
        self,
        price1: pd.Series,
        price2: pd.Series,
    ) -> float:
        """
        Calculate optimal hedge ratio using OLS regression.

        Args:
            price1: Price series of first asset
            price2: Price series of second asset

        Returns:
            Hedge ratio
        """
        if not SCIKIT_AVAILABLE:
            # Simple ratio
            return price1.mean() / price2.mean()

        # Use Ridge regression for robust hedge ratio
        model = Ridge(alpha=1.0)

        # Remove NaN values
        df = pd.DataFrame({"y": price1, "x": price2}).dropna()

        if len(df) < 30:
            return price1.mean() / price2.mean()

        X = df[["x"]].values
        y = df["y"].values

        model.fit(X, y)
        return model.coef_[0]


class MomentumVolatilityFilteringStrategy:
    """
    Momentum strategy with volatility filtering based on 2025 research.

    Sources:
    - https://medium.com/@briplotnik/systematic-crypto-trading-strategies-momentum-mean-reversion-volatility-filtering-8d7da06d60ed
    """

    def __init__(
        self,
        momentum_period: int = 50,
        volatility_period: int = 20,
        volatility_threshold: float = 0.06,  # 6% threshold
        momentum_threshold: float = 0.02,  # 2% momentum threshold
    ):
        """
        Initialize momentum with volatility filtering strategy.

        Args:
            momentum_period: Period for momentum calculation
            volatility_period: Period for volatility calculation
            volatility_threshold: Volatility threshold above which to pause
            momentum_threshold: Minimum momentum to generate signal
        """
        self.momentum_period = momentum_period
        self.volatility_period = volatility_period
        self.volatility_threshold = volatility_threshold
        self.momentum_threshold = momentum_threshold

        logger.info(f"Momentum with Volatility Filtering initialized")

    def generate_signals(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate momentum signals with volatility filtering.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=df.index)

        # Calculate momentum
        momentum = df["close"].pct_change(self.momentum_period)

        # Calculate volatility
        volatility = df["close"].pct_change().rolling(self.volatility_period).std()

        signals["momentum"] = momentum
        signals["volatility"] = volatility

        # Generate signals only when volatility is low
        low_volatility = volatility < self.volatility_threshold

        # Buy when momentum is positive and volatility is low
        signals["signal"] = (momentum > self.momentum_threshold) & low_volatility
        signals["signal"] = signals["signal"].astype(int)

        # Position size based on momentum strength
        signals["position_size"] = (momentum / volatility).clip(0, 2)

        # Exit signal when momentum turns negative
        signals["exit_signal"] = (momentum < 0).astype(int)

        return signals


class MultiFactorStrategy:
    """
    Multi-factor strategy combining traditional and on-chain factors.

    Based on 2025 research:
    https://dl.acm.org/doi/10.1145/3766918.3766922

    Integrates:
    - Traditional technical indicators (RSI, MACD, etc.)
    - Market factors (momentum, value, size, volatility)
    - On-chain metrics (if available)
    """

    def __init__(
        self,
        factors: List[str] = None,
        factor_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-factor strategy.

        Args:
            factors: List of factor names to use
            factor_weights: Weights for each factor
        """
        self.factors = factors or [
            "momentum",
            "mean_reversion",
            "volatility",
            "volume",
            "on_chain_activity",
        ]

        self.factor_weights = factor_weights or {
            "momentum": 0.3,
            "mean_reversion": 0.2,
            "volatility": 0.2,
            "volume": 0.15,
            "on_chain_activity": 0.15,
        }

        logger.info(f"Multi-Factor Strategy initialized with {len(self.factors)} factors")

    def calculate_factors(
        self,
        df: pd.DataFrame,
        on_chain_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate all factors for the given data.

        Args:
            df: DataFrame with OHLCV data
            on_chain_data: Optional DataFrame with on-chain metrics

        Returns:
            DataFrame with all factors calculated
        """
        factors_df = pd.DataFrame(index=df.index)

        # Momentum factor
        factors_df["momentum"] = df["close"].pct_change(50)

        # Mean reversion factor (inverse of momentum)
        factors_df["mean_reversion"] = -factors_df["momentum"]

        # Volatility factor
        factors_df["volatility"] = df["close"].pct_change().rolling(20).std()

        # Volume factor
        factors_df["volume"] = df["volume"].rolling(20).mean() / df["volume"].rolling(50).mean()

        # On-chain activity (if provided)
        if on_chain_data is not None and not on_chain_data.empty:
            # Normalize on-chain data
            factors_df["on_chain_activity"] = on_chain_data["activity_score"]
        else:
            # Use volume as proxy for on-chain activity
            factors_df["on_chain_activity"] = factors_df["volume"]

        # Normalize all factors to 0-1 range
        for factor in self.factors:
            if factor in factors_df.columns:
                min_val = factors_df[factor].min()
                max_val = factors_df[factor].max()
                if max_val > min_val:
                    factors_df[f"{factor}_norm"] = (factors_df[factor] - min_val) / (
                        max_val - min_val
                    )
                else:
                    factors_df[f"{factor}_norm"] = 0

        return factors_df

    def generate_signals(
        self,
        df: pd.DataFrame,
        on_chain_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate signals based on multi-factor model.

        Args:
            df: DataFrame with OHLCV data
            on_chain_data: Optional on-chain metrics

        Returns:
            DataFrame with combined signals
        """
        # Calculate all factors
        factors_df = self.calculate_factors(df, on_chain_data)

        signals = pd.DataFrame(index=df.index)

        # Calculate weighted score
        weighted_score = pd.Series(0.0, index=df.index)

        for factor, weight in self.factor_weights.items():
            col_name = f"{factor}_norm"
            if col_name in factors_df.columns:
                weighted_score += factors_df[col_name] * weight

        signals["factor_score"] = weighted_score

        # Generate signals based on combined score
        # Buy when score > 0.6 (top 40% of scores)
        signals["signal"] = (weighted_score > 0.6).astype(int)

        # Position size based on score strength
        signals["position_size"] = weighted_score.clip(0, 2)

        # Exit signal when score drops below 0.4
        signals["exit_signal"] = (weighted_score < 0.4).astype(int)

        return signals


class ConfidenceThresholdStrategy:
    """
    Confidence-threshold framework for trading decisions.

    Based on 2025 research:
    https://www.mdpi.com/2076-3417/15/20/11145

    Only trades when model confidence exceeds threshold.
    """

    def __init__(
        self,
        base_threshold: float = 0.6,
        aggressive_threshold: float = 0.7,
        conservative_threshold: float = 0.5,
        mode: str = "normal",  # 'aggressive', 'normal', 'conservative'
    ):
        """
        Initialize confidence threshold strategy.

        Args:
            base_threshold: Default confidence threshold
            aggressive_threshold: Lower threshold for aggressive trading
            conservative_threshold: Higher threshold for conservative trading
            mode: Trading mode
        """
        self.mode = mode

        if mode == "aggressive":
            self.threshold = aggressive_threshold
        elif mode == "conservative":
            self.threshold = conservative_threshold
        else:
            self.threshold = base_threshold

        logger.info(f"Confidence Threshold Strategy: mode={mode}, threshold={self.threshold}")

    def adjust_threshold(
        self,
        recent_performance: pd.Series,
        market_volatility: float,
    ) -> float:
        """
        Dynamically adjust threshold based on conditions.

        Args:
            recent_performance: Recent trading returns
            market_volatility: Current market volatility

        Returns:
            Adjusted threshold
        """
        threshold = self.threshold

        # Adjust based on recent performance
        if len(recent_performance) >= 10:
            sharpe = (
                recent_performance.mean() / recent_performance.std()
                if recent_performance.std() > 0
                else 0
            )

            # Lower threshold if Sharpe is good
            if sharpe > 1.5:
                threshold *= 0.9  # 10% lower
            elif sharpe < 0.5:
                threshold *= 1.1  # 10% higher

        # Adjust based on volatility
        if market_volatility > 0.08:  # High volatility
            threshold *= 1.2  # Raise threshold
        elif market_volatility < 0.02:  # Low volatility
            threshold *= 0.9  # Lower threshold

        return max(0.4, min(threshold, 0.8))


def create_advanced_strategy(
    strategy_type: str = "mean_reversion",
    **params,
) -> Any:
    """
    Convenience function to create advanced strategies.

    Args:
        strategy_type: Type of strategy to create
        **params: Additional parameters for the strategy

    Returns:
        Strategy instance
    """
    if strategy_type == "mean_reversion":
        return AdvancedMeanReversionStrategy(**params)
    elif strategy_type == "pairs_trading":
        return PairsTradingStrategy(**params)
    elif strategy_type == "momentum_volatility":
        return MomentumVolatilityFilteringStrategy(**params)
    elif strategy_type == "multi_factor":
        return MultiFactorStrategy(**params)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
