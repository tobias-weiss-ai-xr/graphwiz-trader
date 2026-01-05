"""Feature extraction using Qlib Alpha158 and custom features."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.data.handler import Alpha158

    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("Qlib not available. Some features will be disabled.")

from .config import QlibConfig


class AlphaFeatureExtractor:
    """
    Extract Alpha158 features from market data using Qlib.

    Alpha158 is a set of 158 engineered features commonly used in
    quantitative finance, including price momentum, volume patterns,
    volatility measures, and technical indicators.
    """

    def __init__(
        self,
        config: Optional[QlibConfig] = None,
        enable_graph_features: bool = True,
    ):
        """
        Initialize the feature extractor.

        Args:
            config: Qlib configuration
            enable_graph_features: Whether to enable graph-based features
        """
        self.config = config or QlibConfig()
        self.enable_graph_features = enable_graph_features
        self.alpha_handler = None

        if not QLIB_AVAILABLE:
            logger.warning("Qlib not installed. Feature extraction will use fallback methods.")
            return

        # Initialize Qlib if needed
        try:
            qlib.init(provider=self.config.provider, region=self.config.region)
            logger.info("Qlib initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Qlib: {e}")
            return

    def extract_alpha158(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Extract Alpha158 features from market data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier

        Returns:
            DataFrame with Alpha158 features
        """
        if not QLIB_AVAILABLE:
            return self._extract_fallback_features(df)

        try:
            # Prepare data for Qlib
            # Qlib expects data in a specific format with multi-level columns
            df_prep = self._prepare_qlib_data(df, symbol)

            # Create Alpha158 handler
            if self.alpha_handler is None:
                self.alpha_handler = Alpha158()

            # Extract features
            features_df = self.alpha_handler.fetch(df_prep)

            logger.info(f"Extracted {len(features_df.columns)} features for {symbol}")

            return features_df

        except Exception as e:
            logger.error(f"Error extracting Alpha158 features: {e}")
            logger.info("Falling back to basic features")
            return self._extract_fallback_features(df)

    def _prepare_qlib_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Prepare data in Qlib format.

        Args:
            df: Input DataFrame
            symbol: Symbol identifier

        Returns:
            DataFrame in Qlib format
        """
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        df_prep = df[required_cols].copy()

        # Rename to Qlib format
        df_prep.columns = [col.upper() for col in df_prep.columns]

        # Add instrument column
        df_prep["instrument"] = symbol

        return df_prep

    def _extract_fallback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic features when Qlib is not available.

        This implements a subset of Alpha158 features manually.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with basic features
        """
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features["returns"] = df["close"].pct_change()
        features["high_low_pct"] = (df["high"] - df["low"]) / df["close"]
        features["close_open_pct"] = (df["close"] - df["open"]) / df["open"]

        # Moving averages
        for window in [5, 10, 20, 30, 60]:
            features[f"ma_{window}"] = df["close"].rolling(window=window).mean()
            features[f"close_ma_{window}_pct"] = df["close"] / features[f"ma_{window}"] - 1

        # Momentum
        for window in [5, 10, 20]:
            features[f"momentum_{window}"] = df["close"].pct_change(window)

        # Volatility
        for window in [5, 10, 20]:
            features[f"volatility_{window}"] = features["returns"].rolling(window=window).std()

        # Volume features
        features["volume_ma_5"] = df["volume"].rolling(window=5).mean()
        features["volume_ratio"] = df["volume"] / features["volume_ma_5"]
        features["volume_change"] = df["volume"].pct_change()

        # Price position in range
        features["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        # Bollinger Bands
        for window in [20]:
            ma = df["close"].rolling(window=window).mean()
            std = df["close"].rolling(window=window).std()
            features[f"bb_upper_{window}"] = ma + 2 * std
            features[f"bb_lower_{window}"] = ma - 2 * std
            features[f"bb_position_{window}"] = (df["close"] - features[f"bb_lower_{window}"]) / (
                features[f"bb_upper_{window}"] - features[f"bb_lower_{window}"] + 1e-10
            )

        # RSI (Relative Strength Index)
        features["rsi_14"] = self._calculate_rsi(df["close"], 14)

        # MACD components
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        features["macd"] = exp1 - exp2
        features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
        features["macd_hist"] = features["macd"] - features["macd_signal"]

        logger.info(f"Extracted {len(features.columns)} fallback features")

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def extract_graph_features(
        self,
        neo4j_client,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Extract graph-based features from Neo4j knowledge graph.

        This extracts relationship patterns and network metrics that
        can be combined with Alpha158 features for enhanced prediction.

        Args:
            neo4j_client: Neo4j client instance
            symbol: Symbol identifier
            timestamp: Current timestamp

        Returns:
            Dictionary of graph features
        """
        if not self.enable_graph_features:
            return {}

        features = {}

        try:
            # Query Neo4j for graph features
            query = """
            MATCH (s:Symbol {name: $symbol})
            OPTIONAL MATCH (s)-[r:CORRELATES_WITH]-(other:Symbol)
            WHERE ABS(r.correlation) > 0.7
            WITH s, COUNT(DISTINCT other) as highly_correlated_count
            RETURN highly_correlated_count
            """

            result = neo4j_client.run(query, symbol=symbol)
            single_result = result.single()
            if single_result:
                features["highly_correlated_symbols"] = single_result["highly_correlated_count"]

            # Query for trading patterns
            pattern_query = """
            MATCH (s:Symbol {name: $symbol})<-[:TRADED]-(t:Trade)
            WHERE t.timestamp > datetime($since)
            WITH s, COUNT(t) as recent_trades,
                 AVG(t.profit_loss) as avg_profit_loss,
                 SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(t) as win_rate
            RETURN recent_trades, avg_profit_loss, win_rate
            """

            # Calculate 'since' timestamp (e.g., 7 days ago)
            since = (timestamp - pd.Timedelta(days=7)).isoformat()

            result = neo4j_client.run(pattern_query, symbol=symbol, since=since)
            single_result = result.single()
            if single_result:
                features["recent_trades_7d"] = single_result["recent_trades"]
                features["avg_profit_loss_7d"] = single_result["avg_profit_loss"] or 0
                features["win_rate_7d"] = single_result["win_rate"] or 0

            logger.debug(f"Extracted {len(features)} graph features for {symbol}")

        except Exception as e:
            logger.error(f"Error extracting graph features: {e}")

        return features

    def combine_features(
        self,
        alpha_features: pd.DataFrame,
        graph_features: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Combine Alpha158 features with graph features.

        Args:
            alpha_features: DataFrame with Alpha158 features
            graph_features: Dictionary of graph features

        Returns:
            Combined DataFrame
        """
        if not graph_features:
            return alpha_features

        # Add graph features as columns
        for feature_name, feature_value in graph_features.items():
            alpha_features[feature_name] = feature_value

        logger.info(f"Combined features: {len(alpha_features.columns)} total")

        return alpha_features

    def prepare_features_for_training(
        self,
        df: pd.DataFrame,
        symbol: str,
        neo4j_client=None,
    ) -> pd.DataFrame:
        """
        Complete feature extraction pipeline for training.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            neo4j_client: Optional Neo4j client for graph features

        Returns:
            DataFrame with all features
        """
        # Extract Alpha158 features
        alpha_features = self.extract_alpha158(df, symbol)

        # Extract graph features if available
        graph_features = {}
        if neo4j_client and self.enable_graph_features:
            # Get the latest timestamp
            latest_timestamp = df.index.max()
            graph_features = self.extract_graph_features(
                neo4j_client,
                symbol,
                latest_timestamp,
            )

        # Combine features
        combined_features = self.combine_features(alpha_features, graph_features)

        # Remove NaN values
        combined_features.dropna(inplace=True)

        return combined_features


# Convenience function for quick feature extraction
def extract_features(
    df: pd.DataFrame,
    symbol: str,
    config: Optional[QlibConfig] = None,
    use_graph_features: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to extract features.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol identifier
        config: Qlib configuration
        use_graph_features: Whether to use graph features

    Returns:
        DataFrame with features
    """
    extractor = AlphaFeatureExtractor(
        config=config,
        enable_graph_features=use_graph_features,
    )
    return extractor.prepare_features_for_training(df, symbol)
