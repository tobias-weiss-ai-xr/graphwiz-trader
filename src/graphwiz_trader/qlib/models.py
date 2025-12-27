"""Machine learning models for trading signal generation using Qlib."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import pickle
from loguru import logger

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Signal generation will be limited.")

from .config import QlibConfig
from .features import AlphaFeatureExtractor


class QlibSignalGenerator:
    """
    Generate trading signals using machine learning models.

    This class implements a signal generator that uses LightGBM models
    trained on Alpha158 features to predict price movements and generate
    trading signals.
    """

    def __init__(
        self,
        config: Optional[QlibConfig] = None,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize the signal generator.

        Args:
            config: Qlib configuration
            model_path: Path to saved model file
        """
        self.config = config or QlibConfig()
        self.model_path = model_path
        self.model = None
        self.feature_extractor = AlphaFeatureExtractor(config=config)

        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not installed. Cannot train/predict models.")
            return

        # Load model if path provided
        if model_path and model_path.exists():
            self.load_model(model_path)

    def prepare_labels(
        self,
        df: pd.DataFrame,
        forward_return_period: int = 5,
    ) -> pd.Series:
        """
        Prepare labels for supervised learning.

        Args:
            df: DataFrame with price data
            forward_return_period: Period to calculate forward returns

        Returns:
            Series with forward returns as labels
        """
        # Calculate forward returns
        forward_returns = df['close'].pct_change(forward_return_period).shift(-forward_return_period)

        # Convert to binary classification (1 = up, 0 = down)
        labels = (forward_returns > 0).astype(int)

        logger.info(f"Prepared labels: {labels.sum()} buy signals out of {len(labels)}")

        return labels

    def train(
        self,
        df: pd.DataFrame,
        symbol: str,
        validation_split: float = 0.2,
        **model_params,
    ) -> Dict[str, Any]:
        """
        Train a LightGBM model on historical data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            validation_split: Fraction of data to use for validation
            **model_params: Additional parameters for LightGBM

        Returns:
            Dictionary with training results
        """
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM not available. Cannot train model.")

        logger.info(f"Training model for {symbol}...")

        # Extract features
        features_df = self.feature_extractor.prepare_features_for_training(df, symbol)

        if len(features_df) == 0:
            raise ValueError("No features extracted. Check input data.")

        # Prepare labels
        labels = self.prepare_labels(df)

        # Align features and labels
        aligned_data = pd.concat([features_df, labels.rename('label')], axis=1).dropna()
        X = aligned_data.drop('label', axis=1)
        y = aligned_data['label']

        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Default model parameters
        default_params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
        }
        default_params.update(model_params)

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        logger.info("Starting LightGBM training...")
        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )

        # Evaluate model
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_accuracy = np.mean((train_pred > 0.5).astype(int) == y_train)
        val_accuracy = np.mean((val_pred > 0.5).astype(int) == y_val)

        results = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'num_features': len(X.columns),
            'feature_importance': dict(zip(
                X.columns,
                self.model.feature_importance(importance_type='gain')
            )),
        }

        logger.info(f"Training complete. Train accuracy: {train_accuracy:.4f}, "
                   f"Val accuracy: {val_accuracy:.4f}")

        return results

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Generate trading signals using trained model.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            threshold: Probability threshold for buy signal

        Returns:
            DataFrame with predictions and signals
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first or load a trained model.")

        # Extract features
        features_df = self.feature_extractor.prepare_features_for_training(df, symbol)

        if len(features_df) == 0:
            logger.warning("No features extracted. Cannot generate predictions.")
            return pd.DataFrame()

        # Generate predictions
        predictions = self.model.predict(features_df)

        # Create signals DataFrame
        signals = pd.DataFrame(index=features_df.index)
        signals['probability'] = predictions
        signals['signal'] = (predictions > threshold).astype(int)
        signals['signal_type'] = signals['signal'].map({1: 'BUY', 0: 'HOLD/SELL'})

        logger.info(f"Generated {len(signals)} predictions for {symbol}")

        return signals

    def predict_latest(
        self,
        df: pd.DataFrame,
        symbol: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate prediction for the latest data point.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            threshold: Probability threshold for buy signal

        Returns:
            Dictionary with latest prediction
        """
        signals = self.predict(df, symbol, threshold)

        if len(signals) == 0:
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'probability': 0.0,
                'signal': 'HOLD',
                'error': 'No predictions generated',
            }

        # Get the latest prediction
        latest = signals.iloc[-1]

        return {
            'symbol': symbol,
            'timestamp': latest.name,
            'probability': float(latest['probability']),
            'signal': latest['signal_type'],
            'confidence': 'HIGH' if abs(latest['probability'] - 0.5) > 0.3 else 'MEDIUM' if abs(latest['probability'] - 0.5) > 0.1 else 'LOW',
        }

    def save_model(self, path: Path):
        """
        Save trained model to file.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        """
        Load trained model from file.

        Args:
            path: Path to load model from
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df


class EnsembleSignalGenerator:
    """
    Ensemble of multiple signal generators for robust predictions.

    Combines predictions from multiple models or techniques to generate
    more reliable trading signals.
    """

    def __init__(
        self,
        generators: List[QlibSignalGenerator],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble generator.

        Args:
            generators: List of signal generators
            weights: Optional weights for combining predictions
        """
        self.generators = generators

        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(generators)] * len(generators)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        if len(self.weights) != len(self.generators):
            raise ValueError("Number of weights must match number of generators")

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            threshold: Probability threshold for buy signal

        Returns:
            DataFrame with ensemble predictions
        """
        all_predictions = []

        # Get predictions from each generator
        for generator in self.generators:
            try:
                pred = generator.predict(df, symbol, threshold)
                all_predictions.append(pred['probability'])
            except Exception as e:
                logger.warning(f"Generator failed: {e}")
                continue

        if not all_predictions:
            logger.error("No generators produced predictions")
            return pd.DataFrame()

        # Combine predictions using weighted average
        combined_prob = np.average(all_predictions, axis=0, weights=self.weights)

        # Create signals DataFrame
        signals = pd.DataFrame(index=all_predictions[0].index)
        signals['probability'] = combined_prob
        signals['signal'] = (combined_prob > threshold).astype(int)
        signals['signal_type'] = signals['signal'].map({1: 'BUY', 0: 'HOLD/SELL'})

        logger.info(f"Generated ensemble predictions for {symbol}")

        return signals


def create_signal_generator(
    config: Optional[QlibConfig] = None,
    model_path: Optional[Path] = None,
) -> QlibSignalGenerator:
    """
    Convenience function to create a signal generator.

    Args:
        config: Qlib configuration
        model_path: Path to saved model

    Returns:
        QlibSignalGenerator instance
    """
    return QlibSignalGenerator(config=config, model_path=model_path)
