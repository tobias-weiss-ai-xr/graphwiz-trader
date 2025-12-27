"""Hybrid ML models combining Qlib Alpha158 features with Neo4j graph features.

This module creates unique hybrid models that leverage:
- Alpha158: 158+ engineered time-series features
- Graph features: Network, correlation, trading pattern, regime features

This combination provides predictive power that neither approach can achieve alone.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import pickle
from loguru import logger

try:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    LIGHTGBM_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    logger.warning("LightGBM or sklearn not available. Hybrid models will be limited.")

from .config import QlibConfig
from .features import AlphaFeatureExtractor
from .graph_features import GraphFeatureExtractor
from .models import QlibSignalGenerator


class HybridFeatureGenerator:
    """
    Combines Alpha158 and graph features into unified feature set.

    This is the core innovation - fusing traditional quantitative features
    with knowledge graph-derived features.
    """

    def __init__(
        self,
        alpha_extractor: Optional[AlphaFeatureExtractor] = None,
        graph_extractor: Optional[GraphFeatureExtractor] = None,
    ):
        """
        Initialize hybrid feature generator.

        Args:
            alpha_extractor: Alpha158 feature extractor
            graph_extractor: Graph feature extractor
        """
        self.alpha_extractor = alpha_extractor or AlphaFeatureExtractor()
        self.graph_extractor = graph_extractor

        # Feature names for tracking
        self.alpha_feature_names = []
        self.graph_feature_names = []

        logger.info("Hybrid feature generator initialized")

    def generate_hybrid_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        neo4j_client=None,
    ) -> pd.DataFrame:
        """
        Generate combined Alpha158 + Graph features.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            neo4j_client: Optional Neo4j client

        Returns:
            DataFrame with combined features
        """
        # Extract Alpha158 features
        alpha_features = self.alpha_extractor.prepare_features_for_training(
            df, symbol, neo4j_client=None
        )

        if len(alpha_features) == 0:
            logger.warning("No Alpha158 features extracted")
            return pd.DataFrame()

        # Extract graph features if available
        graph_features = {}
        if self.graph_extractor:
            try:
                graph_features = self.graph_extractor.extract_all_features(symbol)
            except Exception as e:
                logger.warning(f"Failed to extract graph features: {e}")

        # Combine features
        hybrid_features = alpha_features.copy()

        # Add graph features as columns (broadcast to all rows)
        for feature_name, feature_value in graph_features.items():
            hybrid_features[feature_name] = feature_value

        # Track feature names
        self.alpha_feature_names = [col for col in alpha_features.columns]
        self.graph_feature_names = list(graph_features.keys())

        logger.info(f"Generated {len(hybrid_features.columns)} hybrid features "
                   f"({len(self.alpha_feature_names)} Alpha + {len(self.graph_feature_names)} Graph)")

        return hybrid_features

    def get_feature_importance_by_type(
        self,
        model,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate feature importance by type (Alpha vs Graph).

        Args:
            model: Trained model with feature_importances_ attribute

        Returns:
            Tuple of (alpha_importance, graph_importance) DataFrames
        """
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'feature_importance'):
            logger.warning("Model doesn't have feature importances")
            return pd.DataFrame(), pd.DataFrame()

        # Get importances
        if hasattr(model, 'feature_importance'):
            importances = model.feature_importance(importance_type='gain')
            feature_names = model.feature_name()
        else:
            importances = model.feature_importances_
            feature_names = self.alpha_feature_names + self.graph_feature_names

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
        }).sort_values('importance', ascending=False)

        # Split by type
        alpha_importance = importance_df[
            importance_df['feature'].isin(self.alpha_feature_names)
        ].copy()

        graph_importance = importance_df[
            importance_df['feature'].isin(self.graph_feature_names)
        ].copy()

        return alpha_importance, graph_importance


class HybridSignalGenerator(QlibSignalGenerator):
    """
    Enhanced signal generator using hybrid Alpha158 + Graph features.

    Inherits from QlibSignalGenerator and adds graph feature extraction.
    """

    def __init__(
        self,
        config: Optional[QlibConfig] = None,
        model_path: Optional[Path] = None,
        graph_extractor: Optional[GraphFeatureExtractor] = None,
    ):
        """
        Initialize hybrid signal generator.

        Args:
            config: Qlib configuration
            model_path: Path to saved model
            graph_extractor: Graph feature extractor
        """
        super().__init__(config=config, model_path=model_path)

        self.graph_extractor = graph_extractor
        self.hybrid_generator = HybridFeatureGenerator(
            alpha_extractor=self.feature_extractor,
            graph_extractor=graph_extractor,
        )

        self.feature_type = "hybrid" if graph_extractor else "alpha_only"

        logger.info(f"Hybrid signal generator initialized (mode: {self.feature_type})")

    def train(
        self,
        df: pd.DataFrame,
        symbol: str,
        validation_split: float = 0.2,
        **model_params,
    ) -> Dict[str, Any]:
        """
        Train hybrid model on historical data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier
            validation_split: Validation split fraction
            **model_params: Additional model parameters

        Returns:
            Training results dictionary
        """
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM not available. Cannot train hybrid model.")

        logger.info(f"Training hybrid model for {symbol}...")

        # Generate hybrid features
        features_df = self.hybrid_generator.generate_hybrid_features(
            df, symbol, self.graph_extractor
        )

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
        logger.info("Starting LightGBM training with hybrid features...")
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

        # Calculate feature importance by type
        alpha_importance, graph_importance = self.hybrid_generator.get_feature_importance_by_type(
            self.model
        )

        results = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'num_features': len(X.columns),
            'num_alpha_features': len(self.hybrid_generator.alpha_feature_names),
            'num_graph_features': len(self.hybrid_generator.graph_feature_names),
            'alpha_feature_importance': alpha_importance.to_dict('records') if len(alpha_importance) > 0 else [],
            'graph_feature_importance': graph_importance.to_dict('records') if len(graph_importance) > 0 else [],
            'feature_importance': dict(zip(
                X.columns,
                self.model.feature_importance(importance_type='gain')
            )),
        }

        logger.info(f"Hybrid model training complete:")
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Val accuracy: {val_accuracy:.4f}")
        logger.info(f"  Alpha features: {results['num_alpha_features']}")
        logger.info(f"  Graph features: {results['num_graph_features']}")

        return results

    def compare_with_baseline(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Compare hybrid model against Alpha-only baseline.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol identifier

        Returns:
            Comparison results
        """
        logger.info("Comparing Hybrid vs Alpha-only models...")

        # Train hybrid model
        logger.info("Training hybrid model...")
        hybrid_results = self.train(df, symbol, validation_split=0.2)
        hybrid_accuracy = hybrid_results['val_accuracy']

        # Train baseline (Alpha-only) model
        logger.info("Training Alpha-only baseline...")
        baseline_generator = QlibSignalGenerator(config=self.config)
        baseline_results = baseline_generator.train(df, symbol, validation_split=0.2)
        baseline_accuracy = baseline_results['val_accuracy']

        # Calculate improvement
        improvement = (hybrid_accuracy - baseline_accuracy) / baseline_accuracy * 100

        comparison = {
            'baseline_accuracy': baseline_accuracy,
            'hybrid_accuracy': hybrid_accuracy,
            'accuracy_improvement_pct': improvement,
            'baseline_features': baseline_results['num_features'],
            'hybrid_features': hybrid_results['num_features'],
            'graph_features_added': hybrid_results['num_graph_features'],
            'hygraph_better': hybrid_accuracy > baseline_accuracy,
        }

        logger.info(f"\nComparison Results:")
        logger.info(f"  Baseline (Alpha158):  {baseline_accuracy:.4f}")
        logger.info(f"  Hybrid (Alpha+Graph): {hybrid_accuracy:.4f}")
        logger.info(f"  Improvement: {improvement:+.2f}%")

        return comparison


class EnsembleHybridModel:
    """
    Ensemble of multiple models for robust predictions.

    Combines:
    - Alpha-only model
    - Graph-only model (if enough graph features)
    - Hybrid model
    - Weighted average of predictions
    """

    def __init__(
        self,
        alpha_model,
        hybrid_model,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble model.

        Args:
            alpha_model: Alpha-only trained model
            hybrid_model: Hybrid trained model
            weights: Optional weights for averaging
        """
        self.alpha_model = alpha_model
        self.hybrid_model = hybrid_model

        if weights is None:
            # Equal weights by default
            self.weights = [0.4, 0.6]  # Favor hybrid
        else:
            self.weights = weights

        logger.info(f"Ensemble model initialized with weights: {self.weights}")

    def predict(
        self,
        X_alpha: pd.DataFrame,
        X_hybrid: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            X_alpha: Alpha features only
            X_hybrid: Hybrid features (alpha + graph)

        Returns:
            Ensemble predictions
        """
        # Get predictions from both models
        pred_alpha = self.alpha_model.predict(X_alpha)
        pred_hybrid = self.hybrid_model.predict(X_hybrid)

        # Weighted average
        ensemble_pred = (
            self.weights[0] * pred_alpha +
            self.weights[1] * pred_hybrid
        )

        return ensemble_pred


def create_hybrid_signal_generator(
    config: Optional[QlibConfig] = None,
    model_path: Optional[Path] = None,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
) -> HybridSignalGenerator:
    """
    Convenience function to create hybrid signal generator.

    Args:
        config: Qlib configuration
        model_path: Path to saved model
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        HybridSignalGenerator instance
    """
    graph_extractor = GraphFeatureExtractor(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    return HybridSignalGenerator(
        config=config,
        model_path=model_path,
        graph_extractor=graph_extractor,
    )
