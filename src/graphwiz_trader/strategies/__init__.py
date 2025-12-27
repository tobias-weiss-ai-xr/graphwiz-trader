"""
Advanced trading strategies.

Includes:
- Multi-strategy portfolio
- Market regime detection
- Adaptive parameters
- Ensemble methods
- Qlib ML-based strategy (V1: Basic, V2: With portfolio optimization)
- 2025 Research-based strategies (Mean Reversion, Pairs Trading, Momentum, Multi-Factor)
"""

from .regime_detection import MarketRegime, RegimeDetector
from .adaptive_strategy import AdaptiveRSIStrategy
from .qlib_strategy import QlibStrategy, create_qlib_strategy
from .qlib_strategy_v2 import QlibStrategyV2, create_qlib_strategy_v2

# 2025 Research-based strategies
from .advanced_strategies import (
    AdvancedMeanReversionStrategy,
    PairsTradingStrategy,
    MomentumVolatilityFilteringStrategy,
    MultiFactorStrategy,
    ConfidenceThresholdStrategy,
    MeanReversionType,
    create_advanced_strategy,
)

__all__ = [
    "MarketRegime",
    "RegimeDetector",
    "AdaptiveRSIStrategy",
    "QlibStrategy",
    "create_qlib_strategy",
    "QlibStrategyV2",
    "create_qlib_strategy_v2",
    # 2025 Research-based strategies
    "AdvancedMeanReversionStrategy",
    "PairsTradingStrategy",
    "MomentumVolatilityFilteringStrategy",
    "MultiFactorStrategy",
    "ConfidenceThresholdStrategy",
    "MeanReversionType",
    "create_advanced_strategy",
]
