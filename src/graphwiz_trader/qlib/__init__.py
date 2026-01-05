"""Qlib integration for GraphWiz Trader.

This module provides integration with Microsoft's Qlib platform for
AI-oriented quantitative investment, including:
- Data adapters for CCXT to Qlib
- Alpha158 feature extraction
- LightGBM signal generation
- Portfolio optimization
- Advanced backtesting
- Graph-based features (Neo4j)
- Hybrid ML models (Alpha + Graph)
- RL-based execution (slippage reduction)
"""

from .config import QlibConfig, default_config
from .data_adapter import QlibDataAdapter
from .features import AlphaFeatureExtractor, extract_features
from .models import QlibSignalGenerator, EnsembleSignalGenerator, create_signal_generator
from .portfolio import (
    PortfolioOptimizer,
    DynamicPositionSizer,
    PortfolioConstraints,
    OptimizerConfig,
    create_portfolio_optimizer,
)
from .backtest import (
    BacktestEngine,
    BacktestResult,
    BacktestConfig,
    ModelValidator,
    create_backtest_engine,
)
from .graph_features import (
    GraphFeatureExtractor,
    populate_sample_graph_data,
    create_graph_feature_extractor,
)
from .hybrid_models import (
    HybridFeatureGenerator,
    HybridSignalGenerator,
    EnsembleHybridModel,
    create_hybrid_signal_generator,
)
from .rl_execution import (
    ExecutionEnvironment,
    ExecutionAction,
    OrderBook,
    ExecutionState,
    TWAPExecutor,
    SmartOrderRouter,
    ExecutionAnalyzer,
    create_execution_environment,
)
from .execution_strategies import (
    ExecutionStrategy,
    ExecutionPlan,
    OptimalExecutionEngine,
    SlippageMinimizer,
    create_optimal_execution_engine,
)

__all__ = [
    # Configuration
    "QlibConfig",
    "default_config",
    # Data adapter
    "QlibDataAdapter",
    # Feature extraction
    "AlphaFeatureExtractor",
    "extract_features",
    # Signal generation
    "QlibSignalGenerator",
    "EnsembleSignalGenerator",
    "create_signal_generator",
    # Portfolio optimization
    "PortfolioOptimizer",
    "DynamicPositionSizer",
    "PortfolioConstraints",
    "OptimizerConfig",
    "create_portfolio_optimizer",
    # Backtesting
    "BacktestEngine",
    "BacktestResult",
    "BacktestConfig",
    "ModelValidator",
    "create_backtest_engine",
    # Graph features
    "GraphFeatureExtractor",
    "populate_sample_graph_data",
    "create_graph_feature_extractor",
    # Hybrid models
    "HybridFeatureGenerator",
    "HybridSignalGenerator",
    "EnsembleHybridModel",
    "create_hybrid_signal_generator",
    # RL execution
    "ExecutionEnvironment",
    "ExecutionAction",
    "OrderBook",
    "ExecutionState",
    "TWAPExecutor",
    "SmartOrderRouter",
    "ExecutionAnalyzer",
    "create_execution_environment",
    # Execution strategies
    "ExecutionStrategy",
    "ExecutionPlan",
    "OptimalExecutionEngine",
    "SlippageMinimizer",
    "create_optimal_execution_engine",
]

__version__ = "0.4.0"
