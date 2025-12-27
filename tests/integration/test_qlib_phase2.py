"""Test script for Qlib Phase 2: Portfolio Optimization and Backtesting.

This script tests:
1. Portfolio optimization (mean-variance, risk parity, etc.)
2. Dynamic position sizing
3. Advanced backtesting
4. Model validation
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphwiz_trader.qlib import (
    QlibConfig,
    QlibDataAdapter,
    QlibSignalGenerator,
    PortfolioOptimizer,
    DynamicPositionSizer,
    PortfolioConstraints,
    OptimizerConfig,
    BacktestEngine,
    BacktestConfig,
    ModelValidator,
)


async def test_portfolio_optimization():
    """Test 1: Portfolio optimization methods."""
    logger.info("=" * 60)
    logger.info("TEST 1: Portfolio Optimization")
    logger.info("=" * 60)

    try:
        # Create sample returns data
        np.random.seed(42)
        n_assets = 5
        n_periods = 252  # 1 year of daily data

        # Generate correlated returns
        returns = pd.DataFrame(
            np.random.randn(n_periods, n_assets) * 0.02,
            columns=[f'Asset_{i}' for i in range(n_assets)],
        )

        logger.info(f"Generated sample returns data: {returns.shape}")

        # Test different optimization methods
        methods = ['mean_variance', 'max_sharpe', 'min_variance', 'risk_parity', 'equal_weight']

        for method in methods:
            try:
                logger.info(f"\nTesting {method} optimization...")

                optimizer = PortfolioOptimizer(
                    config=OptimizerConfig(optimization_method=method),
                    constraints=PortfolioConstraints(
                        max_position_weight=0.4,
                        min_position_weight=0.0,
                    ),
                )

                weights = optimizer.optimize(returns)

                logger.info(f"✓ {method} weights: {weights.round(4).to_dict()}")
                logger.info(f"  Sum of weights: {weights.sum():.4f}")

                # Calculate portfolio metrics
                metrics = optimizer.calculate_portfolio_metrics(weights, returns)
                logger.info(f"  Expected return: {metrics['annualized_return']:.2%}")
                logger.info(f"  Volatility: {metrics['volatility']:.2%}")
                logger.info(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

            except Exception as e:
                logger.error(f"✗ {method} optimization failed: {e}")

        logger.info("\n✓ Portfolio optimization test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Portfolio optimization test failed: {e}")
        return False


async def test_dynamic_position_sizing():
    """Test 2: Dynamic position sizing."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Dynamic Position Sizing")
    logger.info("=" * 60)

    try:
        sizer = DynamicPositionSizer(
            base_position_size=0.1,
            max_position_size=0.3,
            min_position_size=0.05,
            risk_tolerance=0.02,
        )

        # Test different scenarios
        scenarios = [
            ("High confidence, low volatility", 0.9, 100000, 50000, 0.3),
            ("Low confidence, high volatility", 0.55, 100000, 50000, 0.8),
            ("Medium confidence, medium volatility", 0.7, 100000, 50000, 0.5),
        ]

        for scenario_name, confidence, portfolio_value, price, volatility in scenarios:
            position_size = sizer.calculate_position_size(
                signal_confidence=confidence,
                portfolio_value=portfolio_value,
                asset_price=price,
                asset_volatility=volatility,
            )

            logger.info(f"\n{scenario_name}:")
            logger.info(f"  Confidence: {confidence:.2%}")
            logger.info(f"  Volatility: {volatility:.2%}")
            logger.info(f"  Position size: ${position_size:,.2f}")
            logger.info(f"  Position as % of portfolio: {position_size/portfolio_value:.2%}")

        # Test weight-based sizing
        logger.info(f"\nWeight-based sizing:")
        for weight in [0.1, 0.2, 0.3, 0.4]:
            position_size = sizer.calculate_position_size_by_weight(
                optimal_weight=weight,
                portfolio_value=100000,
                asset_price=50000,
            )
            logger.info(f"  Weight {weight:.1%}: ${position_size:,.2f}")

        logger.info("\n✓ Dynamic position sizing test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Dynamic position sizing test failed: {e}")
        return False


async def test_backtesting():
    """Test 3: Advanced backtesting."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Advanced Backtesting")
    logger.info("=" * 60)

    try:
        # Create sample data
        np.random.seed(42)
        n_periods = 500

        # Generate price data
        returns = np.random.randn(n_periods) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate signals
        signals = pd.Series(
            np.random.randint(0, 2, n_periods),
            index=pd.date_range(start='2024-01-01', periods=n_periods, freq='1h'),
        )

        price_data = pd.DataFrame({
            'close': prices,
        }, index=signals.index)

        # Create backtest engine
        config = BacktestConfig(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
        )

        engine = BacktestEngine(config=config)

        # Run backtest
        logger.info("Running backtest...")
        result = engine.run_backtest(signals, price_data)

        # Display results
        logger.info(f"\n✓ Backtest complete")
        logger.info(f"  Total return: {result.total_return:.2%}")
        logger.info(f"  Annualized return: {result.annualized_return:.2%}")
        logger.info(f"  Volatility: {result.volatility:.2%}")
        logger.info(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Max drawdown: {result.max_drawdown:.2%}")
        logger.info(f"  Win rate: {result.win_rate:.2%}")
        logger.info(f"  Total trades: {result.total_trades}")

        # Generate report
        report = engine.generate_report(result)
        logger.info(f"\n{report}")

        logger.info("\n✓ Backtesting test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Backtesting test failed: {e}")
        return False


async def test_model_validation():
    """Test 4: Model validation and selection."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Model Validation")
    logger.info("=" * 60)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)],
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))

        logger.info(f"Generated sample data: {X.shape}")

        # Create models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        }

        # Create validator
        validator = ModelValidator(n_folds=5)

        # Select best model
        logger.info("\nSelecting best model...")
        best_model, results = validator.select_best_model(models, X, y)

        logger.info(f"\n✓ Best model: {best_model}")
        logger.info(f"  Mean score: {results['mean_score']:.4f}")
        logger.info(f"  Std score: {results['std_score']:.4f}")
        logger.info(f"  Fold scores: {[f'{s:.4f}' for s in results['scores']]}")

        logger.info("\n✓ Model validation test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Model validation test failed: {e}")
        return False


async def test_integration_with_signals():
    """Test 5: Integration with signal generation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Integration with Signal Generation")
    logger.info("=" * 60)

    try:
        # Initialize data adapter
        adapter = QlibDataAdapter(exchange_id="binance")
        await adapter.initialize()

        logger.info("Fetching BTC/USDT data...")
        df = await adapter.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="1h",
            limit=500,
        )

        if len(df) < 100:
            logger.error("Insufficient data for integration test")
            await adapter.close()
            return False

        # Train model
        logger.info("Training model...")
        signal_gen = QlibSignalGenerator()
        results = signal_gen.train(df, "BTC/USDT", validation_split=0.2)

        logger.info(f"✓ Model trained: Val accuracy = {results['val_accuracy']:.2%}")

        # Generate signals
        logger.info("Generating signals...")
        signals = signal_gen.predict(df, "BTC/USDT")

        # Run backtest on signals
        logger.info("Running backtest on generated signals...")
        backtest_config = BacktestConfig(
            initial_capital=100000,
            commission=0.001,
        )
        backtest_engine = BacktestEngine(config=backtest_config)

        price_data = df[['close']].copy()
        backtest_result = backtest_engine.run_backtest(
            signals[['signal']],
            price_data,
        )

        logger.info(f"\n✓ Backtest Results:")
        logger.info(f"  Total return: {backtest_result.total_return:.2%}")
        logger.info(f"  Sharpe ratio: {backtest_result.sharpe_ratio:.2f}")
        logger.info(f"  Win rate: {backtest_result.win_rate:.2%}")
        logger.info(f"  Total trades: {backtest_result.total_trades}")

        await adapter.close()

        logger.info("\n✓ Integration test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all Phase 2 tests."""
    logger.info("\n" + "=" * 60)
    logger.info("QLIB PHASE 2 TEST SUITE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now()}\n")

    test_results = []

    # Test 1: Portfolio Optimization
    success = await test_portfolio_optimization()
    test_results.append(("Portfolio Optimization", success))

    # Test 2: Dynamic Position Sizing
    success = await test_dynamic_position_sizing()
    test_results.append(("Dynamic Position Sizing", success))

    # Test 3: Backtesting
    success = await test_backtesting()
    test_results.append(("Backtesting", success))

    # Test 4: Model Validation
    success = await test_model_validation()
    test_results.append(("Model Validation", success))

    # Test 5: Integration (optional, requires network)
    logger.info("\n" + "=" * 60)
    logger.info("Running integration test (requires network)...")
    logger.info("=" * 60)
    success = await test_integration_with_signals()
    test_results.append(("Integration with Signals", success))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    logger.info(f"\nTotal: {passed}/{total} tests passed")

    logger.info(f"\nCompleted at: {datetime.now()}")

    return passed == total


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Run tests
    result = asyncio.run(run_all_tests())

    # Exit with appropriate code
    sys.exit(0 if result else 1)
