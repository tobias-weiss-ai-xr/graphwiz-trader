"""Test script for Qlib integration with GraphWiz Trader.

This script tests the complete Phase 1 integration:
1. Data adapter (CCXT to Qlib)
2. Alpha158 feature extraction
3. LightGBM model training
4. Signal generation
5. Comparison with technical indicators
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphwiz_trader.qlib import (
    QlibConfig,
    QlibDataAdapter,
    AlphaFeatureExtractor,
    QlibSignalGenerator,
)
from graphwiz_trader.trading.engine import TradingEngine


async def test_data_adapter():
    """Test 1: Data adapter functionality."""
    logger.info("=" * 60)
    logger.info("TEST 1: Data Adapter")
    logger.info("=" * 60)

    adapter = QlibDataAdapter(exchange_id="binance")

    try:
        await adapter.initialize()

        # Test fetching OHLCV data
        logger.info("Fetching BTC/USDT data...")
        df = await adapter.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="1h",
            limit=100,
        )

        logger.info(f"✓ Fetched {len(df)} candles")
        logger.info(f"✓ Columns: {list(df.columns)}")
        logger.info(f"✓ Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"\nSample data:\n{df.head()}")

        # Test Qlib format conversion
        qlib_df = adapter.to_qlib_format(df, "BTCUSDT")
        logger.info(f"✓ Converted to Qlib format: {len(qlib_df)} rows")

        return True, df

    except Exception as e:
        logger.error(f"✗ Data adapter test failed: {e}")
        return False, None
    finally:
        await adapter.close()


async def test_feature_extraction(df):
    """Test 2: Feature extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Feature Extraction")
    logger.info("=" * 60)

    try:
        extractor = AlphaFeatureExtractor()

        # Extract features
        logger.info("Extracting features...")
        features_df = extractor.prepare_features_for_training(df, "BTC/USDT")

        logger.info(f"✓ Extracted {len(features_df.columns)} features")
        logger.info(f"✓ Feature rows: {len(features_df)}")
        logger.info(f"\nSample features:\n{features_df.head()}")

        # Show feature names
        logger.info(f"\nFeature names:\n{list(features_df.columns)[:20]}...")

        return True, features_df

    except Exception as e:
        logger.error(f"✗ Feature extraction test failed: {e}")
        return False, None


async def test_model_training(df):
    """Test 3: Model training."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Model Training")
    logger.info("=" * 60)

    try:
        signal_generator = QlibSignalGenerator()

        # Train model
        logger.info("Training LightGBM model...")
        results = signal_generator.train(
            df=df,
            symbol="BTC/USDT",
            validation_split=0.2,
        )

        logger.info(f"✓ Training complete")
        logger.info(f"✓ Train accuracy: {results['train_accuracy']:.4f}")
        logger.info(f"✓ Val accuracy: {results['val_accuracy']:.4f}")
        logger.info(f"✓ Train samples: {results['train_samples']}")
        logger.info(f"✓ Val samples: {results['val_samples']}")
        logger.info(f"✓ Number of features: {results['num_features']}")

        # Show feature importance
        importance_df = signal_generator.get_feature_importance(top_n=10)
        logger.info(f"\nTop 10 features:\n{importance_df}")

        return True, signal_generator

    except Exception as e:
        logger.error(f"✗ Model training test failed: {e}")
        return False, None


async def test_signal_generation(df, signal_generator):
    """Test 4: Signal generation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Signal Generation")
    logger.info("=" * 60)

    try:
        # Generate signals
        logger.info("Generating signals...")
        signals = signal_generator.predict(
            df=df,
            symbol="BTC/USDT",
            threshold=0.5,
        )

        logger.info(f"✓ Generated {len(signals)} signals")
        logger.info(f"✓ Signal distribution:")
        logger.info(f"  - BUY: {(signals['signal'] == 1).sum()}")
        logger.info(f"  - HOLD/SELL: {(signals['signal'] == 0).sum()}")

        # Get latest prediction
        latest = signal_generator.predict_latest(df, "BTC/USDT")
        logger.info(f"\n✓ Latest signal:")
        logger.info(f"  - Timestamp: {latest['timestamp']}")
        logger.info(f"  - Signal: {latest['signal']}")
        logger.info(f"  - Probability: {latest['probability']:.4f}")
        logger.info(f"  - Confidence: {latest['confidence']}")

        return True

    except Exception as e:
        logger.error(f"✗ Signal generation test failed: {e}")
        return False


async def test_comparison_with_technical_indicators(df):
    """Test 5: Compare Qlib ML signals with technical indicators."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Comparison with Technical Indicators")
    logger.info("=" * 60)

    try:
        # Calculate simple technical indicators for comparison
        logger.info("Calculating technical indicators...")

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Moving averages
        ma_short = df['close'].rolling(window=10).mean()
        ma_long = df['close'].rolling(window=30).mean()

        # Generate technical signals
        tech_signals = pd.DataFrame(index=df.index)
        tech_signals['rsi'] = rsi
        tech_signals['ma_short'] = ma_short
        tech_signals['ma_long'] = ma_long
        tech_signals['ma_crossover'] = (ma_short > ma_long).astype(int)
        tech_signals['rsi_oversold'] = (rsi < 30).astype(int)
        tech_signals['rsi_overbought'] = (rsi > 70).astype(int)

        # Simple rule-based signal
        tech_signals['tech_signal'] = 0
        tech_signals.loc[
            (tech_signals['ma_crossover'] == 1) &
            (tech_signals['rsi_oversold'] == 1),
            'tech_signal'
        ] = 1  # BUY

        logger.info(f"✓ Technical signals generated")
        logger.info(f"  - MA Crossover BUY signals: {tech_signals['ma_crossover'].sum()}")
        logger.info(f"  - RSI oversold: {tech_signals['rsi_oversold'].sum()}")
        logger.info(f"  - Combined BUY signals: {tech_signals['tech_signal'].sum()}")

        # Generate ML signals for comparison
        signal_generator = QlibSignalGenerator()
        signal_generator.train(df, "BTC/USDT", validation_split=0.2)
        ml_signals = signal_generator.predict(df, "BTC/USDT")

        # Compare
        logger.info(f"\n✓ ML signals:")
        logger.info(f"  - Total BUY signals: {ml_signals['signal'].sum()}")

        # Calculate correlation
        comparison = pd.DataFrame({
            'technical': tech_signals['tech_signal'],
            'ml': ml_signals['signal'],
        }).dropna()

        if len(comparison) > 0:
            correlation = comparison['technical'].corr(comparison['ml'])
            logger.info(f"\n✓ Signal correlation: {correlation:.4f}")

            # Agreement rate
            agreement = (comparison['technical'] == comparison['ml']).mean()
            logger.info(f"✓ Agreement rate: {agreement:.2%}")

        return True

    except Exception as e:
        logger.error(f"✗ Comparison test failed: {e}")
        return False


async def run_all_tests():
    """Run all integration tests."""
    logger.info("\n" + "=" * 60)
    logger.info("QLIB INTEGRATION TEST SUITE - PHASE 1")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now()}\n")

    test_results = []

    # Test 1: Data Adapter
    success, df = await test_data_adapter()
    test_results.append(("Data Adapter", success))

    if not success or df is None:
        logger.error("Cannot continue tests without data.")
        return

    # Test 2: Feature Extraction
    success, features_df = await test_feature_extraction(df)
    test_results.append(("Feature Extraction", success))

    # Test 3: Model Training
    success, signal_generator = await test_model_training(df)
    test_results.append(("Model Training", success))

    if not success or signal_generator is None:
        logger.error("Cannot continue tests without trained model.")
        return

    # Test 4: Signal Generation
    success = await test_signal_generation(df, signal_generator)
    test_results.append(("Signal Generation", success))

    # Test 5: Comparison
    success = await test_comparison_with_technical_indicators(df)
    test_results.append(("Technical Indicators Comparison", success))

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
