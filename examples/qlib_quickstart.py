"""Quick start example for Qlib integration.

This script demonstrates the basic usage of the Qlib integration
with GraphWiz Trader.
"""

import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from graphwiz_trader.qlib import (
    QlibConfig,
    QlibDataAdapter,
    QlibSignalGenerator,
)


async def quickstart():
    """Quick start demonstration of Qlib integration."""

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     Qlib Integration - Quick Start Demo                ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Step 1: Initialize data adapter
    logger.info("\n[1/5] Initializing data adapter...")
    adapter = QlibDataAdapter(exchange_id="binance")
    await adapter.initialize()
    logger.info("✓ Data adapter initialized")

    # Step 2: Fetch market data
    logger.info("\n[2/5] Fetching market data for BTC/USDT...")
    df = await adapter.fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=500,
    )
    logger.info(f"✓ Fetched {len(df)} candles")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Step 3: Create and train model
    logger.info("\n[3/5] Training LightGBM model...")
    signal_generator = QlibSignalGenerator()

    try:
        results = signal_generator.train(
            df=df,
            symbol="BTC/USDT",
            validation_split=0.2,
        )

        logger.info("✓ Model trained successfully")
        logger.info(f"  Training accuracy: {results['train_accuracy']:.2%}")
        logger.info(f"  Validation accuracy: {results['val_accuracy']:.2%}")
        logger.info(f"  Features used: {results['num_features']}")

        # Show top 5 features
        importance_df = signal_generator.get_feature_importance(top_n=5)
        logger.info("\n  Top 5 Features:")
        for _, row in importance_df.iterrows():
            logger.info(f"    - {row['feature']}: {row['importance']:.2f}")

    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        logger.info("\nThis might be due to insufficient data or missing dependencies.")
        logger.info("Please ensure you have:")
        logger.info("  - Internet connection (to fetch market data)")
        logger.info("  - Installed Qlib dependencies (pip install qlib lightgbm)")
        await adapter.close()
        return

    # Step 4: Generate signals
    logger.info("\n[4/5] Generating trading signals...")
    signals = signal_generator.predict(
        df=df,
        symbol="BTC/USDT",
        threshold=0.5,
    )

    logger.info(f"✓ Generated {len(signals)} signals")
    logger.info(f"  BUY signals: {(signals['signal'] == 1).sum()}")
    logger.info(f"  HOLD signals: {(signals['signal'] == 0).sum()}")

    # Step 5: Get latest prediction
    logger.info("\n[5/5] Getting latest prediction...")
    latest = signal_generator.predict_latest(df, "BTC/USDT")

    logger.info("✓ Latest trading signal:")
    logger.info(f"  Timestamp: {latest['timestamp']}")
    logger.info(f"  Signal: {latest['signal']}")
    logger.info(f"  Probability: {latest['probability']:.2%}")
    logger.info(f"  Confidence: {latest['confidence']}")

    # Optional: Save model
    logger.info("\n[BONUS] Saving model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "btcusdt_qlib_model.pkl"

    signal_generator.save_model(model_path)
    logger.info(f"✓ Model saved to {model_path}")

    # Cleanup
    await adapter.close()

    logger.info("\n╔══════════════════════════════════════════════════════════╗")
    logger.info("║                    Demo Complete!                       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    logger.info("\nNext Steps:")
    logger.info("  1. Run the full test suite:")
    logger.info("     python tests/integration/test_qlib_integration.py")
    logger.info("\n  2. Integrate with your trading strategy:")
    logger.info("     from graphwiz_trader.strategies import create_qlib_strategy")
    logger.info("\n  3. Read the documentation:")
    logger.info("     docs/QLIB_PHASE1_DOCUMENTATION.md")
    logger.info("\n  4. Review the integration analysis:")
    logger.info("     QLIB_INTEGRATION_ANALYSIS.md")


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Run demo
    try:
        asyncio.run(quickstart())
    except KeyboardInterrupt:
        logger.info("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"\n\nDemo failed with error: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check your internet connection")
        logger.info("  2. Ensure dependencies are installed:")
        logger.info("     pip install qlib lightgbm torch h5py")
        logger.info("  3. Try running the test suite for more details")
        sys.exit(1)
