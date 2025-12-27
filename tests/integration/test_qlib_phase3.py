"""Test script for Qlib Phase 3: Hybrid Graph-ML Models.

This script tests:
1. Graph feature extraction from Neo4j
2. Hybrid feature generation (Alpha158 + Graph)
3. Hybrid model training
4. Comparison: Alpha-only vs Hybrid
5. Performance improvement measurement
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphwiz_trader.qlib import (
    QlibConfig,
    QlibDataAdapter,
    QlibSignalGenerator,
    HybridSignalGenerator,
    GraphFeatureExtractor,
    populate_sample_graph_data,
    create_hybrid_signal_generator,
)


async def test_graph_feature_extraction():
    """Test 1: Graph feature extraction."""
    logger.info("=" * 60)
    logger.info("TEST 1: Graph Feature Extraction")
    logger.info("=" * 60)

    try:
        # Initialize graph extractor
        extractor = GraphFeatureExtractor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )

        # Get graph summary stats
        logger.info("\nFetching graph summary...")
        stats = extractor.get_graph_summary_stats()
        logger.info(f"✓ Graph Statistics:")
        logger.info(f"  Total symbols: {stats['total_symbols']}")
        logger.info(f"  Total correlations: {stats['total_correlations']}")
        logger.info(f"  Total trades: {stats['total_trades']}")
        logger.info(f"  Total patterns: {stats['total_patterns']}")
        logger.info(f"  Avg correlation: {stats['avg_correlation']:.4f}")

        # Extract features for a symbol
        logger.info("\nExtracting graph features for BTC/USDT...")
        features = extractor.extract_all_features('BTC/USDT')

        logger.info(f"✓ Extracted {len(features)} graph features:")
        for feature_name, feature_value in features.items():
            logger.info(f"  {feature_name}: {feature_value:.4f}")

        extractor.close()

        logger.info("\n✓ Graph feature extraction test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Graph feature extraction test failed: {e}")
        logger.info("\nNote: This test requires Neo4j to be running.")
        logger.info("You can start Neo4j with: docker-compose up -d neo4j")
        return False


async def test_hybrid_feature_generation():
    """Test 2: Hybrid feature generation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Hybrid Feature Generation")
    logger.info("=" * 60)

    try:
        from graphwiz_trader.qlib import HybridFeatureGenerator, GraphFeatureExtractor

        # Create generators
        graph_extractor = GraphFeatureExtractor(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )

        hybrid_gen = HybridFeatureGenerator(graph_extractor=graph_extractor)

        # Generate sample data
        logger.info("Generating sample market data...")
        np.random.seed(42)
        n_periods = 200

        dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='1h')
        df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(n_periods) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(n_periods) * 0.5) + 1,
            'low': 100 + np.cumsum(np.random.randn(n_periods) * 0.5) - 1,
            'close': 100 + np.cumsum(np.random.randn(n_periods) * 0.5),
            'volume': np.random.randint(1000, 10000, n_periods),
        }, index=dates)

        # Generate hybrid features
        logger.info("Generating hybrid features (Alpha158 + Graph)...")
        hybrid_features = hybrid_gen.generate_hybrid_features(df, 'BTC/USDT')

        logger.info(f"✓ Generated {len(hybrid_features.columns)} hybrid features")
        logger.info(f"  Alpha features: {len(hybrid_gen.alpha_feature_names)}")
        logger.info(f"  Graph features: {len(hybrid_gen.graph_feature_names)}")

        logger.info("\nSample features:")
        logger.info(f"  Alpha examples: {hybrid_gen.alpha_feature_names[:5]}")
        logger.info(f"  Graph examples: {hybrid_gen.graph_feature_names}")

        graph_extractor.close()

        logger.info("\n✓ Hybrid feature generation test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Hybrid feature generation test failed: {e}")
        return False


async def test_hybrid_model_training():
    """Test 3: Hybrid model training."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Hybrid Model Training")
    logger.info("=" * 60)

    try:
        # Create hybrid signal generator
        generator = create_hybrid_signal_generator(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )

        # Generate sample data
        logger.info("Generating training data...")
        np.random.seed(42)
        n_periods = 500

        dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='1h')
        df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(n_periods) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(n_periods) * 0.5) + 1,
            'low': 100 + np.cumsum(np.random.randn(n_periods) * 0.5) - 1,
            'close': 100 + np.cumsum(np.random.randn(n_periods) * 0.5),
            'volume': np.random.randint(1000, 10000, n_periods),
        }, index=dates)

        # Train hybrid model
        logger.info("Training hybrid model...")
        results = generator.train(df, 'BTC/USDT', validation_split=0.2)

        logger.info(f"✓ Hybrid model trained:")
        logger.info(f"  Train accuracy: {results['train_accuracy']:.4f}")
        logger.info(f"  Val accuracy: {results['val_accuracy']:.4f}")
        logger.info(f"  Total features: {results['num_features']}")
        logger.info(f"  Alpha features: {results['num_alpha_features']}")
        logger.info(f"  Graph features: {results['num_graph_features']}")

        # Show feature importance
        if results['graph_feature_importance']:
            logger.info("\nTop graph features:")
            for feat in results['graph_feature_importance'][:5]:
                logger.info(f"  {feat['feature']}: {feat['importance']:.4f}")

        generator.graph_extractor.close()

        logger.info("\n✓ Hybrid model training test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Hybrid model training test failed: {e}")
        return False


async def test_comparison_alpha_vs_hybrid():
    """Test 4: Compare Alpha-only vs Hybrid models."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Comparison - Alpha-only vs Hybrid")
    logger.info("=" * 60)

    try:
        # Create hybrid generator
        generator = create_hybrid_signal_generator(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )

        # Generate sample data
        logger.info("Generating test data...")
        np.random.seed(42)
        n_periods = 500

        dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='1h')
        df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(n_periods) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(n_periods) * 0.5) + 1,
            'low': 100 + np.cumsum(np.random.randn(n_periods) * 0.5) - 1,
            'close': 100 + np.cumsum(np.random.randn(n_periods) * 0.5),
            'volume': np.random.randint(1000, 10000, n_periods),
        }, index=dates)

        # Run comparison
        logger.info("Running comparison...")
        comparison = generator.compare_with_baseline(df, 'BTC/USDT')

        logger.info("\n✓ Comparison Results:")
        logger.info(f"  Baseline (Alpha158):      {comparison['baseline_accuracy']:.4f}")
        logger.info(f"  Hybrid (Alpha+Graph):      {comparison['hybrid_accuracy']:.4f}")
        logger.info(f"  Improvement:               {comparison['accuracy_improvement_pct']:+.2f}%")
        logger.info(f"  Graph features added:      {comparison['graph_features_added']}")
        logger.info(f"  Hybrid better:              {comparison['hygraph_better']}")

        if comparison['hygraph_better']:
            logger.info("\n✓ Hybrid model OUTPERFORMS baseline!")
        else:
            logger.info("\n⚠ Baseline performs better (graph features may need tuning)")

        generator.graph_extractor.close()

        logger.info("\n✓ Comparison test complete")
        return True

    except Exception as e:
        logger.error(f"✗ Comparison test failed: {e}")
        return False


async def test_end_to_end_workflow():
    """Test 5: End-to-end workflow with Neo4j setup."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: End-to-End Workflow")
    logger.info("=" * 60)

    try:
        # Step 1: Populate sample graph data
        logger.info("\n[Step 1] Populating Neo4j with sample data...")
        await populate_sample_graph_data(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        )
        logger.info("✓ Sample data populated")

        # Step 2: Create hybrid generator
        logger.info("\n[Step 2] Creating hybrid signal generator...")
        generator = create_hybrid_signal_generator(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )
        logger.info("✓ Hybrid generator created")

        # Step 3: Generate sample market data
        logger.info("\n[Step 3] Generating market data...")
        np.random.seed(42)
        n_periods = 500

        dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='1h')
        df = pd.DataFrame({
            'open': 50000 + np.cumsum(np.random.randn(n_periods) * 200),
            'high': 50000 + np.cumsum(np.random.randn(n_periods) * 200) + 100,
            'low': 50000 + np.cumsum(np.random.randn(n_periods) * 200) - 100,
            'close': 50000 + np.cumsum(np.random.randn(n_periods) * 200),
            'volume': np.random.randint(100, 1000, n_periods),
        }, index=dates)
        logger.info(f"✓ Generated {len(df)} candles")

        # Step 4: Train and compare
        logger.info("\n[Step 4] Training models and comparing...")
        comparison = generator.compare_with_baseline(df, 'BTC/USDT')

        logger.info("\n✓ Final Results:")
        logger.info(f"  Accuracy improvement: {comparison['accuracy_improvement_pct']:+.2f}%")
        logger.info(f"  Features added: {comparison['graph_features_added']}")
        logger.info(f"  Hybrid better: {comparison['hygraph_better']}")

        generator.graph_extractor.close()

        logger.info("\n✓ End-to-end workflow test complete")
        return True

    except Exception as e:
        logger.error(f"✗ End-to-end workflow test failed: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Ensure Neo4j is running: docker-compose up -d neo4j")
        logger.info("  2. Check Neo4j credentials in test")
        logger.info("  3. Verify Neo4j is accessible at bolt://localhost:7687")
        return False


async def run_all_tests():
    """Run all Phase 3 tests."""
    logger.info("\n" + "=" * 60)
    logger.info("QLIB PHASE 3 TEST SUITE")
    logger.info("=" * 60)
    logger.info("Testing: Hybrid Graph-ML Models")
    logger.info(f"Started at: {datetime.now()}\n")

    test_results = []

    # Test 1: Graph Feature Extraction
    success = await test_graph_feature_extraction()
    test_results.append(("Graph Feature Extraction", success))

    # Test 2: Hybrid Feature Generation
    success = await test_hybrid_feature_generation()
    test_results.append(("Hybrid Feature Generation", success))

    # Test 3: Hybrid Model Training
    success = await test_hybrid_model_training()
    test_results.append(("Hybrid Model Training", success))

    # Test 4: Comparison
    success = await test_comparison_alpha_vs_hybrid()
    test_results.append(("Comparison Alpha vs Hybrid", success))

    # Test 5: End-to-End Workflow
    success = await test_end_to_end_workflow()
    test_results.append(("End-to-End Workflow", success))

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

    if passed < total:
        logger.info("\nNote: Some tests require Neo4j to be running.")
        logger.info("Start Neo4j with: docker-compose up -d neo4j")

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
