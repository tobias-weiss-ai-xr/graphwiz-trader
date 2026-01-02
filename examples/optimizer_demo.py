#!/usr/bin/env python3
"""Example script demonstrating the optimization orchestrator.

This script shows how to use the OptimizationOrchestrator to run
continuous autonomous optimization of the trading system.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphwiz_trader.optimizer import OptimizationOrchestrator
from loguru import logger


async def main():
    """Run the optimization orchestrator demo."""

    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    logger.info("=== GraphWiz Trader Optimization Demo ===")

    # Initialize orchestrator
    orchestrator = OptimizationOrchestrator(
        project_path="/opt/git/graphwiz-trader",
        enable_auto_approve=False,  # Require manual approval for safety
    )

    # Display configuration
    logger.info(f"Initialized orchestrator with {len(orchestrator.optimization_loops)} optimization loops")

    # Display optimization loops
    for name, loop in orchestrator.optimization_loops.items():
        logger.info(f"  - {name}: {loop.optimization_type.value} (every {loop.frequency_minutes}m)")

    # Start orchestrator
    logger.info("\nStarting orchestrator...")
    await orchestrator.start()

    # Display status
    status = orchestrator.get_status()
    logger.info(f"\nOrchestrator status: {status['state']}")
    logger.info(f"Circuit breaker: {status['circuit_breaker_state']}")

    # Run for a short time (for demo)
    logger.info("\nRunning for 30 seconds (demo mode)...")
    await asyncio.sleep(30)

    # Check status again
    status = orchestrator.get_status()
    logger.info(f"\nActive optimizations: {status['active_optimizations']}")
    logger.info(f"Pending approvals: {status['pending_approvals']}")

    # Get optimization history
    history = orchestrator.get_optimization_history()
    if history:
        logger.info(f"\nRecent optimizations: {len(history)}")
        for opt in history[-5:]:  # Show last 5
            logger.info(f"  - {opt['type']}: {opt['status']} (confidence: {opt['confidence']:.2f})")
    else:
        logger.info("\nNo optimizations yet")

    # Stop orchestrator
    logger.info("\nStopping orchestrator...")
    await orchestrator.stop()

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
