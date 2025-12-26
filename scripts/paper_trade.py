#!/usr/bin/env python3
"""
Paper trading script - test strategies without real money.

Usage:
    # Run once
    python scripts/paper_trade.py --symbol BTC/USDT

    # Run continuously (check every hour)
    python scripts/paper_trade.py --symbol BTC/USDT --continuous

    # Run for specific number of iterations
    python scripts/paper_trade.py --symbol BTC/USDT --iterations 10

    # Custom strategy parameters
    python scripts/paper_trade.py --symbol BTC/USDT --oversold 25 --overbought 65
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphwiz_trader.paper_trading import PaperTradingEngine
from loguru import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Paper trading - test strategies without real money"
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange to use (default: binance)"
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair symbol (default: BTC/USDT)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial virtual capital (default: 10000)"
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Trading commission rate (default: 0.001 = 0.1%%)"
    )
    parser.add_argument(
        "--oversold",
        type=int,
        default=25,
        help="RSI oversold level (default: 25)"
    )
    parser.add_argument(
        "--overbought",
        type=int,
        default=65,
        help="RSI overbought level (default: 65)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously (check every hour)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Interval between checks in seconds (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Maximum number of iterations (default: unlimited)"
    )

    args = parser.parse_args()

    # Strategy configuration
    strategy_config = {
        "oversold": args.oversold,
        "overbought": args.overbought,
    }

    # Initialize paper trading engine
    logger.info(f"\n{'='*80}")
    logger.info("PAPER TRADING SESSION")
    logger.info(f"{'='*80}")
    logger.info(f"Exchange:    {args.exchange}")
    logger.info(f"Symbol:      {args.symbol}")
    logger.info(f"Capital:     ${args.capital:,.2f}")
    logger.info(f"Commission:  {args.commission*100:.2f}%")
    logger.info(f"Strategy:    RSI({args.oversold}/{args.overbought})")
    logger.info(f"{'='*80}\n")

    engine = PaperTradingEngine(
        exchange_name=args.exchange,
        symbol=args.symbol,
        initial_capital=args.capital,
        commission=args.commission,
        strategy_config=strategy_config,
    )

    try:
        if args.continuous or args.iterations:
            # Run continuous or limited iterations
            engine.start(
                interval_seconds=args.interval,
                max_iterations=args.iterations
            )
        else:
            # Run once
            logger.info("Running single iteration...")
            result = engine.run_once()

            if result["status"] == "success":
                logger.success(f"\n✅ Single iteration complete!")
                logger.info(f"Price: ${result['price']:,.2f}")
                logger.info(f"Signal: {result['signal']}")
                logger.info(f"Portfolio Value: ${result['portfolio_value']:,.2f}")
                logger.info(f"Total Return: {result['total_return_pct']:+.2f}%")
            else:
                logger.error(f"❌ Error: {result.get('message')}")

            # Save and print summary
            engine.save_results()
            engine.print_summary()

        return 0

    except KeyboardInterrupt:
        logger.info("\n\nReceived interrupt signal")
        engine.save_results()
        engine.print_summary()
        return 0

    except Exception as e:
        logger.exception(f"Paper trading failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
