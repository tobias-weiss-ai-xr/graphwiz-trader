#!/usr/bin/env python3
"""
Live trading script - executes REAL trades with REAL money.

⚠️ WARNING: This script will execute REAL trades!
Make sure you:
1. Have tested thoroughly with paper trading
2. Understand the risks
3. Have set appropriate safety limits
4. Start with small amounts

Usage:
    # Run once (test mode)
    python scripts/live_trade.py --symbol BTC/USDT --test

    # Run continuous live trading
    python scripts/live_trade.py --symbol BTC/USDT

    # Custom safety limits
    python scripts/live_trade.py --symbol BTC/USDT --max-position 500 --max-daily-loss 200
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphwiz_trader.live_trading import LiveTradingEngine, SafetyLimits
from loguru import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live trading - executes REAL trades with REAL money"
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
        "--api-key",
        default=os.getenv("EXCHANGE_API_KEY"),
        help="Exchange API key (or set EXCHANGE_API_KEY env var)"
    )
    parser.add_argument(
        "--api-secret",
        default=os.getenv("EXCHANGE_API_SECRET"),
        help="Exchange API secret (or set EXCHANGE_API_SECRET env var)"
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
        "--max-position",
        type=float,
        default=100,
        help="Maximum position size in $ (default: 100)"
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.02,
        help="Maximum position %% of portfolio (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=200,
        help="Maximum daily loss in $ (default: 200)"
    )
    parser.add_argument(
        "--max-daily-loss-pct",
        type=float,
        default=0.05,
        help="Maximum daily loss %% of portfolio (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--max-daily-trades",
        type=int,
        default=5,
        help="Maximum trades per day (default: 5)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run once for testing (don't start continuous trading)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Interval between checks in seconds (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution)"
    )

    args = parser.parse_args()

    # Validate API credentials
    if not args.api_key or not args.api_secret:
        logger.error("❌ API credentials required!")
        logger.error("Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET environment variables")
        logger.error("Or use --api-key and --api-secret arguments")
        return 1

    # Safety limits
    safety_limits = SafetyLimits(
        max_position_size=args.max_position,
        max_position_pct=args.max_position_pct,
        max_daily_loss=args.max_daily_loss,
        max_daily_loss_pct=args.max_daily_loss_pct,
        max_daily_trades=args.max_daily_trades,
        require_confirmation=not args.no_confirm,
    )

    # Strategy configuration
    strategy_config = {
        "oversold": args.oversold,
        "overbought": args.overbought,
    }

    # Print warning
    print("\n" + "=" * 80)
    print("⚠️  WARNING: LIVE TRADING - REAL MONEY WILL BE USED")
    print("=" * 80)
    print(f"Exchange:        {args.exchange}")
    print(f"Symbol:          {args.symbol}")
    print(f"Strategy:        RSI({args.oversold}/{args.overbought})")
    print("-" * 80)
    print("Safety Limits:")
    print(f"  Max Position:  ${args.max_position:,.2f} ({args.max_position_pct*100:.1f}%)")
    print(f"  Max Daily Loss: ${args.max_daily_loss:,.2f} ({args.max_daily_loss_pct*100:.1f}%)")
    print(f"  Max Daily Trades: {args.max_daily_trades}")
    print("-" * 80)
    print("⚠️  Make sure you have:")
    print("  1. Tested thoroughly with paper trading")
    print("  2. Started with small amounts")
    print("  3. Understand the risks")
    print("=" * 80 + "\n")

    # Initialize live trading engine
    try:
        engine = LiveTradingEngine(
            exchange_name=args.exchange,
            symbol=args.symbol,
            api_key=args.api_key,
            api_secret=args.api_secret,
            strategy_config=strategy_config,
            safety_limits=safety_limits,
        )

        if args.test:
            # Run once for testing
            logger.info("Running single test iteration...")
            result = engine.run_once()

            if result["status"] == "success":
                logger.success(f"\n✅ Test iteration complete!")
                logger.info(f"Price: ${result['price']:,.2f}")
                logger.info(f"Signal: {result['signal']}")
                logger.info(f"Action: {result['action_taken'] or 'None'}")
                logger.info(f"Portfolio Value: ${result['portfolio_value']:,.2f}")
                logger.info(f"Daily P&L: ${result['daily_pnl']:+,.2f}")
            else:
                logger.error(f"❌ Error: {result.get('message')}")

            # Save and print summary
            engine.save_results()
            engine.print_summary()

        else:
            # Run continuous live trading
            engine.start(interval_seconds=args.interval)

        return 0

    except KeyboardInterrupt:
        logger.info("\n\nReceived interrupt signal")
        if 'engine' in locals():
            engine.save_results()
            engine.print_summary()
        return 0

    except Exception as e:
        logger.exception(f"Live trading failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
