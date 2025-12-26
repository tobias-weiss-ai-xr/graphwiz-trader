#!/usr/bin/env python3
"""
Paper trading service control CLI.

Manages paper trading as a background service.

Usage:
    python scripts/paper_trading_service.py start
    python scripts/paper_trading_service.py stop
    python scripts/paper_trading_service.py restart
    python scripts/paper_trading_service.py status
    python scripts/paper_trading_service.py logs [--symbol BTC/USDT]
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphwiz_trader.paper_trading.service import (
    PaperTradingService,
    PaperTradingConfig,
)
from loguru import logger


def cmd_start(args, service: PaperTradingService):
    """Start the service."""
    logger.info("Starting paper trading service...")
    success = service.start()

    if success:
        logger.success("✅ Service started successfully")
        service.print_status()
        return 0
    else:
        logger.error("❌ Failed to start service")
        return 1


def cmd_stop(args, service: PaperTradingService):
    """Stop the service."""
    logger.info("Stopping paper trading service...")
    success = service.stop()

    if success:
        logger.success("✅ Service stopped successfully")
        return 0
    else:
        logger.error("❌ Failed to stop service")
        return 1


def cmd_restart(args, service: PaperTradingService):
    """Restart the service."""
    logger.info("Restarting paper trading service...")
    success = service.restart()

    if success:
        logger.success("✅ Service restarted successfully")
        service.print_status()
        return 0
    else:
        logger.error("❌ Failed to restart service")
        return 1


def cmd_status(args, service: PaperTradingService):
    """Show service status."""
    service.print_status()
    return 0


def cmd_logs(args, service: PaperTradingService):
    """Show logs."""
    logs = service.logs(symbol=args.symbol, tail=args.tail)

    if not logs:
        logger.warning("No logs found")
        return 1

    print(f"\n{'='*80}")
    print(f"PAPER TRADING LOGS{f' - {args.symbol}' if args.symbol else ''}")
    print(f"{'='*80}\n")

    for log in logs[-args.tail:]:
        print(log.rstrip())

    print(f"\n{'='*80}\n")
    return 0


def cmd_add(args, service: PaperTradingService):
    """Add a new symbol."""
    config = PaperTradingConfig(
        symbol=args.symbol,
        capital=args.capital,
        oversold=args.oversold,
        overbought=args.overbought,
        interval=args.interval,
        enabled=True,
    )

    service.add_symbol(config)
    logger.success(f"✅ Added {args.symbol} to configuration")

    if args.restart:
        logger.info("Restarting service...")
        service.restart()

    return 0


def cmd_remove(args, service: PaperTradingService):
    """Remove a symbol."""
    service.remove_symbol(args.symbol)
    logger.success(f"✅ Removed {args.symbol} from configuration")

    if args.restart:
        logger.info("Restarting service...")
        service.restart()

    return 0


def cmd_enable(args, service: PaperTradingService):
    """Enable a symbol."""
    service.enable_symbol(args.symbol)
    logger.success(f"✅ Enabled {args.symbol}")
    return 0


def cmd_disable(args, service: PaperTradingService):
    """Disable a symbol."""
    service.disable_symbol(args.symbol)
    logger.success(f"✅ Disabled {args.symbol}")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Paper trading service control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start service
  python scripts/paper_trading_service.py start

  # Check status
  python scripts/paper_trading_service.py status

  # View logs
  python scripts/paper_trading_service.py logs

  # Add new symbol
  python scripts/paper_trading_service.py add DOGE/USDT --capital 5000

  # Remove symbol
  python scripts/paper_trading_service.py remove DOGE/USDT

  # Enable/disable symbol
  python scripts/paper_trading_service.py enable DOGE/USDT
  python scripts/paper_trading_service.py disable DOGE/USDT
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start command
    parser_start = subparsers.add_parser("start", help="Start the service")

    # Stop command
    parser_stop = subparsers.add_parser("stop", help="Stop the service")

    # Restart command
    parser_restart = subparsers.add_parser("restart", help="Restart the service")

    # Status command
    parser_status = subparsers.add_parser("status", help="Show service status")

    # Logs command
    parser_logs = subparsers.add_parser("logs", help="Show logs")
    parser_logs.add_argument("--symbol", help="Filter by symbol")
    parser_logs.add_argument("--tail", type=int, default=50, help="Number of lines (default: 50)")

    # Add command
    parser_add = subparsers.add_parser("add", help="Add a new symbol")
    parser_add.add_argument("symbol", help="Symbol to add (e.g., BTC/USDT)")
    parser_add.add_argument("--capital", type=float, default=10000, help="Starting capital")
    parser_add.add_argument("--oversold", type=int, default=25, help="RSI oversold level")
    parser_add.add_argument("--overbought", type=int, default=65, help="RSI overbought level")
    parser_add.add_argument("--interval", type=int, default=3600, help="Check interval (seconds)")
    parser_add.add_argument("--restart", action="store_true", help="Restart service after adding")

    # Remove command
    parser_remove = subparsers.add_parser("remove", help="Remove a symbol")
    parser_remove.add_argument("symbol", help="Symbol to remove")
    parser_remove.add_argument("--restart", action="store_true", help="Restart service after removing")

    # Enable command
    parser_enable = subparsers.add_parser("enable", help="Enable a symbol")
    parser_enable.add_argument("symbol", help="Symbol to enable")

    # Disable command
    parser_disable = subparsers.add_parser("disable", help="Disable a symbol")
    parser_disable.add_argument("symbol", help="Symbol to disable")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create service
    service = PaperTradingService()

    # Execute command
    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "status": cmd_status,
        "logs": cmd_logs,
        "add": cmd_add,
        "remove": cmd_remove,
        "enable": cmd_enable,
        "disable": cmd_disable,
    }

    command_func = commands.get(args.command)
    if command_func:
        return command_func(args, service)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
