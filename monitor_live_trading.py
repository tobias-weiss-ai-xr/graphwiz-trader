#!/usr/bin/env python3
"""
Live trading monitoring dashboard.

Provides real-time monitoring of live trading session including:
- Current positions and P&L
- Account balance
- Recent trades
- System health
- Risk metrics

Usage:
    python monitor_live_trading.py              # Interactive mode
    python monitor_live_trading.py --status     # Quick status
    python monitor_live_trading.py --watch      # Watch mode (auto-refresh)
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


class LiveTradingMonitor:
    """Monitor live trading session."""

    def __init__(self):
        """Initialize monitor."""
        self.log_dir = Path("logs/live_trading")
        self.pid_file = self.log_dir / "live_trading.pid"

    def is_running(self) -> bool:
        """Check if live trading is running."""
        if not self.pid_file.exists():
            return False

        try:
            pid = int(self.pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        status = {
            "running": self.is_running(),
            "timestamp": datetime.now().isoformat(),
            "pid": None,
            "uptime": None,
            "positions": [],
            "trades": [],
            "balance": {},
            "metrics": {}
        }

        if not status["running"]:
            return status

        # Read PID
        try:
            status["pid"] = int(self.pid_file.read_text().strip())
        except:
            pass

        # Get process uptime
        try:
            # Read creation time from /proc
            stat_file = Path(f"/proc/{status['pid']}/stat")
            if stat_file.exists():
                stat = stat_file.read_text()
                starttime = int(stat.split()[21])
                uptime_seconds = time.time() - (starttime / 100)
                status["uptime"] = str(timedelta(seconds=int(uptime_seconds)))
        except:
            pass

        # Read log files
        status.update(self._parse_logs())

        return status

    def _parse_logs(self) -> Dict[str, Any]:
        """Parse log files for trading data."""
        data = {
            "positions": [],
            "trades": [],
            "balance": {},
            "metrics": {}
        }

        # Parse trade history
        history_files = sorted(self.log_dir.glob("*_history_*.json"))

        if history_files:
            latest_history = history_files[-1]
            try:
                with open(latest_history, 'r') as f:
                    data["trades"] = json.load(f)
            except:
                pass

        # Parse summary
        summary_files = sorted(self.log_dir.glob("*_summary_*.json"))

        if summary_files:
            latest_summary = summary_files[-1]
            try:
                with open(latest_summary, 'r') as f:
                    summary = json.load(f)
                    data["balance"] = summary.get("balance", {})
                    data["positions"] = summary.get("positions", [])
                    data["metrics"] = summary.get("metrics", {})
            except:
                pass

        return data

    def print_status(self):
        """Print formatted status."""
        status = self.get_status()

        print("\n" + "=" * 80)
        print("📊 LIVE TRADING STATUS")
        print("=" * 80)
        print(f"Time: {status['timestamp']}")
        print()

        if status["running"]:
            print(f"✅ Status: RUNNING")
            print(f"   PID: {status['pid']}")
            print(f"   Uptime: {status['uptime'] or 'Unknown'}")
        else:
            print("❌ Status: NOT RUNNING")
            print()
            print("Start live trading:")
            print("  ./deploy_live_trading_germany.sh")
            print()
            return

        print()

        # Balance
        if status["balance"]:
            print("💰 Account Balance:")
            for currency, amount in status["balance"].items():
                if isinstance(amount, (int, float)):
                    print(f"   {currency}: {amount:,.2f}")
            print()

        # Positions
        if status["positions"]:
            print("📈 Open Positions:")
            for pos in status["positions"]:
                pnl = pos.get("pnl", 0)
                pnl_pct = pos.get("pnl_percent", 0)
                pnl_symbol = "🟢" if pnl >= 0 else "🔴"

                print(f"   {pnl_symbol} {pos['symbol']}")
                print(f"      Side: {pos['side']}")
                print(f"      Size: {pos['quantity']:.4f}")
                print(f"      Entry: €{pos['entry_price']:,.2f}")
                print(f"      P&L: €{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                print()
        else:
            print("📈 Open Positions: None")
            print()

        # Metrics
        if status["metrics"]:
            print("📊 Performance Metrics:")
            for key, value in status["metrics"].items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
            print()

        # Recent trades
        if status["trades"]:
            recent_trades = status["trades"][-5:]  # Last 5 trades
            print(f"📋 Recent Trades (showing last {len(recent_trades)}):")

            for trade in recent_trades:
                timestamp = trade.get("timestamp", "")
                action = trade.get("action", "").upper()
                symbol = trade.get("symbol", "")
                quantity = trade.get("quantity", 0)

                action_symbol = "🟢" if action == "BUY" else "🔴"

                print(f"   {action_symbol} {timestamp}")
                print(f"      {action} {quantity:.4f} {symbol}")
            print()

        print("=" * 80)
        print()

    def watch_mode(self, interval: int = 30):
        """Watch mode - auto-refresh status."""
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                self.print_status()

                print(f"🔄 Refreshing every {interval}s... (Ctrl+C to exit)")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")

    def show_logs(self, tail: int = 50):
        """Show recent log entries."""
        log_file = self.log_dir / "live_trading_output.log"

        if not log_file.exists():
            print("❌ No log file found")
            return

        print(f"\n📝 Recent {tail} log entries:")
        print("=" * 80)
        print()

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-tail:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"❌ Error reading logs: {e}")

        print()
        print("=" * 80)
        print()


def show_menu():
    """Show interactive menu."""
    print("\n" + "=" * 80)
    print("📊 LIVE TRADING MONITOR")
    print("=" * 80)
    print("\nSelect an option:")
    print()
    print("1) Show Status")
    print("2) Watch Mode (auto-refresh)")
    print("3) Show Logs")
    print("4) Exit")
    print()
    read_input = input("Choice [1-4]: ")

    return read_input.strip()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor live trading session"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show quick status"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode (auto-refresh)"
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Show log entries"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval for watch mode (default: 30s)"
    )

    args = parser.parse_args()

    monitor = LiveTradingMonitor()

    if args.status:
        monitor.print_status()
    elif args.watch:
        monitor.watch_mode(args.interval)
    elif args.logs:
        monitor.show_logs()
    else:
        # Interactive mode
        while True:
            choice = show_menu()

            if choice == "1":
                monitor.print_status()
            elif choice == "2":
                monitor.watch_mode()
            elif choice == "3":
                monitor.show_logs()
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("❌ Invalid choice")

            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)
