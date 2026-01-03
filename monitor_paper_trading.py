#!/usr/bin/env python3
"""
GraphWiz Trader - Paper Trading Monitoring Script

Monitor running paper trading validation sessions.

Usage:
    python monitor_paper_trading.py
    python monitor_paper_trading.py --tail 50
    python monitor_paper_trading.py --report
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def find_latest_validation():
    """Find the latest validation report."""
    report_dir = Path("data/paper_trading")
    if not report_dir.exists():
        return None

    reports = sorted(report_dir.glob("validation_report_*.json"), reverse=True)
    return reports[0] if reports else None


def display_latest_report():
    """Display the latest validation report."""
    report_file = find_latest_report()

    if not report_file:
        print("No validation reports found.")
        print("Start a validation with: python run_extended_paper_trading.py")
        return

    with open(report_file, 'r') as f:
        report = json.load(f)

    print("\n" + "=" * 80)
    print("LATEST VALIDATION REPORT")
    print("=" * 80)
    print(f"Report File: {report_file.name}")
    print("")

    # Summary
    summary = report['validation_summary']
    print(f"Validation Summary:")
    print(f"  Start Time:      {summary['start_time']}")
    print(f"  End Time:        {summary.get('end_time', 'In progress')}")
    print(f"  Duration:        {summary['duration_hours']:.1f} / {summary['target_duration_hours']} hours")
    print(f"  Completion:      {summary['completion_pct']:.1f}%")
    print("")

    # Portfolio
    portfolio = report['portfolio']
    print(f"Portfolio:")
    print(f"  Initial Capital: ${portfolio['initial_capital']:,.2f}")
    print(f"  Final Value:     ${portfolio['final_value']:,.2f}")
    print(f"  Total Return:    ${portfolio['total_return_usd']:+,.2f} ({portfolio['total_return_pct']:+.2f}%)")
    print("")

    # Trading
    trading = report['trading']
    print(f"Trading:")
    print(f"  Total Trades:    {trading['total_trades']}")
    print(f"  Winning Trades:  {trading['winning_trades']}")
    print(f"  Losing Trades:   {trading['losing_trades']}")
    print(f"  Win Rate:        {trading['win_rate_pct']:.1f}%")
    print("")

    # Metrics
    metrics = report['metrics']
    print(f"Metrics:")
    print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print("")

    # Status
    status = report['status']
    status_emoji = {
        "EXCELLENT": "✅",
        "GOOD": "✅",
        "MODERATE": "⚠️",
        "POOR": "❌"
    }

    for key, emoji in status_emoji.items():
        if key in status:
            print(f"Status: {emoji} {status}")
            break

    print("=" * 80 + "\n")

    # Display current portfolio
    final_portfolio = metrics['final_portfolio']
    print("Current Holdings:")
    for asset, amount in final_portfolio.items():
        if amount > 0:
            print(f"  {asset}: {amount}")


def tail_log(lines=50):
    """Display the last N lines from the latest log file."""
    log_dir = Path("logs/paper_trading")
    if not log_dir.exists():
        print("No log files found.")
        return

    log_files = sorted(log_dir.glob("validation_*.log"), reverse=True)
    if not log_files:
        print("No validation logs found.")
        return

    latest_log = log_files[0]

    with open(latest_log, 'r') as f:
        all_lines = f.readlines()

    display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

    print(f"\n{'=' * 80}")
    print(f"LAST {len(display_lines)} LINES FROM {latest_log.name}")
    print("=" * 80 + "\n")

    for line in display_lines:
        print(line.rstrip())


def show_equity_curve():
    """Display equity curve data."""
    equity_dir = Path("logs/paper_trading")
    if not equity_dir.exists():
        print("No equity logs found.")
        return

    equity_files = sorted(equity_dir.glob("equity_*.csv"), reverse=True)
    if not equity_files:
        print("No equity curve logs found.")
        return

    latest_equity = equity_files[0]

    print(f"\n{'=' * 80}")
    print(f"EQUITY CURVE - {latest_equity.name}")
    print("=" * 80 + "\n")

    with open(latest_equity, 'r') as f:
        lines = f.readlines()

    if len(lines) <= 1:
        print("No equity data yet.")
        return

    # Display header and last 10 records
    print(lines[0].rstrip())  # Header

    for line in lines[-10:]:
        print(line.rstrip())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor paper trading validation")
    parser.add_argument('--report', action='store_true', help='Show latest report')
    parser.add_argument('--tail', type=int, default=0, help='Show last N lines from log')
    parser.add_argument('--equity', action='store_true', help='Show equity curve')

    args = parser.parse_args()

    if args.report:
        display_latest_report()
    elif args.tail > 0:
        tail_log(args.tail)
    elif args.equity:
        show_equity_curve()
    else:
        # Show all by default
        display_latest_report()
        print("\n")
        tail_log(20)


if __name__ == "__main__":
    main()
