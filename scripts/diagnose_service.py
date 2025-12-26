#!/usr/bin/env python3
"""
Paper trading service diagnostic tool.

Checks for potential issues:
- Process health
- Memory usage
- Log file sizes
- Network connectivity
- Zombie processes
- File descriptors
- Disk space
"""

import sys
import psutil
import subprocess
from pathlib import Path
from datetime import datetime

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def check(title, condition, message=""):
    """Print check result."""
    if condition:
        status = f"{Colors.GREEN}✓ OK{Colors.RESET}"
    else:
        status = f"{Colors.RED}✗ ISSUE{Colors.RESET}"

    print(f"{status}  {Colors.BOLD}{title}{Colors.RESET}")
    if message:
        print(f"     {message}")
    print()

def diagnose():
    """Run full diagnostic."""
    print("\n" + "="*80)
    print(f"{Colors.BOLD}PAPER TRADING SERVICE DIAGNOSTIC{Colors.RESET}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    issues = []

    # 1. Check running processes
    print(f"\n{Colors.BOLD}1. Process Health{Colors.RESET}\n")
    paper_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'paper_trade.py' in ' '.join(cmdline):
                paper_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if paper_procs:
        check(
            "Paper trading processes",
            len(paper_procs) > 0,
            f"Found {len(paper_procs)} running instances"
        )

        for proc in paper_procs:
            mem_mb = proc.info['memory_info'].rss / 1024 / 1024
            cpu = proc.info['cpu_percent']

            # Get symbol from cmdline
            symbol = "unknown"
            for arg in proc.info['cmdline']:
                if '--symbol' in arg:
                    idx = proc.info['cmdline'].index(arg)
                    if idx + 1 < len(proc.info['cmdline']):
                        symbol = proc.info['cmdline'][idx + 1]
                        break

            # Warn if memory too high
            if mem_mb > 500:
                issues.append(f"High memory usage for {symbol}: {mem_mb:.1f}MB")
                check(
                    f"  • {symbol} memory",
                    mem_mb < 500,
                    f"{mem_mb:.1f}MB (threshold: 500MB)"
                )
            else:
                check(f"  • {symbol} memory", mem_mb < 500, f"{mem_mb:.1f}MB")

            # Warn if CPU too high
            if cpu > 50:
                issues.append(f"High CPU usage for {symbol}: {cpu}%")
                check(f"  • {symbol} CPU", cpu < 50, f"{cpu}% (threshold: 50%)")
            else:
                check(f"  • {symbol} CPU", cpu < 50, f"{cpu}%")
    else:
        issues.append("No paper trading processes found")
        check("Paper trading processes", False, "No running instances found!")

    # 2. Check log file sizes
    print(f"{Colors.BOLD}2. Log File Health{Colors.RESET}\n")
    log_dir = Path("logs")
    total_log_size = 0

    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            size_kb = log_file.stat().st_size / 1024
            total_log_size += size_kb

            if size_kb > 1000:  # 1MB
                issues.append(f"Large log file: {log_file.name} ({size_kb:.1f}KB)")
                check(
                    f"  • {log_file.name}",
                    size_kb < 1000,
                    f"{size_kb:.1f}KB (threshold: 1MB)"
                )
            else:
                check(f"  • {log_file.name}", size_kb < 1000, f"{size_kb:.1f}KB")

        check(
            "Total log size",
            total_log_size < 5000,
            f"{total_log_size:.1f}KB (threshold: 5MB)"
        )
    else:
        check("Log directory", False, "logs/ directory not found")

    # 3. Check for zombie processes
    print(f"{Colors.BOLD}3. Zombie Processes{Colors.RESET}\n")
    zombie_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'status']):
        try:
            if proc.info['status'] == psutil.STATUS_ZOMBIE:
                zombie_count += 1
                issues.append(f"Zombie process: PID {proc.info['pid']} ({proc.info['name']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    check("Zombie processes", zombie_count == 0, f"Found {zombie_count} zombie processes")

    # 4. Check network connections
    print(f"{Colors.BOLD}4. Network Connectivity{Colors.RESET}\n")
    connections = []
    for conn in psutil.net_connections():
        if conn.status == 'ESTABLISHED' and conn.raddr:
            connections.append(conn)

    check(
        "Active network connections",
        len(connections) < 100,
        f"{len(connections)} connections (threshold: 100)"
    )

    # Check for connections to Binance
    binance_conns = 0
    for conn in connections:
        if conn.raddr and 'binance.com' in str(conn.raddr.ip):
            binance_conns += 1

    check(
        "Binance connections",
        binance_conns >= len(paper_procs),
        f"{binance_conns} connections to Binance (expected: {len(paper_procs)})"
    )

    # 5. Check disk space
    print(f"{Colors.BOLD}5. Disk Space{Colors.RESET}\n")
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
    total_gb = disk.total / (1024**3)
    used_pct = disk.percent

    if free_gb < 1:
        issues.append(f"Low disk space: {free_gb:.2f}GB free")
        check(
            "Free disk space",
            free_gb > 1,
            f"{free_gb:.2f}GB / {total_gb:.0f}GB ({used_pct:.1f}% used) - LOW!"
        )
    else:
        check(
            "Free disk space",
            free_gb > 1,
            f"{free_gb:.2f}GB / {total_gb:.0f}GB ({used_pct:.1f}% used)"
        )

    # 6. Check data files
    print(f"{Colors.BOLD}6. Data Output{Colors.RESET}\n")
    data_dir = Path("data/paper_trading")

    if data_dir.exists():
        equity_files = list(data_dir.glob("*_equity_*.csv"))
        summary_files = list(data_dir.glob("*_summary_*.json"))

        check(
            "Equity curve files",
            len(equity_files) > 0,
            f"{len(equity_files)} files found"
        )

        check(
            "Summary files",
            len(summary_files) > 0,
            f"{len(summary_files)} files found"
        )

        # Check for recent updates
        now = datetime.now()
        recent = []
        for f in equity_files + summary_files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            age_minutes = (now - mtime).total_seconds() / 60
            if age_minutes < 15:  # Updated in last 15 minutes
                recent.append(f.name)

        check(
            "Recent file updates",
            len(recent) >= len(paper_procs),
            f"{len(recent)} files updated in last 15 min (expected: {len(paper_procs)})"
        )
    else:
        check("Data directory", False, "data/paper_trading/ not found")

    # 7. Check configuration
    print(f"{Colors.BOLD}7. Configuration{Colors.RESET}\n")
    config_file = Path("config/paper_trading.json")

    if config_file.exists():
        check("Configuration file", True, f"{config_file}")
    else:
        issues.append("Configuration file missing")
        check("Configuration file", False, "config/paper_trading.json not found!")

    # 8. Check for errors in logs
    print(f"{Colors.BOLD}8. Error Detection{Colors.RESET}\n")
    error_count = 0
    for log_file in log_dir.glob("*.log"):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if 'ERROR' in line or 'Exception' in line or 'Traceback' in line:
                        error_count += 1
        except Exception:
            pass

    check("Errors in logs", error_count == 0, f"Found {error_count} error(s) in log files")

    # Summary
    print("\n" + "="*80)
    if issues:
        print(f"{Colors.RED}{Colors.BOLD}DIAGNOSTIC COMPLETE: {len(issues)} ISSUE(S) FOUND{Colors.RESET}")
        print("="*80 + "\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}DIAGNOSTIC COMPLETE: ALL CHECKS PASSED{Colors.RESET}")
        print("="*80 + "\n")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(diagnose())
