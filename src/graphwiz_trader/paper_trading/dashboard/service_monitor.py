"""
Service monitor module for paper trading dashboard.

Monitors paper trading service status and retrieves logs.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from loguru import logger


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format for file lookups.

    Args:
        symbol: Symbol like 'BTC/USDT' or 'BTC_USDT'

    Returns:
        Normalized symbol with underscores (e.g., 'BTC_USDT')
    """
    return symbol.replace("/", "_")


def get_service_status() -> dict[str, dict]:
    """Get status of all paper trading service instances.

    Returns:
        Dictionary mapping symbol to status dict with keys:
        - running: bool
        - pid: Optional[int]
        - uptime_seconds: Optional[float]
        - uptime_str: str
        - memory_mb: Optional[float]
        - cpu_percent: Optional[float]
    """
    status = {}

    # Find all paper trading processes
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "memory_info", "cpu_percent"]):
        try:
            cmdline = proc.info["cmdline"]
            if not cmdline:
                continue

            cmdline_str = " ".join(cmdline)
            if "paper_trade.py" not in cmdline_str:
                continue

            # Extract symbol from command line
            symbol = None
            for i, arg in enumerate(cmdline):
                if "--symbol" in arg:
                    if i + 1 < len(cmdline):
                        symbol = cmdline[i + 1]
                        break

            if not symbol:
                continue

            # Calculate uptime
            create_time = proc.info.get("create_time")
            uptime_seconds = None
            uptime_str = "Unknown"

            if create_time:
                uptime_seconds = datetime.now().timestamp() - create_time
                uptime_str = format_uptime(uptime_seconds)

            # Get memory usage
            memory_info = proc.info.get("memory_info")
            memory_mb = None
            if memory_info:
                memory_mb = memory_info.rss / 1024 / 1024

            # Get CPU usage
            cpu_percent = proc.info.get("cpu_percent")

            status[symbol] = {
                "running": True,
                "pid": proc.info["pid"],
                "uptime_seconds": uptime_seconds,
                "uptime_str": uptime_str,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
            }

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Error accessing process: {e}")
            continue

    return status


def format_uptime(seconds: float) -> str:
    """Format uptime in seconds to human-readable string.

    Args:
        seconds: Uptime in seconds

    Returns:
        Formatted string (e.g., "2h 34m")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days}d {hours}h"


def get_latest_log_lines(
    symbol: str,
    n: int = 10,
) -> list[str]:
    """Get last N lines from symbol's log file.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        n: Number of lines to retrieve

    Returns:
        List of log lines
    """
    log_dir = Path("logs")
    norm_symbol = normalize_symbol(symbol)
    log_file = log_dir / f"{norm_symbol}.log"

    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        return []

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) >= n else lines
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return []


def get_log_summary(symbol: str) -> dict:
    """Get summary of log file for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Dictionary with:
        - total_lines: int
        - last_timestamp: Optional[str]
        - error_count: int
        - latest_entry: Optional[str]
    """
    log_dir = Path("logs")
    norm_symbol = normalize_symbol(symbol)
    log_file = log_dir / f"{norm_symbol}.log"

    if not log_file.exists():
        return {
            "total_lines": 0,
            "last_timestamp": None,
            "error_count": 0,
            "latest_entry": None,
        }

    try:
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Get last timestamp from log
        last_timestamp = None
        timestamp_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"

        for line in reversed(lines):
            match = re.search(timestamp_pattern, line)
            if match:
                last_timestamp = match.group(1)
                break

        # Count errors
        error_count = sum(1 for line in lines if "ERROR" in line.upper())

        # Get latest non-empty entry
        latest_entry = None
        for line in reversed(lines):
            if line.strip():
                latest_entry = line.strip()
                break

        return {
            "total_lines": len(lines),
            "last_timestamp": last_timestamp,
            "error_count": error_count,
            "latest_entry": latest_entry,
        }

    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return {
            "total_lines": 0,
            "last_timestamp": None,
            "error_count": 0,
            "latest_entry": None,
        }


def get_service_config() -> dict:
    """Get service configuration from config file.

    Returns:
        Dictionary of symbol configurations
    """
    import json

    config_file = Path("config/paper_trading.json")

    if not config_file.exists():
        return {}

    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def get_active_symbols() -> list[str]:
    """Get list of active (enabled) symbols from config.

    Returns:
        List of enabled symbols
    """
    config = get_service_config()

    return [symbol for symbol, cfg in config.items() if cfg.get("enabled", True)]


def is_service_running() -> bool:
    """Check if any paper trading service is running.

    Returns:
        True if at least one instance is running
    """
    status = get_service_status()
    return len(status) > 0
