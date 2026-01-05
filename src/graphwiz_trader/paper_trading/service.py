"""
Paper trading service manager.

Manages paper trading as a background service with:
- Start/stop/restart functionality
- Multiple symbol support
- Status monitoring
- Log management
"""

import os
import sys
import signal
import time
import json
import psutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess

from loguru import logger


@dataclass
class PaperTradingConfig:
    """Configuration for a paper trading instance."""

    symbol: str
    capital: float = 10000
    oversold: int = 25
    overbought: int = 65
    interval: int = 3600
    enabled: bool = True


class PaperTradingService:
    """Manages paper trading as a background service."""

    def __init__(self, config_dir: str = "config"):
        """Initialize service manager.

        Args:
            config_dir: Configuration directory
        """
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "paper_trading.json"
        self.pid_file = self.config_dir / "paper_trading.pid"
        self.log_dir = Path("logs")

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load or create default config
        self.configs: Dict[str, PaperTradingConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    for symbol, config_data in data.items():
                        self.configs[symbol] = PaperTradingConfig(**config_data)
                logger.info(f"Loaded {len(self.configs)} symbol configurations")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()

    def _create_default_config(self):
        """Create default configuration."""
        defaults = {
            "BTC/USDT": PaperTradingConfig(
                symbol="BTC/USDT", capital=10000, oversold=25, overbought=65
            ),
            "ETH/USDT": PaperTradingConfig(
                symbol="ETH/USDT", capital=10000, oversold=25, overbought=65
            ),
            "SOL/USDT": PaperTradingConfig(
                symbol="SOL/USDT", capital=10000, oversold=25, overbought=65
            ),
        }
        self.configs = defaults
        self._save_config()
        logger.info("Created default configuration")

    def _save_config(self):
        """Save configuration to file."""
        try:
            data = {symbol: asdict(config) for symbol, config in self.configs.items()}
            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.configs)} symbol configurations")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def add_symbol(self, config: PaperTradingConfig):
        """Add or update a symbol configuration.

        Args:
            config: PaperTradingConfig instance
        """
        self.configs[config.symbol] = config
        self._save_config()
        logger.info(f"Added/updated symbol: {config.symbol}")

    def remove_symbol(self, symbol: str):
        """Remove a symbol from configuration.

        Args:
            symbol: Symbol to remove
        """
        if symbol in self.configs:
            del self.configs[symbol]
            self._save_config()
            logger.info(f"Removed symbol: {symbol}")

    def start(self):
        """Start all paper trading services."""
        if self.is_running():
            logger.warning("Service is already running")
            return False

        logger.info("Starting paper trading service...")

        # Get enabled configs
        enabled_configs = [c for c in self.configs.values() if c.enabled]

        if not enabled_configs:
            logger.error("No enabled symbols found")
            return False

        # Start processes
        processes = []
        for config in enabled_configs:
            try:
                cmd = [
                    sys.executable,
                    "scripts/paper_trade.py",
                    "--symbol",
                    config.symbol,
                    "--capital",
                    str(config.capital),
                    "--oversold",
                    str(config.oversold),
                    "--overbought",
                    str(config.overbought),
                    "--continuous",
                    "--interval",
                    str(config.interval),
                ]

                # Redirect output to log file
                log_file = self.log_dir / f"{config.symbol.replace('/', '_')}.log"

                with open(log_file, "a") as log:
                    process = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    processes.append(process)

                logger.info(f"Started {config.symbol} (PID: {process.pid})")

            except Exception as e:
                logger.error(f"Failed to start {config.symbol}: {e}")

        if processes:
            # Save PID of parent process
            with open(self.pid_file, "w") as f:
                f.write(str(os.getpid()))

            logger.success(f"Started {len(processes)} paper trading instances")
            return True
        else:
            logger.error("Failed to start any instances")
            return False

    def stop(self):
        """Stop all paper trading services."""
        if not self.is_running():
            logger.warning("Service is not running")
            return False

        logger.info("Stopping paper trading service...")

        # Find all paper_trade.py processes
        stopped = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and "paper_trade.py" in " ".join(cmdline):
                    proc.terminate()
                    stopped += 1
                    logger.info(f"Stopped process {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Remove PID file
        if self.pid_file.exists():
            self.pid_file.unlink()

        if stopped > 0:
            logger.success(f"Stopped {stopped} paper trading instances")
            return True
        else:
            logger.warning("No running instances found")
            return False

    def restart(self):
        """Restart all paper trading services."""
        logger.info("Restarting paper trading service...")
        self.stop()
        time.sleep(2)
        return self.start()

    def is_running(self) -> bool:
        """Check if service is running.

        Returns:
            True if any instances are running
        """
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and "paper_trade.py" in " ".join(cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False

    def status(self) -> Dict:
        """Get service status.

        Returns:
            Status dictionary
        """
        running = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and "paper_trade.py" in " ".join(cmdline):
                    # Extract symbol from command line
                    symbol = "unknown"
                    for arg in cmdline:
                        if "--symbol" in arg:
                            idx = cmdline.index(arg)
                            if idx + 1 < len(cmdline):
                                symbol = cmdline[idx + 1]
                                break

                    running.append(
                        {
                            "symbol": symbol,
                            "pid": proc.info["pid"],
                            "uptime": time.time() - proc.info["create_time"],
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return {
            "running": len(running) > 0,
            "instances": running,
            "configured_symbols": list(self.configs.keys()),
            "enabled_symbols": [s for s, c in self.configs.items() if c.enabled],
        }

    def logs(self, symbol: Optional[str] = None, tail: int = 50):
        """Get recent logs.

        Args:
            symbol: Symbol to get logs for (None = all)
            tail: Number of lines to show

        Returns:
            Log lines
        """
        log_files = []

        if symbol:
            log_file = self.log_dir / f"{symbol.replace('/', '_')}.log"
            if log_file.exists():
                log_files.append(log_file)
        else:
            log_files = list(self.log_dir.glob("*.log"))

        logs = []
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if tail:
                        lines = lines[-tail:]
                    logs.extend(lines)
            except Exception as e:
                logger.error(f"Failed to read {log_file}: {e}")

        return logs

    def enable_symbol(self, symbol: str):
        """Enable a symbol.

        Args:
            symbol: Symbol to enable
        """
        if symbol in self.configs:
            self.configs[symbol].enabled = True
            self._save_config()
            logger.info(f"Enabled {symbol}")

    def disable_symbol(self, symbol: str):
        """Disable a symbol.

        Args:
            symbol: Symbol to disable
        """
        if symbol in self.configs:
            self.configs[symbol].enabled = False
            self._save_config()
            logger.info(f"Disabled {symbol}")

    def print_status(self):
        """Print formatted status."""
        status = self.status()

        print("\n" + "=" * 80)
        print("PAPER TRADING SERVICE STATUS")
        print("=" * 80)

        if status["running"]:
            print(f"Status: ✅ Running ({len(status['instances'])} instances)")
            print("\nActive Instances:")
            print("-" * 80)
            print(f"{'Symbol':<20} {'PID':<10} {'Uptime':<15}")
            print("-" * 80)
            for instance in status["instances"]:
                uptime_mins = int(instance["uptime"] / 60)
                print(f"{instance['symbol']:<20} {instance['pid']:<10} {uptime_mins:>6} min")
        else:
            print("Status: ❌ Stopped")

        print("\n" + "-" * 80)
        print("Configured Symbols:")
        print("-" * 80)
        print(f"{'Symbol':<20} {'Capital':<12} {'RSI':<12} {'Status':<10}")
        print("-" * 80)

        for symbol in status["configured_symbols"]:
            config = self.configs[symbol]
            enabled = "✅ Enabled" if config.enabled else "❌ Disabled"
            rsi = f"{config.oversold}/{config.overbought}"
            print(f"{config.symbol:<20} ${config.capital:<11,.0f} {rsi:<12} {enabled:<10}")

        print("=" * 80 + "\n")
