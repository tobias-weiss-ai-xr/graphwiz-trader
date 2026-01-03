#!/usr/bin/env python3
"""
Live Trading Setup Validator

Comprehensive validation that checks:
- Configuration files
- API credentials
- Dependencies
- Account readiness
- Risk management settings
- System health

Run this before starting live trading to ensure everything is ready.

Usage:
    python validate_live_trading_setup.py
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class SetupValidator:
    """Validate live trading setup."""

    def __init__(self):
        """Initialize validator."""
        self.project_root = Path.cwd()
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []

    def log_pass(self, check_name: str, message: str = ""):
        """Log a passed check."""
        self.checks_passed.append((check_name, message))
        print(f"✅ {check_name}")
        if message:
            print(f"   {message}")

    def log_fail(self, check_name: str, message: str = ""):
        """Log a failed check."""
        self.checks_failed.append((check_name, message))
        print(f"❌ {check_name}")
        if message:
            print(f"   {message}")

    def log_warning(self, check_name: str, message: str = ""):
        """Log a warning."""
        self.warnings.append((check_name, message))
        print(f"⚠️  {check_name}")
        if message:
            print(f"   {message}")

    def check_project_structure(self) -> bool:
        """Check project structure."""
        print("\n📁 Checking Project Structure...")

        required_dirs = [
            "src/graphwiz_trader",
            "config",
            "logs",
            "scripts",
            "docs"
        ]

        all_exist = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.log_pass(f"Directory exists: {dir_path}")
            else:
                self.log_fail(f"Directory missing: {dir_path}")
                all_exist = False

        return all_exist

    def check_configuration_files(self) -> bool:
        """Check configuration files."""
        print("\n⚙️  Checking Configuration Files...")

        required_files = {
            "config/germany_live.yaml": "Germany live trading config",
            "deploy_live_trading_germany.sh": "Deployment script",
            "docs/LIVE_TRADING_GERMANY.md": "User documentation",
            ".env": "Environment variables"
        }

        all_exist = True
        for file_path, description in required_files.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_pass(f"{description}: {file_path}")
            else:
                self.log_fail(f"{description} missing: {file_path}")
                all_exist = False

        return all_exist

    def check_dependencies(self) -> bool:
        """Check Python dependencies."""
        print("\n📦 Checking Dependencies...")

        required_packages = [
            "ccxt",
            "loguru",
            "pandas",
            "pydantic",
            "python-dotenv"
        ]

        all_installed = True
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                self.log_pass(f"Package installed: {package}")
            except ImportError:
                self.log_fail(f"Package missing: {package}")
                all_installed = False

        return all_installed

    def check_env_file(self) -> bool:
        """Check .env file configuration."""
        print("\n🔐 Checking Environment Configuration...")

        env_file = self.project_root / ".env"
        if not env_file.exists():
            self.log_fail(".env file does not exist")
            return False

        from dotenv import load_dotenv
        load_dotenv()

        # Check Kraken credentials
        kraken_key = os.getenv("KRAKEN_API_KEY")
        kraken_secret = os.getenv("KRAKEN_API_SECRET")

        if not kraken_key or kraken_key == "your_kraken_api_key_here":
            self.log_fail("KRAKEN_API_KEY not configured", "Add your API key to .env")
            return False
        else:
            self.log_pass("KRAKEN_API_KEY configured")

        if not kraken_secret or kraken_secret == "your_kraken_api_secret_here":
            self.log_fail("KRAKEN_API_SECRET not configured", "Add your API secret to .env")
            return False
        else:
            self.log_pass("KRAKEN_API_SECRET configured")

        # Check Neo4j
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not neo4j_uri:
            self.log_warning("NEO4J_URI not set", "Knowledge graph features may be limited")
        else:
            self.log_pass("NEO4J_URI configured")

        if neo4j_password == "your_neo4j_password":
            self.log_warning("NEO4J_PASSWORD uses default", "Update with real password")
        else:
            self.log_pass("NEO4J_PASSWORD configured")

        return True

    def check_paper_trading_status(self) -> bool:
        """Check paper trading completion status."""
        print("\n📊 Checking Paper Trading Status...")

        log_dir = self.project_root / "logs" / "paper_trading"

        if not log_dir.exists():
            self.log_fail("Paper trading logs not found", "Run paper trading first")
            return False

        # Find most recent validation log
        validation_logs = sorted(log_dir.glob("validation_*.log"))

        if not validation_logs:
            self.log_fail("No paper trading validation logs found")
            return False

        latest_log = validation_logs[-1]

        # Check log contents
        try:
            with open(latest_log, 'r') as f:
                content = f.read()
                if "Elapsed:" in content:
                    # Extract elapsed time
                    for line in content.split('\n'):
                        if 'Elapsed:' in line:
                            self.log_pass("Paper trading validation run", line.strip())
                            break

                    # Check if completed (72 hours)
                    if "71." in content or "72." in content:
                        self.log_pass("Paper trading: COMPLETED ✅")
                        return True
                    else:
                        self.log_warning("Paper trading: INCOMPLETE",
                                      "Complete 72 hours before live trading")
                        return False
        except Exception as e:
            self.log_fail("Could not read paper trading logs", str(e))

        return False

    def check_risk_management(self) -> bool:
        """Check risk management configuration."""
        print("\n⚖️  Checking Risk Management...")

        config_file = self.project_root / "config" / "germany_live.yaml"

        if not config_file.exists():
            self.log_fail("Germany config file not found")
            return False

        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Check risk settings
            live_trading = config.get('live_trading', {})
            risk = config.get('risk', {})

            # Check position limits
            max_position = live_trading.get('max_position_eur', 0)
            if max_position > 0 and max_position <= 1000:
                self.log_pass(f"Max position: €{max_position} (conservative)")
            elif max_position > 1000:
                self.log_warning(f"Max position: €{max_position}",
                               "Consider starting with smaller amounts")
            else:
                self.log_fail("Max position not configured")
                return False

            # Check daily loss limit
            max_daily_loss = live_trading.get('max_daily_loss_eur', 0)
            if max_daily_loss > 0:
                self.log_pass(f"Max daily loss: €{max_daily_loss}")
            else:
                self.log_fail("Max daily loss not configured")
                return False

            # Check stop loss
            stop_loss = risk.get('stop_loss', {})
            if stop_loss.get('enabled'):
                stop_pct = stop_loss.get('default_percent', 0)
                self.log_pass(f"Stop loss: {stop_pct*100:.1f}%")
            else:
                self.log_warning("Stop loss not enabled")

            # Check take profit
            take_profit = risk.get('take_profit', {})
            if take_profit.get('enabled'):
                tp_pct = take_profit.get('default_percent', 0)
                self.log_pass(f"Take profit: {tp_pct*100:.1f}%")
            else:
                self.log_warning("Take profit not enabled")

            return True

        except Exception as e:
            self.log_fail("Could not read risk configuration", str(e))
            return False

    def check_regulatory_compliance(self) -> bool:
        """Check regulatory compliance."""
        print("\n⚖️  Checking Regulatory Compliance...")

        config_file = self.project_root / "config" / "germany_live.yaml"

        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            exchanges = config.get('exchanges', {})

            # Check that Kraken is enabled
            if 'kraken' in exchanges:
                if exchanges['kraken'].get('enabled'):
                    self.log_pass("Kraken: ENABLED ✅",
                                exchanges['kraken'].get('license_status', ''))
                else:
                    self.log_fail("Kraken: DISABLED")
                    return False
            else:
                self.log_fail("Kraken not configured")
                return False

            # Check that Binance is NOT enabled
            if 'binance' in exchanges and exchanges['binance'].get('enabled'):
                self.log_fail("BINANCE IS ENABLED ❌",
                            "Binance is NOT licensed in Germany!")
                return False
            else:
                self.log_pass("Binance: NOT ENABLED ✅",
                            "Correctly disabled for German users")

            return True

        except Exception as e:
            self.log_fail("Could not verify regulatory compliance", str(e))
            return False

    def check_system_resources(self) -> bool:
        """Check system resources."""
        print("\n💻 Checking System Resources...")

        import shutil
        import psutil

        # Check disk space
        disk = shutil.disk_usage(self.project_root)
        disk_free_gb = disk.free / (1024**3)

        if disk_free_gb >= 10:
            self.log_pass(f"Disk space: {disk_free_gb:.1f}GB free")
        else:
            self.log_warning(f"Low disk space: {disk_free_gb:.1f}GB free")

        # Check memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        if memory_percent < 80:
            self.log_pass(f"Memory: {100-memory_percent:.1f}% available")
        else:
            self.log_warning(f"High memory usage: {memory_percent:.1f}% used")

        return True

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("\n" + "=" * 80)
        print("🔍 LIVE TRADING SETUP VALIDATOR")
        print("=" * 80)
        print("\nChecking your system is ready for live trading...\n")

        # Run all checks
        self.check_project_structure()
        self.check_configuration_files()
        self.check_dependencies()
        self.check_env_file()
        self.check_paper_trading_status()
        self.check_risk_management()
        self.check_regulatory_compliance()
        self.check_system_resources()

        # Summary
        print("\n" + "=" * 80)
        print("📋 VALIDATION SUMMARY")
        print("=" * 80)
        print()

        total_passed = len(self.checks_passed)
        total_failed = len(self.checks_failed)
        total_warnings = len(self.warnings)

        print(f"✅ Passed:  {total_passed}")
        print(f"❌ Failed:  {total_failed}")
        print(f"⚠️  Warnings: {total_warnings}")
        print()

        if total_failed == 0:
            print("🎉 ALL CRITICAL CHECKS PASSED!")
            print()
            print("Your system is ready for live trading.")
            print()
            print("Next steps:")
            print("  1. Test Kraken connection:")
            print("     python test_kraken_connection.py")
            print()
            print("  2. Start live trading:")
            print("     ./deploy_live_trading_germany.sh")
            print()

            if total_warnings > 0:
                print("⚠️  Review warnings above before proceeding")
                print()

            return True
        else:
            print("❌ SETUP NOT READY")
            print()
            print("Please fix the failed checks above:")
            for check_name, message in self.checks_failed:
                print(f"  • {check_name}")
                if message:
                    print(f"    {message}")
            print()
            return False


def main():
    """Main entry point."""
    validator = SetupValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
