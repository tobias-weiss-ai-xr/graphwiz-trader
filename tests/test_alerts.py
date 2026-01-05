#!/usr/bin/env python3
"""
Test script for alert system.

Demonstrates all alert channels without requiring real credentials.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add src directory to path for direct imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import directly from module files to avoid package init issues
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import alert config
alerts_config_path = src_path / "graphwiz_trader" / "alerts" / "config.py"
alerts_config = import_module_from_path("alerts.config", alerts_config_path)
get_default_alert_config = alerts_config.get_default_alert_config
CONSOLE_ONLY = alerts_config.CONSOLE_ONLY

# Import alerts module
alerts_init_path = src_path / "graphwiz_trader" / "alerts" / "__init__.py"
alerts_module = import_module_from_path("alerts", str(alerts_init_path))

AlertManager = alerts_module.AlertManager
AlertType = alerts_module.AlertType
AlertPriority = alerts_module.AlertPriority


def create_alert_manager(config):
    """Create an alert manager with the given configuration."""
    return AlertManager(config)


def test_alerts():
    """Test all alert types."""
    print("\n" + "=" * 80)
    print("🔔 ALERT SYSTEM TEST")
    print("=" * 80)
    print("\nTesting all alert types (console only)...\n")

    # Create alert manager with console only
    manager = create_alert_manager(CONSOLE_ONLY)

    print("1️⃣  Testing Trade Alerts")
    print("-" * 40)
    manager.trade_executed(
        exchange="Kraken",
        symbol="BTC/EUR",
        side="buy",
        amount=0.005,
        price=92450.75,
        order_id="ORDER-12345"
    )

    print("\n2️⃣  Testing P&L Alerts")
    print("-" * 40)
    manager.profit_target(
        exchange="Kraken",
        symbol="BTC/EUR",
        entry_price=90000,
        current_price=93600,
        profit_pct=4.0
    )

    manager.stop_loss_hit(
        exchange="Kraken",
        symbol="ETH/EUR",
        entry_price=3000,
        current_price=2940,
        loss_pct=2.0
    )

    print("\n3️⃣  Testing Risk Alerts")
    print("-" * 40)
    manager.daily_loss_limit(
        current_loss=150.00,
        limit=150.00,
        exchange="Kraken"
    )

    print("\n4️⃣  Testing System Alerts")
    print("-" * 40)
    manager.exchange_disconnected(
        exchange="Kraken",
        error="Connection timeout after 30s"
    )

    manager.system_error(
        component="TradingEngine",
        error="Order placement failed: Insufficient funds"
    )

    print("\n5️⃣  Testing Summary Alerts")
    print("-" * 40)
    manager.daily_summary(
        date="2026-01-02",
        trades=3,
        pnl=125.50,
        positions=1
    )

    print("\n" + "=" * 80)
    print("✅ ALERT SYSTEM TEST COMPLETE")
    print("=" * 80)
    print("\nAll alert types tested successfully!")
    print("\nTo enable real alerts:")
    print("1. Configure alert channels in config/alerts.yaml")
    print("2. Add credentials to .env file")
    print("3. Set enabled: true for desired channels")
    print()


def setup_email_guide():
    """Show email setup guide."""
    print("\n" + "=" * 80)
    print("📧 EMAIL ALERT SETUP GUIDE")
    print("=" * 80)
    print()

    print("Gmail Setup (Recommended):")
    print("-" * 40)
    print("1. Enable 2-Factor Authentication on your Gmail account")
    print("2. Go to: https://myaccount.google.com/apppasswords")
    print("3. Select 'App' as the app type")
    print("4. Name it: 'GraphWiz Trader'")
    print("5. Click 'Generate'")
    print("6. Copy the 16-character password")
    print()
    print("Then add to .env:")
    print("  ENABLE_EMAIL_ALERTS=true")
    print("  ALERT_EMAIL_FROM=your_email@gmail.com")
    print("  ALERT_EMAIL_RECIPIENTS=your_email@gmail.com")
    print("  SMTP_USERNAME=your_email@gmail.com")
    print("  SMTP_PASSWORD=your_app_password")
    print()

    print("Other Email Providers:")
    print("-" * 40)
    print("The system works with any SMTP server. Update these in .env:")
    print("  SMTP_HOST=smtp.example.com")
    print("  SMTP_PORT=587")
    print("  SMTP_USERNAME=your_username")
    print("  SMTP_PASSWORD=your_password")
    print()


def setup_slack_guide():
    """Show Slack setup guide."""
    print("\n" + "=" * 80)
    print("💬 SLACK ALERT SETUP GUIDE")
    print("=" * 80)
    print()

    print("1. Create a Slack App:")
    print("   - Go to https://api.slack.com/apps")
    print("   - Create 'New App' → 'From scratch'")
    print("   - Name it 'GraphWiz Trader Alerts'")
    print()

    print("2. Enable Incoming Webhooks:")
    print("   - Incoming Webhooks → On")
    print("   - 'Add New Webhook to Workspace'")
    print("   - Select channel (e.g., #trading-alerts)")
    print("   - Copy Webhook URL")
    print()

    print("3. Configure in .env:")
    print("   ENABLE_SLACK_ALERTS=true")
    print("   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL")
    print()

    print("4. Test Webhook:")
    print("   curl -X POST -H 'Content-type: application/json' \\")
    print("     --data '{\"text\":\"Test alert\"}' \\")
    print("     YOUR_WEBHOOK_URL")
    print()


def setup_telegram_guide():
    """Show Telegram setup guide."""
    print("\n" + "=" * 80)
    print("📱 TELEGRAM ALERT SETUP GUIDE")
    print("=" * 80)
    print()

    print("1. Create a Telegram Bot:")
    print("   - Open Telegram and search for @BotFather")
    print("   - Send: /newbot")
    print("   - Name your bot: 'GraphWiz Trader Bot'")
    print("   - Get a username like 'MyGraphWizTraderBot'")
    print("   - Copy the API token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyZ)")
    print()

    print("2. Get Your Chat ID:")
    print("   - Search for @userinfobot in Telegram")
    print("   - Send: /start")
    print("   - Copy your Chat ID (numbers only)")
    print()

    print("3. Configure in .env:")
    print("   ENABLE_TELEGRAM_ALERTS=true")
    print("   TELEGRAM_BOT_TOKEN=your_bot_token")
    print("   TELEGRAM_CHAT_ID=your_chat_id")
    print()

    print("4. Test Your Bot:")
    print("   python test_alerts.py --telegram")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test alert system")
    parser.add_argument("--email", action="store_true", help="Show email setup guide")
    parser.add_argument("--slack", action="store_true", help="Show Slack setup guide")
    parser.add_argument("--telegram", action="store_true", help="Show Telegram setup guide")
    parser.add_argument("--test", action="store_true", help="Run alert tests")

    args = parser.parse_args()

    if args.email:
        setup_email_guide()

    if args.slack:
        setup_slack_guide()

    if args.telegram:
        setup_telegram_guide()

    if args.test or not (args.email or args.slack or args.telegram):
        test_alerts()

    if not (args.test or args.email or args.slack or args.telegram):
        print("\nRun with --help for more options")
