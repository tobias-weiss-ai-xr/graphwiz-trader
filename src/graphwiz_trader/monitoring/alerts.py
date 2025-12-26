"""
Alert management and notifications.

Supports:
- Telegram notifications
- Discord notifications
- Email notifications (future)
- Webhook notifications (future)
"""

import os
import requests
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from loguru import logger


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert notification."""

    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class Notifier(ABC):
    """Abstract base class for notifiers."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert.

        Args:
            alert: Alert to send

        Returns:
            True if successful, False otherwise
        """
        pass


class TelegramNotifier(Notifier):
    """Send alerts via Telegram bot."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token (or TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat ID (or TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not provided, notifications disabled")

    def send(self, alert: Alert) -> bool:
        """Send alert via Telegram.

        Args:
            alert: Alert to send

        Returns:
            True if successful
        """
        if not self.bot_token or not self.chat_id:
            return False

        try:
            # Format message
            emoji = self._get_emoji(alert.level)
            message = f"{emoji} *{alert.title}*\n\n"
            message += f"{alert.message}\n\n"

            if alert.metadata:
                message += "*Details:*\n"
                for key, value in alert.metadata.items():
                    message += f"• {key}: {value}\n"

            message += f"\n_ {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} _"

            # Send message
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }

            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()

            logger.debug(f"Telegram notification sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    def _get_emoji(self, level: AlertLevel) -> str:
        """Get emoji for alert level.

        Args:
            level: Alert level

        Returns:
            Emoji string
        """
        emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.SUCCESS: "✅",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        return emojis.get(level, "📢")


class DiscordNotifier(Notifier):
    """Send alerts via Discord webhook."""

    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL (or DISCORD_WEBHOOK_URL env var)
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

        if not self.webhook_url:
            logger.warning("Discord webhook URL not provided, notifications disabled")

    def send(self, alert: Alert) -> bool:
        """Send alert via Discord.

        Args:
            alert: Alert to send

        Returns:
            True if successful
        """
        if not self.webhook_url:
            return False

        try:
            # Color based on level
            color = self._get_color(alert.level)

            # Embed
            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": color,
                "timestamp": alert.timestamp.isoformat(),
                "fields": [],
            }

            # Add metadata as fields
            if alert.metadata:
                for key, value in alert.metadata.items():
                    embed["fields"].append({
                        "name": key,
                        "value": str(value),
                        "inline": True,
                    })

            # Send webhook
            data = {"embeds": [embed]}

            response = requests.post(self.webhook_url, json=data, timeout=10)
            response.raise_for_status()

            logger.debug(f"Discord notification sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def _get_color(self, level: AlertLevel) -> int:
        """Get color for alert level.

        Args:
            level: Alert level

        Returns:
            Color integer (decimal)
        """
        colors = {
            AlertLevel.INFO: 0x3498db,  # Blue
            AlertLevel.SUCCESS: 0x2ecc71,  # Green
            AlertLevel.WARNING: 0xf39c12,  # Orange
            AlertLevel.ERROR: 0xe74c3c,  # Red
            AlertLevel.CRITICAL: 0x8e44ad,  # Purple
        }
        return colors.get(level, 0x95a5a6)


class AlertManager:
    """Manages alert notifications across multiple channels."""

    def __init__(self):
        """Initialize alert manager."""
        self.notifiers: list[Notifier] = []
        self.alert_history: list[Alert] = []

        # Add notifiers if credentials available
        if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
            self.notifiers.append(TelegramNotifier())

        if os.getenv("DISCORD_WEBHOOK_URL"):
            self.notifiers.append(DiscordNotifier())

        logger.info(f"Initialized {len(self.notifiers)} notification channels")

    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send an alert to all configured channels.

        Args:
            level: Alert level
            title: Alert title
            message: Alert message
            metadata: Additional metadata

        Returns:
            True if sent to at least one channel
        """
        alert = Alert(level=level, title=title, message=message, metadata=metadata)
        self.alert_history.append(alert)

        # Limit history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        # Send to all notifiers
        success = False
        for notifier in self.notifiers:
            if notifier.send(alert):
                success = True

        # Always log
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.SUCCESS: logger.success,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.error,
        }.get(level, logger.info)

        log_func(f"{title}: {message}")

        return success

    def trade_executed(self, action: str, symbol: str, quantity: float, price: float):
        """Send trade executed notification.

        Args:
            action: Trade action ("buy" or "sell")
            symbol: Trading pair symbol
            quantity: Quantity traded
            price: Execution price
        """
        emoji = "🟢" if action == "buy" else "🔴"
        title = f"{emoji} Trade Executed: {action.upper()}"
        message = f"{action.upper()} {quantity:.4f} {symbol} @ ${price:,.2f}"

        self.send_alert(
            level=AlertLevel.SUCCESS,
            title=title,
            message=message,
            metadata={
                "Action": action.upper(),
                "Symbol": symbol,
                "Quantity": f"{quantity:.4f}",
                "Price": f"${price:,.2f}",
                "Value": f"${quantity * price:,.2f}",
            },
        )

    def position_closed(
        self, symbol: str, quantity: float, entry_price: float, exit_price: float, pnl: float
    ):
        """Send position closed notification.

        Args:
            symbol: Trading pair symbol
            quantity: Position quantity
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss
        """
        if pnl >= 0:
            emoji = "✅"
            level = AlertLevel.SUCCESS
        else:
            emoji = "❌"
            level = AlertLevel.ERROR

        title = f"{emoji} Position Closed: {symbol}"
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        message = (
            f"Closed {quantity:.4f} {symbol}\n"
            f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )

        self.send_alert(
            level=level,
            title=title,
            message=message,
            metadata={
                "Symbol": symbol,
                "Quantity": f"{quantity:.4f}",
                "Entry Price": f"${entry_price:,.2f}",
                "Exit Price": f"${exit_price:,.2f}",
                "P&L": f"${pnl:+,.2f}",
                "P&L %": f"{pnl_pct:+.2f}%",
            },
        )

    def daily_summary(self, portfolio_value: float, daily_pnl: float, trades_count: int):
        """Send daily summary notification.

        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily profit/loss
            trades_count: Number of trades today
        """
        emoji = "📊"
        title = f"{emoji} Daily Trading Summary"
        message = (
            f"Portfolio Value: ${portfolio_value:,.2f}\n"
            f"Daily P&L: ${daily_pnl:+,.2f}\n"
            f"Trades: {trades_count}"
        )

        level = AlertLevel.SUCCESS if daily_pnl >= 0 else AlertLevel.WARNING

        self.send_alert(
            level=level,
            title=title,
            message=message,
            metadata={
                "Portfolio Value": f"${portfolio_value:,.2f}",
                "Daily P&L": f"${daily_pnl:+,.2f}",
                "Daily P&L %": f"{(daily_pnl / portfolio_value) * 100:+.2f}%",
                "Trades": str(trades_count),
            },
        )

    def error_alert(self, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Send error alert.

        Args:
            error_message: Error message
            context: Additional context
        """
        self.send_alert(
            level=AlertLevel.ERROR,
            title="❌ Trading System Error",
            message=error_message,
            metadata=context or {},
        )

    def critical_alert(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Send critical alert.

        Args:
            message: Critical message
            context: Additional context
        """
        self.send_alert(
            level=AlertLevel.CRITICAL,
            title="🚨 CRITICAL ALERT",
            message=message,
            metadata=context or {},
        )
