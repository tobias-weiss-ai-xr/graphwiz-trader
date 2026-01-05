"""
Automated Alerts System for Live Trading.

Provides multi-channel alerting for:
- Trade execution notifications
- Price alerts and thresholds
- Loss/profit warnings
- System health monitoring
- Exchange connection status
- Risk limit breaches

Supported channels:
- Email
- Slack webhook
- Telegram bot
- Console/logging
- Custom webhooks
"""

import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class AlertPriority(str, Enum):
    """Alert priority levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of trading alerts."""

    # Trade alerts
    TRADE_EXECUTED = "trade_executed"
    TRADE_FAILED = "trade_failed"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"

    # Price alerts
    PRICE_THRESHOLD = "price_threshold"
    PRICE_TARGET_HIT = "price_target_hit"

    # P&L alerts
    PROFIT_TARGET = "profit_target"
    STOP_LOSS_HIT = "stop_loss_hit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DAILY_PROFIT_RECORD = "daily_profit_record"

    # Position alerts
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_SIZE_WARNING = "position_size_warning"

    # System alerts
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    SYSTEM_ERROR = "system_error"

    # Risk alerts
    RISK_LIMIT_BREACH = "risk_limit_breach"
    DRAWDOWN_WARNING = "drawdown_warning"
    MARGIN_CALL = "margin_call"

    # Status alerts
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"


@dataclass
class Alert:
    """Alert data structure."""

    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = ""
    symbol: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "type": self.alert_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "exchange": self.exchange,
            "symbol": self.symbol,
            "data": self.data,
        }


class AlertChannel:
    """Base class for alert channels."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize alert channel.

        Args:
            config: Channel configuration
        """
        self.config = config
        self.enabled = config.get("enabled", False)

    def send(self, alert: Alert) -> bool:
        """Send alert.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            return self._send(alert)
        except Exception as e:
            logger.error(f"Failed to send alert via {self.__class__.__name__}: {e}")
            return False

    def _send(self, alert: Alert) -> bool:
        """Send alert implementation.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        raise NotImplementedError


class ConsoleAlertChannel(AlertChannel):
    """Console logging alert channel."""

    def _send(self, alert: Alert) -> bool:
        """Send alert to console."""
        # Color coding based on priority
        colors = {
            AlertPriority.INFO: "\033[0;36m",  # Cyan
            AlertPriority.WARNING: "\033[1;33m",  # Yellow
            AlertPriority.ERROR: "\033[0;31m",  # Red
            AlertPriority.CRITICAL: "\033[1;31m",  # Bold Red
        }
        reset = "\033[0m"

        color = colors.get(alert.priority, "")
        icon = {
            AlertPriority.INFO: "ℹ️",
            AlertPriority.WARNING: "⚠️",
            AlertPriority.ERROR: "❌",
            AlertPriority.CRITICAL: "🚨",
        }.get(alert.priority, "•")

        # Format message
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        header = f"{color}[{alert.priority.value.upper()}]{reset} {icon} {timestamp}"

        print(f"\n{header}")
        print(f"{'─' * 80}")
        print(f"  {alert.title}")
        if alert.symbol:
            print(f"  Symbol: {alert.symbol} @ {alert.exchange}")
        print(f"  {alert.message}")
        if alert.data:
            for key, value in alert.data.items():
                print(f"  {key}: {value}")
        print(f"{'─' * 80}\n")

        return True


class EmailAlertChannel(AlertChannel):
    """Email alert channel."""

    def _send(self, alert: Alert) -> bool:
        """Send email alert."""
        try:
            # Compose email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.priority.value.upper()}] {alert.title}"
            msg["From"] = self.config["from"]
            msg["To"] = ", ".join(self.config["to"])

            # Plain text version
            text = self._format_text(alert)
            part1 = MIMEText(text, "plain")
            msg.attach(part1)

            # HTML version
            html = self._format_html(alert)
            part2 = MIMEText(html, "html")
            msg.attach(part2)

            # Send email
            with smtplib.SMTP(
                self.config["smtp_host"], self.config["smtp_port"], timeout=30
            ) as server:
                server.starttls()
                server.login(self.config["smtp_username"], self.config["smtp_password"])
                server.send_message(msg)

            logger.info(f"✅ Email alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _format_text(self, alert: Alert) -> str:
        """Format plain text email."""
        lines = [
            f"ALERT: {alert.title}",
            f"Priority: {alert.priority.value}",
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Exchange: {alert.exchange}",
            f"Symbol: {alert.symbol}",
            "",
            alert.message,
            "",
        ]

        if alert.data:
            lines.append("Details:")
            for key, value in alert.data.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("-" * 50)
        lines.append("GraphWiz Trader - Live Trading Alerts")

        return "\n".join(lines)

    def _format_html(self, alert: Alert) -> str:
        """Format HTML email."""
        # Color coding
        colors = {
            AlertPriority.INFO: "#0066cc",
            AlertPriority.WARNING: "#ff9900",
            AlertPriority.ERROR: "#cc0000",
            AlertPriority.CRITICAL: "#990000",
        }
        bg_color = colors.get(alert.priority, "#666666")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert-box {{
                    border-left: 5px solid {bg_color};
                    background-color: #f5f5f5;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .title {{ font-size: 18px; font-weight: bold; color: {bg_color}; }}
                .metadata {{ font-size: 12px; color: #666; }}
                .message {{ margin: 20px 0; }}
                .details {{ margin: 20px 0; }}
                .footer {{ font-size: 10px; color: #999; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="alert-box">
                <div class="title">{alert.title}</div>
                <div class="metadata">
                    Priority: {alert.priority.value.upper()} |
                    Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} |
                    Exchange: {alert.exchange} |
                    Symbol: {alert.symbol}
                </div>
                <div class="message">{alert.message}</div>
        """

        if alert.data:
            html += "<div class='details'><b>Details:</b><ul>"
            for key, value in alert.data.items():
                html += f"<li>{key}: {value}</li>"
            html += "</ul></div>"

        html += """
                <div class="footer">
                    GraphWiz Trader - Live Trading Alerts<br/>
                    <a href="https://github.com/your-repo/graphwiz-trader">View Project</a>
                </div>
            </div>
        </body>
        </html>
        """

        return html


class SlackAlertChannel(AlertChannel):
    """Slack webhook alert channel."""

    def _send(self, alert: Alert) -> bool:
        """Send Slack alert."""
        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            return False

        # Color coding
        colors = {
            AlertPriority.INFO: "36a64f",  # Blue
            AlertPriority.WARNING: "ff9900",  # Orange
            AlertPriority.ERROR: "cc0000",  # Red
            AlertPriority.CRITICAL: "990000",  # Dark Red
        }
        color = colors.get(alert.priority, "666666")

        # Build Slack message
        attachment = {
            "color": color,
            "title": alert.title,
            "text": alert.message,
            "fields": [
                {"title": "Priority", "value": alert.priority.value.upper(), "short": True},
                {
                    "title": "Time",
                    "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "short": True,
                },
                {"title": "Exchange", "value": alert.exchange, "short": True},
                {"title": "Symbol", "value": alert.symbol, "short": True},
            ],
            "footer": "GraphWiz Trader",
            "ts": int(alert.alert.timestamp.timestamp()),
        }

        # Add data fields
        if alert.data:
            for key, value in alert.data.items():
                attachment["fields"].append({"title": key, "value": str(value), "short": True})

        payload = {"attachments": [attachment]}

        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info(f"✅ Slack alert sent: {alert.title}")
        return True


class TelegramAlertChannel(AlertChannel):
    """Telegram bot alert channel."""

    def _send(self, alert: Alert) -> bool:
        """Send Telegram alert."""
        bot_token = self.config.get("bot_token")
        chat_id = self.config.get("chat_id")

        if not bot_token or not chat_id:
            return False

        # Build message
        emoji = {
            AlertPriority.INFO: "ℹ️",
            AlertPriority.WARNING: "⚠️",
            AlertPriority.ERROR: "❌",
            AlertPriority.CRITICAL: "🚨",
        }.get(alert.priority, "•")

        message = f"{emoji} *{alert.title}*\n\n"
        message += f"Priority: `{alert.priority.value.upper()}`\n"
        message += f"Time: `{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`\n"
        message += f"Exchange: `{alert.exchange}`\n"
        message += f"Symbol: `{alert.symbol}`\n\n"
        message += f"{alert.message}\n\n"

        if alert.data:
            message += "*Details:*\n"
            for key, value in alert.data.items():
                message += f"• `{key}`: {value}\n"

        # Send message
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info(f"✅ Telegram alert sent: {alert.title}")
        return True


class WebhookAlertChannel(AlertChannel):
    """Custom webhook alert channel."""

    def _send(self, alert: Alert) -> bool:
        """Send webhook alert."""
        url = self.config.get("url")
        if not url:
            return False

        payload = {"alert": alert.to_dict(), "timestamp": datetime.now().isoformat()}

        # Send POST request
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info(f"✅ Webhook alert sent: {alert.title}")
        return True


class AlertManager:
    """Main alerts management system."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config
        self.channels: List[AlertChannel] = []
        self.alert_history: List[Alert] = []
        self._initialize_channels()

        # Alert rules
        self.rules = config.get("rules", {})

        logger.info("✅ Alert Manager initialized")

    def _initialize_channels(self):
        """Initialize alert channels."""
        # Console channel (always enabled)
        self.channels.append(ConsoleAlertChannel({"enabled": True}))

        # Email channel
        if self.config.get("email", {}).get("enabled", False):
            self.channels.append(EmailAlertChannel(self.config["email"]))
            logger.info("✅ Email alerts enabled")

        # Slack channel
        if self.config.get("slack", {}).get("enabled", False):
            self.channels.append(SlackAlertChannel(self.config["slack"]))
            logger.info("✅ Slack alerts enabled")

        # Telegram channel
        if self.config.get("telegram", {}).get("enabled", False):
            self.channels.append(TelegramAlertChannel(self.config["telegram"]))
            logger.info("✅ Telegram alerts enabled")

        # Webhook channel
        if self.config.get("webhook", {}).get("enabled", False):
            self.channels.append(WebhookAlertChannel(self.config["webhook"]))
            logger.info("✅ Webhook alerts enabled")

    def send_alert(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.INFO,
        exchange: str = "",
        symbol: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send alert to all enabled channels.

        Args:
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            priority: Alert priority
            exchange: Exchange name
            symbol: Trading symbol
            data: Additional data

        Returns:
            True if sent successfully to at least one channel
        """
        alert = Alert(
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            exchange=exchange,
            symbol=symbol,
            data=data or {},
        )

        # Add to history
        self.alert_history.append(alert)

        # Keep only last 1000 alerts in memory
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        # Send to all enabled channels
        success = False
        for channel in self.channels:
            if channel.send(alert):
                success = True

        return success

    def trade_executed(
        self, exchange: str, symbol: str, side: str, amount: float, price: float, order_id: str
    ) -> bool:
        """Send trade executed alert.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: Buy/Sell
            amount: Amount traded
            price: Execution price
            order_id: Order ID

        Returns:
            True if sent successfully
        """
        return self.send_alert(
            alert_type=AlertType.TRADE_EXECUTED,
            title=f"Trade Executed: {side.upper()} {symbol}",
            message=f"Successfully {side.lower()} {amount:.6f} {symbol} @ €{price:,.2f}",
            priority=AlertPriority.INFO,
            exchange=exchange,
            symbol=symbol,
            data={
                "side": side.upper(),
                "amount": f"{amount:.6f}",
                "price": f"€{price:,.2f}",
                "value": f"€{amount * price:,.2f}",
                "order_id": order_id,
            },
        )

    def profit_target(
        self,
        exchange: str,
        symbol: str,
        entry_price: float,
        current_price: float,
        profit_pct: float,
    ) -> bool:
        """Send profit target alert.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            entry_price: Entry price
            current_price: Current price
            profit_pct: Profit percentage

        Returns:
            True if sent successfully
        """
        return self.send_alert(
            alert_type=AlertType.PROFIT_TARGET,
            title=f"✅ Profit Target Hit: {symbol}",
            message=f"Position in {symbol} has hit profit target at {profit_pct:.1f}%",
            priority=AlertPriority.INFO,
            exchange=exchange,
            symbol=symbol,
            data={
                "entry_price": f"€{entry_price:,.2f}",
                "current_price": f"€{current_price:,.2f}",
                "profit_percent": f"{profit_pct:.2f}%",
            },
        )

    def stop_loss_hit(
        self, exchange: str, symbol: str, entry_price: float, current_price: float, loss_pct: float
    ) -> bool:
        """Send stop loss alert.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            entry_price: Entry price
            current_price: Current price
            loss_pct: Loss percentage

        Returns:
            True if sent successfully
        """
        return self.send_alert(
            alert_type=AlertType.STOP_LOSS_HIT,
            title=f"🛑 Stop Loss Hit: {symbol}",
            message=f"Position in {symbol} has hit stop loss at {loss_pct:.1f}% loss",
            priority=AlertPriority.WARNING,
            exchange=exchange,
            symbol=symbol,
            data={
                "entry_price": f"€{entry_price:,.2f}",
                "current_price": f"€{current_price:,.2f}",
                "loss_percent": f"{loss_pct:.2f}%",
            },
        )

    def daily_loss_limit(self, current_loss: float, limit: float, exchange: str) -> bool:
        """Send daily loss limit alert.

        Args:
            current_loss: Current daily loss
            limit: Loss limit
            exchange: Exchange name

        Returns:
            True if sent successfully
        """
        return self.send_alert(
            alert_type=AlertType.DAILY_LOSS_LIMIT,
            title=f"⚠️ Daily Loss Limit: €{current_loss:,.2f}",
            message=f"Daily loss of €{current_loss:,.2f} has reached limit of €{limit:,.2f}. "
            f"Consider stopping trading for today.",
            priority=AlertPriority.WARNING,
            exchange=exchange,
            data={
                "current_loss": f"€{current_loss:,.2f}",
                "limit": f"€{limit:,.2f}",
                "utilization": f"{(current_loss/limit)*100:.1f}%",
            },
        )

    def exchange_disconnected(self, exchange: str, error: str) -> bool:
        """Send exchange disconnect alert.

        Args:
            exchange: Exchange name
            error: Error message

        Returns:
            True if sent successfully
        """
        return self.send_alert(
            alert_type=AlertType.EXCHANGE_DISCONNECTED,
            title=f"🔴 Exchange Disconnected: {exchange}",
            message=f"Lost connection to {exchange}. Attempting to reconnect...",
            priority=AlertPriority.ERROR,
            exchange=exchange,
            data={"error": error},
        )

    def system_error(self, component: str, error: str) -> bool:
        """Send system error alert.

        Args:
            component: Component name
            error: Error message

        Returns:
            True if sent successfully
        """
        return self.send_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            title=f"❌ System Error: {component}",
            message=f"Error in {component}: {error}",
            priority=AlertPriority.ERROR,
            data={"component": component, "error": error},
        )

    def daily_summary(self, date: str, trades: int, pnl: float, positions: int) -> bool:
        """Send daily summary alert.

        Args:
            date: Date string
            trades: Number of trades
            pnl: Daily P&L
            positions: Open positions

        Returns:
            True if sent successfully
        """
        pnl_symbol = "🟢" if pnl >= 0 else "🔴"

        return self.send_alert(
            alert_type=AlertType.DAILY_SUMMARY,
            title=f"📊 Daily Summary: {date}",
            message=f"Daily trading summary for {date}",
            priority=AlertPriority.INFO,
            data={
                "trades": trades,
                "pnl": f"{pnl_symbol} €{pnl:+,.2f}",
                "open_positions": positions,
            },
        )

    def position_size_warning(
        self, exchange: str, symbol: str, current_size: float, max_size: float
    ) -> bool:
        """Send position size warning alert.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            current_size: Current position size
            max_size: Maximum allowed position size

        Returns:
            True if sent successfully
        """
        utilization_pct = (current_size / max_size * 100) if max_size > 0 else 0

        return self.send_alert(
            alert_type=AlertType.POSITION_SIZE_WARNING,
            title=f"⚠️ Position Size Warning: {symbol}",
            message=f"Position size warning for {symbol} @ {exchange}",
            priority=AlertPriority.WARNING,
            exchange=exchange,
            symbol=symbol,
            data={
                "current_size": f"{current_size:.4f}",
                "max_size": f"{max_size:.4f}",
                "utilization": f"{utilization_pct:.1f}%",
            },
        )

    def trade_failed(
        self, exchange: str, symbol: str, side: str, amount: float, error: str
    ) -> bool:
        """Send trade failure alert.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            side: Trade side
            amount: Trade amount
            error: Error message

        Returns:
            True if sent successfully
        """
        return self.send_alert(
            alert_type=AlertType.TRADE_FAILED,
            title=f"❌ Trade Failed: {side.upper()} {symbol}",
            message=f"Trade failed for {symbol} @ {exchange}: {error}",
            priority=AlertPriority.ERROR,
            exchange=exchange,
            symbol=symbol,
            data={"side": side.upper(), "amount": f"{amount:.6f}", "error": error},
        )


def create_alert_manager(config: Dict[str, Any]) -> AlertManager:
    """Create alert manager from configuration.

    Args:
        config: Alert configuration

    Returns:
        AlertManager instance

    Example:
        config = {
            "email": {
                "enabled": True,
                "from": "trader@example.com",
                "to": ["trader@example.com"],
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_username": "your@email.com",
                "smtp_password": "your_password"
            },
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
            },
            "telegram": {
                "enabled": False,
                "bot_token": "your_bot_token",
                "chat_id": "your_chat_id"
            }
        }

        manager = create_alert_manager(config)
    """
    return AlertManager(config)
