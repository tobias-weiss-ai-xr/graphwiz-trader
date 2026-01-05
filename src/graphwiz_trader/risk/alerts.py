"""Risk alerting and notification system.

This module provides alerting capabilities for risk threshold breaches,
drawdown warnings, exposure limit violations, and notification integration.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from loguru import logger
import asyncio
from collections import defaultdict


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of risk alerts."""

    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    EXPOSURE_LIMIT_EXCEEDED = "exposure_limit_exceeded"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_WARNING = "drawdown_warning"
    CORRELATION_RISK = "correlation_risk"
    CONCENTRATION_RISK = "concentration_risk"
    VOLATILITY_SPIKE = "volatility_spike"
    TRADING_LIMIT_EXCEEDED = "trading_limit_exceeded"
    MARGIN_CALL_WARNING = "margin_call_warning"
    PORTFOLIO_REBALANCE_NEEDED = "portfolio_rebalance_needed"


@dataclass
class Alert:
    """Risk alert data structure."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: Optional[str] = None
    metric_value: Optional[float] = None
    limit_value: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    notification_sent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "metric_value": self.metric_value,
            "limit_value": self.limit_value,
            "context": self.context,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "notification_sent": self.notification_sent,
        }


class NotificationChannel:
    """Base class for notification channels."""

    async def send(self, alert: Alert) -> bool:
        """Send alert notification.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully, False otherwise
        """
        raise NotImplementedError


class ConsoleNotificationChannel(NotificationChannel):
    """Console-based notification channel."""

    async def send(self, alert: Alert) -> bool:
        """Print alert to console."""
        timestamp_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            AlertSeverity.INFO: "ℹ️  INFO",
            AlertSeverity.WARNING: "⚠️  WARNING",
            AlertSeverity.CRITICAL: "🚨 CRITICAL",
            AlertSeverity.EMERGENCY: "🆘 EMERGENCY",
        }[alert.severity]

        logger.info(
            f"{prefix} [{timestamp_str}] {alert.message} "
            f"(Value: {alert.metric_value}, Limit: {alert.limit_value})"
        )
        return True


class DiscordNotificationChannel(NotificationChannel):
    """Discord webhook notification channel."""

    def __init__(self, webhook_url: str):
        """Initialize Discord notification channel.

        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        """Send alert to Discord webhook."""
        try:
            import aiohttp

            # Color based on severity
            colors = {
                AlertSeverity.INFO: 0x3498DB,  # Blue
                AlertSeverity.WARNING: 0xF39C12,  # Orange
                AlertSeverity.CRITICAL: 0xE74C3C,  # Red
                AlertSeverity.EMERGENCY: 0x8B0000,  # Dark red
            }

            # Build embed
            embed = {
                "title": f"{alert.severity.value.upper()}: {alert.alert_type.value.replace('_', ' ').title()}",
                "description": alert.message,
                "color": colors.get(alert.severity, 0x3498DB),
                "fields": [],
                "timestamp": alert.timestamp.isoformat(),
            }

            # Add fields if available
            if alert.symbol:
                embed["fields"].append({"name": "Symbol", "value": alert.symbol, "inline": True})

            if alert.metric_value is not None:
                embed["fields"].append(
                    {
                        "name": "Current Value",
                        "value": f"{alert.metric_value:.4f}",
                        "inline": True,
                    }
                )

            if alert.limit_value is not None:
                embed["fields"].append(
                    {"name": "Limit", "value": f"{alert.limit_value:.4f}", "inline": True}
                )

            # Add context fields
            for key, value in alert.context.items():
                embed["fields"].append({"name": key, "value": str(value), "inline": False})

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json={"embeds": [embed]},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error("Failed to send Discord notification: {}", e)
            return False


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_emails: List[str],
    ):
        """Initialize email notification channel.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password or app-specific password
            recipient_emails: List of recipient email addresses
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails

    async def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"[{alert.severity.value.upper()}] {alert.alert_type.value}"
            message["From"] = self.sender_email
            message["To"] = ", ".join(self.recipient_emails)

            # Build email body
            text_body = f"""
Risk Alert - {alert.alert_type.value}

Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Message: {alert.message}

"""

            if alert.symbol:
                text_body += f"Symbol: {alert.symbol}\n"

            if alert.metric_value is not None:
                text_body += f"Current Value: {alert.metric_value:.4f}\n"

            if alert.limit_value is not None:
                text_body += f"Limit: {alert.limit_value:.4f}\n"

            if alert.context:
                text_body += "\nAdditional Context:\n"
                for key, value in alert.context.items():
                    text_body += f"  {key}: {value}\n"

            # Attach text body
            text_part = MIMEText(text_body, "plain")
            message.attach(text_part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_emails, message.as_string())

            logger.info("Email notification sent for alert: {}", alert.alert_type.value)
            return True

        except Exception as e:
            logger.error("Failed to send email notification: {}", e)
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack webhook notification channel."""

    def __init__(self, webhook_url: str):
        """Initialize Slack notification channel.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack webhook."""
        try:
            import aiohttp

            # Color based on severity
            colors = {
                AlertSeverity.INFO: "#3498db",  # Blue
                AlertSeverity.WARNING: "#f39c12",  # Orange
                AlertSeverity.CRITICAL: "#e74c3c",  # Red
                AlertSeverity.EMERGENCY: "#8b0000",  # Dark red
            }

            # Build message
            attachment = {
                "color": colors.get(alert.severity, "#3498db"),
                "title": f"{alert.severity.value.upper()}: {alert.alert_type.value.replace('_', ' ').title()}",
                "text": alert.message,
                "fields": [],
                "ts": int(alert.timestamp.timestamp()),
            }

            # Add fields
            if alert.symbol:
                attachment["fields"].append(
                    {"title": "Symbol", "value": alert.symbol, "short": True}
                )

            if alert.metric_value is not None:
                attachment["fields"].append(
                    {"title": "Current Value", "value": f"{alert.metric_value:.4f}", "short": True}
                )

            if alert.limit_value is not None:
                attachment["fields"].append(
                    {"title": "Limit", "value": f"{alert.limit_value:.4f}", "short": True}
                )

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json={"attachments": [attachment]},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error("Failed to send Slack notification: {}", e)
            return False


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    enabled: bool = True
    cooldown_seconds: int = 300  # Minimum time between alerts for same metric


class RiskAlertManager:
    """Manages risk alerts and notifications."""

    def __init__(self, knowledge_graph=None):
        """Initialize risk alert manager.

        Args:
            knowledge_graph: Optional knowledge graph for storing alerts
        """
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.channels: List[NotificationChannel] = []
        self.thresholds: Dict[str, AlertThreshold] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        self.kg = knowledge_graph

        # Add console channel by default
        self.add_channel(ConsoleNotificationChannel())

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel.

        Args:
            channel: Notification channel to add
        """
        self.channels.append(channel)
        logger.info("Added notification channel: {}", type(channel).__name__)

    def set_threshold(self, threshold: AlertThreshold) -> None:
        """Set alert threshold for a metric.

        Args:
            threshold: Alert threshold configuration
        """
        self.thresholds[threshold.metric_name] = threshold
        logger.info(
            "Set alert threshold for {} - Warning: {:.2f}, Critical: {:.2f}",
            threshold.metric_name,
            threshold.warning_threshold,
            threshold.critical_threshold,
        )

    def register_handler(self, alert_type: AlertType, handler: Callable[[Alert], None]) -> None:
        """Register a handler for specific alert type.

        Args:
            alert_type: Type of alert to handle
            handler: Handler function
        """
        self.alert_handlers[alert_type].append(handler)
        logger.info("Registered handler for alert type: {}", alert_type.value)

    def check_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check metrics against thresholds and generate alerts.

        Args:
            metrics: Dictionary of metric names to values

        Returns:
            List of generated alerts
        """
        alerts_generated = []

        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue

            threshold = self.thresholds[metric_name]

            if not threshold.enabled:
                continue

            # Check cooldown
            last_alert_time = self.alert_cooldowns.get(metric_name)
            if last_alert_time and datetime.utcnow() - last_alert_time < timedelta(
                seconds=threshold.cooldown_seconds
            ):
                continue

            # Determine severity and generate alert
            severity = None
            if value >= threshold.emergency_threshold:
                severity = AlertSeverity.EMERGENCY
            elif value >= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif value >= threshold.warning_threshold:
                severity = AlertSeverity.WARNING

            if severity:
                alert = Alert(
                    alert_type=AlertType(f"{metric_name}_exceeded"),
                    severity=severity,
                    message=f"{metric_name} exceeded threshold: {value:.4f}",
                    metric_value=value,
                    limit_value=threshold.warning_threshold,
                )

                self.issue_alert(alert)
                alerts_generated.append(alert)

                # Update cooldown
                self.alert_cooldowns[metric_name] = datetime.utcnow()

        return alerts_generated

    def issue_alert(self, alert: Alert) -> None:
        """Issue a risk alert.

        Args:
            alert: Alert to issue
        """
        self.alerts.append(alert)
        self.alert_history.append(alert)

        logger.warning(
            "Alert issued: {} - {} (Severity: {})",
            alert.alert_type.value,
            alert.message,
            alert.severity.value,
        )

        # Store in knowledge graph if available
        if self.kg:
            self._store_alert_in_graph(alert)

        # Send notifications
        self._send_notifications(alert)

        # Call registered handlers
        for handler in self.alert_handlers.get(alert.alert_type, []):
            try:
                handler(alert)
            except Exception as e:
                logger.error("Error calling alert handler: {}", e)

    def _send_notifications(self, alert: Alert) -> None:
        """Send alert through all notification channels.

        Args:
            alert: Alert to send
        """
        for channel in self.channels:
            try:
                # Try async send first
                if hasattr(asyncio, "get_event_loop"):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(channel.send(alert))
                        else:
                            asyncio.run(channel.send(alert))
                    except RuntimeError:
                        # No event loop, run synchronously
                        asyncio.run(channel.send(alert))
                else:
                    asyncio.run(channel.send(alert))

                alert.notification_sent = True

            except Exception as e:
                logger.error("Failed to send notification via {}: {}", type(channel).__name__, e)

    def _store_alert_in_graph(self, alert: Alert) -> None:
        """Store alert in knowledge graph.

        Args:
            alert: Alert to store
        """
        try:
            cypher = """
            CREATE (a:RiskAlert {
                id: randomUUID(),
                alert_type: $alert_type,
                severity: $severity,
                message: $message,
                timestamp: datetime($timestamp),
                symbol: $symbol,
                metric_value: $metric_value,
                limit_value: $limit_value,
                acknowledged: $acknowledged,
                resolved: $resolved
            })
            """

            self.kg.write(
                cypher,
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                message=alert.message,
                timestamp=alert.timestamp.isoformat(),
                symbol=alert.symbol,
                metric_value=alert.metric_value,
                limit_value=alert.limit_value,
                acknowledged=alert.acknowledged,
                resolved=alert.resolved,
            )

            logger.debug("Stored alert in knowledge graph: {}", alert.alert_type.value)

        except Exception as e:
            logger.error("Failed to store alert in knowledge graph: {}", e)

    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert.

        Args:
            alert_index: Index of alert to acknowledge

        Returns:
            True if acknowledged successfully
        """
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].acknowledged = True
                logger.info("Acknowledged alert: {}", self.alerts[alert_index].alert_type.value)
                return True
        except Exception as e:
            logger.error("Failed to acknowledge alert: {}", e)

        return False

    def resolve_alert(self, alert_index: int) -> bool:
        """Mark an alert as resolved.

        Args:
            alert_index: Index of alert to resolve

        Returns:
            True if resolved successfully
        """
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].resolved = True
                logger.info("Resolved alert: {}", self.alerts[alert_index].alert_type.value)
                return True
        except Exception as e:
            logger.error("Failed to resolve alert: {}", e)

        return False

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active (unresolved) alerts.

        Args:
            severity: Optional severity filter

        Returns:
            List of active alerts
        """
        active_alerts = [a for a in self.alerts if not a.resolved]

        if severity:
            active_alerts = [a for a in active_alerts if a.severity == severity]

        return active_alerts

    def get_alert_history(
        self,
        hours: int = 24,
        alert_type: Optional[AlertType] = None,
    ) -> List[Alert]:
        """Get alert history for specified time period.

        Args:
            hours: Number of hours to look back
            alert_type: Optional alert type filter

        Returns:
            List of historical alerts
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        historical_alerts = [
            a for a in self.alert_history if a.timestamp >= cutoff_time and not a.resolved
        ]

        if alert_type:
            historical_alerts = [a for a in historical_alerts if a.alert_type == alert_type]

        return historical_alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics.

        Returns:
            Dictionary of alert statistics
        """
        total_alerts = len(self.alert_history)

        if total_alerts == 0:
            return {
                "total_alerts": 0,
                "active_alerts": 0,
                "by_severity": {},
                "by_type": {},
                "acknowledged_rate": 0.0,
            }

        active_alerts = len(self.get_active_alerts())
        acknowledged_alerts = sum(1 for a in self.alert_history if a.acknowledged)

        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for a in self.alert_history if a.severity == severity)
            by_severity[severity.value] = count

        by_type = {}
        for alert_type in AlertType:
            count = sum(1 for a in self.alert_history if a.alert_type == alert_type)
            by_type[alert_type.value] = count

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "by_severity": by_severity,
            "by_type": by_type,
            "acknowledged_rate": acknowledged_alerts / total_alerts,
        }
