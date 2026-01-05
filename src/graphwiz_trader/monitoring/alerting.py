"""Multi-channel alerting system."""

import asyncio
import smtplib
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import hashlib
from loguru import logger

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("Websockets library not available. WebSocket notifications disabled.")


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertChannel(Enum):
    """Alert notification channels."""

    DISCORD = "discord"
    SLACK = "slack"
    EMAIL = "email"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert data structure."""

    title: str
    message: str
    severity: AlertSeverity
    channel: AlertChannel
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    alert_id: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def __post_init__(self):
        """Generate alert ID if not provided."""
        if self.alert_id is None:
            content = f"{self.title}:{self.message}:{self.timestamp.isoformat()}"
            self.alert_id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class AlertRule:
    """Alert rule definition."""

    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    channels: List[AlertChannel]
    message_template: str
    title_template: str
    cooldown_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class AlertManager:
    """Multi-channel alerting system with deduplication and cooldowns."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize alert manager.

        Args:
            config: Configuration dictionary with notification settings
        """
        self.config = config
        self.alert_history: List[Alert] = []
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self.last_alert_time: Dict[str, datetime] = {}
        self.cooldown_periods: Dict[str, int] = {}
        self.active_alerts: Dict[str, Alert] = {}

        # Circuit breaker state
        self.circuit_breaker_open: Dict[str, bool] = {}
        self.circuit_breaker_failures: Dict[str, int] = {}
        self.circuit_breaker_last_attempt: Dict[str, datetime] = {}

        # Load configuration
        self.notification_config = config.get("notifications", {})
        self.alert_config = config.get("alerts", {})

        # Initialize notification channels
        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize notification channels from configuration."""
        self.channels = {
            AlertChannel.DISCORD: self.notification_config.get("discord", {}).get("webhook_url"),
            AlertChannel.SLACK: self.notification_config.get("slack", {}).get("webhook_url"),
            AlertChannel.EMAIL: self.notification_config.get("email", {}),
            AlertChannel.TELEGRAM: self.notification_config.get("telegram", {}),
            AlertChannel.WEBHOOK: self.notification_config.get("webhook", {}).get("url"),
            AlertChannel.LOG: True,  # Always available
        }

        # Set default cooldowns
        self.default_cooldowns = {
            AlertSeverity.INFO: self.alert_config.get("cooldown_info", 600),
            AlertSeverity.WARNING: self.alert_config.get("cooldown_warning", 300),
            AlertSeverity.CRITICAL: self.alert_config.get("cooldown_critical", 60),
            AlertSeverity.EMERGENCY: self.alert_config.get("cooldown_emergency", 10),
        }

    def check_alert(
        self, rule_name: str, metrics: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """Check if an alert should be triggered based on rule and metrics.

        Args:
            rule_name: Name of the alert rule
            metrics: Current metrics dictionary
            context: Additional context for the alert

        Returns:
            Alert object if triggered, None otherwise
        """
        rule = self._get_rule(rule_name)
        if not rule or not rule.enabled:
            return None

        # Check if condition is met
        if not rule.condition(metrics):
            # Check if we should resolve an existing alert
            if rule_name in self.active_alerts:
                self.resolve_alert(rule_name)
            return None

        # Check cooldown
        if self._is_in_cooldown(rule_name, rule.cooldown_seconds):
            logger.debug("Alert {} is in cooldown, skipping", rule_name)
            return None

        # Check if alert already exists
        if rule_name in self.active_alerts:
            return None

        # Create alert
        alert = self._create_alert_from_rule(rule, metrics, context or {})
        self.active_alerts[rule_name] = alert

        # Send alert
        asyncio.create_task(self.send_alert(alert))

        return alert

    def _get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """Get alert rule by name.

        Args:
            rule_name: Name of the rule

        Returns:
            AlertRule if found, None otherwise
        """
        # Define built-in rules
        built_in_rules = self._get_builtin_rules()

        if rule_name in built_in_rules:
            return built_in_rules[rule_name]

        # Check for custom rules in config
        custom_rules = self.alert_config.get("custom_rules", {})
        if rule_name in custom_rules:
            rule_config = custom_rules[rule_name]
            return AlertRule(
                name=rule_name,
                condition=lambda m: self._evaluate_condition(rule_config["condition"], m),
                severity=AlertSeverity(rule_config.get("severity", "WARNING")),
                channels=[AlertChannel(c) for c in rule_config.get("channels", ["LOG"])],
                message_template=rule_config.get("message", "{rule_name} triggered"),
                title_template=rule_config.get("title", "{rule_name}"),
                cooldown_seconds=rule_config.get("cooldown_seconds", 300),
                metadata=rule_config.get("metadata", {}),
                enabled=rule_config.get("enabled", True),
            )

        return None

    def _get_builtin_rules(self) -> Dict[str, AlertRule]:
        """Get built-in alert rules.

        Returns:
            Dictionary of built-in AlertRules
        """
        return {
            "high_cpu": AlertRule(
                name="high_cpu",
                condition=lambda m: m.get("system", {}).get("cpu", {}).get("percent", 0) > 90,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DISCORD, AlertChannel.SLACK, AlertChannel.LOG],
                title_template="High CPU Usage Detected",
                message_template="CPU usage is {cpu_percent}% on {hostname}",
                cooldown_seconds=600,
            ),
            "high_memory": AlertRule(
                name="high_memory",
                condition=lambda m: m.get("system", {}).get("memory", {}).get("percent", 0) > 90,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DISCORD, AlertChannel.SLACK, AlertChannel.LOG],
                title_template="High Memory Usage Detected",
                message_template="Memory usage is {memory_percent}% on {hostname}",
                cooldown_seconds=600,
            ),
            "exchange_disconnect": AlertRule(
                name="exchange_disconnect",
                condition=lambda m: any(
                    not status.get("connected", True) for status in m.get("exchanges", {}).values()
                ),
                severity=AlertSeverity.CRITICAL,
                channels=[
                    AlertChannel.DISCORD,
                    AlertChannel.SLACK,
                    AlertChannel.EMAIL,
                    AlertChannel.LOG,
                ],
                title_template="Exchange Disconnected",
                message_template="Exchange {exchange_name} is disconnected",
                cooldown_seconds=60,
            ),
            "high_latency": AlertRule(
                name="high_latency",
                condition=lambda m: any(
                    status.get("p99_latency", 0) > 5.0 for status in m.get("exchanges", {}).values()
                ),
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DISCORD, AlertChannel.SLACK, AlertChannel.LOG],
                title_template="High Exchange Latency",
                message_template="Exchange {exchange_name} latency is {latency}s",
                cooldown_seconds=300,
            ),
            "large_loss": AlertRule(
                name="large_loss",
                condition=lambda m: m.get("trading", {}).get("current_drawdown", 0) > 0.10,
                severity=AlertSeverity.CRITICAL,
                channels=[
                    AlertChannel.DISCORD,
                    AlertChannel.SLACK,
                    AlertChannel.EMAIL,
                    AlertChannel.LOG,
                ],
                title_template="Large Trading Loss Detected",
                message_template="Current drawdown is {drawdown_percent}% (${loss_usd:.2f})",
                cooldown_seconds=300,
            ),
            "rate_limit": AlertRule(
                name="rate_limit",
                condition=lambda m: any(
                    status.get("rate_limit_remaining", 100) < 10
                    for status in m.get("exchanges", {}).values()
                ),
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DISCORD, AlertChannel.LOG],
                title_template="API Rate Limit Warning",
                message_template="Exchange {exchange_name} rate limit: {remaining} requests remaining",
                cooldown_seconds=300,
            ),
            "neo4j_disconnect": AlertRule(
                name="neo4j_disconnect",
                condition=lambda m: not m.get("neo4j", {}).get("connected", True),
                severity=AlertSeverity.CRITICAL,
                channels=[
                    AlertChannel.DISCORD,
                    AlertChannel.SLACK,
                    AlertChannel.EMAIL,
                    AlertChannel.LOG,
                ],
                title_template="Neo4j Disconnected",
                message_template="Neo4j database connection lost",
                cooldown_seconds=60,
            ),
            "agent_failure": AlertRule(
                name="agent_failure",
                condition=lambda m: m.get("agents", {}).get("failure_count", 0) > 5,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DISCORD, AlertChannel.LOG],
                title_template="Agent Failures Detected",
                message_template="{failure_count} agents have failed recently",
                cooldown_seconds=300,
            ),
            "risk_limit_breach": AlertRule(
                name="risk_limit_breach",
                condition=lambda m: m.get("risk", {}).get("leverage", 0) > 3.0,
                severity=AlertSeverity.EMERGENCY,
                channels=[
                    AlertChannel.DISCORD,
                    AlertChannel.SLACK,
                    AlertChannel.EMAIL,
                    AlertChannel.TELEGRAM,
                    AlertChannel.LOG,
                ],
                title_template="RISK LIMIT BREACH - EMERGENCY",
                message_template="Lverage at {leverage}x exceeds maximum. Exposure: ${exposure_usd:.2f}",
                cooldown_seconds=10,
            ),
            "disk_space": AlertRule(
                name="disk_space",
                condition=lambda m: any(
                    usage.get("percent", 0) > 90
                    for usage in m.get("system", {}).get("disk", {}).values()
                ),
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.DISCORD, AlertChannel.LOG],
                title_template="Low Disk Space",
                message_template="Disk {mount} is {percent}% full",
                cooldown_seconds=3600,
            ),
        }

    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate a condition string against metrics.

        Args:
            condition: Condition string (e.g., "system.cpu.percent > 90")
            metrics: Metrics dictionary

        Returns:
            True if condition is met, False otherwise
        """
        try:
            # Create a safe evaluation context
            context = {
                "system": metrics.get("system", {}),
                "trading": metrics.get("trading", {}),
                "exchanges": metrics.get("exchanges", {}),
                "agents": metrics.get("agents", {}),
                "risk": metrics.get("risk", {}),
                "neo4j": metrics.get("neo4j", {}),
            }
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error("Failed to evaluate condition '{}': {}", condition, e)
            return False

    def _is_in_cooldown(self, rule_name: str, cooldown_seconds: int) -> bool:
        """Check if alert is in cooldown period.

        Args:
            rule_name: Name of the alert rule
            cooldown_seconds: Cooldown period in seconds

        Returns:
            True if in cooldown, False otherwise
        """
        if rule_name not in self.last_alert_time:
            return False

        elapsed = (datetime.utcnow() - self.last_alert_time[rule_name]).total_seconds()
        return elapsed < cooldown_seconds

    def _create_alert_from_rule(
        self, rule: AlertRule, metrics: Dict[str, Any], context: Dict[str, Any]
    ) -> Alert:
        """Create Alert from AlertRule.

        Args:
            rule: AlertRule definition
            metrics: Current metrics
            context: Additional context

        Returns:
            Alert object
        """
        # Format templates with context
        format_dict = {
            **metrics,
            **context,
            "rule_name": rule.name,
            "timestamp": datetime.utcnow().isoformat(),
            "hostname": context.get("hostname", "unknown"),
        }

        # Extract specific values for templates
        if "system" in metrics:
            format_dict["cpu_percent"] = metrics["system"].get("cpu", {}).get("percent", 0)
            format_dict["memory_percent"] = metrics["system"].get("memory", {}).get("percent", 0)

        if "trading" in metrics:
            format_dict["drawdown_percent"] = metrics["trading"].get("current_drawdown", 0) * 100
            format_dict["loss_usd"] = abs(metrics["trading"].get("total_pnl", 0))

        if "exchanges" in metrics:
            for exchange_name, exchange_metrics in metrics["exchanges"].items():
                if not exchange_metrics.get("connected", True):
                    format_dict["exchange_name"] = exchange_name
                if exchange_metrics.get("p99_latency", 0) > 5.0:
                    format_dict["exchange_name"] = exchange_name
                    format_dict["latency"] = exchange_metrics["p99_latency"]
                if exchange_metrics.get("rate_limit_remaining", 100) < 10:
                    format_dict["exchange_name"] = exchange_name
                    format_dict["remaining"] = exchange_metrics["rate_limit_remaining"]

        if "risk" in metrics:
            format_dict["leverage"] = metrics["risk"].get("leverage", 0)
            format_dict["exposure_usd"] = metrics["risk"].get("exposure", 0)

        title = rule.title_template.format(**format_dict)
        message = rule.message_template.format(**format_dict)

        alert = Alert(
            title=title,
            message=message,
            severity=rule.severity,
            channel=rule.channels[0],
            metadata={
                "rule_name": rule.name,
                "metrics": metrics,
                "context": context,
                "channels": [c.value for c in rule.channels],
            },
        )

        self.last_alert_time[rule.name] = datetime.utcnow()
        return alert

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to configured channels.

        Args:
            alert: Alert object

        Returns:
            True if sent successfully, False otherwise
        """
        logger.info("Sending alert: {} - {}", alert.severity.value, alert.title)

        # Add to history
        self.alert_history.append(alert)
        self.alert_counts[alert.alert_id] += 1

        # Determine which channels to use
        channels = alert.metadata.get("channels", [])
        if not channels:
            channels = [alert.channel.value]

        success = False
        for channel_str in channels:
            try:
                channel = AlertChannel(channel_str)
                if await self._send_to_channel(alert, channel):
                    success = True
            except Exception as e:
                logger.error("Failed to send alert to {}: {}", channel_str, e)

        return success

    async def _send_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert to specific channel.

        Args:
            alert: Alert object
            channel: AlertChannel to send to

        Returns:
            True if successful, False otherwise
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open(channel.value):
            logger.warning("Circuit breaker is open for {}, skipping", channel.value)
            return False

        try:
            if channel == AlertChannel.DISCORD:
                return await self._send_discord(alert)
            elif channel == AlertChannel.SLACK:
                return await self._send_slack(alert)
            elif channel == AlertChannel.EMAIL:
                return await self._send_email(alert)
            elif channel == AlertChannel.TELEGRAM:
                return await self._send_telegram(alert)
            elif channel == AlertChannel.WEBHOOK:
                return await self._send_webhook(alert)
            elif channel == AlertChannel.LOG:
                return self._send_log(alert)
            else:
                logger.warning("Unknown channel: {}", channel.value)
                return False
        except Exception as e:
            logger.error("Failed to send to {}: {}", channel.value, e)
            self._record_circuit_breaker_failure(channel.value)
            return False

    def _is_circuit_breaker_open(self, channel: str) -> bool:
        """Check if circuit breaker is open for channel.

        Args:
            channel: Channel name

        Returns:
            True if open, False otherwise
        """
        return self.circuit_breaker_open.get(channel, False)

    def _record_circuit_breaker_failure(self, channel: str) -> None:
        """Record a failure for circuit breaker.

        Args:
            channel: Channel name
        """
        self.circuit_breaker_failures[channel] = self.circuit_breaker_failures.get(channel, 0) + 1
        self.circuit_breaker_last_attempt[channel] = datetime.utcnow()

        # Open circuit breaker after 5 consecutive failures
        if self.circuit_breaker_failures[channel] >= 5:
            self.circuit_breaker_open[channel] = True
            logger.warning(
                "Circuit breaker opened for {} after {} failures",
                channel,
                self.circuit_breaker_failures[channel],
            )

    async def _send_discord(self, alert: Alert) -> bool:
        """Send alert to Discord webhook.

        Args:
            alert: Alert object

        Returns:
            True if successful
        """
        webhook_url = self.channels.get(AlertChannel.DISCORD)
        if not webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        # Color based on severity
        colors = {
            AlertSeverity.INFO: 0x3498DB,  # Blue
            AlertSeverity.WARNING: 0xFF9500,  # Orange
            AlertSeverity.CRITICAL: 0xFF0000,  # Red
            AlertSeverity.EMERGENCY: 0x9B59B6,  # Purple
        }

        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": colors.get(alert.severity, 0x3498DB),
            "fields": [],
            "timestamp": alert.timestamp.isoformat(),
        }

        # Add metadata as fields
        if alert.metadata:
            for key, value in alert.metadata.items():
                if key not in ["metrics", "context"]:
                    embed["fields"].append({"name": key, "value": str(value), "inline": True})

        payload = {"embeds": [embed]}

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info("Alert sent to Discord successfully")
                    self._reset_circuit_breaker(AlertChannel.DISCORD.value)
                    return True
                else:
                    logger.error("Failed to send Discord alert: {}", response.status)
                    return False

    async def _send_slack(self, alert: Alert) -> bool:
        """Send alert to Slack webhook.

        Args:
            alert: Alert object

        Returns:
            True if successful
        """
        webhook_url = self.channels.get(AlertChannel.SLACK)
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        # Color based on severity
        colors = {
            AlertSeverity.INFO: "#3498db",
            AlertSeverity.WARNING: "#ff9500",
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.EMERGENCY: "#9b59b6",
        }

        attachment = {
            "color": colors.get(alert.severity, "#3498db"),
            "title": alert.title,
            "text": alert.message,
            "fields": [],
            "footer": f"Severity: {alert.severity.value}",
            "ts": int(alert.timestamp.timestamp()),
        }

        # Add metadata as fields
        if alert.metadata:
            for key, value in alert.metadata.items():
                if key not in ["metrics", "context"]:
                    attachment["fields"].append({"title": key, "value": str(value), "short": True})

        payload = {"attachments": [attachment]}

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info("Alert sent to Slack successfully")
                    self._reset_circuit_breaker(AlertChannel.SLACK.value)
                    return True
                else:
                    logger.error("Failed to send Slack alert: {}", response.status)
                    return False

    async def _send_email(self, alert: Alert) -> bool:
        """Send alert via email.

        Args:
            alert: Alert object

        Returns:
            True if successful
        """
        email_config = self.channels.get(AlertChannel.EMAIL)
        if not email_config or not email_config.get("enabled"):
            logger.warning("Email notifications not enabled")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value}] {alert.title}"
            msg["From"] = email_config.get("from")
            msg["To"] = ", ".join(email_config.get("to", []))

            # HTML body
            html = f"""
            <html>
              <body>
                <h2 style="color: {'red' if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else 'orange'}">
                  {alert.title}
                </h2>
                <p><strong>Severity:</strong> {alert.severity.value}</p>
                <p><strong>Time:</strong> {alert.timestamp.isoformat()}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <hr>
                <h3>Details</h3>
                <pre>{json.dumps(alert.metadata, indent=2)}</pre>
              </body>
            </html>
            """

            msg.attach(MIMEText(html, "html"))

            # Send email
            with smtplib.SMTP(
                email_config.get("smtp_server", "localhost"), email_config.get("smtp_port", 587)
            ) as server:
                if email_config.get("use_tls", True):
                    server.starttls()
                if email_config.get("username") and email_config.get("password"):
                    server.login(email_config["username"], email_config["password"])
                server.send_message(msg)

            logger.info("Alert sent via email successfully")
            self._reset_circuit_breaker(AlertChannel.EMAIL.value)
            return True

        except Exception as e:
            logger.error("Failed to send email alert: {}", e)
            self._record_circuit_breaker_failure(AlertChannel.EMAIL.value)
            return False

    async def _send_telegram(self, alert: Alert) -> bool:
        """Send alert via Telegram.

        Args:
            alert: Alert object

        Returns:
            True if successful
        """
        telegram_config = self.channels.get(AlertChannel.TELEGRAM)
        if not telegram_config or not telegram_config.get("bot_token"):
            logger.warning("Telegram not configured")
            return False

        bot_token = telegram_config["bot_token"]
        chat_id = telegram_config["chat_id"]

        # Emoji based on severity
        emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔴",
        }

        message = f"""
{emojis.get(alert.severity, '')} *{alert.title}*

Severity: {alert.severity.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}
        """.strip()

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Alert sent to Telegram successfully")
                    self._reset_circuit_breaker(AlertChannel.TELEGRAM.value)
                    return True
                else:
                    logger.error("Failed to send Telegram alert: {}", response.status)
                    return False

    async def _send_webhook(self, alert: Alert) -> bool:
        """Send alert to custom webhook.

        Args:
            alert: Alert object

        Returns:
            True if successful
        """
        webhook_url = self.channels.get(AlertChannel.WEBHOOK)
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return False

        payload = {
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.value,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info("Alert sent to webhook successfully")
                    self._reset_circuit_breaker(AlertChannel.WEBHOOK.value)
                    return True
                else:
                    logger.error("Failed to send webhook alert: {}", response.status)
                    return False

    def _send_log(self, alert: Alert) -> bool:
        """Send alert to log.

        Args:
            alert: Alert object

        Returns:
            True (always successful)
        """
        log_level = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(alert.severity, logger.info)

        log_level("{}: {}", alert.title, alert.message)
        return True

    def _reset_circuit_breaker(self, channel: str) -> None:
        """Reset circuit breaker for channel.

        Args:
            channel: Channel name
        """
        self.circuit_breaker_open[channel] = False
        self.circuit_breaker_failures[channel] = 0

    def resolve_alert(self, rule_name: str) -> None:
        """Mark an active alert as resolved.

        Args:
            rule_name: Name of the alert rule
        """
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()

            logger.info("Alert {} resolved", rule_name)
            del self.active_alerts[rule_name]

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts.

        Returns:
            List of active Alert objects
        """
        return list(self.active_alerts.values())

    def get_alert_history(
        self, hours: int = 24, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get alert history.

        Args:
            hours: Hours to look back
            severity: Filter by severity (optional)

        Returns:
            List of Alert objects
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        history = [alert for alert in self.alert_history if alert.timestamp > cutoff]

        if severity:
            history = [alert for alert in history if alert.severity == severity]

        return history

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics.

        Returns:
            Dictionary of alert statistics
        """
        stats = {
            "total_alerts": len(self.alert_history),
            "active_alerts": len(self.active_alerts),
            "by_severity": defaultdict(int),
            "by_channel": defaultdict(int),
            "last_24h": 0,
            "last_1h": 0,
        }

        cutoff_24h = datetime.utcnow() - timedelta(hours=24)
        cutoff_1h = datetime.utcnow() - timedelta(hours=1)

        for alert in self.alert_history:
            stats["by_severity"][alert.severity.value] += 1
            if alert.timestamp > cutoff_24h:
                stats["last_24h"] += 1
            if alert.timestamp > cutoff_1h:
                stats["last_1h"] += 1

        return dict(stats)
