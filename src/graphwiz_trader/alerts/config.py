"""
Alert configuration for automated notifications.

This module provides alert configuration templates for various notification channels.
"""

import os
from typing import Dict, Any, List


def get_default_alert_config() -> Dict[str, Any]:
    """Get default alert configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "enabled": True,
        "channels": {
            "console": {"enabled": True},
            "email": {
                "enabled": False,
                "from": "trader@example.com",
                "to": ["trader@example.com"],
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_username": "your_email@gmail.com",
                "smtp_password": "your_app_password",
            },
            "slack": {
                "enabled": False,
                "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            },
            "telegram": {
                "enabled": False,
                "bot_token": "your_bot_token",
                "chat_id": "your_chat_id",
            },
            "webhook": {"enabled": False, "url": "https://your-webhook-endpoint.com/alerts"},
        },
        "rules": {
            "trade_alerts": {
                "enabled": True,
                "alert_on_execution": True,
                "alert_on_fill": True,
                "alert_on_cancel": True,
                "alert_on_failure": True,
            },
            "profit_loss_alerts": {
                "enabled": True,
                "profit_target": 0.04,  # 4%
                "stop_loss": 0.02,  # 2%
                "daily_loss_limit": 150,  # €150
                "daily_profit_record": True,
            },
            "price_alerts": {"enabled": True, "threshold_change_pct": 0.05},  # 5%
            "system_alerts": {
                "enabled": True,
                "exchange_disconnected": True,
                "api_errors": True,
                "rate_limits": True,
                "system_errors": True,
            },
            "summary_alerts": {
                "enabled": True,
                "daily_summary": True,
                "daily_summary_time": "23:59",
                "weekly_summary": True,
                "weekly_summary_day": "Friday",
            },
        },
    }


def get_email_config(
    from_email: str,
    to_emails: List[str],
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    username: str = "",
    password: str = "",
) -> Dict[str, Any]:
    """Get email configuration.

    Args:
        from_email: Sender email
        to_emails: List of recipient emails
        smtp_host: SMTP server
        smtp_port: SMTP port
        username: SMTP username
        password: SMTP password

    Returns:
        Email configuration dict
    """
    return {
        "enabled": True,
        "from": from_email,
        "to": to_emails,
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "smtp_username": username or from_email,
        "smtp_password": password,
    }


def get_slack_config(webhook_url: str) -> Dict[str, Any]:
    """Get Slack configuration.

    Args:
        webhook_url: Slack webhook URL

    Returns:
        Slack configuration dict
    """
    return {"enabled": True, "webhook_url": webhook_url}


def get_telegram_config(bot_token: str, chat_id: str) -> Dict[str, Any]:
    """Get Telegram configuration.

    Args:
        bot_token: Telegram bot token
        chat_id: Telegram chat ID

    Returns:
        Telegram configuration dict
    """
    return {"enabled": True, "bot_token": bot_token, "chat_id": chat_id}


# Quick setup configurations
CONSOLE_ONLY = {"enabled": True, "channels": {"console": {"enabled": True}}}

EMAIL_ONLY = {
    "enabled": True,
    "channels": {
        "console": {"enabled": True},
        "email": {
            "enabled": True,
            "from": "trader@example.com",
            "to": ["trader@example.com"],
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_username": "your_email@gmail.com",
            "smtp_password": "your_app_password",
        },
    },
}

SLACK_ONLY = {
    "enabled": True,
    "channels": {
        "console": {"enabled": True},
        "slack": {"enabled": True, "webhook_url": os.getenv("SLACK_WEBHOOK_URL", "")},
    },
}

ALL_CHANNELS = {
    "enabled": True,
    "channels": {
        "console": {"enabled": True},
        "email": {
            "enabled": os.getenv("ENABLE_EMAIL_ALERTS", "false") == "true",
            "from": os.getenv("ALERT_EMAIL_FROM", "trader@example.com"),
            "to": os.getenv("ALERT_EMAIL_RECIPIENTS", "trader@example.com").split(","),
            "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "smtp_username": os.getenv("SMTP_USERNAME", ""),
            "smtp_password": os.getenv("SMTP_PASSWORD", ""),
        },
        "slack": {
            "enabled": os.getenv("ENABLE_SLACK_ALERTS", "false") == "true",
            "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
        },
        "telegram": {"enabled": False},  # Requires manual setup
        "webhook": {
            "enabled": os.getenv("ENABLE_WEBHOOK_ALERTS", "false") == "true",
            "url": os.getenv("WEBHOOK_URL", ""),
        },
    },
}
