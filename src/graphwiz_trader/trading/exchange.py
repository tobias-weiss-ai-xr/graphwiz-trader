"""
Exchange integration module.

Provides functions to create and configure exchange connections.
"""

import ccxt
from typing import Optional, Dict, Any
from loguru import logger


def create_exchange(
    exchange_name: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ccxt.Exchange:
    """Create an exchange instance.

    Args:
        exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
        api_key: API key for authenticated requests
        api_secret: API secret for authenticated requests
        config: Additional configuration options

    Returns:
        Configured exchange instance
    """
    # Get exchange class
    if not hasattr(ccxt, exchange_name):
        available = ", ".join(ccxt.exchanges)
        raise ValueError(
            f"Exchange '{exchange_name}' not found. "
            f"Available exchanges: {available}"
        )

    exchange_class = getattr(ccxt, exchange_name)

    # Base configuration
    exchange_config = {
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",  # Spot trading by default
        },
    }

    # Add API credentials if provided
    if api_key:
        exchange_config["apiKey"] = api_key
    if api_secret:
        exchange_config["secret"] = api_secret

    # Merge additional config
    if config:
        exchange_config.update(config)

    # Create exchange instance
    exchange = exchange_class(exchange_config)

    logger.info(f"Created {exchange_name} exchange connection")

    return exchange


def create_sandbox_exchange(
    exchange_name: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> ccxt.Exchange:
    """Create a sandbox/testnet exchange instance.

    Args:
        exchange_name: Name of the exchange
        api_key: Testnet API key
        api_secret: Testnet API secret

    Returns:
        Configured sandbox exchange instance
    """
    # Get exchange class
    if not hasattr(ccxt, exchange_name):
        raise ValueError(f"Exchange '{exchange_name}' not found")

    exchange_class = getattr(ccxt, exchange_name)

    # Sandbox configuration
    exchange_config = {
        "apiKey": api_key or "",
        "secret": api_secret or "",
        "enableRateLimit": True,
    }

    # Set sandbox mode for specific exchanges
    if exchange_name == "binance":
        exchange_config["options"] = {
            "defaultType": "spot",
            "sandboxMode": True,  # Binance testnet
        }
    elif exchange_name == "coinbase":
        exchange_config["sandboxMode"] = True
    elif exchange_name == "kraken":
        exchange_config["urls"] = {
            "api": {
                "public": "https://api.kraken.com/0/public",
                "private": "https://api.kraken.com/0/private",
            }
        }

    exchange = exchange_class(exchange_config)

    logger.info(f"Created {exchange_name} sandbox/testnet exchange")

    return exchange
