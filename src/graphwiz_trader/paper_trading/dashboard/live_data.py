"""
Live data fetcher for paper trading dashboard.

Fetches real-time prices and market data from Binance.
"""

import time
from typing import Optional

import ccxt
import pandas as pd
from loguru import logger


def get_current_price(symbol: str, exchange_id: str = "binance") -> Optional[float]:
    """Get current price for a symbol.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        exchange_id: Exchange ID (default: 'binance')

    Returns:
        Current price or None if fetch fails
    """
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker["last"])
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return None


def get_all_symbols_prices(symbols: list[str]) -> dict[str, float]:
    """Get current prices for multiple symbols.

    Args:
        symbols: List of trading symbols

    Returns:
        Dictionary mapping symbol to price
    """
    prices = {}

    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        tickers = exchange.fetch_tickers(symbols)

        for symbol in symbols:
            if symbol in tickers and tickers[symbol].get("last"):
                prices[symbol] = float(tickers[symbol]["last"])

    except Exception as e:
        logger.error(f"Error fetching prices: {e}")

    return prices


def get_24h_stats(symbol: str) -> dict:
    """Get 24-hour statistics for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Dictionary with 24h stats: high, low, volume, change
    """
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ticker = exchange.fetch_ticker(symbol)

        return {
            "high": float(ticker.get("high", 0)) or 0,
            "low": float(ticker.get("low", 0)) or 0,
            "volume": float(ticker.get("baseVolume", 0)) or 0,
            "change_pct": float(ticker.get("percentage", 0)) or 0,
            "quote_volume": float(ticker.get("quoteVolume", 0)) or 0,
        }
    except Exception as e:
        logger.error(f"Error fetching 24h stats for {symbol}: {e}")
        return {}


def fetch_recent_candles(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> Optional[pd.DataFrame]:
    """Fetch recent OHLCV candles for a symbol.

    Args:
        symbol: Trading symbol
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles to fetch

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        return df
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}")
        return None


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator.

    Args:
        prices: Series of prices
        period: RSI period

    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_market_summary() -> dict:
    """Get overall market summary for configured symbols.

    Returns:
        Dictionary with market stats
    """
    # Load configured symbols
    import json
    from pathlib import Path

    config_file = Path("config/paper_trading.json")
    if not config_file.exists():
        return {}

    with open(config_file, "r") as f:
        config = json.load(f)

    symbols = list(config.keys())

    # Get prices and stats
    prices = get_all_symbols_prices(symbols)
    summary = {}

    for symbol in symbols:
        stats = get_24h_stats(symbol)
        if symbol in prices:
            summary[symbol] = {
                "price": prices[symbol],
                "high_24h": stats.get("high", 0),
                "low_24h": stats.get("low", 0),
                "volume_24h": stats.get("quote_volume", 0),
                "change_24h_pct": stats.get("change_pct", 0),
            }

    return summary
