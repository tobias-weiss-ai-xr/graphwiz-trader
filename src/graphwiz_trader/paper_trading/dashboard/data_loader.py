"""
Data loader module for paper trading dashboard.

Loads equity curves and summary statistics from CSV and JSON files.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


def get_data_dir() -> Path:
    """Get the data directory for paper trading results."""
    return Path("data/paper_trading")


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format for file lookups.

    Args:
        symbol: Symbol like 'BTC/USDT' or 'BTC_USDT'

    Returns:
        Normalized symbol with underscores (e.g., 'BTC_USDT')
    """
    return symbol.replace("/", "_")


def load_equity_curve(symbol: str) -> Optional[pd.DataFrame]:
    """Load equity curve data for a symbol.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')

    Returns:
        DataFrame with columns: timestamp, capital, position, position_value, total_value
        Returns None if file not found
    """
    data_dir = get_data_dir()
    norm_symbol = normalize_symbol(symbol)

    # Find the latest equity file for this symbol
    equity_files = sorted(
        data_dir.glob(f"{norm_symbol}_equity_*.csv"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not equity_files:
        logger.warning(f"No equity curve files found for {symbol}")
        return None

    latest_file = equity_files[0]
    logger.debug(f"Loading equity curve from {latest_file}")

    df = pd.read_csv(latest_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    return df


def load_summary(symbol: str) -> Optional[dict]:
    """Load summary statistics for a symbol.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')

    Returns:
        Dictionary with summary metrics
        Returns None if file not found
    """
    data_dir = get_data_dir()
    norm_symbol = normalize_symbol(symbol)

    # Find the latest summary file for this symbol
    summary_files = sorted(
        data_dir.glob(f"{norm_symbol}_summary_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not summary_files:
        logger.warning(f"No summary files found for {symbol}")
        return None

    latest_file = summary_files[0]
    logger.debug(f"Loading summary from {latest_file}")

    with open(latest_file, "r") as f:
        summary = json.load(f)

    return summary


def get_available_symbols() -> list[str]:
    """Get list of all available symbols with data.

    Returns:
        List of symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
    """
    data_dir = get_data_dir()

    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return []

    # Find all equity files and extract symbols
    equity_files = list(data_dir.glob("*_equity_*.csv"))
    symbols = set()

    for f in equity_files:
        # Extract symbol from filename (e.g., 'BTC_USDT_equity_20250126.csv' -> 'BTC_USDT')
        parts = f.stem.split("_equity_")[0]
        # Convert back to BTC/USDT format
        symbols.add(parts.replace("_", "/"))

    return sorted(list(symbols))


def load_all_symbols() -> dict[str, pd.DataFrame]:
    """Load equity curve data for all available symbols.

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    symbols = get_available_symbols()
    data = {}

    for symbol in symbols:
        df = load_equity_curve(symbol)
        if df is not None:
            data[symbol] = df

    return data


def get_symbol_config(symbol: str) -> Optional[dict]:
    """Get configuration for a symbol from the service config.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')

    Returns:
        Configuration dict or None if not found
    """
    config_file = Path("config/paper_trading.json")

    if not config_file.exists():
        return None

    with open(config_file, "r") as f:
        config = json.load(f)

    return config.get(symbol)
