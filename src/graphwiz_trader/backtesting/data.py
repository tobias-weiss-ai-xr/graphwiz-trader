"""
Data manager for fetching and caching historical market data.

This module handles:
- Fetching historical data from CCXT exchanges
- Local caching for performance
- Multiple timeframe support
- Data validation and cleaning
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import ccxt
import pandas as pd
import numpy as np
from loguru import logger


class DataManager:
    """
    Manage historical market data fetching and caching.

    Supports multiple exchanges, timeframes, and data validation.
    """

    def __init__(
        self,
        cache_dir: str = "/opt/git/graphwiz-trader/data",
        exchange_name: str = "binance",
        enable_cache: bool = True,
    ):
        """
        Initialize DataManager.

        Args:
            cache_dir: Directory to cache downloaded data
            exchange_name: CCXT exchange name (default: binance)
            enable_cache: Whether to enable local caching
        """
        self.cache_dir = Path(cache_dir)
        self.exchange_name = exchange_name
        self.enable_cache = enable_cache

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })
            logger.info(f"Initialized {exchange_name} exchange")
        except AttributeError:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        # Supported timeframes
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

    def _get_cache_path(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Path:
        """Generate cache file path for data."""
        filename = (
            f"{self.exchange_name}_{symbol}_{timeframe}_"
            f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        )
        return self.cache_dir / filename

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for candles (e.g., '1h', '1d')
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of candles to fetch
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if timeframe not in self.timeframes:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {', '.join(self.timeframes)}"
            )

        # Default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Check cache
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
        if use_cache and self.enable_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Fetch data from exchange
        logger.info(
            f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}"
        )

        all_candles = []
        current_date = start_date
        total_days = (end_date - start_date).days

        while current_date < end_date:
            try:
                # Convert to milliseconds for CCXT
                since = int(current_date.timestamp() * 1000)

                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )

                if not candles:
                    logger.warning(f"No candles returned for {current_date}")
                    break

                all_candles.extend(candles)

                # Move to next batch
                last_timestamp = candles[-1][0]
                current_date = datetime.fromtimestamp(last_timestamp / 1000)

                # Rate limiting
                if len(all_candles) % 5000 == 0:
                    logger.info(f"Fetched {len(all_candles)} candles so far...")

                # Safety check
                if current_date >= end_date:
                    break

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

        if not all_candles:
            raise ValueError("No data fetched from exchange")

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        # Filter by date range
        df = df.loc[start_date:end_date]

        # Validate and clean
        df = self._validate_and_clean(df)

        # Cache the data
        if self.enable_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
            logger.info(f"Cached data to {cache_path}")

        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")

        # Check for zero values in price/volume
        for col in ["open", "high", "low", "close", "volume"]:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                logger.warning(f"Found {zero_count} zero values in {col}")
                # Replace with previous value
                df[col] = df[col].replace(0, np.nan).fillna(method="ffill")

        # Check for negative values
        if (df[["open", "high", "low", "close", "volume"]] < 0).any().any():
            raise ValueError("Found negative values in OHLCV data")

        # Check for price consistency
        invalid_high = df["high"] < df[["open", "low", "close"]].max(axis=1)
        invalid_low = df["low"] > df[["open", "high", "close"]].min(axis=1)

        if invalid_high.any() or invalid_low.any():
            logger.warning("Found inconsistent OHLC values, fixing...")

            # Fix high values
            df.loc[invalid_high, "high"] = df.loc[invalid_high, [
                "open", "low", "close"
            ]].max(axis=1)

            # Fix low values
            df.loc[invalid_low, "low"] = df.loc[invalid_low, [
                "open", "high", "close"
            ]].min(axis=1)

        # Detect and handle outliers (price changes > 50% in single candle)
        price_change = df["close"].pct_change().abs()
        outliers = price_change > 0.5
        if outliers.any():
            logger.warning(
                f"Found {outliers.sum()} potential outliers (>50% price change)"
            )

        return df

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe for candles
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(
                    symbol, timeframe, start_date, end_date, use_cache=True
                )
                data[symbol] = df
                logger.info(f"Successfully fetched {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        return data

    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.

        Args:
            df: Source DataFrame
            target_timeframe: Target timeframe (e.g., '1h', '1d')

        Returns:
            Resampled DataFrame
        """
        # Map timeframe to pandas offset alias
        timeframe_map = {
            "1m": "1T", "5m": "5T", "15m": "15T", "30m": "30T",
            "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W",
        }

        if target_timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")

        offset = timeframe_map[target_timeframe]

        resampled = df.resample(offset).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        # Remove any rows with NaN
        resampled = resampled.dropna()

        return resampled

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols from exchange."""
        try:
            markets = self.exchange.load_markets()
            symbols = [
                symbol for symbol, market in markets.items()
                if market.get("active", True) and market.get("type") == "spot"
            ]
            logger.info(f"Found {len(symbols)} active symbols")
            return symbols
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            return []

    def clear_cache(self, older_than_days: int = 7) -> None:
        """
        Clear cached data files.

        Args:
            older_than_days: Remove cache files older than this many days
        """
        cutoff_time = datetime.now() - timedelta(days=older_than_days)

        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time.timestamp():
                cache_file.unlink()
                logger.info(f"Removed old cache file: {cache_file}")
