"""Data adapter for CCXT to Qlib integration."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import ccxt.async_support as ccxt

from .config import QlibConfig


class QlibDataAdapter:
    """
    Bridge between CCXT data sources and Qlib data layer.

    This adapter fetches market data from cryptocurrency exchanges via CCXT
    and converts it to the format expected by Qlib for feature extraction
    and model training.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        config: Optional[QlibConfig] = None,
    ):
        """
        Initialize the data adapter.

        Args:
            exchange_id: CCXT exchange ID (e.g., 'binance', 'okx')
            config: Qlib configuration
        """
        self.exchange_id = exchange_id
        self.config = config or QlibConfig()

        # Initialize CCXT exchange
        self.exchange: Optional[ccxt.Exchange] = None

    async def initialize(self):
        """Initialize CCXT exchange connection."""
        if self.exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
            logger.info(f"Initialized CCXT exchange: {self.exchange_id}")

    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
            logger.info(f"Closed CCXT exchange: {self.exchange_id}")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        since: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d', '5m')
            limit: Number of candles to fetch
            since: Start time for data fetch

        Returns:
            DataFrame with OHLCV data in Qlib format
        """
        await self.initialize()

        try:
            # Convert since to timestamp if provided
            since_timestamp = None
            if since:
                since_timestamp = int(since.timestamp() * 1000)

            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_timestamp,
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Set timestamp as index
            df.set_index("timestamp", inplace=True)

            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe for data
            limit: Number of candles per symbol

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}

        # Fetch data for each symbol
        for symbol in symbols:
            try:
                df = await self.fetch_ohlcv(symbol, timeframe, limit)
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")

        return results

    def to_qlib_format(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Convert CCXT data to Qlib format.

        Qlib expects data with specific column names and structure.
        This method transforms the data accordingly.

        Args:
            df: DataFrame with CCXT OHLCV data
            symbol: Symbol identifier

        Returns:
            DataFrame in Qlib format
        """
        # Ensure we have the required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")

        # Create a copy to avoid modifying original
        qlib_df = df.copy()

        # Rename columns to Qlib format (if needed)
        # Qlib typically uses: open, high, low, close, volume, vwap, amount
        qlib_df.columns = [col.lower() for col in qlib_df.columns]

        # Add instrument column (required by Qlib)
        qlib_df["instrument"] = symbol

        # Reset index to make timestamp a column
        qlib_df.reset_index(inplace=True)
        qlib_df.rename(columns={"timestamp": "datetime"}, inplace=True)

        # Ensure proper sorting
        qlib_df.sort_values("datetime", inplace=True)

        # Remove any duplicate timestamps
        qlib_df.drop_duplicates(subset=["datetime", "instrument"], inplace=True)

        logger.debug(f"Converted {len(qlib_df)} rows to Qlib format for {symbol}")

        return qlib_df

    async def prepare_training_data(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Prepare training data for Qlib models.

        Fetches historical data for multiple symbols and combines them
        into a single DataFrame suitable for training.

        Args:
            symbols: List of symbols to fetch
            timeframe: Data timeframe
            lookback_days: Number of days of historical data to fetch

        Returns:
            Combined DataFrame with all symbols
        """
        # Calculate start time
        since = datetime.now() - timedelta(days=lookback_days)

        # Fetch data for all symbols
        data_dict = await self.fetch_multiple_symbols(
            symbols,
            timeframe=timeframe,
            limit=int(lookback_days * 24),  # Approximate candles needed
        )

        # Convert to Qlib format and combine
        qlib_dfs = []
        for symbol, df in data_dict.items():
            qlib_df = self.to_qlib_format(df, symbol)
            qlib_dfs.append(qlib_df)

        # Combine all DataFrames
        if qlib_dfs:
            combined_df = pd.concat(qlib_dfs, ignore_index=True)
            logger.info(
                f"Prepared training data: {len(combined_df)} rows for {len(qlib_dfs)} symbols"
            )
            return combined_df
        else:
            logger.warning("No data fetched for training")
            return pd.DataFrame()

    async def get_latest_data(
        self,
        symbol: str,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """
        Get the most recent market data for a symbol.

        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe

        Returns:
            DataFrame with latest data
        """
        return await self.fetch_ohlcv(symbol, timeframe, limit=100)

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """
        Get historical data for a date range.

        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe

        Returns:
            DataFrame with historical data
        """
        # Calculate number of candles needed
        time_delta = end_date - start_date
        hours = int(time_delta.total_seconds() / 3600)

        # Timeframe multiplier
        timeframe_multipliers = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        multiplier = timeframe_multipliers.get(timeframe, 60)
        limit = int(hours * multiplier / 60) + 100  # Add buffer

        # Fetch data
        df = await self.fetch_ohlcv(symbol, timeframe, limit=limit, since=start_date)

        # Filter to date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        return df


async def main():
    """Example usage of the data adapter."""
    adapter = QlibDataAdapter(exchange_id="binance")

    try:
        # Fetch some data
        df = await adapter.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)
        print(f"Fetched {len(df)} candles")
        print(df.head())

        # Convert to Qlib format
        qlib_df = adapter.to_qlib_format(df, "BTCUSDT")
        print(f"\nQlib format:")
        print(qlib_df.head())

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
