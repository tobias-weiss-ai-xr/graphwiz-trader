"""
High-Frequency Trading (HFT) Module.

This module provides ultra-low latency trading capabilities for cryptocurrency markets.
"""

from graphwiz_trader.hft.analytics import HFTAnalytics
from graphwiz_trader.hft.engine import HFTEngine
from graphwiz_trader.hft.executor import FastOrderExecutor
from graphwiz_trader.hft.market_data import WebSocketMarketData
from graphwiz_trader.hft.monitoring import PerformanceMonitor
from graphwiz_trader.hft.orderbook import OrderBook, OrderBookManager
from graphwiz_trader.hft.risk import HFTRiskManager
from graphwiz_trader.hft.strategies import (
    CrossExchangeArbitrage,
    HFTStrategy,
    StatisticalArbitrage,
)

__all__ = [
    "HFTEngine",
    "WebSocketMarketData",
    "OrderBookManager",
    "OrderBook",
    "FastOrderExecutor",
    "HFTRiskManager",
    "HFTStrategy",
    "StatisticalArbitrage",
    "CrossExchangeArbitrage",
    "HFTAnalytics",
    "PerformanceMonitor",
]
