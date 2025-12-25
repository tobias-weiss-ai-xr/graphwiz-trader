"""
HFT Trading Strategies.

This module contains various high-frequency trading strategies.
"""

from graphwiz_trader.hft.strategies.base import HFTStrategy
from graphwiz_trader.hft.strategies.cross_exchange_arb import CrossExchangeArbitrage
from graphwiz_trader.hft.strategies.stat_arb import StatisticalArbitrage

__all__ = [
    "HFTStrategy",
    "StatisticalArbitrage",
    "CrossExchangeArbitrage",
]
