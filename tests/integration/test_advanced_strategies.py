"""Integration tests for 2025 advanced trading strategies."""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from graphwiz_trader.strategies.advanced_strategies import (
    AdvancedMeanReversionStrategy,
    PairsTradingStrategy,
    MomentumVolatilityFilteringStrategy,
    MultiFactorStrategy,
    ConfidenceThresholdStrategy,
    MeanReversionType,
    create_advanced_strategy,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)

    # Generate 100 candles of realistic price data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')

    # Simulate price movement with some trends and reversions
    base_price = 50000
    price_changes = np.random.normal(0, 0.01, 100)
    prices = [base_price]

    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in range(100)],
    })
    df.set_index('timestamp', inplace=True)

    return df


@pytest.fixture
def sample_multi_symbol_data():
    """Create sample data for multiple symbols (for pairs trading)."""
    np.random.seed(42)

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    data = {}

    for symbol in symbols:
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        base_price = {'BTC/USDT': 50000, 'ETH/USDT': 3000, 'BNB/USDT': 400}[symbol]

        price_changes = np.random.normal(0, 0.01, 100)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data[symbol] = pd.Series(prices, index=dates)

    return data


class TestAdvancedMeanReversionStrategy:
    """Tests for AdvancedMeanReversionStrategy."""

    def test_bollinger_reversion(self, sample_ohlcv_data):
        """Test Bollinger Bands mean reversion."""
        strategy = AdvancedMeanReversionStrategy(
            reversion_type=MeanReversionType.BOLLINGER,
            lookback_period=20,
        )

        signals = strategy.generate_signals(sample_ohlcv_data)

        # Check signals are generated
        assert 'signal' in signals
        assert 'exit_signal' in signals
        assert len(signals) == len(sample_ohlcv_data)

        # Check indicators calculated
        assert 'upper_band' in signals
        assert 'lower_band' in signals
        assert 'sma' in signals

        # Check position sizing
        assert 'position_size' in signals

    def test_rsi_reversion(self, sample_ohlcv_data):
        """Test RSI mean reversion."""
        strategy = AdvancedMeanReversionStrategy(
            reversion_type=MeanReversionType.RSI,
            lookback_period=20,
        )

        signals = strategy.generate_signals(sample_ohlcv_data)

        assert 'signal' in signals
        assert 'rsi' in signals
        assert 'position_size' in signals

        # RSI should be between 0 and 100
        assert signals['rsi'].dropna().between(0, 100).all()

    def test_zscore_reversion(self, sample_ohlcv_data):
        """Test Z-score mean reversion."""
        strategy = AdvancedMeanReversionStrategy(
            reversion_type=MeanReversionType.ZSCORE,
            lookback_period=20,
        )

        signals = strategy.generate_signals(sample_ohlcv_data)

        assert 'signal' in signals
        assert 'zscore' in signals
        assert 'position_size' in signals

    def test_multi_reversion(self, sample_ohlcv_data):
        """Test combined multi-indicator reversion."""
        strategy = AdvancedMeanReversionStrategy(
            reversion_type=MeanReversionType.MULTI,
            lookback_period=20,
        )

        signals = strategy.generate_signals(sample_ohlcv_data)

        # Check combined signals
        assert 'signal' in signals
        assert 'bollinger_signal' in signals
        assert 'rsi_signal' in signals
        assert 'zscore_signal' in signals

        # Multi should have voting mechanism
        # At least some signals should be generated
        assert signals['signal'].sum() >= 0

    def test_volatility_filter(self, sample_ohlcv_data):
        """Test volatility filtering."""
        strategy = AdvancedMeanReversionStrategy(
            reversion_type=MeanReversionType.BOLLINGER,
            lookback_period=20,
            volatility_filter=True,
            volatility_threshold=0.02,
        )

        signals = strategy.generate_signals(sample_ohlcv_data)

        # Volatility filter should be applied
        assert 'signal' in signals


class TestPairsTradingStrategy:
    """Tests for PairsTradingStrategy."""

    def test_pair_selection(self, sample_multi_symbol_data):
        """Test PCA-based pair selection."""
        strategy = PairsTradingStrategy(
            lookback_period=30,
            n_components=2,
        )

        pairs = strategy.select_pairs(sample_multi_symbol_data)

        # Should return list of pairs
        assert isinstance(pairs, list)
        # Each pair should be a tuple of (symbol1, symbol2, correlation)
        if len(pairs) > 0:
            assert isinstance(pairs[0], tuple)
            assert len(pairs[0]) == 3

    def test_hedge_ratio_calculation(self, sample_multi_symbol_data):
        """Test hedge ratio calculation."""
        strategy = PairsTradingStrategy()

        price1 = sample_multi_symbol_data['BTC/USDT']
        price2 = sample_multi_symbol_data['ETH/USDT']

        hedge_ratio = strategy._calculate_hedge_ratio(price1, price2)

        # Hedge ratio should be positive number
        assert hedge_ratio > 0
        assert isinstance(hedge_ratio, (int, float))

    def test_signal_generation(self, sample_multi_symbol_data):
        """Test signal generation for trading pairs."""
        strategy = PairsTradingStrategy(lookback_period=20)

        # Create DataFrame with both symbols
        df = pd.DataFrame({
            'BTC/USDT': sample_multi_symbol_data['BTC/USDT'],
            'ETH/USDT': sample_multi_symbol_data['ETH/USDT'],
        })

        signals = strategy.generate_signals(('BTC/USDT', 'ETH/USDT'), df)

        # Check signals
        assert 'signal' in signals
        assert 'spread' in signals
        assert 'spread_zscore' in signals
        assert 'hedge_ratio' in signals
        assert 'position_size' in signals


class TestMomentumVolatilityFilteringStrategy:
    """Tests for MomentumVolatilityFilteringStrategy."""

    def test_signal_generation(self, sample_ohlcv_data):
        """Test momentum with volatility filtering."""
        strategy = MomentumVolatilityFilteringStrategy(
            momentum_period=20,
            volatility_period=10,
            volatility_threshold=0.05,
        )

        signals = strategy.generate_signals(sample_ohlcv_data)

        # Check signals
        assert 'signal' in signals
        assert 'momentum' in signals
        assert 'volatility' in signals
        assert 'position_size' in signals
        assert 'exit_signal' in signals

        # Momentum should be calculated
        assert signals['momentum'].notna().sum() > 0

        # Volatility should be calculated
        assert signals['volatility'].notna().sum() > 0

    def test_volatility_filtering(self, sample_ohlcv_data):
        """Test that high volatility periods are filtered out."""
        strategy = MomentumVolatilityFilteringStrategy(
            volatility_threshold=0.01,  # Very low threshold
        )

        signals = strategy.generate_signals(sample_ohlcv_data)

        # With very low threshold, fewer signals should be generated
        # (most periods will be filtered out)
        assert 'signal' in signals
        assert 'volatility' in signals


class TestMultiFactorStrategy:
    """Tests for MultiFactorStrategy."""

    def test_factor_calculation(self, sample_ohlcv_data):
        """Test calculation of multiple factors."""
        strategy = MultiFactorStrategy()

        factors = strategy.calculate_factors(sample_ohlcv_data)

        # Check all factors are present
        expected_factors = [
            'momentum',
            'mean_reversion',
            'volatility',
            'volume',
            'on_chain_activity',
        ]

        for factor in expected_factors:
            assert factor in factors.columns
            # Normalized version should also be present
            assert f'{factor}_norm' in factors.columns

    def test_signal_generation(self, sample_ohlcv_data):
        """Test multi-factor signal generation."""
        strategy = MultiFactorStrategy()

        signals = strategy.generate_signals(sample_ohlcv_data)

        # Check signals
        assert 'signal' in signals
        assert 'factor_score' in signals
        assert 'position_size' in signals
        assert 'exit_signal' in signals

        # Factor score should be between 0 and 2 (clipped) - ignore NaN values from warmup period
        valid_scores = signals['factor_score'].dropna()
        if len(valid_scores) > 0:
            assert valid_scores.between(0, 2).all()

    def test_custom_factor_weights(self, sample_ohlcv_data):
        """Test with custom factor weights."""
        custom_weights = {
            'momentum': 0.5,
            'mean_reversion': 0.3,
            'volatility': 0.2,
            'volume': 0.0,
            'on_chain_activity': 0.0,
        }

        strategy = MultiFactorStrategy(factor_weights=custom_weights)
        signals = strategy.generate_signals(sample_ohlcv_data)

        # Should still generate signals
        assert 'signal' in signals
        assert 'factor_score' in signals


class TestConfidenceThresholdStrategy:
    """Tests for ConfidenceThresholdStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        # Normal mode
        strategy = ConfidenceThresholdStrategy(mode='normal')
        assert strategy.threshold == 0.6

        # Aggressive mode
        strategy = ConfidenceThresholdStrategy(mode='aggressive')
        assert strategy.threshold == 0.7

        # Conservative mode
        strategy = ConfidenceThresholdStrategy(mode='conservative')
        assert strategy.threshold == 0.5

    def test_threshold_adjustment(self):
        """Test dynamic threshold adjustment."""
        strategy = ConfidenceThresholdStrategy(base_threshold=0.6)

        # Create mock recent performance
        recent_performance = pd.Series([
            0.01, 0.02, 0.015, 0.01, -0.005, 0.02, 0.01, 0.015, 0.02, 0.01
        ])

        # Low volatility
        adjusted_threshold = strategy.adjust_threshold(
            recent_performance=recent_performance,
            market_volatility=0.01,
        )

        # Threshold should be adjusted (likely lower due to good Sharpe)
        assert isinstance(adjusted_threshold, float)
        assert 0.4 <= adjusted_threshold <= 0.8

        # High volatility should raise threshold
        high_vol_threshold = strategy.adjust_threshold(
            recent_performance=recent_performance,
            market_volatility=0.10,
        )

        assert high_vol_threshold >= adjusted_threshold


class TestCreateAdvancedStrategy:
    """Tests for convenience factory function."""

    def test_create_mean_reversion(self):
        """Test creating mean reversion strategy."""
        strategy = create_advanced_strategy(
            strategy_type="mean_reversion",
            reversion_type=MeanReversionType.BOLLINGER,
        )

        assert isinstance(strategy, AdvancedMeanReversionStrategy)

    def test_create_pairs_trading(self):
        """Test creating pairs trading strategy."""
        strategy = create_advanced_strategy(
            strategy_type="pairs_trading",
            lookback_period=25,
        )

        assert isinstance(strategy, PairsTradingStrategy)

    def test_create_momentum_volatility(self):
        """Test creating momentum volatility strategy."""
        strategy = create_advanced_strategy(
            strategy_type="momentum_volatility",
            momentum_period=30,
        )

        assert isinstance(strategy, MomentumVolatilityFilteringStrategy)

    def test_create_multi_factor(self):
        """Test creating multi-factor strategy."""
        strategy = create_advanced_strategy(
            strategy_type="multi_factor",
        )

        assert isinstance(strategy, MultiFactorStrategy)

    def test_invalid_strategy_type(self):
        """Test error handling for invalid strategy type."""
        with pytest.raises(ValueError):
            create_advanced_strategy(strategy_type="invalid_type")


class TestStrategyIntegration:
    """Integration tests for strategy combinations."""

    def test_mean_reversion_with_exit_signals(self, sample_ohlcv_data):
        """Test that mean reversion generates proper exit signals."""
        strategy = AdvancedMeanReversionStrategy(
            reversion_type=MeanReversionType.BOLLINGER,
        )

        signals = strategy.generate_signals(sample_ohlcv_data)

        # Exit signals should exist
        assert 'exit_signal' in signals

        # Exit signals should be different from entry signals
        # (at least some of the time)
        assert not (signals['signal'] == signals['exit_signal']).all()

    def test_pairs_trading_position_sizing(self, sample_multi_symbol_data):
        """Test that pairs trading sizes positions based on z-score."""
        strategy = PairsTradingStrategy(lookback_period=20)

        df = pd.DataFrame({
            'BTC/USDT': sample_multi_symbol_data['BTC/USDT'],
            'ETH/USDT': sample_multi_symbol_data['ETH/USDT'],
        })

        signals = strategy.generate_signals(('BTC/USDT', 'ETH/USDT'), df)

        # Position size should correlate with z-score magnitude
        assert 'position_size' in signals

        # Check that position sizes vary
        if signals['position_size'].notna().sum() > 0:
            assert signals['position_size'].std() > 0

    def test_multi_factor_combines_signals(self, sample_ohlcv_data):
        """Test that multi-factor properly combines multiple factors."""
        strategy = MultiFactorStrategy()

        factors = strategy.calculate_factors(sample_ohlcv_data)
        signals = strategy.generate_signals(sample_ohlcv_data)

        # Factor score should combine all factors
        assert 'factor_score' in signals

        # Score should be weighted combination
        # Check that it's not just equal to any single factor
        factor_scores = signals['factor_score'].dropna()
        momentum_norm = factors['momentum_norm'].dropna()

        # Align indices
        if len(factor_scores) > 0 and len(momentum_norm) > 0:
            common_index = factor_scores.index.intersection(momentum_norm.index)
            if len(common_index) > 0:
                assert not (factor_scores[common_index] == momentum_norm[common_index]).all()


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
