"""Tests for technical indicators module."""

import pytest
import numpy as np
from graphwiz_trader.analysis import TechnicalAnalysis, TechnicalIndicators, IndicatorResult


class TestTechnicalIndicators:
    """Test suite for TechnicalIndicators class."""

    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        prices = [10, 12, 15, 14, 16, 18, 20, 22, 25, 23]
        period = 3

        sma = TechnicalIndicators.sma(prices, period)

        # First 2 values should be None
        assert sma[0] is None
        assert sma[1] is None

        # Third value: (10 + 12 + 15) / 3 = 12.33
        assert abs(sma[2] - 12.33) < 0.01

        # Fourth value: (12 + 15 + 14) / 3 = 13.67
        assert abs(sma[3] - 13.67) < 0.01

        # Check last value is calculated
        assert sma[-1] is not None

    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data."""
        prices = [10, 12]
        period = 5

        sma = TechnicalIndicators.sma(prices, period)

        assert all(v is None for v in sma)

    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        prices = [10, 12, 15, 14, 16, 18, 20, 22, 25, 23]
        period = 3

        ema = TechnicalIndicators.ema(prices, period)

        # First 2 values should be None
        assert ema[0] is None
        assert ema[1] is None

        # Check EMA is calculated
        assert ema[2] is not None

        # EMA should react more quickly to recent prices than SMA
        sma = TechnicalIndicators.sma(prices, period)
        # After enough data points, EMA and SMA should diverge
        if ema[-1] is not None and sma[-1] is not None:
            # They should be different for the last value
            assert ema[-1] != sma[-1] or ema[3] != sma[3]

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]

        result = TechnicalIndicators.rsi(prices, period=14)

        assert isinstance(result, IndicatorResult)
        assert result.name == "RSI"
        assert len(result.values) == len(prices)
        assert len(result.signals) == len(prices)

        # RSI should be between 0 and 100
        valid_values = [v for v in result.values if v is not None]
        for val in valid_values:
            assert 0 <= val <= 100

        # Check signal generation
        signal_set = set(result.signals)
        assert "buy" in signal_set or "sell" in signal_set or "neutral" in signal_set

    def test_rsi_oversold_signal(self):
        """Test RSI generates buy signal when oversold."""
        # Create price pattern that drops significantly (RSI should be low)
        prices = [100] * 10 + [50] * 5  # Big drop

        result = TechnicalIndicators.rsi(prices, period=14)

        # Last value should indicate oversold (buy signal) or at least be calculated
        if result.values[-1] is not None:
            # With a 50% drop, RSI should be quite low
            assert result.values[-1] < 50

    def test_rsi_overbought_signal(self):
        """Test RSI generates sell signal when overbought."""
        # Create price pattern that rises significantly
        prices = [50] * 10 + [100] * 5  # Big rise

        result = TechnicalIndicators.rsi(prices, period=14)

        # Last value should indicate overbought (sell signal) or at least be calculated
        if result.values[-1] is not None:
            # With a 100% rise, RSI should be quite high
            assert result.values[-1] > 50

    def test_macd_calculation(self):
        """Test MACD calculation."""
        prices = [100, 102, 105, 108, 107, 110, 112, 115, 118, 120,
                  122, 125, 123, 120, 118, 115, 117, 119, 122, 125,
                  128, 130, 132, 135, 133, 130, 128, 125, 123, 120]

        result = TechnicalIndicators.macd(prices)

        assert isinstance(result, IndicatorResult)
        assert result.name == "MACD"
        assert len(result.values) == len(prices)

        # Check structure of values
        if result.values[-1] is not None:
            assert "macd" in result.values[-1]
            assert "signal" in result.values[-1]
            assert "histogram" in result.values[-1]

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        prices = [100, 102, 105, 108, 107, 110, 112, 115, 118, 110,
                  105, 103, 100, 98, 95, 97, 100, 102, 105, 108]

        result = TechnicalIndicators.bollinger_bands(prices, period=20, std_dev=2.0)

        assert isinstance(result, IndicatorResult)
        assert result.name == "BB"
        assert len(result.values) == len(prices)

        # Check structure of values
        if result.values[-1] is not None:
            assert "upper" in result.values[-1]
            assert "middle" in result.values[-1]
            assert "lower" in result.values[-1]

            # Upper should be > middle > lower
            assert result.values[-1]["upper"] > result.values[-1]["middle"]
            assert result.values[-1]["middle"] > result.values[-1]["lower"]

    def test_bollinger_bands_signals(self):
        """Test Bollinger Bands generates signals at bands."""
        # Use period=5 to ensure we have enough data
        prices = [100] * 10 + [90] * 10  # Price drops to lower band

        result = TechnicalIndicators.bollinger_bands(prices, period=5)

        # Should generate buy signals when price is at lower band
        assert "buy" in result.signals

    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        prices = [100, 102, 105, 108, 107]
        volumes = [1000, 1500, 1200, 800, 2000]

        vwap = TechnicalIndicators.vwap(prices, volumes)

        assert len(vwap) == len(prices)
        assert vwap[0] == prices[0]  # First VWAP equals first price

        # VWAP should be weighted average
        assert all(v is not None for v in vwap)

        # Calculate expected VWAP manually
        expected_cumulative_pv = sum(p * v for p, v in zip(prices, volumes))
        expected_cumulative_v = sum(volumes)
        expected_vwap = expected_cumulative_pv / expected_cumulative_v

        assert abs(vwap[-1] - expected_vwap) < 0.01

    def test_vwap_mismatched_lengths(self):
        """Test VWAP with mismatched price and volume lengths."""
        prices = [100, 102, 105]
        volumes = [1000, 1500]  # Different length

        vwap = TechnicalIndicators.vwap(prices, volumes)

        assert all(v is None for v in vwap)

    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        high = [105, 108, 110, 112, 115, 118, 120]
        low = [100, 102, 105, 107, 110, 112, 115]
        close = [102, 105, 108, 110, 112, 115, 118]

        atr = TechnicalIndicators.atr(high, low, close, period=14)

        assert len(atr) == len(high)
        assert atr[0] is None  # First values should be None

        # ATR should be positive
        valid_atr = [a for a in atr if a is not None]
        for val in valid_atr:
            assert val > 0

    def test_atr_mismatched_lengths(self):
        """Test ATR with mismatched OHLC lengths."""
        high = [105, 108, 110]
        low = [100, 102]  # Different length
        close = [102, 105, 108]

        atr = TechnicalIndicators.atr(high, low, close)

        assert all(a is None for a in atr)


class TestTechnicalAnalysis:
    """Test suite for TechnicalAnalysis class."""

    def test_initialization(self):
        """Test TechnicalAnalysis initialization."""
        analysis = TechnicalAnalysis()

        assert analysis.indicators is not None
        assert isinstance(analysis.indicators, TechnicalIndicators)

    def test_comprehensive_analysis(self):
        """Test comprehensive technical analysis."""
        prices = [100, 102, 105, 108, 107, 110, 112, 115, 118, 110,
                  105, 103, 100, 98, 95, 97, 100, 102, 105, 108,
                  110, 112, 115, 118, 120, 122, 125, 123, 120, 118]

        analysis = TechnicalAnalysis()
        results = analysis.analyze(prices)

        # Check all expected indicators are present
        assert "rsi" in results
        assert "macd" in results
        assert "bollinger_bands" in results
        assert "ema" in results
        assert "overall_signal" in results

    def test_analysis_with_volumes(self):
        """Test analysis with volume data."""
        prices = [100, 102, 105, 108, 107, 110, 112, 115, 118, 110]
        volumes = [1000, 1500, 1200, 800, 2000, 1800, 1500, 1200, 900, 1100]

        analysis = TechnicalAnalysis()
        results = analysis.analyze(prices, volumes=volumes)

        assert "vwap" in results
        assert results["vwap"]["latest"] is not None

    def test_analysis_with_ohlc(self):
        """Test analysis with OHLC data."""
        # Need at least 15 data points for ATR with period 14
        prices = [102, 105, 108, 110, 112, 115, 118, 120, 122, 125,
                  128, 130, 132, 135, 133]
        high = [105, 108, 110, 112, 115, 118, 120, 122, 125, 128,
                130, 132, 135, 138, 135]
        low = [100, 102, 105, 107, 110, 112, 115, 118, 120, 122,
               125, 128, 130, 132, 130]
        close = [102, 105, 108, 110, 112, 115, 118, 120, 122, 125,
                 128, 130, 132, 135, 133]

        analysis = TechnicalAnalysis()
        results = analysis.analyze(prices, high=high, low=low, close=close)

        assert "atr" in results
        # ATR should have a value by the last data point
        assert results["atr"]["latest"] is not None

    def test_overall_signal_calculation(self):
        """Test overall signal is calculated correctly."""
        # Create pattern that should generate buy signals
        prices = [100] * 20 + [90] * 10  # Drop for oversold RSI

        analysis = TechnicalAnalysis()
        results = analysis.analyze(prices)

        assert "overall_signal" in results
        assert "signal" in results["overall_signal"]
        assert "confidence" in results["overall_signal"]
        assert "buy_count" in results["overall_signal"]
        assert "sell_count" in results["overall_signal"]

        # Check signal is one of expected values
        valid_signals = ["strong_buy", "buy", "neutral", "sell", "strong_sell"]
        assert results["overall_signal"]["signal"] in valid_signals

        # Confidence should be between 0 and 1
        assert 0 <= results["overall_signal"]["confidence"] <= 1

    def test_indicator_result_dataclass(self):
        """Test IndicatorResult dataclass."""
        result = IndicatorResult(
            name="Test",
            values=[1.0, 2.0, 3.0],
            signals=["buy", "neutral", "sell"],
            metadata={"period": 14}
        )

        assert result.name == "Test"
        assert result.values == [1.0, 2.0, 3.0]
        assert result.signals == ["buy", "neutral", "sell"]
        assert result.metadata == {"period": 14}

    def test_indicator_result_defaults(self):
        """Test IndicatorResult default values."""
        result = IndicatorResult(name="Test", values=[1.0, 2.0])

        assert result.signals == []  # Default empty list
        assert result.metadata == {}  # Default empty dict


class TestEdgeCases:
    """Edge case tests for indicators."""

    def test_empty_price_list(self):
        """Test indicators with empty price list."""
        analysis = TechnicalAnalysis()
        results = analysis.analyze([])

        # Should handle gracefully
        assert "overall_signal" in results

    def test_single_price(self):
        """Test indicators with single price."""
        analysis = TechnicalAnalysis()
        results = analysis.analyze([100])

        # Should handle gracefully
        assert "overall_signal" in results

    def test_constant_prices(self):
        """Test indicators with constant prices (no volatility)."""
        prices = [100] * 50

        rsi = TechnicalIndicators.rsi(prices)
        bb = TechnicalIndicators.bollinger_bands(prices)

        # RSI for constant prices with no losses should be 100
        if rsi.values[-1] is not None:
            # When there are no losses, avg_loss is 0, so RSI = 100
            assert rsi.values[-1] == 100.0

        # Bollinger Bands should have very small width for constant prices
        if bb.values[-1] is not None:
            # Upper and lower should be very close (minimal std dev)
            width = bb.values[-1]["upper"] - bb.values[-1]["lower"]
            assert width < 1.0  # Should be very small


class TestTechnicalIndicatorsPropertyBased:
    """Property-based tests for TechnicalIndicators using hypothesis."""

    def test_rsi_always_between_0_and_100(self):
        """Test that RSI values are always between 0 and 100."""
        from hypothesis import given, strategies as st

        @given(prices=st.lists(
            st.floats(min_value=1.0, max_value=100000.0, allow_infinity=False, allow_nan=False),
            min_size=20,
            max_size=200
        ))
        def test_rsi_range(prices):
            # Ensure prices are not all the same (would cause division by zero)
            if len(set(prices)) < 2:
                return

            rsi = TechnicalIndicators.rsi(prices, period=14)

            # Check all non-None RSI values are between 0 and 100
            for value in rsi.values:
                if value is not None:
                    assert 0 <= value <= 100, f"RSI {value} outside [0, 100] range"

        test_rsi_range()

    def test_sma_returns_correct_length(self):
        """Test that SMA returns the same length as input."""
        from hypothesis import given, strategies as st

        @given(prices=st.lists(
            st.floats(min_value=1.0, max_value=100000.0, allow_infinity=False, allow_nan=False),
            min_size=5,
            max_size=100
        ))
        def test_sma_length(prices):
            period = min(10, len(prices) // 2 + 1)
            sma = TechnicalIndicators.sma(prices, period)

            # SMA returns a list directly
            assert len(sma) == len(prices), f"SMA length {len(sma)} != input length {len(prices)}"

        test_sma_length()

    def test_bollinger_bands_upper_above_lower(self):
        """Test that upper band is always above lower band."""
        from hypothesis import given, strategies as st

        @given(prices=st.lists(
            st.floats(min_value=10.0, max_value=100000.0, allow_infinity=False, allow_nan=False),
            min_size=20,
            max_size=200
        ))
        def test_bb_structure(prices):
            # Need some price variation
            if len(set(prices)) < 3:
                return

            bb = TechnicalIndicators.bollinger_bands(prices, period=20, std_dev=2)

            # bollinger_bands returns an IndicatorResult with .values attribute
            for value in bb.values:
                if value is not None and isinstance(value, dict):
                    upper = value.get("upper")
                    lower = value.get("lower")

                    # Only check if both values are not None
                    if upper is not None and lower is not None:
                        assert upper >= lower, \
                            f"Upper band {upper} below lower band {lower}"

        test_bb_structure()

    def test_ema_smoothed(self):
        """Test that EMA is smoother (less volatile) than prices."""
        from hypothesis import given, strategies as st

        @given(prices=st.lists(
            st.floats(min_value=100.0, max_value=10000.0, allow_infinity=False, allow_nan=False),
            min_size=20,
            max_size=100
        ))
        def test_ema_smoothing(prices):
            # Need variation
            if len(set(prices)) < 3:
                return

            period = min(10, len(prices) // 2 + 1)
            ema = TechnicalIndicators.ema(prices, period)

            # ema returns a list directly
            # Calculate variance of price changes and EMA changes
            price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            ema_values = [v for v in ema if v is not None]
            ema_changes = [abs(ema_values[i] - ema_values[i-1]) for i in range(1, len(ema_values))]

            if len(ema_changes) > 0 and len(price_changes) > 0:
                # EMA should generally be smoother (lower average change)
                avg_price_change = sum(price_changes) / len(price_changes)
                avg_ema_change = sum(ema_changes) / len(ema_changes)

                # EMA changes should typically be less than or equal to price changes
                assert avg_ema_change <= avg_price_change * 1.1, \
                    f"EMA not smooth enough: {avg_ema_change} vs {avg_price_change}"

        test_ema_smoothing()

    def test_macd_signal_difference(self):
        """Test MACD line and signal line relationship."""
        from hypothesis import given, strategies as st

        @given(prices=st.lists(
            st.floats(min_value=50.0, max_value=50000.0, allow_infinity=False, allow_nan=False),
            min_size=35,
            max_size=200
        ))
        def test_macd_properties(prices):
            # Need variation
            if len(set(prices)) < 5:
                return

            macd = TechnicalIndicators.macd(prices, fast_period=12, slow_period=26, signal_period=9)

            # MACD and signal lines should have same length
            if macd.values and len(macd.values) > 0:
                macd_line = [v.get("macd") for v in macd.values if v and "macd" in v]
                signal_line = [v.get("signal") for v in macd.values if v and "signal" in v]

                # Should have calculated some values
                assert len(macd_line) == len(signal_line)

        test_macd_properties()
