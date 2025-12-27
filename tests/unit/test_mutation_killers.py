"""
Mutation-Killer Tests for Grid Trading Strategy

Cognitive QA: "Testing the Tester"
These tests are specifically designed to kill AST mutants

Target: Kill 80%+ of mutants in modern_strategies.py

Strategy:
- Execute exact code paths where mutations exist
- Validate exact results that would fail if mutated
- Cover arithmetic, comparison, logical, and constant mutations
"""

import pytest
from datetime import datetime
import pandas as pd
import numpy as np
import sys
from unittest.mock import MagicMock

# Mock ccxt before importing
sys.modules['ccxt'] = MagicMock()

from graphwiz_trader.strategies import GridTradingStrategy, GridTradingMode


class TestArithmeticMutations:
    """
    Tests designed to kill arithmetic operator mutants
    Target: +, -, *, / operators in grid calculations
    """

    def test_arithmetic_step_calculation_kills_mutant(self):
        """
        Kill: / mutant in step calculation (line 112)
        Original: step = (upper - lower) / num_grids
        Mutant: step = (upper - lower) * num_grids

        If mutant survives, step would be 2000 * 10 = 20000 instead of 2000
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=10,
            grid_mode=GridTradingMode.ARITHMETIC,
            investment_amount=10000.0,
        )

        # Act
        sorted_levels = sorted(strategy.grid_levels)

        # Assert: Check exact step size
        # Original: (100000 - 80000) / 10 = 2000
        # Mutant: (100000 - 80000) * 10 = 200000
        for i in range(len(sorted_levels) - 1):
            actual_step = sorted_levels[i + 1] - sorted_levels[i]
            assert actual_step == 2000.0, \
                f"Step should be 2000.0, got {actual_step}. Mutant / → * survived!"

    def test_arithmetic_geometric_ratio_kills_mutant(self):
        """
        Kill: * and / mutants in geometric ratio (line 120)
        Original: ratio = (upper / lower) ** (1 / num_grids)
        Mutants: ratio = (upper * lower) ** ... or ratio = ... ** (1 * num_grids)

        For 80k-100k with 10 grids:
        Original ratio: (100000/80000)^(1/10) = 1.25^(0.1) ≈ 1.0225
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
        )

        # Act
        sorted_levels = sorted(strategy.grid_levels)

        # Assert: Verify geometric progression
        expected_ratio = (100000.0 / 80000.0) ** (1 / 10)

        for i in range(len(sorted_levels) - 1):
            actual_ratio = sorted_levels[i + 1] / sorted_levels[i]
            assert abs(actual_ratio - expected_ratio) < 0.0001, \
                f"Ratio should be {expected_ratio:.6f}, got {actual_ratio:.6f}. Mutant survived!"

    def test_arithmetic_investment_division_kills_mutant(self):
        """
        Kill: / → * mutant in position sizing (line 172)
        Original: value_per_grid = investment / num_grids
        Mutant: value_per_grid = investment * num_grids

        Original: 10000 / 10 = 1000 per grid
        Mutant: 10000 * 10 = 100000 per grid (clearly wrong)
        """
        # Arrange
        investment = 10000.0
        num_grids = 10

        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=num_grids,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=investment,
        )

        # Act
        position_sizes = strategy.calculate_position_sizes()

        # Assert: Each grid should have value = investment / num_grids
        expected_value_per_grid = investment / num_grids  # 1000.0

        for level, size in position_sizes.items():
            actual_value = size * level  # size in tokens * price = value
            assert abs(actual_value - expected_value_per_grid) < 1.0, \
                f"Grid value should be ${expected_value_per_grid}, got ${actual_value}. Mutant / → * survived!"

    def test_arithmetic_level_calculation_kills_mutant(self):
        """
        Kill: + → - or * → / mutants in grid level calculation (line 114)
        Original: lower + i * step
        Mutant: lower - i * step (levels go down)
        Mutant: lower + i / step (tiny steps)
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=10,
            grid_mode=GridTradingMode.ARITHMETIC,
            investment_amount=10000.0,
        )

        # Act
        levels = strategy.grid_levels

        # Assert: Levels must be monotonically increasing
        for i in range(len(levels) - 1):
            assert levels[i + 1] > levels[i], \
                f"Levels must increase. Got {levels[i]} → {levels[i+1]}. Mutant + → - survived!"

        # Assert: First level should be exactly lower_price
        assert levels[0] == 80000.0, \
            f"First level should be 80000.0, got {levels[0]}"

    def test_arithmetic_expansion_factor_kills_mutant(self):
        """
        Kill: * → / mutant in volatility expansion (line 263)
        Original: expansion_factor = 1 + (volatility * 2)
        Mutant: expansion_factor = 1 + (volatility / 2)  # Much smaller expansion

        Test by checking _calculate_new_range output
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
        )

        # Act
        current_price = 90000.0
        volatility = 0.1  # 10% volatility

        new_lower, new_upper = strategy._calculate_new_range(current_price, volatility)

        # Assert: expansion_factor = 1 + (0.1 * 2) = 1.2
        # new_upper = 90000 * 1.2 = 108000
        # Mutant: expansion_factor = 1 + (0.1 / 2) = 1.05
        # new_upper_mutant = 90000 * 1.05 = 94500

        expected_upper = current_price * (1 + (volatility * 2))
        assert abs(new_upper - expected_upper) < 1.0, \
            f"Upper should be {expected_upper}, got {new_upper}. Mutant * → / survived!"


class TestComparisonMutations:
    """
    Tests designed to kill comparison operator mutants
    Target: >, <, >=, <=, ==, != operators
    """

    def test_comparison_buy_order_generation_kills_mutant(self):
        """
        Kill: < → <= or < → > mutants in buy order logic (line 228)
        Original: if level < current_price
        Mutant: if level > current_price (would generate wrong orders)
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=5,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
        )

        # Act: Price in middle of grid
        current_price = 90000.0
        signals = strategy.generate_signals(current_price=current_price)

        # Assert: Should have buy orders below current price
        buy_orders = [o for o in signals['orders_to_place'] if o['side'] == 'buy']

        assert len(buy_orders) > 0, "Should generate buy orders"

        # All buy prices must be less than current price
        for order in buy_orders:
            assert order['price'] < current_price, \
                f"Buy price {order['price']} should be < {current_price}. Mutant < → > survived!"

    def test_comparison_sell_order_generation_kills_mutant(self):
        """
        Kill: > → >= or > → < mutants in sell order logic (line 237)
        Original: if next_level > current_price
        Mutant: if next_level < current_price (wrong side)
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=5,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
        )

        # Act
        current_price = 90000.0
        signals = strategy.generate_signals(current_price=current_price)

        # Assert: Should have sell orders above current price
        sell_orders = [o for o in signals['orders_to_place'] if o['side'] == 'sell']

        assert len(sell_orders) > 0, "Should generate sell orders"

        # All sell prices must be greater than current price
        for order in sell_orders:
            assert order['price'] > current_price, \
                f"Sell price {order['price']} should be > {current_price}. Mutant > → < survived!"

    def test_comparison_volatility_threshold_kills_mutant(self):
        """
        Kill: > → < mutant in volatility check (line 215)
        Original: if volatility > volatility_threshold
        Mutant: if volatility < volatility_threshold (inverted logic)

        Test with volatility exactly at threshold + epsilon
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
            dynamic_rebalancing=True,
            volatility_threshold=0.05,
        )

        # Act: Create high volatility data (±20% swings)
        # This creates volatility > 0.05
        historical_data = pd.DataFrame({
            'close': [
                90000, 72000, 108000, 81000, 99000,  # Large swings
                79200, 100800, 80640, 99360, 79488   # ±20% oscillations
            ]
        })
        volatility = historical_data['close'].pct_change().std()

        signals = strategy.generate_signals(
            current_price=90000.0,
            historical_data=historical_data,
        )

        # Assert: Should trigger rebalancing
        assert signals['rebalance_needed'] == True, \
            f"Volatility {volatility:.4f} > 0.05 should trigger rebalance. Mutant > → < survived!"

    def test_comparison_trailing_profit_kills_mutant(self):
        """
        Kill: > → < mutant in trailing profit (line 248)
        Original: if current_price > highest_price * (1 - trailing_pct)
        Mutant: if current_price < highest_price * (1 - trailing_pct) (inverted)
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=5,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
            trailing_profit=True,
            trailing_profit_pct=0.01,  # 1%
        )

        # Act: Price near highest (within 1%)
        current_price = 99000.0  # Just below 100k
        signals = strategy.generate_signals(current_price=current_price)

        # Assert: Should activate trailing profit
        # highest = 100000, threshold = 100000 * 0.99 = 99000
        # current = 99000, should be > threshold (or >=)
        assert signals['trailing_profit_active'] == True, \
            f"Price {current_price} near highest should trigger trailing. Mutant > → < survived!"


class TestLogicalMutations:
    """
    Tests designed to kill logical operator mutants
    Target: and, or, not operators
    """

    def test_logical_dynamic_rebalancing_and_kills_mutant(self):
        """
        Kill: and → or mutant in rebalancing check (line 213)
        Original: if dynamic_rebalancing and historical_data is not None
        Mutant: if dynamic_rebalancing or historical_data is not None

        With mutant, would trigger even with dynamic_rebalancing=False
        """
        # Arrange: dynamic_rebalancing = False
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
            dynamic_rebalancing=False,  # Disabled
        )

        historical_data = pd.DataFrame({
            'close': [80000, 82000, 84000, 86000, 88000, 90000]
        })

        # Act
        signals = strategy.generate_signals(
            current_price=90000.0,
            historical_data=historical_data,
        )

        # Assert: Should NOT rebalance (dynamic_rebalancing is False)
        assert signals['rebalance_needed'] == False, \
            "With dynamic_rebalancing=False, should not rebalance. Mutant and → or survived!"

    def test_logical_trailing_profit_and_kills_mutant(self):
        """
        Kill: and → or in trailing profit activation (implied logic)
        """
        # Arrange: trailing_profit = False
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=5,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
            trailing_profit=False,  # Disabled
        )

        # Act: Price near highest
        current_price = 99000.0
        signals = strategy.generate_signals(current_price=current_price)

        # Assert: Should NOT activate trailing profit
        assert signals['trailing_profit_active'] == False, \
            "With trailing_profit=False, should not activate. Mutant and → or survived!"


class TestConstantMutations:
    """
    Tests designed to kill constant mutations
    Target: 0, 1, True, False constants
    """

    def test_constant_num_grids_kills_mutant(self):
        """
        Kill: num_grids → 0 or num_grids → 1 mutant
        Test ensures we use the actual num_grids value
        """
        # Arrange: Use specific num_grids
        num_grids = 7  # Non-standard number
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=num_grids,
            grid_mode=GridTradingMode.ARITHMETIC,
            investment_amount=10000.0,
        )

        # Act
        actual_levels = len(strategy.grid_levels)

        # Assert: Should have num_grids + 1 levels
        expected_levels = num_grids + 1
        assert actual_levels == expected_levels, \
            f"Should have {expected_levels} levels for num_grids={num_grids}, got {actual_levels}"

    def test_constant_trailing_pct_kills_mutant(self):
        """
        Kill: 0.01 → 0.0 or 0.01 → 1.0 mutant in trailing profit
        """
        # Arrange: Specific trailing percentage
        trailing_pct = 0.02  # 2%
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=5,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
            trailing_profit=True,
            trailing_profit_pct=trailing_pct,
        )

        # Act: Price at highest
        current_price = 100000.0
        signals = strategy.generate_signals(current_price=current_price)

        # Assert: Trailing sell price should use correct percentage
        # Formula: current_price * (1 + trailing_profit_pct)
        expected_sell_price = current_price * (1 + trailing_pct)
        actual_sell_price = signals['trailing_sell_price']

        assert abs(actual_sell_price - expected_sell_price) < 0.01, \
            f"Sell price should be {expected_sell_price}, got {actual_sell_price}. Mutant constant survived!"

    def test_constant_volatility_threshold_kills_mutant(self):
        """
        Kill: 0.05 → 0.0 or 0.05 → 1.0 mutant
        """
        # Arrange: Specific threshold
        threshold = 0.08  # 8%
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
            dynamic_rebalancing=True,
            volatility_threshold=threshold,
        )

        # Create volatility just below threshold
        historical_data = pd.DataFrame({
            'close': [90000, 90500, 91000, 91500, 92000]  # Low volatility
        })
        volatility = historical_data['close'].pct_change().std()

        # Act
        signals = strategy.generate_signals(
            current_price=91000.0,
            historical_data=historical_data,
        )

        # Assert: Should NOT rebalance (volatility < threshold)
        assert signals['rebalance_needed'] == False, \
            f"Volatility {volatility:.4f} < {threshold} should not rebalance. Mutant constant survived!"


class TestEdgeCaseMutations:
    """
    Edge case tests to kill boundary mutants
    """

    def test_boundary_price_at_exact_level_kills_mutant(self):
        """
        Kill: Mutants in boundary comparison when price == grid level
        """
        # Arrange
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=5,
            grid_mode=GridTradingMode.ARITHMETIC,
            investment_amount=10000.0,
        )

        # Act: Price exactly at a grid level
        grid_level = strategy.grid_levels[2]  # Middle level
        signals = strategy.generate_signals(current_price=grid_level)

        # Assert: Should generate orders
        assert len(signals['orders_to_place']) > 0, \
            "Should generate orders even when price equals grid level"

    def test_boundary_zero_investment_kills_mutant(self):
        """
        Kill: Division by zero or investment mutation mutants
        """
        # Arrange: Zero investment
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=5,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=0.0,  # Zero investment
        )

        # Act: Should not crash
        position_sizes = strategy.calculate_position_sizes()

        # Assert: All positions should be zero
        for level, size in position_sizes.items():
            assert size == 0.0, \
                f"With zero investment, position should be 0, got {size}"

    def test_boundary_single_grid_kills_mutant(self):
        """
        Kill: Mutants when num_grids = 1 (minimum)
        """
        # Arrange: Single grid
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=1,  # Minimum
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
        )

        # Act
        levels = strategy.grid_levels

        # Assert: Should have exactly 2 levels
        assert len(levels) == 2, f"num_grids=1 should give 2 levels, got {len(levels)}"
        assert levels[0] == 80000.0, f"Lower should be 80000, got {levels[0]}"
        assert levels[1] == 100000.0, f"Upper should be 100000, got {levels[1]}"

    def test_boundary_large_grid_count_kills_mutant(self):
        """
        Kill: Mutants with large num_grids (stress test)
        """
        # Arrange: Large grid count
        num_grids = 100
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=num_grids,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=100000.0,
        )

        # Act
        levels = strategy.grid_levels

        # Assert: Should have num_grids + 1 levels
        assert len(levels) == num_grids + 1, \
            f"Should have {num_grids + 1} levels, got {len(levels)}"

        # Assert: All levels within bounds
        for level in levels:
            assert 79999 <= level <= 100001, \
                f"Level {level} outside valid range [79999, 100001]"
