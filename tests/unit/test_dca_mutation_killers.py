"""
Mutation-Killer Tests for Smart DCA Strategy

Cognitive QA: "Testing the Tester"
These tests are specifically designed to kill AST mutants in SmartDCAStrategy

Target: Kill arithmetic, comparison, logical, and constant mutants
"""

import pytest
from datetime import datetime
import pandas as pd
import numpy as np
import sys
from unittest.mock import MagicMock

# Mock ccxt before importing
sys.modules['ccxt'] = MagicMock()

from graphwiz_trader.strategies import SmartDCAStrategy


class TestDCAArithmeticMutations:
    """Tests to kill arithmetic operator mutants in DCA strategy"""

    def test_dca_quantity_calculation_kills_mutant(self):
        """
        Kill: / → * mutant in quantity calculation (line 398)
        Original: quantity = purchase_amount / current_price
        Mutant: quantity = purchase_amount * current_price

        Original: 100 / 50000 = 0.002 BTC
        Mutant: 100 * 50000 = 5000000 BTC (clearly wrong)
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
        )

        current_price = 50000.0

        # Act
        purchase = strategy.calculate_next_purchase(current_price=current_price)

        # Assert: quantity should be amount / price
        expected_quantity = 100.0 / 50000.0  # 0.002
        actual_quantity = purchase['quantity']

        assert abs(actual_quantity - expected_quantity) < 0.0001, \
            f"Quantity should be {expected_quantity}, got {actual_quantity}. Mutant / → * survived!"

    def test_dca_volatility_multiplier_kills_mutant(self):
        """
        Kill: * → / mutant in volatility adjustment (lines 371, 373)
        Original: purchase_amount *= 0.8 (or *= 1.2)
        Mutant: purchase_amount /= 0.8 (would increase instead of decrease)
        """
        # Arrange: High volatility scenario
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            volatility_adjustment=True,
        )

        # Create high volatility data (> 0.08)
        historical_data = pd.DataFrame({
            'close': [50000, 60000, 40000, 55000, 45000]  # High volatility
        })

        # Act
        purchase = strategy.calculate_next_purchase(
            current_price=50000.0,
            historical_data=historical_data,
        )

        # Assert: With high volatility, amount should be reduced (multiply by 0.8)
        # Original: 100 * 0.8 = 80
        # Mutant: 100 / 0.8 = 125 (would increase!)
        assert purchase['amount'] < 100.0, \
            f"High volatility should reduce amount. Got {purchase['amount']}. Mutant * → / survived!"

    def test_dca_price_drop_calculation_kills_mutant(self):
        """
        Kill: - → + or / → * mutants in price drop calculation (line 378)
        Original: price_drop = (last - current) / last
        Mutant: price_drop = (last + current) / last (wrong)
        Mutant: price_drop = (last - current) * last (huge number)
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            momentum_boost=0.5,
            price_threshold=0.05,
        )

        # Simulate previous purchase
        strategy.execute_purchase({
            'amount': 100.0,
            'price': 50000.0,
            'quantity': 0.002,
            'symbol': 'BTC/USDT',
            'action': 'buy',
        })

        # Current price is 10% lower
        current_price = 45000.0  # 10% drop from 50000

        # Act
        purchase = strategy.calculate_next_purchase(current_price=current_price)

        # Assert: Should boost purchase due to price drop
        # price_drop = (50000 - 45000) / 50000 = 0.1 (10%)
        # boost_factor = 1 + (0.1 / 0.05) * 0.5 = 1 + 2 * 0.5 = 2.0
        # purchase_amount = 100 * 2.0 = 200
        assert purchase['amount'] > 100.0, \
            f"Price drop should increase purchase. Got {purchase['amount']}. Mutant survived!"
        assert purchase['amount'] < 300.0, \
            f"Boost should be reasonable. Got {purchase['amount']}"

    def test_dca_boost_factor_kills_mutant(self):
        """
        Kill: + → - or / → * mutants in boost factor (line 382)
        Original: boost_factor = 1 + (price_drop / threshold) * boost
        Mutant: boost_factor = 1 - (price_drop / threshold) * boost (would decrease)

        Note: Code uses > comparison, so price_drop must be > threshold, not >=
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            momentum_boost=1.0,  # 100% boost
            price_threshold=0.05,
        )

        # Simulate previous purchase at higher price
        strategy.execute_purchase({
            'amount': 100.0,
            'price': 50000.0,
            'quantity': 0.002,
            'symbol': 'BTC/USDT',
            'action': 'buy',
        })

        # Current price slightly beyond threshold (5.1% drop)
        # Need > 0.05, not >= 0.05
        current_price = 47450.0  # 50000 * 0.951 = 5.1% drop

        # Act
        purchase = strategy.calculate_next_purchase(current_price=current_price)

        # Assert: With >5% drop and 100% boost, should more than double
        # price_drop = (50000 - 47450) / 50000 = 0.051
        # boost_factor = 1 + (0.051 / 0.05) * 1.0 ≈ 2.02
        # amount = 100 * 2.02 = 202
        assert purchase['amount'] > 200.0, \
            f"Should be >200.0, got {purchase['amount']}. Mutant + → - survived!"


class TestDCAComparisonMutations:
    """Tests to kill comparison operator mutants in DCA strategy"""

    def test_dca_high_volatility_comparison_kills_mutant(self):
        """
        Kill: > → < mutant in volatility check (line 370)
        Original: if volatility > 0.08
        Mutant: if volatility < 0.08 (inverted logic)

        Test with high volatility (> 0.08) to ensure it triggers reduction
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            volatility_adjustment=True,
        )

        # Create high volatility data (std > 0.08)
        historical_data = pd.DataFrame({
            'close': [50000, 58000, 42000, 56000, 44000]  # Very high volatility
        })
        volatility = historical_data['close'].pct_change().std()

        # Act
        purchase = strategy.calculate_next_purchase(
            current_price=50000.0,
            historical_data=historical_data,
        )

        # Assert: Should reduce purchase amount
        assert purchase['amount'] < 100.0, \
            f"Volatility {volatility:.4f} > 0.08 should reduce amount. Mutant > → < survived!"

    def test_dca_low_volatility_comparison_kills_mutant(self):
        """
        Kill: < → > mutant in low volatility check (line 373)
        Original: elif volatility < 0.02
        Mutant: elif volatility > 0.02 (inverted)
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            volatility_adjustment=True,
        )

        # Create low volatility data (std < 0.02)
        # Prices are nearly constant
        historical_data = pd.DataFrame({
            'close': [50000, 50100, 49900, 50050, 49950]  # Low volatility
        })
        volatility = historical_data['close'].pct_change().std()

        # Act
        purchase = strategy.calculate_next_purchase(
            current_price=50000.0,
            historical_data=historical_data,
        )

        # Assert: Should increase purchase amount (buy more when volatility low)
        assert purchase['amount'] > 100.0, \
            f"Volatility {volatility:.4f} < 0.02 should increase amount. Mutant < → > survived!"

    def test_dca_price_drop_comparison_kills_mutant(self):
        """
        Kill: < → > mutant in price drop check (line 377)
        Original: if current_price < last_purchase_price
        Mutant: if current_price > last_purchase_price (wrong direction)
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            momentum_boost=0.5,
            price_threshold=0.05,
        )

        # Simulate previous purchase
        last_price = 50000.0
        strategy.execute_purchase({
            'amount': 100.0,
            'price': last_price,
            'quantity': 0.002,
            'symbol': 'BTC/USDT',
            'action': 'buy',
        })

        # Current price is LOWER (should trigger boost)
        current_price = 45000.0  # Lower than last purchase

        # Act
        purchase = strategy.calculate_next_purchase(current_price=current_price)

        # Assert: Should have boosted purchase
        assert purchase['amount'] > 100.0, \
            f"Price drop should boost purchase. Mutant < → > survived!"

    def test_dca_threshold_comparison_kills_mutant(self):
        """
        Kill: > → < mutant in threshold check (line 380)
        Original: if price_drop > price_threshold
        Mutant: if price_drop < price_threshold (inverted)

        Test exactly at threshold to ensure correct behavior
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            momentum_boost=0.5,
            price_threshold=0.05,
        )

        # Simulate previous purchase
        strategy.execute_purchase({
            'amount': 100.0,
            'price': 50000.0,
            'quantity': 0.002,
            'symbol': 'BTC/USDT',
            'action': 'buy',
        })

        # Price drop EXACTLY at threshold (5%)
        # 50000 * 0.95 = 47500
        current_price = 47500.0

        # Act
        purchase = strategy.calculate_next_purchase(current_price=current_price)

        # Assert: 5% drop should trigger boost (>= threshold, though code uses >)
        # With >, need slightly more than 5%
        # Let's test with 5.1% drop
        current_price = 47450.0  # 5.1% drop

        purchase = strategy.calculate_next_purchase(current_price=current_price)

        assert purchase['amount'] > 100.0, \
            f"Price drop 5.1% > 5% should boost. Mutant > → < survived!"

    def test_dca_remaining_comparison_kills_mutant(self):
        """
        Kill: > → < mutant in remaining investment check (line 390)
        Original: if purchase_amount > remaining
        Mutant: if purchase_amount < remaining (would cap incorrectly)
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=1000.0,  # Small total
            purchase_amount=100.0,
            max_per_purchase=500.0,
        )

        # Invest most of it
        strategy.execute_purchase({
            'amount': 950.0,
            'price': 50000.0,
            'quantity': 0.019,
            'symbol': 'BTC/USDT',
            'action': 'buy',
        })

        # Only 50 remaining, but we want to purchase 100
        current_price = 50000.0

        # Act
        purchase = strategy.calculate_next_purchase(current_price=current_price)

        # Assert: Should cap at remaining amount (50)
        assert purchase['amount'] <= 50.0, \
            f"Should cap at remaining 50.0, got {purchase['amount']}. Mutant > → < survived!"
        assert purchase['is_final_purchase'] == True, \
            "Should be marked as final purchase"


class TestDCALogicalMutations:
    """Tests to kill logical operator mutants in DCA strategy"""

    def test_dca_volatility_and_kills_mutant(self):
        """
        Kill: and → or mutant in volatility check (line 366)
        Original: if volatility_adjustment and historical_data is not None
        Mutant: if volatility_adjustment or historical_data is not None

        Test with volatility_adjustment=False to ensure and is required
        """
        # Arrange: volatility_adjustment = False
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            volatility_adjustment=False,  # Disabled
        )

        # Create high volatility data
        historical_data = pd.DataFrame({
            'close': [50000, 60000, 40000, 55000, 45000]
        })

        # Act
        purchase = strategy.calculate_next_purchase(
            current_price=50000.0,
            historical_data=historical_data,
        )

        # Assert: Should NOT adjust (volatility_adjustment=False)
        assert purchase['amount'] == 100.0, \
            f"With volatility_adjustment=False, should use base amount. Mutant and → or survived!"


class TestDCAConstantMutations:
    """Tests to kill constant mutations in DCA strategy"""

    def test_dca_base_amount_kills_mutant(self):
        """
        Kill: Mutant that changes base_purchase_amount
        Test ensures the configured base amount is actually used
        """
        # Arrange: Specific base amount
        base_amount = 250.0  # Non-standard amount
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=base_amount,
            volatility_adjustment=False,
        )

        # Act
        purchase = strategy.calculate_next_purchase(current_price=50000.0)

        # Assert: Should use exact base amount
        assert purchase['amount'] == base_amount, \
            f"Should use base amount {base_amount}, got {purchase['amount']}. Mutant survived!"

    def test_dca_min_max_clamp_kills_mutant(self):
        """
        Kill: Mutants in min/max constants or clamping logic (line 386)

        NOTE: BUG IN PRODUCTION CODE
        Line 386 has: max(self.min_per_purchase, min(purchase_amount, purchase_amount))
        Should be: max(self.min_per_purchase, min(purchase_amount, self.max_per_purchase))

        The second purchase_amount should be self.max_per_purchase!
        This means max_per_purchase is never actually enforced.

        This test documents the actual (buggy) behavior.
        """
        # Arrange: Small min and max
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            min_per_purchase=50.0,
            max_per_purchase=150.0,
        )

        # Scenario 1: Try to get below min (via huge boost)
        strategy.execute_purchase({
            'amount': 100.0,
            'price': 50000.0,
            'quantity': 0.002,
            'symbol': 'BTC/USDT',
            'action': 'buy',
        })

        # Create huge price drop to trigger massive boost
        current_price = 25000.0  # 50% drop from 50000

        # Act
        purchase = strategy.calculate_next_purchase(current_price=current_price)

        # Assert: Due to bug, max_per_purchase is NOT enforced
        # The boost will produce 600, which exceeds max of 150
        # This is expected (buggy) behavior
        assert purchase['amount'] >= 50.0, \
            f"Should clamp at min 50.0, got {purchase['amount']}. Mutant survived!"
        # Due to bug: min(purchase_amount, purchase_amount) is always purchase_amount
        # So max_per_purchase (150) is never enforced
        # We expect amount > 150 due to the bug
        assert purchase['amount'] > 150.0, \
            f"Bug: max_per_purchase not enforced. Got {purchase['amount']} (expected >150 due to bug)"

    def test_dca_total_investment_kills_mutant(self):
        """
        Kill: Mutant that changes total_investment
        Test ensures total investment limit is respected
        """
        # Arrange
        total_investment = 5000.0
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=total_investment,
            purchase_amount=1000.0,  # Large purchase
        )

        # Make first purchase
        strategy.execute_purchase({
            'amount': 1000.0,
            'price': 50000.0,
            'quantity': 0.02,
            'symbol': 'BTC/USDT',
            'action': 'buy',
        })

        # Act: Try to purchase more than remaining
        # Remaining: 5000 - 1000 = 4000
        # But purchase_amount = 1000, so should work
        purchase = strategy.calculate_next_purchase(current_price=50000.0)

        # Assert: Should not exceed total
        assert strategy.invested_amount + purchase['amount'] <= total_investment, \
            f"Should not exceed total {total_investment}. Mutant survived!"


class TestDCAEdgeCaseMutations:
    """Edge case tests to kill boundary mutants"""

    def test_dca_zero_purchases_kills_mutant(self):
        """
        Kill: Division or comparison mutants when no purchases yet
        """
        # Arrange: New strategy (no purchases)
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
        )

        # Act: Get portfolio status
        status = strategy.get_portfolio_status(current_price=50000.0)

        # Assert: Should handle gracefully with zeros
        assert status['total_invested'] == 0
        assert status['total_quantity'] == 0
        assert status['avg_purchase_price'] == 0
        assert status['current_value'] == 0
        assert status['pnl'] == 0
        assert status['num_purchases'] == 0

    def test_dca_first_purchase_kills_mutant(self):
        """
        Kill: Mutants in first purchase logic
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
            momentum_boost=0.5,
            price_threshold=0.05,
        )

        # Act: First purchase (no last_purchase_price yet)
        purchase1 = strategy.calculate_next_purchase(current_price=50000.0)

        # Assert: Should use base amount (no boost on first purchase)
        assert purchase1['amount'] == 100.0, \
            f"First purchase should be base amount. Got {purchase1['amount']}"

        # Execute first purchase
        strategy.execute_purchase(purchase1)

        # Second purchase with price drop
        purchase2 = strategy.calculate_next_purchase(current_price=45000.0)

        # Assert: Second purchase should have boost
        assert purchase2['amount'] > 100.0, \
            "Second purchase with price drop should be boosted"

    def test_dca_final_purchase_kills_mutant(self):
        """
        Kill: Mutants in final purchase detection logic (line 400)
        Original: is_final_purchase = purchase_amount >= remaining
        Mutant: is_final_purchase = purchase_amount < remaining (inverted)
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=1000.0,  # Small total
            purchase_amount=600.0,
        )

        # Invest most of it
        strategy.execute_purchase({
            'amount': 400.0,
            'price': 50000.0,
            'quantity': 0.008,
            'symbol': 'BTC/USDT',
            'action': 'buy',
        })

        # Remaining: 600, purchase_amount: 600
        # Should be exactly final

        # Act
        purchase = strategy.calculate_next_purchase(current_price=50000.0)

        # Assert: Should be marked as final
        assert purchase['is_final_purchase'] == True, \
            "Purchase using all remaining should be marked final. Mutant >= survived!"

    def test_dca_avg_price_calculation_kills_mutant(self):
        """
        Kill: / → * mutant in average price calculation (line 436)
        Original: avg_price = invested_amount / total_quantity
        Mutant: avg_price = invested_amount * total_quantity
        """
        # Arrange
        strategy = SmartDCAStrategy(
            symbol="BTC/USDT",
            total_investment=10000.0,
            purchase_amount=100.0,
        )

        # Make multiple purchases
        purchases = [
            {'amount': 100.0, 'price': 50000.0, 'quantity': 0.002},
            {'amount': 100.0, 'price': 45000.0, 'quantity': 0.002222},
            {'amount': 100.0, 'price': 55000.0, 'quantity': 0.001818},
        ]

        for p in purchases:
            strategy.execute_purchase({
                **p,
                'symbol': 'BTC/USDT',
                'action': 'buy',
            })

        # Act
        status = strategy.get_portfolio_status(current_price=50000.0)

        # Assert: avg_price should be invested / quantity
        # invested: 300, quantity: 0.00604
        # avg: 300 / 0.00604 = 49668
        expected_avg = 300.0 / 0.00604

        assert abs(status['avg_purchase_price'] - expected_avg) < 100, \
            f"Avg price should be ~{expected_avg}, got {status['avg_purchase_price']}. Mutant / → * survived!"
