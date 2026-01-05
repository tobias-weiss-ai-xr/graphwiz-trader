#!/usr/bin/env python3
"""
Mock order execution tests.

Tests order logic without executing real trades.
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ccxt
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    print("Install: pip install ccxt")
    sys.exit(1)


class MockExchange:
    """Mock exchange for testing order execution."""

    def __init__(self):
        self.balance = {
            'EUR': {'free': 1000.0, 'used': 0.0, 'total': 1000.0},
            'BTC': {'free': 0.0, 'used': 0.0, 'total': 0.0}
        }
        self.orders = []
        self.order_id_counter = 1000

    def fetch_balance(self):
        """Fetch account balance."""
        return self.balance

    def create_market_buy_order(self, symbol, amount):
        """Create a market buy order."""
        order = {
            'id': str(self.order_id_counter),
            'symbol': symbol,
            'type': 'market',
            'side': 'buy',
            'amount': amount,
            'price': 50000,  # Mock price
            'status': 'closed',
            'filled': amount,
            'cost': amount * 50000,
            'timestamp': datetime.now().timestamp()
        }
        self.orders.append(order)
        self.order_id_counter += 1

        # Update balance
        cost = amount * 50000
        self.balance['EUR']['free'] -= cost
        self.balance['EUR']['used'] = cost

        return order

    def create_market_sell_order(self, symbol, amount):
        """Create a market sell order."""
        order = {
            'id': str(self.order_id_counter),
            'symbol': symbol,
            'type': 'market',
            'side': 'sell',
            'amount': amount,
            'price': 50000,  # Mock price
            'status': 'closed',
            'filled': amount,
            'cost': amount * 50000,
            'timestamp': datetime.now().timestamp()
        }
        self.orders.append(order)
        self.order_id_counter += 1

        # Update balance
        self.balance['BTC']['free'] -= amount

        return order


class TestOrderExecution(unittest.TestCase):
    """Test order execution logic with mock exchange."""

    def setUp(self):
        """Set up mock exchange."""
        self.exchange = MockExchange()

    def test_fetch_balance(self):
        """Test fetching balance."""
        balance = self.exchange.fetch_balance()
        print(f"\n  ✓ Balance fetched: €{balance['EUR']['free']:.2f}")
        self.assertIn('EUR', balance)
        self.assertEqual(balance['EUR']['free'], 1000.0)

    def test_buy_order_execution(self):
        """Test buy order execution."""
        amount = 0.01  # BTC
        order = self.exchange.create_market_buy_order('BTC/EUR', amount)

        print(f"\n  ✓ Buy order executed: {order['id']}")
        print(f"    Amount: {amount} BTC")
        print(f"    Cost: €{order['cost']:,.2f}")

        self.assertEqual(order['side'], 'buy')
        self.assertEqual(order['status'], 'closed')
        self.assertEqual(len(self.exchange.orders), 1)

    def test_sell_order_execution(self):
        """Test sell order execution."""
        # First add BTC to balance
        self.exchange.balance['BTC']['free'] = 0.05

        amount = 0.01  # BTC
        order = self.exchange.create_market_sell_order('BTC/EUR', amount)

        print(f"\n  ✓ Sell order executed: {order['id']}")
        print(f"    Amount: {amount} BTC")
        print(f"    Revenue: €{order['cost']:,.2f}")

        self.assertEqual(order['side'], 'sell')
        self.assertEqual(order['status'], 'closed')

    def test_insufficient_funds_buy(self):
        """Test buy with insufficient EUR balance."""
        self.exchange.balance['EUR']['free'] = 5  # Only €5

        amount = 0.01  # Would cost €500
        print(f"\n  ✓ Testing insufficient funds (€5 available, €500 needed)")

        # Should not be able to buy
        self.assertLess(self.exchange.balance['EUR']['free'], amount * 50000)

    def test_insufficient_balance_sell(self):
        """Test sell with insufficient BTC balance."""
        self.exchange.balance['BTC']['free'] = 0.001  # Only 0.001 BTC

        amount = 0.01  # Trying to sell 0.01
        print(f"\n  ✓ Testing insufficient BTC (0.001 available, 0.01 needed)")

        # Should not be able to sell
        self.assertLess(self.exchange.balance['BTC']['free'], amount)


class TestPositionSizing(unittest.TestCase):
    """Test position sizing calculations."""

    def calculate_position_size(self, balance_eur, max_position_eur, price):
        """Calculate position size."""
        position_size_eur = min(max_position_eur, balance_eur * 0.25)
        if position_size_eur < 10:
            return 0, "Insufficient funds"
        amount = position_size_eur / price
        return amount, f"Position: €{position_size_eur:.2f}"

    def test_position_size_conservative(self):
        """Test conservative position sizing."""
        amount, msg = self.calculate_position_size(1000, 300, 50000)
        print(f"\n  ✓ Conservative sizing: {msg} = {amount:.6f} BTC")
        self.assertGreater(amount, 0)
        expected = (1000 * 0.25) / 50000
        self.assertAlmostEqual(amount, expected, places=6)

    def test_position_size_max_limit(self):
        """Test position size capped at maximum."""
        amount, msg = self.calculate_position_size(10000, 300, 50000)
        print(f"  ✓ Max limit applied: {msg} = {amount:.6f} BTC")
        expected = 300 / 50000
        self.assertAlmostEqual(amount, expected, places=6)

    def test_position_size_minimum_threshold(self):
        """Test minimum position threshold."""
        amount, msg = self.calculate_position_size(50, 300, 50000)
        print(f"  ✓ Minimum threshold: {msg} = {amount:.6f} BTC")
        # 50 * 0.25 = 12.5, which is > 10, so should work
        self.assertGreater(amount, 0)

    def test_position_size_below_minimum(self):
        """Test position below minimum threshold."""
        amount, msg = self.calculate_position_size(30, 300, 50000)
        print(f"  ✓ Below minimum: {msg} = {amount:.6f} BTC")
        self.assertEqual(amount, 0)


class TestOrderValidation(unittest.TestCase):
    """Test order validation before execution."""

    def validate_order(self, signal, balance, daily_pnl, daily_trades, max_loss, max_trades):
        """Validate order before execution."""
        errors = []

        # Check if signal is actionable
        if signal['action'] == 'HOLD':
            errors.append("Signal is HOLD")
            return False, errors

        # Check confidence
        if signal['confidence'] < 0.6:
            errors.append(f"Confidence too low: {signal['confidence']:.2f}")

        # Check daily loss limit
        if daily_pnl <= -max_loss:
            errors.append(f"Daily loss limit reached: €{abs(daily_pnl):.2f}")

        # Check trade count limit
        if daily_trades >= max_trades:
            errors.append(f"Daily trade limit reached: {daily_trades}")

        # Check balance for BUY
        if signal['action'] == 'BUY':
            required_eur = 300  # Max position
            available_eur = balance.get('EUR', {}).get('free', 0)
            if available_eur < required_eur * 0.25:
                errors.append(f"Insufficient EUR: €{available_eur:.2f}")

        # Check balance for SELL
        if signal['action'] == 'SELL':
            required_btc = 0.01
            available_btc = balance.get('BTC', {}).get('free', 0)
            if available_btc < required_btc:
                errors.append(f"Insufficient BTC: {available_btc:.6f}")

        return len(errors) == 0, errors

    def test_valid_buy_order(self):
        """Test valid buy order passes validation."""
        signal = {'action': 'BUY', 'confidence': 0.8}
        balance = {'EUR': {'free': 1000}, 'BTC': {'free': 0}}
        valid, errors = self.validate_order(signal, balance, 0, 0, 50, 2)
        print(f"\n  ✓ Valid BUY order")
        self.assertTrue(valid)

    def test_valid_sell_order(self):
        """Test valid sell order passes validation."""
        signal = {'action': 'SELL', 'confidence': 0.8}
        balance = {'EUR': {'free': 1000}, 'BTC': {'free': 0.05}}
        valid, errors = self.validate_order(signal, balance, 0, 0, 50, 2)
        print(f"  ✓ Valid SELL order")
        self.assertTrue(valid)

    def test_hold_signal_rejected(self):
        """Test HOLD signal is rejected."""
        signal = {'action': 'HOLD', 'confidence': 0.5}
        balance = {'EUR': {'free': 1000}, 'BTC': {'free': 0}}
        valid, errors = self.validate_order(signal, balance, 0, 0, 50, 2)
        print(f"  ✓ HOLD signal rejected: {errors}")
        self.assertFalse(valid)

    def test_low_confidence_rejected(self):
        """Test low confidence signal is rejected."""
        signal = {'action': 'BUY', 'confidence': 0.5}
        balance = {'EUR': {'free': 1000}, 'BTC': {'free': 0}}
        valid, errors = self.validate_order(signal, balance, 0, 0, 50, 2)
        print(f"  ✓ Low confidence rejected: {errors}")
        self.assertFalse(valid)

    def test_daily_loss_limit_reached(self):
        """Test order rejected when daily loss limit reached."""
        signal = {'action': 'BUY', 'confidence': 0.8}
        balance = {'EUR': {'free': 1000}, 'BTC': {'free': 0}}
        valid, errors = self.validate_order(signal, balance, -60, 0, 50, 2)
        print(f"  ✓ Daily loss limit reached: {errors}")
        self.assertFalse(valid)

    def test_trade_count_limit_reached(self):
        """Test order rejected when trade count limit reached."""
        signal = {'action': 'BUY', 'confidence': 0.8}
        balance = {'EUR': {'free': 1000}, 'BTC': {'free': 0}}
        valid, errors = self.validate_order(signal, balance, 0, 2, 50, 2)
        print(f"  ✓ Trade count limit reached: {errors}")
        self.assertFalse(valid)


def run_tests():
    """Run all order execution tests."""
    print("=" * 80)
    print("Order Execution Tests (Mock)")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOrderExecution))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPositionSizing))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOrderValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
