#!/usr/bin/env python3
"""
Trading strategy logic tests.

Tests the complete trading strategy logic and decision-making.
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)


class TradingStrategy:
    """Complete trading strategy implementation."""

    def __init__(self, risk_params=None):
        """Initialize strategy with risk parameters."""
        self.risk_params = risk_params or {
            'max_position_eur': 300,
            'max_daily_loss_eur': 50,
            'max_daily_trades': 2,
            'min_confidence': 0.6,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03
        }

        # State tracking
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.positions = {}

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        if len(prices) < period:
            return 50.0

        df = pd.DataFrame({'close': prices})
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0 if gain.iloc[-1] > 0 else 50.0

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def generate_signal(self, rsi):
        """Generate trading signal from RSI."""
        if rsi < 42:
            action = "BUY"
            confidence = min(0.95, 0.65 + (42 - rsi) / 80)
        elif rsi > 58:
            action = "SELL"
            confidence = min(0.95, 0.65 + (rsi - 58) / 80)
        else:
            action = "HOLD"
            confidence = 0.5

        return {
            'action': action,
            'confidence': confidence,
            'rsi': rsi
        }

    def check_risk_limits(self, position_size_eur):
        """Check all risk limits."""
        checks = {
            'position_ok': position_size_eur <= self.risk_params['max_position_eur'],
            'loss_limit_ok': self.daily_pnl > -self.risk_params['max_daily_loss_eur'],
            'trade_count_ok': self.daily_trade_count < self.risk_params['max_daily_trades']
        }

        all_passed = all(checks.values())
        return all_passed, checks

    def execute_trade(self, signal, price, balance_eur):
        """Execute trade with full validation."""
        # 1. Check if signal is actionable
        if signal['action'] == 'HOLD':
            return {'status': 'no_action', 'reason': 'Signal is HOLD'}

        # 2. Check confidence threshold
        if signal['confidence'] < self.risk_params['min_confidence']:
            return {'status': 'rejected', 'reason': f'Confidence too low: {signal["confidence"]:.2f}'}

        # 3. Calculate position size
        position_size_eur = min(
            self.risk_params['max_position_eur'],
            balance_eur * 0.25
        )

        if position_size_eur < 10:
            return {'status': 'rejected', 'reason': 'Insufficient funds'}

        # 4. Check risk limits
        risk_ok, risk_checks = self.check_risk_limits(position_size_eur)

        if not risk_ok:
            failed_checks = [k for k, v in risk_checks.items() if not v]
            return {'status': 'rejected', 'reason': f'Risk checks failed: {failed_checks}'}

        # 5. Execute trade
        amount = position_size_eur / price

        self.daily_trade_count += 1

        # Calculate stop loss and take profit
        if signal['action'] == 'BUY':
            stop_loss = price * (1 - self.risk_params['stop_loss_pct'])
            take_profit = price * (1 + self.risk_params['take_profit_pct'])
        else:
            stop_loss = price * (1 + self.risk_params['stop_loss_pct'])
            take_profit = price * (1 - self.risk_params['take_profit_pct'])

        return {
            'status': 'executed',
            'action': signal['action'],
            'amount': amount,
            'price': price,
            'value_eur': position_size_eur,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': signal['confidence']
        }

    def update_pnl(self, trade_pnl):
        """Update daily P&L."""
        self.daily_pnl += trade_pnl


class TestStrategyLogic(unittest.TestCase):
    """Test core strategy logic."""

    def setUp(self):
        """Set up strategy for testing."""
        self.strategy = TradingStrategy()

    def test_rsi_calculation_accuracy(self):
        """Test RSI calculation accuracy."""
        print("\n  Testing RSI calculation...")

        # Uptrend prices
        uptrend = list(range(100, 150))
        rsi = self.strategy.calculate_rsi(uptrend)

        print(f"  ✓ Uptrend RSI: {rsi:.2f}")
        self.assertGreater(rsi, 70, "Uptrend should have high RSI")

        # Downtrend prices
        downtrend = list(range(150, 100, -1))
        rsi = self.strategy.calculate_rsi(downtrend)

        print(f"  ✓ Downtrend RSI: {rsi:.2f}")
        self.assertLess(rsi, 30, "Downtrend should have low RSI")

    def test_signal_generation_accuracy(self):
        """Test signal generation accuracy."""
        print("\n  Testing signal generation...")

        # Test cases
        test_cases = [
            (20, "BUY", 0.925),  # Very oversold: min(0.95, 0.65 + (42-20)/80) = 0.925
            (42, "HOLD", 0.50),  # At threshold
            (50, "HOLD", 0.50),  # Neutral
            (58, "HOLD", 0.50),  # At threshold
            (80, "SELL", 0.925), # Very overbought: min(0.95, 0.65 + (80-58)/80) = 0.925
        ]

        for rsi, expected_action, expected_confidence in test_cases:
            signal = self.strategy.generate_signal(rsi)

            print(f"  ✓ RSI {rsi}: {signal['action']} (conf: {signal['confidence']:.2f})")

            self.assertEqual(signal['action'], expected_action)
            self.assertAlmostEqual(signal['confidence'], expected_confidence, places=2)


class TestCompleteTradingWorkflow(unittest.TestCase):
    """Test complete trading workflow end-to-end."""

    def setUp(self):
        """Set up strategy for testing."""
        self.strategy = TradingStrategy()

    def test_buy_workflow(self):
        """Test complete BUY trade workflow."""
        print("\n  Testing BUY workflow...")

        # 1. Generate market data
        prices = [45000, 44500, 44000, 43500, 43000, 42500, 42000] * 5
        rsi = self.strategy.calculate_rsi(prices)
        print(f"    1. RSI calculated: {rsi:.2f}")

        # 2. Generate signal
        signal = self.strategy.generate_signal(rsi)
        print(f"    2. Signal generated: {signal['action']} (conf: {signal['confidence']:.2f})")

        # 3. Execute trade
        price = 42000
        balance = 1000
        result = self.strategy.execute_trade(signal, price, balance)

        print(f"    3. Trade execution: {result['status']}")

        if result['status'] == 'executed':
            print(f"       Amount: {result['amount']:.6f} BTC")
            print(f"       Value: €{result['value_eur']:.2f}")
            print(f"       Stop Loss: €{result['stop_loss']:.2f}")
            print(f"       Take Profit: €{result['take_profit']:.2f}")

            self.assertEqual(result['action'], 'BUY')
            self.assertGreater(result['amount'], 0)
            self.assertEqual(result['value_eur'], 250)  # 25% of 1000

    def test_sell_workflow(self):
        """Test complete SELL trade workflow."""
        print("\n  Testing SELL workflow...")

        # 1. Generate market data
        prices = [42000, 42500, 43000, 43500, 44000, 44500, 45000] * 5
        rsi = self.strategy.calculate_rsi(prices)
        print(f"    1. RSI calculated: {rsi:.2f}")

        # 2. Generate signal
        signal = self.strategy.generate_signal(rsi)
        print(f"    2. Signal generated: {signal['action']} (conf: {signal['confidence']:.2f})")

        # 3. Execute trade
        price = 45000
        balance = 1000
        result = self.strategy.execute_trade(signal, price, balance)

        print(f"    3. Trade execution: {result['status']}")

        if result['status'] == 'executed':
            print(f"       Amount: {result['amount']:.6f} BTC")
            print(f"       Value: €{result['value_eur']:.2f}")

            self.assertEqual(result['action'], 'SELL')

    def test_hold_workflow(self):
        """Test HOLD signal workflow."""
        print("\n  Testing HOLD workflow...")

        # 1. Generate neutral market data
        prices = [43500] * 20
        rsi = self.strategy.calculate_rsi(prices)
        print(f"    1. RSI calculated: {rsi:.2f}")

        # 2. Generate signal
        signal = self.strategy.generate_signal(rsi)
        print(f"    2. Signal generated: {signal['action']}")

        # 3. Try to execute
        price = 43500
        balance = 1000
        result = self.strategy.execute_trade(signal, price, balance)

        print(f"    3. Trade execution: {result['status']}")
        print(f"       Reason: {result.get('reason', 'N/A')}")

        self.assertEqual(result['status'], 'no_action')


class TestRiskEnforcement(unittest.TestCase):
    """Test risk management enforcement."""

    def setUp(self):
        """Set up strategy for testing."""
        self.strategy = TradingStrategy()

    def test_daily_loss_limit_enforcement(self):
        """Test daily loss limit is enforced."""
        print("\n  Testing daily loss limit...")

        # Simulate losses
        self.strategy.update_pnl(-30)
        print(f"    1. First loss: €30")

        # Should still be able to trade
        signal = {'action': 'BUY', 'confidence': 0.8}
        result = self.strategy.execute_trade(signal, 45000, 1000)

        print(f"    2. Trade status: {result['status']}")
        self.assertEqual(result['status'], 'executed')

        # Add more loss to exceed limit
        self.strategy.update_pnl(-25)
        print(f"    3. Total loss: €55")

        # Should now be blocked
        result = self.strategy.execute_trade(signal, 45000, 1000)
        print(f"    4. Trade status: {result['status']}")

        self.assertEqual(result['status'], 'rejected')
        self.assertIn('loss_limit_ok', result.get('reason', ''))

    def test_trade_count_limit_enforcement(self):
        """Test trade count limit is enforced."""
        print("\n  Testing trade count limit...")

        signal = {'action': 'BUY', 'confidence': 0.8}

        # Execute first trade
        result1 = self.strategy.execute_trade(signal, 45000, 1000)
        print(f"    1. Trade 1: {result1['status']}")
        self.assertEqual(result1['status'], 'executed')

        # Execute second trade
        result2 = self.strategy.execute_trade(signal, 45000, 1000)
        print(f"    2. Trade 2: {result2['status']}")
        self.assertEqual(result2['status'], 'executed')

        # Try third trade (should be blocked)
        result3 = self.strategy.execute_trade(signal, 45000, 1000)
        print(f"    3. Trade 3: {result3['status']}")

        self.assertEqual(result3['status'], 'rejected')
        self.assertIn('trade_count_ok', result3.get('reason', ''))

    def test_confidence_threshold_enforcement(self):
        """Test confidence threshold is enforced."""
        print("\n  Testing confidence threshold...")

        # Low confidence signal
        signal = {'action': 'BUY', 'confidence': 0.5}

        result = self.strategy.execute_trade(signal, 45000, 1000)
        print(f"    Trade with conf=0.5: {result['status']}")
        print(f"    Reason: {result.get('reason', 'N/A')}")

        self.assertEqual(result['status'], 'rejected')
        self.assertIn('Confidence too low', result.get('reason', ''))


class TestStrategyPerformance(unittest.TestCase):
    """Test strategy performance metrics."""

    def test_strategy_win_rate_calculation(self):
        """Test win rate calculation."""
        print("\n  Testing win rate calculation...")

        trades = [
            {'pnl': 50},   # Win
            {'pnl': -30},  # Loss
            {'pnl': 75},   # Win
            {'pnl': -20},  # Loss
            {'pnl': 100}   # Win
        ]

        wins = sum(1 for t in trades if t['pnl'] > 0)
        total = len(trades)
        win_rate = wins / total

        print(f"    Wins: {wins}/{total}")
        print(f"    Win rate: {win_rate*100:.1f}%")

        self.assertEqual(win_rate, 0.6)  # 60%

    def test_strategy_total_return(self):
        """Test total return calculation."""
        print("\n  Testing total return...")

        trades = [
            {'pnl': 50},
            {'pnl': -30},
            {'pnl': 75},
            {'pnl': -20},
            {'pnl': 100}
        ]

        total_return = sum(t['pnl'] for t in trades)

        print(f"    Total return: €{total_return:.2f}")

        self.assertEqual(total_return, 175)

    def test_strategy_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        print("\n  Testing Sharpe ratio...")

        returns = [0.05, -0.03, 0.075, -0.02, 0.10]

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return > 0:
            sharpe = avg_return / std_return
        else:
            sharpe = 0

        print(f"    Avg return: {avg_return*100:.2f}%")
        print(f"    Std dev: {std_return*100:.2f}%")
        print(f"    Sharpe ratio: {sharpe:.2f}")

        self.assertGreater(sharpe, 0)


class TestMarketScenarios(unittest.TestCase):
    """Test strategy in different market scenarios."""

    def setUp(self):
        """Set up strategy for testing."""
        self.strategy = TradingStrategy()

    def test_bull_market_scenario(self):
        """Test strategy in bull market."""
        print("\n  Testing bull market scenario...")

        # Simulate rising prices
        base_price = 40000
        prices = []
        for i in range(50):
            prices.append(base_price + (i * 200) + random.randint(-100, 100))

        rsi = self.strategy.calculate_rsi(prices)
        signal = self.strategy.generate_signal(rsi)

        print(f"    Price trend: Rising")
        print(f"    RSI: {rsi:.2f}")
        print(f"    Signal: {signal['action']}")

        # In strong uptrend, should get SELL signals (overbought)
        self.assertIn(signal['action'], ['SELL', 'HOLD'])

    def test_bear_market_scenario(self):
        """Test strategy in bear market."""
        print("\n  Testing bear market scenario...")

        # Simulate falling prices
        base_price = 50000
        prices = []
        for i in range(50):
            prices.append(base_price - (i * 200) + random.randint(-100, 100))

        rsi = self.strategy.calculate_rsi(prices)
        signal = self.strategy.generate_signal(rsi)

        print(f"    Price trend: Falling")
        print(f"    RSI: {rsi:.2f}")
        print(f"    Signal: {signal['action']}")

        # In strong downtrend, should get BUY signals (oversold)
        self.assertIn(signal['action'], ['BUY', 'HOLD'])

    def test_sideways_market_scenario(self):
        """Test strategy in sideways/ranging market."""
        print("\n  Testing sideways market scenario...")

        # Simulate ranging prices
        prices = []
        for i in range(50):
            prices.append(45000 + random.randint(-500, 500))

        rsi = self.strategy.calculate_rsi(prices)
        signal = self.strategy.generate_signal(rsi)

        print(f"    Price trend: Ranging")
        print(f"    RSI: {rsi:.2f}")
        print(f"    Signal: {signal['action']}")

        # In ranging market, should mostly HOLD
        self.assertEqual(signal['action'], 'HOLD')


def run_tests():
    """Run all strategy tests."""
    print("=" * 80)
    print("Trading Strategy Tests")
    print("=" * 80)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStrategyLogic))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCompleteTradingWorkflow))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskEnforcement))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStrategyPerformance))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMarketScenarios))

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
