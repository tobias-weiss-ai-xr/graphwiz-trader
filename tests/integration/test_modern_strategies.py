"""Integration tests for modern trading strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from graphwiz_trader.strategies.modern_strategies import (
    GridTradingStrategy,
    GridTradingMode,
    SmartDCAStrategy,
    AutomatedMarketMakingStrategy,
    TriangularArbitrageStrategy,
    create_modern_strategy,
)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')

    # Generate ranging market data (good for grid trading)
    base_price = 50000
    prices = []

    for i in range(100):
        # Mean-reverting price movement
        deviation = np.sin(i / 10) * 2000
        noise = np.random.normal(0, 500)
        price = base_price + deviation + noise
        prices.append(price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': [p * (1 + np.random.uniform(-0.002, 0.002)) for p in prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.005))) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in range(100)],
    })
    df.set_index('timestamp', inplace=True)

    return df


class TestGridTradingStrategy:
    """Tests for GridTradingStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = GridTradingStrategy(
            symbol='BTC/USDT',
            upper_price=60000,
            lower_price=40000,
            num_grids=10,
        )

        assert strategy.symbol == 'BTC/USDT'
        assert strategy.upper_price == 60000
        assert strategy.lower_price == 40000
        assert strategy.num_grids == 10
        assert len(strategy.grid_levels) == 11  # num_grids + 1

    def test_arithmetic_grid(self):
        """Test arithmetic grid spacing."""
        strategy = GridTradingStrategy(
            symbol='BTC/USDT',
            upper_price=60000,
            lower_price=40000,
            num_grids=10,
            grid_mode=GridTradingMode.ARITHMETIC,
        )

        # Check equal spacing
        expected_step = 2000  # (60000 - 40000) / 10
        for i in range(len(strategy.grid_levels) - 1):
            diff = strategy.grid_levels[i + 1] - strategy.grid_levels[i]
            assert abs(diff - expected_step) < 1  # Small tolerance for float

    def test_geometric_grid(self):
        """Test geometric grid spacing."""
        strategy = GridTradingStrategy(
            symbol='BTC/USDT',
            upper_price=60000,
            lower_price=40000,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
        )

        # Check percentage spacing
        ratio = strategy.grid_levels[1] / strategy.grid_levels[0]
        for i in range(len(strategy.grid_levels) - 1):
            current_ratio = strategy.grid_levels[i + 1] / strategy.grid_levels[i]
            assert abs(current_ratio - ratio) < 0.001

    def test_position_sizes_geometric(self):
        """Test position size calculation for geometric grid."""
        strategy = GridTradingStrategy(
            symbol='BTC/USDT',
            upper_price=60000,
            lower_price=40000,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000,
        )

        position_sizes = strategy.calculate_position_sizes()

        # Check we have position sizes for all grids except top
        assert len(position_sizes) == len(strategy.grid_levels) - 1

        # All position sizes should be positive
        assert all(size > 0 for size in position_sizes.values())

    def test_signal_generation(self, sample_price_data):
        """Test signal generation."""
        strategy = GridTradingStrategy(
            symbol='BTC/USDT',
            upper_price=55000,
            lower_price=45000,
            num_grids=10,
        )

        current_price = 50000
        signals = strategy.generate_signals(current_price)

        # Check signal structure
        assert 'current_price' in signals
        assert 'grid_levels' in signals
        assert 'orders_to_place' in signals
        assert signals['current_price'] == current_price

        # Should have orders to place
        assert len(signals['orders_to_place']) > 0

    def test_dynamic_rebalancing(self, sample_price_data):
        """Test dynamic grid rebalancing."""
        strategy = GridTradingStrategy(
            symbol='BTC/USDT',
            upper_price=55000,
            lower_price=45000,
            num_grids=10,
            dynamic_rebalancing=True,
            volatility_threshold=0.02,
        )

        current_price = 50000
        signals = strategy.generate_signals(current_price, sample_price_data)

        # Check volatility calculation
        if 'rebalance_needed' in signals:
            assert isinstance(signals['rebalance_needed'], bool)


class TestSmartDCAStrategy:
    """Tests for SmartDCAStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = SmartDCAStrategy(
            symbol='BTC/USDT',
            total_investment=10000,
            purchase_amount=100,
        )

        assert strategy.symbol == 'BTC/USDT'
        assert strategy.total_investment == 10000
        assert strategy.base_purchase_amount == 100
        assert strategy.invested_amount == 0

    def test_basic_purchase(self):
        """Test basic purchase calculation."""
        strategy = SmartDCAStrategy(
            symbol='BTC/USDT',
            total_investment=10000,
            purchase_amount=100,
        )

        current_price = 50000
        purchase = strategy.calculate_next_purchase(current_price)

        assert purchase['action'] == 'buy'
        assert purchase['symbol'] == 'BTC/USDT'
        assert purchase['amount'] == 100
        assert purchase['price'] == current_price
        assert purchase['quantity'] == 100 / current_price

    def test_momentum_boost(self):
        """Test momentum boost on price drop."""
        strategy = SmartDCAStrategy(
            symbol='BTC/USDT',
            total_investment=10000,
            purchase_amount=100,
            momentum_boost=0.5,
            price_threshold=0.05,
        )

        # First purchase at higher price
        strategy.execute_purchase({
            'action': 'buy',
            'amount': 100,
            'price': 55000,
            'quantity': 100 / 55000,
            'symbol': 'BTC/USDT',
        })

        # Calculate next purchase when price dropped
        current_price = 50000  # ~9% drop
        purchase = strategy.calculate_next_purchase(current_price)

        # Should be boosted
        assert purchase['amount'] > 100

    def test_volatility_adjustment(self, sample_price_data):
        """Test volatility-based purchase adjustment."""
        strategy = SmartDCAStrategy(
            symbol='BTC/USDT',
            total_investment=10000,
            purchase_amount=100,
            volatility_adjustment=True,
        )

        # Low volatility - should buy more
        low_vol_data = sample_price_data.copy()
        low_vol_data['close'] = low_vol_data['close'].rolling(5).mean()

        purchase = strategy.calculate_next_purchase(50000, low_vol_data)

        # Amount should be adjusted
        assert isinstance(purchase['amount'], float)

    def test_execute_purchase(self):
        """Test purchase execution."""
        strategy = SmartDCAStrategy(
            symbol='BTC/USDT',
            total_investment=10000,
            purchase_amount=100,
        )

        purchase = {
            'action': 'buy',
            'amount': 100,
            'price': 50000,
            'quantity': 0.002,
            'symbol': 'BTC/USDT',
        }

        initial_invested = strategy.invested_amount
        strategy.execute_purchase(purchase)

        # Check state updated
        assert strategy.invested_amount == initial_invested + 100
        assert len(strategy.purchases) == 1

    def test_portfolio_status(self):
        """Test portfolio status calculation."""
        strategy = SmartDCAStrategy(
            symbol='BTC/USDT',
            total_investment=10000,
            purchase_amount=100,
        )

        # Execute some purchases
        strategy.execute_purchase({
            'action': 'buy',
            'amount': 100,
            'price': 50000,
            'quantity': 0.002,
            'symbol': 'BTC/USDT',
        })

        strategy.execute_purchase({
            'action': 'buy',
            'amount': 100,
            'price': 48000,
            'quantity': 0.00208,
            'symbol': 'BTC/USDT',
        })

        current_price = 52000
        status = strategy.get_portfolio_status(current_price)

        # Check status
        assert status['total_invested'] == 200
        assert status['num_purchases'] == 2
        assert status['total_quantity'] > 0
        assert 'pnl' in status
        assert 'pnl_pct' in status


class TestAutomatedMarketMakingStrategy:
    """Tests for AutomatedMarketMakingStrategy."""

    def test_initialization(self):
        """Test AMM strategy initialization."""
        strategy = AutomatedMarketMakingStrategy(
            token_a='ETH',
            token_b='USDT',
            pool_price=3000,
        )

        assert strategy.token_a == 'ETH'
        assert strategy.token_b == 'USDT'
        assert strategy.pool_price == 3000

    def test_optimal_positions(self):
        """Test optimal position calculation."""
        strategy = AutomatedMarketMakingStrategy(
            token_a='ETH',
            token_b='USDT',
            pool_price=3000,
            price_range=(2400, 3600),
        )

        recommendations = strategy.calculate_optimal_positions(
            current_inventory_a=10,  # 10 ETH
            current_inventory_b=30000,  # 30000 USDT
            current_price=3000,
        )

        # Check recommendations
        assert 'current_ratio_a' in recommendations
        assert 'needs_rebalance' in recommendations
        assert isinstance(recommendations['needs_rebalance'], bool)

    def test_rebalance_needed(self):
        """Test rebalancing trigger."""
        strategy = AutomatedMarketMakingStrategy(
            token_a='ETH',
            token_b='USDT',
            pool_price=3000,
            rebalance_threshold=0.05,  # 5%
        )

        # Very imbalanced inventory
        recommendations = strategy.calculate_optimal_positions(
            current_inventory_a=20,  # 20 ETH
            current_inventory_b=6000,  # 6000 USDT (way less)
            current_price=3000,
        )

        # Should recommend rebalance
        assert recommendations['needs_rebalance'] == True
        assert len(recommendations['actions']) > 0

    def test_trade_simulation(self):
        """Test trade simulation."""
        strategy = AutomatedMarketMakingStrategy(
            token_a='ETH',
            token_b='USDT',
            pool_price=3000,
            base_fee_rate=0.003,
        )

        # Simulate incoming buy trade
        trade = {
            'side': 'buy',
            'amount': 1.0,  # 1 ETH
            'price': 3000,
        }

        result = strategy.simulate_trade(trade)

        assert result['side'] == 'buy'
        assert result['amount_in'] == 1.0
        assert result['fee_earned'] > 0
        assert 'price_impact' in result

    def test_pool_metrics(self):
        """Test pool metrics calculation."""
        strategy = AutomatedMarketMakingStrategy(
            token_a='ETH',
            token_b='USDT',
            pool_price=3000,
        )

        # Simulate some trades
        strategy.simulate_trade({'side': 'buy', 'amount': 1.0, 'price': 3000})
        strategy.simulate_trade({'side': 'sell', 'amount': 0.5, 'price': 3000})

        metrics = strategy.get_pool_metrics()

        assert metrics['total_trades'] == 2
        assert metrics['total_fees'] > 0
        assert 'adverse_selection_count' in metrics


class TestTriangularArbitrageStrategy:
    """Tests for TriangularArbitrageStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = TriangularArbitrageStrategy(
            exchanges=['binance', 'okx'],
            trading_pairs=['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
        )

        assert len(strategy.exchanges) == 2
        assert len(strategy.trading_pairs) == 3

    def test_price_update(self):
        """Test price data update."""
        strategy = TriangularArbitrageStrategy(
            exchanges=['binance'],
            trading_pairs=['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
        )

        price_data = {
            'binance': {
                'BTC/USDT': 50000,
                'ETH/BTC': 0.06,
                'ETH/USDT': 3000,
            }
        }

        strategy.update_prices(price_data)

        assert 'binance' in strategy.price_graph
        assert len(strategy.price_graph['binance']) == 3

    def test_find_opportunities(self):
        """Test arbitrage opportunity detection."""
        strategy = TriangularArbitrageStrategy(
            exchanges=['binance'],
            trading_pairs=['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
            min_profit_threshold=0.001,
        )

        price_data = {
            'binance': {
                'BTC/USDT': 50000,
                'ETH/BTC': 0.06,
                'ETH/USDT': 3000,
            }
        }

        strategy.update_prices(price_data)
        opportunities = strategy.find_arbitrage_opportunities()

        # Should return list (may be empty if no profitable opportunities)
        assert isinstance(opportunities, list)

    def test_execute_arbitrage(self):
        """Test arbitrage execution."""
        strategy = TriangularArbitrageStrategy(
            exchanges=['binance'],
            trading_pairs=['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
        )

        # Create a profitable opportunity
        opportunity = {
            'exchange': 'binance',
            'path': ['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
            'profit_pct': 0.01,  # 1% profit
            'estimated_profit': 10.0,
        }

        # Update prices
        strategy.update_prices({
            'binance': {
                'BTC/USDT': 50000,
                'ETH/BTC': 0.06,
                'ETH/USDT': 3000,
            }
        })

        result = strategy.execute_arbitrage(opportunity, trade_size=1000)

        assert result['success'] == True
        assert 'initial_amount' in result
        assert 'final_amount' in result
        assert 'profit' in result


class TestCreateModernStrategy:
    """Tests for factory function."""

    def test_create_grid_trading(self):
        """Test creating grid trading strategy."""
        strategy = create_modern_strategy(
            strategy_type="grid_trading",
            symbol='BTC/USDT',
            upper_price=60000,
            lower_price=40000,
        )

        assert isinstance(strategy, GridTradingStrategy)

    def test_create_smart_dca(self):
        """Test creating smart DCA strategy."""
        strategy = create_modern_strategy(
            strategy_type="smart_dca",
            symbol='BTC/USDT',
            total_investment=10000,
        )

        assert isinstance(strategy, SmartDCAStrategy)

    def test_create_amm(self):
        """Test creating AMM strategy."""
        strategy = create_modern_strategy(
            strategy_type="amm",
            token_a='ETH',
            token_b='USDT',
            pool_price=3000,
        )

        assert isinstance(strategy, AutomatedMarketMakingStrategy)

    def test_create_triangular_arbitrage(self):
        """Test creating triangular arbitrage strategy."""
        strategy = create_modern_strategy(
            strategy_type="triangular_arbitrage",
            exchanges=['binance'],
            trading_pairs=['BTC/USDT', 'ETH/BTC', 'ETH/USDT'],
        )

        assert isinstance(strategy, TriangularArbitrageStrategy)

    def test_invalid_strategy_type(self):
        """Test error handling for invalid type."""
        with pytest.raises(ValueError):
            create_modern_strategy(strategy_type="invalid_type")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
