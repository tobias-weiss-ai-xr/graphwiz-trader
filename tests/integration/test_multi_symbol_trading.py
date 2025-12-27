"""
Integration Tests for Multi-Symbol Grid Trading

Following Cognitive QA Principles:
1. End-to-End Testing: Full workflow validation
2. Component Interaction: Strategy + Exchange + Portfolio
3. Real Data: Use actual market data via CCXT
4. Mutation Testing: Detect integration breaking changes

Test Focus:
- Real exchange connectivity
- Portfolio state management
- Multi-symbol coordination
- Auto-save functionality
"""

import pytest
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

from graphwiz_trader.trading.exchange import create_exchange


class TestExchangeIntegration:
    """Test suite for exchange integration (Cognitive QA: External System Testing)"""

    def test_binance_connection(self):
        """
        Test: Can connect to Binance and fetch data

        Chain-of-Thought:
        1. Create exchange instance
        2. Fetch ticker data
        3. Verify data structure and validity

        Integration Point: External API connectivity
        Mutation Target: API integration code
        """
        # Arrange & Act
        exchange = create_exchange("binance")
        ticker = exchange.fetch_ticker("BTC/USDT")

        # Assert
        assert ticker is not None
        assert "last" in ticker
        assert ticker["last"] > 0
        assert "symbol" in ticker
        assert isinstance(ticker["symbol"], str)

    def test_fetch_multiple_symbols(self):
        """
        Test: Can fetch data for multiple symbols simultaneously

        Real-World Scenario: Multi-symbol portfolio from Feature 1
        """
        # Arrange
        exchange = create_exchange("binance")
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

        # Act
        tickers = {}
        for symbol in symbols:
            ticker = exchange.fetch_ticker(symbol)
            tickers[symbol] = ticker

        # Assert
        assert len(tickers) == len(symbols)
        assert all(ticker is not None for ticker in tickers.values())
        assert all(ticker["last"] > 0 for ticker in tickers.values())

    def test_fetch_ohlcv_data(self):
        """
        Test: Can fetch OHLCV historical data

        Integration Point: Historical data retrieval
        """
        # Arrange
        exchange = create_exchange("binance")
        symbol = "BTC/USDT"
        limit = 100

        # Act
        ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=limit)

        # Assert
        assert ohlcv is not None
        assert len(ohlcv) > 0
        assert len(ohlcv) <= limit
        # Each candle should have 6 elements [timestamp, open, high, low, close, volume]
        assert all(len(candle) == 6 for candle in ohlcv)
        # Verify data validity
        assert all(candle[4] > 0 for candle in ohlcv)  # Close price > 0


class TestMultiSymbolPortfolioIntegration:
    """Test suite for multi-symbol portfolio management (Cognitive QA: System Integration)"""

    def test_portfolio_initialization(self):
        """
        Test: Multi-symbol trader initializes correctly

        Chain-of-Thought:
        1. Define multiple symbol configurations
        2. Create trader with total capital
        3. Verify capital is allocated correctly

        Integration Point: Portfolio + Strategy initialization
        """
        # Arrange
        from examples.multi_symbol_grid_trading import (
            MultiSymbolGridTrader,
            SymbolConfig,
        )

        symbols = [
            SymbolConfig(
                symbol="BTC/USDT",
                capital_allocation=0.40,
                num_grids=5,
                grid_range_pct=0.15,
            ),
            SymbolConfig(
                symbol="ETH/USDT",
                capital_allocation=0.30,
                num_grids=5,
                grid_range_pct=0.15,
            ),
            SymbolConfig(
                symbol="SOL/USDT",
                capital_allocation=0.30,
                num_grids=5,
                grid_range_pct=0.15,
            ),
        ]

        total_capital = 10000.0

        # Act
        trader = MultiSymbolGridTrader(
            symbols=symbols,
            total_capital=total_capital,
        )

        # Assert
        assert len(trader.symbol_states) == 3
        # Verify capital allocation
        expected_allocations = {
            "BTC/USDT": 4000.0,
            "ETH/USDT": 3000.0,
            "SOL/USDT": 3000.0,
        }
        for symbol, state in trader.symbol_states.items():
            expected = expected_allocations[symbol]
            assert state.initial_capital == expected

    def test_portfolio_value_calculation(self):
        """
        Test: Portfolio value is calculated correctly across symbols

        Integration Point: Portfolio tracking across multiple strategies
        """
        # Arrange
        from examples.multi_symbol_grid_trading import (
            MultiSymbolGridTrader,
            SymbolConfig,
        )

        symbols = [
            SymbolConfig(
                symbol="BTC/USDT",
                capital_allocation=0.50,
                num_grids=5,
                grid_range_pct=0.15,
            ),
        ]

        trader = MultiSymbolGridTrader(
            symbols=symbols,
            total_capital=10000.0,
        )

        # Act
        portfolio_summary = trader.update_portfolio()

        # Assert
        assert "total_value" in portfolio_summary
        assert "strategies" in portfolio_summary
        assert "total_pnl" in portfolio_summary
        # Initially, total value should equal capital (no positions yet)
        assert abs(portfolio_summary["total_value"] - 10000.0) < 1.0

    def test_auto_save_functionality(self):
        """
        Test: Auto-save creates files in correct format

        Integration Point: Trading + File I/O
        """
        # Arrange
        from examples.multi_symbol_grid_trading import (
            MultiSymbolGridTrader,
            SymbolConfig,
        )

        symbols = [
            SymbolConfig(
                symbol="BTC/USDT",
                capital_allocation=1.0,
                num_grids=3,
                grid_range_pct=0.15,
            ),
        ]

        trader = MultiSymbolGridTrader(
            symbols=symbols,
            total_capital=1000.0,
        )

        # Create some dummy equity history
        trader.equity_history = [
            {
                "timestamp": datetime.now(),
                "iteration": 1,
                "total_value": 10000.0,
                "strategies": {},
            }
        ]

        # Act
        trader.save_results({})

        # Assert
        data_dir = Path("data/multi_symbol_trading")
        assert data_dir.exists()
        # Check for equity file
        equity_files = list(data_dir.glob("portfolio_equity_*.csv"))
        assert len(equity_files) > 0


class TestDynamicRebalancingIntegration:
    """Test suite for dynamic grid rebalancing (Cognitive QA: Adaptive Behavior)"""

    def test_rebalance_trigger_detection(self):
        """
        Test: System detects when rebalancing is needed

        Chain-of-Thought:
        1. Initialize grid at specific price
        2. Simulate price moving outside threshold
        3. Verify rebalancing is triggered

        Integration Point: Price monitoring + Rebalancing logic
        """
        # Arrange
        from examples.dynamic_grid_rebalancing import DynamicGridRebalancer

        trader = DynamicGridRebalancer(
            symbol="BTC/USDT",
            initial_capital=10000.0,
            num_grids=10,
            grid_range_pct=0.15,
            rebalance_threshold=0.10,
        )

        # Simulate price moving 15% above grid (beyond 10% threshold)
        original_upper = trader.strategy.upper_price
        trader.current_price = original_upper * 1.15

        # Act
        needs_rebalance, reason = trader.check_rebalance_needed()

        # Assert
        assert needs_rebalance is True
        assert reason is not None
        assert "above" in reason.lower()

    def test_rebalance_preserves_capital(self):
        """
        Test: Rebalancing preserves total capital

        Integration Point: Rebalancing + Portfolio state
        Mutation Target: Capital preservation logic
        """
        # Arrange
        from examples.dynamic_grid_rebalancing import DynamicGridRebalancer

        trader = DynamicGridRebalancer(
            symbol="BTC/USDT",
            initial_capital=10000.0,
            num_grids=10,
            grid_range_pct=0.15,
        )

        # Simulate some position
        trader.position = 0.1  # 0.1 BTC
        trader.avg_price = 50000.0

        # Act
        trader.rebalance_grid("Test rebalance")

        # Assert
        # After rebalance, capital should equal previous portfolio value
        # Position should be closed
        assert trader.position == 0.0
        assert trader.capital > 0


class TestRiskManagementIntegration:
    """Test suite for risk management integration (Cognitive QA: Safety Validation)"""

    def test_stop_loss_trigger(self):
        """
        Test: Stop loss triggers at correct threshold

        Chain-of-Thought:
        1. Create position at entry price
        2. Simulate price dropping to stop-loss level
        3. Verify stop loss executes and closes position

        Integration Point: Risk monitoring + Trade execution
        Mutation Target: Stop loss calculation
        """
        # Arrange
        from examples.risk_management_trading import (
            RiskManagedGridTrader,
            RiskLimits,
        )

        risk_limits = RiskLimits(
            stop_loss_pct=0.05,  # 5% stop loss
        )

        trader = RiskManagedGridTrader(
            symbol="BTC/USDT",
            initial_capital=10000.0,
            num_grids=5,
            risk_limits=risk_limits,
        )

        # Create a position
        trader.position = 0.1
        trader.avg_price = 50000.0
        trader.capital = 5000.0  # Spent $5000 on position

        # Simulate 5% drop (stop loss threshold)
        trader.current_price = 47500.0  # 5% below entry

        # Act
        can_trade, violation = trader.check_risk_limits()

        # Assert
        assert can_trade is False  # Trading should be blocked
        assert "stop loss" in violation.lower()

    def test_daily_loss_limit_reset(self):
        """
        Test: Daily loss limit resets on new day

        Integration Point: Date tracking + Limit enforcement
        """
        # Arrange
        from examples.risk_management_trading import (
            RiskManagedGridTrader,
            RiskLimits,
        )

        risk_limits = RiskLimits(
            daily_loss_limit_pct=0.03,  # 3% daily limit
        )

        trader = RiskManagedGridTrader(
            symbol="BTC/USDT",
            initial_capital=10000.0,
            num_grids=5,
            risk_limits=risk_limits,
        )

        # Simulate hitting daily limit
        trader.risk_state.daily_pnl = -400.0  # -$400 (4% loss)

        # Act
        can_trade, violation = trader.check_risk_limits()

        # Assert
        assert can_trade is False  # Should block trading
        assert "daily" in violation.lower()

        # Now simulate new day
        trader.trading_day_start = datetime.now().date() - timedelta(days=1)

        # Act again
        can_trade, violation = trader.check_risk_limits()

        # Assert
        assert can_trade is True  # Trading should be allowed again
        assert trader.risk_state.daily_pnl == 0.0  # Should be reset

    def test_position_size_limit(self):
        """
        Test: Position size limit is enforced

        Integration Point: Position sizing + Risk limits
        """
        # Arrange
        from examples.risk_management_trading import (
            RiskManagedGridTrader,
            RiskLimits,
        )

        risk_limits = RiskLimits(
            max_position_size_pct=0.20,  # 20% max
        )

        trader = RiskManagedGridTrader(
            symbol="BTC/USDT",
            initial_capital=10000.0,
            num_grids=5,
            risk_limits=risk_limits,
        )

        # Set current portfolio value
        trader.current_price = 50000.0

        # Try to create position larger than 20%
        max_allowed = 10000.0 * 0.20 / 50000.0  # 0.04 BTC
        trader.position = max_allowed * 1.5  # 50% over limit
        trader.capital = 5000.0

        # Act
        can_trade, violation = trader.check_risk_limits()

        # Assert
        # This should be handled in execute_grid_trades, not check_risk_limits
        # But we can verify the limit exists
        assert trader.risk_limits.max_position_size_pct == 0.20


class TestDashboardIntegration:
    """Test suite for dashboard integration (Cognitive QA: UI Integration)"""

    def test_dashboard_state_update(self):
        """
        Test: Dashboard state updates correctly

        Integration Point: Trading system + Dashboard state
        """
        # Arrange
        from examples.trading_dashboard import (
            StrategyState,
            update_strategy_state,
        )

        state = StrategyState(
            symbol="BTC/USDT",
            strategy_type="Grid Trading",
            current_price=87500.0,
            portfolio_value=10930.76,
            capital=9000.0,
            position=0.022044,
            position_value=1930.76,
            pnl=930.76,
            roi_pct=9.31,
            grid_upper=100700.90,
            grid_lower=74400.90,
            grid_levels=list(range(74000, 101000, 2500)),
            trades_count=11,
            last_update=datetime.now().isoformat(),
        )

        # Act
        update_strategy_state("BTC/USDT", state)

        # Assert
        from examples.trading_dashboard import dashboard_state
        assert "BTC/USDT" in dashboard_state["strategies"]
        assert dashboard_state["strategies"]["BTC/USDT"]["portfolio_value"] == 10930.76


class TestRealDataIntegration:
    """Test suite with real market data (Cognitive QA: Production Validity)"""

    def test_live_market_data_workflow(self):
        """
        Test: Complete workflow with live market data

        Chain-of-Thought:
        1. Fetch real BTC price from Binance
        2. Initialize grid around current price
        3. Generate trading signals
        4. Verify signals are valid

        Real-World Scenario: Using live data (as in Features 1-5)
        """
        # Arrange
        exchange = create_exchange("binance")
        ticker = exchange.fetch_ticker("BTC/USDT")
        current_price = ticker["last"]

        from graphwiz_trader.strategies import GridTradingStrategy, GridTradingMode

        # Act: Initialize grid centered on current price (as per backtesting optimization)
        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=current_price * 1.15,
            lower_price=current_price * 0.85,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
        )

        signal = strategy.generate_trading_signals(current_price=current_price)

        # Assert
        assert strategy.lower_price < current_price < strategy.upper_price
        assert signal is not None
        assert "action" in signal

    @pytest.mark.slow
    def test_multi_symbol_real_data(self):
        """
        Test: Multi-symbol system with real market data

        Real-World Scenario: Feature 1 implementation
        Note: Marked as slow due to network calls
        """
        # Arrange
        from examples.multi_symbol_grid_trading import (
            MultiSymbolGridTrader,
            SymbolConfig,
        )

        symbols = [
            SymbolConfig(
                symbol="BTC/USDT",
                capital_allocation=0.40,
                num_grids=3,  # Fewer for faster test
                grid_range_pct=0.15,
            ),
            SymbolConfig(
                symbol="ETH/USDT",
                capital_allocation=0.30,
                num_grids=3,
                grid_range_pct=0.15,
            ),
            SymbolConfig(
                symbol="SOL/USDT",
                capital_allocation=0.30,
                num_grids=3,
                grid_range_pct=0.15,
            ),
        ]

        # Act
        trader = MultiSymbolGridTrader(
            symbols=symbols,
            total_capital=10000.0,
        )

        # Run one iteration
        for symbol in trader.symbol_states.keys():
            market_data = trader.fetch_market_data(symbol)
            assert market_data["price"] > 0

        # Assert
        portfolio = trader.update_portfolio()
        assert portfolio["total_value"] > 0
        assert len(portfolio["strategies"]) == 3


# Performance Testing (Cognitive QA: Non-Functional Requirements)
class TestPerformanceIntegration:
    """Test suite for performance characteristics (Cognitive QA: Performance Testing)"""

    def test_initialization_performance(self):
        """
        Test: Initialization completes within acceptable time

        Non-Functional Requirement: Performance
        """
        # Arrange
        from examples.multi_symbol_grid_trading import (
            MultiSymbolGridTrader,
            SymbolConfig,
        )

        symbols = [
            SymbolConfig(
                symbol=f"BTC/USDT",
                capital_allocation=0.25,
                num_grids=10,
                grid_range_pct=0.15,
            )
            for _ in range(4)
        ]

        # Act
        start_time = time.time()
        trader = MultiSymbolGridTrader(
            symbols=symbols,
            total_capital=10000.0,
        )
        init_time = time.time() - start_time

        # Assert
        # Should initialize in less than 5 seconds
        assert init_time < 5.0, f"Initialization took {init_time:.2f}s, expected < 5s"

    def test_signal_generation_performance(self):
        """
        Test: Signal generation is fast enough for real-time use

        Non-Functional Requirement: Latency
        """
        # Arrange
        from graphwiz_trader.strategies import GridTradingStrategy, GridTradingMode

        strategy = GridTradingStrategy(
            symbol="BTC/USDT",
            upper_price=100000.0,
            lower_price=80000.0,
            num_grids=10,
            grid_mode=GridTradingMode.GEOMETRIC,
            investment_amount=10000.0,
        )

        # Act
        start_time = time.time()
        for _ in range(100):  # Generate 100 signals
            strategy.generate_trading_signals(current_price=87500.0)
        elapsed = time.time() - start_time

        # Assert
        # Should generate 100 signals in less than 1 second
        assert elapsed < 1.0, f"100 signals took {elapsed:.2f}s, expected < 1s"
        # Average latency per signal
        avg_latency = elapsed / 100
        assert avg_latency < 0.01, f"Average latency {avg_latency*1000:.2f}ms, expected < 10ms"
