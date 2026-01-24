#!/usr/bin/env python3
"""Verification script for trading infrastructure implementation."""

import sys
sys.path.insert(0, '/opt/git/graphwiz-trader/src')

from graphwiz_trader.trading import (
    TradingEngine,
    Order,
    OrderManager,
    OrderSide,
    OrderType,
    OrderStatus,
    PortfolioManager,
    Position
)

def test_order_manager():
    """Test OrderManager basic functionality."""
    print("Testing OrderManager...")

    order_mgr = OrderManager(
        min_order_amount=0.001,
        max_order_amount=1000000,
        max_price_deviation=0.5
    )

    # Create order
    order = order_mgr.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=0.1
    )

    assert order.symbol == "BTC/USDT"
    assert order.side == OrderSide.BUY
    assert order.status == OrderStatus.PENDING
    assert float(order.amount) == 0.1

    # Validate order
    assert order_mgr.validate_order(order, current_price=50000)

    # Modify order
    modified = order_mgr.modify_order(order.order_id, amount=0.2)
    assert float(modified.amount) == 0.2

    # Get order
    retrieved = order_mgr.get_order(order.order_id)
    assert retrieved is not None

    print("  OrderManager: OK")


def test_portfolio_manager():
    """Test PortfolioManager basic functionality."""
    print("Testing PortfolioManager...")

    portfolio = PortfolioManager(
        initial_balance={"USDT": 10000.0},
        risk_per_trade=0.02,
        max_position_size=0.3
    )

    # Create position
    position = Position("BTC/USDT", "BTC", "USDT")

    # Update with trade
    realized, unrealized = position.update_position(
        side="buy",
        amount=0.5,
        price=50000,
        fee=10
    )

    assert position.amount == 0.5
    assert position.is_long is True
    assert position.is_open is True

    # Calculate unrealized P&L
    pnl = position.calculate_unrealized_pnl(52000)
    assert pnl > 0  # Should be profit when price goes up

    # Test portfolio position size calculation
    size = portfolio.calculate_position_size(
        symbol="BTC/USDT",
        entry_price=50000,
        stop_loss_price=49000
    )
    assert size > 0

    # Test portfolio statistics
    stats = portfolio.get_portfolio_statistics()
    assert "total_trades" in stats
    assert "balances" in stats

    print("  PortfolioManager: OK")


def test_order_enums():
    """Test order enumerations."""
    print("Testing Order Enums...")

    # Test OrderSide
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"

    # Test OrderType
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"
    assert OrderType.STOP_LOSS.value == "stop_loss"

    # Test OrderStatus
    assert OrderStatus.PENDING.value == "pending"
    assert OrderStatus.OPEN.value == "open"
    assert OrderStatus.FILLED.value == "filled"
    assert OrderStatus.CANCELLED.value == "cancelled"

    print("  Order Enums: OK")


def test_order_serialization():
    """Test order serialization/deserialization."""
    print("Testing Order Serialization...")

    order = Order(
        symbol="ETH/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        amount=1.5,
        price=3000,
        exchange="binance"
    )

    # Convert to dict
    order_dict = order.to_dict()
    assert "order_id" in order_dict
    assert "symbol" in order_dict
    assert "side" in order_dict

    # Reconstruct from dict
    reconstructed = Order.from_dict(order_dict)
    assert reconstructed.symbol == order.symbol
    assert reconstructed.side == order.side
    assert float(reconstructed.amount) == float(order.amount)

    print("  Order Serialization: OK")


def test_position_class():
    """Test Position class."""
    print("Testing Position Class...")

    position = Position("BTC/USDT", "BTC", "USDT")

    # Buy 1 BTC @ 50000
    position.update_position("buy", 1.0, 50000)
    assert position.amount == 1.0
    assert position.avg_entry_price == 50000

    # Sell 0.5 BTC @ 52000
    position.update_position("sell", 0.5, 52000)
    assert position.amount == 0.5
    assert position.realized_pnl > 0  # Should have profit

    # Convert to dict
    pos_dict = position.to_dict()
    assert "symbol" in pos_dict
    assert "amount" in pos_dict
    assert "realized_pnl" in pos_dict

    print("  Position Class: OK")


def main():
    """Run all verification tests."""
    print("\n=== Trading Infrastructure Verification ===\n")

    try:
        test_order_enums()
        test_order_manager()
        test_order_serialization()
        test_position_class()
        test_portfolio_manager()

        print("\n=== All Verification Tests Passed ===\n")
        return 0

    except Exception as e:
        print(f"\n=== Verification Failed: {e} ===\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
