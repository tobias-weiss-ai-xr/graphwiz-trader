"""
Position calculator for live trading.

Helps calculate optimal position sizes based on:
- Account balance
- Risk tolerance
- Stop loss levels
- Trading fees
- Portfolio exposure

Ensures positions are properly sized for risk management.
"""

from decimal import Decimal, getcontext
from typing import Dict, Any, Tuple
import math

# Set precision for decimal calculations
getcontext().prec = 10


class PositionCalculator:
    """Calculate optimal position sizes for trading."""

    def __init__(
        self,
        account_balance_eur: float,
        risk_per_trade_pct: float = 0.01,
        max_position_pct: float = 0.05,
        stop_loss_pct: float = 0.02
    ):
        """Initialize position calculator.

        Args:
            account_balance_eur: Total account balance in EUR
            risk_per_trade_pct: Risk per trade (default: 1%)
            max_position_pct: Maximum position size (default: 5%)
            stop_loss_pct: Stop loss percentage (default: 2%)
        """
        self.account_balance = Decimal(str(account_balance_eur))
        self.risk_per_trade = Decimal(str(risk_per_trade_pct))
        self.max_position_pct = Decimal(str(max_position_pct))
        self.stop_loss_pct = Decimal(str(stop_loss_pct))

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        fees: float = 0.0026
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Calculate optimal position size based on risk.

        Args:
            entry_price: Entry price in EUR
            stop_loss_price: Stop loss price in EUR
            fees: Trading fees (default: 0.26% Kraken taker)

        Returns:
            Tuple of (position_size, cost_eur, details)
        """
        entry = Decimal(str(entry_price))
        stop_loss = Decimal(str(stop_loss_price))
        fee_rate = Decimal(str(fees))

        # Calculate risk amount (1% of account by default)
        risk_amount = self.account_balance * self.risk_per_trade

        # Calculate price difference
        price_diff = abs(entry - stop_loss)

        if price_diff == 0:
            return 0.0, 0.0, {"error": "Entry and stop loss cannot be the same"}

        # Position size based on risk
        # Position Size = Risk Amount / Price Difference
        position_size = float(risk_amount / price_diff)

        # Calculate cost including fees
        cost_before_fees = position_size * entry_price
        fee_amount = cost_before_fees * fees
        total_cost = cost_before_fees + fee_amount

        # Apply maximum position limit
        max_cost = float(self.account_balance * self.max_position_pct)

        if total_cost > max_cost:
            # Reduce position to max allowed
            scale_factor = max_cost / total_cost
            position_size *= scale_factor
            total_cost = max_cost
            cost_before_fees = total_cost / (1 + fees)

        details = {
            "position_size": position_size,
            "entry_price": float(entry),
            "stop_loss": float(stop_loss),
            "cost_before_fees": float(cost_before_fees),
            "fees": float(fee_amount),
            "total_cost": float(total_cost),
            "risk_amount": float(risk_amount),
            "risk_per_trade": f"{self.risk_per_trade * 100:.1f}%",
            "max_position": f"{self.max_position_pct * 100:.1f}%",
            "price_diff": float(price_diff),
            "stop_distance_pct": float((price_diff / entry) * 100)
        }

        return position_size, total_cost, details

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit price based on risk/reward ratio.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            reward_ratio: Risk/reward ratio (default: 2:1)

        Returns:
            Take profit price
        """
        entry = Decimal(str(entry_price))
        stop_loss = Decimal(str(stop_loss_price))

        # Calculate risk distance
        risk_distance = abs(entry - stop_loss)

        # Calculate reward distance
        reward_distance = risk_distance * Decimal(str(reward_ratio))

        # Take profit is reward distance away from entry
        if entry > stop_loss:
            # Long position
            take_profit = float(entry + reward_distance)
        else:
            # Short position
            take_profit = float(entry - reward_distance)

        return take_profit

    def calculate_position_value(
        self,
        symbol: str,
        quantity: float,
        current_price: float
    ) -> Dict[str, float]:
        """Calculate current position value.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            current_price: Current market price

        Returns:
            Dictionary with position details
        """
        value = quantity * current_price

        return {
            "symbol": symbol,
            "quantity": quantity,
            "current_price": current_price,
            "value_eur": value,
            "pct_of_account": (value / float(self.account_balance)) * 100
        }

    def validate_position(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate if position meets risk criteria.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price

        Returns:
            Tuple of (is_valid, reason, details)
        """
        position_value = quantity * price
        position_pct = position_value / float(self.account_balance)

        details = {
            "position_value": position_value,
            "position_pct": position_pct * 100,
            "max_allowed": float(self.account_balance * self.max_position_pct),
            "max_pct": float(self.max_position_pct * 100)
        }

        # Check if position exceeds maximum
        if position_value > details["max_allowed"]:
            return (
                False,
                f"Position exceeds maximum: {position_pct*100:.1f}% > {self.max_position_pct*100:.1f}%",
                details
            )

        # Check if position is too small (below minimum trade)
        min_trade_eur = 10.0  # Kraken minimum
        if position_value < min_trade_eur:
            return (
                False,
                f"Position below minimum trade: €{position_value:.2f} < €{min_trade_eur:.2f}",
                details
            )

        return True, "Position is valid", details

    def calculate_compounding_position(
        self,
        current_balance: float,
        target_risk_pct: float = None
    ) -> Dict[str, Any]:
        """Calculate position size using compounding.

        As account grows, position sizes grow proportionally.

        Args:
            current_balance: Current account balance
            target_risk_pct: Target risk per trade (uses default if None)

        Returns:
            Dictionary with recommended position sizes
        """
        balance = Decimal(str(current_balance))
        risk_pct = Decimal(str(target_risk_pct)) if target_risk_pct else self.risk_per_trade

        # Calculate risk amount
        risk_amount = balance * risk_pct

        # Calculate maximum position
        max_position = balance * self.max_position_pct

        return {
            "account_balance": float(balance),
            "risk_amount": float(risk_amount),
            "max_position": float(max_position),
            "risk_per_trade_pct": f"{risk_pct * 100:.1f}%",
            "max_position_pct": f"{self.max_position_pct * 100:.1f}%",
            "recommended_min": float(risk_amount * 2),  # 2x risk for 2:1 reward
            "recommended_max": float(max_position)
        }


def calculate_quick_position(
    balance_eur: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 0.01
) -> Dict[str, Any]:
    """Quick position calculation helper.

    Args:
        balance_eur: Account balance in EUR
        entry_price: Entry price
        stop_loss_price: Stop loss price
        risk_pct: Risk per trade (default: 1%)

    Returns:
        Dictionary with position calculation results
    """
    calc = PositionCalculator(
        account_balance_eur=balance_eur,
        risk_per_trade_pct=risk_pct
    )

    position_size, cost, details = calc.calculate_position_size(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price
    )

    # Calculate take profit
    take_profit = calc.calculate_take_profit(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        reward_ratio=2.0
    )

    # Validate position
    is_valid, reason, validation = calc.validate_position(
        symbol="BTC/EUR",
        quantity=position_size,
        price=entry_price
    )

    return {
        "valid": is_valid,
        "validation_reason": reason,
        "position_size": position_size,
        "total_cost": cost,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit,
        "details": details,
        "validation": validation
    }


# Example usage and demonstrations
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("📊 POSITION CALCULATOR DEMO")
    print("=" * 80)
    print()

    # Example 1: Basic position calculation
    print("Example 1: Basic Position Calculation")
    print("-" * 40)
    print("Account Balance: €10,000")
    print("Entry Price: €90,000")
    print("Stop Loss: €88,200 (2%)")
    print("Risk per Trade: 1%")
    print()

    result = calculate_quick_position(
        balance_eur=10000,
        entry_price=90000,
        stop_loss_price=88200,
        risk_pct=0.01
    )

    if result["valid"]:
        print(f"✅ Position Size: {result['position_size']:.6f} BTC")
        print(f"✅ Total Cost: €{result['total_cost']:.2f}")
        print(f"✅ Stop Loss: €{result['stop_loss']:,.2f}")
        print(f"✅ Take Profit: €{result['take_profit']:,.2f}")
        print()
        print("Details:")
        for key, value in result["details"].items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ Invalid: {result['validation_reason']}")

    print()

    # Example 2: Live trading recommendation
    print("Example 2: Live Trading Recommendation (€500 Account)")
    print("-" * 40)

    calc = PositionCalculator(account_balance_eur=500)

    recs = calc.calculate_compounding_position(500)

    print("Account: €500.00")
    print(f"Risk Amount (1%): €{recs['risk_amount']:.2f}")
    print(f"Max Position (5%): €{recs['max_position']:.2f}")
    print(f"Recommended Range: €{recs['recommended_min']:.2f} - €{recs['recommended_max']:.2f}")

    print()
    print("=" * 80)
