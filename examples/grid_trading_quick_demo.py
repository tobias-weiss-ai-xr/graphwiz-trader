"""
Quick Grid Trading Paper Trading Demo

Run a quick demo of the grid trading strategy with 3 iterations.
"""

import time
import sys
from loguru import logger

from graphwiz_trader.strategies import GridTradingStrategy, GridTradingMode
from graphwiz_trader.trading.exchange import create_exchange

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

def main():
    print("\n" + "="*80)
    print(" "*25 + "GRID TRADING PAPER TRADING DEMO")
    print("="*80)
    print("\nOptimal Configuration:")
    print("  • 10 grids (Geometric mode)")
    print("  • ±15% range from current price")
    print("  • $10,000 initial capital")
    print("  • Real-time market data from Binance")
    print("\n" + "="*80)

    # Create exchange
    exchange = create_exchange("binance")

    # Fetch current price
    print("\nFetching current BTC/USDT price...")
    ticker = exchange.fetch_ticker("BTC/USDT")
    current_price = ticker["last"]

    print(f"✅ Current BTC/USDT price: ${current_price:,.2f}")

    # Calculate grid range (±15%)
    upper_price = current_price * 1.15
    lower_price = current_price * 0.85

    print(f"Grid range: ${lower_price:,.2f} - ${upper_price:,.2f}")

    # Create strategy
    strategy = GridTradingStrategy(
        symbol="BTC/USDT",
        upper_price=upper_price,
        lower_price=lower_price,
        num_grids=10,
        grid_mode=GridTradingMode.GEOMETRIC,
        investment_amount=10000,
    )

    print(f"\n✅ Grid Trading Strategy initialized")
    print(f"   Grid levels: {len(strategy.grid_levels)}")
    print(f"   Investment: ${strategy.investment_amount:,.2f}")

    # Initialize virtual portfolio
    portfolio = {
        "capital": 10000,
        "position": 0.0,
        "avg_price": 0.0,
    }

    # Run 3 iterations
    print("\n" + "="*80)
    print("Running 3 iterations (fetching real market data each time)...")
    print("="*80 + "\n")

    for iteration in range(1, 4):
        print(f"\n--- Iteration {iteration} ---")

        # Fetch market data
        print("Fetching market data...")
        data = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=100)
        latest_price = data[-1][4]  # Close price
        print(f"Current price: ${latest_price:,.2f}")

        # Check if price is in grid range
        if latest_price < lower_price:
            print(f"⚠️  Price ${latest_price:,.2f} BELOW grid range")
        elif latest_price > upper_price:
            print(f"⚠️  Price ${latest_price:,.2f} ABOVE grid range")
        else:
            print(f"✅ Price ${latest_price:,.2f} within grid range")

            # Find nearby grid levels
            nearby_levels = []
            for level in strategy.grid_levels:
                if abs(level - latest_price) / latest_price < 0.01:  # Within 1%
                    nearby_levels.append(level)

            if nearby_levels:
                print(f"Found {len(nearby_levels)} grid levels within 1% of current price")
                for level in nearby_levels[:3]:
                    if level < latest_price:
                        action = "BUY"
                        profit = (latest_price - level) / level * 100
                        print(f"  {action} at ${level:,.2f} (potential profit: {profit:+.2f}%)")
                    else:
                        action = "SELL"
                        profit = (level - latest_price) / latest_price * 100
                        print(f"  {action} at ${level:,.2f} (potential profit: {profit:+.2f}%)")
            else:
                print("No grid levels within 1% of current price (waiting for price movement)")

        # Calculate portfolio value
        position_value = portfolio["position"] * latest_price
        total_value = portfolio["capital"] + position_value
        print(f"Portfolio: ${total_value:,.2f} (Capital: ${portfolio['capital']:,.2f}, Position: ${position_value:,.2f})")

        # Wait before next iteration
        if iteration < 3:
            print("\nWaiting 5 seconds before next iteration...")
            time.sleep(5)

    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print(f"\nGrid Trading Strategy is ready!")
    print(f"Current Price: ${latest_price:,.2f}")
    print(f"Grid Range: ${lower_price:,.2f} - ${upper_price:,.2f}")
    print(f"Portfolio Value: ${total_value:,.2f}")
    print(f"\nNext Steps:")
    print(f"  1. Run longer paper trading: python examples/grid_trading_paper_trading_deploy.py")
    print(f"  2. Monitor for 1-2 weeks")
    print(f"  3. Deploy to testnet")
    print(f"  4. Graduate to production")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
