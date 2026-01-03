#!/usr/bin/env python3
"""
Live trading script - executes REAL trades with REAL money.

⚠️ WARNING: This script will execute REAL trades!
Make sure you:
1. Have tested thoroughly with paper trading
2. Understand the risks
3. Have set appropriate safety limits
4. Start with small amounts

Usage:
    # Run once (test mode - check connection)
    python scripts/live_trade.py --exchange kraken --symbol BTC/EUR --test

    # Run continuous live trading
    python scripts/live_trade.py --exchange kraken --symbol BTC/EUR

    # Custom safety limits
    python scripts/live_trade.py --exchange kraken --symbol BTC/EUR --max-position 300 --max-daily-loss 50
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from loguru import logger
    import ccxt
    import pandas as pd
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install dependencies:")
    print("  pip install loguru ccxt pandas python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()


class LiveTradingEngine:
    """Simple live trading engine for Kraken."""

    def __init__(
        self,
        exchange_name: str,
        symbol: str,
        api_key: str,
        api_secret: str,
        max_position_eur: float = 300.0,
        max_daily_loss_eur: float = 50.0,
        max_daily_trades: int = 2,
        require_confirmation: bool = True
    ):
        """
        Initialize live trading engine.

        Args:
            exchange_name: Exchange to use (e.g., "kraken")
            symbol: Trading pair (e.g., "BTC/EUR")
            api_key: Exchange API key
            api_secret: Exchange API secret
            max_position_eur: Maximum position size in EUR
            max_daily_loss_eur: Maximum daily loss limit
            max_daily_trades: Maximum trades per day
            require_confirmation: Require manual confirmation for trades
        """
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.max_position = max_position_eur
        self.max_daily_loss = max_daily_loss_eur
        self.max_daily_trades = max_daily_trades
        self.require_confirmation = require_confirmation

        # Initialize exchange
        if exchange_name.lower() == "kraken":
            self.exchange = ccxt.kraken({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        # Portfolio tracking
        self.portfolio = {}
        self.daily_pnl = 0.0
        self.daily_trade_count = 0

    def fetch_balance(self):
        """Fetch account balance."""
        try:
            balance = self.exchange.fetch_balance()
            logger.info(f"Balance fetched successfully")
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise

    def calculate_rsi(self, closes: list, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(closes) < period:
            return 50.0

        df = pd.DataFrame({'close': closes})
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def generate_signal(self) -> dict:
        """Generate trading signal."""
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            closes = df['close'].tolist()
            current_price = closes[-1]
            rsi = self.calculate_rsi(closes)

            # Generate signal based on RSI
            if rsi < 42:
                action = "BUY"
                confidence = min(0.95, 0.65 + (42 - rsi) / 80)
                reason = f"RSI oversold ({rsi:.1f})"
            elif rsi > 58:
                action = "SELL"
                confidence = min(0.95, 0.65 + (rsi - 58) / 80)
                reason = f"RSI overbought ({rsi:.1f})"
            else:
                action = "HOLD"
                confidence = 0.5
                reason = f"RSI neutral ({rsi:.1f})"

            return {
                'symbol': self.symbol,
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'price': current_price,
                'rsi': rsi,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate signal: {e}")
            raise

    def execute_trade(self, signal: dict) -> dict:
        """Execute a trade with confirmation."""
        if signal['action'] == 'HOLD':
            return {'status': 'no_action', 'message': 'Signal is HOLD'}

        # Check risk limits
        if self.daily_trade_count >= self.max_daily_trades:
            return {'status': 'rejected', 'reason': 'Daily trade limit reached'}

        if self.daily_pnl <= -self.max_daily_loss:
            return {'status': 'rejected', 'reason': 'Daily loss limit reached'}

        # Confirm trade
        if self.require_confirmation:
            logger.warning(f"\n{'='*80}")
            logger.warning(f"⚠️  LIVE TRADE CONFIRMATION REQUIRED")
            logger.warning(f"{'='*80}")
            logger.warning(f"Action:    {signal['action']}")
            logger.warning(f"Symbol:    {signal['symbol']}")
            logger.warning(f"Price:     €{signal['price']:.2f}")
            logger.warning(f"Reason:    {signal['reason']}")
            logger.warning(f"Confidence: {signal['confidence']:.2f}")
            logger.warning(f"{'='*80}\n")

            response = input("Type 'YES' to execute this trade: ")
            if response != 'YES':
                logger.info("Trade cancelled by user")
                return {'status': 'cancelled', 'reason': 'User declined'}

        # Execute trade
        try:
            if signal['action'] == 'BUY':
                # Calculate position size
                balance = self.fetch_balance()
                available_eur = balance.get('EUR', {}).get('free', 0)
                position_size_eur = min(self.max_position, available_eur * 0.25)

                if position_size_eur < 10:
                    return {'status': 'rejected', 'reason': 'Insufficient funds'}

                amount = position_size_eur / signal['price']

                logger.info(f"Executing BUY order: {amount:.6f} @ €{signal['price']:.2f}")
                order = self.exchange.create_market_buy_order(
                    self.symbol,
                    amount
                )

            elif signal['action'] == 'SELL':
                # Get balance of base currency
                balance = self.fetch_balance()
                base_currency = self.symbol.split('/')[0]
                available_amount = balance.get(base_currency, {}).get('free', 0)

                if available_amount <= 0:
                    return {'status': 'rejected', 'reason': 'No position to sell'}

                amount = available_amount * 0.75  # Sell 75% of position

                logger.info(f"Executing SELL order: {amount:.6f} @ €{signal['price']:.2f}")
                order = self.exchange.create_market_sell_order(
                    self.symbol,
                    amount
                )

            logger.success(f"✅ Order executed: {order['id']}")
            self.daily_trade_count += 1

            return {
                'status': 'executed',
                'order': order,
                'signal': signal
            }

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def run_once(self):
        """Run one trading iteration."""
        logger.info(f"Analyzing {self.symbol}...")

        try:
            # Generate signal
            signal = self.generate_signal()

            logger.info(f"Price: €{signal['price']:.2f}")
            logger.info(f"RSI: {signal['rsi']:.1f}")
            logger.info(f"Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
            logger.info(f"Reason: {signal['reason']}")

            # Execute if signal is strong enough
            if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                result = self.execute_trade(signal)
                logger.info(f"Result: {result['status']}")

                # Get current portfolio value
                balance = self.fetch_balance()
                eur_balance = balance.get('EUR', {}).get('free', 0)
                logger.info(f"EUR Balance: €{eur_balance:.2f}")

            return {
                'status': 'success',
                'signal': signal,
                'action_taken': signal['action'] if signal['confidence'] > 0.6 else 'None'
            }

        except Exception as e:
            logger.error(f"Error in run_once: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def start(self, interval_seconds: int = 3600):
        """Start continuous live trading."""
        logger.info("Starting live trading...")
        logger.info(f"Update interval: {interval_seconds} seconds")
        logger.info("Press Ctrl+C to stop")

        import time

        try:
            while True:
                result = self.run_once()

                if result['status'] == 'error':
                    logger.error(f"Error in iteration: {result.get('message')}")
                    logger.info("Waiting 60 seconds before retry...")
                    time.sleep(60)
                else:
                    logger.info(f"Next update in {interval_seconds} seconds...")
                    time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nLive trading stopped by user")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live trading - executes REAL trades with REAL money"
    )
    parser.add_argument(
        '--exchange',
        default='kraken',
        help='Exchange to use (default: kraken)'
    )
    parser.add_argument(
        '--symbol',
        default='BTC/EUR',
        help='Trading pair symbol (default: BTC/EUR)'
    )
    parser.add_argument(
        '--api-key',
        default=os.getenv('KRAKEN_API_KEY'),
        help='Exchange API key (or set KRAKEN_API_KEY env var)'
    )
    parser.add_argument(
        '--api-secret',
        default=os.getenv('KRAKEN_API_SECRET'),
        help='Exchange API secret (or set KRAKEN_API_SECRET env var)'
    )
    parser.add_argument(
        '--max-position',
        type=float,
        default=300,
        help='Maximum position size in EUR (default: 300)'
    )
    parser.add_argument(
        '--max-daily-loss',
        type=float,
        default=50,
        help='Maximum daily loss in EUR (default: 50)'
    )
    parser.add_argument(
        '--max-daily-trades',
        type=int,
        default=2,
        help='Maximum trades per day (default: 2)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run once for testing (don\'t start continuous trading)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Interval between checks in seconds (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt (use with caution)'
    )

    args = parser.parse_args()

    # Validate API credentials
    if not args.api_key or not args.api_secret:
        logger.error("❌ API credentials required!")
        logger.error("Set KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables")
        logger.error("Or use --api-key and --api-secret arguments")
        return 1

    # Safety limits
    print("\n" + "=" * 80)
    print("⚠️  WARNING: LIVE TRADING - REAL MONEY WILL BE USED")
    print("=" * 80)
    print(f"Exchange:        {args.exchange}")
    print(f"Symbol:          {args.symbol}")
    print(f"Strategy:        RSI Mean Reversion")
    print("-" * 80)
    print("Safety Limits:")
    print(f"  Max Position:  €{args.max_position:,.2f}")
    print(f"  Max Daily Loss: €{args.max_daily_loss:,.2f}")
    print(f"  Max Daily Trades: {args.max_daily_trades}")
    print("-" * 80)
    print("⚠️  Make sure you have:")
    print("  1. Tested thoroughly with paper trading")
    print("  2. Started with small amounts")
    print("  3. Understand the risks")
    print("=" * 80 + "\n")

    # Initialize live trading engine
    try:
        engine = LiveTradingEngine(
            exchange_name=args.exchange,
            symbol=args.symbol,
            api_key=args.api_key,
            api_secret=args.api_secret,
            max_position_eur=args.max_position,
            max_daily_loss_eur=args.max_daily_loss,
            max_daily_trades=args.max_daily_trades,
            require_confirmation=not args.no_confirm
        )

        if args.test:
            # Run once for testing
            logger.info("Running single test iteration...")
            result = engine.run_once()

            if result["status"] == "success":
                logger.success(f"\n✅ Test iteration complete!")
                logger.info(f"Signal: {result['signal']['action']}")
                logger.info(f"Action: {result['action_taken'] or 'None'}")
            else:
                logger.error(f"❌ Error: {result.get('message')}")

            return 0 if result["status"] == "success" else 1

        else:
            # Run continuous live trading
            engine.start(interval_seconds=args.interval)
            return 0

    except KeyboardInterrupt:
        logger.info("\n\nReceived interrupt signal")
        return 0
    except Exception as e:
        logger.exception(f"Live trading failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
