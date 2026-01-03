#!/usr/bin/env python3
"""
GraphWiz Trader - Live Trading Mock Test

This script simulates live trading with realistic market data but MOCK execution.
It tests the entire live trading pipeline without using real money or API credentials.

Key Features:
- Real market data from Kraken public API (no credentials needed)
- Live trading risk parameters (€300 position, €50 daily loss)
- Mock order execution (no actual trades)
- Complete log generation
- Safety mechanism validation
- Error handling testing

Usage:
    # Run 1-hour mock test
    python test_live_trading_mock.py --duration 1 --interval 5

    # Run 30-minute test
    python test_live_trading_mock.py --duration 0.5 --interval 2
"""

import sys
import os
import signal
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from loguru import logger
    import ccxt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install dependencies:")
    print("  pip install loguru ccxt pandas numpy")
    sys.exit(1)


class LiveTradingMockTester:
    """
    Mock live trading test system.

    Simulates live trading with real market data but mock execution.
    Tests the entire pipeline before deploying to production.
    """

    def __init__(
        self,
        duration_hours: float = 1.0,
        symbols: List[str] = None,
        initial_capital_eur: float = 300.0,
        max_position_eur: float = 300.0,
        max_daily_loss_eur: float = 50.0,
        max_daily_trades: int = 2,
        update_interval_minutes: int = 5
    ):
        """
        Initialize the mock tester.

        Args:
            duration_hours: How long to run (default: 1 hour for testing)
            symbols: Trading pairs (EUR pairs for Germany)
            initial_capital_eur: Starting capital in EUR
            max_position_eur: Maximum position size
            max_daily_loss_eur: Maximum daily loss limit
            max_daily_trades: Maximum trades per day
            update_interval_minutes: How often to check markets
        """
        self.duration_hours = duration_hours
        self.symbols = symbols or ["BTC/EUR", "ETH/EUR"]
        self.initial_capital = initial_capital_eur
        self.max_position = max_position_eur
        self.max_daily_loss = max_daily_loss_eur
        self.max_daily_trades = max_daily_trades
        self.update_interval = update_interval_minutes * 60  # Convert to seconds

        # State
        self.running = False
        self.start_time = None
        self.stop_time = None

        # Portfolio tracking
        self.portfolio = {
            "EUR": initial_capital_eur,
            **{symbol.split('/')[0]: 0.0 for symbol in self.symbols}
        }

        # Trade tracking
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = 0.0
        self.daily_trade_count = 0

        # Mock exchange (Kraken public API - no credentials needed)
        self.exchange = ccxt.kraken({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup structured logging for live trading."""
        log_dir = Path("logs/live_trading")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        logger.add(
            log_dir / "live_trading.log",
            rotation="10 MB",
            retention="90 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
        )

        # Trades log file
        logger.add(
            log_dir / "trades.log",
            rotation="10 MB",
            retention="365 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            filter=lambda record: "TRADE" in record["extra"]
        )

        # Errors log file
        logger.add(
            log_dir / "errors.log",
            rotation="10 MB",
            retention="365 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
        )

    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(closes) < period:
            return 50.0  # Neutral if not enough data

        df = pd.DataFrame({'close': closes})
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def generate_signal(self, symbol: str, rsi: float, price: float) -> Dict[str, Any]:
        """Generate trading signal based on RSI."""
        # Live trading conservative thresholds
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
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'price': price,
            'rsi': rsi,
            'timestamp': datetime.now().isoformat()
        }

    def mock_execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Mock execute a trade (no real order sent)."""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        confidence = signal['confidence']

        # Check risk limits
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_trade_count}/{self.max_daily_trades}")
            return {'status': 'rejected', 'reason': 'Daily trade limit reached'}

        if self.daily_pnl <= -self.max_daily_loss:
            logger.error(f"Daily loss limit reached: €{self.daily_pnl:.2f}")
            return {'status': 'rejected', 'reason': 'Daily loss limit reached'}

        # Check if we have enough capital (for buys)
        base_currency = symbol.split('/')[0]
        if action == "BUY":
            position_size_eur = min(self.max_position, self.portfolio["EUR"] * 0.25)
            if position_size_eur < 10:  # Minimum trade size
                return {'status': 'rejected', 'reason': 'Insufficient funds'}
        else:
            # For sells, check if we have the asset
            if self.portfolio[base_currency] <= 0:
                return {'status': 'rejected', 'reason': 'No position to sell'}
            position_size_eur = self.portfolio[base_currency] * price * 0.75

        # Calculate trade amount
        amount = position_size_eur / price

        # Mock execution (record the trade but don't actually execute)
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'price': price,
            'amount': amount,
            'total_eur': position_size_eur,
            'commission': position_size_eur * 0.0016,  # Kraken fee
            'reason': signal['reason'],
            'confidence': confidence
        }

        # Update portfolio
        if action == "BUY":
            self.portfolio["EUR"] -= (position_size_eur + trade['commission'])
            self.portfolio[base_currency] += amount
        else:  # SELL
            self.portfolio[base_currency] -= amount
            self.portfolio["EUR"] += (position_size_eur - trade['commission'])

        self.trades.append(trade)
        self.daily_trade_count += 1

        # Log trade
        logger.bind(TRADE=True).info(
            f"MOCK TRADE: {action} {amount:.6f} {base_currency} @ €{price:.2f} "
            f"(Total: €{position_size_eur:.2f}, Fee: €{trade['commission']:.4f})"
        )

        return {'status': 'executed', 'trade': trade}

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value in EUR."""
        total = self.portfolio["EUR"]

        for symbol in self.symbols:
            base_currency = symbol.split('/')[0]
            if base_currency in self.portfolio:
                total += self.portfolio[base_currency] * current_prices.get(symbol, 0)

        return total

    def update_equity_curve(self, portfolio_value: float):
        """Update equity curve."""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        initial = self.initial_capital
        return_pct = ((portfolio_value - initial) / initial) * 100

        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'value': portfolio_value,
            'return_pct': return_pct,
            'trades_count': len(self.trades)
        })

        # Also save to CSV
        log_dir = Path("logs/live_trading")
        equity_file = log_dir / f"equity_mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        if len(self.equity_curve) == 1:
            # Write header
            with open(equity_file, 'w') as f:
                f.write("timestamp,value,return_pct,trades_count\n")

        with open(equity_file, 'a') as f:
            f.write(f"{self.equity_curve[-1]['timestamp']},{portfolio_value},{return_pct:.6f},{len(self.trades)}\n")

    def _log_status(self, iteration: int):
        """Log current status."""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        remaining = max(0, self.duration_hours - elapsed)

        # Get current prices
        current_prices = {}
        for symbol in self.symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_prices[symbol] = ticker['last']
            except Exception as e:
                logger.error(f"Failed to fetch ticker for {symbol}: {e}")
                current_prices[symbol] = 0

        portfolio_value = self.calculate_portfolio_value(current_prices)
        self.update_equity_curve(portfolio_value)

        initial = self.initial_capital
        return_pct = ((portfolio_value - initial) / initial) * 100
        drawdown = min(0, return_pct)

        logger.info("=" * 80)
        logger.info("MOCK LIVE TRADING STATUS - {}", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("=" * 80)
        logger.info(f"Elapsed:    {elapsed:.1f}h / {self.duration_hours}h ({elapsed/self.duration_hours*100:.1f}%)")
        logger.info(f"Remaining:  {remaining:.1f} hours")
        logger.info("")
        logger.info("Portfolio:")
        logger.info(f"  Value:     €{portfolio_value:.2f}")
        logger.info(f"  Return:    {return_pct:+.2f}%")
        logger.info(f"  Drawdown:  {drawdown:.2f}%")
        logger.info("")
        logger.info("Trading:")
        logger.info(f"  Total:     {len(self.trades)} trades")
        logger.info(f"  Today:     {self.daily_trade_count}/{self.max_daily_trades}")
        logger.info(f"  Daily P&L: €{self.daily_pnl:+.2f}")
        logger.info("=" * 80)
        logger.info("")

    async def run_once(self):
        """Run one analysis iteration."""
        iteration = 0

        while self.running:
            try:
                iteration += 1
                logger.info(f"Iteration {iteration} - Analyzing markets...")

                current_prices = {}

                # Analyze each symbol
                for symbol in self.symbols:
                    try:
                        # Fetch OHLCV data (public API, no credentials needed)
                        ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                        closes = df['close'].tolist()
                        current_price = closes[-1]
                        current_prices[symbol] = current_price

                        # Calculate RSI
                        rsi = self.calculate_rsi(closes)

                        # Generate signal
                        signal = self.generate_signal(symbol, rsi, current_price)

                        logger.info(f"  {symbol}:")
                        logger.info(f"    Price: €{current_price:.2f} | RSI: {rsi:.1f}")
                        logger.info(f"    Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
                        logger.info(f"    Reason: {signal['reason']}")

                        # Execute if signal is strong enough
                        if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                            result = self.mock_execute_trade(signal)
                            if result['status'] == 'executed':
                                logger.success(f"    ✓ MOCK order executed")
                            else:
                                logger.warning(f"    ✗ Order rejected: {result['reason']}")

                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                        continue

                # Log status
                self._log_status(iteration)

                # Check if we should stop
                elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
                if elapsed >= self.duration_hours:
                    logger.info("Duration reached. Stopping...")
                    self.stop()

                # Emergency stop checks
                if self.daily_pnl <= -self.max_daily_loss:
                    logger.error("Daily loss limit reached. Stopping...")
                    self.stop()

                # Wait for next iteration
                if self.running:
                    logger.info(f"Next update in {self.update_interval // 60} minutes...")
                    logger.info("")
                    await asyncio.sleep(self.update_interval)

            except KeyboardInterrupt:
                logger.info("Interrupt received. Stopping...")
                self.stop()
            except Exception as e:
                logger.exception(f"Unexpected error in iteration {iteration}: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    def start(self):
        """Start the mock trading test."""
        logger.info("=" * 80)
        logger.info("STARTING MOCK LIVE TRADING TEST")
        logger.info("=" * 80)
        logger.info(f"Duration: {self.duration_hours} hours")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Initial Capital: €{self.initial_capital:.2f}")
        logger.info(f"Max Position: €{self.max_position:.2f}")
        logger.info(f"Max Daily Loss: €{self.max_daily_loss:.2f}")
        logger.info(f"Max Daily Trades: {self.max_daily_trades}")
        logger.info(f"Update Interval: {self.update_interval // 60} minutes")
        logger.info("")
        logger.warning("⚠️  MOCK MODE - No real trades will be executed")
        logger.warning("⚠️  This tests the live trading pipeline with fake execution")
        logger.info("=" * 80)
        logger.info("")

        self.running = True
        self.start_time = datetime.now()

        try:
            asyncio.run(self.run_once())
        finally:
            self.save_results()
            self.print_summary()

    def stop(self):
        """Stop the mock trading test."""
        self.running = False
        self.stop_time = datetime.now()

    def save_results(self):
        """Save results to file."""
        log_dir = Path("logs/live_trading")
        results_file = log_dir / f"mock_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results = {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stop_time': self.stop_time.isoformat() if self.stop_time else None,
            'duration_hours': self.duration_hours,
            'symbols': self.symbols,
            'initial_capital': self.initial_capital,
            'final_portfolio': self.portfolio,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': {
                'total_trades': len(self.trades),
                'final_value': self.calculate_portfolio_value({}),
                'daily_trade_count': self.daily_trade_count,
                'daily_pnl': self.daily_pnl
            }
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.success(f"Results saved to: {results_file}")

    def print_summary(self):
        """Print test summary."""
        if not self.start_time:
            return

        elapsed = (self.stop_time - self.start_time).total_seconds() / 3600 if self.stop_time else 0
        final_value = self.calculate_portfolio_value({})
        return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100

        logger.info("")
        logger.info("=" * 80)
        logger.info("MOCK TRADING TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {elapsed:.2f} hours")
        logger.info(f"Trades Executed: {len(self.trades)}")
        logger.info(f"Initial Capital: €{self.initial_capital:.2f}")
        logger.info(f"Final Value: €{final_value:.2f}")
        logger.info(f"Return: {return_pct:+.2f}%")
        logger.info("")
        logger.info("✅ Mock test completed successfully!")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mock live trading test - tests pipeline without real trades"
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=1.0,
        help='Test duration in hours (default: 1.0)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Update interval in minutes (default: 5)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['BTC/EUR', 'ETH/EUR'],
        help='Trading symbols (default: BTC/EUR ETH/EUR)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=300.0,
        help='Initial capital in EUR (default: 300)'
    )

    args = parser.parse_args()

    # Create tester
    tester = LiveTradingMockTester(
        duration_hours=args.duration,
        symbols=args.symbols,
        initial_capital_eur=args.capital,
        max_position_eur=args.capital,
        max_daily_loss_eur=args.capital * 0.16,  # 16% daily loss limit
        max_daily_trades=2,
        update_interval_minutes=args.interval
    )

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        tester.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run test
    try:
        tester.start()
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
