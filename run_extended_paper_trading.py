#!/usr/bin/env python3
"""
GraphWiz Trader - Extended Paper Trading Validation

This script runs a comprehensive 24-72 hour paper trading validation
to test the system thoroughly before live trading.

Features:
- Extended run time (24-72 hours)
- Multiple trading pairs
- Real-time monitoring
- Performance tracking
- Trade logging
- Knowledge graph integration
- Comprehensive metrics

Usage:
    # Run for 24 hours
    python run_extended_paper_trading.py --duration 24

    # Run for 72 hours (recommended)
    python run_extended_paper_trading.py --duration 72

    # Custom symbols
    python run_extended_paper_trading.py --duration 48 --symbols BTC/USDT ETH/USDT SOL/USDT

    # Run in background with nohup
    nohup python run_extended_paper_trading.py --duration 72 > paper_trading.log 2>&1 &
"""

import sys
import os
import asyncio
import signal
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


class ExtendedPaperTradingValidator:
    """
    Extended paper trading validation system.

    Runs paper trading for 24-72 hours with comprehensive monitoring
    and performance tracking.
    """

    def __init__(
        self,
        duration_hours: int = 72,
        symbols: List[str] = None,
        initial_capital: float = 100000.0,
        update_interval_minutes: int = 30
    ):
        """
        Initialize the validator.

        Args:
            duration_hours: How long to run (default: 72 hours)
            symbols: Trading pairs to track
            initial_capital: Starting virtual capital
            update_interval_minutes: How often to check markets (default: 30 min)
        """
        self.duration_hours = duration_hours
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.initial_capital = initial_capital
        self.update_interval = update_interval_minutes * 60  # Convert to seconds

        # State
        self.running = False
        self.start_time = None
        self.stop_time = None

        # Portfolio tracking
        self.portfolio = {
            "USDT": initial_capital,
            **{symbol.split('/')[0]: 0.0 for symbol in self.symbols}
        }

        # Trade tracking
        self.trades = []
        self.equity_curve = []
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0
        }

        # Exchange (simulated)
        self.exchange = self._setup_exchange()

        # Setup logging
        self._setup_logging()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_exchange(self) -> ccxt.Exchange:
        """Setup exchange for paper trading."""
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            },
        })
        logger.info(f"Exchange initialized: {exchange.name}")
        return exchange

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path("logs/paper_trading")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"validation_{timestamp}.log"

        logger.add(
            str(log_file),
            rotation="100 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

        # Trade log
        trade_log = log_dir / f"trades_{timestamp}.csv"
        self.trade_log_file = str(trade_log)

        # Equity curve log
        equity_log = log_dir / f"equity_{timestamp}.csv"
        self.equity_log_file = str(equity_log)

        logger.info(f"Logging initialized: {log_file}")
        logger.info(f"Trade log: {self.trade_log_file}")
        logger.info(f"Equity log: {self.equity_log_file}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current market data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Market data dictionary
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate RSI
            rsi = self._calculate_rsi(df['close'], 14)

            return {
                'symbol': symbol,
                'price': ticker['last'],
                'volume': ticker['baseVolume'],
                'change_24h': ticker['percentage'],
                'rsi': rsi,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except Exception:
            return 50.0

    def _analyze_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals (AGGRESSIVE STRATEGY).

        Args:
            market_data: Market data dictionary

        Returns:
            Signal dictionary with action and confidence
        """
        rsi = market_data.get('rsi', 50)
        change_24h = market_data.get('change_24h', 0)
        price = market_data.get('price', 0)

        # AGGRESSIVE RSI Strategy - Wider trading range
        # Buy when RSI < 42 (was 30), Sell when RSI > 58 (was 70)
        if rsi < 42:
            action = "BUY"
            # Higher base confidence for aggressive trading
            confidence = min(0.95, 0.65 + (42 - rsi) / 80)
            reason = f"RSI oversold/bearish ({rsi:.1f})"

            # Extra boost for very oversold conditions
            if rsi < 35:
                confidence += 0.15
                reason = f"RSI VERY oversold ({rsi:.1f})"

        elif rsi > 58:
            action = "SELL"
            confidence = min(0.95, 0.65 + (rsi - 58) / 80)
            reason = f"RSI overbought/bullish ({rsi:.1f})"

            # Extra boost for very overbought conditions
            if rsi > 65:
                confidence += 0.15
                reason = f"RSI VERY overbought ({rsi:.1f})"

        # Add momentum signals in neutral zone
        elif change_24h < -2.0:
            # Price dropping - buy on dip
            action = "BUY"
            confidence = 0.60 + min(0.15, abs(change_24h) / 20)
            reason = f"Momentum: Price drop {change_24h:.1f}% (dip buying)"

        elif change_24h > 2.0:
            # Price rising - sell into strength
            action = "SELL"
            confidence = 0.60 + min(0.15, change_24h / 20)
            reason = f"Momentum: Price rise {change_24h:.1f}% (profit taking)"

        # Mid-range RSI with no strong momentum - still trade with lower confidence
        elif rsi < 48:
            action = "BUY"
            confidence = 0.55
            reason = f"RSI slightly low ({rsi:.1f}) - slight bullish bias"

        elif rsi > 52:
            action = "SELL"
            confidence = 0.55
            reason = f"RSI slightly high ({rsi:.1f}) - slight bearish bias"

        else:
            # Dead center (RSI 48-52) - trade on momentum if any
            if change_24h < -1.0:
                action = "BUY"
                confidence = 0.52
                reason = f"Slight downtrend ({change_24h:.1f}%) - contrarian buy"
            elif change_24h > 1.0:
                action = "SELL"
                confidence = 0.52
                reason = f"Slight uptrend ({change_24h:.1f}%) - contrarian sell"
            else:
                action = "HOLD"
                confidence = 0.5
                reason = f"No clear signal (RSI: {rsi:.1f}, Change: {change_24h:.1f}%)"

        # Adjust confidence based on 24h trend strength
        if action == "BUY" and change_24h < -3.0:
            confidence += 0.1
            reason += f", strong downtrend ({change_24h:.1f}%)"
        elif action == "SELL" and change_24h > 3.0:
            confidence += 0.1
            reason += f", strong uptrend ({change_24h:.1f}%)"

        return {
            'action': action,
            'confidence': min(confidence, 0.98),  # Cap at 98%
            'reason': reason,
            'rsi': rsi,
            'price': price
        }

    def _execute_trade(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a virtual trade.

        Args:
            symbol: Trading pair
            signal: Trading signal

        Returns:
            Trade execution result
        """
        action = signal['action']
        price = signal['price']
        base_currency = symbol.split('/')[0]

        # Simulate commission and slippage
        commission_rate = 0.001  # 0.1%
        slippage_rate = 0.0005   # 0.05%

        if action == "BUY" and self.portfolio["USDT"] > 100:
            # Calculate position size (25% of available USDT - increased from 10%)
            amount_usdt = self.portfolio["USDT"] * 0.25
            amount = amount_usdt / price

            # Apply commission and slippage
            actual_price = price * (1 + slippage_rate)
            commission = amount_usdt * commission_rate
            total_cost = amount_usdt + commission

            if total_cost <= self.portfolio["USDT"]:
                self.portfolio["USDT"] -= total_cost
                self.portfolio[base_currency] += amount

                trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': actual_price,
                    'amount': amount,
                    'total_usdt': total_cost,
                    'commission': commission,
                    'reason': signal['reason'],
                    'confidence': signal['confidence']
                }

                self.trades.append(trade)
                logger.success(f"BUY {symbol}: {amount:.4f} @ ${actual_price:.2f} | Reason: {signal['reason']}")

                return trade

        elif action == "SELL" and self.portfolio[base_currency] > 0:
            amount = self.portfolio[base_currency] * 0.75  # Sell 75% of position (increased from 50%)

            # Apply commission and slippage
            actual_price = price * (1 - slippage_rate)
            proceeds = amount * actual_price
            commission = proceeds * commission_rate
            net_proceeds = proceeds - commission

            # Calculate P&L
            avg_price = self._calculate_average_price(symbol, base_currency)
            pnl = (actual_price - avg_price) * amount
            pnl_pct = ((actual_price - avg_price) / avg_price) * 100

            self.portfolio[base_currency] -= amount
            self.portfolio["USDT"] += net_proceeds

            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'SELL',
                'price': actual_price,
                'amount': amount,
                'total_usdt': net_proceeds,
                'commission': commission,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': signal['reason'],
                'confidence': signal['confidence']
            }

            self.trades.append(trade)

            if pnl > 0:
                self.metrics['winning_trades'] += 1
                logger.success(f"SELL {symbol}: {amount:.4f} @ ${actual_price:.2f} | P&L: +${pnl:.2f} (+{pnl_pct:.2f}%)")
            else:
                self.metrics['losing_trades'] += 1
                logger.warning(f"SELL {symbol}: {amount:.4f} @ ${actual_price:.2f} | P&L: -${abs(pnl):.2f} ({pnl_pct:.2f}%)")

            return trade

        return None

    def _calculate_average_price(self, symbol: str, base_currency: str) -> float:
        """Calculate average entry price for a position."""
        symbol_trades = [t for t in self.trades if t['symbol'] == symbol and t['action'] == 'BUY']
        if not symbol_trades:
            return 0.0

        total_amount = sum(t['amount'] for t in symbol_trades)
        total_cost = sum(t['total_usdt'] for t in symbol_trades)
        return total_cost / total_amount if total_amount > 0 else 0.0

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value in USDT."""
        total = self.portfolio["USDT"]

        for symbol in self.symbols:
            base_currency = symbol.split('/')[0]
            if self.portfolio[base_currency] > 0:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    total += self.portfolio[base_currency] * ticker['last']
                except Exception:
                    pass

        return total

    def _update_metrics(self):
        """Update performance metrics."""
        current_value = self._calculate_portfolio_value()

        # Total return
        self.metrics['total_return'] = ((current_value - self.initial_capital) / self.initial_capital) * 100

        # Win rate
        total_closed = self.metrics['winning_trades'] + self.metrics['losing_trades']
        if total_closed > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / total_closed) * 100

        # Max drawdown
        if len(self.equity_curve) > 1:
            peak = max(e['value'] for e in self.equity_curve)
            if current_value < peak:
                self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], ((peak - current_value) / peak) * 100)

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 10:
            returns = [self.equity_curve[i]['value'] - self.equity_curve[i-1]['value']
                      for i in range(1, len(self.equity_curve))]
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    self.metrics['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252)  # Annualized

    def _log_status(self):
        """Log current status."""
        current_value = self._calculate_portfolio_value()
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        remaining_hours = max(0, self.duration_hours - elapsed_hours)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"VALIDATION STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info(f"Elapsed:    {elapsed_hours:.1f}h / {self.duration_hours}h ({elapsed_hours/self.duration_hours*100:.1f}%)")
        logger.info(f"Remaining:  {remaining_hours:.1f} hours")
        logger.info("")
        logger.info(f"Portfolio:")
        logger.info(f"  Value:     ${current_value:,.2f}")
        logger.info(f"  Return:    {self.metrics['total_return']:+.2f}%")
        logger.info(f"  Drawdown:  {self.metrics['max_drawdown']:.2f}%")
        logger.info("")
        logger.info(f"Trading:")
        logger.info(f"  Total:     {len(self.trades)} trades")
        logger.info(f"  Wins:      {self.metrics['winning_trades']} ({self.metrics['win_rate']:.1f}%)")
        logger.info(f"  Losses:    {self.metrics['losing_trades']}")
        logger.info("")
        logger.info(f"Metrics:")
        logger.info(f"  Sharpe:    {self.metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Win Rate:  {self.metrics['win_rate']:.1f}%")
        logger.info("=" * 80)
        logger.info("")

        # Log to equity curve
        equity_record = {
            'timestamp': datetime.now().isoformat(),
            'value': current_value,
            'return_pct': self.metrics['total_return'],
            'trades_count': len(self.trades)
        }
        self.equity_curve.append(equity_record)

        # Save to file
        self._save_logs()

    def _save_logs(self):
        """Save logs to files."""
        # Save trades to CSV
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            df_trades.to_csv(self.trade_log_file, index=False)

        # Save equity curve to CSV
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            df_equity.to_csv(self.equity_log_file, index=False)

    def _save_final_report(self):
        """Save final validation report."""
        current_value = self._calculate_portfolio_value()
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        report = {
            'validation_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.stop_time.isoformat() if self.stop_time else None,
                'duration_hours': elapsed_hours,
                'target_duration_hours': self.duration_hours,
                'completion_pct': (elapsed_hours / self.duration_hours) * 100
            },
            'portfolio': {
                'initial_capital': self.initial_capital,
                'final_value': current_value,
                'total_return_usd': current_value - self.initial_capital,
                'total_return_pct': self.metrics['total_return']
            },
            'trading': {
                'total_trades': len(self.trades),
                'winning_trades': self.metrics['winning_trades'],
                'losing_trades': self.metrics['losing_trades'],
                'win_rate_pct': self.metrics['win_rate']
            },
            'metrics': {
                'max_drawdown_pct': self.metrics['max_drawdown'],
                'sharpe_ratio': self.metrics['sharpe_ratio'],
                'final_portfolio': self.portfolio
            },
            'status': self._get_status()
        }

        # Save report
        report_dir = Path("data/paper_trading")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"validation_report_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.success(f"Final report saved: {report_file}")
        logger.success(f"Copy and view: cat {report_file}")

        return report_file

    def _get_status(self) -> str:
        """Get validation status based on metrics."""
        if self.metrics['total_return'] > 5 and self.metrics['max_drawdown'] < 10:
            return "EXCELLENT - Ready for live trading"
        elif self.metrics['total_return'] > 0 and self.metrics['max_drawdown'] < 15:
            return "GOOD - Consider live trading with caution"
        elif self.metrics['total_return'] < 0:
            return "POOR - Not ready for live trading"
        else:
            return "MODERATE - Continue validation"

    async def run(self):
        """Run the extended paper trading validation."""
        self.start_time = datetime.now()
        self.running = True

        logger.success("")
        logger.success("=" * 80)
        logger.success("EXTENDED PAPER TRADING VALIDATION")
        logger.success("=" * 80)
        logger.success(f"Start Time:    {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.success(f"Duration:      {self.duration_hours} hours")
        logger.success(f"Symbols:       {', '.join(self.symbols)}")
        logger.success(f"Capital:       ${self.initial_capital:,.2f}")
        logger.success(f"Update Interval: {self.update_interval / 60:.0f} minutes")
        logger.success("=" * 80)
        logger.success("")

        iteration = 0

        try:
            while self.running:
                iteration += 1
                elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600

                # Check if we've reached the target duration
                if elapsed_hours >= self.duration_hours:
                    logger.success(f"Target duration reached: {elapsed_hours:.1f} hours")
                    break

                logger.info(f"[Iteration {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Analyze each symbol
                for symbol in self.symbols:
                    logger.info(f"  Analyzing {symbol}...")

                    # Fetch market data
                    market_data = self._get_market_data(symbol)
                    if market_data:
                        logger.info(f"    Price: ${market_data['price']:,.2f} | RSI: {market_data['rsi']:.1f}")

                        # Generate signal
                        signal = self._analyze_signals(market_data)
                        logger.info(f"    Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
                        logger.info(f"    Reason: {signal['reason']}")

                        # Execute trade if signal is strong (lowered threshold for aggressive strategy)
                        if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                            self._execute_trade(symbol, signal)

                    await asyncio.sleep(2)  # Small delay between symbols

                # Update metrics and log status
                self._update_metrics()
                self._log_status()

                # Calculate time until next update
                next_update = timedelta(seconds=self.update_interval)
                logger.info(f"Next update in {self.update_interval / 60:.0f} minutes...")
                logger.info("")

                # Wait for next iteration
                await asyncio.sleep(self.update_interval)

        except Exception as e:
            logger.exception(f"Fatal error during validation: {e}")
        finally:
            self.stop_time = datetime.now()
            self.running = False
            self._save_final_report()

            logger.success("")
            logger.success("=" * 80)
            logger.success("VALIDATION COMPLETE")
            logger.success("=" * 80)
            logger.success(f"Total Duration: {elapsed_hours:.1f} hours")
            logger.success(f"Total Trades: {len(self.trades)}")
            logger.success(f"Final Return: {self.metrics['total_return']:+.2f}%")
            logger.success(f"Status: {self._get_status()}")
            logger.success("=" * 80)
            logger.success("")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run extended paper trading validation"
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=72,
        help='Duration in hours (default: 72)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        help='Trading symbols (default: BTC/USDT ETH/USDT SOL/USDT)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital (default: 100000)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Update interval in minutes (default: 30)'
    )

    args = parser.parse_args()

    validator = ExtendedPaperTradingValidator(
        duration_hours=args.duration,
        symbols=args.symbols,
        initial_capital=args.capital,
        update_interval_minutes=args.interval
    )

    await validator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
