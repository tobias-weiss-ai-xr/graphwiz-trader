"""
Trading performance monitor.

Tracks:
- Portfolio performance
- Trade statistics
- Risk metrics
- System health
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import pandas as pd
from loguru import logger

from .alerts import AlertManager, AlertLevel


class TradingMonitor:
    """Monitor trading performance and send alerts."""

    def __init__(
        self,
        symbol: str,
        initial_capital: float,
        alert_manager: Optional[AlertManager] = None,
    ):
        """Initialize trading monitor.

        Args:
            symbol: Trading pair symbol
            initial_capital: Starting capital
            alert_manager: AlertManager instance
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.alert_manager = alert_manager or AlertManager()

        # Performance tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

        # Daily tracking
        self.daily_stats: Dict[str, Any] = {}
        self.current_day = datetime.now().date()

        # Health checks
        self.last_price_update = None
        self.last_trade_time = None
        self.error_count = 0
        self.max_consecutive_errors = 5

    def update_equity(
        self,
        timestamp: datetime,
        portfolio_value: float,
        price: float,
        position: float = 0,
    ):
        """Update equity curve.

        Args:
            timestamp: Current timestamp
            portfolio_value: Total portfolio value
            price: Current asset price
            position: Current position size
        """
        # Check for new day
        if timestamp.date() != self.current_day:
            self._send_daily_summary()

        self.equity_curve.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value,
            "price": price,
            "position": position,
            "return_pct": ((portfolio_value - self.initial_capital) / self.initial_capital) * 100,
        })

        self.last_price_update = timestamp

        # Check for significant events
        self._check_performance_alerts(portfolio_value)

    def record_trade(
        self,
        action: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None,
        pnl: Optional[float] = None,
    ):
        """Record a trade.

        Args:
            action: Trade action ("buy" or "sell")
            quantity: Quantity traded
            price: Execution price
            timestamp: Trade timestamp
            pnl: Profit/loss (for sells)
        """
        if timestamp is None:
            timestamp = datetime.now()

        trade = {
            "timestamp": timestamp,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "pnl": pnl,
        }

        self.trades.append(trade)
        self.last_trade_time = timestamp

        # Update daily stats
        self._update_daily_stats(trade)

        # Send alert
        if action == "buy":
            self.alert_manager.trade_executed(action, self.symbol, quantity, price)
        elif action == "sell" and pnl is not None:
            self.alert_manager.position_closed(
                self.symbol, quantity, self._get_last_entry_price(), price, pnl
            )

        logger.info(f"Trade recorded: {action.upper()} {quantity:.4f} @ ${price:,.2f}")

    def _get_last_entry_price(self) -> float:
        """Get the last entry price.

        Returns:
            Last buy price or 0 if no position
        """
        for trade in reversed(self.trades):
            if trade["action"] == "buy":
                return trade["price"]
        return 0.0

    def _update_daily_stats(self, trade: Dict[str, Any]):
        """Update daily statistics.

        Args:
            trade: Trade dict
        """
        today = datetime.now().date()

        if today != self.current_day:
            self.daily_stats = {}
            self.current_day = today

        if "trades_count" not in self.daily_stats:
            self.daily_stats["trades_count"] = 0
            self.daily_stats["buy_trades"] = 0
            self.daily_stats["sell_trades"] = 0
            self.daily_stats["daily_pnl"] = 0.0

        self.daily_stats["trades_count"] += 1

        if trade["action"] == "buy":
            self.daily_stats["buy_trades"] += 1
        elif trade["action"] == "sell":
            self.daily_stats["sell_trades"] += 1
            if trade["pnl"] is not None:
                self.daily_stats["daily_pnl"] += trade["pnl"]

    def _check_performance_alerts(self, portfolio_value: float):
        """Check for performance alerts.

        Args:
            portfolio_value: Current portfolio value
        """
        total_return = portfolio_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Significant loss alert
        if total_return_pct < -10:
            self.alert_manager.critical_alert(
                f"Portfolio down {total_return_pct:.2f}%",
                {
                    "Portfolio Value": f"${portfolio_value:,.2f}",
                    "Total Return": f"{total_return_pct:.2f}%",
                    "Initial Capital": f"${self.initial_capital:,.2f}",
                },
            )

        # Good performance alert
        elif total_return_pct > 10:
            self.alert_manager.send_alert(
                level=AlertLevel.SUCCESS,
                title="🎉 Great Performance!",
                message=f"Portfolio up {total_return_pct:.2f}%",
                metadata={
                    "Portfolio Value": f"${portfolio_value:,.2f}",
                    "Total Return": f"{total_return_pct:.2f}%",
                },
            )

    def _send_daily_summary(self):
        """Send daily summary alert."""
        if not self.daily_stats:
            return

        portfolio_value = self.equity_curve[-1]["portfolio_value"] if self.equity_curve else self.initial_capital
        daily_pnl = self.daily_stats.get("daily_pnl", 0)
        trades_count = self.daily_stats.get("trades_count", 0)

        self.alert_manager.daily_summary(portfolio_value, daily_pnl, trades_count)

        logger.success(f"Daily summary sent: P&L ${daily_pnl:+,.2f}, {trades_count} trades")

    def check_health(self) -> Dict[str, Any]:
        """Check system health.

        Returns:
            Health status dict
        """
        health = {
            "status": "healthy",
            "issues": [],
        }

        # Check if price updates are recent
        if self.last_price_update:
            time_since_update = datetime.now() - self.last_price_update
            if time_since_update > timedelta(minutes=10):
                health["status"] = "warning"
                health["issues"].append("Price update stale")

        # Check error count
        if self.error_count > self.max_consecutive_errors:
            health["status"] = "critical"
            health["issues"].append(f"Too many errors: {self.error_count}")

        # Check for drawdown
        if self.equity_curve:
            current_value = self.equity_curve[-1]["portfolio_value"]
            max_value = max(e["portfolio_value"] for e in self.equity_curve)
            drawdown_pct = ((current_value - max_value) / max_value) * 100

            if drawdown_pct < -10:
                health["status"] = "critical"
                health["issues"].append(f"Max drawdown exceeded: {drawdown_pct:.2f}%")

        return health

    def increment_error_count(self):
        """Increment error count."""
        self.error_count += 1

        if self.error_count >= self.max_consecutive_errors:
            self.alert_manager.critical_alert(
                f"System error count: {self.error_count}",
                {"Max Consecutive Errors": self.max_consecutive_errors},
            )

    def reset_error_count(self):
        """Reset error count."""
        self.error_count = 0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Performance metrics dict
        """
        if not self.equity_curve:
            return {}

        current_equity = self.equity_curve[-1]
        initial_value = self.initial_capital

        # Calculate metrics
        total_return = current_equity["portfolio_value"] - initial_value
        total_return_pct = (total_return / initial_value) * 100

        # Trade statistics
        buy_trades = [t for t in self.trades if t["action"] == "buy"]
        sell_trades = [t for t in self.trades if t["action"] == "sell"]

        # Calculate win rate
        winning_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0

        # Calculate max drawdown
        equity_values = [e["portfolio_value"] for e in self.equity_curve]
        running_max = pd.Series(equity_values).cummax()
        drawdown = pd.Series(equity_values) - running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = pd.Series(equity_values).pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            "start_time": self.start_time,
            "current_time": datetime.now(),
            "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "initial_capital": initial_value,
            "current_value": current_equity["portfolio_value"],
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_trades": len(self.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "win_rate": win_rate * 100,
            "winning_trades": len(winning_trades),
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "daily_pnl": self.daily_stats.get("daily_pnl", 0),
            "daily_trades": self.daily_stats.get("trades_count", 0),
        }

    def save_results(self, output_dir: str = "data/monitoring"):
        """Save monitoring results.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_clean = self.symbol.replace("/", "_")

        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = output_path / f"{symbol_clean}_equity_{timestamp_str}.csv"
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"Saved equity curve to {equity_file}")

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = output_path / f"{symbol_clean}_trades_{timestamp_str}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved trades to {trades_file}")

        # Save metrics
        metrics = self.get_performance_metrics()
        if metrics:
            metrics_file = output_path / f"{symbol_clean}_metrics_{timestamp_str}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Saved metrics to {metrics_file}")

    def print_summary(self):
        """Print performance summary."""
        metrics = self.get_performance_metrics()

        if not metrics:
            logger.warning("No metrics available")
            return

        print("\n" + "=" * 80)
        print("TRADING PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Symbol:         {self.symbol}")
        print(f"Duration:       {metrics['duration_hours']:.1f} hours")
        print("-" * 80)
        print(f"Initial Capital:  ${metrics['initial_capital']:,.2f}")
        print(f"Current Value:    ${metrics['current_value']:,.2f}")
        print(f"Total Return:     ${metrics['total_return']:+,.2f} ({metrics['total_return_pct']:+.2f}%)")
        print("-" * 80)
        print(f"Total Trades:     {metrics['total_trades']}")
        print(f"Win Rate:         {metrics['win_rate']:.2f}%")
        print(f"Max Drawdown:     ${metrics['max_drawdown']:,.2f}")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print("-" * 80)
        print(f"Daily P&L:        ${metrics['daily_pnl']:+,.2f}")
        print(f"Daily Trades:     {metrics['daily_trades']}")
        print("=" * 80 + "\n")
