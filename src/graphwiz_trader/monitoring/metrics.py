"""Metrics collection and Prometheus integration."""

import time
import psutil
from typing import Dict, Any, Optional, List
from decimal import Decimal
from collections import defaultdict
from datetime import datetime, timedelta
from loguru import logger

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Install with: pip install prometheus_client")


class MetricsCollector:
    """Comprehensive metrics collector for graphwiz-trader.

    Collects system, trading, exchange, agent, and risk metrics with Prometheus export.
    """

    def __init__(self, prometheus_port: int = 8000, enable_prometheus: bool = True):
        """Initialize metrics collector.

        Args:
            prometheus_port: Port for Prometheus metrics server
            enable_prometheus: Enable Prometheus metrics export
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port
        self.start_time = time.time()

        # Internal state tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.exchange_latencies: Dict[str, List[float]] = defaultdict(list)
        self.agent_predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.portfolio_value_history: List[Dict[str, Any]] = []

        # Initialize Prometheus metrics
        self._init_prometheus_metrics()

        if self.enable_prometheus:
            self._start_prometheus_server()

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metric collectors."""
        if not self.enable_prometheus:
            return

        # System Metrics
        self.cpu_usage = Gauge('graphwiz_system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('graphwiz_system_memory_usage_bytes', 'Memory usage in bytes')
        self.memory_available = Gauge('graphwiz_system_memory_available_bytes', 'Available memory in bytes')
        self.disk_usage = Gauge('graphwiz_system_disk_usage_percent', 'Disk usage percentage', ['mount'])
        self.network_sent = Gauge('graphwiz_system_network_sent_bytes', 'Network bytes sent', ['interface'])
        self.network_recv = Gauge('graphwiz_system_network_recv_bytes', 'Network bytes received', ['interface'])

        # Trading Metrics
        self.total_trades = Counter('graphwiz_trading_total_trades', 'Total number of trades')
        self.successful_trades = Counter('graphwiz_trading_successful_trades', 'Number of successful trades')
        self.failed_trades = Counter('graphwiz_trading_failed_trades', 'Number of failed trades')
        self.trade_pnl = Gauge('graphwiz_trading_pnl_usd', 'Total P&L in USD')
        self.trade_pnl_daily = Gauge('graphwiz_trading_pnl_daily_usd', 'Daily P&L in USD')
        self.win_rate = Gauge('graphwiz_trading_win_rate', 'Win rate percentage')
        self.sharpe_ratio = Gauge('graphwiz_trading_sharpe_ratio', 'Sharpe ratio')
        self.max_drawdown = Gauge('graphwiz_trading_max_drawdown', 'Maximum drawdown percentage')
        self.current_drawdown = Gauge('graphwiz_trading_current_drawdown', 'Current drawdown percentage')

        # Trading Performance Histograms
        self.trade_execution_time = Histogram(
            'graphwiz_trading_execution_seconds',
            'Trade execution time in seconds',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        self.trade_size = Summary('graphwiz_trading_size_usd', 'Trade size in USD')

        # Exchange Metrics
        self.exchange_latency = Histogram(
            'graphwiz_exchange_latency_seconds',
            'Exchange API latency in seconds',
            ['exchange'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        self.exchange_order_fill_rate = Gauge('graphwiz_exchange_fill_rate', 'Order fill rate', ['exchange'])
        self.exchange_error_rate = Gauge('graphwiz_exchange_error_rate', 'Error rate', ['exchange'])
        self.exchange_rate_limit_remaining = Gauge('graphwiz_exchange_rate_limit_remaining', 'Rate limit remaining', ['exchange'])
        self.exchange_connection_status = Gauge('graphviz_exchange_connected', 'Exchange connection status', ['exchange'])

        # Agent Metrics
        self.agent_accuracy = Gauge('graphwiz_agent_accuracy', 'Agent prediction accuracy', ['agent'])
        self.agent_confidence = Gauge('graphwiz_agent_confidence', 'Average agent confidence', ['agent'])
        self.agent_consensus_rate = Gauge('graphwiz_agent_consensus_rate', 'Agent consensus rate')
        self.agent_prediction_count = Counter('graphwiz_agent_predictions_total', 'Total agent predictions', ['agent'])
        self.agent_active_agents = Gauge('graphwiz_agent_active_count', 'Number of active agents')

        # Risk Metrics
        self.risk_var_95 = Gauge('graphwiz_risk_var_95_usd', 'Value at Risk (95%) in USD')
        self.risk_var_99 = Gauge('graphwiz_risk_var_99_usd', 'Value at Risk (99%) in USD')
        self.risk_exposure_usd = Gauge('graphwiz_risk_exposure_usd', 'Current exposure in USD')
        self.risk_correlation = Gauge('graphwiz_risk_correlation_avg', 'Average portfolio correlation')
        self.risk_position_count = Gauge('graphwiz_risk_position_count', 'Number of open positions')
        self.risk_leverage = Gauge('graphwiz_risk_leverage', 'Current leverage ratio')

        # Portfolio Metrics
        self.portfolio_value = Gauge('graphwiz_portfolio_value_usd', 'Portfolio value in USD')
        self.portfolio_cash = Gauge('graphwiz_portfolio_cash_usd', 'Available cash in USD')
        self.portfolio_positions_value = Gauge('graphwiz_portfolio_positions_value_usd', 'Position value in USD')
        self.portfolio_returns_1h = Gauge('graphwiz_portfolio_returns_1h', '1-hour returns')
        self.portfolio_returns_24h = Gauge('graphwiz_portfolio_returns_24h', '24-hour returns')
        self.portfolio_returns_7d = Gauge('graphwiz_portfolio_returns_7d', '7-day returns')

        # System Info
        self.system_info = Info('graphwiz_system', 'Graphwiz Trader system info')

    def _start_prometheus_server(self) -> None:
        """Start Prometheus HTTP server."""
        try:
            start_http_server(self.prometheus_port)
            logger.info("Prometheus metrics server started on port {}", self.prometheus_port)
        except Exception as e:
            logger.error("Failed to start Prometheus server: {}", e)

    # System Metrics Collection
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics.

        Returns:
            Dictionary of system metrics
        """
        metrics = {
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used,
                'free': psutil.virtual_memory().free
            },
            'disk': {},
            'network': {
                'io_counters': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None
            },
            'load_avg': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None,
            'boot_time': psutil.boot_time()
        }

        # Disk usage for all mounts
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                metrics['disk'][partition.mountpoint] = {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                }
            except PermissionError:
                continue

        # Network I/O per interface
        metrics['network']['interfaces'] = {}
        for name, addrs in psutil.net_if_addrs().items():
            metrics['network']['interfaces'][name] = {
                'addresses': [addr.address for addr in addrs]
            }

        # Update Prometheus metrics
        if self.enable_prometheus:
            self.cpu_usage.set(metrics['cpu']['percent'])
            self.memory_usage.set(metrics['memory']['used'])
            self.memory_available.set(metrics['memory']['available'])
            for mount, disk in metrics['disk'].items():
                self.disk_usage.labels(mount=mount).set(disk['percent'])

        return metrics

    # Trading Metrics
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a trade and update metrics.

        Args:
            trade: Trade dictionary with symbol, side, price, quantity, pnl, etc.
        """
        self.trade_history.append({
            **trade,
            'timestamp': datetime.utcnow().isoformat()
        })

        if self.enable_prometheus:
            self.total_trades.inc()

            if trade.get('success', True):
                self.successful_trades.inc()
            else:
                self.failed_trades.inc()

            if 'pnl' in trade:
                pnl_value = float(trade['pnl'])
                current_total = self.trade_pnl._value.get() if hasattr(self.trade_pnl, '_value') else 0
                self.trade_pnl.set(current_total + pnl_value)
                self.trade_size.observe(float(trade.get('quantity', 0) * trade.get('price', 0)))

            if 'execution_time' in trade:
                self.trade_execution_time.observe(trade['execution_time'])

        logger.debug("Recorded trade: {}", trade)

    def calculate_trading_metrics(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Calculate trading performance metrics.

        Args:
            lookback_hours: Hours to look back for calculations

        Returns:
            Dictionary of trading metrics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent_trades = [
            t for t in self.trade_history
            if datetime.fromisoformat(t['timestamp']) > cutoff_time
        ]

        if not recent_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0
            }

        # Basic metrics
        total_trades = len(recent_trades)
        winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_pnl = sum(t.get('pnl', 0) for t in recent_trades)

        # Calculate returns for Sharpe ratio
        returns = [t.get('pnl', 0) for t in recent_trades]
        if returns:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            sharpe_ratio = (avg_return / std_dev) if std_dev > 0 else 0
        else:
            sharpe_ratio = 0

        # Drawdown calculation
        cumulative_pnl = []
        running_total = 0
        for t in recent_trades:
            running_total += t.get('pnl', 0)
            cumulative_pnl.append(running_total)

        if cumulative_pnl:
            peak = max(cumulative_pnl)
            current_value = cumulative_pnl[-1]
            max_drawdown = ((peak - min(cumulative_pnl)) / peak) if peak > 0 else 0
            current_drawdown = ((peak - current_value) / peak) if peak > 0 else 0
        else:
            max_drawdown = 0
            current_drawdown = 0

        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': total_trades - len(winning_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown
        }

        # Update Prometheus
        if self.enable_prometheus:
            self.win_rate.set(win_rate * 100)
            self.sharpe_ratio.set(sharpe_ratio)
            self.max_drawdown.set(max_drawdown * 100)
            self.current_drawdown.set(current_drawdown * 100)
            self.trade_pnl.set(total_pnl)

            # Daily P&L (last 24 hours)
            daily_pnl = sum(t.get('pnl', 0) for t in recent_trades)
            self.trade_pnl_daily.set(daily_pnl)

        return metrics

    # Exchange Metrics
    def record_exchange_latency(self, exchange: str, latency: float) -> None:
        """Record exchange API latency.

        Args:
            exchange: Exchange name
            latency: Latency in seconds
        """
        self.exchange_latencies[exchange].append(latency)

        # Keep only last 1000 measurements
        if len(self.exchange_latencies[exchange]) > 1000:
            self.exchange_latencies[exchange] = self.exchange_latencies[exchange][-1000:]

        if self.enable_prometheus:
            self.exchange_latency.labels(exchange=exchange).observe(latency)

        logger.debug("Recorded {} latency: {:.3f}s", exchange, latency)

    def get_exchange_metrics(self, exchange: str) -> Dict[str, Any]:
        """Get metrics for specific exchange.

        Args:
            exchange: Exchange name

        Returns:
            Dictionary of exchange metrics
        """
        latencies = self.exchange_latencies.get(exchange, [])

        if not latencies:
            return {
                'exchange': exchange,
                'avg_latency': 0,
                'p50_latency': 0,
                'p95_latency': 0,
                'p99_latency': 0,
                'sample_count': 0
            }

        sorted_latencies = sorted(latencies)
        metrics = {
            'exchange': exchange,
            'avg_latency': sum(latencies) / len(latencies),
            'p50_latency': sorted_latencies[len(sorted_latencies) // 2],
            'p95_latency': sorted_latencies[int(len(sorted_latencies) * 0.95)],
            'p99_latency': sorted_latencies[int(len(sorted_latencies) * 0.99)],
            'sample_count': len(latencies)
        }

        return metrics

    def update_exchange_connection_status(self, exchange: str, connected: bool) -> None:
        """Update exchange connection status.

        Args:
            exchange: Exchange name
            connected: Connection status
        """
        if self.enable_prometheus:
            self.exchange_connection_status.labels(exchange=exchange).set(1 if connected else 0)

    def update_exchange_rate_limit(self, exchange: str, remaining: int) -> None:
        """Update exchange rate limit remaining.

        Args:
            exchange: Exchange name
            remaining: Remaining requests
        """
        if self.enable_prometheus:
            self.exchange_rate_limit_remaining.labels(exchange=exchange).set(remaining)

    # Agent Metrics
    def record_agent_prediction(
        self,
        agent_name: str,
        prediction: Dict[str, Any],
        actual_outcome: Optional[float] = None
    ) -> None:
        """Record agent prediction and optionally actual outcome.

        Args:
            agent_name: Name of the agent
            prediction: Prediction dictionary with confidence, etc.
            actual_outcome: Actual outcome (if known) for accuracy calculation
        """
        record = {
            **prediction,
            'timestamp': datetime.utcnow().isoformat()
        }
        if actual_outcome is not None:
            record['actual_outcome'] = actual_outcome

        self.agent_predictions[agent_name].append(record)

        # Keep only last 1000 predictions per agent
        if len(self.agent_predictions[agent_name]) > 1000:
            self.agent_predictions[agent_name] = self.agent_predictions[agent_name][-1000:]

        if self.enable_prometheus:
            self.agent_prediction_count.labels(agent=agent_name).inc()
            if 'confidence' in prediction:
                self.agent_confidence.labels(agent=agent_name).set(prediction['confidence'])

        logger.debug("Recorded prediction from {}: {}", agent_name, prediction)

    def calculate_agent_metrics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Calculate agent performance metrics.

        Args:
            agent_name: Specific agent or None for all agents

        Returns:
            Dictionary of agent metrics
        """
        if agent_name:
            predictions = self.agent_predictions.get(agent_name, [])
            agents_to_analyze = {agent_name: predictions}
        else:
            agents_to_analyze = self.agent_predictions

        metrics = {}

        for agent, preds in agents_to_analyze.items():
            if not preds:
                metrics[agent] = {
                    'total_predictions': 0,
                    'avg_confidence': 0,
                    'accuracy': 0,
                    'recent_predictions': 0
                }
                continue

            total = len(preds)
            avg_confidence = sum(p.get('confidence', 0) for p in preds) / total

            # Calculate accuracy if we have actual outcomes
            predictions_with_outcome = [p for p in preds if 'actual_outcome' in p]
            if predictions_with_outcome:
                correct = sum(
                    1 for p in predictions_with_outcome
                    if (p.get('prediction', 0) > 0) == (p.get('actual_outcome', 0) > 0)
                )
                accuracy = correct / len(predictions_with_outcome)
            else:
                accuracy = 0

            # Recent predictions (last hour)
            cutoff = datetime.utcnow() - timedelta(hours=1)
            recent = sum(1 for p in preds if datetime.fromisoformat(p['timestamp']) > cutoff)

            metrics[agent] = {
                'total_predictions': total,
                'avg_confidence': avg_confidence,
                'accuracy': accuracy,
                'recent_predictions': recent
            }

            if self.enable_prometheus:
                self.agent_accuracy.labels(agent=agent).set(accuracy * 100)
                self.agent_confidence.labels(agent=agent).set(avg_confidence)

        if self.enable_prometheus and agent_name is None:
            self.agent_active_agents.set(len(agents_to_analyze))

        return metrics

    # Portfolio Metrics
    def update_portfolio_metrics(self, portfolio: Dict[str, Any]) -> None:
        """Update portfolio metrics.

        Args:
            portfolio: Portfolio dictionary with value, cash, positions, etc.
        """
        self.portfolio_value_history.append({
            **portfolio,
            'timestamp': datetime.utcnow().isoformat()
        })

        # Keep only last 10000 data points
        if len(self.portfolio_value_history) > 10000:
            self.portfolio_value_history = self.portfolio_value_history[-10000:]

        if self.enable_prometheus:
            self.portfolio_value.set(portfolio.get('total_value', 0))
            self.portfolio_cash.set(portfolio.get('cash', 0))
            self.portfolio_positions_value.set(portfolio.get('positions_value', 0))

            # Calculate returns for different periods
            for hours, gauge in [(1, self.portfolio_returns_1h),
                                 (24, self.portfolio_returns_24h),
                                 (168, self.portfolio_returns_7d)]:
                cutoff = datetime.utcnow() - timedelta(hours=hours)
                historical_values = [
                    p for p in self.portfolio_value_history
                    if datetime.fromisoformat(p['timestamp']) > cutoff
                ]
                if historical_values:
                    initial_value = historical_values[0].get('total_value', portfolio.get('total_value', 1))
                    current_value = portfolio.get('total_value', 0)
                    returns = ((current_value - initial_value) / initial_value) if initial_value > 0 else 0
                    gauge.set(returns)

    # Risk Metrics
    def calculate_risk_metrics(
        self,
        portfolio_value: float,
        positions: List[Dict[str, Any]],
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Any]:
        """Calculate risk metrics.

        Args:
            portfolio_value: Current portfolio value
            positions: List of position dictionaries
            confidence_levels: Confidence levels for VaR calculation

        Returns:
            Dictionary of risk metrics
        """
        # Calculate exposure
        total_exposure = sum(abs(p.get('value', 0)) for p in positions)

        # Calculate leverage
        cash = portfolio_value - total_exposure
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # Simple VaR calculation using historical simulation
        # In production, use proper risk models (Cornish-Fisher, Monte Carlo, etc.)
        returns = [
            t.get('pnl', 0) / portfolio_value
            for t in self.trade_history[-100:]  # Last 100 trades
            if portfolio_value > 0
        ]

        var_metrics = {}
        if returns:
            sorted_returns = sorted(returns)
            for confidence in confidence_levels:
                var_index = int((1 - confidence) * len(sorted_returns))
                var_value = abs(sorted_returns[var_index] * portfolio_value) if var_index < len(sorted_returns) else 0
                var_metrics[f'var_{int(confidence * 100)}'] = var_value

                if self.enable_prometheus:
                    if confidence == 0.95:
                        self.risk_var_95.set(var_value)
                    elif confidence == 0.99:
                        self.risk_var_99.set(var_value)

        # Average correlation (simplified - use proper correlation matrix in production)
        var_metrics.update({
            'exposure': total_exposure,
            'position_count': len(positions),
            'leverage': leverage,
            'avg_correlation': 0.0  # Placeholder - implement proper correlation
        })

        if self.enable_prometheus:
            self.risk_exposure_usd.set(total_exposure)
            self.risk_position_count.set(len(positions))
            self.risk_leverage.set(leverage)

        return var_metrics

    # Utility Methods
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dictionary of all metrics
        """
        return {
            'system': self.collect_system_metrics(),
            'trading': self.calculate_trading_metrics(),
            'exchanges': {
                exchange: self.get_exchange_metrics(exchange)
                for exchange in self.exchange_latencies.keys()
            },
            'agents': self.calculate_agent_metrics(),
            'uptime_seconds': time.time() - self.start_time
        }

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.trade_history.clear()
        self.exchange_latencies.clear()
        self.agent_predictions.clear()
        self.portfolio_value_history.clear()
        logger.info("Metrics reset")
