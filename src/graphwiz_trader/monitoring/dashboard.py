"""Dashboard configuration for Grafana and real-time monitoring."""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from loguru import logger

try:
    import aiohttp
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False
    logger.warning("Asyncio/aiohttp not available. Real-time updates disabled.")


class GrafanaDashboard:
    """Grafana dashboard configuration generator."""

    def __init__(self, dashboard_name: str = "Graphwiz Trader"):
        """Initialize Grafana dashboard generator.

        Args:
            dashboard_name: Name of the dashboard
        """
        self.dashboard_name = dashboard_name
        self.panels = []
        self.variables = []

    def add_variable(
        self,
        name: str,
        query: str,
        var_type: str = "query",
        label: Optional[str] = None
    ) -> None:
        """Add a dashboard variable.

        Args:
            name: Variable name
            query: Query string
            var_type: Variable type (query, interval, etc.)
            label: Display label
        """
        self.variables.append({
            "name": name,
            "type": var_type,
            "query": query,
            "label": label or name
        })

    def add_panel(
        self,
        title: str,
        targets: List[Dict[str, Any]],
        panel_type: str = "graph",
        grid_pos: Optional[Dict[str, int]] = None,
        description: Optional[str] = None
    ) -> None:
        """Add a panel to the dashboard.

        Args:
            title: Panel title
            targets: List of query targets
            panel_type: Type of panel (graph, stat, table, etc.)
            grid_pos: Position in grid (x, y, w, h)
            description: Panel description
        """
        if grid_pos is None:
            # Auto-position based on existing panels
            panel_count = len(self.panels)
            grid_pos = {
                "x": panel_count % 3 * 8,
                "y": (panel_count // 3) * 8,
                "w": 8,
                "h": 8
            }

        panel = {
            "title": title,
            "type": panel_type,
            "gridPos": grid_pos,
            "targets": targets,
            "description": description
        }

        self.panels.append(panel)

    def generate_dashboard_json(self) -> Dict[str, Any]:
        """Generate complete dashboard JSON.

        Returns:
            Dashboard JSON dictionary
        """
        dashboard = {
            "title": self.dashboard_name,
            "tags": ["trading", "graphwiz"],
            "timezone": "browser",
            "schemaVersion": 36,
            "version": 1,
            "refresh": "10s",
            "panels": self.panels,
            "templating": {
                "list": self._generate_variables_json()
            }
        }

        return {
            "dashboard": dashboard,
            "overwrite": True
        }

    def _generate_variables_json(self) -> List[Dict[str, Any]]:
        """Generate variables JSON.

        Returns:
            List of variable configurations
        """
        vars_json = []

        for var in self.variables:
            vars_json.append({
                "name": var["name"],
                "type": var["type"],
                "query": var["query"],
                "label": var["label"],
                "refresh": 1,
                "includeAll": True,
                "multi": False
            })

        return vars_json


class DashboardConfig:
    """Dashboard configuration manager."""

    @staticmethod
    def get_default_dashboard() -> Dict[str, Any]:
        """Get default Grafana dashboard configuration.

        Returns:
            Dashboard JSON
        """
        dashboard = GrafanaDashboard("Graphwiz Trading Dashboard")

        # Add variables
        dashboard.add_variable("exchange", "label_values(graphwiz_exchange_connected, exchange)")
        dashboard.add_variable("agent", "label_values(graphwiz_agent_accuracy, agent)")
        dashboard.add_variable("interval", "1m,5m,15m,1h", "interval")

        # System Metrics Row
        dashboard.add_panel(
            "CPU Usage",
            [{
                "expr": "rate(graphwiz_system_cpu_usage_percent[5m])",
                "legendFormat": "CPU %",
                "refId": "A"
            }],
            "graph",
            {"x": 0, "y": 0, "w": 8, "h": 8},
            "CPU usage percentage over time"
        )

        dashboard.add_panel(
            "Memory Usage",
            [{
                "expr": "graphwiz_system_memory_usage_bytes / 1024 / 1024 / 1024",
                "legendFormat": "Memory (GB)",
                "refId": "A"
            }],
            "graph",
            {"x": 8, "y": 0, "w": 8, "h": 8},
            "Memory usage in GB"
        )

        dashboard.add_panel(
            "Network I/O",
            [{
                "expr": "rate(graphwiz_system_network_sent_bytes[5m]) / 1024 / 1024",
                "legendFormat": "Sent (MB/s)",
                "refId": "A"
            }, {
                "expr": "rate(graphwiz_system_network_recv_bytes[5m]) / 1024 / 1024",
                "legendFormat": "Recv (MB/s)",
                "refId": "B"
            }],
            "graph",
            {"x": 16, "y": 0, "w": 8, "h": 8},
            "Network throughput"
        )

        # Trading Metrics Row
        dashboard.add_panel(
            "Total P&L",
            [{
                "expr": "graphwiz_trading_pnl_usd",
                "legendFormat": "Total P&L",
                "refId": "A"
            }],
            "stat",
            {"x": 0, "y": 8, "w": 6, "h": 6},
            "Total profit and loss in USD"
        )

        dashboard.add_panel(
            "Daily P&L",
            [{
                "expr": "graphwiz_trading_pnl_daily_usd",
                "legendFormat": "Daily P&L",
                "refId": "A"
            }],
            "stat",
            {"x": 6, "y": 8, "w": 6, "h": 6},
            "Today's profit and loss"
        )

        dashboard.add_panel(
            "Win Rate",
            [{
                "expr": "graphwiz_trading_win_rate",
                "legendFormat": "Win Rate %",
                "refId": "A"
            }],
            "stat",
            {"x": 12, "y": 8, "w": 6, "h": 6},
            "Win rate percentage"
        )

        dashboard.add_panel(
            "Sharpe Ratio",
            [{
                "expr": "graphwiz_trading_sharpe_ratio",
                "legendFormat": "Sharpe",
                "refId": "A"
            }],
            "stat",
            {"x": 18, "y": 8, "w": 6, "h": 6},
            "Sharpe ratio"
        )

        dashboard.add_panel(
            "Drawdown",
            [{
                "expr": "graphwiz_trading_current_drawdown",
                "legendFormat": "Current Drawdown",
                "refId": "A"
            }, {
                "expr": "graphwiz_trading_max_drawdown",
                "legendFormat": "Max Drawdown",
                "refId": "B"
            }],
            "graph",
            {"x": 0, "y": 14, "w": 12, "h": 8},
            "Portfolio drawdown over time"
        )

        dashboard.add_panel(
            "Trade Count",
            [{
                "expr": "rate(graphwiz_trading_total_trades[5m]) * 60",
                "legendFormat": "Trades/min",
                "refId": "A"
            }],
            "graph",
            {"x": 12, "y": 14, "w": 12, "h": 8},
            "Trade rate per minute"
        )

        # Exchange Metrics Row
        dashboard.add_panel(
            "Exchange Latency",
            [{
                "expr": "histogram_quantile(0.99, rate(graphwiz_exchange_latency_seconds_bucket[5m]))",
                "legendFormat": "{{exchange}} p99",
                "refId": "A"
            }, {
                "expr": "histogram_quantile(0.95, rate(graphwiz_exchange_latency_seconds_bucket[5m]))",
                "legendFormat": "{{exchange}} p95",
                "refId": "B"
            }],
            "graph",
            {"x": 0, "y": 22, "w": 12, "h": 8},
            "Exchange API latency (p95, p99)"
        )

        dashboard.add_panel(
            "Exchange Connection Status",
            [{
                "expr": "graphviz_exchange_connected",
                "legendFormat": "{{exchange}}",
                "refId": "A"
            }],
            "stat",
            {"x": 12, "y": 22, "w": 12, "h": 8},
            "Exchange connection status (1=connected, 0=disconnected)"
        )

        # Agent Metrics Row
        dashboard.add_panel(
            "Agent Accuracy",
            [{
                "expr": "graphwiz_agent_accuracy",
                "legendFormat": "{{agent}}",
                "refId": "A"
            }],
            "graph",
            {"x": 0, "y": 30, "w": 12, "h": 8},
            "Agent prediction accuracy over time"
        )

        dashboard.add_panel(
            "Agent Confidence",
            [{
                "expr": "avg(graphwiz_agent_confidence) by (agent)",
                "legendFormat": "{{agent}}",
                "refId": "A"
            }],
            "graph",
            {"x": 12, "y": 30, "w": 12, "h": 8},
            "Average agent confidence levels"
        )

        # Risk Metrics Row
        dashboard.add_panel(
            "Value at Risk",
            [{
                "expr": "graphwiz_risk_var_95_usd",
                "legendFormat": "VaR 95%",
                "refId": "A"
            }, {
                "expr": "graphwiz_risk_var_99_usd",
                "legendFormat": "VaR 99%",
                "refId": "B"
            }],
            "graph",
            {"x": 0, "y": 38, "w": 12, "h": 8},
            "Value at Risk at 95% and 99% confidence"
        )

        dashboard.add_panel(
            "Portfolio Exposure",
            [{
                "expr": "graphwiz_risk_exposure_usd",
                "legendFormat": "Exposure",
                "refId": "A"
            }, {
                "expr": "graphwiz_risk_leverage",
                "legendFormat": "Leverage",
                "refId": "B"
            }],
            "graph",
            {"x": 12, "y": 38, "w": 12, "h": 8},
            "Portfolio exposure and leverage"
        )

        dashboard.add_panel(
            "Portfolio Value",
            [{
                "expr": "graphwiz_portfolio_value_usd",
                "legendFormat": "Portfolio Value",
                "refId": "A"
            }, {
                "expr": "graphwiz_portfolio_cash_usd",
                "legendFormat": "Cash",
                "refId": "B"
            }, {
                "expr": "graphwiz_portfolio_positions_value_usd",
                "legendFormat": "Positions",
                "refId": "C"
            }],
            "graph",
            {"x": 0, "y": 46, "w": 12, "h": 8},
            "Portfolio value breakdown"
        )

        dashboard.add_panel(
            "Portfolio Returns",
            [{
                "expr": "graphwiz_portfolio_returns_1h * 100",
                "legendFormat": "1h %",
                "refId": "A"
            }, {
                "expr": "graphwiz_portfolio_returns_24h * 100",
                "legendFormat": "24h %",
                "refId": "B"
            }, {
                "expr": "graphwiz_portfolio_returns_7d * 100",
                "legendFormat": "7d %",
                "refId": "C"
            }],
            "graph",
            {"x": 12, "y": 46, "w": 12, "h": 8},
            "Portfolio returns at different time horizons"
        )

        return dashboard.generate_dashboard_json()

    @staticmethod
    def get_alerts_dashboard() -> Dict[str, Any]:
        """Get alerts dashboard configuration.

        Returns:
            Dashboard JSON
        """
        dashboard = GrafanaDashboard("Graphwiz Alerts Dashboard")

        # Alert Statistics
        dashboard.add_panel(
            "Active Alerts",
            [{
                "expr": "count(graphwiz_alert_active == 1)",
                "legendFormat": "Active",
                "refId": "A"
            }],
            "stat",
            {"x": 0, "y": 0, "w": 6, "h": 6},
            "Number of active alerts"
        )

        dashboard.add_panel(
            "Alerts Last 24h",
            [{
                "expr": "increase(graphwiz_alert_total[24h])",
                "legendFormat": "Total",
                "refId": "A"
            }],
            "stat",
            {"x": 6, "y": 0, "w": 6, "h": 6},
            "Total alerts in last 24 hours"
        )

        dashboard.add_panel(
            "Alerts by Severity",
            [{
                "expr": "count by (severity) (graphwiz_alert_active == 1)",
                "legendFormat": "{{severity}}",
                "refId": "A"
            }],
            "piechart",
            {"x": 12, "y": 0, "w": 12, "h": 6},
            "Active alerts by severity level"
        )

        # Alert Timeline
        dashboard.add_panel(
            "Alert Timeline",
            [{
                "expr": "rate(graphwiz_alert_total[5m]) * 60",
                "legendFormat": "{{severity}}",
                "refId": "A"
            }],
            "graph",
            {"x": 0, "y": 6, "w": 24, "h": 8},
            "Alert rate over time"
        )

        return dashboard.generate_dashboard_json()


class RealTimeMonitor:
    """Real-time monitoring with WebSocket updates."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize real-time monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.websocket_config = config.get('websocket', {})
        self.clients = set()
        self.running = False

    async def start(self) -> None:
        """Start WebSocket server for real-time updates."""
        if not ASYNCIO_AVAILABLE or not WEBSOCKETS_AVAILABLE:
            logger.warning("Real-time monitoring disabled (websockets not available)")
            return

        self.running = True
        host = self.websocket_config.get('host', '0.0.0.0')
        port = self.websocket_config.get('port', 8765)

        logger.info("Starting WebSocket server on {}:{}", host, port)

        async def handler(websocket, path):
            """Handle WebSocket connection."""
            self.clients.add(websocket)
            logger.info("WebSocket client connected. Total clients: {}", len(self.clients))

            try:
                async for message in websocket:
                    # Handle incoming messages if needed
                    pass
            except Exception as e:
                logger.error("WebSocket error: {}", e)
            finally:
                self.clients.remove(websocket)
                logger.info("WebSocket client disconnected. Total clients: {}", len(self.clients))

        # Start WebSocket server
        import websockets
        server = await websockets.serve(handler, host, port)
        logger.info("WebSocket server started on ws://{}:{}", host, port)

        # Keep server running
        await server.wait_closed()

    async def broadcast_update(self, update: Dict[str, Any]) -> None:
        """Broadcast update to all connected clients.

        Args:
            update: Update dictionary to broadcast
        """
        if not self.clients:
            return

        message = json.dumps(update)

        # Create a list of disconnected clients to remove
        disconnected = set()

        for client in self.clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.warning("Failed to send to client: {}", e)
                disconnected.add(client)

        # Remove disconnected clients
        self.clients -= disconnected

    async def broadcast_metrics(self, metrics: Dict[str, Any]) -> None:
        """Broadcast metrics update.

        Args:
            metrics: Metrics dictionary
        """
        await self.broadcast_update({
            'type': 'metrics',
            'data': metrics,
            'timestamp': datetime.utcnow().isoformat()
        })

    async def broadcast_alert(self, alert: Dict[str, Any]) -> None:
        """Broadcast alert update.

        Args:
            alert: Alert dictionary
        """
        await self.broadcast_update({
            'type': 'alert',
            'data': alert,
            'timestamp': datetime.utcnow().isoformat()
        })

    async def broadcast_health(self, health: Dict[str, Any]) -> None:
        """Broadcast health check update.

        Args:
            health: Health check results
        """
        await self.broadcast_update({
            'type': 'health',
            'data': health,
            'timestamp': datetime.utcnow().isoformat()
        })

    def stop(self) -> None:
        """Stop real-time monitor."""
        self.running = False
        logger.info("Real-time monitor stopped")


class PrometheusExporter:
    """Helper for Prometheus metric queries."""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """Initialize Prometheus exporter.

        Args:
            prometheus_url: Prometheus server URL
        """
        self.prometheus_url = prometheus_url
        self.query_endpoint = f"{prometheus_url}/api/v1/query"
        self.query_range_endpoint = f"{prometheus_url}/api/v1/query_range"

    async def query(self, query: str, time: Optional[datetime] = None) -> Dict[str, Any]:
        """Execute PromQL query.

        Args:
            query: PromQL query string
            time: Query time (defaults to now)

        Returns:
            Query results dictionary
        """
        if not ASYNCIO_AVAILABLE:
            return {}

        params = {"query": query}
        if time:
            params["time"] = time.timestamp()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.query_endpoint, params=params) as response:
                    data = await response.json()
                    if data.get("status") == "success":
                        return data.get("data", {})
                    else:
                        logger.error("Prometheus query failed: {}", data.get("error", "Unknown error"))
                        return {}
        except Exception as e:
            logger.error("Failed to query Prometheus: {}", e)
            return {}

    async def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "15s"
    ) -> Dict[str, Any]:
        """Execute PromQL range query.

        Args:
            query: PromQL query string
            start: Start time
            end: End time
            step: Query step interval

        Returns:
            Query results dictionary
        """
        if not ASYNCIO_AVAILABLE:
            return {}

        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.query_range_endpoint, params=params) as response:
                    data = await response.json()
                    if data.get("status") == "success":
                        return data.get("data", {})
                    else:
                        logger.error("Prometheus range query failed: {}", data.get("error", "Unknown error"))
                        return {}
        except Exception as e:
            logger.error("Failed to query Prometheus range: {}", e)
            return {}

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of key metrics.

        Returns:
            Dictionary with metrics summary
        """
        queries = {
            "total_trades": "graphwiz_trading_total_trades",
            "total_pnl": "graphwiz_trading_pnl_usd",
            "win_rate": "graphwiz_trading_win_rate",
            "portfolio_value": "graphwiz_portfolio_value_usd",
            "cpu_usage": "graphwiz_system_cpu_usage_percent",
            "memory_usage": "graphwiz_system_memory_usage_bytes"
        }

        summary = {}

        for key, query in queries.items():
            result = await self.query(query)
            if result and result.get("result"):
                value = result["result"][0].get("value", [0, "0"])[1]
                try:
                    summary[key] = float(value)
                except (ValueError, TypeError):
                    summary[key] = 0

        return summary


def export_dashboard_to_file(dashboard: Dict[str, Any], filepath: str) -> None:
    """Export dashboard JSON to file.

    Args:
        dashboard: Dashboard JSON dictionary
        filepath: Path to save dashboard
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(dashboard, f, indent=2)
        logger.info("Dashboard exported to {}", filepath)
    except Exception as e:
        logger.error("Failed to export dashboard: {}", e)


def load_dashboard_from_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load dashboard JSON from file.

    Args:
        filepath: Path to dashboard file

    Returns:
        Dashboard JSON dictionary or None
    """
    try:
        with open(filepath, 'r') as f:
            dashboard = json.load(f)
        logger.info("Dashboard loaded from {}", filepath)
        return dashboard
    except Exception as e:
        logger.error("Failed to load dashboard: {}", e)
        return None
