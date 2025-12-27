"""
Real-Time Trading Performance Dashboard

Web-based dashboard for monitoring paper trading performance with live updates,
equity curves, grid visualization, and trade history.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from loguru import logger

try:
    from flask import Flask, render_template, jsonify
    from flask_sock import Sock
except ImportError:
    logger.error("Flask or Flask-Sock not installed. Install with: pip install flask flask-sock")
    sys.exit(1)

import pandas as pd

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

app = Flask(__name__)
sock = Sock(app)

# Global state for dashboard
dashboard_state = {
    "strategies": {},
    "equity_history": [],
    "trades": [],
    "last_update": None,
}


@dataclass
class StrategyState:
    """State of a trading strategy for dashboard display."""
    symbol: str
    strategy_type: str
    current_price: float
    portfolio_value: float
    capital: float
    position: float
    position_value: float
    pnl: float
    roi_pct: float
    grid_upper: float
    grid_lower: float
    grid_levels: List[float]
    trades_count: int
    last_update: str

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def create_html_template():
    """Create the HTML template for the dashboard."""
    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .last-update {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .metric-card .value.positive { color: #10b981; }
        .metric-card .value.negative { color: #ef4444; }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .chart-card h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.2em;
        }
        .trades-table {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow-x: auto;
        }
        .trades-table h2 {
            color: #667eea;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        th {
            background: #f9fafb;
            font-weight: 600;
            color: #667eea;
        }
        tr:hover {
            background: #f9fafb;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .status-badge.buy {
            background: #d1fae5;
            color: #065f46;
        }
        .status-badge.sell {
            background: #fee2e2;
            color: #991b1b;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Trading Performance Dashboard</h1>
            <p class="last-update">Last Update: <span id="lastUpdate">Loading...</span></p>
        </div>

        <div class="metrics-grid" id="metrics">
            <!-- Metrics will be populated dynamically -->
        </div>

        <div class="charts-grid">
            <div class="chart-card">
                <h2>💰 Portfolio Equity Curve</h2>
                <canvas id="equityChart"></canvas>
            </div>
            <div class="chart-card">
                <h2>📊 Grid Levels & Current Price</h2>
                <canvas id="gridChart"></canvas>
            </div>
        </div>

        <div class="trades-table">
            <h2>📋 Recent Trades</h2>
            <table id="tradesTable">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>Amount</th>
                        <th>Profit</th>
                    </tr>
                </thead>
                <tbody id="tradesBody">
                    <!-- Trades will be populated dynamically -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let equityChart = null;
        let gridChart = null;

        function initCharts() {
            const equityCtx = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(equityCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: value => '$' + value.toLocaleString()
                            }
                        }
                    }
                }
            });

            const gridCtx = document.getElementById('gridChart').getContext('2d');
            gridChart = new Chart(gridCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Grid Levels',
                        data: [],
                        backgroundColor: 'rgba(102, 126, 234, 0.5)',
                        borderColor: '#667eea',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            ticks: {
                                callback: value => '$' + value.toLocaleString()
                            }
                        }
                    }
                }
            });
        }

        function updateDashboard(data) {
            // Update last update time
            document.getElementById('lastUpdate').textContent = new Date().toLocaleString();

            // Update metrics
            const metricsHtml = Object.entries(data.strategies).map(([symbol, strategy]) => `
                <div class="metric-card">
                    <h3>${symbol} - ${strategy.strategy_type}</h3>
                    <div class="value ${strategy.pnl >= 0 ? 'positive' : 'negative'}">
                        $${strategy.portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                    </div>
                    <div style="margin-top: 10px; font-size: 0.9em;">
                        <div>P&L: <span class="${strategy.pnl >= 0 ? 'positive' : 'negative'}">$${strategy.pnl.toFixed(2)} (${strategy.roi_pct.toFixed(2)}%)</span></div>
                        <div>Price: $${strategy.current_price.toLocaleString()}</div>
                        <div>Trades: ${strategy.trades_count}</div>
                    </div>
                </div>
            `).join('');
            document.getElementById('metrics').innerHTML = metricsHtml;

            // Update equity chart
            if (data.equity_history && data.equity_history.length > 0) {
                equityChart.data.labels = data.equity_history.map((_, i) => i + 1);
                equityChart.data.datasets[0].data = data.equity_history.map(e => e.total_value || e.portfolio_value);
                equityChart.update();
            }

            // Update grid chart
            if (data.strategies && Object.keys(data.strategies).length > 0) {
                const firstStrategy = Object.values(data.strategies)[0];
                gridChart.data.labels = firstStrategy.grid_levels.map((_, i) => `L${i + 1}`);
                gridChart.data.datasets[0].data = firstStrategy.grid_levels;
                gridChart.update();
            }

            // Update trades table
            if (data.trades && data.trades.length > 0) {
                const recentTrades = data.trades.slice(-20).reverse();
                const tradesHtml = recentTrades.map(trade => `
                    <tr>
                        <td>${new Date(trade.timestamp).toLocaleString()}</td>
                        <td>${trade.symbol}</td>
                        <td><span class="status-badge ${trade.side}">${trade.side.toUpperCase()}</span></td>
                        <td>$${trade.price.toFixed(2)}</td>
                        <td>${trade.amount.toFixed(6)}</td>
                        <td>${trade.profit ? '$' + trade.profit.toFixed(2) : '-'}</td>
                    </tr>
                `).join('');
                document.getElementById('tradesBody').innerHTML = tradesHtml;
            }
        }

        // Initialize WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        ws.onopen = () => {
            console.log('WebSocket connected');
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected, attempting to reconnect...');
            setTimeout(() => location.reload(), 3000);
        };

        // Initialize charts on load
        window.onload = initCharts;
    </script>
</body>
</html>"""
    return template


@app.route("/")
def index():
    """Render dashboard homepage."""
    return render_template_string(create_html_template())


@app.route("/api/state")
def get_state():
    """Return current dashboard state as JSON."""
    return jsonify(dashboard_state)


@sock.route("/ws")
def websocket_connection(ws):
    """WebSocket endpoint for real-time updates."""
    logger.info("Dashboard client connected")
    try:
        while True:
            # Send current state
            ws.send(json.dumps(dashboard_state))
            # Wait for new data or heartbeat
            ws.receive(timeout=1)
    except Exception as e:
        logger.info(f"Dashboard client disconnected: {e}")


def update_strategy_state(symbol: str, state: StrategyState):
    """Update state for a strategy."""
    dashboard_state["strategies"][symbol] = state.to_dict()
    dashboard_state["last_update"] = datetime.now().isoformat()


def update_equity_history(equity_data: List[Dict]):
    """Update equity history."""
    dashboard_state["equity_history"] = equity_data


def update_trades(trades: List[Dict]):
    """Update trades list."""
    dashboard_state["trades"] = trades[-100:]  # Keep last 100 trades


def run_dashboard(host="0.0.0.0", port=5000, debug=False):
    """Run the dashboard server."""
    logger.info("=" * 80)
    logger.info("🚀 Starting Trading Performance Dashboard")
    logger.info("=" * 80)
    logger.info(f"URL: http://{host}:{port}")
    logger.info(f"WebSocket: ws://{host}:{port}/ws")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 80 + "\n")

    app.run(host=host, port=port, debug=debug)


# Example usage and integration helper
class DashboardIntegrator:
    """
    Helper class to integrate dashboard with trading strategies.

    Usage:
        dashboard_integrator = DashboardIntegrator()
        dashboard_integrator.start()

        # In your trading loop:
        dashboard_integrator.update(strategy_state, equity_history, trades)
    """

    def __init__(self):
        self.integrator_thread = None

    def start(self, host="0.0.0.0", port=5000):
        """Start dashboard in a separate thread."""
        import threading
        self.integrator_thread = threading.Thread(
            target=run_dashboard,
            kwargs={"host": host, "port": port, "debug": False},
            daemon=True
        )
        self.integrator_thread.start()
        logger.success(f"✅ Dashboard started at http://{host}:{port}")

    def update(
        self,
        strategies: Dict[str, StrategyState],
        equity_history: List[Dict],
        trades: List[Dict],
    ):
        """Update dashboard with latest data."""
        for symbol, state in strategies.items():
            update_strategy_state(symbol, state)

        update_equity_history(equity_history)
        update_trades(trades)


def main():
    """Main entry point for standalone dashboard."""
    # Create sample data for demonstration
    from graphwiz_trader.strategies import GridTradingStrategy, GridTradingMode

    # Sample strategy state
    sample_state = StrategyState(
        symbol="BTC/USDT",
        strategy_type="Grid Trading",
        current_price=87588.00,
        portfolio_value=10930.76,
        capital=9000.00,
        position=0.022044,
        position_value=1930.76,
        pnl=930.76,
        roi_pct=9.31,
        grid_upper=100726.19,
        grid_lower=74449.79,
        grid_levels=list(range(74000, 101000, 2500)),
        trades_count=11,
        last_update=datetime.now().isoformat(),
    )

    update_strategy_state("BTC/USDT", sample_state)

    # Sample equity history
    equity_history = [
        {"timestamp": "2024-01-01T00:00:00", "total_value": 10000},
        {"timestamp": "2024-01-01T01:00:00", "total_value": 10100},
        {"timestamp": "2024-01-01T02:00:00", "total_value": 10200},
        {"timestamp": "2024-01-01T03:00:00", "total_value": 10930.76},
    ]
    update_equity_history(equity_history)

    # Run dashboard
    run_dashboard()


if __name__ == "__main__":
    main()
