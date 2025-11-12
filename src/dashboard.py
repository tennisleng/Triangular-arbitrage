"""
Real-time arbitrage dashboard with WebSocket support.
Provides live visualization of arbitrage opportunities and trading activity.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
import plotly.utils
from collections import deque
import logging


class ArbitrageDashboard:
    """Real-time dashboard for arbitrage monitoring."""

    def __init__(self, exchange_manager, arbitrage_algorithms):
        self.exchange_manager = exchange_manager
        self.arbitrage_algorithms = arbitrage_algorithms
        self.logger = logging.getLogger(__name__)

        # Dashboard data
        self.opportunity_history = deque(maxlen=1000)
        self.price_history = {}
        self.active_opportunities = []
        self.performance_metrics = {
            'total_scanned': 0,
            'opportunities_found': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'total_profit_usd': 0.0
        }

        # Flask app setup
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Setup routes
        self._setup_routes()

        # Background threads
        self.monitoring_thread = None
        self.is_monitoring = False

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            return render_template_string(self._get_html_template())

        @self.app.route('/api/stats')
        def get_stats():
            return self._get_stats_json()

        @self.app.route('/api/opportunities')
        def get_opportunities():
            return self._get_opportunities_json()

        @self.app.route('/api/performance')
        def get_performance():
            return self._get_performance_json()

        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client connected to dashboard")
            emit('initial_data', self._get_initial_data())

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected from dashboard")

    def _get_html_template(self) -> str:
        """Get HTML template for the dashboard."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Triangular Arbitrage Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .opportunities-list { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-height: 400px; overflow-y: auto; }
        .opportunity-item { padding: 10px; border-bottom: 1px solid #ecf0f1; display: flex; justify-content: space-between; }
        .profit-positive { color: #27ae60; }
        .profit-negative { color: #e74c3c; }
        .status-active { background: #d4edda; border-left: 4px solid #27ae60; }
        .status-inactive { background: #f8d7da; border-left: 4px solid #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Triangular Arbitrage Dashboard</h1>
            <p>Real-time arbitrage opportunity monitoring</p>
        </div>

        <div class="stats-grid" id="stats-grid">
            <!-- Stats will be populated by JavaScript -->
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <h3>Profit Over Time</h3>
                <div id="profit-chart"></div>
            </div>
            <div class="chart-container">
                <h3>Opportunity Frequency</h3>
                <div id="opportunity-chart"></div>
            </div>
        </div>

        <div class="opportunities-list">
            <h3>Recent Opportunities</h3>
            <div id="opportunities-list">
                <!-- Opportunities will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let profitChart, opportunityChart;

        socket.on('initial_data', function(data) {
            updateStats(data.stats);
            updateCharts(data);
            updateOpportunities(data.opportunities);
        });

        socket.on('stats_update', function(stats) {
            updateStats(stats);
        });

        socket.on('opportunities_update', function(opportunities) {
            updateOpportunities(opportunities);
        });

        socket.on('charts_update', function(data) {
            updateCharts(data);
        });

        function updateStats(stats) {
            const statsGrid = document.getElementById('stats-grid');
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.total_scanned || 0}</div>
                    <div class="stat-label">Total Scanned</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.opportunities_found || 0}</div>
                    <div class="stat-label">Opportunities Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.trades_executed || 0}</div>
                    <div class="stat-label">Trades Executed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">$${stats.total_profit_usd || 0}</div>
                    <div class="stat-label">Total Profit</div>
                </div>
            `;
        }

        function updateCharts(data) {
            // Profit chart
            if (!profitChart) {
                profitChart = Plotly.newPlot('profit-chart', [{
                    x: data.profit_timestamps || [],
                    y: data.profit_values || [],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Profit (USD)',
                    line: {color: '#27ae60'}
                }], {
                    margin: {t: 20, b: 40, l: 60, r: 20},
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'Profit (USD)'}
                });
            } else {
                Plotly.update('profit-chart', {
                    x: [data.profit_timestamps],
                    y: [data.profit_values]
                });
            }

            // Opportunity chart
            if (!opportunityChart) {
                opportunityChart = Plotly.newPlot('opportunity-chart', [{
                    x: data.opportunity_timestamps || [],
                    y: data.opportunity_counts || [],
                    type: 'bar',
                    name: 'Opportunities',
                    marker: {color: '#3498db'}
                }], {
                    margin: {t: 20, b: 40, l: 60, r: 20},
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'Count'}
                });
            } else {
                Plotly.update('opportunity-chart', {
                    x: [data.opportunity_timestamps],
                    y: [data.opportunity_counts]
                });
            }
        }

        function updateOpportunities(opportunities) {
            const list = document.getElementById('opportunities-list');
            list.innerHTML = opportunities.slice(0, 20).map(opp => `
                <div class="opportunity-item ${opp.active ? 'status-active' : 'status-inactive'}">
                    <div>
                        <strong>${opp.exchange} - ${opp.base_currency}/${opp.alt_currency}/${opp.quote_currency}</strong><br>
                        <small>${opp.direction} | ${opp.profit_percentage.toFixed(4)}% | $${opp.profit_usd.toFixed(2)}</small>
                    </div>
                    <div class="profit-${opp.profit_percentage > 0 ? 'positive' : 'negative'}">
                        ${opp.profit_percentage > 0 ? '+' : ''}${opp.profit_percentage.toFixed(4)}%
                    </div>
                </div>
            `).join('');
        }

        // Refresh data every 5 seconds
        setInterval(() => {
            fetch('/api/stats')
                .then(r => r.json())
                .then(updateStats);

            fetch('/api/opportunities')
                .then(r => r.json())
                .then(data => updateOpportunities(data.opportunities || []));
        }, 5000);
    </script>
</body>
</html>
        """

    def _get_initial_data(self) -> Dict[str, Any]:
        """Get initial data for dashboard."""
        return {
            'stats': self.performance_metrics,
            'opportunities': [self._opportunity_to_dict(opp) for opp in list(self.opportunity_history)[-20:]],
            'profit_timestamps': [],
            'profit_values': [],
            'opportunity_timestamps': [],
            'opportunity_counts': []
        }

    def _get_stats_json(self) -> str:
        """Get performance statistics as JSON."""
        return json.dumps(self.performance_metrics)

    def _get_opportunities_json(self) -> str:
        """Get recent opportunities as JSON."""
        opportunities = [self._opportunity_to_dict(opp) for opp in list(self.opportunity_history)[-50:]]
        return json.dumps({'opportunities': opportunities})

    def _get_performance_json(self) -> str:
        """Get performance metrics as JSON."""
        stats = self.arbitrage_algorithms.get_arbitrage_statistics()
        stats.update(self.performance_metrics)
        return json.dumps(stats)

    def _opportunity_to_dict(self, opportunity) -> Dict[str, Any]:
        """Convert ArbitrageOpportunity to dictionary."""
        return {
            'exchange': opportunity.exchange,
            'base_currency': opportunity.base_currency,
            'quote_currency': opportunity.quote_currency,
            'alt_currency': opportunity.alt_currency,
            'direction': opportunity.direction,
            'profit_percentage': opportunity.profit_percentage,
            'profit_usd': opportunity.profit_usd,
            'path': opportunity.path,
            'timestamp': opportunity.timestamp.isoformat(),
            'active': opportunity in self.active_opportunities
        }

    def add_opportunity(self, opportunity):
        """Add new arbitrage opportunity to dashboard."""
        self.opportunity_history.append(opportunity)
        self.performance_metrics['opportunities_found'] += 1

        # Emit to connected clients
        if self.socketio:
            self.socketio.emit('opportunities_update',
                             [self._opportunity_to_dict(opp) for opp in list(self.opportunity_history)[-20:]])

    def record_trade(self, opportunity, success: bool, profit_usd: float = 0.0):
        """Record trade execution."""
        self.performance_metrics['trades_executed'] += 1

        if success:
            self.performance_metrics['successful_trades'] += 1
            self.performance_metrics['total_profit_usd'] += profit_usd

        # Emit stats update
        if self.socketio:
            self.socketio.emit('stats_update', self.performance_metrics)

    def update_price_data(self, exchange: str, symbol: str, price: float):
        """Update price data for charts."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)

        self.price_history[symbol].append({
            'timestamp': datetime.now(),
            'price': price,
            'exchange': exchange
        })

    def start_monitoring(self, host: str = '0.0.0.0', port: int = 5001):
        """Start the dashboard monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._run_dashboard,
            args=(host, port),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Dashboard started on http://{host}:{port}")

    def stop_monitoring(self):
        """Stop the dashboard monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    def _run_dashboard(self, host: str, port: int):
        """Run the Flask dashboard server."""
        try:
            self.socketio.run(self.app, host=host, port=port, debug=False)
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary for external access."""
        return {
            'performance_metrics': self.performance_metrics,
            'active_opportunities_count': len(self.active_opportunities),
            'total_opportunities_tracked': len(self.opportunity_history),
            'exchanges_monitored': list(self.exchange_manager.get_enabled_exchanges()),
            'uptime': str(datetime.now() - datetime.fromtimestamp(time.time())),
        }
