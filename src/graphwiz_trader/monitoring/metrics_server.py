#!/usr/bin/env python3
"""Simple HTTP server for health checks and basic metrics."""

import json
import time
from datetime import datetime
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import psutil


class MetricsHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for health and metrics endpoints."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self.send_health_response()
        elif self.path == '/metrics':
            self.send_metrics_response()
        else:
            self.send_error(404, 'Not Found')

    def send_health_response(self):
        """Send health check response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - start_time
        }
        
        self.wfile.write(json.dumps(health_data).encode())

    def send_metrics_response(self):
        """Send basic metrics response."""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        
        # Basic Prometheus-style metrics
        metrics = [
            f'# HELP trader_uptime_seconds Trading bot uptime in seconds',
            f'# TYPE trader_uptime_seconds counter',
            f'trader_uptime_seconds {time.time() - start_time}',
            '',
            f'# HELP process_cpu_percent CPU usage percentage',
            f'# TYPE process_cpu_percent gauge', 
            f'process_cpu_percent {psutil.cpu_percent()}',
            '',
            f'# HELP process_memory_bytes Memory usage in bytes',
            f'# TYPE process_memory_bytes gauge',
            f'process_memory_bytes {psutil.virtual_memory().used}',
            '',
            f'# HELP trader_status Trading bot status (1=running)',
            f'# TYPE trader_status gauge',
            f'trader_status 1'
        ]
        
        self.wfile.write('\n'.join(metrics).encode())

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


# Global state
start_time = time.time()
http_server = None


def start_metrics_server(port=8080):
    """Start the metrics HTTP server in a background thread."""
    global http_server
    
    def run_server():
        server = HTTPServer(('0.0.0.0', port), MetricsHandler)
        global http_server
        http_server = server
        server.serve_forever()
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print(f"Metrics server started on port {port}")


def stop_metrics_server():
    """Stop the metrics HTTP server."""
    global http_server
    if http_server:
        http_server.shutdown()
        print("Metrics server stopped")


if __name__ == '__main__':
    # Test server standalone
    start_metrics_server()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_metrics_server()