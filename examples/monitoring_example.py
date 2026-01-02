"""Example usage of the monitoring system."""

import asyncio
from loguru import logger

from graphwiz_trader.monitoring import (
    create_monitor,
    AlertSeverity,
    AlertChannel
)


async def main():
    """Demonstrate monitoring system usage."""

    # Load configuration
    config = {
        'monitoring': {
            'prometheus_port': 8000,
            'enable_prometheus': True,
            'metrics_interval_seconds': 15,
            'health_check_interval_seconds': 30,
            'enable_realtime': True
        },
        'notifications': {
            'discord': {
                'webhook_url': 'https://discord.com/api/webhooks/YOUR_WEBHOOK'
            },
            'email': {
                'enabled': False
            }
        },
        'alerts': {
            'cooldown_warning': 300,
            'cooldown_critical': 60
        },
        'health_checks': {
            'exchange_connectivity': {'enabled': True},
            'neo4j_connectivity': {'enabled': True},
            'system_resources': {'enabled': True}
        }
    }

    # Create monitor
    monitor = create_monitor(config)

    # Set up event callbacks
    async def on_alert(alert):
        logger.warning("Alert received: {} - {}", alert.severity.value, alert.title)

    async def on_health_change(health_summary):
        logger.info("Health status: {}", health_summary['overall_status'])

    monitor.on_alert = on_alert
    monitor.on_health_change = on_health_change

    # Start monitoring
    await monitor.start()

    try:
        # Simulate trading activity
        logger.info("Simulating trading activity...")

        # Record some trades
        trades = [
            {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 50000,
                'quantity': 0.1,
                'pnl': 150.50,
                'success': True,
                'execution_time': 0.125
            },
            {
                'symbol': 'ETH/USDT',
                'side': 'sell',
                'price': 3000,
                'quantity': 1.0,
                'pnl': -25.30,
                'success': True,
                'execution_time': 0.089
            },
            {
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'price': 51000,
                'quantity': 0.1,
                'pnl': 95.50,
                'success': True,
                'execution_time': 0.156
            }
        ]

        for trade in trades:
            monitor.record_trade(trade)
            await asyncio.sleep(1)

        # Record agent predictions
        monitor.record_agent_prediction(
            'momentum_agent',
            {
                'prediction': 1,  # Buy signal
                'confidence': 0.85,
                'symbol': 'BTC/USDT'
            }
        )

        monitor.record_agent_prediction(
            'mean_reversion_agent',
            {
                'prediction': -1,  # Sell signal
                'confidence': 0.72,
                'symbol': 'ETH/USDT'
            }
        )

        # Wait and check metrics
        await asyncio.sleep(5)

        # Get current metrics
        metrics = monitor.get_metrics()
        logger.info("Current metrics: {}", metrics)

        # Get health status
        health = monitor.get_health_status()
        logger.info("Health status: {}", health)

        # Get active alerts
        active_alerts = monitor.get_active_alerts()
        logger.info("Active alerts: {}", len(active_alerts))

        # Create manual alert
        monitor.create_manual_alert(
            title="Test Alert",
            message="This is a test alert from the monitoring system",
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.LOG],
            metadata={'test': True}
        )

        # Keep running for a bit to see monitoring in action
        logger.info("Monitoring running... Press Ctrl+C to stop")
        await asyncio.sleep(30)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
