"""
HFT Performance Monitoring.

Real-time performance monitoring for HFT strategies and system health.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import psutil
from loguru import logger


class PerformanceMonitor:
    """Monitor system and strategy performance in real-time."""

    def __init__(self, knowledge_graph: Any, interval: int = 5) -> None:
        """
        Initialize performance monitor.

        Args:
            knowledge_graph: Neo4j knowledge graph instance
            interval: Monitoring interval in seconds
        """
        self.kg = knowledge_graph
        self.interval = interval
        self.running = False
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = 1000  # Keep last 1000 data points

        # Latency tracking
        self.latencies: Dict[str, List[float]] = {
            "order_execution": [],
            "market_data": [],
            "strategy_signal": [],
        }
        self.max_latency_samples = 10000

        # Performance thresholds
        self.thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 90.0,
            "order_latency_ms": 10.0,
            "market_data_latency_ms": 1.0,
            "strategy_latency_ms": 5.0,
        }

    async def start(self) -> None:
        """Start performance monitoring."""
        self.running = True
        logger.info("Performance monitoring started")
        asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop performance monitoring."""
        self.running = False
        logger.info("Performance monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.interval)

    async def _collect_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)

            # Disk metrics (optional)
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            # Network metrics (optional)
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024**2)
            network_recv_mb = network.bytes_recv / (1024**2)

            # Latency metrics
            latency_metrics = self.get_latency_metrics()

            # Build metrics dictionary
            metrics = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory_used_gb,
                "disk_percent": disk_percent,
                "network_sent_mb": network_sent_mb,
                "network_recv_mb": network_recv_mb,
                **latency_metrics,
            }

            # Add to history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)

            # Check thresholds
            await self._check_thresholds(metrics)

            # Store in knowledge graph
            await self._store_metrics(metrics)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def _check_thresholds(self, metrics: Dict[str, Any]) -> None:
        """
        Check if metrics exceed thresholds.

        Args:
            metrics: Current metrics
        """
        # CPU check
        if metrics["cpu_percent"] > self.thresholds["cpu_percent"]:
            logger.warning(
                f"High CPU usage: {metrics['cpu_percent']:.1f}% "
                f"(threshold: {self.thresholds['cpu_percent']}%)"
            )

        # Memory check
        if metrics["memory_percent"] > self.thresholds["memory_percent"]:
            logger.warning(
                f"High memory usage: {metrics['memory_percent']:.1f}% "
                f"(threshold: {self.thresholds['memory_percent']}%)"
            )

        # Order latency check
        if (
            "avg_order_latency_ms" in metrics
            and metrics["avg_order_latency_ms"]
            > self.thresholds["order_latency_ms"]
        ):
            logger.warning(
                f"High order latency: {metrics['avg_order_latency_ms']:.2f}ms "
                f"(threshold: {self.thresholds['order_latency_ms']}ms)"
            )

        # Market data latency check
        if (
            "avg_market_data_latency_ms" in metrics
            and metrics["avg_market_data_latency_ms"]
            > self.thresholds["market_data_latency_ms"]
        ):
            logger.warning(
                f"High market data latency: {metrics['avg_market_data_latency_ms']:.2f}ms "
                f"(threshold: {self.thresholds['market_data_latency_ms']}ms)"
            )

    async def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Store metrics in knowledge graph.

        Args:
            metrics: Metrics to store
        """
        if not self.kg:
            return

        try:
            await self.kg.write(
                """
                CREATE (m:SystemMetrics {
                    timestamp: datetime(),
                    cpu_percent: $cpu_percent,
                    memory_percent: $memory_percent,
                    memory_used_gb: $memory_used_gb,
                    disk_percent: $disk_percent,
                    network_sent_mb: $network_sent_mb,
                    network_recv_mb: $network_recv_mb,
                    avg_order_latency_ms: $avg_order_latency_ms,
                    avg_market_data_latency_ms: $avg_market_data_latency_ms,
                    avg_strategy_latency_ms: $avg_strategy_latency_ms
                })
                """,
                cpu_percent=metrics.get("cpu_percent"),
                memory_percent=metrics.get("memory_percent"),
                memory_used_gb=metrics.get("memory_used_gb"),
                disk_percent=metrics.get("disk_percent"),
                network_sent_mb=metrics.get("network_sent_mb"),
                network_recv_mb=metrics.get("network_recv_mb"),
                avg_order_latency_ms=metrics.get("avg_order_latency_ms", 0.0),
                avg_market_data_latency_ms=metrics.get(
                    "avg_market_data_latency_ms", 0.0
                ),
                avg_strategy_latency_ms=metrics.get("avg_strategy_latency_ms", 0.0),
            )
        except Exception as e:
            logger.error(f"Failed to store metrics in knowledge graph: {e}")

    def record_latency(self, operation: str, latency_ms: float) -> None:
        """
        Record latency for an operation.

        Args:
            operation: Operation type (order_execution, market_data, strategy_signal)
            latency_ms: Latency in milliseconds
        """
        if operation in self.latencies:
            self.latencies[operation].append(latency_ms)
            if len(self.latencies[operation]) > self.max_latency_samples:
                self.latencies[operation].pop(0)

    def get_latency_metrics(self) -> Dict[str, float]:
        """
        Get latency metrics.

        Returns:
            Dictionary with latency statistics
        """
        metrics = {}

        for operation, latencies in self.latencies.items():
            if latencies:
                metrics[f"avg_{operation}_latency_ms"] = sum(latencies) / len(
                    latencies
                )
                metrics[f"max_{operation}_latency_ms"] = max(latencies)
                metrics[f"min_{operation}_latency_ms"] = min(latencies)
                metrics[f"{operation}_sample_count"] = len(latencies)

        return metrics

    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent metrics.

        Returns:
            Latest metrics dictionary or None
        """
        if self.metrics_history:
            return self.metrics_history[-1].copy()
        return None

    def get_metrics_history(
        self, duration_seconds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics.

        Args:
            duration_seconds: Time period to retrieve (None for all)

        Returns:
            List of metrics dictionaries
        """
        if duration_seconds is None:
            return self.metrics_history.copy()

        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m["timestamp"] >= cutoff_time]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.

        Returns:
            Summary statistics
        """
        if not self.metrics_history:
            return {}

        recent_metrics = self.get_metrics_history(duration_seconds=300)  # Last 5 minutes

        if not recent_metrics:
            return {}

        return {
            "avg_cpu_percent": sum(m["cpu_percent"] for m in recent_metrics)
            / len(recent_metrics),
            "max_cpu_percent": max(m["cpu_percent"] for m in recent_metrics),
            "avg_memory_percent": sum(m["memory_percent"] for m in recent_metrics)
            / len(recent_metrics),
            "max_memory_percent": max(m["memory_percent"] for m in recent_metrics),
            "latency_summary": self.get_latency_metrics(),
            "samples": len(recent_metrics),
        }

    def clear_latency_history(self) -> None:
        """Clear latency history."""
        for operation in self.latencies:
            self.latencies[operation].clear()
        logger.info("Latency history cleared")

    def set_threshold(self, metric: str, value: float) -> None:
        """
        Set threshold for a metric.

        Args:
            metric: Metric name
            value: Threshold value
        """
        if metric in self.thresholds:
            self.thresholds[metric] = value
            logger.info(f"Threshold for {metric} set to {value}")
        else:
            logger.warning(f"Unknown metric: {metric}")
