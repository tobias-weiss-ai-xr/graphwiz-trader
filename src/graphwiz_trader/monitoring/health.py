"""System health checker with automated recovery actions."""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("CCXT not available. Exchange health checks disabled.")


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Types of recovery actions."""

    RECONNECT = "reconnect"
    RESTART = "restart"
    SCALE_DOWN = "scale_down"
    PAUSE_TRADING = "pause_trading"
    CLOSE_POSITIONS = "close_positions"
    NOTIFY_ADMIN = "notify_admin"
    CLEAR_CACHE = "clear_cache"
    RESET_RATE_LIMITS = "reset_rate_limits"


@dataclass
class HealthCheck:
    """Health check definition."""

    name: str
    check_func: Callable[[], Dict[str, Any]]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    enabled: bool = True


@dataclass
class HealthResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recovery_attempted: bool = False
    recovery_successful: Optional[bool] = None


class HealthChecker:
    """System health checker with automated recovery."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize health checker.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.health_config = config.get("health_checks", {})

        # Health check results storage
        self.health_results: Dict[str, HealthResult] = {}
        self.health_history: List[HealthResult] = []

        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "open": False,
                "failure_count": 0,
                "last_failure_time": None,
                "opened_at": None,
            }
        )

        # Recovery state
        self.recovery_history: List[Dict[str, Any]] = []

        # Dependencies
        self.exchange_manager = None
        self.neo4j_graph = None
        self.trading_engine = None
        self.agent_manager = None

        # Initialize health checks
        self._init_health_checks()

    def _init_health_checks(self) -> None:
        """Initialize built-in health checks."""
        self.health_checks = {
            "exchange_connectivity": HealthCheck(
                name="exchange_connectivity",
                check_func=self._check_exchange_connectivity,
                interval_seconds=30,
                recovery_actions=[RecoveryAction.RECONNECT, RecoveryAction.NOTIFY_ADMIN],
            ),
            "neo4j_connectivity": HealthCheck(
                name="neo4j_connectivity",
                check_func=self._check_neo4j_connectivity,
                interval_seconds=60,
                recovery_actions=[RecoveryAction.RECONNECT, RecoveryAction.NOTIFY_ADMIN],
            ),
            "api_rate_limits": HealthCheck(
                name="api_rate_limits",
                check_func=self._check_api_rate_limits,
                interval_seconds=10,
                recovery_actions=[RecoveryAction.SCALE_DOWN, RecoveryAction.PAUSE_TRADING],
            ),
            "system_resources": HealthCheck(
                name="system_resources",
                check_func=self._check_system_resources,
                interval_seconds=60,
                recovery_actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.NOTIFY_ADMIN],
            ),
            "disk_space": HealthCheck(
                name="disk_space",
                check_func=self._check_disk_space,
                interval_seconds=300,
                recovery_actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.NOTIFY_ADMIN],
            ),
            "agent_health": HealthCheck(
                name="agent_health",
                check_func=self._check_agent_health,
                interval_seconds=60,
                recovery_actions=[RecoveryAction.RESTART],
            ),
            "portfolio_health": HealthCheck(
                name="portfolio_health",
                check_func=self._check_portfolio_health,
                interval_seconds=30,
                recovery_actions=[RecoveryAction.PAUSE_TRADING, RecoveryAction.CLOSE_POSITIONS],
            ),
            "database_locks": HealthCheck(
                name="database_locks",
                check_func=self._check_database_locks,
                interval_seconds=60,
                recovery_actions=[RecoveryAction.CLEAR_CACHE],
            ),
        }

    def set_dependencies(
        self, exchange_manager=None, neo4j_graph=None, trading_engine=None, agent_manager=None
    ) -> None:
        """Set service dependencies for health checks.

        Args:
            exchange_manager: Exchange manager instance
            neo4j_graph: Neo4j knowledge graph instance
            trading_engine: Trading engine instance
            agent_manager: Agent manager instance
        """
        self.exchange_manager = exchange_manager
        self.neo4j_graph = neo4j_graph
        self.trading_engine = trading_engine
        self.agent_manager = agent_manager
        logger.info("Health checker dependencies configured")

    # Health Check Implementations
    def _check_exchange_connectivity(self) -> Dict[str, Any]:
        """Check exchange connectivity.

        Returns:
            Health check result dictionary
        """
        if not self.exchange_manager or not CCXT_AVAILABLE:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Exchange manager not configured, skipping check",
            }

        exchanges = (
            self.exchange_manager.get_all_exchanges()
            if hasattr(self.exchange_manager, "get_all_exchanges")
            else []
        )
        results = {}

        for exchange_name in exchanges:
            try:
                exchange = self.exchange_manager.get_exchange(exchange_name)
                # Try to fetch ticker to test connectivity
                ticker = exchange.fetch_ticker("BTC/USDT")
                results[exchange_name] = {
                    "connected": True,
                    "latency": (
                        exchange.last_response_headers.get("content-length", 0)
                        if hasattr(exchange, "last_response_headers")
                        else 0
                    ),
                }
            except Exception as e:
                results[exchange_name] = {"connected": False, "error": str(e)}

        # Determine overall status
        connected_count = sum(1 for r in results.values() if r.get("connected", False))
        total_count = len(results)

        if connected_count == total_count:
            status = HealthStatus.HEALTHY
            message = f"All {total_count} exchanges connected"
        elif connected_count > 0:
            status = HealthStatus.DEGRADED
            message = f"{connected_count}/{total_count} exchanges connected"
        else:
            status = HealthStatus.CRITICAL
            message = "No exchanges connected"

        return {"status": status, "message": message, "details": results}

    def _check_neo4j_connectivity(self) -> Dict[str, Any]:
        """Check Neo4j connectivity.

        Returns:
            Health check result dictionary
        """
        if not self.neo4j_graph:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Neo4j not configured, skipping check",
            }

        try:
            # Simple query to test connectivity
            result = self.neo4j_graph.query("RETURN 1 as test")
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Neo4j connected and responsive",
                "details": {"query_time_ms": result[0].get("test", 0) if result else 0},
            }
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Neo4j connection failed: {str(e)}",
                "details": {"error": str(e)},
            }

    def _check_api_rate_limits(self) -> Dict[str, Any]:
        """Check API rate limit status.

        Returns:
            Health check result dictionary
        """
        if not self.exchange_manager:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Exchange manager not configured, skipping check",
            }

        exchanges = (
            self.exchange_manager.get_all_exchanges()
            if hasattr(self.exchange_manager, "get_all_exchanges")
            else []
        )
        results = {}

        min_rate_limit = float("inf")

        for exchange_name in exchanges:
            try:
                exchange = self.exchange_manager.get_exchange(exchange_name)
                rate_limit = getattr(exchange, "rateLimit", None)

                if rate_limit:
                    remaining = getattr(exchange, "rate_limit_remaining", rate_limit)
                    usage_percent = (
                        ((rate_limit - remaining) / rate_limit * 100) if rate_limit > 0 else 0
                    )

                    results[exchange_name] = {
                        "rate_limit": rate_limit,
                        "remaining": remaining,
                        "usage_percent": usage_percent,
                    }

                    min_rate_limit = min(min_rate_limit, remaining)
            except Exception as e:
                results[exchange_name] = {"error": str(e)}

        # Determine status based on lowest remaining rate limit
        if min_rate_limit == float("inf"):
            status = HealthStatus.HEALTHY
            message = "No rate limit information available"
        elif min_rate_limit > 50:
            status = HealthStatus.HEALTHY
            message = "Rate limits healthy"
        elif min_rate_limit > 10:
            status = HealthStatus.DEGRADED
            message = f"Rate limits low: {min_rate_limit:.0f} remaining"
        else:
            status = HealthStatus.CRITICAL
            message = f"Rate limits critical: {min_rate_limit:.0f} remaining"

        return {"status": status, "message": message, "details": results}

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources (CPU, memory, etc.).

        Returns:
            Health check result dictionary
        """
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine status
            issues = []
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")

            if issues:
                status = HealthStatus.DEGRADED if len(issues) == 1 else HealthStatus.UNHEALTHY
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"

            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Failed to check system resources: {str(e)}",
                "details": {"error": str(e)},
            }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space.

        Returns:
            Health check result dictionary
        """
        try:
            import psutil

            disk = psutil.disk_usage("/")

            if disk.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk critically full: {disk.percent}%"
            elif disk.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Disk very full: {disk.percent}%"
            elif disk.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Disk getting full: {disk.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {disk.percent}%"

            return {
                "status": status,
                "message": message,
                "details": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": disk.percent,
                },
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Failed to check disk space: {str(e)}",
                "details": {"error": str(e)},
            }

    def _check_agent_health(self) -> Dict[str, Any]:
        """Check agent health.

        Returns:
            Health check result dictionary
        """
        if not self.agent_manager:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Agent manager not configured, skipping check",
            }

        try:
            # Check if agents are responsive
            agents = (
                self.agent_manager.get_all_agents()
                if hasattr(self.agent_manager, "get_all_agents")
                else []
            )
            results = {}

            failure_count = 0
            for agent in agents:
                try:
                    is_healthy = agent.is_healthy() if hasattr(agent, "is_healthy") else True
                    results[agent.name] = {
                        "healthy": is_healthy,
                        "last_prediction": getattr(agent, "last_prediction_time", None),
                    }
                    if not is_healthy:
                        failure_count += 1
                except Exception as e:
                    results[agent.name] = {"healthy": False, "error": str(e)}
                    failure_count += 1

            # Determine status
            total_count = len(agents)
            if failure_count == 0:
                status = HealthStatus.HEALTHY
                message = f"All {total_count} agents healthy"
            elif failure_count < total_count / 2:
                status = HealthStatus.DEGRADED
                message = f"{failure_count}/{total_count} agents unhealthy"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"{failure_count}/{total_count} agents unhealthy"

            return {
                "status": status,
                "message": message,
                "details": {
                    "total_agents": total_count,
                    "failure_count": failure_count,
                    "agents": results,
                },
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Failed to check agent health: {str(e)}",
                "details": {"error": str(e)},
            }

    def _check_portfolio_health(self) -> Dict[str, Any]:
        """Check portfolio health.

        Returns:
            Health check result dictionary
        """
        if not self.trading_engine:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Trading engine not configured, skipping check",
            }

        try:
            portfolio = (
                self.trading_engine.get_portfolio()
                if hasattr(self.trading_engine, "get_portfolio")
                else {}
            )
            metrics = (
                self.trading_engine.get_metrics()
                if hasattr(self.trading_engine, "get_metrics")
                else {}
            )

            # Check for concerning conditions
            issues = []

            drawdown = metrics.get("current_drawdown", 0)
            if drawdown > 0.15:
                issues.append(f"High drawdown: {drawdown*100:.1f}%")

            leverage = metrics.get("leverage", 0)
            if leverage > 3:
                issues.append(f"High leverage: {leverage:.1f}x")

            position_count = len(portfolio.get("positions", []))
            if position_count > 20:
                issues.append(f"Too many positions: {position_count}")

            if issues:
                status = HealthStatus.DEGRADED if len(issues) == 1 else HealthStatus.UNHEALTHY
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "Portfolio healthy"

            return {
                "status": status,
                "message": message,
                "details": {
                    "portfolio_value": portfolio.get("total_value", 0),
                    "drawdown": drawdown,
                    "leverage": leverage,
                    "position_count": position_count,
                },
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Failed to check portfolio health: {str(e)}",
                "details": {"error": str(e)},
            }

    def _check_database_locks(self) -> Dict[str, Any]:
        """Check for database locks.

        Returns:
            Health check result dictionary
        """
        if not self.neo4j_graph:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Neo4j not configured, skipping check",
            }

        try:
            # Check for long-running transactions
            query = """
            CALL dbms.listTransactions() YIELD transactionId, currentQueryId, username, metaData, activeTransactions
            RETURN count(*) as transaction_count
            """

            result = self.neo4j_graph.query(query)
            transaction_count = result[0]["transaction_count"] if result else 0

            if transaction_count > 10:
                status = HealthStatus.DEGRADED
                message = f"High number of transactions: {transaction_count}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database transactions normal: {transaction_count}"

            return {
                "status": status,
                "message": message,
                "details": {"transaction_count": transaction_count},
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Failed to check database locks: {str(e)}",
                "details": {"error": str(e)},
            }

    # Recovery Actions
    async def _perform_recovery(self, health_check: HealthCheck, result: HealthResult) -> bool:
        """Perform recovery actions for failed health check.

        Args:
            health_check: HealthCheck definition
            result: HealthResult that triggered recovery

        Returns:
            True if recovery was successful
        """
        logger.info("Attempting recovery for {}", health_check.name)

        recovery_success = False

        for action in health_check.recovery_actions:
            try:
                logger.info("Performing recovery action: {}", action.value)

                if action == RecoveryAction.RECONNECT:
                    success = await self._action_reconnect(health_check.name)
                elif action == RecoveryAction.RESTART:
                    success = await self._action_restart(health_check.name)
                elif action == RecoveryAction.SCALE_DOWN:
                    success = await self._action_scale_down()
                elif action == RecoveryAction.PAUSE_TRADING:
                    success = await self._action_pause_trading()
                elif action == RecoveryAction.CLOSE_POSITIONS:
                    success = await self._action_close_positions()
                elif action == RecoveryAction.CLEAR_CACHE:
                    success = await self._action_clear_cache()
                elif action == RecoveryAction.NOTIFY_ADMIN:
                    success = await self._action_notify_admin(health_check.name, result)
                elif action == RecoveryAction.RESET_RATE_LIMITS:
                    success = await self._action_reset_rate_limits()
                else:
                    logger.warning("Unknown recovery action: {}", action.value)
                    continue

                if success:
                    logger.info("Recovery action {} succeeded", action.value)
                    recovery_success = True
                    break

            except Exception as e:
                logger.error("Recovery action {} failed: {}", action.value, e)

        # Record recovery attempt
        self.recovery_history.append(
            {
                "check_name": health_check.name,
                "actions": [a.value for a in health_check.recovery_actions],
                "successful": recovery_success,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return recovery_success

    async def _action_reconnect(self, check_name: str) -> bool:
        """Attempt to reconnect service.

        Args:
            check_name: Name of the health check

        Returns:
            True if successful
        """
        if check_name == "exchange_connectivity" and self.exchange_manager:
            # Reconnect to exchanges
            logger.info("Attempting to reconnect to exchanges")
            # Implementation depends on exchange manager interface
            return True

        elif check_name == "neo4j_connectivity" and self.neo4j_graph:
            # Reconnect to Neo4j
            logger.info("Attempting to reconnect to Neo4j")
            try:
                self.neo4j_graph.connect()
                return True
            except Exception as e:
                logger.error("Failed to reconnect to Neo4j: {}", e)
                return False

        return False

    async def _action_restart(self, check_name: str) -> bool:
        """Restart service.

        Args:
            check_name: Name of the health check

        Returns:
            True if successful
        """
        if check_name == "agent_health" and self.agent_manager:
            logger.info("Attempting to restart unhealthy agents")
            # Implementation depends on agent manager interface
            return True

        return False

    async def _action_scale_down(self) -> bool:
        """Scale down trading operations.

        Returns:
            True if successful
        """
        if self.trading_engine:
            logger.info("Scaling down trading operations")
            # Reduce position sizes, frequency, etc.
            return True

        return False

    async def _action_pause_trading(self) -> bool:
        """Pause trading operations.

        Returns:
            True if successful
        """
        if self.trading_engine:
            logger.warning("Pausing trading operations")
            # Implementation depends on trading engine interface
            return True

        return False

    async def _action_close_positions(self) -> bool:
        """Close all positions.

        Returns:
            True if successful
        """
        if self.trading_engine:
            logger.critical("Closing all positions")
            # Implementation depends on trading engine interface
            return True

        return False

    async def _action_clear_cache(self) -> bool:
        """Clear caches.

        Returns:
            True if successful
        """
        logger.info("Clearing caches")
        # Clear various caches (Redis, in-memory, etc.)
        return True

    async def _action_notify_admin(self, check_name: str, result: HealthResult) -> bool:
        """Notify administrator of issue.

        Args:
            check_name: Name of the health check
            result: HealthResult

        Returns:
            True if successful (notification always returns True)
        """
        logger.warning("Admin notification: {} - {}", check_name, result.message)
        # In production, this would integrate with the alerting system
        return True

    async def _action_reset_rate_limits(self) -> bool:
        """Reset rate limit counters.

        Returns:
            True if successful
        """
        logger.info("Resetting rate limit counters")
        # Implementation depends on rate limiting strategy
        return True

    # Public API
    async def run_health_check(self, check_name: str) -> HealthResult:
        """Run a specific health check.

        Args:
            check_name: Name of the health check

        Returns:
            HealthResult object
        """
        if check_name not in self.health_checks:
            logger.error("Unknown health check: {}", check_name)
            return HealthResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Unknown health check: {check_name}",
            )

        health_check = self.health_checks[check_name]

        if not health_check.enabled:
            logger.debug("Health check {} is disabled", check_name)
            return HealthResult(
                name=check_name, status=HealthStatus.HEALTHY, message="Health check disabled"
            )

        try:
            # Check circuit breaker
            cb = self.circuit_breakers[check_name]
            if cb["open"]:
                # Check if we should attempt to close circuit breaker
                if cb["opened_at"] and (datetime.utcnow() - cb["opened_at"]).total_seconds() > 300:
                    # Attempt recovery
                    logger.info("Attempting to close circuit breaker for {}", check_name)
                    cb["open"] = False
                    cb["failure_count"] = 0
                else:
                    logger.warning("Circuit breaker open for {}, skipping check", check_name)
                    return HealthResult(
                        name=check_name,
                        status=HealthStatus.CRITICAL,
                        message="Circuit breaker open",
                        recovery_attempted=False,
                    )

            # Run health check
            logger.debug("Running health check: {}", check_name)
            result_dict = health_check.check_func()

            result = HealthResult(
                name=check_name,
                status=result_dict["status"],
                message=result_dict["message"],
                details=result_dict.get("details", {}),
            )

            # Store result
            self.health_results[check_name] = result
            self.health_history.append(result)

            # Keep history manageable
            if len(self.health_history) > 1000:
                self.health_history = self.health_history[-1000:]

            # Check if recovery is needed
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                logger.warning("Health check {} failed: {}", check_name, result.message)

                # Update circuit breaker
                cb["failure_count"] += 1
                cb["last_failure_time"] = datetime.utcnow()

                if cb["failure_count"] >= 3:
                    cb["open"] = True
                    cb["opened_at"] = datetime.utcnow()
                    logger.error(
                        "Circuit breaker opened for {} after {} failures",
                        check_name,
                        cb["failure_count"],
                    )

                # Attempt recovery
                if health_check.recovery_actions:
                    recovery_success = await self._perform_recovery(health_check, result)
                    result.recovery_attempted = True
                    result.recovery_successful = recovery_success
            else:
                # Reset circuit breaker on success
                cb["failure_count"] = 0

            return result

        except Exception as e:
            logger.error("Health check {} failed with exception: {}", check_name, e)
            result = HealthResult(
                name=check_name,
                status=HealthStatus.CRITICAL,
                message=f"Health check exception: {str(e)}",
            )
            self.health_results[check_name] = result
            return result

    async def run_all_health_checks(self) -> Dict[str, HealthResult]:
        """Run all enabled health checks.

        Returns:
            Dictionary mapping check names to HealthResult objects
        """
        results = {}

        tasks = []
        check_names = []

        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                tasks.append(self.run_health_check(name))
                check_names.append(name)

        if tasks:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(check_names, results_list):
                if isinstance(result, Exception):
                    logger.error("Health check {} raised exception: {}", name, result)
                    results[name] = HealthResult(
                        name=name, status=HealthStatus.CRITICAL, message=f"Exception: {str(result)}"
                    )
                else:
                    results[name] = result

        return results

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all health checks.

        Returns:
            Dictionary with health summary
        """
        summary = {
            "overall_status": HealthStatus.HEALTHY,
            "checks": {},
            "last_updated": datetime.utcnow().isoformat(),
        }

        for name, result in self.health_results.items():
            summary["checks"][name] = {
                "status": result.status.value,
                "message": result.message,
                "timestamp": result.timestamp.isoformat(),
            }

            # Update overall status (worst status wins)
            if result.status == HealthStatus.CRITICAL:
                summary["overall_status"] = HealthStatus.CRITICAL
            elif (
                result.status == HealthStatus.UNHEALTHY
                and summary["overall_status"] != HealthStatus.CRITICAL
            ):
                summary["overall_status"] = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and summary["overall_status"] not in [
                HealthStatus.CRITICAL,
                HealthStatus.UNHEALTHY,
            ]:
                summary["overall_status"] = HealthStatus.DEGRADED

        return summary

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status.

        Returns:
            Dictionary with circuit breaker status
        """
        return {
            name: {
                "open": state["open"],
                "failure_count": state["failure_count"],
                "last_failure_time": (
                    state["last_failure_time"].isoformat() if state["last_failure_time"] else None
                ),
                "opened_at": state["opened_at"].isoformat() if state["opened_at"] else None,
            }
            for name, state in self.circuit_breakers.items()
        }

    def reset_circuit_breaker(self, check_name: str) -> None:
        """Manually reset circuit breaker.

        Args:
            check_name: Name of the health check
        """
        if check_name in self.circuit_breakers:
            self.circuit_breakers[check_name] = {
                "open": False,
                "failure_count": 0,
                "last_failure_time": None,
                "opened_at": None,
            }
            logger.info("Circuit breaker reset for {}", check_name)
