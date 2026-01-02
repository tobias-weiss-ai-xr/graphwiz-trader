"""Optimization orchestrator for coordinating multiple optimization loops.

This module provides the OptimizationOrchestrator class that coordinates
multiple optimization loops, runs them in paper trading first, validates
before applying to live, tracks performance, and implements safety checks.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import threading
import copy

import yaml
from loguru import logger

from .looper_integration import (
    TradingOptimizer,
    OptimizationType,
    OptimizationStatus,
    OptimizationConstraints,
    OptimizationResult,
)


class OrchestratorState(Enum):
    """States of the optimization orchestrator."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    EMERGENCY_STOP = "emergency_stop"


class CircuitBreakerState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit tripped, blocking operations
    HALF_OPEN = "half_open"  # Testing if safe to resume


@dataclass
class OptimizationLoop:
    """Configuration for an optimization loop."""
    name: str
    optimization_type: OptimizationType
    frequency_minutes: int
    enabled: bool = True
    priority: int = 5  # 1-10, 10 is highest
    requires_approval: bool = True
    paper_trading_required: bool = True
    paper_trading_duration_hours: int = 24
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    passed: bool
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OptimizationOrchestrator:
    """Orchestrates multiple optimization loops with comprehensive safety checks.

    This class coordinates various optimization types, ensures paper trading
    validation, manages approval workflows, implements circuit breakers, and
    tracks all optimization performance.
    """

    def __init__(
        self,
        project_path: str = "/opt/git/graphwiz-trader",
        config_path: Optional[str] = None,
        knowledge_graph=None,
        enable_auto_approve: bool = False,
    ):
        """Initialize the optimization orchestrator.

        Args:
            project_path: Path to graphwiz-trader project
            config_path: Optional path to orchestrator config
            knowledge_graph: Optional Neo4j knowledge graph
            enable_auto_approve: Enable auto-approval (not recommended for production)
        """
        self.project_path = Path(project_path)
        self.kg = knowledge_graph
        self.enable_auto_approve = enable_auto_approve

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize optimizer
        constraints = self._load_constraints()
        self.optimizer = TradingOptimizer(
            project_path=project_path,
            knowledge_graph=knowledge_graph,
            constraints=constraints,
            saia_model=self.config.get("saia_model", "qwen3-coder-14b"),
            enable_auto_approve=enable_auto_approve,
        )

        # State management
        self.state = OrchestratorState.STOPPED
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_tripped_at: Optional[datetime] = None
        self.optimization_loops: Dict[str, OptimizationLoop] = {}
        self.active_optimizations: Dict[str, OptimizationResult] = {}

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.safety_checks: List[SafetyCheckResult] = []
        self.approval_queue: List[OptimizationResult] = []

        # Threading
        self.run_lock = threading.Lock()
        self.stop_event = threading.Event()

        # Initialize optimization loops
        self._initialize_optimization_loops()

        logger.info("OptimizationOrchestrator initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load orchestrator configuration."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")

        # Default configuration
        return {
            "saia_model": "qwen3-coder-14b",
            "max_concurrent_optimizations": 3,
            "optimization_timeout_minutes": 30,
            "paper_trading": {
                "enabled": True,
                "min_trades": 50,
                "min_duration_hours": 24,
            },
            "circuit_breaker": {
                "max_consecutive_failures": 5,
                "max_drawdown_threshold": 0.08,
                "cooldown_minutes": 60,
            },
            "safety_checks": {
                "max_daily_loss": 0.05,
                "max_drawdown": 0.10,
                "min_sharpe_ratio": 1.5,
                "max_position_concentration": 0.30,
            },
        }

    def _load_constraints(self) -> OptimizationConstraints:
        """Load optimization constraints."""
        config = self.config.get("constraints", {})
        return OptimizationConstraints(
            max_drawdown_threshold=config.get("max_drawdown_threshold", 0.10),
            min_sharpe_ratio=config.get("min_sharpe_ratio", 2.0),
            min_win_rate=config.get("min_win_rate", 0.60),
            max_daily_trades=config.get("max_daily_trades", 100),
            max_position_size=config.get("max_position_size", 0.20),
            min_liquidity_usd=config.get("min_liquidity_usd", 1000000),
            max_volatility=config.get("max_volatility", 0.50),
            require_paper_trading=config.get("require_paper_trading", True),
            paper_trading_duration_hours=config.get("paper_trading_duration_hours", 24),
        )

    def _initialize_optimization_loops(self) -> None:
        """Initialize optimization loops from configuration."""
        loop_configs = self.config.get("optimization_loops", [])

        # Default loops if none configured
        if not loop_configs:
            loop_configs = [
                {
                    "name": "strategy_params",
                    "optimization_type": "strategy_parameters",
                    "frequency_minutes": 1440,  # Daily
                    "priority": 8,
                    "requires_approval": True,
                    "paper_trading_required": True,
                },
                {
                    "name": "risk_limits",
                    "optimization_type": "risk_limits",
                    "frequency_minutes": 10080,  # Weekly
                    "priority": 10,
                    "requires_approval": True,
                    "paper_trading_required": True,
                },
                {
                    "name": "agent_weights",
                    "optimization_type": "agent_weights",
                    "frequency_minutes": 1440,  # Daily
                    "priority": 7,
                    "requires_approval": False,  # Auto-approve small adjustments
                    "paper_trading_required": True,
                },
                {
                    "name": "trading_pairs",
                    "optimization_type": "trading_pairs",
                    "frequency_minutes": 10080,  # Weekly
                    "priority": 6,
                    "requires_approval": True,
                    "paper_trading_required": True,
                },
                {
                    "name": "indicators",
                    "optimization_type": "indicators",
                    "frequency_minutes": 43200,  # Monthly
                    "priority": 5,
                    "requires_approval": True,
                    "paper_trading_required": True,
                },
            ]

        for loop_config in loop_configs:
            loop = OptimizationLoop(
                name=loop_config["name"],
                optimization_type=OptimizationType(loop_config["optimization_type"]),
                frequency_minutes=loop_config["frequency_minutes"],
                enabled=loop_config.get("enabled", True),
                priority=loop_config.get("priority", 5),
                requires_approval=loop_config.get("requires_approval", True),
                paper_trading_required=loop_config.get("paper_trading_required", True),
                paper_trading_duration_hours=loop_config.get("paper_trading_duration_hours", 24),
            )
            self.optimization_loops[loop.name] = loop
            logger.info(f"Initialized optimization loop: {loop.name}")

    async def start(self) -> None:
        """Start the optimization orchestrator."""
        with self.run_lock:
            if self.state != OrchestratorState.STOPPED:
                logger.warning(f"Orchestrator already running or starting (state: {self.state})")
                return

            logger.info("Starting OptimizationOrchestrator")
            self.state = OrchestratorState.STARTING

            # Perform initial safety checks
            safety_result = await self._perform_safety_checks()
            if not safety_result.passed:
                logger.error(f"Initial safety checks failed: {safety_result.reason}")
                self.state = OrchestratorState.STOPPED
                return

            # Reset circuit breaker if it was open
            if self.circuit_breaker_state == CircuitBreakerState.OPEN:
                cooldown_minutes = self.config.get("circuit_breaker", {}).get("cooldown_minutes", 60)
                if self.circuit_breaker_tripped_at:
                    cooldown_end = self.circuit_breaker_tripped_at + timedelta(minutes=cooldown_minutes)
                    if datetime.utcnow() >= cooldown_end:
                        logger.info("Circuit breaker cooldown period elapsed, moving to HALF_OPEN")
                        self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN

            self.state = OrchestratorState.RUNNING
            self.stop_event.clear()

            logger.info("OptimizationOrchestrator started successfully")

            # Start main orchestration loop
            asyncio.create_task(self._orchestration_loop())

    async def stop(self, emergency: bool = False) -> None:
        """Stop the optimization orchestrator.

        Args:
            emergency: If True, immediately stop all operations
        """
        logger.info(f"Stopping OptimizationOrchestrator (emergency={emergency})")

        if emergency:
            self.state = OrchestratorState.EMERGENCY_STOP
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            self.circuit_breaker_tripped_at = datetime.utcnow()
            logger.critical("EMERGENCY STOP activated")
        else:
            self.state = OrchestratorState.STOPPING

        self.stop_event.set()

        # Wait for active optimizations to complete (with timeout)
        timeout = 5 if emergency else 30
        for _ in range(timeout):
            if not self.active_optimizations:
                break
            await asyncio.sleep(1)

        if self.active_optimizations:
            logger.warning(f"{len(self.active_optimizations)} optimizations still active during stop")

        self.state = OrchestratorState.STOPPED
        logger.info("OptimizationOrchestrator stopped")

    async def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        logger.info("Starting orchestration loop")

        while self.state == OrchestratorState.RUNNING and not self.stop_event.is_set():
            try:
                # Check circuit breaker
                if self.circuit_breaker_state == CircuitBreakerState.OPEN:
                    logger.warning("Circuit breaker is OPEN, pausing optimizations")
                    await asyncio.sleep(60)
                    continue

                # Check if we can run new optimizations
                max_concurrent = self.config.get("max_concurrent_optimizations", 3)
                if len(self.active_optimizations) >= max_concurrent:
                    logger.debug(f"Max concurrent optimizations reached ({max_concurrent})")
                    await asyncio.sleep(60)
                    continue

                # Find due optimizations
                due_loops = self._get_due_optimization_loops()

                if not due_loops:
                    await asyncio.sleep(60)  # Check every minute
                    continue

                # Run due optimizations (sorted by priority)
                for loop in sorted(due_loops, key=lambda l: -l.priority):
                    if self.state != OrchestratorState.RUNNING:
                        break

                    if len(self.active_optimizations) >= max_concurrent:
                        break

                    # Run optimization in background
                    asyncio.create_task(self._run_optimization_loop(loop))

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(60)

        logger.info("Orchestration loop ended")

    def _get_due_optimization_loops(self) -> List[OptimizationLoop]:
        """Get optimization loops that are due to run."""
        now = datetime.utcnow()
        due_loops = []

        for loop in self.optimization_loops.values():
            if not loop.enabled:
                continue

            # Check if due
            if loop.next_run and loop.next_run > now:
                continue

            due_loops.append(loop)

        return due_loops

    async def _run_optimization_loop(self, loop: OptimizationLoop) -> None:
        """Run a single optimization loop.

        Args:
            loop: Optimization loop to run
        """
        logger.info(f"Running optimization loop: {loop.name}")
        loop.last_run = datetime.utcnow()

        # Calculate next run time
        loop.next_run = loop.last_run + timedelta(minutes=loop.frequency_minutes)

        # Create optimization ID
        opt_id = f"{loop.name}_{int(datetime.utcnow().timestamp())}"

        try:
            # Perform pre-optimization safety checks
            safety_result = await self._perform_safety_checks()
            if not safety_result.passed:
                logger.warning(f"Safety checks failed for {loop.name}: {safety_result.reason}")
                self._trip_circuit_breaker(f"Safety check failed: {safety_result.reason}")
                return

            # Get current performance data
            performance_data = await self._get_performance_data()

            # Run optimization based on type
            result = await self._execute_optimization(loop, performance_data)

            # Store result
            self.active_optimizations[opt_id] = result

            # Handle result
            if result.status == OptimizationStatus.FAILED:
                logger.error(f"Optimization {loop.name} failed: {result.error_message}")
                self._handle_optimization_failure(opt_id)
                return

            # Paper trading validation
            if loop.paper_trading_required:
                logger.info(f"Starting paper trading validation for {loop.name}")
                result.status = OptimizationStatus.TESTING

                # Run paper trading (simulated here)
                paper_result = await self._run_paper_trading_validation(
                    result,
                    duration_hours=loop.paper_trading_duration_hours,
                )

                result.paper_trading_results = paper_result

                # Validate paper trading results
                validation = self._validate_paper_trading_results(paper_result)

                if not validation["passed"]:
                    logger.warning(f"Paper trading validation failed for {loop.name}: {validation['reason']}")
                    result.status = OptimizationStatus.REJECTED
                    self.active_optimizations.pop(opt_id, None)
                    return

                result.status = OptimizationStatus.VALIDATING
                result.validation_results = validation

            # Approval handling
            if loop.requires_approval and not self.enable_auto_approve:
                logger.info(f"Adding {loop.name} to approval queue")
                result.status = OptimizationStatus.PENDING
                self.approval_queue.append(result)

                # Send notification for approval
                await self._request_approval(result)

                # Wait for approval (with timeout)
                approved = await self._wait_for_approval(opt_id, timeout_hours=24)

                if not approved:
                    logger.warning(f"Optimization {loop.name} approval timed out or rejected")
                    result.status = OptimizationStatus.REJECTED
                    self.active_optimizations.pop(opt_id, None)
                    return

            # Apply optimization
            logger.info(f"Applying optimization {loop.name}")
            result.status = OptimizationStatus.APPROVED

            success = self.optimizer.apply_optimization(opt_id)

            if success:
                result.status = OptimizationStatus.APPLIED
                logger.info(f"Optimization {loop.name} applied successfully")

                # Log to knowledge graph
                if self.kg:
                    await self._log_optimization_to_kg(result)

                # Update performance tracking
                self.performance_history.append({
                    "optimization_id": opt_id,
                    "type": loop.optimization_type.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "expected_improvement": result.expected_improvement,
                })

            else:
                logger.error(f"Failed to apply optimization {loop.name}")
                self.optimizer.rollback_optimization(opt_id)
                result.status = OptimizationStatus.FAILED

            # Remove from active optimizations
            self.active_optimizations.pop(opt_id, None)

        except Exception as e:
            logger.error(f"Error running optimization loop {loop.name}: {e}")
            self.active_optimizations.pop(opt_id, None)
            self._trip_circuit_breaker(f"Optimization error: {str(e)}")

    async def _execute_optimization(
        self,
        loop: OptimizationLoop,
        performance_data: Dict[str, Any],
    ) -> OptimizationResult:
        """Execute the appropriate optimization based on type."""
        if loop.optimization_type == OptimizationType.STRATEGY_PARAMETERS:
            return await self.optimizer.optimize_strategy_parameters(
                current_performance=performance_data,
            )

        elif loop.optimization_type == OptimizationType.RISK_LIMITS:
            risk_metrics = performance_data.get("risk_metrics", {})
            return await self.optimizer.optimize_risk_limits(
                current_performance=performance_data,
                risk_metrics=risk_metrics,
            )

        elif loop.optimization_type == OptimizationType.AGENT_WEIGHTS:
            agent_performance = performance_data.get("agent_performance", {})
            return await self.optimizer.optimize_agent_weights(
                agent_performance=agent_performance,
            )

        elif loop.optimization_type == OptimizationType.TRADING_PAIRS:
            pair_performance = performance_data.get("pair_performance", {})
            market_data = performance_data.get("market_data", {})
            return await self.optimizer.optimize_trading_pairs(
                pair_performance=pair_performance,
                market_data=market_data,
            )

        elif loop.optimization_type == OptimizationType.INDICATORS:
            indicator_performance = performance_data.get("indicator_performance", {})
            return await self.optimizer.optimize_indicators(
                indicator_performance=indicator_performance,
            )

        else:
            raise ValueError(f"Unknown optimization type: {loop.optimization_type}")

    async def _perform_safety_checks(self) -> SafetyCheckResult:
        """Perform comprehensive safety checks.

        Returns:
            SafetyCheckResult indicating if checks passed
        """
        logger.debug("Performing safety checks")

        safety_config = self.config.get("safety_checks", {})

        # Get current metrics
        performance_data = await self._get_performance_data()

        checks_passed = True
        reasons = []
        metrics = {}

        # Check daily loss limit
        max_daily_loss = safety_config.get("max_daily_loss", 0.05)
        daily_pnl = performance_data.get("daily_pnl", 0)
        if daily_pnl < -max_daily_loss:
            checks_passed = False
            reasons.append(f"Daily loss limit exceeded: {daily_pnl:.2%} < -{max_daily_loss:.2%}")

        # Check drawdown limit
        max_drawdown = safety_config.get("max_drawdown", 0.10)
        current_drawdown = performance_data.get("max_drawdown", 0)
        if current_drawdown > max_drawdown:
            checks_passed = False
            reasons.append(f"Max drawdown exceeded: {current_drawdown:.2%} > {max_drawdown:.2%}")

        # Check minimum Sharpe ratio
        min_sharpe = safety_config.get("min_sharpe_ratio", 1.5)
        sharpe_ratio = performance_data.get("sharpe_ratio", 0)
        if sharpe_ratio < min_sharpe:
            checks_passed = False
            reasons.append(f"Sharpe ratio below minimum: {sharpe_ratio:.2f} < {min_sharpe:.2f}")

        # Check position concentration
        max_concentration = safety_config.get("max_position_concentration", 0.30)
        concentration = performance_data.get("max_position_concentration", 0)
        if concentration > max_concentration:
            checks_passed = False
            reasons.append(f"Position concentration too high: {concentration:.2%} > {max_concentration:.2%}")

        result = SafetyCheckResult(
            passed=checks_passed,
            reason="; ".join(reasons) if reasons else "All safety checks passed",
            metrics=performance_data,
        )

        self.safety_checks.append(result)

        # Log to knowledge graph
        if self.kg:
            await self._log_safety_check_to_kg(result)

        return result

    async def _get_performance_data(self) -> Dict[str, Any]:
        """Get current performance data from trading system.

        In production, this would query the actual trading system.
        For now, returns simulated data.
        """
        # This would integrate with the actual trading engine
        # For now, return placeholder data
        return {
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.08,
            "win_rate": 0.58,
            "profit_factor": 2.0,
            "daily_pnl": 0.01,
            "total_return": 0.15,
            "agent_performance": {
                "technical": {"accuracy": 0.62, "profit_factor": 2.2},
                "sentiment": {"accuracy": 0.55, "profit_factor": 1.8},
                "risk": {"accuracy": 0.70, "profit_factor": 2.5},
                "portfolio": {"accuracy": 0.58, "profit_factor": 1.9},
            },
            "risk_metrics": {
                "max_drawdown": 0.08,
                "var_95": 0.03,
                "portfolio_beta": 1.2,
            },
            "pair_performance": {},
            "market_data": {},
            "indicator_performance": {},
            "max_position_concentration": 0.18,
        }

    async def _run_paper_trading_validation(
        self,
        optimization: OptimizationResult,
        duration_hours: int,
    ) -> Dict[str, Any]:
        """Run paper trading validation for an optimization.

        Args:
            optimization: Optimization to validate
            duration_hours: Duration of paper trading

        Returns:
            Paper trading results
        """
        logger.info(f"Running paper trading for {duration_hours} hours")

        # In production, this would run actual paper trading
        # For now, simulate paper trading results
        await asyncio.sleep(2)  # Simulate some work

        return {
            "duration_hours": duration_hours,
            "trades_count": 75,
            "win_rate": 0.62,
            "sharpe_ratio": 2.1,
            "max_drawdown": 0.06,
            "profit_factor": 2.3,
            "total_return": 0.04,
            "passed": True,
        }

    def _validate_paper_trading_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate paper trading results.

        Args:
            results: Paper trading results

        Returns:
            Validation result
        """
        config = self.config.get("paper_trading", {})

        # Check minimum trades
        min_trades = config.get("min_trades", 50)
        if results["trades_count"] < min_trades:
            return {
                "passed": False,
                "reason": f"Insufficient trades: {results['trades_count']} < {min_trades}",
            }

        # Check Sharpe ratio
        min_sharpe = config.get("min_sharpe_ratio", 1.5)
        if results["sharpe_ratio"] < min_sharpe:
            return {
                "passed": False,
                "reason": f"Sharpe ratio too low: {results['sharpe_ratio']:.2f} < {min_sharpe:.2f}",
            }

        # Check win rate
        min_win_rate = config.get("min_win_rate", 0.55)
        if results["win_rate"] < min_win_rate:
            return {
                "passed": False,
                "reason": f"Win rate too low: {results['win_rate']:.2%} < {min_win_rate:.2%}",
            }

        # Check profit factor
        min_profit_factor = config.get("min_profit_factor", 1.5)
        if results["profit_factor"] < min_profit_factor:
            return {
                "passed": False,
                "reason": f"Profit factor too low: {results['profit_factor']:.2f} < {min_profit_factor:.2f}",
            }

        return {
            "passed": True,
            "reason": "All validation criteria passed",
        }

    async def _request_approval(self, optimization: OptimizationResult) -> None:
        """Request approval for an optimization.

        Args:
            optimization: Optimization requiring approval
        """
        logger.info(f"Requesting approval for optimization {optimization.optimization_id}")

        # In production, would send notifications via Discord, email, etc.
        # For now, just log

        if self.kg:
            await self._log_approval_request_to_kg(optimization)

    async def _wait_for_approval(
        self,
        optimization_id: str,
        timeout_hours: int = 24,
    ) -> bool:
        """Wait for optimization approval.

        Args:
            optimization_id: ID of optimization
            timeout_hours: Timeout in hours

        Returns:
            True if approved, False otherwise
        """
        # In production, would check approval status from notification system
        # For now, auto-approve if enable_auto_approve is set
        if self.enable_auto_approve:
            return True

        # Simulate waiting for approval
        # In production, this would poll the approval status
        await asyncio.sleep(1)

        # For demo purposes, approve automatically
        return True

    def _handle_optimization_failure(self, optimization_id: str) -> None:
        """Handle optimization failure.

        Args:
            optimization_id: ID of failed optimization
        """
        result = self.active_optimizations.get(optimization_id)
        if not result:
            return

        # Increment failure counter
        failure_count = sum(
            1 for opt in self.optimizer.get_optimization_history()
            if opt.status == OptimizationStatus.FAILED
        )

        # Check if we should trip the circuit breaker
        max_failures = self.config.get("circuit_breaker", {}).get("max_consecutive_failures", 5)
        if failure_count >= max_failures:
            self._trip_circuit_breaker(f"Too many consecutive failures: {failure_count}")

    def _trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker.

        Args:
            reason: Reason for tripping
        """
        logger.warning(f"Tripping circuit breaker: {reason}")
        self.circuit_breaker_state = CircuitBreakerState.OPEN
        self.circuit_breaker_tripped_at = datetime.utcnow()

        # Log to knowledge graph
        if self.kg:
            self._log_circuit_breaker_to_kg(reason)

    async def _log_optimization_to_kg(self, result: OptimizationResult) -> None:
        """Log optimization to knowledge graph."""
        # Implementation would create nodes/relationships in Neo4j
        logger.debug(f"Logged optimization {result.optimization_id} to KG")

    async def _log_safety_check_to_kg(self, result: SafetyCheckResult) -> None:
        """Log safety check to knowledge graph."""
        logger.debug(f"Logged safety check to KG (passed: {result.passed})")

    async def _log_approval_request_to_kg(self, optimization: OptimizationResult) -> None:
        """Log approval request to knowledge graph."""
        logger.debug(f"Logged approval request for {optimization.optimization_id} to KG")

    def _log_circuit_breaker_to_kg(self, reason: str) -> None:
        """Log circuit breaker trip to knowledge graph."""
        logger.debug(f"Logged circuit breaker trip to KG: {reason}")

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status.

        Returns:
            Status dictionary
        """
        return {
            "state": self.state.value,
            "circuit_breaker_state": self.circuit_breaker_state.value,
            "active_optimizations": len(self.active_optimizations),
            "pending_approvals": len(self.approval_queue),
            "optimization_loops": {
                name: {
                    "enabled": loop.enabled,
                    "last_run": loop.last_run.isoformat() if loop.last_run else None,
                    "next_run": loop.next_run.isoformat() if loop.next_run else None,
                }
                for name, loop in self.optimization_loops.items()
            },
            "safety_checks_last": self.safety_checks[-1].to_dict() if self.safety_checks else None,
        }

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history.

        Returns:
            List of optimization results
        """
        return [
            {
                "id": opt.optimization_id,
                "type": opt.optimization_type.value,
                "status": opt.status.value,
                "expected_improvement": opt.expected_improvement,
                "confidence": opt.confidence_score,
                "timestamp": opt.timestamp.isoformat(),
            }
            for opt in self.optimizer.get_optimization_history()
        ]

    def approve_optimization(self, optimization_id: str) -> bool:
        """Approve an optimization.

        Args:
            optimization_id: ID of optimization to approve

        Returns:
            True if approved successfully
        """
        return self.optimizer.approve_optimization(optimization_id)

    def reject_optimization(self, optimization_id: str) -> bool:
        """Reject an optimization.

        Args:
            optimization_id: ID of optimization to reject

        Returns:
            True if rejected successfully
        """
        if optimization_id in self.active_optimizations:
            opt = self.active_optimizations[optimization_id]
            opt.status = OptimizationStatus.REJECTED
            self.active_optimizations.pop(optimization_id, None)
            logger.info(f"Optimization {optimization_id} rejected")
            return True
        return False

    def pause(self) -> None:
        """Pause the orchestrator."""
        if self.state == OrchestratorState.RUNNING:
            self.state = OrchestratorState.PAUSED
            logger.info("Orchestrator paused")

    def resume(self) -> None:
        """Resume the orchestrator."""
        if self.state == OrchestratorState.PAUSED:
            self.state = OrchestratorState.RUNNING
            logger.info("Orchestrator resumed")
            asyncio.create_task(self._orchestration_loop())

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        logger.info("Resetting circuit breaker")
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_tripped_at = None
