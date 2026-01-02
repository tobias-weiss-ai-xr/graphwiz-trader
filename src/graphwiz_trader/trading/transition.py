"""Transition manager for progressing from paper trading to live trading."""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from graphwiz_trader.trading.paper_trading import PaperTradingEngine
from graphwiz_trader.trading.safety import SafetyChecks


class TransitionRequirements:
    """Requirements for transitioning from paper to live trading."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize transition requirements.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._load_requirements()

    def _load_requirements(self) -> None:
        """Load transition requirements from config."""
        req_config = self.config.get("transition_requirements", {})

        # Minimum requirements
        self.min_paper_days = req_config.get("min_paper_days", 3)
        self.min_trades = req_config.get("min_trades", 100)
        self.max_drawdown_pct = req_config.get("max_drawdown_pct", 10.0)
        self.min_win_rate = req_config.get("min_win_rate", 55.0)
        self.min_sharpe_ratio = req_config.get("min_sharpe_ratio", 1.5)

        # Consistency requirements
        self.consecutive_profitable_days = req_config.get("consecutive_profitable_days", 2)
        self.max_single_loss_pct = req_config.get("max_single_loss_pct", 5.0)

        # Gradual transition settings
        self.initial_capital_pct = req_config.get("initial_capital_pct", 10.0)  # Start with 10%
        self.capital_increase_steps = req_config.get("capital_increase_steps", [10, 25, 50, 100])
        self.min_days_per_step = req_config.get("min_days_per_step", 3)

        # Monitoring requirements during transition
        self.monitoring_interval_hours = req_config.get("monitoring_interval_hours", 1)
        self.max_drawdown_rollback_pct = req_config.get("max_drawdown_rollback_pct", 5.0)


class TransitionManager:
    """Manages transition from paper trading to live trading."""

    def __init__(
        self,
        paper_engine: PaperTradingEngine,
        safety_checks: SafetyChecks,
        knowledge_graph,
        config: Optional[Dict[str, Any]] = None,
        alert_callback: Optional[Callable] = None
    ):
        """Initialize transition manager.

        Args:
            paper_engine: Paper trading engine
            safety_checks: Safety checks instance
            knowledge_graph: Knowledge graph instance
            config: Transition configuration
            alert_callback: Optional callback for alerts
        """
        self.paper_engine = paper_engine
        self.safety = safety_checks
        self.kg = knowledge_graph
        self.config = config or {}
        self.alert_callback = alert_callback

        # Requirements
        self.requirements = TransitionRequirements(config)

        # Transition state
        self.transition_stage = "paper"  # paper, gradual, full
        self.current_capital_pct = 0
        self.transition_start_time: Optional[datetime] = None
        self.last_stage_change: Optional[datetime] = None

        # Performance tracking
        self.paper_performance: Dict[str, Any] = {}
        self.live_performance: Dict[str, Any] = {}

        # Rollback tracking
        self.rollback_count = 0
        self.rollback_history: List[Dict[str, Any]] = []

        logger.info("Transition manager initialized")

    async def validate_paper_trading_readiness(
        self,
        current_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Validate if paper trading meets live trading requirements.

        Args:
            current_prices: Current market prices for valuation

        Returns:
            Validation results
        """
        logger.info("Validating paper trading readiness...")

        # Get paper trading metrics
        metrics = self.paper_engine.calculate_performance_metrics(current_prices or {})

        # Calculate runtime
        start_time = self.paper_engine.start_time
        runtime_days = (datetime.now(timezone.utc) - start_time).days

        # Validate each requirement
        validations = {
            "min_paper_days": self._validate_min_days(runtime_days),
            "min_trades": self._validate_min_trades(metrics["total_trades"]),
            "max_drawdown": self._validate_max_drawdown(metrics["max_drawdown_pct"]),
            "win_rate": self._validate_win_rate(metrics["win_rate"] * 100),
            "sharpe_ratio": self._validate_sharpe_ratio(metrics["sharpe_ratio"]),
            "profitability": self._validate_profitability(metrics),
            "consistency": self._validate_consistency(runtime_days, metrics)
        }

        # Overall status
        all_passed = all(v["passed"] for v in validations.values())

        # Store performance
        self.paper_performance = metrics

        # Generate recommendations
        recommendations = self._generate_recommendations(validations, metrics)

        result = {
            "ready_for_transition": all_passed,
            "validations": validations,
            "current_metrics": metrics,
            "recommendations": recommendations,
            "runtime_days": runtime_days
        }

        # Log to knowledge graph
        await self._log_validation_result(result)

        if all_passed:
            logger.info("Paper trading validation PASSED - ready for transition")
        else:
            logger.warning("Paper trading validation FAILED - not ready for transition")
            for check_name, validation in validations.items():
                if not validation["passed"]:
                    logger.warning("  - {}: {}", check_name, validation["message"])

        return result

    def _validate_min_days(self, runtime_days: int) -> Dict[str, Any]:
        """Validate minimum paper trading days."""
        passed = runtime_days >= self.requirements.min_paper_days

        return {
            "passed": passed,
            "requirement": f"Minimum {self.requirements.min_paper_days} days",
            "actual": f"{runtime_days} days",
            "message": (
                f"Paper trading duration: {runtime_days} days "
                f"(minimum: {self.requirements.min_paper_days} days)"
            )
        }

    def _validate_min_trades(self, trade_count: int) -> Dict[str, Any]:
        """Validate minimum trade count."""
        passed = trade_count >= self.requirements.min_trades

        return {
            "passed": passed,
            "requirement": f"Minimum {self.requirements.min_trades} trades",
            "actual": f"{trade_count} trades",
            "message": (
                f"Trade count: {trade_count} "
                f"(minimum: {self.requirements.min_trades} trades)"
            )
        }

    def _validate_max_drawdown(self, drawdown_pct: float) -> Dict[str, Any]:
        """Validate maximum drawdown."""
        passed = drawdown_pct <= self.requirements.max_drawdown_pct

        return {
            "passed": passed,
            "requirement": f"Maximum {self.requirements.max_drawdown_pct}%",
            "actual": f"{drawdown_pct:.2f}%",
            "message": (
                f"Maximum drawdown: {drawdown_pct:.2f}% "
                f"(limit: {self.requirements.max_drawdown_pct}%)"
            )
        }

    def _validate_win_rate(self, win_rate_pct: float) -> Dict[str, Any]:
        """Validate win rate."""
        passed = win_rate_pct >= self.requirements.min_win_rate

        return {
            "passed": passed,
            "requirement": f"Minimum {self.requirements.min_win_rate}%",
            "actual": f"{win_rate_pct:.2f}%",
            "message": (
                f"Win rate: {win_rate_pct:.2f}% "
                f"(minimum: {self.requirements.min_win_rate}%)"
            )
        }

    def _validate_sharpe_ratio(self, sharpe_ratio: float) -> Dict[str, Any]:
        """Validate Sharpe ratio."""
        passed = sharpe_ratio >= self.requirements.min_sharpe_ratio

        return {
            "passed": passed,
            "requirement": f"Minimum {self.requirements.min_sharpe_ratio}",
            "actual": f"{sharpe_ratio:.2f}",
            "message": (
                f"Sharpe ratio: {sharpe_ratio:.2f} "
                f"(minimum: {self.requirements.min_sharpe_ratio})"
            )
        }

    def _validate_profitability(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall profitability."""
        total_return = metrics["total_return_pct"]
        passed = total_return > 0

        return {
            "passed": passed,
            "requirement": "Positive returns",
            "actual": f"{total_return:.2f}%",
            "message": f"Total return: {total_return:.2f}%"
        }

    def _validate_consistency(
        self,
        runtime_days: int,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate trading consistency."""
        # Check if winning trades > losing trades
        win_rate = metrics["win_rate"]
        passed = win_rate > 0.5

        return {
            "passed": passed,
            "requirement": "Win rate > 50%",
            "actual": f"{win_rate*100:.2f}%",
            "message": f"Consistency check: {win_rate*100:.2f}% win rate"
        }

    def _generate_recommendations(
        self,
        validations: Dict[str, Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improvement.

        Args:
            validations: Validation results
            metrics: Performance metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        if not validations["min_days"]["passed"]:
            days_needed = self.requirements.min_paper_days - validations["min_days"]["actual"].split()[0]
            recommendations.append(f"Continue paper trading for {days_needed} more days")

        if not validations["min_trades"]["passed"]:
            trades_needed = self.requirements.min_trades - metrics["total_trades"]
            recommendations.append(f"Execute {trades_needed} more trades")

        if not validations["max_drawdown"]["passed"]:
            recommendations.append(
                "Review and improve risk management to reduce maximum drawdown"
            )

        if not validations["win_rate"]["passed"]:
            recommendations.append(
                "Refine trading strategy to improve win rate"
            )

        if not validations["sharpe_ratio"]["passed"]:
            recommendations.append(
                "Improve risk-adjusted returns (higher returns with lower volatility)"
            )

        if not validations["profitability"]["passed"]:
            recommendations.append(
                "Ensure positive returns before transitioning to live trading"
            )

        return recommendations

    async def start_gradual_transition(self, initial_capital_pct: Optional[float] = None) -> Dict[str, Any]:
        """Start gradual transition to live trading.

        Args:
            initial_capital_pct: Initial capital percentage (default from config)

        Returns:
            Transition status
        """
        # First validate paper trading is ready
        validation = await self.validate_paper_trading_readiness()

        if not validation["ready_for_transition"]:
            return {
                "success": False,
                "message": "Paper trading does not meet requirements",
                "validation": validation
            }

        # Determine initial capital percentage
        if initial_capital_pct is None:
            initial_capital_pct = self.requirements.initial_capital_pct

        # Start transition
        self.transition_stage = "gradual"
        self.current_capital_pct = initial_capital_pct
        self.transition_start_time = datetime.now(timezone.utc)
        self.last_stage_change = datetime.now(timezone.utc)

        logger.warning("Starting gradual transition to live trading")
        logger.warning("Initial capital allocation: {}%", initial_capital_pct)

        # Log to knowledge graph
        await self._log_transition_start(initial_capital_pct)

        # Start monitoring task
        asyncio.create_task(self._monitor_transition())

        return {
            "success": True,
            "stage": "gradual",
            "capital_allocation_pct": initial_capital_pct,
            "message": f"Started gradual transition with {initial_capital_pct}% of capital"
        }

    async def increase_capital_allocation(self) -> Dict[str, Any]:
        """Increase capital allocation in gradual transition.

        Returns:
            New allocation status
        """
        if self.transition_stage != "gradual":
            return {
                "success": False,
                "message": "Not in gradual transition stage"
            }

        # Check if enough time has passed since last change
        if self.last_stage_change:
            days_since_change = (datetime.now(timezone.utc) - self.last_stage_change).days
            if days_since_change < self.requirements.min_days_per_step:
                return {
                    "success": False,
                    "message": f"Need {self.requirements.min_days_per_step} days at current level"
                }

        # Find next capital level
        current_level_idx = -1
        for i, level in enumerate(self.requirements.capital_increase_steps):
            if level == self.current_capital_pct:
                current_level_idx = i
                break

        if current_level_idx == -1 or current_level_idx >= len(self.requirements.capital_increase_steps) - 1:
            # Already at max
            self.transition_stage = "full"
            logger.warning("Reached full live trading - transition complete")

            await self._log_transition_complete()

            return {
                "success": True,
                "stage": "full",
                "capital_allocation_pct": 100,
                "message": "Transition complete - now at 100% capital allocation"
            }

        # Move to next level
        new_allocation = self.requirements.capital_increase_steps[current_level_idx + 1]
        self.current_capital_pct = new_allocation
        self.last_stage_change = datetime.now(timezone.utc)

        logger.warning(
            "Increased capital allocation: {}% -> {}%",
            self.requirements.capital_increase_steps[current_level_idx],
            new_allocation
        )

        # Log to knowledge graph
        await self._log_capital_increase(
            self.requirements.capital_increase_steps[current_level_idx],
            new_allocation
        )

        return {
            "success": True,
            "stage": "gradual",
            "previous_allocation_pct": self.requirements.capital_increase_steps[current_level_idx],
            "capital_allocation_pct": new_allocation,
            "message": f"Increased capital allocation to {new_allocation}%"
        }

    async def check_rollback_conditions(
        self,
        live_performance: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Check if rollback to paper trading is needed.

        Args:
            live_performance: Current live trading performance

        Returns:
            Tuple of (should_rollback, reason)
        """
        if self.transition_stage == "paper":
            return False, None

        # Check drawdown
        if "drawdown_pct" in live_performance:
            drawdown = live_performance["drawdown_pct"]
            if drawdown > self.requirements.max_drawdown_rollback_pct:
                return True, f"Drawdown exceeded rollback threshold: {drawdown:.2f}%"

        # Check consecutive losses
        if "consecutive_losses" in live_performance:
            consecutive_losses = live_performance["consecutive_losses"]
            if consecutive_losses >= 5:
                return True, f"Too many consecutive losses: {consecutive_losses}"

        # Check safety violations
        safety_status = self.safety.get_status()
        if safety_status.get("circuit_breaker_active"):
            return True, "Circuit breaker activated"

        return False, None

    async def execute_rollback(self, reason: str) -> Dict[str, Any]:
        """Execute rollback to paper trading.

        Args:
            reason: Reason for rollback

        Returns:
            Rollback status
        """
        logger.error("Executing ROLLBACK to paper trading")
        logger.error("Reason: {}", reason)

        # Record rollback
        rollback_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_stage": self.transition_stage,
            "to_stage": "paper",
            "capital_allocation": self.current_capital_pct,
            "reason": reason
        }
        self.rollback_history.append(rollback_record)
        self.rollback_count += 1

        # Reset state
        self.transition_stage = "paper"
        self.current_capital_pct = 0
        self.transition_start_time = None

        # Log to knowledge graph
        await self._log_rollback(reason)

        # Send alert
        if self.alert_callback:
            try:
                await self.alert_callback(
                    alert_type="rollback",
                    message=f"Rolled back to paper trading: {reason}",
                    details=rollback_record
                )
            except Exception as e:
                logger.error("Alert callback failed: {}", e)

        return {
            "success": True,
            "stage": "paper",
            "rollback_count": self.rollback_count,
            "message": f"Rolled back to paper trading: {reason}"
        }

    async def _monitor_transition(self) -> None:
        """Monitor gradual transition and alert on issues."""
        if self.transition_stage != "gradual":
            return

        logger.info("Starting gradual transition monitoring")

        while self.transition_stage == "gradual":
            try:
                # Check performance (would integrate with live metrics)
                # For now, just log monitoring status
                logger.debug(
                    "Transition monitoring: stage={}, capital_allocation={}%",
                    self.transition_stage,
                    self.current_capital_pct
                )

                # Sleep for monitoring interval
                await asyncio.sleep(self.requirements.monitoring_interval_hours * 3600)

            except Exception as e:
                logger.error("Error in transition monitoring: {}", e)
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _log_validation_result(self, result: Dict[str, Any]) -> None:
        """Log validation result to knowledge graph.

        Args:
            result: Validation result
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (vr:ValidationResult {
                        timestamp: datetime($timestamp),
                        ready_for_transition: $ready,
                        validations: $validations,
                        metrics: $metrics,
                        recommendations: $recommendations
                    })
                    RETURN vr
                    """,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    ready=result["ready_for_transition"],
                    validations=str(result["validations"]),
                        metrics=str(result["current_metrics"]),
                    recommendations=result["recommendations"]
                )
                logger.debug("Logged validation result to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log validation result: {}", e)

    async def _log_transition_start(self, capital_pct: float) -> None:
        """Log transition start to knowledge graph.

        Args:
            capital_pct: Initial capital percentage
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (ts:TransitionStart {
                        timestamp: datetime($timestamp),
                        stage: 'gradual',
                        capital_allocation_pct: $capital_pct
                    })
                    RETURN ts
                    """,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    capital_pct=capital_pct
                )
                logger.debug("Logged transition start to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log transition start: {}", e)

    async def _log_capital_increase(self, old_pct: float, new_pct: float) -> None:
        """Log capital increase to knowledge graph.

        Args:
            old_pct: Old capital percentage
            new_pct: New capital percentage
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (ci:CapitalIncrease {
                        timestamp: datetime($timestamp),
                        old_allocation_pct: $old_pct,
                        new_allocation_pct: $new_pct
                    })
                    RETURN ci
                    """,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    old_pct=old_pct,
                    new_pct=new_pct
                )
                logger.debug("Logged capital increase to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log capital increase: {}", e)

    async def _log_transition_complete(self) -> None:
        """Log transition completion to knowledge graph."""
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (tc:TransitionComplete {
                        timestamp: datetime($timestamp),
                        stage: 'full',
                        capital_allocation_pct: 100.0
                    })
                    RETURN tc
                    """,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                logger.debug("Logged transition completion to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log transition completion: {}", e)

    async def _log_rollback(self, reason: str) -> None:
        """Log rollback to knowledge graph.

        Args:
            reason: Reason for rollback
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (rb:Rollback {
                        timestamp: datetime($timestamp),
                        reason: $reason,
                        rollback_count: $count
                    })
                    RETURN rb
                    """,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    reason=reason,
                    count=self.rollback_count
                )
                logger.debug("Logged rollback to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log rollback: {}", e)

    def get_transition_status(self) -> Dict[str, Any]:
        """Get current transition status.

        Returns:
            Status dictionary
        """
        status = {
            "stage": self.transition_stage,
            "capital_allocation_pct": self.current_capital_pct,
            "rollback_count": self.rollback_count,
            "requirements": {
                "min_paper_days": self.requirements.min_paper_days,
                "min_trades": self.requirements.min_trades,
                "max_drawdown_pct": self.requirements.max_drawdown_pct,
                "min_win_rate": self.requirements.min_win_rate,
                "min_sharpe_ratio": self.requirements.min_sharpe_ratio
            }
        }

        if self.transition_start_time:
            status["transition_duration_days"] = (
                datetime.now(timezone.utc) - self.transition_start_time
            ).days

        if self.last_stage_change:
            status["days_at_current_stage"] = (
                datetime.now(timezone.utc) - self.last_stage_change
            ).days

        return status

    def __repr__(self) -> str:
        return f"TransitionManager(stage={self.transition_stage}, capital={self.current_capital_pct}%)"
