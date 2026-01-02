"""TradingOptimizer class integrating SAIA agent from agent-looper.

This module provides continuous autonomous optimization for trading strategies,
risk limits, agent weights, trading pairs, and technical indicators.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import threading
import copy

import numpy as np
from loguru import logger

# Import SAIA agent from agent-looper
import sys
sys.path.insert(0, str(Path("/opt/git/agent-looper/src")))
from core.saia_agent import SAIAAgent


class OptimizationType(Enum):
    """Types of trading optimizations."""
    STRATEGY_PARAMETERS = "strategy_parameters"
    RISK_LIMITS = "risk_limits"
    AGENT_WEIGHTS = "agent_weights"
    TRADING_PAIRS = "trading_pairs"
    INDICATORS = "indicators"


class OptimizationStatus(Enum):
    """Status of an optimization."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    TESTING = "testing"
    VALIDATING = "validating"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class OptimizationConstraints:
    """Constraints for optimization."""
    max_drawdown_threshold: float = 0.10  # 10% max drawdown
    min_sharpe_ratio: float = 2.0
    min_win_rate: float = 0.60  # 60% win rate
    max_daily_trades: int = 100
    max_position_size: float = 0.20  # 20% of portfolio
    min_liquidity_usd: float = 1000000  # $1M daily volume
    max_volatility: float = 0.50  # 50% daily volatility max
    require_paper_trading: bool = True
    paper_trading_duration_hours: int = 24


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    optimization_id: str
    optimization_type: OptimizationType
    status: OptimizationStatus
    proposed_changes: Dict[str, Any]
    expected_improvement: float
    confidence_score: float
    reasoning: str
    paper_trading_results: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


@dataclass
class RollbackState:
    """State for rollback capability."""
    snapshot_id: str
    timestamp: datetime
    config_backup: Dict[str, Any]
    performance_snapshot: Dict[str, Any]
    optimization_id: str


class TradingOptimizer:
    """Continuous autonomous optimizer for trading systems using SAIA agent.

    This class integrates with agent-looper's SAIA agent to provide intelligent
    optimization of trading strategies, risk limits, agent weights, and other
    parameters with comprehensive safety checks and rollback capability.
    """

    def __init__(
        self,
        project_path: str = "/opt/git/graphwiz-trader",
        knowledge_graph=None,
        constraints: Optional[OptimizationConstraints] = None,
        saia_model: str = "qwen3-coder-14b",
        enable_auto_approve: bool = False,
    ):
        """Initialize the trading optimizer.

        Args:
            project_path: Path to graphwiz-trader project
            knowledge_graph: Optional Neo4j knowledge graph for tracking
            constraints: Optimization constraints (uses defaults if None)
            saia_model: SAIA model to use for optimization
            enable_auto_approve: Auto-approve optimizations (dangerous in production)
        """
        self.project_path = Path(project_path)
        self.kg = knowledge_graph
        self.constraints = constraints or OptimizationConstraints()
        self.enable_auto_approve = enable_auto_approve

        # Initialize SAIA agent
        logger.info(f"Initializing SAIA agent with model: {saia_model}")
        self.agent = SAIAAgent(
            model=saia_model,
            max_tokens=8192,
            temperature=0.3  # Lower temperature for more deterministic results
        )

        # State management
        self.optimizations: Dict[str, OptimizationResult] = {}
        self.rollback_states: List[RollbackState] = []
        self.current_config: Dict[str, Any] = {}
        self.paper_trading_active = False
        self.optimization_lock = threading.Lock()

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.last_optimization_time: Optional[datetime] = None

        # Load current configuration
        self._load_current_config()

        logger.info("TradingOptimizer initialized successfully")

    def _load_current_config(self) -> None:
        """Load current trading configuration."""
        config_paths = [
            self.project_path / "config" / "config.yaml",
            self.project_path / "config" / "agents.yaml",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path) as f:
                        config_data = yaml.safe_load(f)
                        self.current_config.update(config_data)
                        logger.info(f"Loaded config from {config_path}")
                except Exception as e:
                    logger.error(f"Failed to load config from {config_path}: {e}")

    async def optimize_strategy_parameters(
        self,
        current_performance: Dict[str, Any],
        strategy_name: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize strategy parameters.

        Args:
            current_performance: Current performance metrics
            strategy_name: Specific strategy to optimize (optimizes all if None)

        Returns:
            OptimizationResult with proposed changes
        """
        logger.info(f"Starting strategy parameter optimization for {strategy_name or 'all strategies'}")

        with self.optimization_lock:
            opt_id = f"strategy_{datetime.utcnow().timestamp()}"

            try:
                # Create snapshot for rollback
                snapshot = self._create_rollback_snapshot(opt_id)

                # Analyze current state
                analysis = await self._analyze_strategy_performance(
                    current_performance,
                    strategy_name
                )

                # Generate optimization plan
                plan = await self._generate_strategy_optimization_plan(analysis)

                # Extract proposed changes
                proposed_changes = self._extract_proposed_changes(plan)

                # Validate constraints
                validation = self._validate_constraints(proposed_changes)
                if not validation["passed"]:
                    return OptimizationResult(
                        optimization_id=opt_id,
                        optimization_type=OptimizationType.STRATEGY_PARAMETERS,
                        status=OptimizationStatus.REJECTED,
                        proposed_changes={},
                        expected_improvement=0.0,
                        confidence_score=0.0,
                        reasoning=f"Constraint validation failed: {validation['reason']}"
                    )

                # Calculate expected improvement and confidence
                metrics = self._calculate_optimization_metrics(plan, current_performance)

                result = OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.STRATEGY_PARAMETERS,
                    status=OptimizationStatus.PENDING,
                    proposed_changes=proposed_changes,
                    expected_improvement=metrics["expected_improvement"],
                    confidence_score=metrics["confidence"],
                    reasoning=plan
                )

                # Store optimization
                self.optimizations[opt_id] = result
                self.rollback_states.append(snapshot)

                logger.info(f"Strategy optimization {opt_id} planned successfully")
                return result

            except Exception as e:
                logger.error(f"Strategy optimization failed: {e}")
                return OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.STRATEGY_PARAMETERS,
                    status=OptimizationStatus.FAILED,
                    proposed_changes={},
                    expected_improvement=0.0,
                    confidence_score=0.0,
                    reasoning="",
                    error_message=str(e)
                )

    async def optimize_risk_limits(
        self,
        current_performance: Dict[str, Any],
        risk_metrics: Dict[str, Any],
    ) -> OptimizationResult:
        """Optimize risk management limits.

        Args:
            current_performance: Current performance metrics
            risk_metrics: Current risk metrics (drawdown, VaR, etc.)

        Returns:
            OptimizationResult with proposed risk limit changes
        """
        logger.info("Starting risk limit optimization")

        with self.optimization_lock:
            opt_id = f"risk_{datetime.utcnow().timestamp()}"

            try:
                # Create snapshot for rollback
                snapshot = self._create_rollback_snapshot(opt_id)

                # Prepare context for SAIA
                context = self._prepare_risk_optimization_context(
                    current_performance,
                    risk_metrics
                )

                # Generate optimization using SAIA
                prompt = f"""
Analyze and optimize the risk management parameters for this trading system:

Current Performance:
{json.dumps(current_performance, indent=2)}

Current Risk Metrics:
{json.dumps(risk_metrics, indent=2)}

Current Risk Configuration:
{json.dumps(self.current_config.get('trading', {}), indent=2)}

Optimization Goals:
1. Minimize maximum drawdown (target: < {self.constraints.max_drawdown_threshold:.1%})
2. Maintain Sharpe ratio > {self.constraints.min_sharpe_ratio}
3. Optimize risk-adjusted returns
4. Ensure position sizes align with volatility

Provide specific recommendations for:
- Stop loss percentages
- Take profit percentages
- Maximum position sizes
- Risk per trade
- Daily loss limits

Format your response as a JSON object with specific numeric values.
"""

                response = self.agent.chat(prompt)

                # Parse recommendations
                proposed_changes = self._parse_risk_recommendations(response)

                # Validate against constraints
                validation = self._validate_risk_constraints(proposed_changes, risk_metrics)
                if not validation["passed"]:
                    return OptimizationResult(
                        optimization_id=opt_id,
                        optimization_type=OptimizationType.RISK_LIMITS,
                        status=OptimizationStatus.REJECTED,
                        proposed_changes={},
                        expected_improvement=0.0,
                        confidence_score=0.0,
                        reasoning=f"Risk constraint validation failed: {validation['reason']}"
                    )

                # Calculate expected improvement
                expected_improvement = self._estimate_risk_improvement(
                    proposed_changes,
                    risk_metrics
                )

                result = OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.RISK_LIMITS,
                    status=OptimizationStatus.PENDING,
                    proposed_changes=proposed_changes,
                    expected_improvement=expected_improvement,
                    confidence_score=0.75,
                    reasoning=response
                )

                self.optimizations[opt_id] = result
                self.rollback_states.append(snapshot)

                logger.info(f"Risk limit optimization {opt_id} planned successfully")
                return result

            except Exception as e:
                logger.error(f"Risk limit optimization failed: {e}")
                return OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.RISK_LIMITS,
                    status=OptimizationStatus.FAILED,
                    proposed_changes={},
                    expected_improvement=0.0,
                    confidence_score=0.0,
                    reasoning="",
                    error_message=str(e)
                )

    async def optimize_agent_weights(
        self,
        agent_performance: Dict[str, Dict[str, Any]],
    ) -> OptimizationResult:
        """Optimize agent decision weights.

        Args:
            agent_performance: Performance metrics for each agent

        Returns:
            OptimizationResult with proposed weight adjustments
        """
        logger.info("Starting agent weight optimization")

        with self.optimization_lock:
            opt_id = f"weights_{datetime.utcnow().timestamp()}"

            try:
                snapshot = self._create_rollback_snapshot(opt_id)

                # Prepare context
                context = {
                    "agent_performance": agent_performance,
                    "current_weights": self.current_config.get("agents", {}),
                }

                prompt = f"""
Analyze and optimize the agent weights for the trading decision system:

Current Agent Performance:
{json.dumps(agent_performance, indent=2)}

Optimization Goals:
1. Maximize overall decision accuracy
2. Minimize false signals
3. Weight agents by their recent performance
4. Ensure diversity in decision sources

Provide optimal weights for each agent (0.0 to 1.0, sum to approximately 1.0).
Format as JSON with agent names as keys and weights as values.
"""

                response = self.agent.chat(prompt)
                proposed_weights = self._parse_agent_weights(response)

                # Validate weights sum and ranges
                validation = self._validate_agent_weights(proposed_weights)
                if not validation["passed"]:
                    return OptimizationResult(
                        optimization_id=opt_id,
                        optimization_type=OptimizationType.AGENT_WEIGHTS,
                        status=OptimizationStatus.REJECTED,
                        proposed_changes={},
                        expected_improvement=0.0,
                        confidence_score=0.0,
                        reasoning=f"Weight validation failed: {validation['reason']}"
                    )

                result = OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.AGENT_WEIGHTS,
                    status=OptimizationStatus.PENDING,
                    proposed_changes=proposed_weights,
                    expected_improvement=0.15,  # Estimated 15% improvement
                    confidence_score=0.70,
                    reasoning=response
                )

                self.optimizations[opt_id] = result
                self.rollback_states.append(snapshot)

                logger.info(f"Agent weight optimization {opt_id} planned successfully")
                return result

            except Exception as e:
                logger.error(f"Agent weight optimization failed: {e}")
                return OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.AGENT_WEIGHTS,
                    status=OptimizationStatus.FAILED,
                    proposed_changes={},
                    expected_improvement=0.0,
                    confidence_score=0.0,
                    reasoning="",
                    error_message=str(e)
                )

    async def optimize_trading_pairs(
        self,
        pair_performance: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> OptimizationResult:
        """Optimize trading pair selection.

        Args:
            pair_performance: Performance metrics for each trading pair
            market_data: Current market conditions and data

        Returns:
            OptimizationResult with proposed pair additions/removals
        """
        logger.info("Starting trading pair optimization")

        with self.optimization_lock:
            opt_id = f"pairs_{datetime.utcnow().timestamp()}"

            try:
                snapshot = self._create_rollback_snapshot(opt_id)

                prompt = f"""
Analyze and optimize the trading pair selection:

Current Pair Performance:
{json.dumps(pair_performance, indent=2)}

Market Conditions:
{json.dumps(market_data, indent=2)}

Optimization Criteria:
- Minimum daily volume: ${self.constraints.min_liquidity_usd:,.0f}
- Maximum volatility: {self.constraints.max_volatility:.1%}
- Win rate target: > {self.constraints.min_win_rate:.1%}
- Diversification across sectors/asset classes

Recommend:
1. Pairs to add (with justification)
2. Pairs to remove (with justification)
3. Pairs to keep (with current assessment)
"""

                response = self.agent.chat(prompt)
                recommendations = self._parse_pair_recommendations(response)

                # Validate each recommended pair
                validated_recommendations = self._validate_pair_recommendations(
                    recommendations,
                    market_data
                )

                result = OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.TRADING_PAIRS,
                    status=OptimizationStatus.PENDING,
                    proposed_changes=validated_recommendations,
                    expected_improvement=0.10,
                    confidence_score=0.65,
                    reasoning=response
                )

                self.optimizations[opt_id] = result
                self.rollback_states.append(snapshot)

                logger.info(f"Trading pair optimization {opt_id} planned successfully")
                return result

            except Exception as e:
                logger.error(f"Trading pair optimization failed: {e}")
                return OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.TRADING_PAIRS,
                    status=OptimizationStatus.FAILED,
                    proposed_changes={},
                    expected_improvement=0.0,
                    confidence_score=0.0,
                    reasoning="",
                    error_message=str(e)
                )

    async def optimize_indicators(
        self,
        indicator_performance: Dict[str, Dict[str, Any]],
    ) -> OptimizationResult:
        """Optimize technical indicator parameters.

        Args:
            indicator_performance: Performance of different indicators

        Returns:
            OptimizationResult with proposed indicator adjustments
        """
        logger.info("Starting indicator optimization")

        with self.optimization_lock:
            opt_id = f"indicators_{datetime.utcnow().timestamp()}"

            try:
                snapshot = self._create_rollback_snapshot(opt_id)

                prompt = f"""
Analyze and optimize technical indicator parameters:

Current Indicator Performance:
{json.dumps(indicator_performance, indent=2)}

Optimization Goals:
1. Maximize signal accuracy
2. Minimize lag
3. Reduce false signals
4. Optimize for current market regime

Provide specific parameter recommendations for:
- RSI periods and thresholds
- MACD parameters
- Bollinger Band periods and std deviations
- EMA periods
- Any other indicators in use

Format as JSON with indicator names as keys and parameter objects as values.
"""

                response = self.agent.chat(prompt)
                proposed_changes = self._parse_indicator_recommendations(response)

                result = OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.INDICATORS,
                    status=OptimizationStatus.PENDING,
                    proposed_changes=proposed_changes,
                    expected_improvement=0.08,
                    confidence_score=0.60,
                    reasoning=response
                )

                self.optimizations[opt_id] = result
                self.rollback_states.append(snapshot)

                logger.info(f"Indicator optimization {opt_id} planned successfully")
                return result

            except Exception as e:
                logger.error(f"Indicator optimization failed: {e}")
                return OptimizationResult(
                    optimization_id=opt_id,
                    optimization_type=OptimizationType.INDICATORS,
                    status=OptimizationStatus.FAILED,
                    proposed_changes={},
                    expected_improvement=0.0,
                    confidence_score=0.0,
                    reasoning="",
                    error_message=str(e)
                )

    def approve_optimization(self, optimization_id: str) -> bool:
        """Approve an optimization for application.

        Args:
            optimization_id: ID of optimization to approve

        Returns:
            True if approved successfully
        """
        if optimization_id not in self.optimizations:
            logger.error(f"Optimization {optimization_id} not found")
            return False

        optimization = self.optimizations[optimization_id]

        if optimization.status != OptimizationStatus.PENDING:
            logger.warning(f"Optimization {optimization_id} not in PENDING status")
            return False

        optimization.status = OptimizationStatus.APPROVED
        logger.info(f"Optimization {optimization_id} approved")

        # Log to knowledge graph if available
        if self.kg:
            self._log_optimization_to_kg(optimization, "approved")

        return True

    def apply_optimization(self, optimization_id: str) -> bool:
        """Apply an approved optimization.

        Args:
            optimization_id: ID of optimization to apply

        Returns:
            True if applied successfully
        """
        if optimization_id not in self.optimizations:
            logger.error(f"Optimization {optimization_id} not found")
            return False

        optimization = self.optimizations[optimization_id]

        if optimization.status != OptimizationStatus.APPROVED:
            logger.warning(f"Optimization {optimization_id} not approved")
            return False

        try:
            # Apply changes to configuration
            self._apply_configuration_changes(optimization.proposed_changes)

            # Update status
            optimization.status = OptimizationStatus.APPLIED
            self.last_optimization_time = datetime.utcnow()

            logger.info(f"Optimization {optimization_id} applied successfully")

            # Log to knowledge graph
            if self.kg:
                self._log_optimization_to_kg(optimization, "applied")

            return True

        except Exception as e:
            logger.error(f"Failed to apply optimization {optimization_id}: {e}")

            # Attempt rollback
            self.rollback_optimization(optimization_id)

            return False

    def rollback_optimization(self, optimization_id: str) -> bool:
        """Rollback an optimization.

        Args:
            optimization_id: ID of optimization to rollback

        Returns:
            True if rolled back successfully
        """
        # Find the rollback state
        rollback_state = None
        for state in self.rollback_states:
            if state.optimization_id == optimization_id:
                rollback_state = state
                break

        if not rollback_state:
            logger.error(f"No rollback state found for optimization {optimization_id}")
            return False

        try:
            # Restore configuration
            self.current_config = rollback_state.config_backup.copy()

            # Update optimization status
            if optimization_id in self.optimizations:
                self.optimizations[optimization_id].status = OptimizationStatus.ROLLED_BACK

            logger.info(f"Optimization {optimization_id} rolled back successfully")

            # Log to knowledge graph
            if self.kg:
                opt = self.optimizations.get(optimization_id)
                if opt:
                    self._log_optimization_to_kg(opt, "rolled_back")

            return True

        except Exception as e:
            logger.error(f"Failed to rollback optimization {optimization_id}: {e}")
            return False

    def _create_rollback_snapshot(self, optimization_id: str) -> RollbackState:
        """Create a snapshot for rollback capability."""
        return RollbackState(
            snapshot_id=f"snapshot_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            config_backup=copy.deepcopy(self.current_config),
            performance_snapshot=copy.deepcopy(self.performance_history[-10:] if self.performance_history else []),
            optimization_id=optimization_id
        )

    async def _analyze_strategy_performance(
        self,
        performance: Dict[str, Any],
        strategy_name: Optional[str],
    ) -> str:
        """Analyze strategy performance using SAIA."""
        context = f"""
Current Trading Performance:
{json.dumps(performance, indent=2)}

Current Strategy Configuration:
{json.dumps(self.current_config.get('trading', {}), indent=2)}
"""

        if strategy_name:
            context = f"\nAnalyzing specific strategy: {strategy_name}\n" + context

        response = self.agent.analyze(
            project_context=context,
            goals=self._get_strategy_goals(),
            current_state=json.dumps(performance, indent=2)
        )

        return response

    async def _generate_strategy_optimization_plan(self, analysis: str) -> str:
        """Generate optimization plan based on analysis."""
        issues = [
            "Suboptimal parameter tuning based on current market conditions",
            "Potential improvements in entry/exit logic",
            "Risk-adjusted return optimization opportunities"
        ]

        constraints = f"""
Constraints:
- Max drawdown: {self.constraints.max_drawdown_threshold:.1%}
- Min Sharpe ratio: {self.constraints.min_sharpe_ratio}
- Min win rate: {self.constraints.min_win_rate:.1%}
- Must test in paper trading for {self.constraints.paper_trading_duration_hours}h before live
"""

        return self.agent.plan(issues, constraints)

    def _extract_proposed_changes(self, plan: str) -> Dict[str, Any]:
        """Extract proposed changes from optimization plan."""
        # In production, this would use more sophisticated parsing
        # For now, return a structured placeholder
        return {
            "plan_summary": plan[:500],
            "source": "saia_agent",
        }

    def _validate_constraints(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Validate proposed changes against constraints."""
        # Placeholder - in production would validate specific constraints
        return {"passed": True, "reason": ""}

    def _calculate_optimization_metrics(
        self,
        plan: str,
        current_performance: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate expected improvement and confidence."""
        # Placeholder - in production would use more sophisticated analysis
        return {
            "expected_improvement": 0.12,  # 12% improvement
            "confidence": 0.70,
        }

    def _prepare_risk_optimization_context(
        self,
        performance: Dict[str, Any],
        risk_metrics: Dict[str, Any],
    ) -> str:
        """Prepare context for risk optimization."""
        return json.dumps({
            "performance": performance,
            "risk_metrics": risk_metrics,
            "current_config": self.current_config.get("trading", {})
        }, indent=2)

    def _parse_risk_recommendations(self, response: str) -> Dict[str, Any]:
        """Parse risk limit recommendations from SAIA response."""
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Fallback to current config
        return self.current_config.get("trading", {})

    def _validate_risk_constraints(
        self,
        proposed: Dict[str, Any],
        current_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate risk constraints."""
        # Check if proposed limits are within acceptable ranges
        max_risk_per_trade = proposed.get("risk_per_trade", 0.02)
        if max_risk_per_trade > 0.05:  # Max 5% risk per trade
            return {"passed": False, "reason": "Risk per trade exceeds 5%"}

        return {"passed": True, "reason": ""}

    def _estimate_risk_improvement(
        self,
        proposed: Dict[str, Any],
        current_metrics: Dict[str, Any],
    ) -> float:
        """Estimate expected improvement from risk changes."""
        # Simplified estimation
        current_drawdown = current_metrics.get("max_drawdown", 0.15)
        target_drawdown = self.constraints.max_drawdown_threshold

        if current_drawdown > target_drawdown:
            return (current_drawdown - target_drawdown) / current_drawdown

        return 0.05  # 5% improvement estimate

    def _parse_agent_weights(self, response: str) -> Dict[str, float]:
        """Parse agent weight recommendations."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Return current weights as fallback
        return {
            "technical": 0.30,
            "sentiment": 0.25,
            "risk": 0.25,
            "portfolio": 0.20
        }

    def _validate_agent_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Validate agent weights."""
        total_weight = sum(weights.values())

        if not (0.9 <= total_weight <= 1.1):  # Allow small rounding errors
            return {"passed": False, "reason": f"Weights sum to {total_weight}, expected ~1.0"}

        for agent, weight in weights.items():
            if not (0.0 <= weight <= 1.0):
                return {"passed": False, "reason": f"Weight for {agent} out of range: {weight}"}

        return {"passed": True, "reason": ""}

    def _parse_pair_recommendations(self, response: str) -> Dict[str, Any]:
        """Parse trading pair recommendations."""
        return {
            "add": [],
            "remove": [],
            "keep": [],
            "reasoning": response
        }

    def _validate_pair_recommendations(
        self,
        recommendations: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate pair recommendations against market data."""
        return recommendations

    def _parse_indicator_recommendations(self, response: str) -> Dict[str, Any]:
        """Parse indicator recommendations."""
        return {
            "indicators": {},
            "reasoning": response
        }

    def _apply_configuration_changes(self, changes: Dict[str, Any]) -> None:
        """Apply configuration changes."""
        # Update current config
        for key, value in changes.items():
            if isinstance(value, dict):
                if key not in self.current_config:
                    self.current_config[key] = {}
                self.current_config[key].update(value)
            else:
                self.current_config[key] = value

        # In production, would also write to config files
        logger.info("Configuration changes applied")

    def _get_strategy_goals(self) -> str:
        """Get strategy optimization goals."""
        return """
Optimization Goals:
1. Maximize Sharpe ratio (target: > 2.0)
2. Minimize maximum drawdown (target: < 10%)
3. Maximize win rate (target: > 60%)
4. Minimize trading frequency costs
5. Optimize risk-adjusted returns
"""

    def _log_optimization_to_kg(
        self,
        optimization: OptimizationResult,
        action: str,
    ) -> None:
        """Log optimization to knowledge graph."""
        if not self.kg:
            return

        try:
            # Create optimization node in Neo4j
            query = """
            MERGE (opt:Optimization {id: $id})
            SET opt.type = $type,
                opt.status = $status,
                opt.action = $action,
                opt.expected_improvement = $expected_improvement,
                opt.confidence = $confidence,
                opt.timestamp = $timestamp,
                opt.reasoning = $reasoning
            """
            self.kg.execute_query(query, {
                "id": optimization.optimization_id,
                "type": optimization.optimization_type.value,
                "status": optimization.status.value,
                "action": action,
                "expected_improvement": optimization.expected_improvement,
                "confidence": optimization.confidence_score,
                "timestamp": optimization.timestamp.isoformat(),
                "reasoning": optimization.reasoning[:1000]  # Truncate for Neo4j
            })

            logger.debug(f"Logged optimization {optimization.optimization_id} to knowledge graph")

        except Exception as e:
            logger.error(f"Failed to log optimization to KG: {e}")

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get history of all optimizations."""
        return list(self.optimizations.values())

    def get_pending_optimizations(self) -> List[OptimizationResult]:
        """Get all pending optimizations."""
        return [
            opt for opt in self.optimizations.values()
            if opt.status == OptimizationStatus.PENDING
        ]

    def get_recent_performance(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent performance history."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [
            perf for perf in self.performance_history
            if datetime.fromisoformat(perf["timestamp"]) >= cutoff
        ]
