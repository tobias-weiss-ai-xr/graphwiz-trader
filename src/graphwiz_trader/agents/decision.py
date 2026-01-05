"""Decision engine for aggregating and resolving multi-agent signals."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from graphwiz_trader.agents.trading_agents import (
    AgentDecision,
    AgentPerformance,
    TradingAgent,
    TradingSignal,
)


class ConsensusMethod(Enum):
    """Methods for reaching consensus among agents."""

    MAJORITY_VOTE = "majority_vote"  # Simple majority
    WEIGHTED_VOTE = "weighted_vote"  # Weighted by agent performance
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weighted by confidence
    BEST_PERFORMER = "best_performer"  # Follow best performing agent
    UNANIMOUS = "unanimous"  # Only act if all agree


class ConflictResolution(Enum):
    """Methods for resolving conflicts between agents."""

    HIGH_CONFIDENCE_WINS = "high_confidence"  # Highest confidence wins
    BEST_PERFORMER_WINS = "best_performer"  # Best performing agent wins
    RISK_AVERSE = "risk_averse"  # Default to HOLD in conflicts
    MAJORITY_RULES = "majority"  # Follow majority
    MANUAL_REVIEW = "manual"  # Flag for manual review


@dataclass
class DecisionResult:
    """Final trading decision after aggregation.

    Attributes:
        signal: Final trading signal
        confidence: Overall confidence in the decision
        reasoning: Explanation of the decision
        participating_agents: Agents that participated
        agent_signals: Individual agent signals
        consensus_method: Method used for consensus
        conflict_score: Degree of conflict (0=unanimous, 1=maximum conflict)
        metadata: Additional decision metadata
        timestamp: When the decision was made
    """

    signal: TradingSignal
    confidence: float
    reasoning: str
    participating_agents: List[str]
    agent_signals: Dict[str, AgentDecision]
    consensus_method: ConsensusMethod
    conflict_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "participating_agents": self.participating_agents,
            "agent_signals": {
                name: decision.to_dict() for name, decision in self.agent_signals.items()
            },
            "consensus_method": self.consensus_method.value,
            "conflict_score": self.conflict_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class DecisionEngine:
    """Engine for aggregating signals from multiple trading agents.

    The DecisionEngine is responsible for:
    - Collecting signals from multiple agents
    - Applying weighted voting/consensus mechanisms
    - Resolving conflicts when agents disagree
    - Determining final action (BUY/SELL/HOLD)
    """

    def __init__(
        self,
        consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_VOTE,
        conflict_resolution: ConflictResolution = ConflictResolution.HIGH_CONFIDENCE_WINS,
        min_confidence_threshold: float = 0.6,
        enable_disagreement_tracking: bool = True,
    ):
        """Initialize decision engine.

        Args:
            consensus_method: Method for reaching consensus
            conflict_resolution: Method for resolving conflicts
            min_confidence_threshold: Minimum confidence for action
            enable_disagreement_tracking: Whether to track agent disagreements
        """
        self.consensus_method = consensus_method
        self.conflict_resolution = conflict_resolution
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_disagreement_tracking = enable_disagreement_tracking

        # Historical tracking
        self.decision_history: List[DecisionResult] = []
        self.agent_disagreements: Dict[str, int] = {}

        logger.info(
            f"Initialized DecisionEngine with consensus={consensus_method.value}, "
            f"conflict_resolution={conflict_resolution.value}"
        )

    async def make_decision(
        self,
        agent_decisions: Dict[str, AgentDecision],
        agent_performances: Dict[str, AgentPerformance],
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        """Make final trading decision from multiple agent signals.

        Args:
            agent_decisions: Dictionary of agent decisions
            agent_performances: Dictionary of agent performance metrics
            context: Additional context for decision making

        Returns:
            Final decision result
        """
        if not agent_decisions:
            logger.warning("No agent decisions provided")
            return self._create_hold_decision("No agent decisions available")

        # Filter out disabled agents or HOLD signals with low confidence
        active_decisions = {
            name: decision
            for name, decision in agent_decisions.items()
            if decision.signal != TradingSignal.HOLD or decision.confidence > 0.7
        }

        if not active_decisions:
            return self._create_hold_decision(
                "All agents recommend HOLD or have low confidence", agent_decisions
            )

        # Calculate conflict score
        conflict_score = self._calculate_conflict_score(active_decisions)

        # Apply consensus method
        if self.consensus_method == ConsensusMethod.MAJORITY_VOTE:
            signal, confidence = self._majority_vote(active_decisions)
        elif self.consensus_method == ConsensusMethod.WEIGHTED_VOTE:
            signal, confidence = self._weighted_vote(active_decisions, agent_performances)
        elif self.consensus_method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            signal, confidence = self._confidence_weighted_vote(active_decisions)
        elif self.consensus_method == ConsensusMethod.BEST_PERFORMER:
            signal, confidence = self._best_performer_vote(active_decisions, agent_performances)
        elif self.consensus_method == ConsensusMethod.UNANIMOUS:
            signal, confidence = self._unanimous_vote(active_decisions)
        else:
            signal, confidence = self._weighted_vote(active_decisions, agent_performances)

        # Apply conflict resolution if needed
        if conflict_score > 0.5:  # Significant conflict
            signal, confidence = self._resolve_conflict(
                signal, confidence, active_decisions, agent_performances
            )

        # Apply minimum confidence threshold
        if confidence < self.min_confidence_threshold:
            signal = TradingSignal.HOLD
            confidence = max(confidence, 0.5)

        # Build reasoning
        reasoning = self._build_reasoning(active_decisions, signal, confidence, conflict_score)

        # Create result
        result = DecisionResult(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            participating_agents=list(active_decisions.keys()),
            agent_signals=active_decisions,
            consensus_method=self.consensus_method,
            conflict_score=conflict_score,
            metadata={
                "total_agents": len(agent_decisions),
                "active_agents": len(active_decisions),
                "conflict_resolution_applied": conflict_score > 0.5,
            },
        )

        # Track decision
        self.decision_history.append(result)
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

        # Track disagreements
        if self.enable_disagreement_tracking:
            self._track_disagreements(active_decisions)

        logger.info(
            f"Decision: {signal.value} (confidence={confidence:.2f}, "
            f"conflict={conflict_score:.2f}, agents={len(active_decisions)})"
        )

        return result

    def _calculate_conflict_score(self, decisions: Dict[str, AgentDecision]) -> float:
        """Calculate conflict score between agent decisions.

        Args:
            decisions: Agent decisions

        Returns:
            Conflict score (0=unanimous, 1=maximum conflict)
        """
        if len(decisions) <= 1:
            return 0.0

        signals = [d.signal for d in decisions.values()]
        buy_count = sum(1 for s in signals if s == TradingSignal.BUY)
        sell_count = sum(1 for s in signals if s == TradingSignal.SELL)
        hold_count = sum(1 for s in signals if s == TradingSignal.HOLD)

        total = len(signals)

        # If all agree, no conflict
        if buy_count == total or sell_count == total or hold_count == total:
            return 0.0

        # Calculate entropy-based conflict
        proportions = [buy_count / total, sell_count / total, hold_count / total]

        # Remove zeros
        proportions = [p for p in proportions if p > 0]

        # Calculate normalized entropy
        entropy = -sum(p * np.log(p) for p in proportions)
        max_entropy = np.log(min(3, total))  # Max entropy for 3 outcomes

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _majority_vote(self, decisions: Dict[str, AgentDecision]) -> Tuple[TradingSignal, float]:
        """Simple majority vote.

        Args:
            decisions: Agent decisions

        Returns:
            Tuple of (signal, confidence)
        """
        signals = [d.signal for d in decisions.values()]
        buy_count = sum(1 for s in signals if s == TradingSignal.BUY)
        sell_count = sum(1 for s in signals if s == TradingSignal.SELL)
        hold_count = sum(1 for s in signals if s == TradingSignal.HOLD)

        # Find majority
        counts = {
            TradingSignal.BUY: buy_count,
            TradingSignal.SELL: sell_count,
            TradingSignal.HOLD: hold_count,
        }

        max_signal = max(counts, key=counts.get)
        max_count = counts[max_signal]
        total = len(signals)

        # Confidence based on agreement level
        confidence = max_count / total if total > 0 else 0.5

        return max_signal, confidence

    def _weighted_vote(
        self, decisions: Dict[str, AgentDecision], performances: Dict[str, AgentPerformance]
    ) -> Tuple[TradingSignal, float]:
        """Weighted vote based on agent performance.

        Args:
            decisions: Agent decisions
            performances: Agent performance metrics

        Returns:
            Tuple of (signal, confidence)
        """
        weights = {}
        for name in decisions.keys():
            perf = performances.get(name)
            if perf:
                # Weight based on accuracy and profit factor
                weight = perf.accuracy * perf.profit_factor
                weights[name] = max(weight, 0.1)  # Minimum weight
            else:
                weights[name] = 0.5  # Default weight for new agents

        # Calculate weighted votes
        buy_weight = sum(
            weights[name] * decision.confidence
            for name, decision in decisions.items()
            if decision.signal == TradingSignal.BUY
        )
        sell_weight = sum(
            weights[name] * decision.confidence
            for name, decision in decisions.items()
            if decision.signal == TradingSignal.SELL
        )
        hold_weight = sum(
            weights[name] * decision.confidence
            for name, decision in decisions.items()
            if decision.signal == TradingSignal.HOLD
        )

        # Find maximum weight
        weights_dict = {
            TradingSignal.BUY: buy_weight,
            TradingSignal.SELL: sell_weight,
            TradingSignal.HOLD: hold_weight,
        }

        max_signal = max(weights_dict, key=weights_dict.get)
        max_weight = weights_dict[max_signal]
        total_weight = sum(weights_dict.values())

        confidence = max_weight / total_weight if total_weight > 0 else 0.5

        return max_signal, confidence

    def _confidence_weighted_vote(
        self, decisions: Dict[str, AgentDecision]
    ) -> Tuple[TradingSignal, float]:
        """Vote weighted by agent confidence levels.

        Args:
            decisions: Agent decisions

        Returns:
            Tuple of (signal, confidence)
        """
        buy_confidence = sum(
            d.confidence for d in decisions.values() if d.signal == TradingSignal.BUY
        )
        sell_confidence = sum(
            d.confidence for d in decisions.values() if d.signal == TradingSignal.SELL
        )
        hold_confidence = sum(
            d.confidence for d in decisions.values() if d.signal == TradingSignal.HOLD
        )

        confidences = {
            TradingSignal.BUY: buy_confidence,
            TradingSignal.SELL: sell_confidence,
            TradingSignal.HOLD: hold_confidence,
        }

        max_signal = max(confidences, key=confidences.get)
        max_conf = confidences[max_signal]
        total_conf = sum(confidences.values())

        confidence = max_conf / total_conf if total_conf > 0 else 0.5

        return max_signal, confidence

    def _best_performer_vote(
        self, decisions: Dict[str, AgentDecision], performances: Dict[str, AgentPerformance]
    ) -> Tuple[TradingSignal, float]:
        """Follow the best performing agent.

        Args:
            decisions: Agent decisions
            performances: Agent performance metrics

        Returns:
            Tuple of (signal, confidence)
        """
        # Find best performing agent
        best_agent = None
        best_score = -1

        for name in decisions.keys():
            perf = performances.get(name)
            if perf:
                # Composite score: accuracy * profit factor
                score = perf.accuracy * perf.profit_factor
                if score > best_score:
                    best_score = score
                    best_agent = name

        # If no performance data, fall back to confidence-weighted
        if best_agent is None:
            return self._confidence_weighted_vote(decisions)

        best_decision = decisions[best_agent]
        confidence = best_decision.confidence * 0.9  # Slight reduction for following single agent

        return best_decision.signal, confidence

    def _unanimous_vote(self, decisions: Dict[str, AgentDecision]) -> Tuple[TradingSignal, float]:
        """Only act if all agents agree.

        Args:
            decisions: Agent decisions

        Returns:
            Tuple of (signal, confidence)
        """
        signals = [d.signal for d in decisions.values()]

        # Check if unanimous
        if len(set(signals)) == 1:
            # All agree
            signal = signals[0]
            # High confidence due to unanimity
            avg_confidence = sum(d.confidence for d in decisions.values()) / len(decisions)
            confidence = min(avg_confidence * 1.1, 0.95)
            return signal, confidence
        else:
            # Not unanimous - default to HOLD
            return TradingSignal.HOLD, 0.5

    def _resolve_conflict(
        self,
        initial_signal: TradingSignal,
        initial_confidence: float,
        decisions: Dict[str, AgentDecision],
        performances: Dict[str, AgentPerformance],
    ) -> Tuple[TradingSignal, float]:
        """Resolve conflicts when agents disagree.

        Args:
            initial_signal: Initial signal from consensus
            initial_confidence: Initial confidence
            decisions: Agent decisions
            performances: Agent performances

        Returns:
            Tuple of (resolved_signal, resolved_confidence)
        """
        if self.conflict_resolution == ConflictResolution.HIGH_CONFIDENCE_WINS:
            # Find highest confidence decision
            max_conf_decision = max(decisions.values(), key=lambda d: d.confidence)
            return max_conf_decision.signal, max_conf_decision.confidence * 0.9

        elif self.conflict_resolution == ConflictResolution.BEST_PERFORMER_WINS:
            # Follow best performer
            signal, confidence = self._best_performer_vote(decisions, performances)
            return signal, confidence * 0.85

        elif self.conflict_resolution == ConflictResolution.RISK_AVERSE:
            # Default to HOLD with moderate confidence
            return TradingSignal.HOLD, max(initial_confidence * 0.7, 0.5)

        elif self.conflict_resolution == ConflictResolution.MAJORITY_RULES:
            # Use majority vote
            return self._majority_vote(decisions)

        else:  # MANUAL_REVIEW
            # Flag for review but still make decision
            return TradingSignal.HOLD, 0.5

    def _build_reasoning(
        self,
        decisions: Dict[str, AgentDecision],
        final_signal: TradingSignal,
        final_confidence: float,
        conflict_score: float,
    ) -> str:
        """Build reasoning explanation for the decision.

        Args:
            decisions: Agent decisions
            final_signal: Final signal
            final_confidence: Final confidence
            conflict_score: Conflict score

        Returns:
            Reasoning string
        """
        parts = []

        # Count signals
        signal_counts = {}
        for decision in decisions.values():
            sig = decision.signal.value
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        parts.append(f"Decision: {final_signal.value} (confidence: {final_confidence:.2f})")

        # Agent breakdown
        agent_summary = ", ".join(
            [f"{name}:{dec.signal.value}({dec.confidence:.2f})" for name, dec in decisions.items()]
        )
        parts.append(f"Agents: {agent_summary}")

        # Conflict level
        if conflict_score > 0.7:
            parts.append("High agent disagreement")
        elif conflict_score > 0.4:
            parts.append("Moderate agent disagreement")
        else:
            parts.append("Good agent consensus")

        # Top reasons from supporting agents
        supporting_agents = [
            (name, dec) for name, dec in decisions.items() if dec.signal == final_signal
        ]

        if supporting_agents:
            top_supporters = sorted(supporting_agents, key=lambda x: x[1].confidence, reverse=True)[
                :2
            ]

            reasons = []
            for name, dec in top_supporters:
                # Extract first sentence of reasoning
                reason_first_line = dec.reasoning.split(".")[0]
                reasons.append(f"{name}: {reason_first_line}")

            if reasons:
                parts.append(f"Key reasons: {'; '.join(reasons)}")

        return " | ".join(parts)

    def _create_hold_decision(
        self, reason: str, agent_decisions: Optional[Dict[str, AgentDecision]] = None
    ) -> DecisionResult:
        """Create a HOLD decision.

        Args:
            reason: Reason for HOLD
            agent_decisions: Optional agent decisions

        Returns:
            HOLD decision result
        """
        return DecisionResult(
            signal=TradingSignal.HOLD,
            confidence=0.5,
            reasoning=reason,
            participating_agents=list(agent_decisions.keys()) if agent_decisions else [],
            agent_signals=agent_decisions or {},
            consensus_method=self.consensus_method,
            conflict_score=0.0,
        )

    def _track_disagreements(self, decisions: Dict[str, AgentDecision]) -> None:
        """Track agent disagreements for performance analysis.

        Args:
            decisions: Agent decisions
        """
        signals = [d.signal for d in decisions.values()]

        # Check for disagreement
        if len(set(signals)) > 1:
            for name in decisions.keys():
                self.agent_disagreements[name] = self.agent_disagreements.get(name, 0) + 1

    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about decisions made.

        Returns:
            Dictionary of decision statistics
        """
        if not self.decision_history:
            return {"total_decisions": 0}

        signal_counts = {}
        confidences = []
        conflict_scores = []

        for decision in self.decision_history:
            sig = decision.signal.value
            signal_counts[sig] = signal_counts.get(sig, 0) + 1
            confidences.append(decision.confidence)
            conflict_scores.append(decision.conflict_score)

        return {
            "total_decisions": len(self.decision_history),
            "signal_distribution": signal_counts,
            "average_confidence": np.mean(confidences) if confidences else 0.0,
            "average_conflict": np.mean(conflict_scores) if conflict_scores else 0.0,
            "agent_disagreements": self.agent_disagreements.copy(),
        }
