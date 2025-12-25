"""Agent orchestrator for coordinating multiple AI agents."""

from loguru import logger
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class AgentDecision:
    """Decision from an individual agent."""
    agent_name: str
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConsensusDecision:
    """Consensus decision from all agents."""
    action: str
    confidence: float
    reasoning: str
    individual_decisions: List[AgentDecision]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseAgent:
    """Base class for AI agents."""

    def __init__(self, name: str, config: Dict[str, Any], knowledge_graph):
        """Initialize agent.

        Args:
            name: Agent name
            config: Agent configuration
            knowledge_graph: Knowledge graph instance
        """
        self.name = name
        self.config = config
        self.kg = knowledge_graph
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 1000)

    def get_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Get trading decision from this agent.

        Args:
            context: Trading context information

        Returns:
            AgentDecision with recommendation
        """
        raise NotImplementedError("Subclasses must implement get_decision")

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for the AI model.

        Args:
            context: Trading context

        Returns:
            Prompt string
        """
        raise NotImplementedError("Subclasses must implement _build_prompt")


class TechnicalAnalysisAgent(BaseAgent):
    """Agent for technical analysis decisions."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build technical analysis prompt."""
        prompt = f"""You are a technical analysis expert. Analyze the following market data and provide a trading recommendation.

Market Data:
- Symbol: {context.get('symbol', 'N/A')}
- Current Price: {context.get('current_price', 'N/A')}
- Price Change (24h): {context.get('price_change_24h', 'N/A')}%

Technical Indicators:
"""

        # Add technical indicators if available
        indicators = context.get('technical_indicators', {})
        if indicators:
            if 'rsi' in indicators:
                rsi = indicators['rsi'].get('latest', 'N/A')
                prompt += f"- RSI: {rsi}\n"
            if 'macd' in indicators:
                macd_signal = indicators['macd'].get('signal', 'N/A')
                prompt += f"- MACD Signal: {macd_signal}\n"
            if 'bollinger_bands' in indicators:
                bb_signal = indicators['bollinger_bands'].get('signal', 'N/A')
                prompt += f"- Bollinger Bands Signal: {bb_signal}\n"
            if 'overall_signal' in indicators:
                overall = indicators['overall_signal'].get('signal', 'N/A')
                prompt += f"- Overall Technical Signal: {overall}\n"

        prompt += """
Provide your recommendation in the following JSON format:
{
    "action": "buy" or "sell" or "hold",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your analysis"
}
"""
        return prompt

    def get_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Get technical analysis decision."""
        try:
            # Check if we have historical decisions in knowledge graph
            kg_context = self._get_kg_context(context.get('symbol', ''))
            if kg_context:
                context['kg_context'] = kg_context

            # For now, return a rule-based decision
            # In production, this would call OpenAI/Anthropic API
            indicators = context.get('technical_indicators', {})
            overall_signal = indicators.get('overall_signal', {}).get('signal', 'neutral')

            # Map technical signal to action
            action_map = {
                'strong_buy': 'buy',
                'buy': 'buy',
                'neutral': 'hold',
                'sell': 'sell',
                'strong_sell': 'sell'
            }

            action = action_map.get(overall_signal, 'hold')
            confidence = indicators.get('overall_signal', {}).get('confidence', 0.5)
            reasoning = f"Technical analysis indicates {overall_signal} signal"

            return AgentDecision(
                agent_name=self.name,
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                metadata={"signal": overall_signal}
            )

        except Exception as e:
            logger.error("Error in technical analysis agent: {}", e)
            return AgentDecision(
                agent_name=self.name,
                action="hold",
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )

    def _get_kg_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get historical context from knowledge graph."""
        try:
            if self.kg is None:
                return None

            cypher = """
            MATCH (t:Trade {symbol: $symbol})
            WITH t ORDER BY t.timestamp DESC LIMIT 10
            RETURN avg(t.amount) as avg_amount,
                   count(t) as trade_count,
                   max(t.timestamp) as last_trade
            """

            result = self.kg.query(cypher, {"symbol": symbol})
            if result and len(result) > 0:
                return result[0]
            return None

        except Exception as e:
            logger.warning("Failed to get KG context: {}", e)
            return None


class SentimentAnalysisAgent(BaseAgent):
    """Agent for sentiment analysis decisions."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build sentiment analysis prompt."""
        return f"""Analyze market sentiment and provide a trading recommendation.

Market Context:
{json.dumps(context, indent=2)}

Consider news sentiment, social media trends, and market sentiment indicators.
"""

    def get_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Get sentiment analysis decision."""
        # Stub implementation - would integrate with sentiment APIs
        # For now, return neutral with low confidence
        return AgentDecision(
            agent_name=self.name,
            action="hold",
            confidence=0.3,
            reasoning="Sentiment analysis not yet implemented - using neutral stance",
            metadata={"status": "stub"}
        )


class RiskManagementAgent(BaseAgent):
    """Agent for risk management decisions."""

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build risk management prompt."""
        return f"""Analyze risk factors and provide risk-adjusted recommendation.

Context:
{json.dumps(context, indent=2)}
"""

    def get_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Get risk management decision."""
        # Check portfolio risk
        current_positions = context.get('positions', [])
        max_position_size = context.get('max_position_size', 0.1)

        risk_level = "low"
        if len(current_positions) >= 5:
            risk_level = "medium"
        if len(current_positions) >= 8:
            risk_level = "high"

        if risk_level == "high":
            return AgentDecision(
                agent_name=self.name,
                action="hold",
                confidence=0.8,
                reasoning=f"Portfolio risk level is {risk_level} - recommend holding",
                metadata={"risk_level": risk_level, "position_count": len(current_positions)}
            )
        elif risk_level == "medium":
            return AgentDecision(
                agent_name=self.name,
                action="hold",
                confidence=0.5,
                reasoning=f"Portfolio risk level is {risk_level} - cautious approach",
                metadata={"risk_level": risk_level}
            )
        else:
            return AgentDecision(
                agent_name=self.name,
                action="buy",  # Allow trades when risk is low
                confidence=0.6,
                reasoning=f"Portfolio risk level is {risk_level} - within limits",
                metadata={"risk_level": risk_level}
            )


class AgentOrchestrator:
    """Orchestrates multiple AI agents for trading decisions."""

    def __init__(self, config: Dict[str, Any], knowledge_graph):
        """Initialize agent orchestrator.

        Args:
            config: Agent configuration
            knowledge_graph: Knowledge graph instance
        """
        self.config = config
        self.kg = knowledge_graph
        self.agents: Dict[str, BaseAgent] = {}
        self.decision_history: List[ConsensusDecision] = []
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """Initialize AI agents based on configuration."""
        for agent_name, agent_config in self.config.items():
            if not agent_config.get("enabled", False):
                continue

            # Create agent instance based on type
            agent = self._create_agent(agent_name, agent_config)
            if agent:
                self.agents[agent_name] = agent
                logger.info("Initialized agent: {} with model {}", agent_name, agent_config.get("model", "unknown"))

    def _create_agent(self, name: str, config: Dict[str, Any]) -> Optional[BaseAgent]:
        """Create agent instance based on name/type.

        Args:
            name: Agent name
            config: Agent configuration

        Returns:
            Agent instance or None
        """
        # Map agent names to agent classes
        agent_classes = {
            "technical": TechnicalAnalysisAgent,
            "sentiment": SentimentAnalysisAgent,
            "risk": RiskManagementAgent
        }

        agent_class = agent_classes.get(name)
        if agent_class:
            return agent_class(name, config, self.kg)

        # Default to technical analysis agent for unknown types
        logger.warning("Unknown agent type {}, using TechnicalAnalysisAgent", name)
        return TechnicalAnalysisAgent(name, config, self.kg)

    def get_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading decision from all agents.

        Args:
            context: Trading context information

        Returns:
            Trading decision dictionary with consensus
        """
        if not self.agents:
            logger.warning("No agents enabled")
            return {
                "action": "hold",
                "confidence": 0.0,
                "reasoning": "No agents available for decision",
                "agent_count": 0
            }

        # Get decisions from all agents
        individual_decisions = []
        for agent_name, agent in self.agents.items():
            try:
                decision = agent.get_decision(context)
                individual_decisions.append(decision)
                logger.debug("Agent {} recommended: {} (confidence: {})",
                           agent_name, decision.action, decision.confidence)
            except Exception as e:
                logger.error("Error getting decision from agent {}: {}", agent_name, e)

        # Calculate consensus
        consensus = self._calculate_consensus(individual_decisions)

        # Store in history
        consensus_decision = ConsensusDecision(
            action=consensus["action"],
            confidence=consensus["confidence"],
            reasoning=consensus["reasoning"],
            individual_decisions=individual_decisions
        )
        self.decision_history.append(consensus_decision)

        # Store in knowledge graph if available
        self._store_decision_in_kg(consensus_decision, context)

        logger.info("Consensus decision: {} with confidence {}", consensus["action"], consensus["confidence"])

        return {
            "action": consensus["action"],
            "confidence": consensus["confidence"],
            "reasoning": consensus["reasoning"],
            "agent_count": len(individual_decisions),
            "individual_decisions": [
                {
                    "agent": d.agent_name,
                    "action": d.action,
                    "confidence": d.confidence
                }
                for d in individual_decisions
            ]
        }

    def _calculate_consensus(self, decisions: List[AgentDecision]) -> Dict[str, Any]:
        """Calculate consensus from multiple agent decisions.

        Args:
            decisions: List of individual agent decisions

        Returns:
            Dictionary with consensus action and confidence
        """
        if not decisions:
            return {"action": "hold", "confidence": 0.0, "reasoning": "No decisions available"}

        # Count votes
        votes = {"buy": 0, "sell": 0, "hold": 0}
        weighted_votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

        for decision in decisions:
            votes[decision.action] += 1
            weighted_votes[decision.action] += decision.confidence

        # Find action with most votes
        winning_action = max(votes, key=votes.get)

        # Calculate confidence based on vote distribution
        total_votes = sum(votes.values())
        vote_confidence = votes[winning_action] / total_votes if total_votes > 0 else 0.0

        # Also consider average confidence of winning action
        if votes[winning_action] > 0:
            avg_confidence = weighted_votes[winning_action] / votes[winning_action]
        else:
            avg_confidence = 0.0

        # Combine vote confidence and average confidence
        final_confidence = (vote_confidence + avg_confidence) / 2

        # Build reasoning
        reasoning_parts = []
        for decision in decisions:
            reasoning_parts.append(f"{decision.agent_name}: {decision.action} ({decision.confidence:.2f})")

        reasoning = f"Consensus: {winning_action} ({vote_confidence:.1%} vote majority). "
        reasoning += f"Agents: {', '.join(reasoning_parts)}"

        return {
            "action": winning_action,
            "confidence": round(final_confidence, 2),
            "reasoning": reasoning
        }

    def _store_decision_in_kg(self, decision: ConsensusDecision, context: Dict[str, Any]) -> None:
        """Store decision in knowledge graph.

        Args:
            decision: Consensus decision
            context: Trading context
        """
        try:
            if self.kg is None:
                return

            cypher = """
            MERGE (d:Decision {timestamp: datetime($timestamp)})
            SET d.action = $action,
                d.confidence = $confidence,
                d.reasoning = $reasoning,
                d.symbol = $symbol
            """

            self.kg.write(cypher, {
                "timestamp": decision.timestamp.isoformat(),
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "symbol": context.get("symbol", "unknown")
            })

        except Exception as e:
            logger.warning("Failed to store decision in knowledge graph: {}", e)

    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get decision history.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of decision dictionaries
        """
        recent_decisions = self.decision_history[-limit:] if len(self.decision_history) > limit else self.decision_history

        return [
            {
                "action": d.action,
                "confidence": d.confidence,
                "reasoning": d.reasoning,
                "timestamp": d.timestamp.isoformat(),
                "agent_count": len(d.individual_decisions)
            }
            for d in recent_decisions
        ]

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents.

        Returns:
            Dictionary with agent information
        """
        return {
            agent_name: {
                "model": agent.model,
                "temperature": agent.temperature,
                "max_tokens": agent.max_tokens
            }
            for agent_name, agent in self.agents.items()
        }
