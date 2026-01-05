"""Comprehensive tests for AI trading agents system."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from graphwiz_trader.agents import (
    AgentOrchestrator,
    TechnicalAnalysisAgent,
    SentimentAnalysisAgent,
    RiskManagementAgent,
    MomentumAgent,
    MeanReversionAgent,
    TradingSignal,
    AgentDecision,
    DecisionEngine,
    ConsensusMethod,
    ConflictResolution
)


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "symbol": "BTC/USDT",
        "close": 50000.0,
        "open": 49500.0,
        "high": 50500.0,
        "low": 49200.0,
        "volume": 1000.0,
        "price_history": [49000, 49200, 49500, 49800, 50000]
    }


@pytest.fixture
def sample_indicators():
    """Sample technical indicators for testing."""
    return {
        "RSI": {"value": 65.0},
        "MACD": {"macd": 100, "signal": 95, "histogram": 5},
        "BB": {"upper": 51000, "lower": 49000, "middle": 50000},
        "EMA": {"short": 50100, "long": 49800},
        "ROC": {"value": 2.0},
        "ADX": {"value": 25, "di_plus": 22, "di_minus": 18},
        "volume_ma": {"value": 800},
        "zscore": {"value": 1.8},
        "Stochastic": {"k": 75, "d": 70},
        "SMA": {"value": 49900},
        "volatility": {"value": 0.03}
    }


@pytest.fixture
def agents_config():
    """Configuration for agents."""
    return {
        "technical": {"enabled": True, "model": "gpt-4", "min_confidence": 0.6},
        "sentiment": {"enabled": True, "model": "gpt-4", "min_confidence": 0.6},
        "risk": {"enabled": True, "model": "gpt-4", "min_confidence": 0.7},
        "momentum": {"enabled": True},
        "mean_reversion": {"enabled": True},
        "consensus_method": "weighted_vote",
        "conflict_resolution": "high_confidence"
    }


class TestTechnicalAnalysisAgent:
    """Test suite for TechnicalAnalysisAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = TechnicalAnalysisAgent(
            name="technical",
            config={"enabled": True, "min_confidence": 0.6},
            knowledge_graph=None
        )

        assert agent.name == "technical"
        assert agent.enabled is True
        assert agent.min_confidence == 0.6

    @pytest.mark.asyncio
    async def test_analyze_rsi_oversold(self, sample_market_data):
        """Test analysis with oversold RSI."""
        agent = TechnicalAnalysisAgent(
            name="technical",
            config={"enabled": True},
            knowledge_graph=None
        )

        indicators = {"RSI": {"value": 25.0}}
        decision = await agent.analyze(sample_market_data, indicators)

        assert decision.signal in [TradingSignal.BUY, TradingSignal.HOLD]
        assert decision.confidence >= 0.0
        assert len(decision.reasoning) > 0

    @pytest.mark.asyncio
    async def test_analyze_rsi_overbought(self, sample_market_data):
        """Test analysis with overbought RSI."""
        agent = TechnicalAnalysisAgent(
            name="technical",
            config={"enabled": True},
            knowledge_graph=None
        )

        indicators = {"RSI": {"value": 75.0}}
        decision = await agent.analyze(sample_market_data, indicators)

        assert decision.signal in [TradingSignal.SELL, TradingSignal.HOLD]

    @pytest.mark.asyncio
    async def test_analyze_macd_bullish(self, sample_market_data):
        """Test analysis with bullish MACD."""
        agent = TechnicalAnalysisAgent(
            name="technical",
            config={"enabled": True},
            knowledge_graph=None
        )

        indicators = {
            "MACD": {"macd": 100, "signal": 95, "histogram": 5}
        }
        decision = await agent.analyze(sample_market_data, indicators)

        assert isinstance(decision, AgentDecision)

    @pytest.mark.asyncio
    async def test_analyze_bollinger_bands(self, sample_market_data):
        """Test analysis with Bollinger Bands."""
        agent = TechnicalAnalysisAgent(
            name="technical",
            config={"enabled": True},
            knowledge_graph=None
        )

        # Price near lower band
        indicators = {
            "BB": {"upper": 51000, "lower": 49200, "middle": 50000}
        }
        decision = await agent.analyze(sample_market_data, indicators)

        assert isinstance(decision, AgentDecision)

    @pytest.mark.asyncio
    async def test_analyze_combined_indicators(
        self, sample_market_data, sample_indicators
    ):
        """Test analysis with multiple indicators."""
        agent = TechnicalAnalysisAgent(
            name="technical",
            config={"enabled": True},
            knowledge_graph=None
        )

        decision = await agent.analyze(sample_market_data, sample_indicators)

        assert isinstance(decision, AgentDecision)
        assert decision.agent_name == "technical"
        assert decision.metadata["method"] == "technical_analysis"


class TestSentimentAnalysisAgent:
    """Test suite for SentimentAnalysisAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = SentimentAnalysisAgent(
            name="sentiment",
            config={"enabled": True},
            knowledge_graph=None
        )

        assert agent.name == "sentiment"

    @pytest.mark.asyncio
    async def test_analyze_positive_sentiment(self, sample_market_data):
        """Test analysis with positive sentiment."""
        agent = SentimentAnalysisAgent(
            name="sentiment",
            config={"enabled": True},
            knowledge_graph=None
        )

        context = {
            "news_sentiment": {"score": 0.6, "count": 10},
            "social_sentiment": {"score": 0.4, "volume": 500},
            "overall_sentiment": 0.5
        }

        decision = await agent.analyze(
            sample_market_data,
            {},
            context
        )

        assert decision.signal in [TradingSignal.BUY, TradingSignal.HOLD]
        assert isinstance(decision, AgentDecision)

    @pytest.mark.asyncio
    async def test_analyze_negative_sentiment(self, sample_market_data):
        """Test analysis with negative sentiment."""
        agent = SentimentAnalysisAgent(
            name="sentiment",
            config={"enabled": True},
            knowledge_graph=None
        )

        context = {
            "news_sentiment": {"score": -0.6, "count": 10},
            "overall_sentiment": -0.5
        }

        decision = await agent.analyze(
            sample_market_data,
            {},
            context
        )

        assert decision.signal in [TradingSignal.SELL, TradingSignal.HOLD]

    @pytest.mark.asyncio
    async def test_analyze_insufficient_data(self, sample_market_data):
        """Test analysis with insufficient sentiment data."""
        agent = SentimentAnalysisAgent(
            name="sentiment",
            config={"enabled": True},
            knowledge_graph=None
        )

        decision = await agent.analyze(sample_market_data, {}, {})

        assert decision.signal == TradingSignal.HOLD


class TestRiskManagementAgent:
    """Test suite for RiskManagementAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = RiskManagementAgent(
            name="risk",
            config={"enabled": True, "max_position_size": 0.1},
            knowledge_graph=None
        )

        assert agent.name == "risk"
        assert agent.config["max_position_size"] == 0.1

    @pytest.mark.asyncio
    async def test_analyze_high_volatility(self, sample_market_data):
        """Test analysis with high volatility."""
        agent = RiskManagementAgent(
            name="risk",
            config={"enabled": True},
            knowledge_graph=None
        )

        indicators = {"volatility": {"value": 0.08}}
        decision = await agent.analyze(sample_market_data, indicators)

        # Should recommend HOLD or reduce position size
        assert decision.signal in [TradingSignal.HOLD, TradingSignal.SELL]
        assert isinstance(decision, AgentDecision)

    @pytest.mark.asyncio
    async def test_analyze_high_exposure(self, sample_market_data):
        """Test analysis with high portfolio exposure."""
        agent = RiskManagementAgent(
            name="risk",
            config={"enabled": True},
            knowledge_graph=None
        )

        context = {
            "portfolio": {"exposure": 0.85, "drawdown": -0.02}
        }

        decision = await agent.analyze(
            sample_market_data,
            {},
            context
        )

        assert decision.signal in [TradingSignal.HOLD, TradingSignal.SELL]

    @pytest.mark.asyncio
    async def test_analyze_significant_drawdown(self, sample_market_data):
        """Test analysis with significant drawdown."""
        agent = RiskManagementAgent(
            name="risk",
            config={"enabled": True},
            knowledge_graph=None
        )

        context = {
            "portfolio": {"exposure": 0.5, "drawdown": -0.12}
        }

        decision = await agent.analyze(
            sample_market_data,
            {},
            context
        )

        assert decision.signal == TradingSignal.HOLD


class TestMomentumAgent:
    """Test suite for MomentumAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = MomentumAgent(
            name="momentum",
            config={"enabled": True},
            knowledge_graph=None
        )

        assert agent.name == "momentum"

    @pytest.mark.asyncio
    async def test_analyze_positive_momentum(self, sample_market_data):
        """Test analysis with positive momentum."""
        agent = MomentumAgent(
            name="momentum",
            config={"enabled": True},
            knowledge_graph=None
        )

        indicators = {
            "ROC": {"value": 3.5},
            "ADX": {"value": 28, "di_plus": 25, "di_minus": 18}
        }

        decision = await agent.analyze(sample_market_data, indicators)

        assert decision.signal in [TradingSignal.BUY, TradingSignal.HOLD]

    @pytest.mark.asyncio
    async def test_analyze_negative_momentum(self, sample_market_data):
        """Test analysis with negative momentum."""
        agent = MomentumAgent(
            name="momentum",
            config={"enabled": True},
            knowledge_graph=None
        )

        indicators = {
            "ROC": {"value": -3.5},
            "ADX": {"value": 28, "di_plus": 18, "di_minus": 25}
        }

        decision = await agent.analyze(sample_market_data, indicators)

        assert decision.signal in [TradingSignal.SELL, TradingSignal.HOLD]


class TestMeanReversionAgent:
    """Test suite for MeanReversionAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = MeanReversionAgent(
            name="mean_reversion",
            config={"enabled": True},
            knowledge_graph=None
        )

        assert agent.name == "mean_reversion"

    @pytest.mark.asyncio
    async def test_analyze_overbought(self, sample_market_data):
        """Test analysis when price is overbought."""
        agent = MeanReversionAgent(
            name="mean_reversion",
            config={"enabled": True},
            knowledge_graph=None
        )

        indicators = {
            "zscore": {"value": 2.2},
            "BB": {"upper": 51000, "lower": 49000}
        }

        decision = await agent.analyze(sample_market_data, indicators)

        assert decision.signal in [TradingSignal.SELL, TradingSignal.HOLD]

    @pytest.mark.asyncio
    async def test_analyze_oversold(self, sample_market_data):
        """Test analysis when price is oversold."""
        agent = MeanReversionAgent(
            name="mean_reversion",
            config={"enabled": True},
            knowledge_graph=None
        )

        indicators = {
            "zscore": {"value": -2.2},
            "BB": {"upper": 51000, "lower": 49000}
        }

        decision = await agent.analyze(sample_market_data, indicators)

        assert decision.signal in [TradingSignal.BUY, TradingSignal.HOLD]


class TestDecisionEngine:
    """Test suite for DecisionEngine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = DecisionEngine(
            consensus_method=ConsensusMethod.WEIGHTED_VOTE,
            conflict_resolution=ConflictResolution.HIGH_CONFIDENCE_WINS
        )

        assert engine.consensus_method == ConsensusMethod.WEIGHTED_VOTE
        assert engine.conflict_resolution == ConflictResolution.HIGH_CONFIDENCE_WINS

    @pytest.mark.asyncio
    async def test_majority_vote(self):
        """Test majority vote consensus."""
        engine = DecisionEngine(consensus_method=ConsensusMethod.MAJORITY_VOTE)

        decisions = {
            "agent1": AgentDecision(
                signal=TradingSignal.BUY,
                confidence=0.7,
                reasoning="Agent 1",
                agent_name="agent1"
            ),
            "agent2": AgentDecision(
                signal=TradingSignal.BUY,
                confidence=0.6,
                reasoning="Agent 2",
                agent_name="agent2"
            ),
            "agent3": AgentDecision(
                signal=TradingSignal.SELL,
                confidence=0.8,
                reasoning="Agent 3",
                agent_name="agent3"
            )
        }

        result = await engine.make_decision(decisions, {}, {})

        assert result.signal == TradingSignal.BUY
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_confidence_weighted(self):
        """Test confidence-weighted consensus."""
        engine = DecisionEngine(
            consensus_method=ConsensusMethod.CONFIDENCE_WEIGHTED
        )

        decisions = {
            "agent1": AgentDecision(
                signal=TradingSignal.BUY,
                confidence=0.9,
                reasoning="Agent 1",
                agent_name="agent1"
            ),
            "agent2": AgentDecision(
                signal=TradingSignal.SELL,
                confidence=0.4,
                reasoning="Agent 2",
                agent_name="agent2"
            )
        }

        result = await engine.make_decision(decisions, {}, {})

        # High confidence BUY should win
        assert result.signal == TradingSignal.BUY

    @pytest.mark.asyncio
    async def test_conflict_score_calculation(self):
        """Test conflict score calculation."""
        engine = DecisionEngine()

        # Unanimous decisions
        unanimous_decisions = {
            "agent1": AgentDecision(
                signal=TradingSignal.BUY,
                confidence=0.7,
                reasoning="Agent 1",
                agent_name="agent1"
            ),
            "agent2": AgentDecision(
                signal=TradingSignal.BUY,
                confidence=0.8,
                reasoning="Agent 2",
                agent_name="agent2"
            )
        }

        conflict_score = engine._calculate_conflict_score(unanimous_decisions)
        assert conflict_score == 0.0

        # Conflicting decisions
        conflicting_decisions = {
            "agent1": AgentDecision(
                signal=TradingSignal.BUY,
                confidence=0.7,
                reasoning="Agent 1",
                agent_name="agent1"
            ),
            "agent2": AgentDecision(
                signal=TradingSignal.SELL,
                confidence=0.7,
                reasoning="Agent 2",
                agent_name="agent2"
            ),
            "agent3": AgentDecision(
                signal=TradingSignal.HOLD,
                confidence=0.5,
                reasoning="Agent 3",
                agent_name="agent3"
            )
        }

        conflict_score = engine._calculate_conflict_score(conflicting_decisions)
        assert conflict_score > 0.5


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator."""

    def test_initialization(self, agents_config):
        """Test orchestrator initialization."""
        orchestrator = AgentOrchestrator(agents_config)

        assert len(orchestrator.agents) == 5
        assert "technical" in orchestrator.agents
        assert "momentum" in orchestrator.agents
        assert "mean_reversion" in orchestrator.agents

    @pytest.mark.asyncio
    async def test_get_decision(
        self, agents_config, sample_market_data, sample_indicators
    ):
        """Test getting trading decision."""
        orchestrator = AgentOrchestrator(agents_config)

        decision = await orchestrator.get_decision(
            sample_market_data,
            sample_indicators,
            {"symbol": "BTC/USDT"}
        )

        assert decision.signal in [
            TradingSignal.BUY,
            TradingSignal.SELL,
            TradingSignal.HOLD
        ]
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.reasoning) > 0
        assert len(decision.participating_agents) > 0

    @pytest.mark.asyncio
    async def test_get_decision_empty_agents(self, sample_market_data):
        """Test decision with no agents."""
        config = {"technical": {"enabled": False}}
        orchestrator = AgentOrchestrator(config)

        decision = await orchestrator.get_decision(
            sample_market_data,
            {},
            {}
        )

        assert decision.signal == TradingSignal.HOLD

    @pytest.mark.asyncio
    async def test_update_performance(self, agents_config):
        """Test performance update."""
        orchestrator = AgentOrchestrator(agents_config)

        # Create a mock decision
        from graphwiz_trader.agents.decision import DecisionResult

        decision = DecisionResult(
            signal=TradingSignal.BUY,
            confidence=0.8,
            reasoning="Test decision",
            participating_agents=["technical", "momentum"],
            agent_signals={},
            consensus_method=ConsensusMethod.WEIGHTED_VOTE,
            conflict_score=0.2
        )

        # Update performance
        performance = await orchestrator.update_performance(
            symbol="BTC/USDT",
            decision=decision,
            entry_price=50000.0,
            current_price=50500.0,
            position_size=0.1,
            action_taken="BUY"
        )

        assert isinstance(performance, dict)

    def test_get_agent_weights(self, agents_config):
        """Test getting agent weights."""
        orchestrator = AgentOrchestrator(agents_config)

        weights = orchestrator.get_agent_weights()

        assert isinstance(weights, dict)
        assert len(weights) == 5
        for weight in weights.values():
            assert weight > 0

    def test_get_agent_performance(self, agents_config):
        """Test getting agent performance."""
        orchestrator = AgentOrchestrator(agents_config)

        performance = orchestrator.get_agent_performance()

        assert isinstance(performance, dict)
        for agent_perf in performance.values():
            assert "accuracy" in agent_perf
            assert "profit_factor" in agent_perf

    def test_get_decision_statistics(self, agents_config):
        """Test getting decision statistics."""
        orchestrator = AgentOrchestrator(agents_config)

        stats = orchestrator.get_decision_statistics()

        assert "orchestrator" in stats
        assert "decision_engine" in stats
        assert stats["orchestrator"]["total_agents"] == 5


class TestAgentDecision:
    """Test suite for AgentDecision dataclass."""

    def test_decision_creation(self):
        """Test creating an agent decision."""
        decision = AgentDecision(
            signal=TradingSignal.BUY,
            confidence=0.8,
            reasoning="Strong bullish signal",
            agent_name="test_agent"
        )

        assert decision.signal == TradingSignal.BUY
        assert decision.confidence == 0.8
        assert decision.reasoning == "Strong bullish signal"
        assert decision.agent_name == "test_agent"

    def test_decision_to_dict(self):
        """Test converting decision to dictionary."""
        decision = AgentDecision(
            signal=TradingSignal.SELL,
            confidence=0.7,
            reasoning="Bearish trend",
            metadata={"test": "value"},
            agent_name="test_agent"
        )

        decision_dict = decision.to_dict()

        assert decision_dict["signal"] == "SELL"
        assert decision_dict["confidence"] == 0.7
        assert "timestamp" in decision_dict
        assert decision_dict["metadata"]["test"] == "value"
