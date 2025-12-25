"""Tests for agent orchestrator module."""

import pytest
from unittest.mock import MagicMock

from graphwiz_trader.agents import AgentOrchestrator


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator class."""

    def test_initialization(self, agents_config, mock_kg):
        """Test AgentOrchestrator initialization."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        assert orchestrator.config == agents_config
        assert orchestrator.kg == mock_kg
        assert isinstance(orchestrator.agents, dict)

    def test_initialize_agents(self, agents_config, mock_kg):
        """Test agent initialization from configuration."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        assert "technical" in orchestrator.agents
        assert "sentiment" in orchestrator.agents
        assert "risk" not in orchestrator.agents  # Disabled in config

    def test_initialize_agents_all_enabled(self, mock_kg):
        """Test initializing all agents when enabled."""
        config = {
            "technical": {"enabled": True, "model": "gpt-4"},
            "sentiment": {"enabled": True, "model": "gpt-3.5-turbo"},
            "risk": {"enabled": True, "model": "gpt-4"}
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        assert len(orchestrator.agents) == 3
        assert "technical" in orchestrator.agents
        assert "sentiment" in orchestrator.agents
        assert "risk" in orchestrator.agents

    def test_initialize_agents_all_disabled(self, mock_kg):
        """Test when all agents are disabled."""
        config = {
            "technical": {"enabled": False},
            "sentiment": {"enabled": False}
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        assert len(orchestrator.agents) == 0

    def test_agent_config_storage(self, agents_config, mock_kg):
        """Test that agent configuration is stored correctly."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        technical_agent = orchestrator.agents["technical"]
        # Agents are now BaseAgent objects with attributes
        assert technical_agent.config["enabled"] is True
        assert technical_agent.model == "gpt-4"

    def test_agent_model_default(self, mock_kg):
        """Test default model when not specified in config."""
        config = {
            "custom_agent": {"enabled": True}
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        # Unknown agent type defaults to TechnicalAnalysisAgent with "gpt-4" as default
        assert orchestrator.agents["custom_agent"].model == "gpt-4"
        # The config should not have model specified
        assert orchestrator.agents["custom_agent"].config.get("model") is None

    def test_agent_model_custom(self, mock_kg):
        """Test custom model specification."""
        config = {
            "custom_agent": {
                "enabled": True,
                "model": "claude-3-opus"
            }
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        assert orchestrator.agents["custom_agent"].model == "claude-3-opus"

    def test_get_decision_stub(self, agents_config, mock_kg):
        """Test get_decision implementation."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        context = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "technical_indicators": {}
        }

        decision = orchestrator.get_decision(context)

        assert decision["action"] in ["buy", "sell", "hold"]
        assert "confidence" in decision

    def test_get_decision_with_context(self, agents_config, mock_kg):
        """Test get_decision with trading context."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        context = {
            "symbol": "ETH/USDT",
            "current_price": 3000,
            "technical_indicators": {
                "overall_signal": {"signal": "buy", "confidence": 0.7}
            }
        }

        decision = orchestrator.get_decision(context)

        assert isinstance(decision, dict)
        assert "action" in decision
        assert "confidence" in decision
        assert "agent_count" in decision

    def test_get_decision_empty_agents(self, mock_kg):
        """Test get_decision when no agents are initialized."""
        config = {"agent": {"enabled": False}}
        orchestrator = AgentOrchestrator(config, mock_kg)

        decision = orchestrator.get_decision({})

        assert decision["action"] == "hold"
        assert decision["confidence"] == 0.0
        assert decision["agent_count"] == 0

    def test_agent_with_temperature(self, mock_kg):
        """Test agent with temperature parameter."""
        config = {
            "creative_agent": {
                "enabled": True,
                "model": "gpt-4",
                "temperature": 0.9
            }
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        assert orchestrator.agents["creative_agent"].temperature == 0.9

    def test_multiple_agents_different_configs(self, mock_kg):
        """Test multiple agents with different configurations."""
        config = {
            "conservative": {
                "enabled": True,
                "model": "gpt-4",
                "temperature": 0.1
            },
            "aggressive": {
                "enabled": True,
                "model": "gpt-3.5-turbo",
                "temperature": 0.9
            },
            "balanced": {
                "enabled": True,
                "model": "gpt-4",
                "temperature": 0.5
            }
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        assert len(orchestrator.agents) == 3
        # All these unknown agent types will be TechnicalAnalysisAgent instances
        assert orchestrator.agents["conservative"].temperature == 0.1
        assert orchestrator.agents["aggressive"].temperature == 0.9
        assert orchestrator.agents["balanced"].temperature == 0.5

    def test_agent_count(self, agents_config, mock_kg):
        """Test that only enabled agents are initialized."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        # Only 2 agents enabled in the fixture
        assert len(orchestrator.agents) == 2

    def test_agents_object_structure(self, agents_config, mock_kg):
        """Test that agents are BaseAgent objects with expected attributes."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        for agent_name, agent in orchestrator.agents.items():
            # Agents are now objects, not dicts
            assert hasattr(agent, 'config')
            assert hasattr(agent, 'model')
            assert hasattr(agent, 'temperature')
            assert hasattr(agent, 'name')
            assert hasattr(agent, 'get_decision')

    def test_consensus_calculation(self, mock_kg):
        """Test consensus calculation from multiple agents."""
        config = {
            "technical": {"enabled": True, "model": "gpt-4"},
            "sentiment": {"enabled": True, "model": "gpt-3.5-turbo"},
            "risk": {"enabled": True, "model": "gpt-4"}
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        context = {
            "symbol": "BTC/USDT",
            "technical_indicators": {
                "overall_signal": {"signal": "buy", "confidence": 0.8}
            },
            "positions": []  # No positions for risk agent
        }

        decision = orchestrator.get_decision(context)

        assert "action" in decision
        assert "confidence" in decision
        assert "reasoning" in decision
        assert decision["agent_count"] == 3
        assert "individual_decisions" in decision
        assert len(decision["individual_decisions"]) == 3

    def test_decision_history(self, agents_config, mock_kg):
        """Test decision history tracking."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        context = {"symbol": "BTC/USDT", "technical_indicators": {}}

        # Make multiple decisions
        orchestrator.get_decision(context)
        orchestrator.get_decision(context)

        history = orchestrator.get_decision_history()

        assert len(history) == 2
        assert all("action" in h for h in history)
        assert all("timestamp" in h for h in history)

    def test_get_agent_status(self, agents_config, mock_kg):
        """Test getting agent status."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        status = orchestrator.get_agent_status()

        assert isinstance(status, dict)
        assert "technical" in status
        assert "sentiment" in status

        # Check status structure
        for agent_name, agent_status in status.items():
            assert "model" in agent_status
            assert "temperature" in agent_status
            assert "max_tokens" in agent_status
