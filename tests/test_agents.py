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
        assert technical_agent["config"]["enabled"] is True
        assert technical_agent["model"] == "gpt-4"

    def test_agent_model_default(self, mock_kg):
        """Test default model when not specified in config."""
        config = {
            "custom_agent": {"enabled": True}
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        assert orchestrator.agents["custom_agent"]["model"] == "gpt-4"

    def test_agent_model_custom(self, mock_kg):
        """Test custom model specification."""
        config = {
            "custom_agent": {
                "enabled": True,
                "model": "claude-3-opus"
            }
        }

        orchestrator = AgentOrchestrator(config, mock_kg)

        assert orchestrator.agents["custom_agent"]["model"] == "claude-3-opus"

    def test_get_decision_stub(self, agents_config, mock_kg):
        """Test get_decision stub implementation."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        context = {
            "symbol": "BTC/USDT",
            "price": 50000,
            "indicators": {}
        }

        decision = orchestrator.get_decision(context)

        assert decision["action"] == "hold"
        assert "confidence" in decision

    def test_get_decision_with_context(self, agents_config, mock_kg):
        """Test get_decision with trading context."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        context = {
            "symbol": "ETH/USDT",
            "price": 3000,
            "volume": 1000000,
            "trend": "upward"
        }

        decision = orchestrator.get_decision(context)

        assert isinstance(decision, dict)
        assert "action" in decision
        assert "confidence" in decision

    def test_get_decision_empty_agents(self, mock_kg):
        """Test get_decision when no agents are initialized."""
        config = {"agent": {"enabled": False}}
        orchestrator = AgentOrchestrator(config, mock_kg)

        decision = orchestrator.get_decision({})

        assert decision["action"] == "hold"
        assert "confidence" in decision

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

        assert orchestrator.agents["creative_agent"]["config"]["temperature"] == 0.9

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
        assert orchestrator.agents["conservative"]["temperature"] == 0.1
        assert orchestrator.agents["aggressive"]["temperature"] == 0.9
        assert orchestrator.agents["balanced"]["temperature"] == 0.5

    def test_agent_count(self, agents_config, mock_kg):
        """Test that only enabled agents are initialized."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        # Only 2 agents enabled in the fixture
        assert len(orchestrator.agents) == 2

    def test_agents_dict_structure(self, agents_config, mock_kg):
        """Test that agents have the expected structure."""
        orchestrator = AgentOrchestrator(agents_config, mock_kg)

        for agent_name, agent_data in orchestrator.agents.items():
            assert isinstance(agent_data, dict)
            assert "config" in agent_data
            assert "model" in agent_data
