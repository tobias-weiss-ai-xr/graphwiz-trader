"""Tests for the optimizer module."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from graphwiz_trader.optimizer import (
    TradingOptimizer,
    OptimizationOrchestrator,
)
from graphwiz_trader.optimizer.looper_integration import (
    OptimizationType,
    OptimizationStatus,
    OptimizationConstraints,
)
from graphwiz_trader.optimizer.orchestrator import (
    OrchestratorState,
    CircuitBreakerState,
)


@pytest.fixture
def mock_knowledge_graph():
    """Create a mock knowledge graph."""
    kg = Mock()
    kg.execute_query = Mock()
    return kg


@pytest.fixture
def optimization_constraints():
    """Create optimization constraints for testing."""
    return OptimizationConstraints(
        max_drawdown_threshold=0.10,
        min_sharpe_ratio=2.0,
        min_win_rate=0.60,
        require_paper_trading=True,
        paper_trading_duration_hours=24,
    )


class TestTradingOptimizer:
    """Tests for TradingOptimizer class."""

    def test_initialization(self, optimization_constraints, mock_knowledge_graph):
        """Test optimizer initialization."""
        optimizer = TradingOptimizer(
            project_path="/opt/git/graphwiz-trader",
            knowledge_graph=mock_knowledge_graph,
            constraints=optimization_constraints,
        )

        assert optimizer.constraints == optimization_constraints
        assert optimizer.kg == mock_knowledge_graph
        assert optimizer.enable_auto_approve is False
        assert len(optimizer.optimizations) == 0
        assert len(optimizer.rollback_states) == 0

    @pytest.mark.asyncio
    async def test_optimize_strategy_parameters(self, optimization_constraints, mock_knowledge_graph):
        """Test strategy parameter optimization."""
        with patch('graphwiz_trader.optimizer.looper_integration.SAIAAgent'):
            optimizer = TradingOptimizer(
                project_path="/opt/git/graphwiz-trader",
                knowledge_graph=mock_knowledge_graph,
                constraints=optimization_constraints,
            )

            # Mock the agent methods
            optimizer.agent.analyze = AsyncMock(return_value="Analysis results")
            optimizer.agent.plan = AsyncMock(return_value="Optimization plan")

            result = await optimizer.optimize_strategy_parameters(
                current_performance={
                    "sharpe_ratio": 1.8,
                    "max_drawdown": 0.08,
                    "win_rate": 0.58,
                }
            )

            assert result is not None
            assert result.optimization_type == OptimizationType.STRATEGY_PARAMETERS
            assert result.status in [OptimizationStatus.PENDING, OptimizationStatus.FAILED]

    def test_approve_optimization(self, optimization_constraints, mock_knowledge_graph):
        """Test optimization approval."""
        with patch('graphwiz_trader.optimizer.looper_integration.SAIAAgent'):
            optimizer = TradingOptimizer(
                project_path="/opt/git/graphwiz-trader",
                knowledge_graph=mock_knowledge_graph,
                constraints=optimization_constraints,
            )

            # Create a mock optimization
            from graphwiz_trader.optimizer.looper_integration import OptimizationResult
            opt = OptimizationResult(
                optimization_id="test_opt",
                optimization_type=OptimizationType.STRATEGY_PARAMETERS,
                status=OptimizationStatus.PENDING,
                proposed_changes={},
                expected_improvement=0.1,
                confidence_score=0.8,
                reasoning="Test optimization"
            )

            optimizer.optimizations["test_opt"] = opt

            # Approve it
            result = optimizer.approve_optimization("test_opt")

            assert result is True
            assert opt.status == OptimizationStatus.APPROVED

    def test_rollback_optimization(self, optimization_constraints, mock_knowledge_graph):
        """Test optimization rollback."""
        with patch('graphwiz_trader.optimizer.looper_integration.SAIAAgent'):
            optimizer = TradingOptimizer(
                project_path="/opt/git/graphwiz-trader",
                knowledge_graph=mock_knowledge_graph,
                constraints=optimization_constraints,
            )

            # Create a mock optimization and rollback state
            from graphwiz_trader.optimizer.looper_integration import (
                OptimizationResult,
                RollbackState,
            )

            opt = OptimizationResult(
                optimization_id="test_opt",
                optimization_type=OptimizationType.STRATEGY_PARAMETERS,
                status=OptimizationStatus.APPLIED,
                proposed_changes={},
                expected_improvement=0.1,
                confidence_score=0.8,
                reasoning="Test optimization"
            )

            snapshot = RollbackState(
                snapshot_id="snapshot_1",
                timestamp=datetime.utcnow(),
                config_backup={},
                performance_snapshot=[],
                optimization_id="test_opt"
            )

            optimizer.optimizations["test_opt"] = opt
            optimizer.rollback_states.append(snapshot)

            # Rollback
            result = optimizer.rollback_optimization("test_opt")

            assert result is True
            assert opt.status == OptimizationStatus.ROLLED_BACK


class TestOptimizationOrchestrator:
    """Tests for OptimizationOrchestrator class."""

    def test_initialization(self, mock_knowledge_graph):
        """Test orchestrator initialization."""
        orchestrator = OptimizationOrchestrator(
            project_path="/opt/git/graphwiz-trader",
            knowledge_graph=mock_knowledge_graph,
        )

        assert orchestrator.state == OrchestratorState.STOPPED
        assert orchestrator.circuit_breaker_state == CircuitBreakerState.CLOSED
        assert len(orchestrator.optimization_loops) > 0
        assert len(orchestrator.active_optimizations) == 0

    def test_get_status(self, mock_knowledge_graph):
        """Test getting orchestrator status."""
        orchestrator = OptimizationOrchestrator(
            project_path="/opt/git/graphwiz-trader",
            knowledge_graph=mock_knowledge_graph,
        )

        status = orchestrator.get_status()

        assert "state" in status
        assert "circuit_breaker_state" in status
        assert "active_optimizations" in status
        assert "optimization_loops" in status

    def test_pause_resume(self, mock_knowledge_graph):
        """Test pause and resume functionality."""
        orchestrator = OptimizationOrchestrator(
            project_path="/opt/git/graphwiz-trader",
            knowledge_graph=mock_knowledge_graph,
        )

        # Pause
        orchestrator.pause()
        assert orchestrator.state == OrchestratorState.PAUSED

        # Resume
        orchestrator.resume()
        assert orchestrator.state == OrchestratorState.RUNNING

    def test_circuit_breaker(self, mock_knowledge_graph):
        """Test circuit breaker functionality."""
        orchestrator = OptimizationOrchestrator(
            project_path="/opt/git/graphwiz-trader",
            knowledge_graph=mock_knowledge_graph,
        )

        # Trip circuit breaker
        orchestrator._trip_circuit_breaker("Test failure")

        assert orchestrator.circuit_breaker_state == CircuitBreakerState.OPEN
        assert orchestrator.circuit_breaker_tripped_at is not None

        # Reset
        orchestrator.reset_circuit_breaker()

        assert orchestrator.circuit_breaker_state == CircuitBreakerState.CLOSED
        assert orchestrator.circuit_breaker_tripped_at is None

    def test_approve_optimization(self, mock_knowledge_graph):
        """Test optimization approval through orchestrator."""
        with patch('graphwiz_trader.optimizer.looper_integration.SAIAAgent'):
            orchestrator = OptimizationOrchestrator(
                project_path="/opt/git/graphwiz-trader",
                knowledge_graph=mock_knowledge_graph,
            )

            # Create a mock optimization
            from graphwiz_trader.optimizer.looper_integration import OptimizationResult
            opt = OptimizationResult(
                optimization_id="test_opt",
                optimization_type=OptimizationType.STRATEGY_PARAMETERS,
                status=OptimizationStatus.PENDING,
                proposed_changes={},
                expected_improvement=0.1,
                confidence_score=0.8,
                reasoning="Test optimization"
            )

            orchestrator.optimizer.optimizations["test_opt"] = opt

            # Approve through orchestrator
            result = orchestrator.approve_optimization("test_opt")

            assert result is True
            assert opt.status == OptimizationStatus.APPROVED

    def test_reject_optimization(self, mock_knowledge_graph):
        """Test optimization rejection through orchestrator."""
        with patch('graphwiz_trader.optimizer.looper_integration.SAIAAgent'):
            orchestrator = OptimizationOrchestrator(
                project_path="/opt/git/graphwiz-trader",
                knowledge_graph=mock_knowledge_graph,
            )

            # Create a mock optimization
            from graphwiz_trader.optimizer.looper_integration import OptimizationResult
            opt = OptimizationResult(
                optimization_id="test_opt",
                optimization_type=OptimizationType.STRATEGY_PARAMETERS,
                status=OptimizationStatus.PENDING,
                proposed_changes={},
                expected_improvement=0.1,
                confidence_score=0.8,
                reasoning="Test optimization"
            )

            orchestrator.active_optimizations["test_opt"] = opt

            # Reject through orchestrator
            result = orchestrator.reject_optimization("test_opt")

            assert result is True
            assert opt.status == OptimizationStatus.REJECTED
            assert "test_opt" not in orchestrator.active_optimizations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
