"""
Custom exception hierarchy for GraphWiz Trader.

This module defines a structured exception system with specific error types,
error codes, and retryability flags for better error handling and debugging.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


class GraphWizError(Exception):
    """
    Base exception for all GraphWiz Trader errors.

    All custom exceptions should inherit from this class to provide
    consistent error handling, logging, and user feedback.

    Attributes:
        message: Human-readable error message
        error_code: Unique error code for programmatic handling
        retryable: Whether the operation can be safely retried
        context: Additional error context for debugging
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        retryable: bool = False,
        **context
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.retryable = retryable
        self.context = context
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
            "context": self.context
        }

    def __str__(self) -> str:
        context_str = f" | Context: {self.context}" if self.context else ""
        return f"[{self.error_code}] {self.message}{context_str}"


# ============================================================================
# Trading & Exchange Errors
# ============================================================================

class TradingError(GraphWizError):
    """Base exception for trading-related errors."""
    pass


class InsufficientFundsError(TradingError):
    """
    Raised when account has insufficient funds for a trade.

    Attributes:
        required: Amount required
        available: Amount currently available
        symbol: Trading symbol
    """

    def __init__(self, required: float, available: float, symbol: str):
        super().__init__(
            message=f"Insufficient funds for {symbol}: required {required}, available {available}",
            error_code="INSUFFICIENT_FUNDS",
            retryable=False,
            required=required,
            available=available,
            symbol=symbol
        )


class OrderExecutionError(TradingError):
    """
    Raised when order execution fails.

    Attributes:
        order_id: Order identifier
        reason: Specific failure reason
    """

    def __init__(self, order_id: str, reason: str, retryable: bool = True):
        super().__init__(
            message=f"Order {order_id} execution failed: {reason}",
            error_code="ORDER_EXECUTION_FAILED",
            retryable=retryable,
            order_id=order_id,
            reason=reason
        )


class ExchangeConnectionError(TradingError):
    """
    Raised when connection to exchange fails.

    This is typically a transient error that can be retried.
    """

    def __init__(self, exchange: str, reason: str):
        super().__init__(
            message=f"Connection to {exchange} failed: {reason}",
            error_code="EXCHANGE_CONNECTION",
            retryable=True,
            exchange=exchange,
            reason=reason
        )


class ExchangeAPIError(TradingError):
    """
    Raised when exchange API returns an error.

    Attributes:
        exchange: Exchange name
        status_code: HTTP status code (if applicable)
        response: API response details
    """

    def __init__(self, exchange: str, status_code: Optional[int] = None,
                 response: Optional[str] = None):
        super().__init__(
            message=f"Exchange API error from {exchange}: {response}",
            error_code="EXCHANGE_API_ERROR",
            retryable=status_code is not None and 500 <= status_code < 600,
            exchange=exchange,
            status_code=status_code,
            response=response
        )


class RateLimitError(TradingError):
    """
    Raised when exchange rate limit is exceeded.

    Attributes:
        exchange: Exchange name
        retry_after: Seconds to wait before retrying
    """

    def __init__(self, exchange: str, retry_after: int):
        super().__init__(
            message=f"Rate limit exceeded for {exchange}. Retry after {retry_after}s",
            error_code="RATE_LIMIT_EXCEEDED",
            retryable=True,
            exchange=exchange,
            retry_after=retry_after
        )


class InvalidOrderError(TradingError):
    """
    Raised when order parameters are invalid.

    Attributes:
        reason: Specific validation failure
    """

    def __init__(self, reason: str, **params):
        super().__init__(
            message=f"Invalid order parameters: {reason}",
            error_code="INVALID_ORDER",
            retryable=False,
            reason=reason,
            params=params
        )


# ============================================================================
# Risk Management Errors
# ============================================================================

class RiskError(GraphWizError):
    """Base exception for risk management errors."""
    pass


class RiskLimitExceededError(RiskError):
    """
    Raised when a trade would violate risk limits.

    Attributes:
        limit_type: Type of limit violated
        limit_value: Limit value
        attempted_value: Value that would exceed limit
    """

    def __init__(self, limit_type: str, limit_value: float, attempted_value: float):
        super().__init__(
            message=f"Risk limit '{limit_type}' exceeded: {attempted_value} > {limit_value}",
            error_code="RISK_LIMIT_EXCEEDED",
            retryable=False,
            limit_type=limit_type,
            limit_value=limit_value,
            attempted_value=attempted_value
        )


class PositionSizeError(RiskError):
    """
    Raised when position size calculation fails or is invalid.

    Attributes:
        reason: Calculation failure reason
    """

    def __init__(self, reason: str):
        super().__init__(
            message=f"Position size error: {reason}",
            error_code="POSITION_SIZE_ERROR",
            retryable=False,
            reason=reason
        )


class DrawdownExceededError(RiskError):
    """
    Raised when maximum drawdown limit is exceeded.

    This is a critical error that may trigger circuit breakers.
    """

    def __init__(self, current_drawdown: float, max_allowed: float):
        super().__init__(
            message=f"Maximum drawdown exceeded: {current_drawdown:.2%} > {max_allowed:.2%}",
            error_code="DRAWDOWN_EXCEEDED",
            retryable=False,
            current_drawdown=current_drawdown,
            max_allowed=max_allowed
        )


# ============================================================================
# Knowledge Graph Errors
# ============================================================================

class KnowledgeGraphError(GraphWizError):
    """Base exception for knowledge graph operations."""
    pass


class GraphConnectionError(KnowledgeGraphError):
    """
    Raised when connection to Neo4j fails.

    Attributes:
        uri: Neo4j connection URI
        reason: Connection failure reason
    """

    def __init__(self, uri: str, reason: str):
        super().__init__(
            message=f"Failed to connect to knowledge graph at {uri}: {reason}",
            error_code="GRAPH_CONNECTION",
            retryable=True,
            uri=uri,
            reason=reason
        )


class GraphQueryError(KnowledgeGraphError):
    """
    Raised when a graph query fails.

    Attributes:
        query: Query that failed
        reason: Failure reason
    """

    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Graph query failed: {reason}",
            error_code="GRAPH_QUERY",
            retryable=False,
            query=query,
            reason=reason
        )


class GraphValidationError(KnowledgeGraphError):
    """
    Raised when graph data validation fails.

    Attributes:
        reason: Validation failure reason
    """

    def __init__(self, reason: str, **data):
        super().__init__(
            message=f"Graph validation failed: {reason}",
            error_code="GRAPH_VALIDATION",
            retryable=False,
            reason=reason,
            data=data
        )


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(GraphWizError):
    """Base exception for configuration errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """
    Raised when configuration is invalid.

    Attributes:
        config_path: Path to configuration file
        errors: List of validation errors
    """

    def __init__(self, config_path: str, errors: list):
        super().__init__(
            message=f"Invalid configuration in {config_path}: {', '.join(errors)}",
            error_code="INVALID_CONFIGURATION",
            retryable=False,
            config_path=config_path,
            errors=errors
        )


class MissingConfigurationError(ConfigurationError):
    """
    Raised when required configuration is missing.

    Attributes:
        missing_keys: List of missing configuration keys
    """

    def __init__(self, missing_keys: list):
        super().__init__(
            message=f"Missing required configuration: {', '.join(missing_keys)}",
            error_code="MISSING_CONFIGURATION",
            retryable=False,
            missing_keys=missing_keys
        )


# ============================================================================
# Agent Errors
# ============================================================================

class AgentError(GraphWizError):
    """Base exception for agent-related errors."""
    pass


class AgentInitializationError(AgentError):
    """
    Raised when agent fails to initialize.

    Attributes:
        agent_type: Type of agent that failed
        reason: Initialization failure reason
    """

    def __init__(self, agent_type: str, reason: str):
        super().__init__(
            message=f"Failed to initialize {agent_type} agent: {reason}",
            error_code="AGENT_INITIALIZATION",
            retryable=False,
            agent_type=agent_type,
            reason=reason
        )


class AgentExecutionError(AgentError):
    """
    Raised when agent execution fails.

    Attributes:
        agent_type: Type of agent that failed
        stage: Execution stage that failed
    """

    def __init__(self, agent_type: str, stage: str, reason: str):
        super().__init__(
            message=f"{agent_type} agent failed at {stage}: {reason}",
            error_code="AGENT_EXECUTION",
            retryable=True,
            agent_type=agent_type,
            stage=stage,
            reason=reason
        )


# ============================================================================
# Data Errors
# ============================================================================

class DataError(GraphWizError):
    """Base exception for data-related errors."""
    pass


class DataValidationError(DataError):
    """
    Raised when data validation fails.

    Attributes:
        field: Field that failed validation
        reason: Validation failure reason
    """

    def __init__(self, field: str, reason: str, value: Any = None):
        super().__init__(
            message=f"Data validation failed for {field}: {reason}",
            error_code="DATA_VALIDATION",
            retryable=False,
            field=field,
            reason=reason,
            value=value
        )


class MarketDataError(DataError):
    """
    Raised when market data fetch fails.

    Attributes:
        symbol: Symbol that failed
        source: Data source
        reason: Failure reason
    """

    def __init__(self, symbol: str, source: str, reason: str):
        super().__init__(
            message=f"Failed to fetch market data for {symbol} from {source}: {reason}",
            error_code="MARKET_DATA",
            retryable=True,
            symbol=symbol,
            source=source,
            reason=reason
        )


# ============================================================================
# Backtesting Errors
# ============================================================================

class BacktestError(GraphWizError):
    """Base exception for backtesting errors."""
    pass


class BacktestValidationError(BacktestError):
    """
    Raised when backtest parameters are invalid.

    Attributes:
        parameter: Invalid parameter
        reason: Validation failure reason
    """

    def __init__(self, parameter: str, reason: str):
        super().__init__(
            message=f"Invalid backtest parameter '{parameter}': {reason}",
            error_code="BACKTEST_VALIDATION",
            retryable=False,
            parameter=parameter,
            reason=reason
        )


class BacktestExecutionError(BacktestError):
    """
    Raised when backtest execution fails.

    Attributes:
        stage: Stage that failed
        reason: Failure reason
    """

    def __init__(self, stage: str, reason: str):
        super().__init__(
            message=f"Backtest failed at {stage}: {reason}",
            error_code="BACKTEST_EXECUTION",
            retryable=False,
            stage=stage,
            reason=reason
        )


# ============================================================================
# Utility Functions
# ============================================================================

def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error dictionary.

    Args:
        error: The exception to handle
        context: Additional context to include

    Returns:
        Standardized error dictionary
    """
    if isinstance(error, GraphWizError):
        error_dict = error.to_dict()
    else:
        error_dict = {
            "error_type": error.__class__.__name__,
            "error_code": "UNKNOWN_ERROR",
            "message": str(error),
            "retryable": False,
            "context": {}
        }

    if context:
        error_dict["context"].update(context)

    return error_dict


def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error can be safely retried
    """
    if isinstance(error, GraphWizError):
        return error.retryable
    return False
