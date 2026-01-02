"""
Configuration validation and management using Pydantic.

This module provides type-safe configuration models with validation,
environment variable support, and secrets management.
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
import yaml

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import BaseSettings

from .exceptions import InvalidConfigurationError, MissingConfigurationError


class TradingMode(str, Enum):
    """Trading execution modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class RiskLevel(str, Enum):
    """Risk management levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ExchangeConfig(BaseModel):
    """Exchange configuration with validation."""

    name: str = Field(..., description="Exchange name (e.g., binance, coinbase)")
    api_key: Optional[str] = Field(None, description="API key (load from env if not provided)")
    api_secret: Optional[str] = Field(None, description="API secret (load from env if not provided)")
    enabled: bool = Field(True, description="Whether this exchange is enabled")
    test_mode: bool = Field(False, description="Use testnet/sandbox mode")
    rate_limit: int = Field(1200, ge=1, le=10000, description="Requests per minute limit")
    timeout: int = Field(30, ge=1, le=120, description="Request timeout in seconds")

    @validator('api_key', 'api_secret', pre=True)
    def load_from_env(cls, v, field):
        """Load API credentials from environment if not provided."""
        if v is None:
            env_var = f"EXCHANGE_{field.name.upper()}"
            v = os.getenv(env_var)
            if v is None and field.name == 'api_key':
                # Only api_key is required, api_secret can be optional for read-only
                pass
        return v

    @validator('name')
    def name_must_be_valid_exchange(cls, v):
        """Validate exchange name."""
        valid_exchanges = ['binance', 'coinbase', 'kraken', 'bitstamp', 'gemini']
        if v.lower() not in valid_exchanges:
            raise ValueError(f"Invalid exchange: {v}. Must be one of {valid_exchanges}")
        return v.lower()


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size_pct: float = Field(
        0.10,
        ge=0.01,
        le=1.0,
        description="Maximum position size as percentage of portfolio"
    )
    max_daily_loss_pct: float = Field(
        0.05,
        ge=0.01,
        le=0.5,
        description="Maximum daily loss as percentage of portfolio"
    )
    max_drawdown_pct: float = Field(
        0.20,
        ge=0.05,
        le=1.0,
        description="Maximum drawdown before circuit breaker"
    )
    stop_loss_pct: float = Field(
        0.02,
        ge=0.005,
        le=0.20,
        description="Default stop-loss percentage"
    )
    take_profit_pct: float = Field(
        0.05,
        ge=0.01,
        le=0.50,
        description="Default take-profit percentage"
    )
    risk_per_trade: float = Field(
        0.01,
        ge=0.001,
        le=0.10,
        description="Risk per trade as percentage of portfolio (Kelly criterion)"
    )
    max_correlation_exposure: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Maximum exposure to correlated assets"
    )
    enable_circuit_breaker: bool = Field(
        True,
        description="Enable automatic circuit breaker on drawdown"
    )

    @root_validator
    def validate_risk_params(cls, values):
        """Validate risk parameter relationships."""
        stop_loss = values.get('stop_loss_pct', 0)
        take_profit = values.get('take_profit_pct', 0)

        if take_profit <= stop_loss:
            raise ValueError(
                f"Take profit ({take_profit}) must be greater than stop loss ({stop_loss})"
            )

        max_daily = values.get('max_daily_loss_pct', 0)
        max_dd = values.get('max_drawdown_pct', 0)

        if max_daily >= max_dd:
            raise ValueError(
                f"Max daily loss ({max_daily}) must be less than max drawdown ({max_dd})"
            )

        return values


class KnowledgeGraphConfig(BaseModel):
    """Neo4j knowledge graph configuration."""

    uri: str = Field(..., description="Neo4j connection URI")
    username: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(..., description="Neo4j password (load from env if not provided)")
    database: str = Field(default="neo4j", description="Neo4j database name")
    max_connection_lifetime: int = Field(3600, ge=60, description="Connection lifetime in seconds")
    max_connection_pool_size: int = Field(50, ge=1, le=200, description="Max connection pool size")
    connection_acquisition_timeout: int = Field(60, ge=1, description="Connection timeout in seconds")

    @validator('password', pre=True)
    def load_password_from_env(cls, v):
        """Load password from environment if not provided."""
        if v is None or v == "your_neo4j_password":
            v = os.getenv("NEO4J_PASSWORD")
            if v is None:
                raise ValueError(
                    "Neo4j password must be provided or set in NEO4J_PASSWORD environment variable"
                )
        return v

    @validator('uri')
    def validate_uri(cls, v):
        """Validate Neo4j URI format."""
        if not v.startswith(('bolt://', 'bolt+s://', 'neo4j://', 'neo4j+s://')):
            raise ValueError(
                f"Invalid Neo4j URI: {v}. Must start with bolt://, bolt+s://, neo4j://, or neo4j+s://"
            )
        return v


class AgentConfig(BaseModel):
    """AI agent configuration."""

    enabled: bool = Field(True, description="Enable AI agents")
    provider: str = Field(
        "openai",
        description="LLM provider (openai, anthropic, etc.)"
    )
    model: str = Field("gpt-4", description="LLM model name")
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature (randomness)"
    )
    max_tokens: int = Field(
        2000,
        ge=100,
        le=8000,
        description="Maximum tokens per response"
    )
    timeout: int = Field(30, ge=5, le=120, description="Agent timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retries on failure")
    consensus_threshold: float = Field(
        0.7,
        ge=0.5,
        le=1.0,
        description="Minimum consensus threshold for agent decisions"
    )

    @validator('provider')
    def validate_provider(cls, v):
        """Validate LLM provider."""
        valid_providers = ['openai', 'anthropic', 'cohere', 'huggingface']
        if v.lower() not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v.lower()


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    initial_capital: float = Field(
        100000.0,
        gt=0,
        description="Initial capital for backtesting"
    )
    commission_pct: float = Field(
        0.001,
        ge=0.0,
        le=0.01,
        description="Commission percentage per trade"
    )
    slippage_pct: float = Field(
        0.0005,
        ge=0.0,
        le=0.01,
        description="Slippage percentage per trade"
    )
    data_start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    data_end_date: str = Field(..., description="End date (YYYY-MM-DD)")

    @validator('data_start_date', 'data_end_date')
    def validate_date_format(cls, v):
        """Validate date format."""
        from datetime import datetime
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Must be YYYY-MM-DD")
        return v

    @root_validator
    def validate_date_range(cls, values):
        """Validate date range."""
        from datetime import datetime

        start = values.get('data_start_date')
        end = values.get('data_end_date')

        if start and end:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')

            if start_dt >= end_dt:
                raise ValueError(
                    f"Start date ({start}) must be before end date ({end})"
                )

        return values


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""

    enabled: bool = Field(True, description="Enable monitoring")
    log_level: str = Field(
        "INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    metrics_port: int = Field(9090, ge=1024, le=65535, description="Prometheus metrics port")
    dashboard_port: int = Field(8050, ge=1024, le=65535, description="Dashboard port")
    enable_slack_alerts: bool = Field(False, description="Enable Slack alerts")
    slack_webhook_url: Optional[str] = Field(None, description="Slack webhook URL")
    enable_email_alerts: bool = Field(False, description="Enable email alerts")
    email_recipients: List[str] = Field(default_factory=list, description="Email recipients")

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class GraphWizConfig(BaseSettings):
    """
    Main configuration class for GraphWiz Trader.

    This class loads and validates all configuration parameters from
    YAML files and environment variables.
    """

    # Core settings
    trading_mode: TradingMode = Field(TradingMode.PAPER, description="Trading mode")
    risk_level: RiskLevel = Field(RiskLevel.MODERATE, description="Risk level")

    # Sub-configurations
    exchanges: List[ExchangeConfig] = Field(default_factory=list)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(...)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    backtest: Optional[BacktestConfig] = Field(None, description="Backtesting config (required for backtest mode)")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Additional settings
    symbols: List[str] = Field(
        default_factory=lambda: ['BTC/USDT', 'ETH/USDT'],
        description="Trading symbols"
    )
    update_interval_seconds: int = Field(
        60,
        ge=1,
        le=3600,
        description="Market data update interval"
    )
    enable_logging: bool = Field(True, description="Enable file logging")
    log_file_path: str = Field("logs/graphwiz_trader.log", description="Log file path")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in config

    @classmethod
    def from_yaml(cls, config_path: str) -> 'GraphWizConfig':
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Validated GraphWizConfig instance

        Raises:
            InvalidConfigurationError: If configuration is invalid
            MissingConfigurationError: If required fields are missing
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise MissingConfigurationError(
                missing_keys=[f"config_file:{config_path}"]
            )

        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise InvalidConfigurationError(
                config_path=str(config_file),
                errors=[f"Invalid YAML: {str(e)}"]
            )

        # Extract environment-specific overrides
        env = os.getenv("GRAPHWIZ_ENV", "development")
        env_overrides = config_data.get('environments', {}).get(env, {})

        # Merge base config with environment overrides
        if env_overrides:
            config_data = {**config_data, **env_overrides}
            # Remove environments section to avoid validation error
            config_data.pop('environments', None)

        try:
            return cls(**config_data)
        except Exception as e:
            errors = []
            if hasattr(e, 'errors'):
                for error in e.errors():
                    loc = ' -> '.join(str(l) for l in error['loc'])
                    errors.append(f"{loc}: {error['msg']}")
            else:
                errors = [str(e)]

            raise InvalidConfigurationError(
                config_path=str(config_file),
                errors=errors
            )

    def to_yaml(self, output_path: str) -> None:
        """
        Export configuration to YAML file.

        Args:
            output_path: Path to output YAML file
        """
        config_dict = self.dict()

        # Remove sensitive data
        if 'knowledge_graph' in config_dict:
            config_dict['knowledge_graph']['password'] = '***REMOVED***'

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @root_validator
    def validate_trading_mode(cls, values):
        """Validate configuration based on trading mode."""
        mode = values.get('trading_mode')

        if mode == TradingMode.BACKTEST:
            if values.get('backtest') is None:
                raise ValueError("Backtest configuration is required for backtest mode")

        return values

    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """
        Get configuration for a specific exchange.

        Args:
            exchange_name: Name of the exchange

        Returns:
            ExchangeConfig if found, None otherwise
        """
        for exchange in self.exchanges:
            if exchange.name == exchange_name.lower():
                return exchange
        return None


def load_config(config_path: Optional[str] = None) -> GraphWizConfig:
    """
    Load and validate GraphWiz Trader configuration.

    Args:
        config_path: Path to configuration file. If None, searches standard locations.

    Returns:
        Validated GraphWizConfig instance

    Raises:
        InvalidConfigurationError: If configuration is invalid
        MissingConfigurationError: If configuration file not found
    """
    # Standard config file locations
    config_paths = [
        config_path,
        os.getenv("GRAPHWIZ_CONFIG"),
        "config/config.yaml",
        "config/production.yaml",
        "config/paper_trading.yaml",
    ]

    # Remove None values
    config_paths = [p for p in config_paths if p is not None]

    # Try each location
    for path in config_paths:
        if path and Path(path).exists():
            return GraphWizConfig.from_yaml(path)

    raise MissingConfigurationError(
        missing_keys=[f"config_file (searched: {', '.join(config_paths)})"]
    )


# Export main classes
__all__ = [
    'GraphWizConfig',
    'ExchangeConfig',
    'RiskConfig',
    'KnowledgeGraphConfig',
    'AgentConfig',
    'BacktestConfig',
    'MonitoringConfig',
    'TradingMode',
    'RiskLevel',
    'load_config',
]
