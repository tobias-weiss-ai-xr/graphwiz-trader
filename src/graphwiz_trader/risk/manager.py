"""Risk management system orchestrator.

This module provides the main RiskManager class that coordinates position sizing,
portfolio risk monitoring, correlation analysis, and exposure limits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger

from .calculators import (
    calculate_position_size,
    calculate_portfolio_risk,
    calculate_correlation_matrix,
    calculate_max_drawdown,
    PositionSizingStrategy,
)
from .limits import RiskLimits, RiskLimitsConfig, StopLossCalculator
from .alerts import RiskAlertManager, Alert, AlertType, AlertSeverity, AlertThreshold


@dataclass
class Position:
    """Position data structure."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: str = "long"  # 'long' or 'short'
    sector: Optional[str] = None
    asset_class: Optional[str] = None
    entry_time: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    correlation_cluster: Optional[str] = None


@dataclass
class PortfolioState:
    """Portfolio state snapshot."""

    total_value: float
    cash_balance: float
    positions: List[Position]
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(
        self,
        account_balance: float,
        limits_config: Optional[RiskLimitsConfig] = None,
        knowledge_graph=None,
        default_risk_per_trade: float = 0.02,
    ):
        """Initialize risk manager.

        Args:
            account_balance: Initial account balance
            limits_config: Risk limits configuration (uses defaults if None)
            knowledge_graph: Optional knowledge graph for tracking metrics
            default_risk_per_trade: Default risk percentage per trade (2%)
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.default_risk_per_trade = default_risk_per_trade

        # Initialize components
        self.limits = RiskLimits(limits_config)
        self.stop_loss_calculator = StopLossCalculator()
        self.alert_manager = RiskAlertManager(knowledge_graph)
        self.kg = knowledge_graph

        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.realized_pnl = 0.0
        self.trades_today = 0
        self.last_update = datetime.utcnow()

        # Historical data for calculations
        self.price_history: Dict[str, pd.Series] = {}
        self.portfolio_value_history: List[float] = []
        self.correlation_cache: Optional[Dict[Tuple[str, str], float]] = None
        self.correlation_cache_time: Optional[datetime] = None

        # Metrics cache
        self.last_portfolio_risk: Optional[Dict[str, float]] = None
        self.last_drawdown_analysis: Optional[Dict[str, Any]] = None

        # Setup default alert thresholds
        self._setup_default_alerts()

        logger.info(
            "Risk Manager initialized with ${:.2f} account balance",
            self.account_balance,
        )

    def _setup_default_alerts(self) -> None:
        """Setup default alert thresholds."""
        # Drawdown alerts
        self.alert_manager.set_threshold(
            AlertThreshold(
                metric_name="drawdown",
                warning_threshold=0.05,  # 5%
                critical_threshold=0.10,  # 10%
                emergency_threshold=0.15,  # 15%
            )
        )

        # Daily loss alerts
        self.alert_manager.set_threshold(
            AlertThreshold(
                metric_name="daily_loss",
                warning_threshold=0.02,  # 2%
                critical_threshold=0.04,  # 4%
                emergency_threshold=0.05,  # 5%
            )
        )

        # Portfolio exposure alerts
        self.alert_manager.set_threshold(
            AlertThreshold(
                metric_name="portfolio_exposure",
                warning_threshold=0.80,  # 80%
                critical_threshold=0.95,  # 95%
                emergency_threshold=1.00,  # 100%
            )
        )

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        strategy: PositionSizingStrategy = PositionSizingStrategy.FIXED_FRACTIONAL,
        strategy_params: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Calculate optimal position size with risk checks.

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            strategy: Position sizing strategy
            strategy_params: Additional strategy parameters

        Returns:
            Dictionary with position size details

        Raises:
            ValueError: If position violates risk limits
        """
        # Calculate base position size
        result = calculate_position_size(
            account_balance=self.account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            risk_per_trade=self.default_risk_per_trade,
            strategy=strategy,
            strategy_params=strategy_params,
        )

        position_value = result["position_value"]

        # Check against position size limits
        allowed, message = self.limits.check_position_size(
            position_value=position_value,
            portfolio_value=self.account_balance,
            symbol=symbol,
            hard_limit=True,
        )

        if not allowed:
            logger.warning("Position size rejected: {}", message)
            raise ValueError(message)

        # Check if adding position would exceed total exposure
        current_exposure = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        total_exposure = current_exposure + position_value

        allowed, message = self.limits.check_total_exposure(
            total_exposure=total_exposure,
            portfolio_value=self.account_balance,
            hard_limit=True,
        )

        if not allowed:
            logger.warning("Total exposure would be exceeded: {}", message)
            raise ValueError(message)

        # Check correlation exposure
        if len(self.positions) > 0:
            correlation_data = self.get_correlation_matrix()
            positions_list = [
                {
                    "symbol": s,
                    "value": p.quantity * p.current_price,
                }
                for s, p in self.positions.items()
            ]
            positions_list.append({"symbol": symbol, "value": position_value})

            allowed, message = self.limits.check_correlation_exposure(
                positions=positions_list,
                correlation_matrix=correlation_data,
                hard_limit=True,
            )

            if not allowed:
                logger.warning("Correlation exposure would be exceeded: {}", message)
                raise ValueError(message)

        return result

    def add_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        side: str = "long",
        sector: Optional[str] = None,
        asset_class: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Position:
        """Add a position to the portfolio.

        Args:
            symbol: Asset symbol
            quantity: Position quantity
            entry_price: Entry price
            side: Position side ('long' or 'short')
            sector: Asset sector
            asset_class: Asset class
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Created Position object

        Raises:
            ValueError: If position violates risk limits
        """
        position_value = quantity * entry_price

        # Check position size limit
        allowed, message = self.limits.check_position_size(
            position_value=position_value,
            portfolio_value=self.account_balance,
            symbol=symbol,
            hard_limit=True,
        )

        if not allowed:
            raise ValueError(f"Position rejected: {message}")

        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            side=side,
            sector=sector,
            asset_class=asset_class,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.positions[symbol] = position
        self.trades_today += 1

        # Store in knowledge graph
        if self.kg:
            self._store_position_in_graph(position)

        logger.info(
            "Position added: {} {} {} at ${:.2f} (Value: ${:.2f})",
            side.upper(),
            quantity,
            symbol,
            entry_price,
            position_value,
        )

        # Run risk checks after adding position
        self._run_portfolio_risk_checks()

        return position

    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update position's current price.

        Args:
            symbol: Asset symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            logger.warning("Cannot update price for non-existent position: {}", symbol)
            return

        position = self.positions[symbol]
        position.current_price = current_price

        # Check stop-loss and take-profit
        if position.side == "long":
            if position.stop_loss and current_price <= position.stop_loss:
                self._trigger_stop_loss(symbol)
            elif position.take_profit and current_price >= position.take_profit:
                self._trigger_take_profit(symbol)
        else:  # short
            if position.stop_loss and current_price >= position.stop_loss:
                self._trigger_stop_loss(symbol)
            elif position.take_profit and current_price <= position.take_profit:
                self._trigger_take_profit(symbol)

        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = pd.Series(dtype=float)

        self.price_history[symbol][datetime.utcnow()] = current_price

    def close_position(self, symbol: str, exit_price: Optional[float] = None) -> float:
        """Close a position.

        Args:
            symbol: Asset symbol
            exit_price: Exit price (uses current_price if None)

        Returns:
            Realized P&L

        Raises:
            ValueError: If position doesn't exist
        """
        if symbol not in self.positions:
            raise ValueError(f"Position {symbol} does not exist")

        position = self.positions[symbol]
        exit_price = exit_price or position.current_price

        # Calculate P&L
        if position.side == "long":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # short
            pnl = (position.entry_price - exit_price) * position.quantity

        # Update account balance
        self.account_balance += pnl
        self.daily_pnl += pnl
        self.realized_pnl += pnl

        logger.info(
            "Position closed: {} | Entry: ${:.2f} -> Exit: ${:.2f} | P&L: ${:.2f}",
            symbol,
            position.entry_price,
            exit_price,
            pnl,
        )

        # Remove position
        del self.positions[symbol]

        # Store in knowledge graph
        if self.kg:
            self._store_closed_trade_in_graph(position, exit_price, pnl)

        return pnl

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state.

        Returns:
            PortfolioState object with current portfolio information
        """
        total_position_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        unrealized_pnl = sum(
            (pos.current_price - pos.entry_price) * pos.quantity
            for pos in self.positions.values()
            if pos.side == "long"
        ) + sum(
            (pos.entry_price - pos.current_price) * pos.quantity
            for pos in self.positions.values()
            if pos.side == "short"
        )

        return PortfolioState(
            total_value=self.account_balance,
            cash_balance=self.account_balance - total_position_value,
            positions=list(self.positions.values()),
            daily_pnl=self.daily_pnl,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
        )

    def calculate_portfolio_risk(
        self,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics.

        Args:
            confidence_level: Confidence level for VaR (default 95%)
            method: VaR calculation method

        Returns:
            Dictionary with risk metrics
        """
        if not self.positions:
            return {
                "portfolio_value": self.account_balance,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "portfolio_std": 0.0,
                "worst_case_loss": 0.0,
            }

        # Build price DataFrame
        prices_df = self._build_prices_dataframe()

        if prices_df is None or prices_df.empty:
            logger.warning("Insufficient price data for portfolio risk calculation")
            return {
                "portfolio_value": self.account_balance,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "portfolio_std": 0.0,
                "worst_case_loss": 0.0,
            }

        # Build positions list
        positions_list = [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
            }
            for pos in self.positions.values()
        ]

        # Calculate risk
        risk_metrics = calculate_portfolio_risk(
            positions=positions_list,
            prices=prices_df,
            confidence_level=confidence_level,
            method=method,
        )

        self.last_portfolio_risk = risk_metrics

        # Store in knowledge graph
        if self.kg:
            self._store_portfolio_risk_in_graph(risk_metrics)

        return risk_metrics

    def get_correlation_matrix(self, recalculate: bool = False) -> Dict[Tuple[str, str], float]:
        """Get correlation matrix for current positions.

        Args:
            recalculate: Force recalculation even if cached

        Returns:
            Dictionary of (symbol1, symbol2) -> correlation
        """
        # Check cache
        cache_age = (
            datetime.utcnow() - self.correlation_cache_time
            if self.correlation_cache_time
            else timedelta(hours=1)
        )

        if not recalculate and self.correlation_cache and cache_age < timedelta(minutes=15):
            return self.correlation_cache

        # Build price DataFrame
        prices_df = self._build_prices_dataframe()

        if prices_df is None or len(prices_df.columns) < 2:
            return {}

        # Calculate correlation matrix
        corr_matrix = calculate_correlation_matrix(prices_df, method="pearson")

        # Convert to dict format
        self.correlation_cache = {}
        for i, symbol1 in enumerate(corr_matrix.columns):
            for symbol2 in corr_matrix.columns[i + 1 :]:
                correlation = corr_matrix.loc[symbol1, symbol2]
                self.correlation_cache[(symbol1, symbol2)] = correlation
                self.correlation_cache[(symbol2, symbol1)] = correlation

        self.correlation_cache_time = datetime.utcnow()

        return self.correlation_cache

    def calculate_max_drawdown(self) -> Dict[str, Any]:
        """Calculate maximum drawdown analysis.

        Returns:
            Dictionary with drawdown metrics
        """
        if len(self.portfolio_value_history) < 2:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_abs": 0.0,
                "max_drawdown_duration": 0,
                "current_drawdown": 0.0,
            }

        # Create series from history
        portfolio_series = pd.Series(self.portfolio_value_history)

        # Calculate drawdown
        drawdown_analysis = calculate_max_drawdown(portfolio_series)

        self.last_drawdown_analysis = drawdown_analysis

        # Check drawdown limits
        allowed, message = self.limits.check_drawdown_limit(
            current_drawdown=drawdown_analysis["current_drawdown"],
            hard_limit=True,
        )

        if not allowed:
            # Issue critical alert
            alert = Alert(
                alert_type=AlertType.DRAWDOWN_WARNING,
                severity=AlertSeverity.CRITICAL,
                message=message,
                metric_value=drawdown_analysis["current_drawdown"],
                limit_value=self.limits.config.max_drawdown_pct,
            )
            self.alert_manager.issue_alert(alert)

        return drawdown_analysis

    def _build_prices_dataframe(self) -> Optional[pd.DataFrame]:
        """Build DataFrame of historical prices for all positions.

        Returns:
            DataFrame with datetime index and symbol columns
        """
        if not self.price_history:
            return None

        # Combine all price histories
        series_list = []
        for symbol, series in self.price_history.items():
            if symbol in self.positions and len(series) > 1:
                series_list.append(series.rename(symbol))

        if not series_list:
            return None

        # Concat and align
        try:
            prices_df = pd.concat(series_list, axis=1).dropna()
            return prices_df
        except Exception as e:
            logger.error("Error building prices DataFrame: {}", e)
            return None

    def _run_portfolio_risk_checks(self) -> None:
        """Run all portfolio-level risk checks."""
        # Check daily loss limits
        allowed, message = self.limits.check_daily_loss(
            daily_pnl=self.daily_pnl,
            portfolio_value=self.account_balance,
            hard_limit=True,
        )

        if not allowed:
            alert = Alert(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                severity=AlertSeverity.CRITICAL,
                message=message,
                metric_value=abs(self.daily_pnl) / self.account_balance if self.account_balance > 0 else 0,
                limit_value=self.limits.config.max_daily_loss_pct,
            )
            self.alert_manager.issue_alert(alert)

        # Check trading limits
        allowed, message = self.limits.check_trading_limits(
            num_trades_today=self.trades_today,
            hard_limit=True,
        )

        if not allowed:
            alert = Alert(
                alert_type=AlertType.TRADING_LIMIT_EXCEEDED,
                severity=AlertSeverity.WARNING,
                message=message,
                metric_value=self.trades_today,
                limit_value=self.limits.config.max_trades_per_day,
            )
            self.alert_manager.issue_alert(alert)

        # Check correlation exposure
        if len(self.positions) > 1:
            correlation_data = self.get_correlation_matrix()
            positions_list = [
                {"symbol": s, "value": p.quantity * p.current_price}
                for s, p in self.positions.items()
            ]

            allowed, message = self.limits.check_correlation_exposure(
                positions=positions_list,
                correlation_matrix=correlation_data,
                hard_limit=False,  # Warn only
            )

            if not allowed:
                alert = Alert(
                    alert_type=AlertType.CORRELATION_RISK,
                    severity=AlertSeverity.WARNING,
                    message=message,
                )
                self.alert_manager.issue_alert(alert)

    def _trigger_stop_loss(self, symbol: str) -> None:
        """Trigger stop-loss for a position.

        Args:
            symbol: Asset symbol
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            logger.warning(
                "Stop-loss triggered for {} at ${:.2f}",
                symbol,
                position.current_price,
            )

            alert = Alert(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                severity=AlertSeverity.WARNING,
                message=f"Stop-loss triggered for {symbol} at ${position.current_price:.2f}",
                symbol=symbol,
                metric_value=position.current_price,
                limit_value=position.stop_loss,
            )
            self.alert_manager.issue_alert(alert)

            # Close position
            self.close_position(symbol, position.stop_loss)

    def _trigger_take_profit(self, symbol: str) -> None:
        """Trigger take-profit for a position.

        Args:
            symbol: Asset symbol
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            logger.info(
                "Take-profit triggered for {} at ${:.2f}",
                symbol,
                position.current_price,
            )

            alert = Alert(
                alert_type=AlertType.PORTFOLIO_REBALANCE_NEEDED,
                severity=AlertSeverity.INFO,
                message=f"Take-profit triggered for {symbol} at ${position.current_price:.2f}",
                symbol=symbol,
                metric_value=position.current_price,
                limit_value=position.take_profit,
            )
            self.alert_manager.issue_alert(alert)

            # Close position
            self.close_position(symbol, position.take_profit)

    def _store_position_in_graph(self, position: Position) -> None:
        """Store position in knowledge graph.

        Args:
            position: Position to store
        """
        try:
            cypher = """
            CREATE (p:Position {
                id: randomUUID(),
                symbol: $symbol,
                quantity: $quantity,
                entry_price: $entry_price,
                current_price: $current_price,
                side: $side,
                sector: $sector,
                asset_class: $asset_class,
                entry_time: datetime($entry_time),
                stop_loss: $stop_loss,
                take_profit: $take_profit
            })
            """

            self.kg.write(
                cypher,
                symbol=position.symbol,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=position.current_price,
                side=position.side,
                sector=position.sector,
                asset_class=position.asset_class,
                entry_time=position.entry_time.isoformat(),
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
            )

        except Exception as e:
            logger.error("Failed to store position in knowledge graph: {}", e)

    def _store_closed_trade_in_graph(
        self,
        position: Position,
        exit_price: float,
        pnl: float,
    ) -> None:
        """Store closed trade in knowledge graph.

        Args:
            position: Closed position
            exit_price: Exit price
            pnl: Realized P&L
        """
        try:
            cypher = """
            CREATE (t:ClosedTrade {
                id: randomUUID(),
                symbol: $symbol,
                entry_price: $entry_price,
                exit_price: $exit_price,
                quantity: $quantity,
                side: $side,
                pnl: $pnl,
                close_time: datetime($close_time),
                entry_time: datetime($entry_time)
            })
            """

            self.kg.write(
                cypher,
                symbol=position.symbol,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                side=position.side,
                pnl=pnl,
                close_time=datetime.utcnow().isoformat(),
                entry_time=position.entry_time.isoformat(),
            )

        except Exception as e:
            logger.error("Failed to store closed trade in knowledge graph: {}", e)

    def _store_portfolio_risk_in_graph(self, risk_metrics: Dict[str, float]) -> None:
        """Store portfolio risk metrics in knowledge graph.

        Args:
            risk_metrics: Risk metrics dictionary
        """
        try:
            cypher = """
            CREATE (r:PortfolioRisk {
                id: randomUUID(),
                portfolio_value: $portfolio_value,
                var_95: $var_95,
                cvar_95: $cvar_95,
                portfolio_std: $portfolio_std,
                worst_case_loss: $worst_case_loss,
                timestamp: datetime($timestamp)
            })
            """

            self.kg.write(
                cypher,
                portfolio_value=risk_metrics.get("portfolio_value", 0),
                var_95=risk_metrics.get("var_95", 0),
                cvar_95=risk_metrics.get("cvar_95", 0),
                portfolio_std=risk_metrics.get("portfolio_std", 0),
                worst_case_loss=risk_metrics.get("worst_case_loss", 0),
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            logger.error("Failed to store portfolio risk in knowledge graph: {}", e)

    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (called at start of trading day)."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        logger.info("Daily metrics reset")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary.

        Returns:
            Dictionary with risk summary
        """
        portfolio_state = self.get_portfolio_state()

        summary = {
            "account_balance": self.account_balance,
            "total_pnl": self.account_balance - self.initial_balance,
            "daily_pnl": self.daily_pnl,
            "num_positions": len(self.positions),
            "portfolio_risk": self.last_portfolio_risk,
            "drawdown": self.last_drawdown_analysis,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "alert_statistics": self.alert_manager.get_alert_statistics(),
        }

        return summary
