"""Trading mode system for paper trading, simulated trading, and live trading."""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from loguru import logger


class TradingMode(Enum):
    """Trading modes supported by the system."""

    PAPER = "paper"  # Simulated execution with real market data
    SIMULATED = "simulated"  # Historical data replay for backtesting
    LIVE = "live"  # Real trading with actual funds


class ModeTransitionError(Exception):
    """Raised when mode transition fails validation."""

    pass


class TradingModeManager:
    """Manages trading modes and validates transitions."""

    def __init__(
        self,
        knowledge_graph,
        config: Optional[Dict[str, Any]] = None,
        approval_callback: Optional[Callable] = None,
    ):
        """Initialize trading mode manager.

        Args:
            knowledge_graph: Knowledge graph instance for logging
            config: Trading mode configuration
            approval_callback: Optional callback for live trading approval
        """
        self.kg = knowledge_graph
        self.config = config or {}
        self.approval_callback = approval_callback

        # Default to paper trading mode
        self.current_mode = TradingMode.PAPER
        self.previous_mode: Optional[TradingMode] = None
        self.mode_history: List[Dict[str, Any]] = []

        # Mode-specific settings
        self.mode_settings = {
            TradingMode.PAPER: {
                "enabled": True,
                "execution_delay": 0.0,
                "slippage_model": "realistic",
                "fee_rate": 0.001,
                "requires_approval": False,
            },
            TradingMode.SIMULATED: {
                "enabled": True,
                "data_source": "historical",
                "time_multiplier": 1.0,
                "requires_approval": False,
            },
            TradingMode.LIVE: {
                "enabled": False,  # Must be explicitly enabled
                "requires_approval": True,
                "emergency_stop": True,
                "max_drawdown_pct": 10.0,
                "daily_loss_limit_pct": 5.0,
            },
        }

        # Override with config
        if config:
            self._load_config(config)

        # Mode transition requirements
        self.transition_requirements = {
            "to_live": {
                "min_paper_days": 3,
                "min_trades": 100,
                "max_drawdown_pct": 10.0,
                "min_win_rate": 55.0,
                "min_sharpe_ratio": 1.5,
            }
        }

        # Emergency stop flag
        self.emergency_stop = False

        logger.info("Trading mode manager initialized in {} mode", self.current_mode.value)

    def _load_config(self, config: Dict[str, Any]) -> None:
        """Load configuration settings.

        Args:
            config: Configuration dictionary
        """
        mode_config = config.get("modes", {})

        for mode_name, settings in mode_config.items():
            try:
                mode = TradingMode(mode_name)
                if mode in self.mode_settings:
                    self.mode_settings[mode].update(settings)
            except ValueError:
                logger.warning("Invalid trading mode in config: {}", mode_name)

        # Load transition requirements
        if "transitions" in config:
            self.transition_requirements.update(config.get("transitions", {}))

    def get_current_mode(self) -> TradingMode:
        """Get current trading mode.

        Returns:
            Current trading mode
        """
        return self.current_mode

    def is_paper_trading(self) -> bool:
        """Check if currently in paper trading mode.

        Returns:
            True if in paper trading mode
        """
        return self.current_mode == TradingMode.PAPER

    def is_simulated_trading(self) -> bool:
        """Check if currently in simulated trading mode.

        Returns:
            True if in simulated trading mode
        """
        return self.current_mode == TradingMode.SIMULATED

    def is_live_trading(self) -> bool:
        """Check if currently in live trading mode.

        Returns:
            True if in live trading mode
        """
        return self.current_mode == TradingMode.LIVE

    def can_execute_live_trades(self) -> bool:
        """Check if live trades can be executed.

        Returns:
            True if live trading is enabled and emergency stop is not active
        """
        return (
            self.current_mode == TradingMode.LIVE
            and not self.emergency_stop
            and self.mode_settings[TradingMode.LIVE]["enabled"]
        )

    async def switch_mode(
        self, new_mode: Union[TradingMode, str], force: bool = False, reason: Optional[str] = None
    ) -> bool:
        """Switch to a different trading mode.

        Args:
            new_mode: New trading mode
            force: Force transition without validation (DANGEROUS)
            reason: Reason for mode switch

        Returns:
            True if mode switch successful

        Raises:
            ModeTransitionError: If transition validation fails
        """
        # Normalize mode
        if isinstance(new_mode, str):
            try:
                new_mode = TradingMode(new_mode.lower())
            except ValueError:
                raise ModeTransitionError(f"Invalid trading mode: {new_mode}")

        # Check if already in this mode
        if new_mode == self.current_mode:
            logger.info("Already in {} mode", new_mode.value)
            return True

        # Validate mode is enabled
        if not self.mode_settings[new_mode]["enabled"]:
            raise ModeTransitionError(
                f"{new_mode.value.upper()} mode is not enabled in configuration"
            )

        # Special validation for LIVE mode
        if new_mode == TradingMode.LIVE and not force:
            # Check if requires approval
            if self.mode_settings[TradingMode.LIVE]["requires_approval"]:
                logger.warning("ATTEMPTING TO SWITCH TO LIVE TRADING MODE")
                logger.warning("This will trade with REAL MONEY")

                # Call approval callback if provided
                if self.approval_callback:
                    approved = await self._get_approval(reason)
                    if not approved:
                        raise ModeTransitionError("Live trading approval denied")
                else:
                    logger.error("No approval callback configured for live trading")
                    raise ModeTransitionError(
                        "Cannot switch to live trading without approval mechanism"
                    )

        # Validate transition requirements
        if not force:
            await self._validate_transition(new_mode)

        # Perform mode switch
        previous_mode = self.current_mode
        self.previous_mode = previous_mode
        self.current_mode = new_mode

        # Log mode change
        timestamp = datetime.now(timezone.utc)
        mode_change = {
            "event_id": str(uuid4()),
            "timestamp": timestamp.isoformat(),
            "previous_mode": previous_mode.value,
            "new_mode": new_mode.value,
            "reason": reason or "Manual mode switch",
            "forced": force,
        }
        self.mode_history.append(mode_change)

        # Log to knowledge graph
        await self._log_mode_change(mode_change)

        # Log the change
        logger.warning(
            "TRADING MODE CHANGED: {} -> {} (reason: {})",
            previous_mode.value.upper(),
            new_mode.value.upper(),
            reason or "manual switch",
        )

        if new_mode == TradingMode.LIVE:
            logger.warning("!!! LIVE TRADING IS NOW ACTIVE !!!")
            logger.warning("!!! REAL MONEY IS AT RISK !!!")

        return True

    async def _validate_transition(self, new_mode: TradingMode) -> None:
        """Validate mode transition requirements.

        Args:
            new_mode: Target trading mode

        Raises:
            ModeTransitionError: If validation fails
        """
        # Transition to LIVE requires validation
        if new_mode == TradingMode.LIVE:
            logger.info("Validating transition to LIVE trading mode...")
            errors = []

            # Check if emergency stop is active
            if self.emergency_stop:
                errors.append("Emergency stop is active - clear before switching to live")

            # Validate paper trading performance
            if self.previous_mode == TradingMode.PAPER or self.current_mode == TradingMode.PAPER:
                validation = await self._validate_paper_trading_requirements()
                if not validation["passed"]:
                    errors.extend(validation["errors"])

            if errors:
                error_msg = "Live trading validation failed:\n" + "\n".join(
                    f"  - {e}" for e in errors
                )
                raise ModeTransitionError(error_msg)

            logger.info("Live trading validation passed")

    async def _validate_paper_trading_requirements(self) -> Dict[str, Any]:
        """Validate paper trading requirements for live trading.

        Returns:
            Dictionary with validation results
        """
        # This will be called with paper trading statistics
        # For now, return placeholder - will be implemented in transition manager
        return {"passed": True, "errors": [], "warnings": []}

    async def _get_approval(self, reason: Optional[str]) -> bool:
        """Get approval for live trading mode.

        Args:
            reason: Reason for mode switch

        Returns:
            True if approved
        """
        if self.approval_callback:
            try:
                result = await self.approval_callback(
                    current_mode=self.current_mode.value, target_mode="live", reason=reason
                )
                return result
            except Exception as e:
                logger.error("Approval callback failed: {}", e)
                return False

        # Default: require manual confirmation
        logger.warning("!!! LIVE TRADING APPROVAL REQUIRED !!!")
        logger.warning("To approve, set force=True in switch_mode()")
        return False

    async def emergency_stop_mode(self, reason: Optional[str] = None) -> None:
        """Activate emergency stop for live trading.

        Args:
            reason: Reason for emergency stop
        """
        if self.current_mode == TradingMode.LIVE:
            self.emergency_stop = True
            logger.error("!!! EMERGENCY STOP ACTIVATED !!!")
            logger.error("All trading operations will be halted")
            logger.error("Reason: {}", reason or "Manual emergency stop")

            # Log to knowledge graph
            await self._log_emergency_stop(reason)

    async def clear_emergency_stop(self, reason: Optional[str] = None) -> None:
        """Clear emergency stop status.

        Args:
            reason: Reason for clearing
        """
        if self.emergency_stop:
            self.emergency_stop = False
            logger.warning("Emergency stop cleared: {}", reason or "Manual clear")
            logger.warning("Trading operations can resume")

            # Log to knowledge graph
            await self._log_emergency_stop_clear(reason)

    def get_mode_settings(self, mode: Optional[TradingMode] = None) -> Dict[str, Any]:
        """Get settings for a trading mode.

        Args:
            mode: Trading mode (uses current if None)

        Returns:
            Mode settings dictionary
        """
        if mode is None:
            mode = self.current_mode

        return self.mode_settings.get(mode, {}).copy()

    def get_mode_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get mode change history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of mode change events
        """
        return self.mode_history[-limit:]

    async def _log_mode_change(self, mode_change: Dict[str, Any]) -> None:
        """Log mode change to knowledge graph.

        Args:
            mode_change: Mode change event data
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (mc:TradingModeChange {
                        event_id: $event_id,
                        timestamp: datetime($timestamp),
                        previous_mode: $previous_mode,
                        new_mode: $new_mode,
                        reason: $reason,
                        forced: $forced
                    })
                    RETURN mc
                    """,
                    **mode_change,
                )
                logger.debug("Logged mode change to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log mode change to graph: {}", e)

    async def _log_emergency_stop(self, reason: Optional[str]) -> None:
        """Log emergency stop to knowledge graph.

        Args:
            reason: Reason for emergency stop
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (es:EmergencyStop {
                        event_id: $event_id,
                        timestamp: datetime($timestamp),
                        mode: $mode,
                        reason: $reason,
                        active: true
                    })
                    RETURN es
                    """,
                    event_id=str(uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    mode=self.current_mode.value,
                    reason=reason or "Manual emergency stop",
                )
                logger.debug("Logged emergency stop to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log emergency stop to graph: {}", e)

    async def _log_emergency_stop_clear(self, reason: Optional[str]) -> None:
        """Log emergency stop clear to knowledge graph.

        Args:
            reason: Reason for clearing
        """
        try:
            if self.kg:
                # Update any active emergency stops
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    MATCH (es:EmergencyStop {active: true})
                    SET es.active = false,
                        es.cleared_at = datetime($timestamp),
                        es.clear_reason = $reason
                    RETURN es
                    """,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    reason=reason or "Manual clear",
                )
                logger.debug("Logged emergency stop clear to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log emergency stop clear to graph: {}", e)

    def get_status(self) -> Dict[str, Any]:
        """Get current trading mode status.

        Returns:
            Status dictionary
        """
        return {
            "current_mode": self.current_mode.value,
            "previous_mode": self.previous_mode.value if self.previous_mode else None,
            "can_execute_live": self.can_execute_live_trades(),
            "emergency_stop_active": self.emergency_stop,
            "mode_settings": {
                mode.value: settings for mode, settings in self.mode_settings.items()
            },
            "mode_changes_count": len(self.mode_history),
        }

    def __repr__(self) -> str:
        return f"TradingModeManager(mode={self.current_mode.value}, emergency_stop={self.emergency_stop})"
