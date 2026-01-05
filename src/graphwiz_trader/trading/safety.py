"""Comprehensive safety checks for trading operations."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


class SafetyCheckError(Exception):
    """Raised when safety check fails."""

    pass


class SafetyViolation:
    """Represents a safety violation."""

    def __init__(
        self, check_type: str, severity: str, message: str, details: Optional[Dict[str, Any]] = None
    ):
        """Initialize safety violation.

        Args:
            check_type: Type of safety check
            severity: Severity level (critical, warning, info)
            message: Violation message
            details: Additional details
        """
        self.check_type = check_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class SafetyChecks:
    """Comprehensive safety checks for trading operations."""

    def __init__(
        self,
        knowledge_graph,
        config: Optional[Dict[str, Any]] = None,
        violation_callback: Optional[Callable] = None,
    ):
        """Initialize safety checks.

        Args:
            knowledge_graph: Knowledge graph instance
            config: Safety configuration
            violation_callback: Optional callback for safety violations
        """
        self.kg = knowledge_graph
        self.config = config or {}
        self.violation_callback = violation_callback

        # Load configuration
        self._load_config()

        # State tracking
        self.daily_pnl = Decimal("0")
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now(timezone.utc).date()

        # Violation tracking
        self.violations: List[SafetyViolation] = []
        self.circuit_breaker_active = False

        # API rate limiting
        self.api_calls: Dict[str, List[datetime]] = {}

        logger.info("Safety checks initialized")

    def _load_config(self) -> None:
        """Load safety configuration."""
        safety_config = self.config.get("safety", {})

        # Position size limits
        self.max_position_size_pct = Decimal(
            str(safety_config.get("max_position_size_pct", 30.0))
        ) / Decimal("100")

        self.max_position_value = Decimal(str(safety_config.get("max_position_value", 50000.0)))

        # Daily loss limits
        self.daily_loss_limit_soft = Decimal(
            str(safety_config.get("daily_loss_limit_soft_pct", 3.0))
        ) / Decimal("100")

        self.daily_loss_limit_hard = Decimal(
            str(safety_config.get("daily_loss_limit_hard_pct", 5.0))
        ) / Decimal("100")

        # Drawdown limits
        self.max_drawdown_pct = Decimal(str(safety_config.get("max_drawdown_pct", 10.0))) / Decimal(
            "100"
        )

        # Trade limits
        self.max_daily_trades = safety_config.get("max_daily_trades", 100)
        self.max_trades_per_hour = safety_config.get("max_trades_per_hour", 20)

        # API rate limits
        self.api_rate_limit = safety_config.get("api_rate_limit", 1200)  # per minute
        self.api_burst_limit = safety_config.get("api_burst_limit", 100)  # per second

        # Circuit breaker
        self.circuit_breaker_threshold = Decimal(
            str(safety_config.get("circuit_breaker_threshold_pct", 15.0))
        ) / Decimal("100")

        self.circuit_breaker_duration = safety_config.get("circuit_breaker_duration_minutes", 30)

        # Market conditions
        self.volatility_threshold = Decimal(
            str(safety_config.get("volatility_threshold", 5.0))
        ) / Decimal("100")

        # Balance checks
        self.min_balance_threshold = Decimal(str(safety_config.get("min_balance_threshold", 100.0)))

        # Whitelist/blacklist
        self.symbol_whitelist = set(safety_config.get("symbol_whitelist", []))
        self.symbol_blacklist = set(safety_config.get("symbol_blacklist", []))

    async def pre_trade_validation(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        exchange_name: str,
    ) -> Tuple[bool, List[SafetyViolation]]:
        """Comprehensive pre-trade validation.

        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            amount: Trade amount
            price: Trade price
            portfolio_value: Current portfolio value
            current_positions: Current positions
            exchange_name: Exchange name

        Returns:
            Tuple of (passed, violations)
        """
        violations = []
        all_passed = True

        try:
            # Reset daily counters if new day
            self._check_daily_reset()

            # Check circuit breaker
            if self.circuit_breaker_active:
                if not await self._check_circuit_breaker_expired():
                    violations.append(
                        SafetyViolation(
                            check_type="circuit_breaker",
                            severity="critical",
                            message="Circuit breaker is active - trading halted",
                            details={"active_since": self.circuit_breaker_active},
                        )
                    )
                    all_passed = False
                else:
                    self.circuit_breaker_active = False
                    logger.warning("Circuit breaker expired - trading can resume")

            # Check if trading is allowed
            if not all_passed:
                return False, violations

            # Run all safety checks
            checks = [
                self._check_daily_limits,
                self._check_position_size,
                self._check_daily_loss_limit,
                self._check_symbol_allowed,
                self._check_api_rate_limit,
                self._check_portfolio_balance,
                self._check_trade_frequency,
            ]

            for check in checks:
                try:
                    result = await check(
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=price,
                        portfolio_value=portfolio_value,
                        current_positions=current_positions,
                        exchange_name=exchange_name,
                    )

                    if not result["passed"]:
                        all_passed = False
                        if "violation" in result:
                            violations.append(result["violation"])

                except Exception as e:
                    logger.error("Safety check failed with error: {}", e)
                    violations.append(
                        SafetyViolation(
                            check_type="check_error",
                            severity="warning",
                            message=f"Safety check error: {str(e)}",
                            details={"check": check.__name__},
                        )
                    )

            # Log violations
            if violations:
                for violation in violations:
                    self.violations.append(violation)
                    await self._log_violation(violation)

                    # Call violation callback
                    if self.violation_callback:
                        try:
                            await self.violation_callback(violation)
                        except Exception as e:
                            logger.error("Violation callback failed: {}", e)

            return all_passed, violations

        except Exception as e:
            logger.error("Pre-trade validation failed: {}", e)
            return False, [
                SafetyViolation(
                    check_type="validation_error",
                    severity="critical",
                    message=f"Validation error: {str(e)}",
                )
            ]

    async def _check_daily_limits(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        exchange_name: str,
    ) -> Dict[str, Any]:
        """Check daily trade limits.

        Returns:
            Check result
        """
        if self.daily_trade_count >= self.max_daily_trades:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="daily_trade_limit",
                    severity="critical",
                    message=f"Daily trade limit reached: {self.daily_trade_count}/{self.max_daily_trades}",
                    details={"current": self.daily_trade_count, "limit": self.max_daily_trades},
                ),
            }

        return {"passed": True}

    async def _check_position_size(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        exchange_name: str,
    ) -> Dict[str, Any]:
        """Check position size limits.

        Returns:
            Check result
        """
        position_value = Decimal(str(amount * price))

        # Check max position value
        if position_value > self.max_position_value:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="position_size",
                    severity="critical",
                    message=f"Position value exceeds maximum: ${float(position_value):.2f} > ${float(self.max_position_value):.2f}",
                    details={
                        "position_value": float(position_value),
                        "max_value": float(self.max_position_value),
                    },
                ),
            }

        # Check position as percentage of portfolio
        position_pct = (
            position_value / Decimal(str(portfolio_value)) if portfolio_value > 0 else Decimal("0")
        )

        if position_pct > self.max_position_size_pct:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="position_size",
                    severity="critical",
                    message=f"Position size exceeds maximum: {float(position_pct*100):.2f}% > {float(self.max_position_size_pct*100):.2f}%",
                    details={
                        "position_pct": float(position_pct * 100),
                        "max_pct": float(self.max_position_size_pct * 100),
                    },
                ),
            }

        return {"passed": True}

    async def _check_daily_loss_limit(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        exchange_name: str,
    ) -> Dict[str, Any]:
        """Check daily loss limits.

        Returns:
            Check result
        """
        initial_portfolio = Decimal(str(portfolio_value)) + self.daily_pnl

        if initial_portfolio <= 0:
            return {"passed": True}

        loss_pct = abs(self.daily_pnl) / initial_portfolio if self.daily_pnl < 0 else Decimal("0")

        # Hard limit
        if loss_pct >= self.daily_loss_limit_hard:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="daily_loss_limit",
                    severity="critical",
                    message=f"Hard daily loss limit reached: {float(loss_pct*100):.2f}%",
                    details={
                        "loss_pct": float(loss_pct * 100),
                        "limit": float(self.daily_loss_limit_hard * 100),
                        "daily_pnl": float(self.daily_pnl),
                    },
                    severity="critical",
                ),
            }

        # Soft limit (warning only)
        if loss_pct >= self.daily_loss_limit_soft:
            return {
                "passed": True,
                "violation": SafetyViolation(
                    check_type="daily_loss_limit",
                    severity="warning",
                    message=f"Soft daily loss limit reached: {float(loss_pct*100):.2f}%",
                    details={
                        "loss_pct": float(loss_pct * 100),
                        "soft_limit": float(self.daily_loss_limit_soft * 100),
                        "hard_limit": float(self.daily_loss_limit_hard * 100),
                    },
                ),
            }

        return {"passed": True}

    async def _check_symbol_allowed(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        exchange_name: str,
    ) -> Dict[str, Any]:
        """Check if symbol is allowed for trading.

        Returns:
            Check result
        """
        # Check blacklist
        if self.symbol_blacklist and symbol in self.symbol_blacklist:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="symbol_restriction",
                    severity="critical",
                    message=f"Symbol is blacklisted: {symbol}",
                    details={"symbol": symbol},
                ),
            }

        # Check whitelist (if configured)
        if self.symbol_whitelist and symbol not in self.symbol_whitelist:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="symbol_restriction",
                    severity="critical",
                    message=f"Symbol not in whitelist: {symbol}",
                    details={"symbol": symbol, "whitelist": list(self.symbol_whitelist)},
                ),
            }

        return {"passed": True}

    async def _check_api_rate_limit(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        exchange_name: str,
    ) -> Dict[str, Any]:
        """Check API rate limits.

        Returns:
            Check result
        """
        now = datetime.now(timezone.utc)

        # Clean old calls
        if exchange_name in self.api_calls:
            # Remove calls older than 1 minute
            cutoff = now - timedelta(minutes=1)
            self.api_calls[exchange_name] = [
                call_time for call_time in self.api_calls[exchange_name] if call_time > cutoff
            ]

        # Check burst limit (per second)
        second_ago = now - timedelta(seconds=1)
        recent_calls = len(
            [
                call_time
                for call_time in self.api_calls.get(exchange_name, [])
                if call_time > second_ago
            ]
        )

        if recent_calls >= self.api_burst_limit:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="api_rate_limit",
                    severity="warning",
                    message="API burst limit reached - backing off",
                    details={"recent_calls": recent_calls, "burst_limit": self.api_burst_limit},
                ),
            }

        # Check rate limit (per minute)
        minute_calls = len(self.api_calls.get(exchange_name, []))

        if minute_calls >= self.api_rate_limit:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="api_rate_limit",
                    severity="warning",
                    message="API rate limit reached - backing off",
                    details={"minute_calls": minute_calls, "rate_limit": self.api_rate_limit},
                ),
            }

        # Record this call
        if exchange_name not in self.api_calls:
            self.api_calls[exchange_name] = []
        self.api_calls[exchange_name].append(now)

        return {"passed": True}

    async def _check_portfolio_balance(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        exchange_name: str,
    ) -> Dict[str, Any]:
        """Check portfolio balance is above minimum threshold.

        Returns:
            Check result
        """
        if portfolio_value < self.min_balance_threshold:
            return {
                "passed": False,
                "violation": SafetyViolation(
                    check_type="portfolio_balance",
                    severity="critical",
                    message=f"Portfolio balance below minimum: ${portfolio_value:.2f} < ${float(self.min_balance_threshold):.2f}",
                    details={
                        "current_balance": portfolio_value,
                        "min_threshold": float(self.min_balance_threshold),
                    },
                ),
            }

        return {"passed": True}

    async def _check_trade_frequency(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, Any],
        exchange_name: str,
    ) -> Dict[str, Any]:
        """Check trade frequency limits.

        Returns:
            Check result
        """
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)

        # Count trades in last hour (would need proper tracking in production)
        # For now, use daily_trade_count as proxy
        if self.daily_trade_count > 0:
            # Simplified check
            pass

        return {"passed": True}

    async def check_exchange_connectivity(self, exchange_name: str) -> bool:
        """Check if exchange connectivity is healthy.

        Args:
            exchange_name: Exchange to check

        Returns:
            True if exchange is healthy
        """
        # This would integrate with actual exchange health checks
        # For now, return True
        return True

    async def validate_api_keys(self, exchange_config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate API keys are properly configured.

        Args:
            exchange_config: Exchange configuration

        Returns:
            Tuple of (valid, message)
        """
        api_key = exchange_config.get("api_key")
        api_secret = exchange_config.get("api_secret")

        if not api_key or not api_secret:
            return False, "API key or secret is missing"

        if len(api_key) < 10 or len(api_secret) < 10:
            return False, "API key or secret appears invalid (too short)"

        # Check for default/test keys
        if api_key in ["your_api_key", "test_key", ""] or api_secret in [
            "your_api_secret",
            "test_secret",
            "",
        ]:
            return False, "Default or test API keys detected"

        return True, "API keys appear valid"

    async def activate_circuit_breaker(self, reason: str) -> None:
        """Activate circuit breaker.

        Args:
            reason: Reason for activation
        """
        self.circuit_breaker_active = datetime.now(timezone.utc)

        logger.error("!!! CIRCUIT BREAKER ACTIVATED !!!")
        logger.error("Reason: {}", reason)
        logger.error("Trading will be halted for {} minutes", self.circuit_breaker_duration)

        # Log to knowledge graph
        await self._log_circuit_breaker_activation(reason)

    async def _check_circuit_breaker_expired(self) -> bool:
        """Check if circuit breaker has expired.

        Returns:
            True if expired
        """
        if not isinstance(self.circuit_breaker_active, datetime):
            return False

        expiry_time = self.circuit_breaker_active + timedelta(minutes=self.circuit_breaker_duration)
        return datetime.now(timezone.utc) >= expiry_time

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        current_date = datetime.now(timezone.utc).date()

        if current_date != self.last_reset_date:
            self.daily_pnl = Decimal("0")
            self.daily_trade_count = 0
            self.last_reset_date = current_date
            logger.info("Daily safety counters reset")

    def update_daily_pnl(self, pnl: Decimal) -> None:
        """Update daily P&L.

        Args:
            pnl: Profit/loss to add
        """
        self.daily_pnl += pnl

    def increment_trade_count(self) -> None:
        """Increment daily trade count."""
        self.daily_trade_count += 1

    async def _log_violation(self, violation: SafetyViolation) -> None:
        """Log safety violation to knowledge graph.

        Args:
            violation: Safety violation
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (sv:SafetyViolation {
                        check_type: $check_type,
                        severity: $severity,
                        message: $message,
                        details: $details,
                        timestamp: datetime($timestamp)
                    })
                    RETURN sv
                    """,
                    **violation.to_dict(),
                )
                logger.debug("Logged safety violation to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log safety violation to graph: {}", e)

    async def _log_circuit_breaker_activation(self, reason: str) -> None:
        """Log circuit breaker activation to knowledge graph.

        Args:
            reason: Reason for activation
        """
        try:
            if self.kg:
                await asyncio.to_thread(
                    self.kg.write,
                    """
                    CREATE (cb:CircuitBreaker {
                        timestamp: datetime($timestamp),
                        reason: $reason,
                        duration_minutes: $duration,
                        active: true
                    })
                    RETURN cb
                    """,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    reason=reason,
                    duration=self.circuit_breaker_duration,
                )
                logger.debug("Logged circuit breaker activation to knowledge graph")
        except Exception as e:
            logger.warning("Failed to log circuit breaker to graph: {}", e)

    def get_violations(
        self, severity: Optional[str] = None, limit: int = 100
    ) -> List[SafetyViolation]:
        """Get safety violations.

        Args:
            severity: Filter by severity (optional)
            limit: Maximum number to return

        Returns:
            List of violations
        """
        violations = self.violations

        if severity:
            violations = [v for v in violations if v.severity == severity]

        return violations[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get current safety status.

        Returns:
            Status dictionary
        """
        return {
            "circuit_breaker_active": bool(self.circuit_breaker_active),
            "daily_pnl": float(self.daily_pnl),
            "daily_trade_count": self.daily_trade_count,
            "total_violations": len(self.violations),
            "api_rate_limit": self.api_rate_limit,
            "api_burst_limit": self.api_burst_limit,
        }

    def __repr__(self) -> str:
        return f"SafetyChecks(circuit_breaker={bool(self.circuit_breaker_active)})"
