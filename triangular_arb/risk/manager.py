"""
Risk manager — the last line of defense before execution.

Enforces position limits, drawdown circuit breakers, and staleness checks.
Every opportunity must pass through here before execution.
The risk manager can only reject — it never modifies opportunities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

import structlog

from triangular_arb.config import RiskConfig
from triangular_arb.types import ArbitrageResult, Opportunity

log = structlog.get_logger()


class RejectionReason(Enum):
    BELOW_MIN_PROFIT = "below_min_profit_threshold"
    STALE_BOOKS = "order_books_too_stale"
    DAILY_LOSS_LIMIT = "daily_loss_limit_reached"
    CONSECUTIVE_LOSSES = "consecutive_loss_limit_reached"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_is_open"
    TOO_MANY_OPEN = "too_many_open_triangles"


@dataclass
class RiskState:
    """Mutable risk state that tracks session statistics."""
    daily_pnl: Decimal = Decimal("0")
    starting_balance: Decimal = Decimal("0")
    consecutive_losses: int = 0
    circuit_breaker_open: bool = False
    circuit_breaker_until_ns: int = 0
    open_triangles: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_pnl: Decimal = Decimal("0")
    session_start_ns: int = field(default_factory=time.time_ns)


class RiskManager:
    """
    Pre-trade risk gate.

    This component exists to prevent the system from destroying itself.
    A firm like Jane Street would have this as an independent process
    with kill switches — here we implement the same logic inline.
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._state = RiskState()

    def check(self, opportunity: Opportunity) -> RejectionReason | None:
        """
        Evaluate whether an opportunity should be executed.

        Returns None if the opportunity passes all checks,
        or a RejectionReason if it should be rejected.
        """
        now_ns = time.time_ns()

        # Circuit breaker check
        if self._state.circuit_breaker_open:
            if now_ns < self._state.circuit_breaker_until_ns:
                return RejectionReason.CIRCUIT_BREAKER_OPEN
            else:
                log.info("circuit_breaker_reset")
                self._state.circuit_breaker_open = False
                self._state.consecutive_losses = 0

        # Minimum profit threshold
        if opportunity.net_profit_bps < self._config.min_profit_bps:
            return RejectionReason.BELOW_MIN_PROFIT

        # Order book staleness
        now_ns = time.time_ns()
        for book in opportunity.books:
            age_ms = (now_ns - book.timestamp_ns) / 1_000_000
            if age_ms > self._config.stale_book_ms:
                return RejectionReason.STALE_BOOKS

        # Daily loss limit
        if self._state.starting_balance > 0:
            loss_pct = (-self._state.daily_pnl / self._state.starting_balance) * 100
            if loss_pct >= self._config.max_daily_loss_pct:
                return RejectionReason.DAILY_LOSS_LIMIT

        # Consecutive losses
        if self._state.consecutive_losses >= self._config.max_consecutive_losses:
            self._trip_circuit_breaker(duration_s=300)  # 5-minute cooldown
            return RejectionReason.CONSECUTIVE_LOSSES

        # Open position limit
        if self._state.open_triangles >= self._config.max_open_triangles:
            return RejectionReason.TOO_MANY_OPEN

        return None

    def record_result(self, result: ArbitrageResult) -> None:
        """Update risk state after a trade completes."""
        self._state.total_trades += 1
        self._state.total_pnl += result.net_profit
        self._state.daily_pnl += result.net_profit

        if result.success and result.net_profit > 0:
            self._state.total_wins += 1
            self._state.consecutive_losses = 0
        elif not result.success or result.net_profit < 0:
            self._state.consecutive_losses += 1

        log.info(
            "risk_state_updated",
            total_trades=self._state.total_trades,
            win_rate=self.win_rate,
            daily_pnl=float(self._state.daily_pnl),
            consecutive_losses=self._state.consecutive_losses,
        )

    def set_starting_balance(self, balance: Decimal) -> None:
        """Set the reference balance for drawdown calculations."""
        self._state.starting_balance = balance

    def _trip_circuit_breaker(self, duration_s: int) -> None:
        """Activate the circuit breaker for a specified duration."""
        self._state.circuit_breaker_open = True
        self._state.circuit_breaker_until_ns = time.time_ns() + duration_s * 1_000_000_000
        log.warning(
            "circuit_breaker_tripped",
            consecutive_losses=self._state.consecutive_losses,
            cooldown_s=duration_s,
        )

    @property
    def win_rate(self) -> float:
        if self._state.total_trades == 0:
            return 0.0
        return self._state.total_wins / self._state.total_trades * 100

    @property
    def state(self) -> RiskState:
        return self._state
