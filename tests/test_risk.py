"""
Tests for risk management.

These tests verify that the risk manager correctly gates trades
based on profitability, staleness, drawdown, and consecutive losses.
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from triangular_arb.config import RiskConfig
from triangular_arb.risk.manager import RejectionReason, RiskManager
from triangular_arb.types import (
    ArbitrageResult,
    Direction,
    Opportunity,
    OrderBook,
    Pair,
    PriceLevel,
    Side,
    Symbol,
    Triangle,
)


def _make_opportunity(
    net_profit_bps: str = "10",
    book_age_ns: int | None = None,
) -> Opportunity:
    """Create a test opportunity."""
    ts = book_age_ns or time.time_ns()
    triangle = Triangle(
        base=Symbol("ETH"),
        leg1_pair=Pair("LTC/ETH"),
        leg1_side=Side.BUY,
        leg2_pair=Pair("LTC/BTC"),
        leg2_side=Side.SELL,
        leg3_pair=Pair("ETH/BTC"),
        leg3_side=Side.BUY,
        intermediate_a=Symbol("LTC"),
        intermediate_b=Symbol("BTC"),
    )
    book = OrderBook(
        pair=Pair("LTC/ETH"),
        bids=(PriceLevel(Decimal("100"), Decimal("10")),),
        asks=(PriceLevel(Decimal("101"), Decimal("10")),),
        timestamp_ns=ts,
    )
    return Opportunity(
        triangle=triangle,
        direction=Direction.FORWARD,
        gross_profit_bps=Decimal(net_profit_bps) + Decimal("3"),
        net_profit_bps=Decimal(net_profit_bps),
        estimated_size=Decimal("1"),
        books=(book, book, book),
    )


def _make_result(success: bool, profit: str = "0.01") -> ArbitrageResult:
    """Create a test result."""
    opp = _make_opportunity()
    return ArbitrageResult(
        opportunity=opp,
        fills=(),
        net_profit=Decimal(profit) if success else Decimal(f"-{profit}"),
        net_profit_bps=Decimal("5") if success else Decimal("-5"),
        total_fees=Decimal("0.001"),
        total_latency_ms=2.5,
        success=success,
    )


class TestRiskManager:
    """Tests for the RiskManager class."""

    def test_accepts_profitable_opportunity(self) -> None:
        """Opportunity above min threshold should be accepted."""
        rm = RiskManager(RiskConfig(min_profit_bps=Decimal("5")))
        opp = _make_opportunity(net_profit_bps="10")
        assert rm.check(opp) is None

    def test_rejects_below_min_profit(self) -> None:
        """Opportunity below min threshold should be rejected."""
        rm = RiskManager(RiskConfig(min_profit_bps=Decimal("20")))
        opp = _make_opportunity(net_profit_bps="10")
        assert rm.check(opp) == RejectionReason.BELOW_MIN_PROFIT

    def test_rejects_stale_books(self) -> None:
        """Books older than threshold should be rejected."""
        rm = RiskManager(RiskConfig(stale_book_ms=1000))
        old_ts = time.time_ns() - 5_000_000_000  # 5 seconds ago
        opp = _make_opportunity(net_profit_bps="10", book_age_ns=old_ts)
        assert rm.check(opp) == RejectionReason.STALE_BOOKS

    def test_circuit_breaker_after_consecutive_losses(self) -> None:
        """After N consecutive losses, circuit breaker should trip."""
        rm = RiskManager(RiskConfig(max_consecutive_losses=3))

        for _ in range(3):
            rm.record_result(_make_result(success=False))

        opp = _make_opportunity(net_profit_bps="50")
        result = rm.check(opp)
        assert result == RejectionReason.CONSECUTIVE_LOSSES

    def test_consecutive_losses_reset_on_win(self) -> None:
        """A winning trade should reset the consecutive loss counter."""
        rm = RiskManager(RiskConfig(max_consecutive_losses=5))

        # 2 losses then a win
        rm.record_result(_make_result(success=False))
        rm.record_result(_make_result(success=False))
        rm.record_result(_make_result(success=True))

        assert rm.state.consecutive_losses == 0

    def test_daily_loss_limit(self) -> None:
        """Exceeding daily loss limit should reject new trades."""
        rm = RiskManager(RiskConfig(max_daily_loss_pct=Decimal("5")))
        rm.set_starting_balance(Decimal("100"))

        # Simulate a large loss
        result = _make_result(success=False, profit="6")
        rm.record_result(result)

        opp = _make_opportunity(net_profit_bps="50")
        assert rm.check(opp) == RejectionReason.DAILY_LOSS_LIMIT

    def test_win_rate_calculation(self) -> None:
        """Win rate should be calculated correctly."""
        rm = RiskManager(RiskConfig())

        rm.record_result(_make_result(success=True))
        rm.record_result(_make_result(success=True))
        rm.record_result(_make_result(success=False))

        assert rm.win_rate == pytest.approx(66.67, rel=0.01)

    def test_win_rate_zero_trades(self) -> None:
        """Win rate with no trades should be 0."""
        rm = RiskManager(RiskConfig())
        assert rm.win_rate == 0.0
