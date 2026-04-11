"""Tests for stealth execution and anti-frontrunning."""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from triangular_arb.execution.stealth import (
    CompetitionState,
    adaptive_threshold,
    opportunity_decay,
    randomize_size,
    triangle_heat,
)
from triangular_arb.types import (
    Direction,
    Opportunity,
    OrderBook,
    Pair,
    PriceLevel,
    Side,
    Symbol,
    Triangle,
)


def _make_opportunity(age_ms: float = 0) -> Opportunity:
    """Create a test opportunity with a specific age."""
    ts = time.time_ns() - int(age_ms * 1_000_000)
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
        gross_profit_bps=Decimal("15"),
        net_profit_bps=Decimal("10"),
        estimated_size=Decimal("1"),
        books=(book, book, book),
        detected_at_ns=ts,
    )


class TestRandomizeSize:
    def test_output_differs_from_input(self) -> None:
        """Noise should change the size (probabilistically)."""
        size = Decimal("1.0")
        results = {randomize_size(size) for _ in range(20)}
        # With 20 samples at 5% noise, extremely unlikely all are identical
        assert len(results) > 1

    def test_output_is_positive(self) -> None:
        """Noisy size should always be positive."""
        for _ in range(100):
            result = randomize_size(Decimal("0.001"))
            assert result > 0

    def test_output_precision(self) -> None:
        """Output should have at most 8 decimal places."""
        result = randomize_size(Decimal("1.23456789012345"))
        assert result == result.quantize(Decimal("0.00000001"))


class TestOpportunityDecay:
    def test_fresh_opportunity_no_decay(self) -> None:
        """Age ~0 should return approximately full value."""
        opp = _make_opportunity(age_ms=0)
        decayed = opportunity_decay(opp, half_life_ms=200)
        assert decayed >= Decimal("9.5")  # ~10 bps, small rounding tolerance

    def test_old_opportunity_decays(self) -> None:
        """After several half-lives, value should be near zero."""
        opp = _make_opportunity(age_ms=1000)  # 5 half-lives at 200ms
        decayed = opportunity_decay(opp, half_life_ms=200)
        assert decayed < Decimal("1")  # Should be ~0.31 bps

    def test_half_life_halves_value(self) -> None:
        """At exactly one half-life, value should be ~half."""
        opp = _make_opportunity(age_ms=200)
        decayed = opportunity_decay(opp, half_life_ms=200)
        # 10 * 0.5 = 5.0 bps
        assert Decimal("4") < decayed < Decimal("6")


class TestAdaptiveThreshold:
    def test_high_competition_raises_threshold(self) -> None:
        """When we're losing, threshold should increase."""
        state = CompetitionState()
        for _ in range(20):
            state.record(False)  # All losses

        base = Decimal("5")
        adjusted = adaptive_threshold(base, state)
        assert adjusted > base

    def test_low_competition_lowers_threshold(self) -> None:
        """When we're winning, threshold should decrease."""
        state = CompetitionState()
        for _ in range(20):
            state.record(True)  # All wins

        base = Decimal("5")
        adjusted = adaptive_threshold(base, state)
        assert adjusted < base

    def test_neutral_competition(self) -> None:
        """50% win rate should roughly maintain threshold."""
        state = CompetitionState()
        for i in range(20):
            state.record(i % 2 == 0)

        base = Decimal("5")
        adjusted = adaptive_threshold(base, state)
        assert Decimal("4") < adjusted < Decimal("8")


class TestTriangleHeat:
    def test_cold_triangle_not_hot(self) -> None:
        """Never-traded triangle should be allowed."""
        state = CompetitionState()
        assert not triangle_heat("ETH-LTC-BTC", state)

    def test_frequently_traded_triangle_is_hot(self) -> None:
        """Heavily traded triangle should be cooled off."""
        state = CompetitionState()
        # Simulate many executions
        for _ in range(10):
            state.record_execution("ETH-LTC-BTC")
        assert triangle_heat("ETH-LTC-BTC", state)


class TestCompetitionState:
    def test_win_rate_calculation(self) -> None:
        state = CompetitionState()
        state.record(True)
        state.record(True)
        state.record(False)
        assert state.win_rate == pytest.approx(2 / 3, rel=0.01)

    def test_window_size_limits(self) -> None:
        """Old results should roll off."""
        state = CompetitionState(window_size=5)
        for _ in range(5):
            state.record(False)
        for _ in range(5):
            state.record(True)
        # Only last 5 (all True) should remain
        assert state.win_rate == 1.0

    def test_competition_intensity_range(self) -> None:
        """Intensity should always be in [0, 1]."""
        state = CompetitionState()
        assert 0 <= state.competition_intensity <= 1

        for _ in range(20):
            state.record(True)
        assert 0 <= state.competition_intensity <= 1

        state2 = CompetitionState()
        for _ in range(20):
            state2.record(False)
        assert 0 <= state2.competition_intensity <= 1
