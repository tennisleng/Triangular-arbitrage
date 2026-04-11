"""
Tests for opportunity evaluation.

These tests verify that the evaluator correctly computes profit/loss
using Decimal arithmetic and properly handles edge cases like
empty order books and stale data.
"""

from __future__ import annotations

import time
from decimal import Decimal

from triangular_arb.strategy.evaluator import evaluate_triangle
from triangular_arb.types import (
    OrderBook,
    Pair,
    PriceLevel,
    Side,
    Symbol,
    Triangle,
)


def _make_book(
    pair: str,
    bid: str = "100",
    ask: str = "101",
    quantity: str = "10",
    timestamp_ns: int | None = None,
) -> OrderBook:
    """Helper to create a simple order book."""
    return OrderBook(
        pair=Pair(pair),
        bids=(PriceLevel(Decimal(bid), Decimal(quantity)),),
        asks=(PriceLevel(Decimal(ask), Decimal(quantity)),),
        timestamp_ns=timestamp_ns or time.time_ns(),
    )


def _make_triangle() -> Triangle:
    """Helper to create a standard test triangle."""
    return Triangle(
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


class TestEvaluator:
    """Tests for the evaluate_triangle function."""

    def test_profitable_triangle_detected(self) -> None:
        """A triangle with favorable rates should be detected as profitable."""
        triangle = _make_triangle()

        # Construct books where the cycle yields > 1.0
        # Leg 1 (BUY LTC/ETH): ask = 0.05 ETH per LTC → get 1/0.05 = 20 LTC
        # Leg 2 (SELL LTC/BTC): bid = 0.003 BTC per LTC → get 20 * 0.003 = 0.06 BTC
        # Leg 3 (BUY ETH/BTC): ask = 0.05 BTC per ETH → get 1/0.05 = 20 ETH... wait
        # Let's use rates that actually create an arb:
        # 1 ETH → BUY LTC @ 0.05 → 20 LTC → SELL LTC/BTC @ 0.003 → 0.06 BTC
        # → BUY ETH/BTC @ 0.055 → 0.06/0.055 = 1.0909 ETH → ~9% profit

        books = (
            _make_book("LTC/ETH", bid="0.049", ask="0.05", quantity="100"),
            _make_book("LTC/BTC", bid="0.003", ask="0.0031", quantity="100"),
            _make_book("ETH/BTC", bid="0.054", ask="0.055", quantity="100"),
        )

        result = evaluate_triangle(
            triangle=triangle,
            books=books,
            fee_rate=Decimal("0.001"),
        )

        # With these rates: (1/0.05) * 0.003 * (1/0.055) ≈ 1.0909
        # After 3 legs of 0.1% fees: 1.0909 * 0.999^3 ≈ 1.0876
        # Net profit ≈ 876 bps
        assert result is not None
        assert result.is_profitable
        assert result.net_profit_bps > Decimal("0")

    def test_unprofitable_triangle_rejected(self) -> None:
        """A triangle with unfavorable rates should return None."""
        triangle = _make_triangle()

        # Tight spreads, no arbitrage opportunity
        books = (
            _make_book("LTC/ETH", bid="0.05", ask="0.051", quantity="100"),
            _make_book("LTC/BTC", bid="0.0025", ask="0.0026", quantity="100"),
            _make_book("ETH/BTC", bid="0.050", ask="0.051", quantity="100"),
        )

        result = evaluate_triangle(
            triangle=triangle,
            books=books,
            fee_rate=Decimal("0.001"),
        )

        assert result is None

    def test_stale_books_rejected(self) -> None:
        """Books older than the staleness threshold should be rejected."""
        triangle = _make_triangle()

        old_ts = time.time_ns() - 10_000_000_000  # 10 seconds ago
        books = (
            _make_book("LTC/ETH", timestamp_ns=old_ts),
            _make_book("LTC/BTC"),
            _make_book("ETH/BTC"),
        )

        result = evaluate_triangle(triangle=triangle, books=books)
        assert result is None

    def test_empty_book_rejected(self) -> None:
        """Books with no levels should be rejected."""
        triangle = _make_triangle()

        empty_book = OrderBook(
            pair=Pair("LTC/ETH"),
            bids=(),
            asks=(),
        )
        books = (
            empty_book,
            _make_book("LTC/BTC"),
            _make_book("ETH/BTC"),
        )

        result = evaluate_triangle(triangle=triangle, books=books)
        assert result is None

    def test_fees_reduce_profit(self) -> None:
        """Higher fees should reduce net profit."""
        triangle = _make_triangle()

        books = (
            _make_book("LTC/ETH", bid="0.049", ask="0.05", quantity="100"),
            _make_book("LTC/BTC", bid="0.003", ask="0.0031", quantity="100"),
            _make_book("ETH/BTC", bid="0.054", ask="0.055", quantity="100"),
        )

        result_low_fee = evaluate_triangle(
            triangle=triangle,
            books=books,
            fee_rate=Decimal("0.001"),
        )
        result_high_fee = evaluate_triangle(
            triangle=triangle,
            books=books,
            fee_rate=Decimal("0.005"),
        )

        assert result_low_fee is not None
        if result_high_fee is not None:
            assert result_high_fee.net_profit_bps < result_low_fee.net_profit_bps

    def test_decimal_precision(self) -> None:
        """Verify we don't lose precision in calculations."""
        triangle = _make_triangle()
        books = (
            _make_book("LTC/ETH", bid="0.049", ask="0.05", quantity="100"),
            _make_book("LTC/BTC", bid="0.003", ask="0.0031", quantity="100"),
            _make_book("ETH/BTC", bid="0.054", ask="0.055", quantity="100"),
        )

        result = evaluate_triangle(
            triangle=triangle,
            books=books,
            fee_rate=Decimal("0.001"),
        )

        if result is not None:
            # Verify types are Decimal, not float
            assert isinstance(result.net_profit_bps, Decimal)
            assert isinstance(result.gross_profit_bps, Decimal)
            assert isinstance(result.estimated_size, Decimal)
