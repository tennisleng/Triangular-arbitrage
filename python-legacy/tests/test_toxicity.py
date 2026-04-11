"""Tests for order book toxicity detection."""

from __future__ import annotations

from decimal import Decimal

from triangular_arb.strategy.toxicity import (
    ToxicitySignal,
    analyze_toxicity,
)
from triangular_arb.types import OrderBook, Pair, PriceLevel


def _book(
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
) -> OrderBook:
    """Helper: build a book from (price, qty) string pairs."""
    return OrderBook(
        pair=Pair("ETH/BTC"),
        bids=tuple(PriceLevel(Decimal(p), Decimal(q)) for p, q in bids),
        asks=tuple(PriceLevel(Decimal(p), Decimal(q)) for p, q in asks),
    )


class TestToxicity:
    def test_clean_book(self) -> None:
        """Normal book with tight spread and varied sizes."""
        book = _book(
            bids=[
                ("100.00", "5"),
                ("99.99", "8"),
                ("99.98", "3"),
                ("99.97", "12"),
                ("99.96", "7"),
            ],
            asks=[
                ("100.01", "6"),
                ("100.02", "4"),
                ("100.03", "9"),
                ("100.04", "2"),
                ("100.05", "11"),
            ],
        )
        report = analyze_toxicity(book)
        assert report.signal == ToxicitySignal.CLEAN
        assert report.safe_to_trade

    def test_thin_book_detected(self) -> None:
        """Book with fewer than 3 levels."""
        book = _book(bids=[("100", "5")], asks=[("100.01", "6")])
        report = analyze_toxicity(book)
        assert report.signal == ToxicitySignal.THIN_BOOK
        assert not report.safe_to_trade

    def test_wide_spread_detected(self) -> None:
        """Spread > 50 bps."""
        book = _book(
            bids=[("100", "5"), ("99.99", "8"), ("99.98", "3")],
            asks=[("101", "6"), ("101.01", "4"), ("101.02", "9")],
        )
        report = analyze_toxicity(book)
        assert report.signal == ToxicitySignal.WIDE_SPREAD
        assert not report.safe_to_trade

    def test_imbalanced_book_detected(self) -> None:
        """Lopsided volume → informed flow."""
        book = _book(
            bids=[
                ("100.00", "100"),
                ("99.99", "100"),
                ("99.98", "100"),
                ("99.97", "100"),
                ("99.96", "100"),
            ],
            asks=[
                ("100.01", "1"),
                ("100.02", "1"),
                ("100.03", "1"),
                ("100.04", "1"),
                ("100.05", "1"),
            ],
        )
        report = analyze_toxicity(book)
        assert report.signal == ToxicitySignal.IMBALANCED
        assert not report.safe_to_trade

    def test_layered_book_detected(self) -> None:
        """Multiple identical-size levels → spoofing."""
        book = _book(
            bids=[("100.00", "7"), ("99.99", "3"), ("99.98", "9")],
            asks=[
                ("100.01", "10.00"),
                ("100.02", "10.00"),
                ("100.03", "10.00"),
                ("100.04", "10.00"),
                ("100.05", "5"),
            ],
        )
        report = analyze_toxicity(book)
        assert report.signal == ToxicitySignal.LAYERED
        assert not report.safe_to_trade

    def test_disappearing_liquidity_detected(self) -> None:
        """Single level with >80% of volume."""
        book = _book(
            bids=[("100.00", "15"), ("99.99", "25"), ("99.98", "20")],
            asks=[
                ("100.01", "50"),  # >80% of ask depth
                ("100.02", "3"),
                ("100.03", "2"),
                ("100.04", "4"),
                ("100.05", "1"),
            ],
        )
        report = analyze_toxicity(book)
        assert report.signal == ToxicitySignal.DISAPPEARING_LIQUIDITY
        assert not report.safe_to_trade

    def test_score_is_decimal(self) -> None:
        """Toxicity scores should always be Decimal."""
        book = _book(
            bids=[
                ("100.00", "5"),
                ("99.99", "8"),
                ("99.98", "3"),
                ("99.97", "12"),
                ("99.96", "7"),
            ],
            asks=[
                ("100.01", "6"),
                ("100.02", "4"),
                ("100.03", "9"),
                ("100.04", "2"),
                ("100.05", "11"),
            ],
        )
        report = analyze_toxicity(book)
        assert isinstance(report.score, Decimal)
