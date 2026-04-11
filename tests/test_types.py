"""Tests for domain types."""

from decimal import Decimal

from triangular_arb.types import OrderBook, Pair, PriceLevel, Side, Symbol, Triangle


class TestOrderBook:
    """Tests for OrderBook properties."""

    def test_spread_bps_calculation(self) -> None:
        """Spread should be correctly computed in basis points."""
        book = OrderBook(
            pair=Pair("ETH/BTC"),
            bids=(PriceLevel(Decimal("100"), Decimal("10")),),
            asks=(PriceLevel(Decimal("101"), Decimal("10")),),
        )
        # Spread = (101-100)/100.5 * 10000 ≈ 99.50 bps
        assert book.spread_bps > Decimal("99")
        assert book.spread_bps < Decimal("100")

    def test_spread_bps_empty_book(self) -> None:
        """Empty book should have infinite spread."""
        book = OrderBook(pair=Pair("ETH/BTC"), bids=(), asks=())
        assert book.spread_bps == Decimal("Infinity")


class TestTriangle:
    def test_str_representation(self) -> None:
        """Triangle should have a readable string representation."""
        t = Triangle(
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
        s = str(t)
        assert "ETH" in s
        assert "LTC" in s
        assert "BTC" in s
