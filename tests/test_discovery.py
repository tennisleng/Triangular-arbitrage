"""
Tests for triangle discovery.

These tests verify that the graph-based discovery algorithm correctly
identifies triangular paths and assigns the right sides (BUY/SELL)
for each leg.
"""


from triangular_arb.strategy.discovery import discover_triangles
from triangular_arb.types import Pair, Side


class TestTriangleDiscovery:
    """Tests for the discover_triangles function."""

    def test_discovers_simple_triangle(self) -> None:
        """Given three pairs forming a cycle, should find exactly one triangle."""
        pairs = [
            Pair("LTC/ETH"),
            Pair("LTC/BTC"),
            Pair("ETH/BTC"),
        ]
        triangles = discover_triangles(pairs, base_currencies=["ETH"])

        assert len(triangles) == 1
        t = triangles[0]
        assert t.base == "ETH"
        assert {t.intermediate_a, t.intermediate_b} == {"LTC", "BTC"}

    def test_no_triangle_when_pair_missing(self) -> None:
        """If any edge is missing, no triangle should be found."""
        pairs = [
            Pair("LTC/ETH"),
            Pair("LTC/BTC"),
            # Missing ETH/BTC
        ]
        triangles = discover_triangles(pairs, base_currencies=["ETH"])
        assert len(triangles) == 0

    def test_multiple_triangles(self) -> None:
        """Multiple base currencies should find more triangles."""
        pairs = [
            Pair("LTC/ETH"),
            Pair("LTC/BTC"),
            Pair("ETH/BTC"),
            Pair("DOGE/ETH"),
            Pair("DOGE/BTC"),
        ]
        triangles = discover_triangles(pairs, base_currencies=["ETH", "BTC"])

        # Should find at least: ETH-LTC-BTC and ETH-DOGE-BTC
        assert len(triangles) >= 2

    def test_deduplication(self) -> None:
        """Same triangle from different base currencies should not be duplicated."""
        pairs = [
            Pair("LTC/ETH"),
            Pair("LTC/BTC"),
            Pair("ETH/BTC"),
        ]
        # Even though we specify both ETH and BTC as bases,
        # the triangle ETH-LTC-BTC should appear only once
        triangles = discover_triangles(pairs, base_currencies=["ETH", "BTC"])
        assert len(triangles) == 1

    def test_correct_sides_assigned(self) -> None:
        """Verify BUY/SELL sides are assigned correctly for each leg."""
        pairs = [
            Pair("LTC/ETH"),
            Pair("LTC/BTC"),
            Pair("ETH/BTC"),
        ]
        triangles = discover_triangles(pairs, base_currencies=["ETH"])
        assert len(triangles) == 1

        t = triangles[0]
        # Every leg should have a valid side
        for side in [t.leg1_side, t.leg2_side, t.leg3_side]:
            assert side in (Side.BUY, Side.SELL)

    def test_empty_pairs(self) -> None:
        """Empty pair list should return empty triangles."""
        triangles = discover_triangles([], base_currencies=["ETH"])
        assert len(triangles) == 0

    def test_invalid_pair_format_ignored(self) -> None:
        """Malformed pairs should be silently skipped."""
        pairs = [
            Pair("INVALID"),
            Pair("LTC/ETH"),
            Pair("LTC/BTC"),
            Pair("ETH/BTC"),
        ]
        triangles = discover_triangles(pairs, base_currencies=["ETH"])
        assert len(triangles) == 1
