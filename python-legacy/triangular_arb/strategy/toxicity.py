"""
Order book toxicity detection.

In competitive HFT, other firms will:
1. Spoof/layer the book to bait your algo into trading
2. Detect your pattern and frontrun you
3. Place phantom liquidity that disappears when you hit it

This module scores order books for toxicity signals
_before_ we commit capital. A toxic book means someone
is trying to exploit us — we skip and move on.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from triangular_arb.types import OrderBook


class ToxicitySignal(Enum):
    CLEAN = "clean"
    THIN_BOOK = "thin_book"  # Insufficient liquidity to hide in
    IMBALANCED = "imbalanced"  # Lopsided book — likely informed flow
    WIDE_SPREAD = "wide_spread"  # Spread too wide — adverse selection risk
    LAYERED = "layered"  # Suspicious equal-size levels (spoofing)
    DISAPPEARING_LIQUIDITY = "disappearing"  # Liquidity concentrated at one level


@dataclass(frozen=True)
class ToxicityReport:
    """Immutable result of toxicity analysis."""

    signal: ToxicitySignal
    score: Decimal  # 0 = clean, 100 = extremely toxic
    detail: str
    safe_to_trade: bool


# ── Thresholds (tuned conservatively — false negatives cost money) ─────────
_MIN_LEVELS = 3
_MAX_SPREAD_BPS = Decimal("50")
_IMBALANCE_RATIO = Decimal("5")  # 5:1 bid/ask volume ratio = suspicious
_LAYER_SIZE_TOLERANCE = Decimal("0.05")  # 5% size similarity = potential spoof
_MIN_LAYER_COUNT = 3  # Need 3+ identical levels to flag


def analyze_toxicity(book: OrderBook) -> ToxicityReport:
    """
    Score an order book for manipulation signals.

    Run this on every book before committing capital.
    Cheap to compute (O(depth)), expensive to skip.
    """
    if len(book.bids) < _MIN_LEVELS or len(book.asks) < _MIN_LEVELS:
        return ToxicityReport(
            signal=ToxicitySignal.THIN_BOOK,
            score=Decimal("80"),
            detail=f"Only {min(len(book.bids), len(book.asks))} levels",
            safe_to_trade=False,
        )

    # ── Spread check ──────────────────────────────────────────────────
    spread = book.spread_bps
    if spread > _MAX_SPREAD_BPS:
        return ToxicityReport(
            signal=ToxicitySignal.WIDE_SPREAD,
            score=Decimal("70"),
            detail=f"Spread {spread:.1f} bps > {_MAX_SPREAD_BPS} threshold",
            safe_to_trade=False,
        )

    # ── Volume imbalance (informed flow detection) ────────────────────
    bid_vol = sum(level.quantity for level in book.bids[:5])
    ask_vol = sum(level.quantity for level in book.asks[:5])

    if bid_vol > 0 and ask_vol > 0:
        ratio = max(bid_vol / ask_vol, ask_vol / bid_vol)
        if ratio > _IMBALANCE_RATIO:
            heavier = "bid" if bid_vol > ask_vol else "ask"
            return ToxicityReport(
                signal=ToxicitySignal.IMBALANCED,
                score=min(Decimal("90"), Decimal(str(ratio)) * Decimal("10")),
                detail=f"{ratio:.1f}:1 {heavier}-heavy (informed flow likely)",
                safe_to_trade=False,
            )

    # ── Layering/spoofing detection ───────────────────────────────────
    # Spoofers place many orders of nearly identical size to create
    # fake depth. Real order flow has varied sizes.
    for side_name, levels in [("ask", book.asks), ("bid", book.bids)]:
        if len(levels) >= _MIN_LAYER_COUNT:
            sizes = [level.quantity for level in levels[:8]]
            identical = _count_similar_sizes(sizes)
            if identical >= _MIN_LAYER_COUNT:
                return ToxicityReport(
                    signal=ToxicitySignal.LAYERED,
                    score=Decimal("85"),
                    detail=f"{identical} near-identical {side_name} levels (spoofing pattern)",
                    safe_to_trade=False,
                )

    # ── Disappearing liquidity ────────────────────────────────────────
    # If >80% of volume is at a single level, it's likely to be pulled
    for side_name, levels in [("ask", book.asks), ("bid", book.bids)]:
        if len(levels) >= 2:
            total = sum(lvl.quantity for lvl in levels[:5])
            if total > 0 and levels[0].quantity / total > Decimal("0.8"):
                pct = levels[0].quantity / total * 100
                return ToxicityReport(
                    signal=ToxicitySignal.DISAPPEARING_LIQUIDITY,
                    score=Decimal("60"),
                    detail=f"Top {side_name} level is {pct:.0f}% of depth",
                    safe_to_trade=False,
                )

    return ToxicityReport(
        signal=ToxicitySignal.CLEAN,
        score=Decimal("0"),
        detail="No manipulation signals detected",
        safe_to_trade=True,
    )


def _count_similar_sizes(sizes: list[Decimal]) -> int:
    """Count how many sizes are within tolerance of the median."""
    if not sizes:
        return 0
    sorted_sizes = sorted(sizes)
    median = sorted_sizes[len(sorted_sizes) // 2]
    if median == 0:
        return 0
    return sum(1 for s in sizes if abs(s - median) / median <= _LAYER_SIZE_TOLERANCE)
