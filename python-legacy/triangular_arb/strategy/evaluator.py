"""
Opportunity evaluator — determines if a triangle is profitable.

All arithmetic uses Decimal. The key insight: we compute the effective
exchange rate around the triangle. If starting with 1 unit and ending
with >1 unit after fees and slippage, it's profitable.
"""

from __future__ import annotations

import time
from decimal import ROUND_DOWN, Decimal

import structlog

from triangular_arb.types import (
    Direction,
    Opportunity,
    OrderBook,
    Side,
    Triangle,
)

log = structlog.get_logger()

# Minimum profitability to consider (avoids noise from rounding)
_MIN_PROFIT_BPS = Decimal("0.1")


def evaluate_triangle(
    triangle: Triangle,
    books: tuple[OrderBook, OrderBook, OrderBook],
    fee_rate: Decimal = Decimal("0.001"),
    max_slippage_bps: Decimal = Decimal("10"),
) -> Opportunity | None:
    """
    Evaluate whether a triangle presents a profitable opportunity.

    Simulates executing each leg against the order book to account
    for actual liquidity and slippage, not just top-of-book prices.

    Args:
        triangle: The triangle path to evaluate
        books: Order books for each leg (in order)
        fee_rate: Trading fee per leg (as fraction, e.g., 0.001 = 0.1%)
        max_slippage_bps: Maximum acceptable slippage per leg

    Returns:
        Opportunity if profitable, None otherwise
    """
    book1, book2, book3 = books

    # Validate books aren't stale
    now_ns = time.time_ns()
    for book in books:
        age_ms = (now_ns - book.timestamp_ns) / 1_000_000
        if age_ms > 5_000:  # 5 second staleness threshold
            log.debug("stale_book", pair=book.pair, age_ms=age_ms)
            return None

    # Validate books have sufficient depth
    for book in books:
        if len(book.bids) < 1 or len(book.asks) < 1:
            return None

    # Compute effective rate for each leg
    # Starting with 1 unit of the base currency
    fee_multiplier = Decimal("1") - fee_rate

    # Leg 1
    rate1 = _effective_rate(book1, triangle.leg1_side)
    if rate1 is None:
        return None

    # Leg 2
    rate2 = _effective_rate(book2, triangle.leg2_side)
    if rate2 is None:
        return None

    # Leg 3
    rate3 = _effective_rate(book3, triangle.leg3_side)
    if rate3 is None:
        return None

    # Gross profit: rate around the triangle
    gross_rate = rate1 * rate2 * rate3
    gross_profit_bps = (gross_rate - Decimal("1")) * Decimal("10000")

    # Net profit: after 3 legs of fees
    net_rate = rate1 * fee_multiplier * rate2 * fee_multiplier * rate3 * fee_multiplier
    net_profit_bps = (net_rate - Decimal("1")) * Decimal("10000")

    if net_profit_bps < _MIN_PROFIT_BPS:
        return None

    # Estimate max executable size (bottleneck liquidity)
    size = _estimate_max_size(books, triangle)

    return Opportunity(
        triangle=triangle,
        direction=Direction.FORWARD,
        gross_profit_bps=gross_profit_bps.quantize(Decimal("0.01")),
        net_profit_bps=net_profit_bps.quantize(Decimal("0.01")),
        estimated_size=size,
        books=books,
    )


def _effective_rate(book: OrderBook, side: Side) -> Decimal | None:
    """
    Get the effective exchange rate for one leg.

    BUY: We're buying base with quote → rate = 1/ask
    SELL: We're selling base for quote → rate = bid
    """
    if side == Side.BUY:
        if not book.asks:
            return None
        return Decimal("1") / book.asks[0].price
    else:
        if not book.bids:
            return None
        return book.bids[0].price


def _estimate_max_size(
    books: tuple[OrderBook, OrderBook, OrderBook],
    triangle: Triangle,
) -> Decimal:
    """
    Estimate the maximum tradeable size through the triangle.

    The bottleneck is the leg with the least available liquidity.
    We take the minimum across all three legs, converted to base currency.
    """
    sizes: list[Decimal] = []

    sides = [triangle.leg1_side, triangle.leg2_side, triangle.leg3_side]
    for book, side in zip(books, sides):
        levels = book.asks if side == Side.BUY else book.bids
        # Sum available liquidity across top levels
        total_qty = sum(level.quantity for level in levels[:5])
        sizes.append(total_qty)

    if not sizes:
        return Decimal("0")

    return min(sizes).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
