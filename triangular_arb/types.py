"""
Domain types for the arbitrage engine.

All financial values use Decimal to avoid floating-point errors that would
silently eat basis points of profit — the exact margin this system operates on.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import NewType, Optional, Tuple

# ─── Branded newtypes prevent passing a bid where an ask is expected ──────────
Symbol = NewType("Symbol", str)  # e.g. "ETH"
Pair = NewType("Pair", str)  # e.g. "ETH/BTC"
ExchangeId = NewType("ExchangeId", str)


class Side(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(enum.Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Direction(enum.Enum):
    """Which way around the triangle we're going."""

    FORWARD = "forward"  # A → B → C → A
    BACKWARD = "backward"  # A → C → B → A


@dataclass(frozen=True)
class PriceLevel:
    """Single level in an order book."""

    price: Decimal
    quantity: Decimal


@dataclass(frozen=True)
class OrderBook:
    """Snapshot of an order book at a point in time."""

    pair: Pair
    bids: Tuple[PriceLevel, ...]  # Best bid first (descending price)
    asks: Tuple[PriceLevel, ...]  # Best ask first (ascending price)
    timestamp_ns: int = field(default_factory=time.time_ns)

    @property
    def spread_bps(self) -> Decimal:
        """Bid-ask spread in basis points."""
        if not self.bids or not self.asks:
            return Decimal("Infinity")
        mid = (self.bids[0].price + self.asks[0].price) / 2
        if mid == 0:
            return Decimal("Infinity")
        return (self.asks[0].price - self.bids[0].price) / mid * Decimal("10000")


@dataclass(frozen=True)
class Triangle:
    """
    A triangular path through three pairs.

    Example: ETH → LTC/ETH → LTC/BTC → ETH/BTC
    The legs define which pairs to trade and in what order.
    """

    base: Symbol  # The currency we start and end with (e.g., ETH)
    leg1_pair: Pair  # e.g. LTC/ETH
    leg1_side: Side  # BUY or SELL
    leg2_pair: Pair  # e.g. LTC/BTC
    leg2_side: Side
    leg3_pair: Pair  # e.g. ETH/BTC
    leg3_side: Side
    intermediate_a: Symbol  # e.g. LTC
    intermediate_b: Symbol  # e.g. BTC

    def __str__(self) -> str:
        return (
            f"{self.base} →({self.leg1_side.value} {self.leg1_pair})→ "
            f"{self.intermediate_a} →({self.leg2_side.value} {self.leg2_pair})→ "
            f"{self.intermediate_b} →({self.leg3_side.value} {self.leg3_pair})→ {self.base}"
        )


@dataclass(frozen=True)
class Opportunity:
    """A detected arbitrage opportunity with estimated profit."""

    triangle: Triangle
    direction: Direction
    gross_profit_bps: Decimal  # Before fees/slippage
    net_profit_bps: Decimal  # After fees/slippage
    estimated_size: Decimal  # Max executable size (bottleneck liquidity)
    books: Tuple[OrderBook, OrderBook, OrderBook]
    detected_at_ns: int = field(default_factory=time.time_ns)

    @property
    def is_profitable(self) -> bool:
        return self.net_profit_bps > 0

    @property
    def age_ms(self) -> float:
        return (time.time_ns() - self.detected_at_ns) / 1_000_000


@dataclass(frozen=True)
class Fill:
    """Result of a single leg execution."""

    pair: Pair
    side: Side
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_currency: Symbol
    status: OrderStatus
    exchange_order_id: str
    latency_ms: float


@dataclass(frozen=True)
class ArbitrageResult:
    """
    Complete result of a triangular arbitrage attempt.

    Immutable record for audit trail. Every trade ever executed
    can be reconstructed from a sequence of these.
    """

    opportunity: Opportunity
    fills: Tuple[Fill, ...]
    net_profit: Decimal
    net_profit_bps: Decimal
    total_fees: Decimal
    total_latency_ms: float
    success: bool
    error: Optional[str] = None
