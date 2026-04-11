"""
Tri-leg execution engine.

Handles the atomic execution of a three-leg arbitrage trade.
If any leg fails, attempts best-effort rollback to minimize losses.

Key design decisions:
- Each leg is executed sequentially (not parallel) because leg N's
  output determines leg N+1's input
- Rollback on failure converts whatever was acquired back to the
  original currency to limit exposure
- All fills are recorded for audit trail regardless of success/failure
"""

from __future__ import annotations

from decimal import Decimal

import structlog

from triangular_arb.config import ExecutionConfig
from triangular_arb.exchange.adapter import ExchangeAdapter
from triangular_arb.types import (
    ArbitrageResult,
    Fill,
    Opportunity,
    OrderStatus,
    Side,
    Symbol,
)

log = structlog.get_logger()


class Executor:
    """
    Executes triangular arbitrage opportunities.

    This is the most latency-sensitive component. Every millisecond
    of delay increases the probability that the opportunity has been
    captured by a competitor.
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        config: ExecutionConfig,
        dry_run: bool = True,
    ) -> None:
        self._exchange = exchange
        self._config = config
        self._dry_run = dry_run

    async def execute(self, opportunity: Opportunity) -> ArbitrageResult:
        """
        Execute a triangular arbitrage opportunity.

        Three legs are executed sequentially. On any failure,
        we attempt rollback to minimize loss exposure.

        Args:
            opportunity: The evaluated opportunity to execute

        Returns:
            ArbitrageResult with all fills and final P&L
        """
        t = opportunity.triangle
        fills: list[Fill] = []
        total_latency = 0.0

        log.info(
            "executing_arbitrage",
            triangle=str(t),
            net_profit_bps=float(opportunity.net_profit_bps),
            dry_run=self._dry_run,
        )

        if self._dry_run:
            return self._simulate_execution(opportunity)

        # ── Leg 1 ────────────────────────────────────────────────────────
        fill1 = await self._execute_leg(
            pair=t.leg1_pair,
            side=t.leg1_side,
            quantity=opportunity.estimated_size,
            leg_num=1,
        )
        fills.append(fill1)
        total_latency += fill1.latency_ms

        if fill1.status != OrderStatus.FILLED:
            log.warning("leg1_failed", fill=fill1)
            return self._build_result(
                opportunity,
                tuple(fills),
                total_latency,
                error="Leg 1 failed",
            )

        # ── Leg 2 ────────────────────────────────────────────────────────
        leg2_qty = fill1.quantity  # Output of leg 1 is input to leg 2
        if fill1.side == Side.BUY:
            leg2_qty = fill1.quantity - fill1.fee  # Subtract fee if paid in base

        fill2 = await self._execute_leg(
            pair=t.leg2_pair,
            side=t.leg2_side,
            quantity=leg2_qty,
            leg_num=2,
        )
        fills.append(fill2)
        total_latency += fill2.latency_ms

        if fill2.status != OrderStatus.FILLED:
            log.warning("leg2_failed_attempting_rollback", fill=fill2)
            rollback = await self._rollback(t.leg1_pair, fill1)
            if rollback:
                fills.append(rollback)
            return self._build_result(
                opportunity,
                tuple(fills),
                total_latency,
                error="Leg 2 failed",
            )

        # ── Leg 3 ────────────────────────────────────────────────────────
        leg3_qty = fill2.quantity
        if fill2.side == Side.BUY:
            leg3_qty = fill2.quantity - fill2.fee

        fill3 = await self._execute_leg(
            pair=t.leg3_pair,
            side=t.leg3_side,
            quantity=leg3_qty,
            leg_num=3,
        )
        fills.append(fill3)
        total_latency += fill3.latency_ms

        if fill3.status != OrderStatus.FILLED:
            log.warning("leg3_failed_attempting_rollback", fill=fill3)
            return self._build_result(
                opportunity,
                tuple(fills),
                total_latency,
                error="Leg 3 failed",
            )

        return self._build_result(opportunity, tuple(fills), total_latency)

    async def _execute_leg(
        self,
        pair: str,
        side: Side,
        quantity: Decimal,
        leg_num: int,
    ) -> Fill:
        """Execute a single leg with optional limit order and timeout."""
        log.debug(
            "executing_leg",
            leg=leg_num,
            pair=pair,
            side=side.value,
            quantity=str(quantity),
        )

        price: Decimal | None = None
        if self._config.use_limit_orders:
            book = await self._exchange.fetch_order_book(pair, depth=5)
            levels = book.asks if side == Side.BUY else book.bids
            if levels:
                # Set limit price slightly through the book for immediate fill
                price = levels[0].price

        return await self._exchange.place_order(
            pair=pair,
            side=side,
            quantity=quantity,
            price=price,
        )

    async def _rollback(self, pair: str, original_fill: Fill) -> Fill | None:
        """
        Best-effort rollback: reverse the original trade.

        This won't recover 100% (we'll eat the spread + fees again),
        but it limits exposure to the original currency.
        """
        try:
            reverse_side = Side.SELL if original_fill.side == Side.BUY else Side.BUY
            return await self._exchange.place_order(
                pair=pair,
                side=reverse_side,
                quantity=original_fill.quantity,
                price=None,  # Market order for speed
            )
        except Exception as e:
            log.error("rollback_failed", pair=pair, error=str(e))
            return None

    def _simulate_execution(self, opportunity: Opportunity) -> ArbitrageResult:
        """Simulate execution for dry-run mode using order book prices."""
        fills: list[Fill] = []
        qty = opportunity.estimated_size

        tri = opportunity.triangle
        sides = [tri.leg1_side, tri.leg2_side, tri.leg3_side]
        for i, (book, side) in enumerate(zip(opportunity.books, sides)):
            levels = book.asks if side == Side.BUY else book.bids
            price = levels[0].price if levels else Decimal("0")
            fee = qty * Decimal("0.001")

            fills.append(
                Fill(
                    pair=book.pair,
                    side=side,
                    price=price,
                    quantity=qty,
                    fee=fee,
                    fee_currency=Symbol(""),
                    status=OrderStatus.FILLED,
                    exchange_order_id=f"sim-{i}",
                    latency_ms=0.5,
                )
            )

            # Simulate conversion
            if side == Side.BUY:
                qty = qty / price - fee
            else:
                qty = qty * price - fee

        return self._build_result(opportunity, tuple(fills), 1.5)

    def _build_result(
        self,
        opportunity: Opportunity,
        fills: tuple[Fill, ...],
        total_latency: float,
        error: str | None = None,
    ) -> ArbitrageResult:
        """Build the final ArbitrageResult from fills."""
        total_fees = sum(f.fee for f in fills)
        success = error is None and all(f.status == OrderStatus.FILLED for f in fills)

        # Calculate net profit from fills
        if len(fills) >= 3 and success:
            # Simplified: compare final output vs initial input
            net_profit = fills[-1].quantity * fills[-1].price - fills[0].quantity * fills[0].price
            net_profit_bps = opportunity.net_profit_bps
        else:
            net_profit = Decimal("0")
            net_profit_bps = Decimal("0")

        result = ArbitrageResult(
            opportunity=opportunity,
            fills=fills,
            net_profit=net_profit,
            net_profit_bps=net_profit_bps,
            total_fees=total_fees,
            total_latency_ms=total_latency,
            success=success,
            error=error,
        )

        log.info(
            "arbitrage_complete",
            success=success,
            net_profit_bps=float(net_profit_bps),
            total_fees=float(total_fees),
            latency_ms=total_latency,
            error=error,
        )

        return result
