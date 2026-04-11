"""
Main engine — the event loop that ties discovery, evaluation, risk, and execution.

Pipeline per triangle scan:
  1. Fetch order books (concurrent)
  2. Toxicity check (reject manipulated books)
  3. Evaluate profitability
  4. Decay-adjust for competition (stale opportunities are worth less)
  5. Risk gate (circuit breakers, drawdown limits)
  6. Triangle heat check (avoid detectable patterns)
  7. Execute with stealth (randomized size, timing jitter)
"""

from __future__ import annotations

import asyncio

import structlog

from triangular_arb.config import Config
from triangular_arb.exchange.adapter import ExchangeAdapter
from triangular_arb.exchange.binance import BinanceAdapter
from triangular_arb.execution.executor import Executor
from triangular_arb.execution.stealth import (
    CompetitionState,
    adaptive_threshold,
    jitter_delay,
    opportunity_decay,
    randomize_size,
    triangle_heat,
)
from triangular_arb.risk.manager import RiskManager
from triangular_arb.strategy.discovery import discover_triangles
from triangular_arb.strategy.evaluator import evaluate_triangle
from triangular_arb.strategy.toxicity import analyze_toxicity
from triangular_arb.types import Opportunity, Symbol, Triangle
from triangular_arb.utils.logging import setup_logging

log = structlog.get_logger()


class Engine:
    """
    Main arbitrage engine.

    Lifecycle:
        engine = Engine(config)
        await engine.start()  # Runs until interrupted
        await engine.stop()   # Cleanup
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._exchange: ExchangeAdapter | None = None
        self._executor: Executor | None = None
        self._risk: RiskManager | None = None
        self._competition = CompetitionState()
        self._triangles: list[Triangle] = []
        self._running = False
        self._scan_count = 0
        self._opportunity_count = 0
        self._toxic_rejections = 0
        self._decay_rejections = 0
        self._heat_rejections = 0

    async def start(self) -> None:
        """Initialize components and start the scan loop."""
        setup_logging(
            level=self._config.logging.level,
            json_output=self._config.logging.json_output,
            log_dir=self._config.logging.log_dir,
        )

        log.info(
            "engine_starting",
            exchange=self._config.exchange.exchange_id,
            dry_run=self._config.dry_run,
        )

        # Initialize exchange adapter
        self._exchange = BinanceAdapter(self._config.exchange)

        # Initialize risk manager
        self._risk = RiskManager(self._config.risk)

        # Get initial balance for risk baseline
        base_currencies = self._config.scanner.base_currencies
        base = base_currencies[0] if base_currencies else "ETH"
        balance = await self._exchange.fetch_balance(Symbol(base))
        self._risk.set_starting_balance(balance)
        log.info("initial_balance", currency=base, balance=float(balance))

        # Initialize executor
        self._executor = Executor(
            exchange=self._exchange,
            config=self._config.execution,
            dry_run=self._config.dry_run,
        )

        # Discover triangles
        pairs = await self._exchange.get_all_pairs()
        self._triangles = discover_triangles(
            pairs=pairs,
            base_currencies=self._config.scanner.base_currencies,
        )

        if not self._triangles:
            log.error("no_triangles_found")
            await self.stop()
            return

        log.info(
            "engine_ready",
            triangles=len(self._triangles),
            scan_interval_ms=self._config.scanner.scan_interval_ms,
        )

        # Start the scan loop
        self._running = True
        try:
            await self._scan_loop()
        except asyncio.CancelledError:
            log.info("engine_cancelled")
        finally:
            await self.stop()

    async def _scan_loop(self) -> None:
        """Continuously scan triangles for opportunities."""
        interval_s = self._config.scanner.scan_interval_ms / 1000

        while self._running:
            self._scan_count += 1

            for triangle in self._triangles:
                if not self._running:
                    break
                await self._evaluate_and_execute(triangle)

            if self._scan_count % 100 == 0:
                self._log_stats()

            await asyncio.sleep(interval_s)

    async def _evaluate_and_execute(self, triangle: Triangle) -> None:
        """Full pipeline: fetch → toxicity → evaluate → decay → risk → heat → execute."""
        assert self._exchange is not None
        assert self._executor is not None
        assert self._risk is not None

        try:
            # ── 1. Fetch order books concurrently ─────────────────────
            depth = self._config.execution.order_book_depth
            books = await asyncio.gather(
                self._exchange.fetch_order_book(triangle.leg1_pair, depth),
                self._exchange.fetch_order_book(triangle.leg2_pair, depth),
                self._exchange.fetch_order_book(triangle.leg3_pair, depth),
                return_exceptions=True,
            )

            for book in books:
                if isinstance(book, Exception):
                    return

            # ── 2. Toxicity check (reject manipulated books) ──────────
            for book in books:
                report = analyze_toxicity(book)  # type: ignore[arg-type]
                if not report.safe_to_trade:
                    self._toxic_rejections += 1
                    log.debug(
                        "toxic_book_rejected",
                        pair=book.pair,  # type: ignore[union-attr]
                        signal=report.signal.value,
                        detail=report.detail,
                    )
                    return

            # ── 3. Evaluate profitability ─────────────────────────────
            _, taker_fee = await self._exchange.get_trading_fees(
                triangle.leg1_pair,
            )

            opportunity = evaluate_triangle(
                triangle=triangle,
                books=(books[0], books[1], books[2]),  # type: ignore[arg-type]
                fee_rate=taker_fee,
                max_slippage_bps=self._config.execution.max_slippage_bps,
            )

            if opportunity is None:
                return

            self._opportunity_count += 1

            # ── 4. Decay-adjust for competition ───────────────────────
            decayed_bps = opportunity_decay(opportunity, half_life_ms=200)
            threshold = adaptive_threshold(
                self._config.risk.min_profit_bps,
                self._competition,
            )

            if decayed_bps < threshold:
                self._decay_rejections += 1
                log.debug(
                    "opportunity_decayed",
                    triangle=str(triangle),
                    original_bps=float(opportunity.net_profit_bps),
                    decayed_bps=float(decayed_bps),
                    threshold_bps=float(threshold),
                    age_ms=opportunity.age_ms,
                )
                return

            # ── 5. Risk gate ──────────────────────────────────────────
            rejection = self._risk.check(opportunity)
            if rejection is not None:
                log.debug(
                    "opportunity_rejected",
                    triangle=str(triangle),
                    reason=rejection.value,
                )
                return

            # ── 6. Triangle heat check ────────────────────────────────
            tri_key = str(triangle)
            if triangle_heat(tri_key, self._competition):
                self._heat_rejections += 1
                log.debug("triangle_too_hot", triangle=tri_key)
                return

            # ── 7. Execute with stealth ───────────────────────────────
            log.info(
                "executing_opportunity",
                triangle=tri_key,
                net_profit_bps=float(decayed_bps),
                size=float(opportunity.estimated_size),
                competition=f"{self._competition.competition_intensity:.2f}",
            )

            # Randomize size and add jitter
            stealth_opportunity = self._apply_stealth(opportunity)
            await jitter_delay(
                competition=self._competition.competition_intensity,
            )

            result = await self._executor.execute(stealth_opportunity)
            self._risk.record_result(result)
            self._competition.record(result.success and result.net_profit > 0)
            self._competition.record_execution(tri_key)

        except Exception:
            log.exception("scan_error", triangle=str(triangle))

    def _apply_stealth(self, opportunity: Opportunity) -> Opportunity:
        """Apply size randomization to make our flow harder to fingerprint."""
        noisy_size = randomize_size(opportunity.estimated_size)
        return Opportunity(
            triangle=opportunity.triangle,
            direction=opportunity.direction,
            gross_profit_bps=opportunity.gross_profit_bps,
            net_profit_bps=opportunity.net_profit_bps,
            estimated_size=noisy_size,
            books=opportunity.books,
            detected_at_ns=opportunity.detected_at_ns,
        )

    def _log_stats(self) -> None:
        """Periodic stats including competition metrics."""
        assert self._risk is not None
        state = self._risk.state
        log.info(
            "scan_stats",
            scans=self._scan_count,
            opportunities_found=self._opportunity_count,
            toxic_rejections=self._toxic_rejections,
            decay_rejections=self._decay_rejections,
            heat_rejections=self._heat_rejections,
            competition_intensity=(f"{self._competition.competition_intensity:.2f}"),
            win_rate=f"{self._competition.win_rate:.2f}",
            risk_state={
                "total_trades": state.total_trades,
                "win_rate": self._risk.win_rate,
                "daily_pnl": float(state.daily_pnl),
            },
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False

        if self._risk:
            state = self._risk.state
            log.info(
                "engine_stopped",
                total_scans=self._scan_count,
                total_trades=state.total_trades,
                win_rate=self._risk.win_rate,
                total_pnl=float(state.total_pnl),
                toxic_rejections=self._toxic_rejections,
                decay_rejections=self._decay_rejections,
                heat_rejections=self._heat_rejections,
            )

        if self._exchange:
            await self._exchange.close()
