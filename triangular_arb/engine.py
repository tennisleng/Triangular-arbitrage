"""
Main engine — the event loop that ties discovery, evaluation, risk, and execution.

This is the central orchestrator. It:
1. Discovers all valid triangles on startup
2. Continuously scans order books for each triangle
3. Evaluates profitability
4. Passes through risk checks
5. Executes if approved

The engine is designed to be the single entry point for running the system.
Configuration, exchange adapters, and strategy components are injected.
"""

from __future__ import annotations

import asyncio

import structlog

from triangular_arb.config import Config
from triangular_arb.exchange.adapter import ExchangeAdapter
from triangular_arb.exchange.binance import BinanceAdapter
from triangular_arb.execution.executor import Executor
from triangular_arb.risk.manager import RiskManager
from triangular_arb.strategy.discovery import discover_triangles
from triangular_arb.strategy.evaluator import evaluate_triangle
from triangular_arb.types import Symbol, Triangle
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
        self._triangles: list[Triangle] = []
        self._running = False
        self._scan_count = 0
        self._opportunity_count = 0

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
                log.info(
                    "scan_stats",
                    scans=self._scan_count,
                    opportunities_found=self._opportunity_count,
                    risk_state={
                        "total_trades": self._risk.state.total_trades if self._risk else 0,
                        "win_rate": self._risk.win_rate if self._risk else 0,
                        "daily_pnl": float(self._risk.state.daily_pnl) if self._risk else 0,
                    },
                )

            await asyncio.sleep(interval_s)

    async def _evaluate_and_execute(self, triangle: Triangle) -> None:
        """Fetch books, evaluate, check risk, and execute if profitable."""
        assert self._exchange is not None
        assert self._executor is not None
        assert self._risk is not None

        try:
            # Fetch order books for all three legs concurrently
            depth = self._config.execution.order_book_depth
            books = await asyncio.gather(
                self._exchange.fetch_order_book(triangle.leg1_pair, depth),
                self._exchange.fetch_order_book(triangle.leg2_pair, depth),
                self._exchange.fetch_order_book(triangle.leg3_pair, depth),
                return_exceptions=True,
            )

            # Check for fetch errors
            for book in books:
                if isinstance(book, Exception):
                    return

            # Get fee rate
            _, taker_fee = await self._exchange.get_trading_fees(triangle.leg1_pair)

            # Evaluate profitability
            opportunity = evaluate_triangle(
                triangle=triangle,
                books=(books[0], books[1], books[2]),  # type: ignore[arg-type]
                fee_rate=taker_fee,
                max_slippage_bps=self._config.execution.max_slippage_bps,
            )

            if opportunity is None:
                return

            self._opportunity_count += 1

            # Risk check
            rejection = self._risk.check(opportunity)
            if rejection is not None:
                log.debug(
                    "opportunity_rejected",
                    triangle=str(triangle),
                    reason=rejection.value,
                )
                return

            # Execute!
            log.info(
                "opportunity_found",
                triangle=str(triangle),
                net_profit_bps=float(opportunity.net_profit_bps),
                size=float(opportunity.estimated_size),
                age_ms=opportunity.age_ms,
            )

            result = await self._executor.execute(opportunity)
            self._risk.record_result(result)

        except Exception:
            log.exception("scan_error", triangle=str(triangle))

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
            )

        if self._exchange:
            await self._exchange.close()
