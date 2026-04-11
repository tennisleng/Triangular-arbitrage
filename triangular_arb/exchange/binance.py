"""
Binance exchange adapter via ccxt.

Wraps ccxt's async client with proper error handling, retry logic,
and conversion to our domain types. All Decimal conversion happens
at this boundary — internal code never sees raw floats from the API.
"""

from __future__ import annotations

import time
from decimal import Decimal

import ccxt.async_support as ccxt_async
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from triangular_arb.config import ExchangeConfig
from triangular_arb.exchange.adapter import ExchangeAdapter
from triangular_arb.types import (
    ExchangeId,
    Fill,
    OrderBook,
    OrderStatus,
    Pair,
    PriceLevel,
    Side,
    Symbol,
)

log = structlog.get_logger()

# Transient errors worth retrying
_RETRYABLE = (
    ccxt_async.NetworkError,
    ccxt_async.RequestTimeout,
    ccxt_async.ExchangeNotAvailable,
)


class BinanceAdapter(ExchangeAdapter):
    """
    Production Binance adapter.

    Converts between ccxt's float-based responses and our Decimal domain.
    All retries and rate-limiting are handled here so callers don't need to.
    """

    def __init__(self, config: ExchangeConfig) -> None:
        self._config = config
        self._client = ccxt_async.binance(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "timeout": config.timeout_ms,
                "enableRateLimit": config.rate_limit,
                "options": {"defaultType": "spot"},
            }
        )

        if config.testnet:
            self._client.set_sandbox_mode(True)

        self._fee_cache: dict[Pair, tuple[Decimal, Decimal]] = {}

    @property
    def exchange_id(self) -> ExchangeId:
        return ExchangeId("binance")

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    async def fetch_order_book(self, pair: Pair, depth: int = 10) -> OrderBook:
        raw = await self._client.fetch_order_book(pair, limit=depth)
        ts = time.time_ns()

        bids = tuple(
            PriceLevel(price=Decimal(str(p)), quantity=Decimal(str(q)))
            for p, q in raw.get("bids", [])
        )
        asks = tuple(
            PriceLevel(price=Decimal(str(p)), quantity=Decimal(str(q)))
            for p, q in raw.get("asks", [])
        )

        return OrderBook(pair=pair, bids=bids, asks=asks, timestamp_ns=ts)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    async def fetch_balance(self, symbol: Symbol) -> Decimal:
        balance = await self._client.fetch_balance()
        info = balance.get(symbol, {})
        free = info.get("free", 0)
        return Decimal(str(free)) if free else Decimal("0")

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.1, max=1),
        reraise=True,
    )
    async def place_order(
        self,
        pair: Pair,
        side: Side,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> Fill:
        t0 = time.monotonic()
        order_type = "limit" if price is not None else "market"

        try:
            if order_type == "limit":
                raw = await self._client.create_order(
                    symbol=pair,
                    type="limit",
                    side=side.value,
                    amount=float(quantity),
                    price=float(price),  # type: ignore[arg-type]
                )
            else:
                raw = await self._client.create_order(
                    symbol=pair,
                    type="market",
                    side=side.value,
                    amount=float(quantity),
                )

            latency = (time.monotonic() - t0) * 1000

            status_map = {
                "closed": OrderStatus.FILLED,
                "open": OrderStatus.PENDING,
                "canceled": OrderStatus.CANCELLED,
                "partially_filled": OrderStatus.PARTIAL,
            }

            fill_price = Decimal(str(raw.get("average", raw.get("price", 0))))
            fill_qty = Decimal(str(raw.get("filled", quantity)))
            fee_info = raw.get("fee", {})
            fee_cost = Decimal(str(fee_info.get("cost", 0)))
            fee_currency = Symbol(fee_info.get("currency", ""))

            return Fill(
                pair=pair,
                side=side,
                price=fill_price,
                quantity=fill_qty,
                fee=fee_cost,
                fee_currency=fee_currency,
                status=status_map.get(raw.get("status", ""), OrderStatus.FAILED),
                exchange_order_id=str(raw.get("id", "")),
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            log.error("order_failed", pair=pair, side=side.value, error=str(e))
            return Fill(
                pair=pair,
                side=side,
                price=Decimal("0"),
                quantity=Decimal("0"),
                fee=Decimal("0"),
                fee_currency=Symbol(""),
                status=OrderStatus.FAILED,
                exchange_order_id="",
                latency_ms=latency,
            )

    async def cancel_order(self, pair: Pair, order_id: str) -> bool:
        try:
            await self._client.cancel_order(order_id, pair)
            return True
        except ccxt_async.OrderNotFound:
            log.warning("cancel_not_found", pair=pair, order_id=order_id)
            return False
        except Exception as e:
            log.error("cancel_failed", pair=pair, order_id=order_id, error=str(e))
            return False

    async def get_trading_fees(self, pair: Pair) -> tuple[Decimal, Decimal]:
        if pair in self._fee_cache:
            return self._fee_cache[pair]

        try:
            fees = await self._client.fetch_trading_fee(pair)
            maker = Decimal(str(fees.get("maker", "0.001")))
            taker = Decimal(str(fees.get("taker", "0.001")))
            self._fee_cache[pair] = (maker, taker)
            return maker, taker
        except Exception:
            # Default Binance fees
            default = (Decimal("0.001"), Decimal("0.001"))
            self._fee_cache[pair] = default
            return default

    async def get_all_pairs(self) -> list[Pair]:
        markets = await self._client.load_markets()
        return [
            Pair(symbol)
            for symbol, market in markets.items()
            if market.get("active", False) and market.get("spot", False)
        ]

    async def close(self) -> None:
        await self._client.close()
