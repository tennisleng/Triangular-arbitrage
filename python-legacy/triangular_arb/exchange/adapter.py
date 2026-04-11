"""
Exchange adapter protocol.

Any exchange backend must implement this interface. This decouples
strategy logic from exchange specifics and makes testing trivial
(swap in a mock exchange, no network calls needed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from triangular_arb.types import (
    ExchangeId,
    Fill,
    OrderBook,
    Pair,
    Side,
    Symbol,
)


class ExchangeAdapter(ABC):
    """
    Abstract interface for exchange operations.

    Implementations handle the specifics of each exchange's API,
    rate limiting, and error handling. Strategy code never touches
    raw exchange APIs directly.
    """

    @property
    @abstractmethod
    def exchange_id(self) -> ExchangeId:
        """Unique identifier for this exchange."""

    @abstractmethod
    async def fetch_order_book(self, pair: Pair, depth: int = 10) -> OrderBook:
        """
        Fetch a snapshot of the order book.

        Args:
            pair: Trading pair (e.g., "ETH/BTC")
            depth: Number of levels to fetch on each side

        Returns:
            OrderBook with bids descending and asks ascending by price.
        """

    @abstractmethod
    async def fetch_balance(self, symbol: Symbol) -> Decimal:
        """
        Get available (non-locked) balance for a symbol.

        Returns:
            Available balance as Decimal. Zero if the symbol is not held.
        """

    @abstractmethod
    async def place_order(
        self,
        pair: Pair,
        side: Side,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> Fill:
        """
        Place an order on the exchange.

        Args:
            pair: Trading pair
            side: BUY or SELL
            quantity: Amount of base currency
            price: Limit price. None = market order.

        Returns:
            Fill result with execution details.
        """

    @abstractmethod
    async def cancel_order(self, pair: Pair, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""

    @abstractmethod
    async def get_trading_fees(self, pair: Pair) -> tuple[Decimal, Decimal]:
        """
        Get maker and taker fees for a pair.

        Returns:
            (maker_fee, taker_fee) as fractions (e.g., 0.001 for 0.1%).
        """

    @abstractmethod
    async def get_all_pairs(self) -> list[Pair]:
        """Get all actively trading pairs on the exchange."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""
