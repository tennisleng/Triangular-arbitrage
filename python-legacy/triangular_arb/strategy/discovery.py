"""
Triangle discovery via graph search.

Instead of hardcoding token lists, we build a graph of all trading pairs
and enumerate valid triangles. This automatically adapts to new listings
and delistings without config changes.
"""

from __future__ import annotations

from collections import defaultdict

import structlog

from triangular_arb.types import Pair, Side, Symbol, Triangle

log = structlog.get_logger()


def _parse_pair(pair: Pair) -> tuple[Symbol, Symbol]:
    """Split 'ETH/BTC' into (Symbol('ETH'), Symbol('BTC'))."""
    parts = pair.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair format: {pair}")
    return Symbol(parts[0]), Symbol(parts[1])


def discover_triangles(
    pairs: list[Pair],
    base_currencies: list[str],
) -> list[Triangle]:
    """
    Discover all valid triangular arbitrage paths.

    Algorithm:
    1. Build adjacency graph from all trading pairs
    2. For each base currency, find all 3-node cycles that
       start and end at that currency
    3. Deduplicate (A→B→C→A is the same triangle as C→A→B→C)

    Args:
        pairs: All available trading pairs on the exchange
        base_currencies: Currencies to use as the start/end of triangles

    Returns:
        List of Triangle objects representing valid arbitrage paths
    """
    # Build adjacency: symbol → set of symbols it can trade against
    adjacency: dict[str, set[str]] = defaultdict(set)
    pair_set: set[str] = set()

    for pair in pairs:
        try:
            base, quote = _parse_pair(pair)
            adjacency[base].add(quote)
            adjacency[quote].add(base)
            pair_set.add(pair)
        except ValueError:
            continue

    triangles: list[Triangle] = []
    seen: set[frozenset[str]] = set()

    for start in base_currencies:
        if start not in adjacency:
            continue

        # Find all 2-hop paths from start that return to start
        for mid_a in adjacency[start]:
            for mid_b in adjacency[mid_a]:
                if mid_b == start or mid_b == mid_a:
                    continue
                if start not in adjacency[mid_b]:
                    continue

                # Dedup: {ETH, LTC, BTC} is the same triangle regardless of direction
                key = frozenset([start, mid_a, mid_b])
                if key in seen:
                    continue
                seen.add(key)

                # Determine the pairs and sides for the forward direction
                # Forward: start → mid_a → mid_b → start
                triangle = _build_triangle(
                    base=Symbol(start),
                    mid_a=Symbol(mid_a),
                    mid_b=Symbol(mid_b),
                    pair_set=pair_set,
                )
                if triangle is not None:
                    triangles.append(triangle)

    log.info(
        "triangle_discovery_complete",
        total_pairs=len(pairs),
        triangles_found=len(triangles),
        base_currencies=base_currencies,
    )
    return triangles


def _build_triangle(
    base: Symbol,
    mid_a: Symbol,
    mid_b: Symbol,
    pair_set: set[str],
) -> Triangle | None:
    """
    Build a Triangle with correct pair directions and sides.

    For each leg, we need to determine:
    - Which pair exists (e.g., LTC/ETH vs ETH/LTC)
    - Whether we BUY or SELL on that pair
    """
    # Leg 1: base → mid_a
    leg1 = _resolve_leg(base, mid_a, pair_set)
    if leg1 is None:
        return None

    # Leg 2: mid_a → mid_b
    leg2 = _resolve_leg(mid_a, mid_b, pair_set)
    if leg2 is None:
        return None

    # Leg 3: mid_b → base
    leg3 = _resolve_leg(mid_b, base, pair_set)
    if leg3 is None:
        return None

    return Triangle(
        base=base,
        leg1_pair=leg1[0],
        leg1_side=leg1[1],
        leg2_pair=leg2[0],
        leg2_side=leg2[1],
        leg3_pair=leg3[0],
        leg3_side=leg3[1],
        intermediate_a=mid_a,
        intermediate_b=mid_b,
    )


def _resolve_leg(
    from_sym: Symbol,
    to_sym: Symbol,
    pair_set: set[str],
) -> tuple[Pair, Side] | None:
    """
    Determine the correct pair and side to go from `from_sym` to `to_sym`.

    If pair is FROM/TO: we SELL (sell FROM to get TO)
    If pair is TO/FROM: we BUY (buy TO with FROM)
    """
    forward = f"{from_sym}/{to_sym}"
    backward = f"{to_sym}/{from_sym}"

    if forward in pair_set:
        # Pair is FROM/TO — selling FROM gives us TO
        return Pair(forward), Side.SELL
    elif backward in pair_set:
        # Pair is TO/FROM — buying TO costs FROM
        return Pair(backward), Side.BUY
    else:
        return None
