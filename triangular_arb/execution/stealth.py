"""
Stealth execution — anti-frontrunning countermeasures.

When you trade the same triangle repeatedly, competitors will:
1. Detect your pattern from order flow
2. Frontrun you by snapping the opportunity first
3. Bait you with phantom liquidity

This module makes your execution unpredictable:
- Randomized order sizes (never round numbers)
- Timing jitter between legs
- Opportunity decay modeling (skip stale edges)
- Adaptive thresholds based on recent win rate
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal

import structlog

from triangular_arb.types import Opportunity

log = structlog.get_logger()

# Seed with hardware entropy, not predictable seed
_rng = random.SystemRandom()


@dataclass
class CompetitionState:
    """
    Tracks how competitive the market is right now.

    If win rate drops, it means competitors are eating our lunch.
    We respond by raising thresholds (only take the best opportunities)
    and increasing execution jitter (harder to detect our pattern).
    """

    recent_results: list[bool] = field(default_factory=list)
    window_size: int = 50
    last_execution_ns: int = 0
    executions_per_triangle: dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        if not self.recent_results:
            return 0.5  # Assume neutral before we have data
        return sum(self.recent_results) / len(self.recent_results)

    @property
    def competition_intensity(self) -> float:
        """0.0 = no competition, 1.0 = extremely competitive."""
        # Below 30% win rate = very competitive market
        # Above 70% win rate = low competition
        return max(0.0, min(1.0, 1.0 - (self.win_rate - 0.2) / 0.6))

    def record(self, won: bool) -> None:
        self.recent_results.append(won)
        if len(self.recent_results) > self.window_size:
            self.recent_results.pop(0)

    def record_execution(self, triangle_key: str) -> None:
        self.last_execution_ns = time.time_ns()
        self.executions_per_triangle[triangle_key] = (
            self.executions_per_triangle.get(triangle_key, 0) + 1
        )


def randomize_size(size: Decimal, noise_pct: float = 0.05) -> Decimal:
    """
    Add noise to order size so competitors can't fingerprint us.

    Round-number orders (0.5, 1.0, 10.0) are trivially identifiable
    on the tape. Adding ±5% noise makes our flow look like retail.

    Args:
        size: Base order size
        noise_pct: Maximum noise as fraction (default 5%)

    Returns:
        Size with random noise applied, always positive
    """
    noise = Decimal(str(_rng.uniform(-noise_pct, noise_pct)))
    noisy = size * (Decimal("1") + noise)
    # Truncate to 8 decimal places (standard crypto precision)
    return max(noisy.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN), Decimal("0.00000001"))


async def jitter_delay(
    base_ms: float = 0,
    max_jitter_ms: float = 50,
    competition: float = 0.0,
) -> None:
    """
    Add random delay between legs to break timing patterns.

    Competitors correlate the timing of sequential orders to
    detect multi-leg strategies. Random jitter breaks this signal.

    In high-competition environments, we add MORE jitter (counterintuitive
    but correct — speed alone won't win against faster firms, but
    unpredictability prevents them from modeling us).

    Args:
        base_ms: Minimum delay in milliseconds
        max_jitter_ms: Maximum additional random delay
        competition: Competition intensity (0-1), increases jitter
    """
    # Scale jitter with competition — more competitive = more random
    scaled_max = max_jitter_ms * (1 + competition)
    delay_ms = base_ms + _rng.uniform(0, scaled_max)
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000)


def opportunity_decay(
    opportunity: Opportunity,
    half_life_ms: float = 200,
) -> Decimal:
    """
    Model opportunity decay due to competition.

    In a Malthusian market, arbitrage opportunities don't persist —
    they decay exponentially as competitors race to capture them.
    A 10bps opportunity that's 500ms old is worth much less than
    a fresh one, because someone probably already took it.

    Args:
        opportunity: The opportunity to evaluate
        half_life_ms: Time for opportunity value to halve

    Returns:
        Decay-adjusted net profit in basis points
    """
    age_ms = opportunity.age_ms
    if age_ms <= 0:
        return opportunity.net_profit_bps

    # Exponential decay: value = initial * 2^(-age/half_life)
    decay_factor = Decimal(str(math.pow(2, -age_ms / half_life_ms)))
    return (opportunity.net_profit_bps * decay_factor).quantize(Decimal("0.01"))


def adaptive_threshold(
    base_threshold_bps: Decimal,
    competition: CompetitionState,
) -> Decimal:
    """
    Adjust minimum profit threshold based on competition.

    When win rate drops (competitors are faster/smarter):
    - Raise the threshold to only take the safest opportunities
    - This reduces volume but improves expected value per trade

    When win rate is high (low competition):
    - Lower the threshold to capture more volume
    - Accept thinner margins when we're likely to win

    Args:
        base_threshold_bps: Default minimum profit threshold
        competition: Current competition state

    Returns:
        Adjusted threshold in basis points
    """
    intensity = competition.competition_intensity

    # Scale: 0 competition → 0.7x threshold, 1.0 competition → 2.0x threshold
    multiplier = Decimal(str(0.7 + 1.3 * intensity))
    adjusted = base_threshold_bps * multiplier
    return adjusted.quantize(Decimal("0.1"))


def triangle_heat(
    triangle_key: str,
    competition: CompetitionState,
    cooldown_per_execution: int = 3,
) -> bool:
    """
    Check if a triangle is "hot" (traded too recently/frequently).

    Repeatedly hitting the same triangle creates a detectable pattern.
    Competitors will learn: "when X/Y/Z books tighten, someone always
    hits the arb within 200ms" — and they'll frontrun it.

    Rotating across triangles makes the pattern harder to detect.

    Returns:
        True if the triangle should be skipped (too hot)
    """
    count = competition.executions_per_triangle.get(triangle_key, 0)
    if count == 0:
        return False

    # Cooldown scales with how many times we've hit this triangle
    min_interval_ns = cooldown_per_execution * count * 1_000_000_000
    elapsed_ns = time.time_ns() - competition.last_execution_ns
    return elapsed_ns < min_interval_ns
