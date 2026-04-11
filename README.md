# Triangular Arbitrage Engine

High-frequency triangular arbitrage for crypto exchanges. Detects and executes three-leg arbitrage cycles (e.g., ETH → LTC → BTC → ETH) when price inefficiencies create risk-free profit opportunities.

Built with the same engineering rigor you'd find at a quantitative trading firm: Decimal arithmetic for financial math, immutable domain types, circuit breakers, and a fully async execution pipeline.

[![CI](https://github.com/tennisleng/Triangular-arbitrage/actions/workflows/ci.yml/badge.svg)](https://github.com/tennisleng/Triangular-arbitrage/actions/workflows/ci.yml)

---

## What It Does

Triangular arbitrage exploits pricing inconsistencies between three trading pairs on the same exchange. When the cross-rates between three currencies don't perfectly align, a cycle trade can extract the difference as profit.

```
ETH ──buy LTC/ETH──▶ LTC ──sell LTC/BTC──▶ BTC ──buy ETH/BTC──▶ ETH
 1.000                                                           1.003
                                                                  ↑
                                                          0.3% profit
```

This engine automates the entire pipeline: discovering valid triangles, monitoring order books for opportunities, evaluating profitability after fees and slippage, and executing all three legs atomically.

## Quick Start

```bash
git clone https://github.com/tennisleng/Triangular-arbitrage.git
cd Triangular-arbitrage
pip install -e ".[dev]"

cp config.example.yaml config.yaml
# Add your exchange API keys to config.yaml

triangular-arb              # Runs in paper trading mode by default
triangular-arb --dry-run    # Explicitly force paper trading
```

## Architecture

```
triangular_arb/
├── types.py              # Immutable domain types (Decimal, frozen dataclasses)
├── config.py             # Pydantic-validated YAML configuration
├── engine.py             # Async event loop orchestrator (7-stage pipeline)
├── cli.py                # CLI entry point with signal handling
├── exchange/
│   ├── adapter.py        # Abstract exchange interface (ABC)
│   └── binance.py        # Binance implementation via ccxt async
├── strategy/
│   ├── discovery.py      # Graph-based triangle enumeration
│   ├── evaluator.py      # Order-book-aware profit calculation
│   └── toxicity.py       # Order book manipulation detection
├── execution/
│   ├── executor.py       # Atomic tri-leg execution with rollback
│   └── stealth.py        # Anti-frontrunning countermeasures
├── risk/
│   └── manager.py        # Circuit breakers, drawdown guards
└── utils/
    └── logging.py        # Structured JSON logging (structlog)
```

The system is composed of five independent layers, each with a single responsibility:

**Discovery** — Builds a graph of all trading pairs and enumerates every valid 3-cycle. No hardcoded token lists; automatically adapts when exchanges add or remove pairs.

**Evaluation** — For each triangle, concurrently fetches all three order books and computes the effective round-trip rate using actual book prices (not just top-of-book). Accounts for three legs of fees, bid-ask spread, and available liquidity.

**Toxicity Detection** — Scores each order book for manipulation signals before committing capital. Detects spoofing (identical-size layering), informed flow (lopsided volume), disappearing liquidity, and thin/wide markets. A toxic book means someone is trying to exploit us — we skip and move on.

**Risk** — A pre-execution gate that can only reject, never modify. Checks minimum profit threshold, order book freshness (rejects stale data), daily drawdown limits, and consecutive loss circuit breakers.

**Stealth Execution** — Anti-frontrunning countermeasures that make our order flow harder to detect and exploit. Randomized order sizes, timing jitter between legs, exponential opportunity decay, adaptive thresholds, and triangle rotation.

**Exchange Adapter** — Abstract interface (ABC) that decouples all strategy logic from exchange specifics. The Binance implementation handles ccxt async calls, automatic retries with exponential backoff, and Decimal conversion at the boundary.

## Configuration

All configuration lives in a single YAML file validated by Pydantic at startup. If the config is invalid, the process fails immediately with a clear error instead of silently misbehaving mid-trade.

```yaml
exchange:
  exchange_id: binance
  api_key: ""
  api_secret: ""

risk:
  min_profit_bps: "5"        # Need at least 5 basis points net profit
  max_daily_loss_pct: "5"    # Circuit breaker at 5% daily drawdown
  max_consecutive_losses: 5  # Pause after 5 losses in a row

execution:
  use_limit_orders: true
  max_slippage_bps: "10"

scanner:
  base_currencies: ["ETH", "BTC", "USDT"]
  scan_interval_ms: 500

dry_run: true  # Paper trading (no real orders placed)
```

See [`config.example.yaml`](config.example.yaml) for the full reference with all available options.

## How Profit Is Calculated

The evaluator computes the net exchange rate around the triangle:

```
gross_rate = rate_leg1 × rate_leg2 × rate_leg3
net_rate   = gross_rate × (1 - fee)³
profit_bps = (net_rate - 1) × 10000
```

All arithmetic uses `decimal.Decimal`. A 64-bit float has ~15 significant digits — enough to silently round away the 1–5 basis point margins this system operates on. Every financial value enters the system as a string and is converted to Decimal exactly once, at the exchange adapter boundary.

## Risk Management

The risk manager enforces five checks before any trade is executed:

| Check | What It Does |
|-------|-------------|
| **Min profit threshold** | Rejects opportunities below the configured basis point minimum |
| **Stale book detection** | Rejects order books older than the staleness threshold (default: 2s) |
| **Daily drawdown limit** | Halts trading if cumulative daily losses exceed the configured % of starting balance |
| **Consecutive loss breaker** | Triggers a 5-minute cooldown after N consecutive losing trades |
| **Position limit** | Caps the number of concurrent open triangle positions |

The risk manager is designed as a pure gate: it can reject an opportunity, but it can never modify one. This makes every risk decision auditable — you can always answer "why was this trade rejected?" by checking the rejection reason enum.

## Anti-Frontrunning & Competitive Defense

In practice, triangular arbitrage is a Malthusian game. Opportunities are a finite resource consumed by the first firm to execute. This engine includes defenses against the specific ways competitors try to exploit algorithmic traders:

### Order Book Toxicity Detection

Before committing capital, every order book is scored for manipulation signals:

| Signal | What It Detects |
|--------|----------------|
| **Thin book** | Insufficient depth to absorb our order without moving price |
| **Wide spread** | Adverse selection risk — the market maker knows something |
| **Imbalanced volume** | Informed flow on one side (someone has an information edge) |
| **Layered orders** | Spoofing — identical-size levels creating fake depth |
| **Disappearing liquidity** | One dominant level that will be pulled when we hit it |

### Stealth Execution

| Defense | How It Works |
|---------|--------------|
| **Size randomization** | ±5% noise on every order. Round-number orders are trivially identifiable on the tape. |
| **Timing jitter** | Random delay between legs breaks the timing correlation that reveals multi-leg strategies |
| **Opportunity decay** | Exponential decay model — a 10bps opportunity that's 500ms old is likely already taken |
| **Adaptive thresholds** | When win rate drops (competitors are faster), we raise the bar and only take the safest trades |
| **Triangle heat** | Rotating across triangles prevents competitors from learning our patterns |

The key insight: when you can't outrun faster firms, you make yourself unpredictable. Randomized sizes, jittered timing, and rotating targets mean competitors can't build a reliable model of your behavior.

## Testing

```bash
pytest tests/ -v                                                # 45 tests
pytest tests/ --cov=triangular_arb --cov-report=term-missing    # With coverage
ruff check triangular_arb/ tests/                               # Lint
ruff format --check triangular_arb/ tests/                      # Format check
mypy triangular_arb/                                            # Type check
```

## Design Principles

- **Decimal, not float** — Financial math uses `decimal.Decimal` everywhere
- **Immutable types** — Frozen dataclasses prevent mutation across async boundaries
- **Adapter pattern** — Exchange interface is abstract; swap in a mock for testing
- **Fail fast** — Invalid config crashes at startup, not mid-trade
- **Structured logging** — JSON log entries, not text files parsed with regex
- **Single responsibility** — Each module does one thing; no God classes
- **Unpredictability** — Randomized execution makes frontrunning expensive

## Lineage

Originally forked from [Cherecho/Triangular-arbitrage](https://github.com/Cherecho/Triangular-arbitrage). Completely rewritten — no original code remains.

## License

[GPL-3.0](LICENSE)
