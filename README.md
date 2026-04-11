# Triangular Arbitrage Engine

A high-frequency triangular arbitrage engine for cryptocurrency exchanges, built with institutional-grade engineering practices.

## Architecture

```
triangular_arb/
├── types.py              # Immutable domain types (Decimal arithmetic, no floats)
├── config.py             # Pydantic-validated YAML configuration
├── engine.py             # Main async event loop orchestrator
├── cli.py                # CLI with signal handling
├── exchange/
│   ├── adapter.py        # Abstract exchange interface (ABC)
│   └── binance.py        # Binance implementation via ccxt async
├── strategy/
│   ├── discovery.py      # Graph-based triangle enumeration
│   └── evaluator.py      # Order-book-aware profit calculation
├── execution/
│   └── executor.py       # Atomic tri-leg execution with rollback
├── risk/
│   └── manager.py        # Circuit breakers, drawdown guards, position limits
└── utils/
    └── logging.py        # Structured JSON logging via structlog
```

## Design Decisions

**Decimal arithmetic everywhere.** Financial calculations use `decimal.Decimal`, never `float`. A 64-bit float has ~15 significant digits — enough to silently round away the 1-5 basis point margins this system operates on. Decimal conversion happens at the exchange adapter boundary; internal code never sees floats.

**Immutable domain types.** All core types (`OrderBook`, `Triangle`, `Opportunity`, `Fill`, `ArbitrageResult`) are frozen dataclasses. This prevents accidental mutation across async tasks and makes the audit trail trivially reproducible.

**Exchange adapter pattern.** The `ExchangeAdapter` ABC decouples strategy logic from exchange specifics. Swap in a mock adapter and every strategy test runs without network calls. Adding a new exchange means implementing one interface, not modifying strategy code.

**Graph-based triangle discovery.** Instead of hardcoding token lists, we build an adjacency graph from all trading pairs and enumerate valid 3-cycles. This automatically adapts to new listings and delistings without config changes.

**Risk manager as a gate, not a modifier.** The risk manager can only reject opportunities — it never modifies them. This makes risk decisions auditable and prevents subtle bugs where risk adjustments interact with execution logic.

**Structured logging.** Every log entry is a JSON object with `timestamp`, `level`, and `event` fields. No more parsing text files with regex. Pipe to any log aggregation system.

## Quick Start

```bash
# Clone and install
git clone https://github.com/tennisleng/Triangular-arbitrage.git
cd Triangular-arbitrage
pip install -e ".[dev]"

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your API keys

# Run (paper trading by default)
triangular-arb --config config.yaml

# Run with explicit dry-run
triangular-arb --dry-run
```

## Configuration

Configuration is validated at startup via Pydantic. If the config is invalid, the process refuses to start rather than discovering misconfiguration mid-trade.

See [`config.example.yaml`](config.example.yaml) for all available options with documentation.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk.min_profit_bps` | `5` | Minimum net profit (basis points) to execute |
| `risk.max_daily_loss_pct` | `5` | Circuit breaker threshold (% of starting balance) |
| `risk.max_consecutive_losses` | `5` | Consecutive losses before cooldown |
| `execution.max_slippage_bps` | `10` | Max acceptable slippage per leg |
| `scanner.scan_interval_ms` | `500` | Time between full triangle scans |
| `dry_run` | `true` | Paper trading mode |

## How It Works

### 1. Triangle Discovery
On startup, the engine fetches all active trading pairs and builds an adjacency graph. It enumerates all valid 3-node cycles starting from configured base currencies (default: ETH, BTC, USDT). This typically finds 100-500+ triangles depending on the exchange.

### 2. Continuous Scanning
The async event loop scans each triangle by concurrently fetching all three order books (`asyncio.gather`). For each set of books, the evaluator computes the effective exchange rate around the triangle using actual order book prices (not just top-of-book).

### 3. Profit Evaluation
Profit is calculated as:
```
net_rate = (rate_leg1 × rate_leg2 × rate_leg3) × (1 - fee)³
net_profit_bps = (net_rate - 1) × 10000
```
The evaluator uses Decimal arithmetic and accounts for:
- Three legs of trading fees (compounded)
- Bid-ask spread on each leg
- Available liquidity (bottleneck sizing)

### 4. Risk Gating
Every opportunity passes through the risk manager, which checks:
- Net profit exceeds minimum threshold
- Order books are fresh (not stale)
- Daily drawdown is within limits
- No consecutive loss streak
- Circuit breaker is not tripped

### 5. Execution
The executor handles tri-leg execution sequentially (output of leg N is input to leg N+1). On any leg failure, it attempts best-effort rollback by reversing completed legs. All fills are recorded for audit trail regardless of outcome.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=triangular_arb --cov-report=term-missing

# Lint
ruff check triangular_arb/ tests/

# Type check
mypy triangular_arb/
```

## Project Lineage

Originally forked from [Cherecho/Triangular-arbitrage](https://github.com/Cherecho/Triangular-arbitrage). Completely rewritten with:
- Async architecture (was: blocking threads)
- Decimal financial math (was: floating point)
- Typed domain model (was: raw dicts)
- Exchange adapter pattern (was: hardcoded Binance calls)
- Graph-based discovery (was: hardcoded token list)
- Risk management with circuit breakers (was: none)
- Structured JSON logging (was: text file append)
- Pydantic config validation (was: Python files as config)
- Comprehensive test suite (was: zero tests)

## License

[GPL-3.0](LICENSE)
