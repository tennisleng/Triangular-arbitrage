# Triangular Arbitrage Bot

This is a crypto arbitrage bot I built that scans Binance for triangular arbitrage opportunities across ETH/BTC/ALT pairs and executes trades automatically when it finds an edge.

The basic idea: if you can go ETH → ALT → BTC → ETH (or the reverse) and end up with more ETH than you started with after fees, that's free money. The catch is these windows are tiny and close fast, so the bot has to be quick about it.

## How it works

The bot continuously scans all available ALT pairs on Binance, checking both forward (ETH → ALT → BTC → ETH) and backward (ETH → BTC → ALT → ETH) paths. It uses multi-threading to scan batches of tokens in parallel — the market moves fast and sequential scanning would miss most opportunities.

For each path, it looks at the actual order book (not just ticker prices) to get realistic execution prices. Once it finds a spread that exceeds the minimum profit threshold after accounting for fees and estimated slippage, it fires off limit orders and walks the book at the best available prices.

### Key design choices

- **Maker fees over taker fees** — Limit orders cost 0.05% vs 0.1% for market orders. That difference matters a lot when you're chasing sub-1% spreads.
- **BNB fee discount** — If you hold BNB in your account, Binance gives a 25% fee reduction. The bot checks for this automatically and adjusts its math.
- **Dynamic position sizing** — Bigger opportunities get more capital allocated (up to 80% of balance), smaller ones get more conservative sizing. No point risking a lot on a 0.3% edge.
- **Slippage estimation** — Before committing to a trade, the bot scans the top 10 order book levels to estimate how much slippage you'd actually take at a given size. This prevents the classic "looked profitable on paper, lost money in execution" problem.

## Project structure

```
ini.py                  — Entry point. Spawns checker threads, runs the main loop.
src/model.py            — Core trading logic: order execution, price fetching, arbitrage estimation.
src/strategies.py       — Alternative strategy implementations (grid trading, funding rate, etc.)
src/arbitrage_algorithms.py — Advanced detection algorithms
src/dashboard.py        — Flask-based real-time dashboard with WebSocket updates
src/order_executor.py   — Order management and execution engine
src/ml_predictor.py     — ML-based opportunity prediction (XGBoost, LightGBM, PyTorch)
src/exchange_manager.py — Exchange connection management
src/telegram_notifier.py — Trade alerts via Telegram
data/settings.py        — All configurable parameters in one place
data/secrets.py         — API keys (not committed, obviously)
data/tokens.py          — List of ALT tokens to scan
```

## Getting started

```bash
pip install -r requirements.txt
```

Add your Binance API keys to `data/secrets.py`:
```python
BINANCE_KEY = "your-api-key"
BINANCE_SECRET = "your-api-secret"
```

Then run:
```bash
python ini.py
```

The bot will start scanning immediately and log everything to `logs.txt`. You can monitor profit in a separate terminal:
```bash
python check_profit.py
```

## Configuration

Everything lives in `data/settings.py`. The important ones:

| Setting | Default | What it does |
|---|---|---|
| `MIN_PROFIT_USD` | $1.00 | Won't execute a trade unless expected profit exceeds this |
| `AGGRESSIVE_MODE` | True | Lowers detection threshold by 20%, uses 8 threads instead of 5 |
| `ENABLE_DYNAMIC_POSITION_SIZING` | True | Scales position size based on opportunity quality |
| `ENABLE_SLIPPAGE_OPTIMIZATION` | True | Uses order book depth to estimate real execution costs |
| `MAX_POSITION_SIZE` | 0.5 | Max fraction of ETH balance to use per trade |

## Dashboard

There's a real-time web dashboard built with Flask + Socket.IO + Plotly that shows live opportunity scanning, profit over time, and trade history. It starts automatically on port 5001 when enabled, or you can access the REST API on port 5000 if you have a premium subscription.

## Backtesting

I ran a 10-year backtest across six strategies. All came back positive EV:

- **Funding Rate Arbitrage**: $17,269 total PnL (Sharpe: 58.07)
- **Grid Trading**: $6,485 across 34k trades
- **Futures Basis**: $4,903 (Sharpe: 6.30)
- **Market Making**: $581 across 8.5k trades
- **Cross-Exchange**: $3.91 across 42 trades
- **Triangular Arbitrage**: $2.09 across 53 trades

Full results in `backtest_results.json`. Run your own with `python run_backtest.py`.

## Telegram notifications

Optional but useful — set up a bot through [@BotFather](https://t.me/BotFather), drop the token and chat ID into `data/secrets.py`, and you'll get push notifications for every opportunity detected, trade executed, and daily summaries.

## What I learned building this

Triangular arbitrage on a single exchange is much harder than it sounds. The spreads are razor-thin — we're talking fractions of a percent — and by the time you detect an opportunity and get your orders filled, it's often gone. The math is straightforward; the execution is the hard part.

The biggest improvements came from switching to maker fees (cut costs in half), adding proper slippage estimation (stopped the bot from taking bad trades), and dynamic position sizing (stopped it from over-committing on marginal opportunities).

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.
