# Triangular Arbitrage Bot

A multi-threaded trading bot that exploits price inefficiencies across ETH/BTC/ALT pairs on Binance.

---

### The concept

When the price of a token is slightly different depending on which pair you trade through, there's a small window to profit. This bot catches those windows.

The bot checks both directions — forward (ETH → ALT → BTC → ETH) and backward (ETH → BTC → ALT → ETH) — in parallel across every available ALT pair, checks the actual order book depth (not just ticker price), and only executes when the math works out after fees and slippage.

---

### What makes this different

**It's not a toy.** Most open-source arb bots use ticker prices and ignore execution costs — they look profitable on paper but lose money in practice. This one:

- Reads order book depth before every trade to estimate real slippage
- Uses limit orders (0.05% maker fee) instead of market orders (0.1% taker fee)
- Auto-detects BNB balance for an additional 25% fee discount
- Scales position size dynamically — bigger edge = more capital, small edge = conservative
- Runs 8 parallel threads in aggressive mode for faster scanning

---

### Quickstart

```bash
pip install -r requirements.txt
```

Add your keys to `data/secrets.py`, then:

```bash
python ini.py              # start the bot
python check_profit.py     # monitor profit (separate terminal)
```

---

### Config

All in `data/settings.py`:

| | Default | |
|---|---|---|
| **Min profit threshold** | `$1.00` | Won't trade below this |
| **Aggressive mode** | `On` | 20% lower threshold, 8 threads |
| **Dynamic sizing** | `On` | 35-80% of balance per trade |
| **Slippage optimization** | `On` | Order book depth analysis |
| **Max position** | `50%` | Safety cap per trade |

---

### Project layout

```
ini.py                       entry point — spawns threads, runs the loop
src/
  model.py                   core logic — pricing, execution, arbitrage math
  strategies.py              alternative strategies (grid, funding rate, etc)
  arbitrage_algorithms.py    detection algorithms
  dashboard.py               live web dashboard (Flask + WebSocket)
  ml_predictor.py            ML-based prediction (XGBoost / LightGBM / PyTorch)
  order_executor.py          order management
  exchange_manager.py        exchange connections
  telegram_notifier.py       trade alerts via Telegram
data/
  settings.py                all tunable parameters
  secrets.py                 API keys (not committed)
  tokens.py                  token list to scan
```

---

### Backtest results (10yr simulated)

| Strategy | PnL | Trades |
|---|---|---|
| Funding Rate Arb | **$17,269** | 5,419 collections |
| Grid Trading | **$6,485** | 34,141 |
| Futures Basis | **$4,903** | 120 |
| Market Making | **$581** | 8,571 |
| Cross-Exchange | **$3.91** | 42 |
| Triangular Arb | **$2.09** | 53 |

All strategies positive EV. Full data in `backtest_results.json`, run your own with `python run_backtest.py`.

---

### Extras

- **Dashboard** — real-time web UI on port 5001 showing opportunities, profit charts, trade history
- **Telegram alerts** — set up via [@BotFather](https://t.me/BotFather), drop creds in `data/secrets.py`
- **REST API** — stats, trade history, subscription management on port 5000

---

*Built with Python, ccxt, and a lot of staring at order books.*
