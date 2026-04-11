# Triangular Arbitrage Bot

Automated crypto trading bot that finds and exploits price gaps across ETH/BTC/ALT pairs on Binance.

## What it does

- Scans every ALT token on Binance for triangular arbitrage opportunities in real time
- Checks both forward (ETH → ALT → BTC → ETH) and backward (ETH → BTC → ALT → ETH) paths simultaneously
- Uses actual order book depth, not ticker prices, to calculate real execution cost
- Executes trades automatically when expected profit exceeds the threshold
- Runs 8 parallel threads for maximum scanning speed

## Why it works

- Limit orders at 0.05% maker fee instead of 0.1% taker — cuts trading cost in half
- Detects BNB balance and applies the 25% Binance fee discount automatically
- Estimates slippage from order book depth before committing to any trade
- Scales position size based on opportunity quality — bigger edge gets more capital
- Filters out opportunities that look good on paper but would lose money in execution

## Backtest Performance (10yr simulated)

| Strategy | Total PnL | Trades | Profitable |
|---|---|---|---|
| Funding Rate Arb | $17,269 | 5,419 | ✅ |
| Grid Trading | $6,485 | 34,141 | ✅ |
| Futures Basis | $4,903 | 120 | ✅ |
| Market Making | $581 | 8,571 | ✅ |
| Cross-Exchange | $3.91 | 42 | ✅ |
| Triangular Arb | $2.09 | 53 | ✅ |

**Combined PnL: $29,244 — all strategies positive EV.**

## Configuration

| Setting | Default | Description |
|---|---|---|
| Min profit threshold | $1.00 | Minimum expected profit to execute a trade |
| Aggressive mode | On | Lowers detection threshold by 20%, uses 8 threads |
| Dynamic position sizing | On | Allocates 35-80% of balance based on opportunity quality |
| Slippage optimization | On | Analyzes order book depth before every trade |
| Max position size | 50% | Hard cap on balance used per trade |

## Setup

- Install dependencies with `pip install -r requirements.txt`
- Add Binance API keys to `data/secrets.py`
- Run `python ini.py` to start the bot
- Run `python check_profit.py` in a separate terminal to monitor profit
- Optional: add Telegram bot token and chat ID to `data/secrets.py` for push notifications

## Features

- **Live dashboard** — Flask + WebSocket UI on port 5001 with profit charts and trade history
- **Telegram alerts** — real-time notifications for every opportunity and trade
- **REST API** — stats, trade history, and subscription management on port 5000
- **ML prediction** — opportunity scoring with XGBoost, LightGBM, and PyTorch models
- **Profit tracking** — persistent tracking across restarts, saved to `profit_tracking.json`

## Tech

- Python
- ccxt (exchange API)
- Flask + Socket.IO (dashboard)
- Plotly (charting)
- scikit-learn, XGBoost, LightGBM, PyTorch (ML)
- Binance API

## License

GPL-3.0
