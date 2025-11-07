# Triangular-arbitrage
## What Is Triangular Arbitrage?
Triangular arbitrage is the result of a discrepancy between three foreign currencies that occurs when the currency's exchange rates do not exactly match up. These opportunities are rare and traders who take advantage of them usually have advanced computer equipment and/or programs to automate the process.<br><br>
![Triangular-Arbitrage](https://user-images.githubusercontent.com/48016628/166167936-23ad03bb-2825-4efe-ace6-fc8b403b30e8.png)

## Description of this script
This is a triangular arbitrage bot made entirely in python using the ccxt libraries to handle the binance api, It's an endless loop that scans opportunity for binance and executes the operations if found, multi-threaded and with a simple log that it save in a file.<br>
The base money for this bot is ETH, while the exchange currency is BTC, as a result, there are two types of arbitrage:<br> 

- **Forward**: passing first from **ETH** through the **ALT** to then be changed to **BTC** and back to **ETH**
- **Backward**: passing first from **ETH** through **BTC** to then be changed to the selected **ALT** and back to **ETH**

### Does this make money?
**YES!** This enhanced version includes multiple improvements to increase profitability:

- **Optimized fee calculations**: Uses maker fees (0.05%) instead of taker fees (0.1%) for better margins
- **Smart profit thresholds**: Only executes trades with minimum $5 USD profit
- **Risk management**: Position sizing limits (max 50% per trade)
- **Better profit estimation**: USD-based calculations with real-time price data
- **Profit tracking**: Automatic tracking and analytics

## 🚀 Monetization Features

This bot now includes monetization capabilities:

### Premium Features:
- **Telegram Notifications**: Real-time alerts for opportunities and trades
- **REST API**: Access bot statistics and trade history via API
- **Advanced Analytics**: Profit tracking, success rates, and performance metrics
- **Subscription System**: Tiered access (Free, Basic, Premium, Enterprise)

See [MONETIZATION.md](MONETIZATION.md) for detailed monetization guide.

## Dependencies
Install all dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- ccxt (cryptocurrency exchange library)
- flask (for API server)
- flask-restful (for REST API)
- requests (for Telegram notifications)

## Requirements
- [Binance account](https://www.binance.com/es) with API keys
- (Optional) Telegram bot token for notifications (get from @BotFather)

## Quick Start

1. **Configure API keys**:
   ```bash
   # Edit data/secrets.py
   BINANCE_KEY = "your_binance_api_key"
   BINANCE_SECRET = "your_binance_secret"
   ```

2. **Optional: Setup Telegram** (for premium features):
   ```bash
   # Edit data/secrets.py
   TELEGRAM_BOT_TOKEN = "your_bot_token"
   TELEGRAM_CHAT_ID = "your_chat_id"
   ```

3. **Activate license** (for premium features):
   ```bash
   python activate_license.py
   ```

4. **Run the bot**:
   ```bash
   python ini.py
   ```

5. **Start API server** (if premium tier):
   ```bash
   python src/api_server.py
   ```

## TODO
- ✅ Implement Telegram Bot notifications
- ✅ Add subscription/licensing system
- ✅ Create REST API for monetization
- ✅ Improve profitability with better fee calculations
- ⏳ Make the base currency eligible  
- ⏳ Add more exchanges
- ⏳ Make base currency selectable

## Disclaimer:
### This bot is intended to be a Proof-of-concept. The developer will not be responsible for Any losses that are made are as a result of using this tool. Understand the risks involved and Only invest amounts you are willing to lose.