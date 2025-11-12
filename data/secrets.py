# Multi-exchange API configuration
EXCHANGES = {
    'binance': {
        'api_key': '',
        'secret': '',
        'enabled': True,
        'testnet': False
    },
    'coinbase': {
        'api_key': '',
        'secret': '',
        'passphrase': '',
        'enabled': False,
        'sandbox': False
    },
    'kraken': {
        'api_key': '',
        'secret': '',
        'enabled': False
    },
    'kucoin': {
        'api_key': '',
        'secret': '',
        'passphrase': '',
        'enabled': False,
        'sandbox': False
    },
    'okx': {
        'api_key': '',
        'secret': '',
        'passphrase': '',
        'enabled': False
    }
}

# Legacy support (deprecated - use EXCHANGES dict above)
BINANCE_KEY = EXCHANGES['binance']['api_key']
BINANCE_SECRET = EXCHANGES['binance']['secret']

# Telegram Bot Configuration (optional, for premium features)
# Get bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN = ""
# Get chat ID by messaging @userinfobot on Telegram
TELEGRAM_CHAT_ID = ""
