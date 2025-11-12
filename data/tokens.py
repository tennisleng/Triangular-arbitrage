# Multi-exchange token configurations
EXCHANGE_TOKENS = {
    'binance': [
        'LTC', 'BNB', 'NEO', 'QTUM', 'ZRX', 'KNC', 'IOTA', 'LINK', 'XVG',
        'MTL', 'ETC', 'ZEC', 'DASH', 'XRP', 'ENJ', 'STORJ', 'BAT', 'LSK',
        'MANA', 'ADA', 'XLM', 'WAVES', 'ICX', 'STEEM', 'NANO', 'ONT', 'ZIL',
        'THETA', 'VET', 'HOT', 'DOGE', 'SOL', 'AVAX', 'DOT', 'MATIC'
    ],
    'coinbase': [
        'BTC', 'ETH', 'LTC', 'BCH', 'ETC', 'ZRX', 'BAT', 'LINK', 'ADA',
        'XLM', 'ALGO', 'DOGE', 'AVAX', 'SOL', 'DOT', 'MATIC', 'UNI'
    ],
    'kraken': [
        'ADA', 'ALGO', 'ANT', 'BAT', 'COMP', 'DOT', 'ETH', 'FIL', 'FLOW',
        'GRT', 'ICP', 'KAR', 'KAVA', 'KEEP', 'KNC', 'LINK', 'LSK', 'LTC',
        'MANA', 'OXT', 'QTUM', 'REP', 'SC', 'STORJ', 'TRX', 'UNI', 'WAVES',
        'XMR', 'XRP', 'XTZ', 'ZEC'
    ],
    'kucoin': [
        'BTC', 'ETH', 'LTC', 'ADA', 'DOT', 'SOL', 'AVAX', 'MATIC', 'LINK',
        'UNI', 'ALGO', 'VET', 'ICP', 'FIL', 'TRX', 'ETC', 'XLM', 'DOGE',
        'SHIB', 'SUSHI', 'CAKE', 'XRP', 'BCH'
    ],
    'okx': [
        'BTC', 'ETH', 'LTC', 'ADA', 'DOT', 'SOL', 'AVAX', 'MATIC', 'LINK',
        'UNI', 'ALGO', 'FIL', 'ICP', 'TRX', 'ETC', 'XLM', 'DOGE', 'SHIB',
        'SUSHI', 'CAKE', 'XRP', 'BCH', 'THETA', 'VET'
    ]
}

# Legacy support (deprecated - use EXCHANGE_TOKENS dict above)
binance_tokens = EXCHANGE_TOKENS['binance']
