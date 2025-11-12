# Multi-exchange settings
ENABLED_EXCHANGES = ['binance']  # List of enabled exchanges
DEFAULT_BASE_CURRENCY = 'ETH'  # Default base currency for arbitrage
DEFAULT_QUOTE_CURRENCY = 'BTC'  # Default quote currency for arbitrage

# Arbitrage settings
MIN_DIFFERENCE = 0.15  # Minimum profit percentage to execute arbitrage
MIN_PROFIT_USD = 1.0   # Minimum profit in USD to execute trade
MAX_POSITION_SIZE = 0.5  # Maximum position size as percentage of balance

# Order execution settings
ESTIMATION_ORDERBOOK = 2  # Number of orderbook levels to use for estimation
MAX_TRIES_ORDERBOOK = 14  # Maximum tries for orderbook operations
WAIT_BETWEEN_ORDER = 1  # Seconds to wait between order attempts
MIN_ORDERBOOK_DEPTH = 3  # Minimum orderbook depth required

# Advanced algorithm settings
ENABLE_DYNAMIC_POSITION_SIZING = True  # Adjust position size based on opportunity
ENABLE_SLIPPAGE_OPTIMIZATION = True   # Optimize for price slippage
ENABLE_MULTI_PATH_DETECTION = True    # Detect multiple arbitrage paths
AGGRESSIVE_MODE = True               # More aggressive opportunity detection

# Cross-exchange arbitrage settings
ENABLE_CROSS_EXCHANGE_ARBITRAGE = False  # Enable arbitrage between different exchanges
CROSS_EXCHANGE_MIN_PROFIT_USD = 5.0     # Minimum profit for cross-exchange arbitrage
MAX_CROSS_EXCHANGE_LATENCY = 2.0        # Maximum allowed latency between exchanges (seconds)

# Risk management settings
MAX_DAILY_LOSS_PERCENTAGE = 10.0  # Maximum daily loss as percentage of starting balance
MAX_CONSECUTIVE_LOSSES = 5        # Maximum consecutive losing trades before pausing
ENABLE_CIRCUIT_BREAKER = True     # Enable circuit breaker on consecutive losses

# Performance settings
THREAD_COUNT = 8  # Number of threads for parallel processing
CACHE_TTL_SECONDS = 30  # Cache time-to-live in seconds
ENABLE_PRICE_CACHE = True  # Enable price caching for performance

# Premium features
PREMIUM_FEATURES_ENABLED = False

# Exchange-specific settings
EXCHANGE_SETTINGS = {
    'binance': {
        'fee': 0.001,      # 0.1% taker fee
        'maker_fee': 0.0005,  # 0.05% maker fee
        'min_order_size': 0.0001,
        'max_leverage': 1  # Spot trading only
    },
    'coinbase': {
        'fee': 0.005,      # 0.5% fee
        'maker_fee': 0.0035,  # 0.35% maker fee
        'min_order_size': 0.0001,
        'max_leverage': 1
    },
    'kraken': {
        'fee': 0.0026,     # 0.26% fee
        'maker_fee': 0.0016,  # 0.16% maker fee
        'min_order_size': 0.0001,
        'max_leverage': 1
    },
    'kucoin': {
        'fee': 0.001,      # 0.1% fee
        'maker_fee': 0.0009,  # 0.09% maker fee
        'min_order_size': 0.0001,
        'max_leverage': 5  # Up to 5x leverage
    },
    'okx': {
        'fee': 0.001,      # 0.1% fee
        'maker_fee': 0.0008,  # 0.08% maker fee
        'min_order_size': 0.0001,
        'max_leverage': 10  # Up to 10x leverage
    }
}
