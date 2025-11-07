# The min difference to do the arbitrage (improved for actual profitability)
MIN_DIFFERENCE = 0.15  # Changed to positive - need actual profit after fees
# The orderbook to take for estimation
ESTIMATION_ORDERBOOK = 2
# The tries that it will make
MAX_TRIES_ORDERBOOK = 14
# How many seconds should we wait between best buy/ best sell attempts
WAIT_BETWEEN_ORDER = 1
# Minimum profit in USD to execute trade (lowered to $1 for more opportunities)
MIN_PROFIT_USD = 1.0
# Position size as percentage of available balance (risk management)
MAX_POSITION_SIZE = 0.5  # Use max 50% of balance per trade
# Advanced algorithm settings
ENABLE_DYNAMIC_POSITION_SIZING = True  # Adjust position size based on opportunity
ENABLE_SLIPPAGE_OPTIMIZATION = True  # Optimize for slippage
ENABLE_MULTI_PATH_DETECTION = True  # Detect multiple arbitrage paths
MIN_ORDERBOOK_DEPTH = 3  # Minimum order book depth required
AGGRESSIVE_MODE = True  # More aggressive opportunity detection
# Enable premium features
PREMIUM_FEATURES_ENABLED = False
