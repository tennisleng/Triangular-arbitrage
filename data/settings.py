# The min difference to do the arbitrage (improved for actual profitability)
MIN_DIFFERENCE = 0.15  # Changed to positive - need actual profit after fees
# The orderbook to take for estimation
ESTIMATION_ORDERBOOK = 2
# The tries that it will make
MAX_TRIES_ORDERBOOK = 14
# How many seconds should we wait between best buy/ best sell attempts
WAIT_BETWEEN_ORDER = 1
# Minimum profit in USD to execute trade
MIN_PROFIT_USD = 5.0
# Position size as percentage of available balance (risk management)
MAX_POSITION_SIZE = 0.5  # Use max 50% of balance per trade
# Enable premium features
PREMIUM_FEATURES_ENABLED = False
