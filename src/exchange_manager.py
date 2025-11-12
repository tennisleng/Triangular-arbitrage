"""
Multi-exchange manager for triangular arbitrage bot.
Handles connections and operations across multiple cryptocurrency exchanges.
"""

import ccxt
import time
import logging
from typing import Dict, List, Optional, Any
from data import secrets, settings, tokens


class ExchangeManager:
    """Manages multiple cryptocurrency exchange connections and operations."""

    def __init__(self):
        self.exchanges = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize all enabled exchanges."""
        for exchange_name in settings.ENABLED_EXCHANGES:
            try:
                self._initialize_single_exchange(exchange_name)
                self.logger.info(f"Successfully initialized {exchange_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {e}")

    def _initialize_single_exchange(self, exchange_name: str):
        """Initialize a single exchange connection."""
        exchange_config = secrets.EXCHANGES.get(exchange_name, {})
        if not exchange_config.get('enabled', False):
            return

        config = {
            'apiKey': exchange_config.get('api_key', ''),
            'secret': exchange_config.get('secret', ''),
            'timeout': 30000,
            'enableRateLimit': True
        }

        # Exchange-specific configurations
        if exchange_name == 'binance':
            config['options'] = {'testnet': exchange_config.get('testnet', False)}
        elif exchange_name == 'coinbase':
            config['password'] = exchange_config.get('passphrase', '')
            config['sandbox'] = exchange_config.get('sandbox', False)
        elif exchange_name == 'kucoin':
            config['password'] = exchange_config.get('passphrase', '')
            config['sandbox'] = exchange_config.get('sandbox', False)
        elif exchange_name == 'okx':
            config['password'] = exchange_config.get('passphrase', '')

        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class(config)

        self.exchanges[exchange_name] = {
            'instance': exchange,
            'config': exchange_config,
            'settings': settings.EXCHANGE_SETTINGS.get(exchange_name, {}),
            'tokens': tokens.EXCHANGE_TOKENS.get(exchange_name, [])
        }

    def get_exchange(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """Get exchange instance and configuration."""
        return self.exchanges.get(exchange_name)

    def get_enabled_exchanges(self) -> List[str]:
        """Get list of enabled exchange names."""
        return list(self.exchanges.keys())

    def get_exchange_tokens(self, exchange_name: str) -> List[str]:
        """Get supported tokens for an exchange."""
        exchange = self.get_exchange(exchange_name)
        return exchange['tokens'] if exchange else []

    def get_exchange_fee(self, exchange_name: str, maker: bool = False) -> float:
        """Get trading fee for an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return 0.001  # Default 0.1% fee

        fee_key = 'maker_fee' if maker else 'fee'
        return exchange['settings'].get(fee_key, 0.001)

    def get_balance(self, exchange_name: str, currency: str) -> float:
        """Get balance for a currency on an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return 0.0

        try:
            balance = exchange['instance'].fetch_balance()
            return balance.get(currency, {}).get('free', 0.0)
        except Exception as e:
            self.logger.error(f"Failed to get balance for {currency} on {exchange_name}: {e}")
            return 0.0

    def get_ticker(self, exchange_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for a symbol on an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return None

        try:
            return exchange['instance'].fetch_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol} on {exchange_name}: {e}")
            return None

    def get_order_book(self, exchange_name: str, symbol: str, limit: int = 5) -> Optional[Dict[str, Any]]:
        """Get order book for a symbol on an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return None

        try:
            return exchange['instance'].fetch_order_book(symbol, limit)
        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol} on {exchange_name}: {e}")
            return None

    def create_market_order(self, exchange_name: str, symbol: str, side: str,
                           amount: float) -> Optional[Dict[str, Any]]:
        """Create a market order on an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return None

        try:
            if side.lower() == 'buy':
                return exchange['instance'].create_market_buy_order(symbol, amount)
            elif side.lower() == 'sell':
                return exchange['instance'].create_market_sell_order(symbol, amount)
            else:
                self.logger.error(f"Invalid order side: {side}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create {side} order for {symbol} on {exchange_name}: {e}")
            return None

    def create_limit_order(self, exchange_name: str, symbol: str, side: str,
                          amount: float, price: float) -> Optional[Dict[str, Any]]:
        """Create a limit order on an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return None

        try:
            if side.lower() == 'buy':
                return exchange['instance'].create_limit_buy_order(symbol, amount, price)
            elif side.lower() == 'sell':
                return exchange['instance'].create_limit_sell_order(symbol, amount, price)
            else:
                self.logger.error(f"Invalid order side: {side}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create {side} limit order for {symbol} on {exchange_name}: {e}")
            return None

    def cancel_order(self, exchange_name: str, order_id: str, symbol: str) -> bool:
        """Cancel an order on an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return False

        try:
            exchange['instance'].cancel_order(order_id, symbol)
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id} on {exchange_name}: {e}")
            return False

    def get_open_orders(self, exchange_name: str, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open orders for an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return []

        try:
            return exchange['instance'].fetch_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get open orders on {exchange_name}: {e}")
            return []

    def test_connectivity(self, exchange_name: str) -> bool:
        """Test connectivity to an exchange."""
        exchange = self.get_exchange(exchange_name)
        if not exchange:
            return False

        try:
            # Simple ping test
            exchange['instance'].load_markets()
            return True
        except Exception as e:
            self.logger.error(f"Connectivity test failed for {exchange_name}: {e}")
            return False

    def get_exchange_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all exchanges."""
        info = {}
        for name, exchange_data in self.exchanges.items():
            info[name] = {
                'connected': self.test_connectivity(name),
                'tokens_count': len(exchange_data['tokens']),
                'fee': self.get_exchange_fee(name),
                'maker_fee': self.get_exchange_fee(name, maker=True)
            }
        return info
