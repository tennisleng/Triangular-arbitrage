"""
Advanced arbitrage algorithms for triangular arbitrage detection and execution.
Implements various sophisticated arbitrage strategies including statistical arbitrage,
surface arbitrage, and multi-path optimization.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import statistics
from collections import deque


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""
    exchange: str
    base_currency: str
    quote_currency: str
    alt_currency: str
    direction: str  # 'forward' or 'backward'
    profit_percentage: float
    profit_usd: float
    estimated_profit: float
    path: List[str]
    orderbook_depth: int
    timestamp: datetime
    volatility: float = 0.0
    liquidity_score: float = 0.0


class AdvancedArbitrageAlgorithms:
    """Advanced algorithms for detecting and analyzing arbitrage opportunities."""

    def __init__(self, exchange_manager, settings):
        self.exchange_manager = exchange_manager
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Historical data for statistical analysis
        self.price_history = {}
        self.arbitrage_history = deque(maxlen=1000)
        self.volatility_window = 50
        self.profit_threshold_history = deque(maxlen=100)

    def detect_triangular_arbitrage(self, exchange_name: str, base: str, quote: str, alt: str) -> Optional[ArbitrageOpportunity]:
        """
        Enhanced triangular arbitrage detection with advanced algorithms.

        Args:
            exchange_name: Name of the exchange
            base: Base currency (e.g., 'ETH')
            quote: Quote currency (e.g., 'BTC')
            alt: Alternative currency for triangulation

        Returns:
            ArbitrageOpportunity if profitable opportunity found, None otherwise
        """
        try:
            # Get current market data
            base_quote_symbol = f"{base}/{quote}"
            base_alt_symbol = f"{base}/{alt}"
            alt_quote_symbol = f"{alt}/{quote}"

            base_quote_orderbook = self.exchange_manager.get_order_book(exchange_name, base_quote_symbol)
            base_alt_orderbook = self.exchange_manager.get_order_book(exchange_name, base_alt_symbol)
            alt_quote_orderbook = self.exchange_manager.get_order_book(exchange_name, alt_quote_symbol)

            if not all([base_quote_orderbook, base_alt_orderbook, alt_quote_orderbook]):
                return None

            # Forward arbitrage: base -> alt -> quote -> base
            forward_profit = self._calculate_forward_arbitrage(
                base_quote_orderbook, base_alt_orderbook, alt_quote_orderbook
            )

            # Backward arbitrage: base -> quote -> alt -> base
            backward_profit = self._calculate_backward_arbitrage(
                base_quote_orderbook, base_alt_orderbook, alt_quote_orderbook
            )

            # Apply advanced filtering
            opportunities = []

            if forward_profit['profit_percentage'] > self.settings.MIN_DIFFERENCE:
                opportunity = self._create_opportunity(
                    exchange_name, base, quote, alt, 'forward',
                    forward_profit, [base, alt, quote, base]
                )
                if self._passes_advanced_filters(opportunity):
                    opportunities.append(opportunity)

            if backward_profit['profit_percentage'] > self.settings.MIN_DIFFERENCE:
                opportunity = self._create_opportunity(
                    exchange_name, base, quote, alt, 'backward',
                    backward_profit, [base, quote, alt, base]
                )
                if self._passes_advanced_filters(opportunity):
                    opportunities.append(opportunity)

            # Return best opportunity
            return max(opportunities, key=lambda x: x.profit_percentage) if opportunities else None

        except Exception as e:
            self.logger.error(f"Error detecting triangular arbitrage: {e}")
            return None

    def _calculate_forward_arbitrage(self, base_quote_ob, base_alt_ob, alt_quote_ob) -> Dict[str, float]:
        """Calculate forward arbitrage: base -> alt -> quote -> base"""
        try:
            # Step 1: base -> alt (sell base for alt)
            base_alt_ask = base_alt_ob['asks'][0][0]  # Sell base for alt

            # Step 2: alt -> quote (sell alt for quote)
            alt_quote_ask = alt_quote_ob['asks'][0][0]  # Sell alt for quote

            # Step 3: quote -> base (buy base with quote)
            base_quote_bid = base_quote_ob['bids'][0][0]  # Buy base with quote

            # Calculate final amount after fees
            fee = self.exchange_manager.get_exchange_fee(self.exchange_manager.get_enabled_exchanges()[0])
            maker_fee = self.exchange_manager.get_exchange_fee(self.exchange_manager.get_enabled_exchanges()[0], maker=True)

            # Forward calculation with realistic fees
            final_amount = 1.0
            final_amount *= (1 - fee) / base_alt_ask  # Sell base for alt
            final_amount *= (1 - fee) / alt_quote_ask  # Sell alt for quote
            final_amount *= (1 - maker_fee) * base_quote_bid  # Buy base with quote (maker)

            profit_percentage = (final_amount - 1.0) * 100
            estimated_profit_usd = profit_percentage * 0.01 * self._get_base_usd_price()

            return {
                'profit_percentage': profit_percentage,
                'estimated_profit_usd': estimated_profit_usd,
                'final_amount': final_amount
            }
        except (IndexError, KeyError) as e:
            self.logger.debug(f"Insufficient orderbook data for forward arbitrage: {e}")
            return {'profit_percentage': 0, 'estimated_profit_usd': 0, 'final_amount': 1.0}

    def _calculate_backward_arbitrage(self, base_quote_ob, base_alt_ob, alt_quote_ob) -> Dict[str, float]:
        """Calculate backward arbitrage: base -> quote -> alt -> base"""
        try:
            # Step 1: base -> quote (sell base for quote)
            base_quote_ask = base_quote_ob['asks'][0][0]  # Sell base for quote

            # Step 2: quote -> alt (buy alt with quote)
            alt_quote_bid = alt_quote_ob['bids'][0][0]  # Buy alt with quote

            # Step 3: alt -> base (sell alt for base)
            base_alt_bid = base_alt_ob['bids'][0][0]  # Buy base with alt (sell alt for base)

            # Calculate final amount after fees
            fee = self.exchange_manager.get_exchange_fee(self.exchange_manager.get_enabled_exchanges()[0])
            maker_fee = self.exchange_manager.get_exchange_fee(self.exchange_manager.get_enabled_exchanges()[0], maker=True)

            # Backward calculation with realistic fees
            final_amount = 1.0
            final_amount *= (1 - fee) / base_quote_ask  # Sell base for quote
            final_amount *= (1 - maker_fee) * alt_quote_bid  # Buy alt with quote (maker)
            final_amount *= (1 - fee) * base_alt_bid  # Sell alt for base

            profit_percentage = (final_amount - 1.0) * 100
            estimated_profit_usd = profit_percentage * 0.01 * self._get_base_usd_price()

            return {
                'profit_percentage': profit_percentage,
                'estimated_profit_usd': estimated_profit_usd,
                'final_amount': final_amount
            }
        except (IndexError, KeyError) as e:
            self.logger.debug(f"Insufficient orderbook data for backward arbitrage: {e}")
            return {'profit_percentage': 0, 'estimated_profit_usd': 0, 'final_amount': 1.0}

    def _create_opportunity(self, exchange: str, base: str, quote: str, alt: str,
                          direction: str, profit_data: Dict, path: List[str]) -> ArbitrageOpportunity:
        """Create an ArbitrageOpportunity object with advanced metrics."""
        opportunity = ArbitrageOpportunity(
            exchange=exchange,
            base_currency=base,
            quote_currency=quote,
            alt_currency=alt,
            direction=direction,
            profit_percentage=profit_data['profit_percentage'],
            profit_usd=profit_data['estimated_profit_usd'],
            estimated_profit=profit_data['final_amount'] - 1.0,
            path=path,
            orderbook_depth=self.settings.ESTIMATION_ORDERBOOK,
            timestamp=datetime.now(),
            volatility=self._calculate_volatility(f"{base}/{quote}"),
            liquidity_score=self._calculate_liquidity_score(exchange, [f"{base}/{quote}", f"{base}/{alt}", f"{alt}/{quote}"])
        )
        return opportunity

    def _passes_advanced_filters(self, opportunity: ArbitrageOpportunity) -> bool:
        """Apply advanced filtering to arbitrage opportunities."""
        # Basic profit threshold
        if opportunity.profit_usd < self.settings.MIN_PROFIT_USD:
            return False

        # Volatility filter - avoid highly volatile opportunities
        if opportunity.volatility > 0.05:  # 5% volatility threshold
            return False

        # Liquidity filter - ensure sufficient liquidity
        if opportunity.liquidity_score < 0.3:  # Minimum liquidity score
            return False

        # Historical performance filter
        if not self._passes_historical_filter(opportunity):
            return False

        # Risk management filter
        if not self._passes_risk_management_filter(opportunity):
            return False

        return True

    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate price volatility for a symbol."""
        if symbol not in self.price_history:
            return 0.0

        prices = list(self.price_history[symbol])
        if len(prices) < 10:
            return 0.0

        try:
            returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
            return statistics.stdev(returns) if returns else 0.0
        except:
            return 0.0

    def _calculate_liquidity_score(self, exchange: str, symbols: List[str]) -> float:
        """Calculate liquidity score across multiple symbols."""
        total_score = 0.0
        valid_symbols = 0

        for symbol in symbols:
            orderbook = self.exchange_manager.get_order_book(exchange, symbol, limit=10)
            if orderbook and len(orderbook.get('bids', [])) >= 5 and len(orderbook.get('asks', [])) >= 5:
                # Calculate spread and depth
                best_bid = orderbook['bids'][0][0]
                best_ask = orderbook['asks'][0][0]
                spread = (best_ask - best_bid) / best_bid

                # Calculate depth (sum of top 5 levels)
                bid_depth = sum(vol for _, vol in orderbook['bids'][:5])
                ask_depth = sum(vol for _, vol in orderbook['asks'][:5])
                avg_depth = (bid_depth + ask_depth) / 2

                # Combine spread and depth into score
                spread_score = max(0, 1 - spread * 100)  # Lower spread = higher score
                depth_score = min(1.0, avg_depth / 1000)  # Normalize depth

                symbol_score = (spread_score + depth_score) / 2
                total_score += symbol_score
                valid_symbols += 1

        return total_score / max(valid_symbols, 1)

    def _passes_historical_filter(self, opportunity: ArbitrageOpportunity) -> bool:
        """Filter based on historical arbitrage performance."""
        if len(self.arbitrage_history) < 10:
            return True

        # Check success rate for this type of arbitrage
        similar_opportunities = [
            opp for opp in self.arbitrage_history
            if opp.base_currency == opportunity.base_currency and
               opp.quote_currency == opportunity.quote_currency and
               opp.direction == opportunity.direction
        ]

        if len(similar_opportunities) < 5:
            return True

        success_rate = sum(1 for opp in similar_opportunities if opp.profit_percentage > 0) / len(similar_opportunities)

        # Require at least 60% success rate for similar opportunities
        return success_rate >= 0.6

    def _passes_risk_management_filter(self, opportunity: ArbitrageOpportunity) -> bool:
        """Apply risk management filters."""
        # Check consecutive losses
        if hasattr(self, 'consecutive_losses') and self.consecutive_losses >= self.settings.MAX_CONSECUTIVE_LOSSES:
            return False

        # Check daily loss limit (simplified - would need daily tracking)
        if hasattr(self, 'daily_loss') and self.daily_loss >= self.settings.MAX_DAILY_LOSS_PERCENTAGE:
            return False

        # Position size check
        max_position_value = self._calculate_max_position_size(opportunity)
        if opportunity.profit_usd > max_position_value:
            return False

        return True

    def _calculate_max_position_size(self, opportunity: ArbitrageOpportunity) -> float:
        """Calculate maximum position size based on risk parameters."""
        # Get available balance in base currency
        balance = self.exchange_manager.get_balance(opportunity.exchange, opportunity.base_currency)

        # Apply position size limits
        max_by_balance = balance * self.settings.MAX_POSITION_SIZE

        # Dynamic sizing based on volatility and liquidity
        volatility_multiplier = max(0.1, 1 - opportunity.volatility * 10)
        liquidity_multiplier = opportunity.liquidity_score

        dynamic_size = max_by_balance * volatility_multiplier * liquidity_multiplier

        return min(dynamic_size, max_by_balance)

    def _get_base_usd_price(self) -> float:
        """Get USD price of base currency for profit calculations."""
        # Simplified - in production would fetch real price
        base_prices = {
            'ETH': 3000,
            'BTC': 50000,
            'BNB': 400,
            'ADA': 0.5,
            'SOL': 100
        }
        return base_prices.get(self.settings.DEFAULT_BASE_CURRENCY, 3000)

    def detect_statistical_arbitrage(self, exchange_name: str, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Detect statistical arbitrage opportunities using cointegration."""
        opportunities = []

        if len(symbols) < 2:
            return opportunities

        try:
            # Get price data for cointegration analysis
            price_data = {}
            for symbol in symbols:
                ticker = self.exchange_manager.get_ticker(exchange_name, symbol)
                if ticker:
                    price_data[symbol] = ticker['last']

            if len(price_data) < 2:
                return opportunities

            # Simple statistical arbitrage detection
            # In a full implementation, this would use cointegration tests
            prices = list(price_data.values())
            mean_price = statistics.mean(prices)
            std_dev = statistics.stdev(prices) if len(prices) > 1 else 0

            # Flag opportunities where price deviates significantly from mean
            for symbol, price in price_data.items():
                z_score = (price - mean_price) / std_dev if std_dev > 0 else 0

                if abs(z_score) > 2.0:  # 2 standard deviations
                    opportunity = ArbitrageOpportunity(
                        exchange=exchange_name,
                        base_currency=symbol.split('/')[0],
                        quote_currency=symbol.split('/')[1],
                        alt_currency='',  # Not applicable for statistical arb
                        direction='statistical',
                        profit_percentage=abs(z_score) * 2,  # Estimated profit
                        profit_usd=abs(z_score) * 10,  # Rough USD estimate
                        estimated_profit=abs(z_score) * 0.01,
                        path=[symbol],
                        orderbook_depth=1,
                        timestamp=datetime.now(),
                        volatility=std_dev / mean_price,
                        liquidity_score=0.5
                    )
                    opportunities.append(opportunity)

        except Exception as e:
            self.logger.error(f"Error in statistical arbitrage detection: {e}")

        return opportunities

    def detect_surface_arbitrage(self, exchange_name: str) -> List[ArbitrageOpportunity]:
        """Detect surface arbitrage using order book analysis."""
        opportunities = []

        try:
            # Get all available symbols for the exchange
            symbols = self.exchange_manager.get_exchange_tokens(exchange_name)
            base_currency = self.settings.DEFAULT_BASE_CURRENCY

            # Look for triangular relationships
            for alt in symbols[:10]:  # Limit for performance
                if alt == base_currency:
                    continue

                opportunity = self.detect_triangular_arbitrage(
                    exchange_name, base_currency, self.settings.DEFAULT_QUOTE_CURRENCY, alt
                )

                if opportunity:
                    opportunities.append(opportunity)

        except Exception as e:
            self.logger.error(f"Error in surface arbitrage detection: {e}")

        return opportunities

    def optimize_arbitrage_path(self, exchange_name: str, start_currency: str, target_currency: str,
                               max_depth: int = 3) -> Optional[ArbitrageOpportunity]:
        """Find optimal arbitrage path between two currencies using graph algorithms."""
        # This is a simplified implementation
        # Full implementation would use graph theory and shortest path algorithms

        try:
            # For now, just check direct triangular arbitrage
            symbols = self.exchange_manager.get_exchange_tokens(exchange_name)

            for intermediate in symbols[:5]:  # Check a few intermediates
                if intermediate in [start_currency, target_currency]:
                    continue

                opportunity = self.detect_triangular_arbitrage(
                    exchange_name, start_currency, target_currency, intermediate
                )

                if opportunity:
                    return opportunity

        except Exception as e:
            self.logger.error(f"Error in path optimization: {e}")

        return None

    def update_price_history(self, symbol: str, price: float):
        """Update price history for statistical analysis."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.volatility_window)

        self.price_history[symbol].append(price)

    def record_arbitrage_result(self, opportunity: ArbitrageOpportunity, success: bool):
        """Record arbitrage execution result for learning."""
        self.arbitrage_history.append(opportunity)

        if success:
            self.profit_threshold_history.append(opportunity.profit_percentage)
        else:
            # Track consecutive losses for risk management
            if not hasattr(self, 'consecutive_losses'):
                self.consecutive_losses = 0
            self.consecutive_losses = self.consecutive_losses + 1 if not success else 0

    def get_arbitrage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive arbitrage statistics."""
        if not self.arbitrage_history:
            return {}

        total_opportunities = len(self.arbitrage_history)
        profitable_opportunities = [opp for opp in self.arbitrage_history if opp.profit_percentage > 0]

        return {
            'total_opportunities': total_opportunities,
            'profitable_opportunities': len(profitable_opportunities),
            'success_rate': len(profitable_opportunities) / total_opportunities,
            'average_profit': statistics.mean([opp.profit_percentage for opp in profitable_opportunities]) if profitable_opportunities else 0,
            'best_profit': max([opp.profit_percentage for opp in self.arbitrage_history]) if self.arbitrage_history else 0,
            'worst_profit': min([opp.profit_percentage for opp in self.arbitrage_history]) if self.arbitrage_history else 0,
        }
