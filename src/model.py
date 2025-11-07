from datetime import datetime
from operator import itemgetter
from data import secrets , settings

import time
import ccxt

# Import Telegram notifier if available
try:
    from src.telegram_notifier import TelegramNotifier
    telegram_available = True
except:
    telegram_available = False


class Model:

    # Exchange
    binance = None
    fee = 0.999  # Taker fee (0.1%)
    maker_fee = 0.9995  # Maker fee (0.05% with potential BNB discount)
    
    # Profit tracking
    total_profit_eth = 0.0
    total_profit_usd = 0.0
    trades_executed = 0
    successful_trades = 0

    # Cache
    cache_prices = []
    cache_order_books = []

    # Order status
    not_filled = 0
    in_progress = 1
    filled = 2

    # Init binance, create connections with secrets file.
    def __init__(self):
        # Check subscription/license
        try:
            from src.subscription import SubscriptionManager
            self.subscription = SubscriptionManager()
            self.subscription.check_premium_features()
        except:
            self.subscription = None
        
        self.binance = ccxt.binance({
            'apiKey': secrets.BINANCE_KEY , 'secret': secrets.BINANCE_SECRET , 'timeout': 30000 ,
            'enableRateLimit': True
        })
        self.load_profit_tracking()
        # Initialize Telegram notifier
        if telegram_available:
            self.telegram = TelegramNotifier()
        else:
            self.telegram = None

    # Create a buy order. 'amount' or 'amount_percentage' should be specified.
    def buy(self , exchange , asset1 , asset2 , amount_percentage=None , amount=None , limit=None , timeout=None):
        try:
            if amount_percentage:
                asset2_available = self.get_balance(exchange , asset2) * amount_percentage
                amount = asset2_available / self.get_price(exchange , asset1 , asset2 , mode='ask')

            self.log("Buying {:.6} {} with {} on {}.".format(
                amount , asset1 , asset2 , exchange
            ))

            if limit:
                self.log("Limit @{}.".format(limit))

            if not limit:
                self.log("Buying at market price.")
                exchange.createMarketBuyOrder(
                    '{}/{}'.format(asset1 , asset2) , amount
                )
                return True

            else:
                exchange.createLimitBuyOrder(
                    '{}/{}'.format(asset1 , asset2) , amount , limit
                )

                if timeout:
                    time.sleep(timeout)
                    result = self.is_open_order(exchange , asset1 , asset2)

                    if result == Model.not_filled:
                        if self.cancel_orders(exchange , asset1 , asset2):
                            self.log("Canceled limit order for {}/{} after timeout.".format(asset1 , asset2))

                        return False

                    elif result == Model.in_progress:
                        n = 0
                        while result == Model.in_progress:
                            n += 1
                            if n >= 20:
                                self.log("Order cannot be terminated, selling {} to {}.".format(asset1 , asset2))
                                self.sell(exchange , asset1 , asset2 , amount_percentage=1)
                                return False

                            self.log("Order for {}/{} is in progress, waiting...".format(asset1 , asset2))
                            time.sleep(timeout)
                            result = self.is_open_order(exchange , asset1 , asset2)

                        self.log("Limit order executed.")
                        return True

                    else:
                        self.log("Limit order executed.")
                        return True

                else:
                    return False

        except Exception as e:
            self.log("Error while buying: {}".format(str(e)))
            return False

    # Create a sell order. 'amount' or 'amount_percentage' should be specified.
    def sell(self , exchange , asset1 , asset2 , amount_percentage=None , amount=None , limit=None , timeout=None):
        try:
            if amount_percentage:
                amount = self.get_balance(exchange , asset1) * amount_percentage

            self.log("Selling {:.6f} {} to {} on {}.".format(
                amount , asset1 , asset2 , exchange
            ))

            if limit:
                self.log("Limit @{}.".format(limit))

            if not limit:
                self.log("Selling at market price.")
                exchange.createMarketSellOrder(
                    '{}/{}'.format(asset1 , asset2) , amount
                )
                return True

            else:
                exchange.createLimitSellOrder(
                    '{}/{}'.format(asset1 , asset2) , amount , limit
                )

                if timeout:
                    time.sleep(timeout)
                    result = self.is_open_order(exchange , asset1 , asset2)

                    if result == Model.not_filled:
                        if self.cancel_orders(exchange , asset1 , asset2):
                            self.log("Canceled limit order for {}/{} after timeout.".format(asset1 , asset2))
                        return False

                    elif result == Model.in_progress:
                        n = 0
                        while result == Model.in_progress:
                            n += 1
                            if n >= 20:
                                self.log("Order cannot be terminated, buying {} with {}.".format(asset1 , asset2))
                                self.buy(exchange , asset1 , asset2 , amount_percentage=1)
                                return False

                            self.log("Order for {}/{} is in progress, waiting...".format(asset1 , asset2))
                            time.sleep(timeout)
                            result = self.is_open_order(exchange , asset1 , asset2)

                        self.log("Limit order executed.")
                        return True

                    else:
                        self.log("Limit order executed.")
                        return True

                else:
                    return False

        except Exception as e:
            self.log("Error while selling: {}".format(str(e)))
            return False

    # Reset the cache
    def reset_cache(self):
        self.cache_prices = []
        self.cache_order_books = []

    # Check if the price is cached and return if it is.
    def get_price_cache(self , exchange , asset1 , asset2):
        for item in self.cache_prices:
            if item['asset1'] == asset1 and item['asset2'] == asset2 and item['exchange'] == str(exchange):
                return item['ticker']
        return None

    # Check if the order book is cached, and return if so.
    def get_order_book_cache(self , exchange , asset1 , asset2):
        for item in self.cache_order_books:
            if item['asset1'] == asset1 and item['asset2'] == asset2 and item['exchange'] == str(exchange):
                return item['book']
        return None

    # Put price in cache.
    def cache_add_price(self , exchange , asset1 , asset2 , ticker):
        self.cache_prices.append({
            'exchange': str(exchange) , 'asset1': asset1 , 'asset2': asset2 , 'ticker': ticker
        })

    # Put the order book in the cache.
    def cache_order_book(self , exchange , asset1 , asset2 , book):
        self.cache_order_books.append({
            'exchange': str(exchange) , 'asset1': asset1 , 'asset2': asset2 , 'book': book
        })

    # Get your balance for given asset.
    def get_balance(self , exchange , asset):
        try:
            balance = exchange.fetchBalance()

            if asset in balance:
                return balance[asset]['free']

            return 0
        except Exception as e:
            self.log("Error while getting balance: {}".format(str(e)))
            raise

    # Get an asset price
    def get_price(self , exchange , asset1 , asset2 , mode='average'):
        try:
            if self.get_price_cache(exchange , asset1 , asset2):
                ticker = self.get_price_cache(exchange , asset1 , asset2)

            else:
                ticker = exchange.fetchTicker('{}/{}'.format(asset1 , asset2))
                self.cache_add_price(exchange , asset1 , asset2 , ticker)

            if mode == 'bid':
                return ticker['bid']

            if mode == 'ask':
                return ticker['ask']

            return (ticker['ask'] + ticker['bid']) / 2
        except Exception as e:
            self.log("Error while fetching price for {}/{}: {}".format(asset1 , asset2 , str(e)))
            return None

    # Get order book for given asset.
    def get_order_book(self , exchange , asset1 , asset2 , mode="bids"):
        try:
            order_book = self.get_order_book_cache(exchange , asset1 , asset2)
            if not order_book:
                order_book = exchange.fetchOrderBook('{}/{}'.format(asset1 , asset2))
                self.cache_order_book(exchange , asset1 , asset2 , order_book)

            return order_book[mode]
        except Exception as e:
            self.log("Error while fetching order book for {}/{}: {}".format(asset1 , asset2 , str(e)))
            return None

    # Get the safest and lowest price to limit buy the given asset.
    def get_buy_limit_price(self , exchange , asset1 , asset2 , amount=1):
        bids = self.get_order_book(exchange , asset1 , asset2 , mode="asks")

        if not bids:
            return None

        bids.sort()

        if len(bids) < settings.ESTIMATION_ORDERBOOK:
            return None

        for bid in bids[settings.ESTIMATION_ORDERBOOK:]:
            if bid[1] >= amount:
                return bid[0]

    # Get the safest and highest price to limit sell the given asset.
    def get_sell_limit_price(self , exchange , asset1 , asset2 , amount=1):
        asks = self.get_order_book(exchange , asset1 , asset2 , mode="bids")

        if not asks:
            return None

        asks.sort(reverse=True)

        if len(asks) < settings.ESTIMATION_ORDERBOOK:
            return None

        for ask in asks[settings.ESTIMATION_ORDERBOOK:]:
            if ask[1] >= amount:
                return ask[0]

    # Check if at least one order is open for the given asset and exchange.
    def is_open_order(self , exchange , asset1 , asset2):
        try:
            data = exchange.fetchOpenOrders('{}/{}'.format(asset1 , asset2))
            for item in data:
                if item['filled'] > 0:
                    return Model.in_progress

            if len(data) > 0:
                return Model.not_filled

            return Model.filled
        except Exception as e:
            self.log("Error while fetching open orders for {}/{}: {}".format(asset1 , asset2 , str(e)))
            return None

    # Cancel all orders for given assets.
    def cancel_orders(self , exchange , asset1 , asset2):
        for _ in range(5):
            try:
                orders = exchange.fetchOpenOrders('{}/{}'.format(asset1 , asset2))
                for order in orders:
                    exchange.cancelOrder(order['id'] , '{}/{}'.format(asset1 , asset2))

                return True
            except Exception as e:
                self.log("Error while canceling orders for {}/{}: {}. Retrying.".format(asset1 , asset2 , str(e)))

        self.log("Cannot cancel orders for {}/{}.".format(asset1 , asset2 , str(e)))
        return

    # Advanced: Calculate optimal position size based on opportunity
    def calculate_optimal_position_size(self, exchange, profit_pct, eth_price_usd):
        """Dynamic position sizing based on opportunity size"""
        if not settings.ENABLE_DYNAMIC_POSITION_SIZING:
            return settings.MAX_POSITION_SIZE
        
        balance_eth = self.get_balance(exchange, 'ETH')
        if balance_eth == 0:
            return 0
        
        # Scale position size based on profit percentage
        # Higher profit = larger position (up to max)
        base_size = settings.MAX_POSITION_SIZE
        if profit_pct > 1.0:  # Very profitable opportunity
            scale_factor = min(1.0, profit_pct / 2.0)  # Scale up to 2x for 2%+ profit
            optimal_size = base_size * (1 + scale_factor * 0.5)  # Up to 1.5x base
            return min(optimal_size, 0.8)  # Cap at 80% for safety
        elif profit_pct > 0.5:  # Good opportunity
            return base_size
        else:  # Smaller opportunity
            return base_size * 0.7  # Use smaller position
        
    # Advanced: Optimize for slippage using order book depth
    def estimate_slippage(self, exchange, asset1, asset2, amount):
        """Estimate slippage based on order book depth"""
        if not settings.ENABLE_SLIPPAGE_OPTIMIZATION:
            return 0
        
        try:
            order_book = self.get_order_book(exchange, asset1, asset2, mode="asks")
            if not order_book:
                return 0.001  # Default 0.1% slippage
            
            total_volume = 0
            weighted_price = 0
            for price, volume in order_book[:10]:  # Check top 10 levels
                total_volume += volume
                weighted_price += price * volume
                if total_volume >= amount:
                    break
            
            if total_volume > 0:
                avg_price = weighted_price / total_volume
                best_price = order_book[0][0] if order_book else avg_price
                slippage = abs(avg_price - best_price) / best_price
                return slippage
            return 0.001
        except:
            return 0.001
    
    # Estimate the profit for forward arbitrage on given asset (improved with maker fees).
    def estimate_arbitrage_forward(self , exchange , asset):
        try:
            alt_eth = self.get_buy_limit_price(exchange , asset , 'ETH')
            alt_btc = self.get_sell_limit_price(exchange , asset , 'BTC')

            if not alt_btc or not alt_eth:
                if settings.AGGRESSIVE_MODE:
                    # In aggressive mode, try with market prices as fallback
                    alt_eth = self.get_price(exchange, asset, 'ETH', mode='ask')
                    alt_btc = self.get_price(exchange, asset, 'BTC', mode='bid')
                    if not alt_btc or not alt_eth:
                        return -100
                else:
                    self.log("Less than {} orders for {}, skipping.".format(settings.MIN_ORDERBOOK_DEPTH, asset))
                    return -100

            # Use maker fees for limit orders (better profitability)
            # Optimize fee if BNB is available (potential 25% discount)
            maker_fee_optimized = self.maker_fee
            try:
                bnb_balance = self.get_balance(exchange, 'BNB')
                if bnb_balance > 0.1:  # Has BNB for fee payment
                    maker_fee_optimized = 0.99975  # 0.025% fee with BNB discount
            except:
                pass
            
            step_1 = (1 / alt_eth) * maker_fee_optimized
            step_2 = (step_1 * alt_btc) * maker_fee_optimized
            eth_btc_price = self.get_price(exchange , 'ETH' , 'BTC' , mode='ask')
            step_3 = (step_2 / eth_btc_price) * maker_fee_optimized

            profit_pct = (step_3 - 1) * 100
            
            # Calculate estimated profit in USD with dynamic position sizing
            eth_price_usd = self.get_price(exchange, 'ETH', 'USDT', mode='average')
            if eth_price_usd:
                balance_eth = self.get_balance(exchange, 'ETH')
                if balance_eth == 0:
                    return -100
                
                # Use dynamic position sizing
                position_size_pct = self.calculate_optimal_position_size(exchange, profit_pct, eth_price_usd)
                position_size = balance_eth * position_size_pct
                
                # Account for slippage
                slippage_penalty = 0
                if settings.ENABLE_SLIPPAGE_OPTIMIZATION:
                    slippage_penalty = self.estimate_slippage(exchange, asset, 'ETH', position_size)
                    slippage_penalty += self.estimate_slippage(exchange, asset, 'BTC', position_size * alt_btc)
                
                profit_usd = (profit_pct / 100 - slippage_penalty) * position_size * eth_price_usd
                
                if profit_usd < settings.MIN_PROFIT_USD:
                    return -100  # Not profitable enough
                
                # In aggressive mode, accept slightly lower profits
                if settings.AGGRESSIVE_MODE and profit_usd >= settings.MIN_PROFIT_USD * 0.8:
                    return profit_pct
            
            return profit_pct
        except ZeroDivisionError:
            return -1
        except Exception as e:
            self.log("Error estimating forward arbitrage: {}".format(str(e)))
            return -1

    # Estimate the profit for backward arbitrage on given asset (improved with maker fees).
    def estimate_arbitrage_backward(self , exchange , asset):
        try:
            alt_btc = self.get_buy_limit_price(exchange , asset , 'BTC')
            alt_eth = self.get_sell_limit_price(exchange , asset , 'ETH')

            if not alt_btc or not alt_eth:
                if settings.AGGRESSIVE_MODE:
                    # In aggressive mode, try with market prices as fallback
                    alt_btc = self.get_price(exchange, asset, 'BTC', mode='ask')
                    alt_eth = self.get_price(exchange, asset, 'ETH', mode='bid')
                    if not alt_btc or not alt_eth:
                        return -100
                else:
                    self.log("Less than {} orders for {} on {}, skipping.".format(settings.MIN_ORDERBOOK_DEPTH, asset, str(exchange)))
                    return -100

            # Use maker fees for limit orders (better profitability)
            # Optimize fee if BNB is available (potential 25% discount)
            maker_fee_optimized = self.maker_fee
            try:
                bnb_balance = self.get_balance(exchange, 'BNB')
                if bnb_balance > 0.1:  # Has BNB for fee payment
                    maker_fee_optimized = 0.99975  # 0.025% fee with BNB discount
            except:
                pass
            
            eth_btc_price = self.get_price(exchange , 'ETH' , 'BTC' , mode='bid')
            step_1 = eth_btc_price * maker_fee_optimized
            step_2 = (step_1 / alt_btc) * maker_fee_optimized
            step_3 = (step_2 * alt_eth) * maker_fee_optimized

            profit_pct = (step_3 - 1) * 100
            
            # Calculate estimated profit in USD with dynamic position sizing
            eth_price_usd = self.get_price(exchange, 'ETH', 'USDT', mode='average')
            if eth_price_usd:
                balance_eth = self.get_balance(exchange, 'ETH')
                if balance_eth == 0:
                    return -100
                
                # Use dynamic position sizing
                position_size_pct = self.calculate_optimal_position_size(exchange, profit_pct, eth_price_usd)
                position_size = balance_eth * position_size_pct
                
                # Account for slippage
                slippage_penalty = 0
                if settings.ENABLE_SLIPPAGE_OPTIMIZATION:
                    slippage_penalty = self.estimate_slippage(exchange, 'ETH', 'BTC', position_size)
                    slippage_penalty += self.estimate_slippage(exchange, asset, 'BTC', position_size * eth_btc_price / alt_btc)
                
                profit_usd = (profit_pct / 100 - slippage_penalty) * position_size * eth_price_usd
                
                if profit_usd < settings.MIN_PROFIT_USD:
                    return -100  # Not profitable enough
                
                # In aggressive mode, accept slightly lower profits
                if settings.AGGRESSIVE_MODE and profit_usd >= settings.MIN_PROFIT_USD * 0.8:
                    return profit_pct
            
            return profit_pct
        except ZeroDivisionError:
            return -1
        except Exception as e:
            self.log("Error estimating backward arbitrage: {}".format(str(e)))
            return -1

    # Executes forward arbitrage on given asset: ETH -> ALT -> BTC -> ETH.
    def run_arbitrage_forward(self , exchange , asset):
        self.log("Arbitrage on {}: ETH -> {} -> BTC -> ETH".format(exchange , asset))
        balance_before = self.get_balance(exchange , "ETH")
        
        # Calculate optimal position size based on opportunity
        profit_estimate = self.estimate_arbitrage_forward(exchange, asset)
        eth_price_usd = self.get_price(exchange, 'ETH', 'USDT', mode='average') or 2000
        position_size_pct = self.calculate_optimal_position_size(exchange, profit_estimate, eth_price_usd)
        position_size = min(position_size_pct, 1.0)
        
        result1 = self.best_buy(exchange , asset , 'ETH' , position_size)

        if not result1:
            self.log("Failed to convert {} to ETH, canceling arbitrage.".format(asset))
            return

        result2 = self.best_sell(exchange , asset , 'BTC' , position_size)

        if not result2:
            self.log(
                "Failed to convert {} to BTC, canceling arbitrage. Will convert back {} to ETH.".format(asset , asset))
            self.sell(exchange , asset , 'ETH' , amount_percentage=1)
            self.summarize_arbitrage(exchange , balance_before , asset)
            return

        self.buy(exchange , "ETH" , "BTC" , amount_percentage=1)
        self.summarize_arbitrage(exchange , balance_before , asset)

    # Executes backward arbitrage on given asset: ETH -> BTC -> ALT -> ETH.
    def run_arbitrage_backward(self , exchange , asset):
        self.log("Arbitrage on {}: ETH -> BTC -> {} -> ETH".format(exchange , asset))
        balance_before = self.get_balance(exchange , "ETH")
        
        # Calculate optimal position size based on opportunity
        profit_estimate = self.estimate_arbitrage_backward(exchange, asset)
        eth_price_usd = self.get_price(exchange, 'ETH', 'USDT', mode='average') or 2000
        position_size_pct = self.calculate_optimal_position_size(exchange, profit_estimate, eth_price_usd)
        position_size = min(position_size_pct, 1.0)
        
        self.sell(exchange , "ETH" , "BTC" , position_size)
        result1 = self.best_buy(exchange , asset , 'BTC' , position_size)

        if not result1:
            self.log("Failed to convert BTC to {}, canceling arbitrage. Will convert BTC to ETH.".format(asset))
            self.buy(exchange , 'ETH' , 'BTC' , amount_percentage=1)
            self.summarize_arbitrage(exchange , balance_before , asset)
            return

        result2 = self.best_sell(exchange , asset , 'ETH' , position_size)

        if not result2:
            self.log(
                "Failed to convert {} to ETH, canceling arbitrage. Forcing conversion from {} to ETH.".format(asset ,
                                                                                                              asset))
            self.sell(exchange , asset , 'ETH' , amount_percentage=1)
            self.summarize_arbitrage(exchange , balance_before , asset)
            return

        self.summarize_arbitrage(exchange , balance_before , asset)

    # Logs given string.
    @staticmethod
    def log(text):
        formatted_text = "[{}] {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S") , text)

        with open('logs.txt' , 'a+') as file:
            file.write(formatted_text)
            file.write("\n")

    # Buy at best price possible using decreasing buy limit orders.
    def best_buy(self , exchange , asset1 , asset2 , amount_percentage):
        order_book = self.get_order_book(exchange , asset1 , asset2 , mode="asks")
        order_book.sort(key=itemgetter(0))

        for price in order_book[:settings.MAX_TRIES_ORDERBOOK]:
            self.log("Trying to buy {} with {} @{:.8f}.".format(asset1 , asset2 , price[0]))
            result = self.buy(exchange , asset1 , asset2 , amount_percentage=amount_percentage , limit=price[0] ,
                              timeout=settings.WAIT_BETWEEN_ORDER)

            if result:
                self.log("Bought {} with {} @{:.8f}.".format(asset1 , asset2 , price[0]))
                return True

            else:
                self.log("Failed to buy {} with {} at {:.8f}.".format(asset1 , asset2 , price[0]))

        self.log("Was not able to buy {} with {}".format(asset1 , asset2))
        return False

    # Sell at best price possible using decreasing buy sell orders.
    def best_sell(self , exchange , asset1 , asset2 , amount_percentage):
        order_book = self.get_order_book(exchange , asset1 , asset2 , mode="bids")
        order_book.sort(key=itemgetter(0) , reverse=True)

        for price in order_book[:settings.MAX_TRIES_ORDERBOOK]:
            self.log("Trying to sell {} to {} @{:.8f}.".format(asset1 , asset2 , price[0]))
            result = self.sell(exchange , asset1 , asset2 , amount_percentage=amount_percentage , limit=price[0] ,
                               timeout=settings.WAIT_BETWEEN_ORDER)

            if result:
                self.log("Sold {} to {} @{:.8f}.".format(asset1 , asset2 , price[0]))
                return True

            else:
                self.log("Failed to sell {} to {} at {:.8f}.".format(asset1 , asset2 , price[0]))

        self.log("Was not able to sell {} to {}".format(asset1 , asset2))
        return False

    # Summarize arbitrage, calculate loss/gain, print it, and save it on the file.
    def summarize_arbitrage(self , exchange , balance_before , asset):
        balance_after = self.get_balance(exchange , "ETH")
        diff = balance_after - balance_before
        
        # Get USD price for better tracking
        try:
            eth_price_usd = self.get_price(exchange, 'ETH', 'USDT', mode='average')
            diff_usd = diff * eth_price_usd if eth_price_usd else 0
        except:
            diff_usd = 0
        
        # Track profits
        self.trades_executed += 1
        if diff > 0:
            self.successful_trades += 1
            self.total_profit_eth += diff
            self.total_profit_usd += diff_usd
        
        success = diff > 0
        self.log("Arbitrage {:5} on binance, diff: {:8.6f}ETH (${:.2f} USD). Total profit: {:8.6f}ETH (${:.2f} USD) | Success rate: {:.1f}%".format(
            asset, diff, diff_usd, self.total_profit_eth, self.total_profit_usd, 
            (self.successful_trades / self.trades_executed * 100) if self.trades_executed > 0 else 0
        ))
        self.save_profit_tracking()
        
        # Send Telegram notification
        if self.telegram:
            self.telegram.notify_trade_executed('Binance', asset, diff, diff_usd, success)
    
    # Load profit tracking from file
    def load_profit_tracking(self):
        try:
            import json
            import os
            if os.path.exists('profit_tracking.json'):
                with open('profit_tracking.json', 'r') as f:
                    data = json.load(f)
                    self.total_profit_eth = data.get('total_profit_eth', 0.0)
                    self.total_profit_usd = data.get('total_profit_usd', 0.0)
                    self.trades_executed = data.get('trades_executed', 0)
                    self.successful_trades = data.get('successful_trades', 0)
        except Exception as e:
            self.log("Error loading profit tracking: {}".format(str(e)))
    
    # Save profit tracking to file
    def save_profit_tracking(self):
        try:
            import json
            data = {
                'total_profit_eth': self.total_profit_eth,
                'total_profit_usd': self.total_profit_usd,
                'trades_executed': self.trades_executed,
                'successful_trades': self.successful_trades,
                'last_updated': datetime.now().isoformat()
            }
            with open('profit_tracking.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log("Error saving profit tracking: {}".format(str(e)))
    
    # Get profit statistics
    def get_profit_stats(self):
        success_rate = (self.successful_trades / self.trades_executed * 100) if self.trades_executed > 0 else 0
        return {
            'total_profit_eth': self.total_profit_eth,
            'total_profit_usd': self.total_profit_usd,
            'trades_executed': self.trades_executed,
            'successful_trades': self.successful_trades,
            'success_rate': success_rate
        }
