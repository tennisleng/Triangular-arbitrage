from src.model import Model
from data import tokens , settings
import threading
import time


def checker(model , exchange , alt):
    change_f = model.estimate_arbitrage_forward(exchange , alt)
    change_b = model.estimate_arbitrage_backward(exchange , alt)

    model.log("Binance | {:5}: {:8.5f}% / {:8.5f}%".format(alt , change_f , change_b))

    # In aggressive mode, use lower threshold for opportunities
    min_diff = settings.MIN_DIFFERENCE
    if settings.AGGRESSIVE_MODE:
        min_diff = max(0.1, settings.MIN_DIFFERENCE * 0.8)  # 20% lower threshold
    
    # Execute the better opportunity
    if change_f > change_b and change_f > min_diff:
        model.log("Got opportunity for {:5} @{:.4f}% on Binance (Forward)".format(alt , change_f))
        # Notify about opportunity
        if model.telegram:
            model.telegram.notify_opportunity('Binance', alt, change_f, 'Forward')
        model.run_arbitrage_forward(exchange , alt)

    elif change_b > min_diff:
        model.log("Got opportunity for {:5} @{:.4f}% on Binance (Backward)".format(alt , change_b))
        # Notify about opportunity
        if model.telegram:
            model.telegram.notify_opportunity('Binance', alt, change_b, 'Backward')
        model.run_arbitrage_backward(exchange , alt)


def run(model , exchange , thread_number):
    alts = tokens.binance_tokens
    last_summary_time = time.time()
    
    while True:
        for i in range(0 , len(alts) , thread_number):
            alts_batch = alts[i:i + thread_number]
            threads = []
            for asset in alts_batch:
                threads.append(threading.Thread(target=checker , args=(model , exchange , asset)))
                threads[-1].start()
            for thread in threads:
                thread.join()
            model.reset_cache()
        
        # Send daily summary every 24 hours
        if time.time() - last_summary_time > 86400:  # 24 hours
            if model.telegram:
                stats = model.get_profit_stats()
                model.telegram.notify_daily_summary(stats)
            last_summary_time = time.time()


if __name__ == "__main__":
    print("=" * 60)
    print("Triangular Arbitrage Bot - Monetized Edition")
    print("=" * 60)
    
    model = Model()
    exchange = model.binance
    
    # Display subscription info
    if model.subscription:
        print(f"Subscription Tier: {model.subscription.get_tier()}")
        print(f"Premium Features: {'Enabled' if settings.PREMIUM_FEATURES_ENABLED else 'Disabled'}")
    
    # Display profit stats
    stats = model.get_profit_stats()
    print(f"\nCurrent Statistics:")
    print(f"  Total Profit: {stats['total_profit_eth']:.6f} ETH (${stats['total_profit_usd']:.2f})")
    print(f"  Trades Executed: {stats['trades_executed']}")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    
    # Start API server in background if premium
    api_thread = None
    if model.subscription and model.subscription.has_feature('api_access'):
        try:
            from src.api_server import start_api_server
            import threading as th
            api_thread = th.Thread(target=start_api_server, args=(5000, model), daemon=True)
            api_thread.start()
            print(f"\nAPI Server started on port 5000")
        except Exception as e:
            print(f"Could not start API server: {e}")
    
    print(f"\nStarting to listen to Binance markets...")
    print("Advanced Algorithms Enabled:")
    print(f"  - Dynamic Position Sizing: {settings.ENABLE_DYNAMIC_POSITION_SIZING}")
    print(f"  - Slippage Optimization: {settings.ENABLE_SLIPPAGE_OPTIMIZATION}")
    print(f"  - Aggressive Mode: {settings.AGGRESSIVE_MODE}")
    print(f"  - Minimum Profit Threshold: ${settings.MIN_PROFIT_USD}")
    print("=" * 60)
    
    model.log("Starting to listen the binance markets with advanced algorithms")
    # Increase thread number for faster scanning in aggressive mode
    thread_number = 8 if settings.AGGRESSIVE_MODE else 5
    run(model , exchange , thread_number)
