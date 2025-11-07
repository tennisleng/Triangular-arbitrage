#!/usr/bin/env python3
"""
Profit checker - monitors profit tracking and alerts when $1 profit is reached
"""
import json
import os
import time
from datetime import datetime

def check_profit():
    """Check current profit and return status"""
    profit_file = 'profit_tracking.json'
    
    if not os.path.exists(profit_file):
        return {
            'status': 'no_data',
            'profit_usd': 0.0,
            'profit_eth': 0.0,
            'trades': 0,
            'success_rate': 0.0
        }
    
    try:
        with open(profit_file, 'r') as f:
            data = json.load(f)
        
        profit_usd = data.get('total_profit_usd', 0.0)
        profit_eth = data.get('total_profit_eth', 0.0)
        trades = data.get('trades_executed', 0)
        successful = data.get('successful_trades', 0)
        success_rate = (successful / trades * 100) if trades > 0 else 0.0
        
        return {
            'status': 'success' if profit_usd >= 1.0 else 'in_progress',
            'profit_usd': profit_usd,
            'profit_eth': profit_eth,
            'trades': trades,
            'successful_trades': successful,
            'success_rate': success_rate,
            'last_updated': data.get('last_updated', 'Unknown')
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def main():
    """Main monitoring loop"""
    print("=" * 60)
    print("Profit Monitor - Advanced Arbitrage Bot")
    print("=" * 60)
    print(f"Target: $1.00 USD profit")
    print("=" * 60)
    
    last_profit = 0.0
    
    while True:
        stats = check_profit()
        
        if stats['status'] == 'success':
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! Profit goal reached!")
            print("=" * 60)
            print(f"Total Profit: ${stats['profit_usd']:.2f} USD")
            print(f"Total Profit: {stats['profit_eth']:.6f} ETH")
            print(f"Trades Executed: {stats['trades']}")
            print(f"Successful Trades: {stats['successful_trades']}")
            print(f"Success Rate: {stats['success_rate']:.1f}%")
            print(f"Last Updated: {stats['last_updated']}")
            print("=" * 60)
            break
        
        elif stats['status'] == 'in_progress':
            current_profit = stats['profit_usd']
            if current_profit != last_profit:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                print(f"  Current Profit: ${current_profit:.2f} USD ({stats['profit_eth']:.6f} ETH)")
                print(f"  Progress: {(current_profit / 1.0 * 100):.1f}% of $1.00 goal")
                print(f"  Trades: {stats['trades']} | Success Rate: {stats['success_rate']:.1f}%")
                last_profit = current_profit
        
        elif stats['status'] == 'no_data':
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for trading data...")
        
        time.sleep(5)  # Check every 5 seconds

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        stats = check_profit()
        if stats['status'] != 'error':
            print(f"\nFinal Status:")
            print(f"  Profit: ${stats['profit_usd']:.2f} USD")
            print(f"  Trades: {stats['trades']}")
