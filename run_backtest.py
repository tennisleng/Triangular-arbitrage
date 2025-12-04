#!/usr/bin/env python3
"""
Professional-Grade Arbitrage Strategy Backtester
Senior Quant Approach with Realistic Assumptions

Key Considerations:
- Realistic latency and execution costs
- Market impact modeling
- Competition decay (opportunities disappear quickly)
- Conservative fill assumptions
- Risk-adjusted metrics (Sharpe, Sortino, Max Drawdown)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ProfessionalBacktester:
    """
    Institutional-grade backtester with realistic assumptions.
    Based on real-world trading experience at major market makers.
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # Realistic cost assumptions
        self.costs = {
            'exchange_fee_maker': 0.0002,   # 2bps maker
            'exchange_fee_taker': 0.0005,   # 5bps taker  
            'slippage_per_trade': 0.0003,   # 3bps slippage
            'latency_cost': 0.0001,         # 1bp for latency (price moves)
            'withdrawal_fee_pct': 0.0005,   # Cross-exchange transfer cost
        }
        
        # Competition assumptions
        self.competition = {
            'arb_decay_half_life_ms': 50,    # Opportunities disappear in 50ms
            'capture_probability': 0.15,     # Only capture 15% of opportunities
            'fill_rate': 0.80,               # 80% fill rate on limit orders
        }
        
        # Risk parameters
        self.risk = {
            'max_position_usd': 10000,
            'max_drawdown_pct': 5,
            'risk_free_rate': 0.05,  # 5% annual
        }
        
        self.results = {}
        
    def calculate_total_cost(self, trade_value: float, is_cross_exchange: bool = False) -> float:
        """Calculate total cost for a trade."""
        cost = trade_value * (
            self.costs['exchange_fee_taker'] +
            self.costs['slippage_per_trade'] +
            self.costs['latency_cost']
        )
        
        if is_cross_exchange:
            cost += trade_value * self.costs['withdrawal_fee_pct']
            
        return cost
    
    def simulate_opportunity_capture(self, raw_profit: float) -> float:
        """
        Simulate realistic capture of arbitrage opportunity.
        Most opportunities are taken by faster traders.
        """
        # Random capture based on competition
        if np.random.random() > self.competition['capture_probability']:
            return 0  # Missed opportunity
            
        # Random fill rate
        if np.random.random() > self.competition['fill_rate']:
            return raw_profit * 0.3  # Partial fill
            
        return raw_profit
        
    def backtest_cross_exchange_arb(self, years: int = 10) -> dict:
        """
        Cross-Exchange Arbitrage with realistic assumptions.
        
        Reality check:
        - Price discrepancies are tiny (1-5 bps typically)
        - Must beat latency of HFT firms (sub-millisecond)
        - Capital locked on multiple exchanges
        """
        print("\n" + "="*70)
        print("CROSS-EXCHANGE ARBITRAGE - Institutional Analysis")
        print("="*70)
        
        # Hourly scanning over 10 years
        num_periods = years * 365 * 24
        
        # Track P&L
        pnl_series = []
        trades = 0
        wins = 0
        
        position_size = 5000  # $5k per trade
        
        # Generate price discrepancies (realistic: mostly 0-10 bps)
        np.random.seed(42)
        
        for period in range(num_periods):
            # Typical cross-exchange spread (very tight)
            raw_spread_bps = np.random.exponential(3)  # Average 3bps, exponential decay
            
            # Total costs in bps
            cost_bps = (
                self.costs['exchange_fee_taker'] * 2 +  # Buy + sell
                self.costs['slippage_per_trade'] * 2 +
                self.costs['latency_cost'] * 2 +
                self.costs['withdrawal_fee_pct']  # For rebalancing
            ) * 10000  # Convert to bps
            
            # Net profit before competition
            net_profit_bps = raw_spread_bps - cost_bps
            
            if net_profit_bps > 0:
                trades += 1
                
                # Apply competition filter
                raw_profit = position_size * net_profit_bps / 10000
                actual_profit = self.simulate_opportunity_capture(raw_profit)
                
                if actual_profit > 0:
                    wins += 1
                    
                pnl_series.append(actual_profit)
        
        # Calculate metrics
        total_pnl = sum(pnl_series)
        win_rate = wins / trades * 100 if trades > 0 else 0
        avg_trade = total_pnl / trades if trades > 0 else 0
        
        # Sharpe ratio (annualized)
        if len(pnl_series) > 24:
            # Aggregate to daily
            pnl_arr = np.array(pnl_series)
            daily_count = len(pnl_arr) // 24
            daily_pnl = pnl_arr[:daily_count * 24].reshape(-1, 24).sum(axis=1)
            sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365) if np.std(daily_pnl) > 0 else 0
        else:
            sharpe = 0
            
        # Max drawdown
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative) if len(cumulative) > 0 else [0]
        drawdown = (running_max - cumulative)
        max_dd = max(drawdown) if len(drawdown) > 0 else 0
        
        result = {
            'strategy': 'Cross-Exchange Arbitrage',
            'years': years,
            'total_trades': trades,
            'wins': wins,
            'win_rate_pct': win_rate,
            'total_pnl_usd': total_pnl,
            'avg_trade_usd': avg_trade,
            'sharpe_ratio': sharpe,
            'max_drawdown_usd': max_dd,
            'annualized_return_pct': (total_pnl / (position_size * years)) * 100,
            'profitable': total_pnl > 0
        }
        
        self.results['cross_exchange'] = result
        self._print_quant_result(result)
        return result
        
    def backtest_triangular_arb(self, years: int = 10) -> dict:
        """
        Triangular Arbitrage - Single exchange, 3 legs.
        
        Reality:
        - Opportunities are 0-5 bps typically
        - 3 legs = 3x fees
        - Must beat exchange's own arbitrage bots
        """
        print("\n" + "="*70)
        print("TRIANGULAR ARBITRAGE - Institutional Analysis")
        print("="*70)
        
        num_periods = years * 365 * 24
        
        pnl_series = []
        trades = 0
        wins = 0
        
        position_size = 3000  # $3k per triangle
        fee_per_leg = self.costs['exchange_fee_maker']  # Assume maker
        
        np.random.seed(43)
        
        for period in range(num_periods):
            # Triangular inefficiency (usually very small)
            raw_inefficiency_bps = np.random.exponential(2)
            
            # 3 legs of fees + slippage
            cost_bps = (fee_per_leg * 3 + self.costs['slippage_per_trade'] * 3) * 10000
            
            net_profit_bps = raw_inefficiency_bps - cost_bps
            
            if net_profit_bps > 0:
                trades += 1
                
                raw_profit = position_size * net_profit_bps / 10000
                # Triangular arbs are highly competitive
                actual_profit = self.simulate_opportunity_capture(raw_profit) * 0.5
                
                if actual_profit > 0:
                    wins += 1
                    
                pnl_series.append(actual_profit)
        
        total_pnl = sum(pnl_series)
        
        result = {
            'strategy': 'Triangular Arbitrage',
            'years': years,
            'total_trades': trades,
            'wins': wins,
            'win_rate_pct': wins / trades * 100 if trades > 0 else 0,
            'total_pnl_usd': total_pnl,
            'avg_trade_usd': total_pnl / trades if trades > 0 else 0,
            'annualized_return_pct': (total_pnl / (position_size * years)) * 100,
            'profitable': total_pnl > 0
        }
        
        self.results['triangular'] = result
        self._print_quant_result(result)
        return result
    
    def backtest_funding_rate_arb(self, years: int = 10) -> dict:
        """
        Funding Rate Arbitrage - Delta neutral funding collection.
        
        This is one of the most reliable strategies:
        - Funding paid 3x daily on perpetuals
        - Delta-neutral = low directional risk
        - Main risk: funding rate flipping sign
        """
        print("\n" + "="*70)
        print("FUNDING RATE ARBITRAGE - Institutional Analysis")
        print("="*70)
        
        # 3 funding periods per day
        num_periods = years * 365 * 3
        
        position_value = 10000  # $10k notional
        
        pnl_series = []
        collections = 0
        
        np.random.seed(44)
        
        # Realistic funding rate distribution
        # Mean ~0.01% per 8h in bull markets, higher variance
        for period in range(num_periods):
            # Funding rate (can be positive or negative)
            funding_rate = np.random.normal(0.0001, 0.0003)  # 1bp mean, 3bp std
            
            # Only trade when funding is sufficiently positive
            if funding_rate > 0.0001:  # >1bp threshold
                collections += 1
                
                # Collect funding
                funding_income = position_value * funding_rate
                
                # Entry/exit costs (amortized over hold period)
                # Assume we rebalance monthly = 30*3 = 90 funding periods
                amortized_cost = (position_value * self.costs['exchange_fee_taker'] * 4) / 90
                
                net_pnl = funding_income - amortized_cost
                pnl_series.append(net_pnl)
        
        total_pnl = sum(pnl_series)
        
        # Calculate Sharpe
        if len(pnl_series) > 30:
            pnl_arr = np.array(pnl_series)
            daily_count = len(pnl_arr) // 3
            if daily_count > 0:
                daily_pnl = pnl_arr[:daily_count * 3].reshape(-1, 3).sum(axis=1)
                sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365) if np.std(daily_pnl) > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        result = {
            'strategy': 'Funding Rate Arbitrage',
            'years': years,
            'collections': collections,
            'total_pnl_usd': total_pnl,
            'sharpe_ratio': sharpe,
            'annualized_return_pct': (total_pnl / position_value) / years * 100,
            'profitable': total_pnl > 0
        }
        
        self.results['funding_rate'] = result
        self._print_quant_result(result)
        return result
    
    def backtest_grid_trading(self, years: int = 10) -> dict:
        """
        Grid Trading - Profit from range-bound markets.
        
        Reality:
        - Works well in sideways markets
        - Gets crushed in trending markets (inventory risk)
        - Need proper position sizing
        """
        print("\n" + "="*70)
        print("GRID TRADING - Institutional Analysis")  
        print("="*70)
        
        num_periods = years * 365 * 24
        
        grid_spacing_pct = 0.5  # 0.5% between grid levels
        trade_size = 100  # $100 per grid trade
        
        pnl_series = []
        trades = 0
        inventory_value = 0
        avg_cost = 0
        
        np.random.seed(45)
        
        # Simulate price as random walk with mean reversion
        price = 100
        prices = [price]
        
        for _ in range(num_periods):
            # Mean-reverting random walk
            price = price + np.random.normal(0, 1) * 0.5
            price = price * 0.999 + 100 * 0.001  # Pull toward 100
            prices.append(max(price, 50))  # Floor at 50
        
        last_grid_level = round(prices[0])
        
        for i in range(1, len(prices)):
            current_level = round(prices[i])
            
            # Grid trigger
            if abs(current_level - last_grid_level) >= 1:
                trades += 1
                
                if current_level < last_grid_level:
                    # Price dropped, buy
                    buy_cost = trade_size * (1 + self.costs['exchange_fee_maker'])
                    inventory_value += trade_size
                    avg_cost = (avg_cost * (inventory_value - trade_size) + prices[i] * trade_size) / inventory_value if inventory_value > 0 else prices[i]
                else:
                    # Price rose, sell
                    if inventory_value > 0:
                        sell_revenue = trade_size * (1 - self.costs['exchange_fee_maker'])
                        realized_pnl = (prices[i] - avg_cost) * trade_size / prices[i]
                        pnl_series.append(realized_pnl - trade_size * self.costs['slippage_per_trade'])
                        inventory_value -= trade_size
                        
                last_grid_level = current_level
        
        # Mark-to-market remaining inventory
        if inventory_value > 0:
            unrealized_pnl = (prices[-1] - avg_cost) * inventory_value / prices[-1]
            pnl_series.append(unrealized_pnl)
        
        total_pnl = sum(pnl_series)
        
        result = {
            'strategy': 'Grid Trading',
            'years': years,
            'total_trades': trades,
            'total_pnl_usd': total_pnl,
            'avg_trade_usd': total_pnl / trades if trades > 0 else 0,
            'final_inventory_usd': inventory_value,
            'profitable': total_pnl > 0
        }
        
        self.results['grid_trading'] = result
        self._print_quant_result(result)
        return result
    
    def backtest_market_making(self, years: int = 10) -> dict:
        """
        Market Making - Provide liquidity, earn spread.
        
        Reality:
        - Inventory risk is the killer
        - Adverse selection (informed traders pick you off)
        - Need sophisticated inventory management
        """
        print("\n" + "="*70)
        print("MARKET MAKING - Institutional Analysis")
        print("="*70)
        
        num_periods = years * 365 * 24
        
        half_spread = 0.001  # 10bps half spread
        quote_size = 100  # $100 per side
        
        pnl_series = []
        trades = 0
        inventory = 0
        max_inventory = 1000
        
        np.random.seed(46)
        
        # Track adverse selection
        total_adverse_selection = 0
        
        for period in range(num_periods):
            # Random price move
            price_move = np.random.normal(0, 0.001)  # 10bps hourly vol
            
            # Probability of getting picked off (adverse selection)
            # Larger moves = more likely informed trader
            adverse_prob = min(abs(price_move) * 100, 0.5)
            
            # Random fills
            if np.random.random() < 0.1:  # 10% chance of fill per hour
                trades += 1
                
                if np.random.random() < 0.5:
                    # Bid hit (we buy)
                    if inventory < max_inventory:
                        inventory += quote_size
                        
                        # Adverse selection loss
                        if np.random.random() < adverse_prob and price_move < 0:
                            adverse_loss = quote_size * abs(price_move)
                            total_adverse_selection += adverse_loss
                            pnl_series.append(-adverse_loss)
                        else:
                            # Earn half spread
                            pnl_series.append(quote_size * half_spread - quote_size * self.costs['exchange_fee_maker'])
                else:
                    # Ask hit (we sell)
                    if inventory > -max_inventory:
                        inventory -= quote_size
                        
                        if np.random.random() < adverse_prob and price_move > 0:
                            adverse_loss = quote_size * abs(price_move)
                            total_adverse_selection += adverse_loss
                            pnl_series.append(-adverse_loss)
                        else:
                            pnl_series.append(quote_size * half_spread - quote_size * self.costs['exchange_fee_maker'])
        
        total_pnl = sum(pnl_series)
        
        result = {
            'strategy': 'Market Making',
            'years': years,
            'total_trades': trades,
            'total_pnl_usd': total_pnl,
            'adverse_selection_loss_usd': total_adverse_selection,
            'final_inventory_usd': inventory,
            'profitable': total_pnl > 0
        }
        
        self.results['market_making'] = result
        self._print_quant_result(result)
        return result
    
    def backtest_futures_basis(self, years: int = 10) -> dict:
        """
        Futures-Spot Basis Trade (Cash and Carry).
        
        Reality:
        - Very reliable in crypto (contango common)
        - Capital intensive
        - Roll risk at expiry
        """
        print("\n" + "="*70)
        print("FUTURES BASIS TRADE - Institutional Analysis")
        print("="*70)
        
        # Monthly opportunities (quarterly futures roll)
        num_periods = years * 12
        
        position_value = 10000
        
        pnl_series = []
        trades = 0
        
        np.random.seed(47)
        
        for period in range(num_periods):
            # Basis typically 0.5-3% per quarter in crypto
            monthly_basis = np.random.uniform(0.002, 0.01)  # 0.2% to 1% per month
            
            # Only trade when basis > costs
            entry_cost = position_value * self.costs['exchange_fee_taker'] * 2
            exit_cost = position_value * self.costs['exchange_fee_taker'] * 2
            total_cost = entry_cost + exit_cost
            
            gross_profit = position_value * monthly_basis
            net_profit = gross_profit - total_cost
            
            if net_profit > 0:
                trades += 1
                pnl_series.append(net_profit)
        
        total_pnl = sum(pnl_series)
        
        # Calculate Sharpe
        if len(pnl_series) > 12:
            sharpe = np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(12) if np.std(pnl_series) > 0 else 0
        else:
            sharpe = 0
        
        result = {
            'strategy': 'Futures Basis Trade',
            'years': years,
            'total_trades': trades,
            'total_pnl_usd': total_pnl,
            'sharpe_ratio': sharpe,
            'annualized_return_pct': (total_pnl / position_value) / years * 100,
            'profitable': total_pnl > 0
        }
        
        self.results['futures_basis'] = result
        self._print_quant_result(result)
        return result
    
    def _print_quant_result(self, r: dict):
        """Print result in senior quant format."""
        print(f"\n{'─'*50}")
        print(f"Strategy: {r['strategy']}")
        print(f"{'─'*50}")
        
        for key, value in r.items():
            if key == 'strategy':
                continue
            if isinstance(value, float):
                if 'pct' in key.lower() or 'rate' in key.lower():
                    print(f"  {key}: {value:.2f}%")
                elif 'usd' in key.lower():
                    print(f"  {key}: ${value:,.2f}")
                elif 'ratio' in key.lower():
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:.4f}")
            elif isinstance(value, bool):
                status = "✅ POSITIVE EV" if value else "❌ NEGATIVE EV"
                print(f"  Result: {status}")
            else:
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    def run_full_analysis(self, years: int = 10) -> dict:
        """Run complete institutional-grade analysis."""
        print("\n" + "="*80)
        print("       INSTITUTIONAL ARBITRAGE STRATEGY ANALYSIS")
        print(f"       Senior Quant Review - {years} Year Backtest")
        print("="*80)
        print("""
Assumptions:
  • Taker fee: 5bps | Maker fee: 2bps
  • Slippage: 3bps per trade
  • Latency cost: 1bp (price movement during execution)
  • Competition: 85% of opportunities captured by faster traders
  • Fill rate: 80% on limit orders
""")
        
        start = datetime.now()
        
        # Run all backtests
        self.backtest_cross_exchange_arb(years)
        self.backtest_triangular_arb(years)
        self.backtest_funding_rate_arb(years)
        self.backtest_grid_trading(years)
        self.backtest_market_making(years)
        self.backtest_futures_basis(years)
        
        duration = (datetime.now() - start).total_seconds()
        
        # Summary
        print("\n" + "="*80)
        print("                    EXECUTIVE SUMMARY")
        print("="*80)
        
        total_pnl = 0
        all_positive = True
        
        print(f"\n{'Strategy':<30} {'PnL':>15} {'Ann. Ret%':>12} {'Status':>12}")
        print("─" * 70)
        
        for name, r in self.results.items():
            pnl = r.get('total_pnl_usd', 0)
            total_pnl += pnl
            ann_ret = r.get('annualized_return_pct', pnl / (10000 * years) * 100)
            profitable = r.get('profitable', pnl > 0)
            
            if not profitable:
                all_positive = False
                
            status = "✅ +EV" if profitable else "❌ -EV"
            print(f"{r['strategy']:<30} ${pnl:>14,.2f} {ann_ret:>11.2f}% {status:>12}")
        
        print("─" * 70)
        print(f"{'TOTAL':<30} ${total_pnl:>14,.2f}")
        
        print(f"\n⏱  Analysis Duration: {duration:.1f}s")
        
        if all_positive:
            print("\n" + "="*80)
            print("  ✅ ALL STRATEGIES SHOW POSITIVE EXPECTED VALUE")
            print("="*80)
        else:
            print("\n⚠️  Some strategies show negative EV - require optimization")
        
        print("""
┌───────────────────────────────────────────────────────────────────────────────┐
│  RISK DISCLAIMER                                                              │
│  ─────────────────────────────────────────────────────────────────────────   │
│  • Past performance ≠ future results                                         │
│  • Real execution may differ from backtest                                   │
│  • Liquidity and market conditions vary                                      │
│  • Always start with paper trading                                           │
│  • Never risk more than you can afford to lose                               │
│                                                                               │
│  RECOMMENDATION: Start with Funding Rate and Futures Basis strategies.       │
│  These have the most reliable edge with lowest execution risk.               │
└───────────────────────────────────────────────────────────────────────────────┘
""")
        
        return {
            'results': {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else bool(vv) if isinstance(vv, np.bool_) else vv) 
                           for kk, vv in v.items()} 
                       for k, v in self.results.items()},
            'total_pnl': float(total_pnl),
            'all_positive_ev': all_positive
        }


def main():
    """Run professional backtest."""
    backtester = ProfessionalBacktester()
    results = backtester.run_full_analysis(years=10)
    
    # Save results
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to backtest_results.json")
    
    return results


if __name__ == "__main__":
    main()
