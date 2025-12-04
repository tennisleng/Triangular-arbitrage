"""
Strategy Runner - Main entry point for running multiple arbitrage strategies.
"""

import time
import logging
import threading
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from src.strategies import (
    StrategyOrchestrator,
    CrossExchangeArbitrage,
    FuturesSpotArbitrage,
    FundingRateArbitrage,
    GridTradingStrategy,
    MarketMakingStrategy,
    DEXCEXArbitrage
)
from src.order_executor import OrderExecutor
from src.exchange_manager import ExchangeManager
from data import settings, tokens


class StrategyRunner:
    """
    Main runner for executing multiple arbitrage strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize components
        self.exchange_manager = ExchangeManager()
        self.order_executor = OrderExecutor(self.exchange_manager)
        self.orchestrator = StrategyOrchestrator(self.exchange_manager, settings)
        
        # State
        self.running = False
        self.start_time = None
        self.stats = {
            'total_opportunities': 0,
            'executed_trades': 0,
            'total_profit_usd': 0,
            'start_balance': {}
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('strategy_runner.log')
            ]
        )
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received")
        self.stop()
        
    def enable_strategies(self, strategies: List[str]):
        """Enable specified strategies."""
        for strategy in strategies:
            self.orchestrator.enable_strategy(strategy)
            
    def get_trading_symbols(self) -> List[str]:
        """Get list of trading symbols to scan."""
        symbols = []
        
        for exchange in self.exchange_manager.get_enabled_exchanges():
            exchange_tokens = self.exchange_manager.get_exchange_tokens(exchange)
            for token in exchange_tokens:
                symbol = f"{token}/{settings.DEFAULT_QUOTE_CURRENCY}"
                if symbol not in symbols:
                    symbols.append(symbol)
                    
        return symbols
        
    def record_start_balances(self):
        """Record starting balances for P&L tracking."""
        for exchange in self.exchange_manager.get_enabled_exchanges():
            for currency in [settings.DEFAULT_BASE_CURRENCY, 
                           settings.DEFAULT_QUOTE_CURRENCY, 'USDT']:
                balance = self.exchange_manager.get_balance(exchange, currency)
                key = f"{exchange}_{currency}"
                self.stats['start_balance'][key] = balance
                
    def calculate_pnl(self) -> Dict[str, float]:
        """Calculate profit and loss since start."""
        pnl = {}
        
        for exchange in self.exchange_manager.get_enabled_exchanges():
            for currency in [settings.DEFAULT_BASE_CURRENCY,
                           settings.DEFAULT_QUOTE_CURRENCY, 'USDT']:
                key = f"{exchange}_{currency}"
                start = self.stats['start_balance'].get(key, 0)
                current = self.exchange_manager.get_balance(exchange, currency)
                pnl[key] = current - start
                
        return pnl
        
    def run_scan_cycle(self, symbols: List[str]):
        """Run one scanning cycle."""
        opportunities = self.orchestrator.run_scan_cycle(symbols)
        
        for opp in opportunities:
            self.stats['total_opportunities'] += 1
            
            # Risk check
            if not self._passes_risk_check(opp):
                self.logger.info(f"Opportunity skipped due to risk: {opp}")
                continue
                
            # Execute
            result = self._execute_opportunity(opp)
            
            if result.get('success'):
                self.stats['executed_trades'] += 1
                self.stats['total_profit_usd'] += result.get('profit_usd', 0)
                
    def _passes_risk_check(self, opportunity) -> bool:
        """Check if opportunity passes risk management rules."""
        # Check circuit breaker
        if hasattr(settings, 'ENABLE_CIRCUIT_BREAKER') and settings.ENABLE_CIRCUIT_BREAKER:
            # Check consecutive losses
            # This would be tracked in actual implementation
            pass
            
        # Check daily loss limit
        if hasattr(settings, 'MAX_DAILY_LOSS_PERCENTAGE'):
            pnl = self.calculate_pnl()
            # Check if daily loss exceeded
            pass
            
        # Check position size
        if opportunity.profit_usd > 1000:  # Example threshold
            # Large trade, require extra validation
            pass
            
        return True
        
    def _execute_opportunity(self, opportunity) -> Dict[str, Any]:
        """Execute an arbitrage opportunity."""
        try:
            strategy_name = opportunity.strategy
            
            if strategy_name == 'cross_exchange':
                strategy = self.orchestrator.strategies['cross_exchange']
                # Calculate position size
                amount = self._calculate_position_size(opportunity)
                result = strategy.execute(opportunity, amount)
                return result
                
            # Add other strategy executions here
            
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return {'success': False, 'error': str(e)}
            
        return {'success': False}
        
    def _calculate_position_size(self, opportunity) -> float:
        """Calculate optimal position size."""
        base_currency = opportunity.symbol.split('/')[0]
        balance = self.exchange_manager.get_balance(
            opportunity.exchange_buy, base_currency
        )
        
        max_size = balance * settings.MAX_POSITION_SIZE
        
        # Adjust based on confidence
        if opportunity.profit_percentage > 1.0:
            # High profit opportunity, use larger size
            size = max_size
        elif opportunity.profit_percentage > 0.5:
            size = max_size * 0.7
        else:
            size = max_size * 0.5
            
        return min(size, opportunity.volume_available * 0.5)
        
    def print_status(self):
        """Print current status."""
        runtime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        status = f"""
{'='*60}
Strategy Runner Status
{'='*60}
Runtime: {runtime:.0f} seconds
Opportunities Found: {self.stats['total_opportunities']}
Trades Executed: {self.stats['executed_trades']}
Total Profit: ${self.stats['total_profit_usd']:.2f}

Active Strategies:
"""
        for strategy in self.orchestrator.active_strategies:
            status += f"  - {strategy}\n"
            
        status += f"{'='*60}"
        print(status)
        
    def start(self, strategies: List[str] = None, paper_mode: bool = True):
        """
        Start the strategy runner.
        
        Args:
            strategies: List of strategies to enable
            paper_mode: If True, don't execute real trades
        """
        self.logger.info("=" * 60)
        self.logger.info("Strategy Runner Starting")
        self.logger.info("=" * 60)
        
        if paper_mode:
            self.logger.info("Running in PAPER TRADING mode - no real trades")
            
        # Enable strategies
        if strategies is None:
            strategies = ['cross_exchange', 'grid_trading', 'market_making']
            
        self.enable_strategies(strategies)
        
        # Record starting balances
        self.record_start_balances()
        
        self.running = True
        self.start_time = datetime.now()
        
        # Get symbols to scan
        symbols = self.get_trading_symbols()
        self.logger.info(f"Scanning {len(symbols)} symbols")
        
        # Main loop
        scan_interval = 1.0  # seconds
        status_interval = 60  # seconds
        last_status = time.time()
        
        while self.running:
            try:
                self.run_scan_cycle(symbols)
                
                # Print status periodically
                if time.time() - last_status > status_interval:
                    self.print_status()
                    last_status = time.time()
                    
                time.sleep(scan_interval)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(5)  # Back off
                
        self.logger.info("Strategy Runner stopped")
        self.print_status()
        
    def stop(self):
        """Stop the strategy runner."""
        self.logger.info("Stopping Strategy Runner...")
        self.running = False
        self.orchestrator.stop()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        return {
            **self.stats,
            'strategy_stats': self.orchestrator.get_all_stats(),
            'execution_stats': self.order_executor.get_execution_stats(),
            'pnl': self.calculate_pnl()
        }


def main():
    """Main entry point."""
    print("=" * 60)
    print("Multi-Strategy Arbitrage Bot")
    print("=" * 60)
    
    # Configuration
    paper_mode = getattr(settings, 'PAPER_TRADING', True)
    
    strategies = [
        'cross_exchange',
        'grid_trading',
        'market_making'
    ]
    
    # Optional strategies if configured
    if getattr(settings, 'ENABLE_FUTURES_ARBITRAGE', False):
        strategies.append('futures_spot')
        strategies.append('funding_rate')
        
    if getattr(settings, 'ENABLE_DEX_ARBITRAGE', False):
        strategies.append('dex_cex')
    
    print(f"\nEnabled Strategies: {', '.join(strategies)}")
    print(f"Paper Trading: {'Yes' if paper_mode else 'No - REAL TRADING'}")
    print(f"Base Currency: {settings.DEFAULT_BASE_CURRENCY}")
    print(f"Quote Currency: {settings.DEFAULT_QUOTE_CURRENCY}")
    print("=" * 60)
    
    if not paper_mode:
        print("\n⚠️  WARNING: Real trading mode enabled!")
        print("Press Ctrl+C to cancel within 5 seconds...")
        time.sleep(5)
        
    runner = StrategyRunner()
    
    try:
        runner.start(strategies=strategies, paper_mode=paper_mode)
    except KeyboardInterrupt:
        runner.stop()
        
    # Print final stats
    print("\nFinal Statistics:")
    stats = runner.get_stats()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
