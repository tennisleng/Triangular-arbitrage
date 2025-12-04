"""
Smart Order Execution Module
Implements advanced order execution algorithms for optimal trade execution.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import math


@dataclass
class OrderSlice:
    """Represents a slice of a larger order."""
    slice_id: int
    amount: float
    price: Optional[float]
    side: str
    status: str  # 'pending', 'submitted', 'filled', 'cancelled'
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    fill_price: Optional[float]
    order_id: Optional[str]


@dataclass
class ExecutionResult:
    """Result of order execution."""
    success: bool
    total_amount: float
    filled_amount: float
    average_price: float
    total_cost: float
    slippage: float
    execution_time: float
    order_slices: List[OrderSlice]
    error: Optional[str]


class TWAPExecutor:
    """
    Time-Weighted Average Price (TWAP) Executor
    Splits large orders into smaller slices executed over time.
    """
    
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        self.active_executions = {}
        
    def execute(self, exchange: str, symbol: str, side: str, total_amount: float,
               duration_seconds: float, num_slices: int = 10,
               price_limit: Optional[float] = None) -> ExecutionResult:
        """
        Execute a TWAP order.
        
        Args:
            exchange: Exchange to execute on
            symbol: Trading pair
            side: 'buy' or 'sell'
            total_amount: Total amount to execute
            duration_seconds: Time period to execute over
            num_slices: Number of order slices
            price_limit: Optional limit price
            
        Returns:
            ExecutionResult with details
        """
        start_time = datetime.now()
        slice_amount = total_amount / num_slices
        slice_interval = duration_seconds / num_slices
        
        slices: List[OrderSlice] = []
        filled_amount = 0
        total_cost = 0
        
        execution_id = f"{exchange}_{symbol}_{start_time.timestamp()}"
        self.active_executions[execution_id] = {
            'status': 'running',
            'slices': slices
        }
        
        try:
            for i in range(num_slices):
                # Check if execution was cancelled
                if self.active_executions.get(execution_id, {}).get('status') == 'cancelled':
                    break
                    
                slice_obj = OrderSlice(
                    slice_id=i,
                    amount=slice_amount,
                    price=price_limit,
                    side=side,
                    status='pending',
                    submitted_at=None,
                    filled_at=None,
                    fill_price=None,
                    order_id=None
                )
                
                # Get current market price
                ticker = self.exchange_manager.get_ticker(exchange, symbol)
                if not ticker:
                    slice_obj.status = 'cancelled'
                    slices.append(slice_obj)
                    continue
                    
                current_price = ticker['ask'] if side == 'buy' else ticker['bid']
                
                # Check price limit
                if price_limit:
                    if side == 'buy' and current_price > price_limit:
                        slice_obj.status = 'cancelled'
                        slices.append(slice_obj)
                        time.sleep(slice_interval)
                        continue
                    elif side == 'sell' and current_price < price_limit:
                        slice_obj.status = 'cancelled'
                        slices.append(slice_obj)
                        time.sleep(slice_interval)
                        continue
                
                # Submit order
                slice_obj.submitted_at = datetime.now()
                slice_obj.status = 'submitted'
                
                order = self.exchange_manager.create_market_order(
                    exchange, symbol, side, slice_amount
                )
                
                if order:
                    slice_obj.order_id = order.get('id')
                    slice_obj.status = 'filled'
                    slice_obj.filled_at = datetime.now()
                    slice_obj.fill_price = order.get('average', current_price)
                    
                    filled_amount += slice_amount
                    total_cost += slice_amount * slice_obj.fill_price
                else:
                    slice_obj.status = 'cancelled'
                    
                slices.append(slice_obj)
                
                # Wait before next slice (except for last)
                if i < num_slices - 1:
                    time.sleep(slice_interval)
                    
        except Exception as e:
            self.logger.error(f"TWAP execution error: {e}")
            
        finally:
            del self.active_executions[execution_id]
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        average_price = total_cost / filled_amount if filled_amount > 0 else 0
        
        # Calculate slippage compared to first price
        initial_price = slices[0].fill_price if slices and slices[0].fill_price else 0
        slippage = ((average_price - initial_price) / initial_price * 100) if initial_price > 0 else 0
        
        return ExecutionResult(
            success=filled_amount > 0,
            total_amount=total_amount,
            filled_amount=filled_amount,
            average_price=average_price,
            total_cost=total_cost,
            slippage=abs(slippage),
            execution_time=execution_time,
            order_slices=slices,
            error=None if filled_amount > 0 else "No orders filled"
        )


class VWAPExecutor:
    """
    Volume-Weighted Average Price (VWAP) Executor
    Executes orders based on historical volume profile.
    """
    
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        self.volume_profiles = {}
        
    def get_volume_profile(self, exchange: str, symbol: str, periods: int = 24) -> List[float]:
        """
        Get historical volume profile.
        Returns percentage of daily volume for each hour.
        """
        # In real implementation, this would fetch historical data
        # Using typical crypto volume profile as placeholder
        typical_profile = [
            0.032, 0.028, 0.025, 0.024, 0.028, 0.035,  # 0-5 (low Asia)
            0.045, 0.055, 0.060, 0.058, 0.055, 0.050,  # 6-11 (Europe open)
            0.055, 0.065, 0.070, 0.068, 0.062, 0.055,  # 12-17 (US open)
            0.048, 0.042, 0.038, 0.035, 0.033, 0.035   # 18-23 (evening)
        ]
        
        # Normalize to sum to 1
        total = sum(typical_profile)
        return [v / total for v in typical_profile]
    
    def calculate_execution_schedule(self, total_amount: float, start_hour: int,
                                    duration_hours: int) -> Dict[int, float]:
        """
        Calculate execution schedule based on volume profile.
        
        Returns:
            Dict mapping hour to amount to execute
        """
        profile = self.get_volume_profile('', '', 24)
        
        # Get relevant hours
        hours = [(start_hour + i) % 24 for i in range(duration_hours)]
        relevant_volumes = [profile[h] for h in hours]
        total_volume = sum(relevant_volumes)
        
        # Allocate amounts proportionally
        schedule = {}
        for hour, volume in zip(hours, relevant_volumes):
            schedule[hour] = total_amount * (volume / total_volume)
            
        return schedule
    
    def execute(self, exchange: str, symbol: str, side: str, total_amount: float,
               duration_hours: int = 4, slices_per_hour: int = 4,
               price_limit: Optional[float] = None) -> ExecutionResult:
        """
        Execute a VWAP order.
        
        Args:
            exchange: Exchange to execute on
            symbol: Trading pair
            side: 'buy' or 'sell'
            total_amount: Total amount to execute
            duration_hours: Duration to execute over (in hours)
            slices_per_hour: Number of slices per hour
            price_limit: Optional limit price
            
        Returns:
            ExecutionResult with details
        """
        start_time = datetime.now()
        current_hour = start_time.hour
        
        # Get execution schedule
        schedule = self.calculate_execution_schedule(
            total_amount, current_hour, duration_hours
        )
        
        slices: List[OrderSlice] = []
        filled_amount = 0
        total_cost = 0
        slice_id = 0
        
        try:
            for hour in range(duration_hours):
                actual_hour = (current_hour + hour) % 24
                hour_amount = schedule.get(actual_hour, 0)
                slice_amount = hour_amount / slices_per_hour
                
                for i in range(slices_per_hour):
                    slice_obj = OrderSlice(
                        slice_id=slice_id,
                        amount=slice_amount,
                        price=price_limit,
                        side=side,
                        status='pending',
                        submitted_at=None,
                        filled_at=None,
                        fill_price=None,
                        order_id=None
                    )
                    
                    # Get current price
                    ticker = self.exchange_manager.get_ticker(exchange, symbol)
                    if ticker:
                        current_price = ticker['ask'] if side == 'buy' else ticker['bid']
                        
                        # Check price limit
                        execute = True
                        if price_limit:
                            if side == 'buy' and current_price > price_limit:
                                execute = False
                            elif side == 'sell' and current_price < price_limit:
                                execute = False
                        
                        if execute:
                            slice_obj.submitted_at = datetime.now()
                            order = self.exchange_manager.create_market_order(
                                exchange, symbol, side, slice_amount
                            )
                            
                            if order:
                                slice_obj.status = 'filled'
                                slice_obj.filled_at = datetime.now()
                                slice_obj.fill_price = order.get('average', current_price)
                                slice_obj.order_id = order.get('id')
                                
                                filled_amount += slice_amount
                                total_cost += slice_amount * slice_obj.fill_price
                            else:
                                slice_obj.status = 'cancelled'
                        else:
                            slice_obj.status = 'cancelled'
                    else:
                        slice_obj.status = 'cancelled'
                        
                    slices.append(slice_obj)
                    slice_id += 1
                    
                    # Wait between slices
                    time.sleep(3600 / slices_per_hour)
                    
        except Exception as e:
            self.logger.error(f"VWAP execution error: {e}")
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        average_price = total_cost / filled_amount if filled_amount > 0 else 0
        initial_price = slices[0].fill_price if slices and slices[0].fill_price else 0
        slippage = ((average_price - initial_price) / initial_price * 100) if initial_price > 0 else 0
        
        return ExecutionResult(
            success=filled_amount > 0,
            total_amount=total_amount,
            filled_amount=filled_amount,
            average_price=average_price,
            total_cost=total_cost,
            slippage=abs(slippage),
            execution_time=execution_time,
            order_slices=slices,
            error=None if filled_amount > 0 else "No orders filled"
        )


class IcebergExecutor:
    """
    Iceberg Order Executor
    Hides large orders by only showing small portions.
    """
    
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        
    def execute(self, exchange: str, symbol: str, side: str, total_amount: float,
               visible_amount: float, price: float,
               price_variance: float = 0.001) -> ExecutionResult:
        """
        Execute an iceberg order.
        
        Args:
            exchange: Exchange to execute on
            symbol: Trading pair
            side: 'buy' or 'sell'
            total_amount: Total hidden amount
            visible_amount: Visible order size
            price: Limit price
            price_variance: Random price variance to hide pattern
            
        Returns:
            ExecutionResult
        """
        start_time = datetime.now()
        slices: List[OrderSlice] = []
        filled_amount = 0
        total_cost = 0
        slice_id = 0
        remaining = total_amount
        
        try:
            while remaining > 0:
                slice_amount = min(visible_amount, remaining)
                
                # Add small random variance to price
                import random
                variance = random.uniform(-price_variance, price_variance)
                slice_price = price * (1 + variance)
                
                slice_obj = OrderSlice(
                    slice_id=slice_id,
                    amount=slice_amount,
                    price=slice_price,
                    side=side,
                    status='pending',
                    submitted_at=datetime.now(),
                    filled_at=None,
                    fill_price=None,
                    order_id=None
                )
                
                # Place limit order
                order = self.exchange_manager.create_limit_order(
                    exchange, symbol, side, slice_amount, slice_price
                )
                
                if order:
                    slice_obj.order_id = order.get('id')
                    slice_obj.status = 'submitted'
                    
                    # Wait for fill (simplified - real implementation would monitor)
                    time.sleep(2)
                    
                    # Check if filled (simplified)
                    slice_obj.status = 'filled'
                    slice_obj.filled_at = datetime.now()
                    slice_obj.fill_price = slice_price
                    
                    filled_amount += slice_amount
                    total_cost += slice_amount * slice_price
                    remaining -= slice_amount
                else:
                    slice_obj.status = 'cancelled'
                    break
                    
                slices.append(slice_obj)
                slice_id += 1
                
                # Small delay between slices
                time.sleep(0.5)
                
        except Exception as e:
            self.logger.error(f"Iceberg execution error: {e}")
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        average_price = total_cost / filled_amount if filled_amount > 0 else 0
        slippage = ((average_price - price) / price * 100) if price > 0 else 0
        
        return ExecutionResult(
            success=filled_amount >= total_amount * 0.95,  # 95% fill rate
            total_amount=total_amount,
            filled_amount=filled_amount,
            average_price=average_price,
            total_cost=total_cost,
            slippage=abs(slippage),
            execution_time=execution_time,
            order_slices=slices,
            error=None
        )


class SmartOrderRouter:
    """
    Smart Order Router
    Routes orders to optimal execution venues.
    """
    
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        
    def get_best_execution_venue(self, symbol: str, side: str, 
                                 amount: float) -> Dict[str, Any]:
        """
        Find the best exchange for order execution.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            
        Returns:
            Dict with best exchange and expected execution price
        """
        exchanges = self.exchange_manager.get_enabled_exchanges()
        best_exchange = None
        best_price = float('inf') if side == 'buy' else 0
        best_depth = 0
        
        for exchange in exchanges:
            orderbook = self.exchange_manager.get_order_book(exchange, symbol, limit=10)
            if not orderbook:
                continue
                
            orders = orderbook['asks'] if side == 'buy' else orderbook['bids']
            if not orders:
                continue
                
            # Calculate effective price for the amount
            remaining = amount
            total_cost = 0
            
            for price, volume in orders:
                if remaining <= 0:
                    break
                fill_amount = min(remaining, volume)
                total_cost += fill_amount * price
                remaining -= fill_amount
                
            if remaining > 0:
                continue  # Not enough liquidity
                
            effective_price = total_cost / amount
            
            # Calculate depth score (more depth = better)
            depth = sum(v for _, v in orders[:5])
            
            # Check if this is better
            is_better = False
            if side == 'buy' and effective_price < best_price:
                is_better = True
            elif side == 'sell' and effective_price > best_price:
                is_better = True
                
            if is_better:
                best_exchange = exchange
                best_price = effective_price
                best_depth = depth
                
        if best_exchange:
            return {
                'exchange': best_exchange,
                'expected_price': best_price,
                'depth': best_depth,
                'fee': self.exchange_manager.get_exchange_fee(best_exchange)
            }
            
        return {}
    
    def split_order_across_exchanges(self, symbol: str, side: str,
                                    total_amount: float) -> List[Dict[str, Any]]:
        """
        Split order across multiple exchanges for optimal execution.
        
        Returns:
            List of (exchange, amount) pairs
        """
        exchanges = self.exchange_manager.get_enabled_exchanges()
        allocation = []
        
        # Get liquidity at each exchange
        liquidity = {}
        for exchange in exchanges:
            orderbook = self.exchange_manager.get_order_book(exchange, symbol, limit=10)
            if not orderbook:
                continue
                
            orders = orderbook['asks'] if side == 'buy' else orderbook['bids']
            if orders:
                liquidity[exchange] = sum(v for _, v in orders[:5])
                
        if not liquidity:
            return []
            
        # Allocate proportionally to liquidity
        total_liquidity = sum(liquidity.values())
        
        for exchange, liq in liquidity.items():
            proportion = liq / total_liquidity
            amount = total_amount * proportion
            
            allocation.append({
                'exchange': exchange,
                'amount': amount,
                'liquidity': liq,
                'proportion': proportion
            })
            
        return allocation
    
    def execute_smart(self, symbol: str, side: str, amount: float,
                     strategy: str = 'best_price') -> ExecutionResult:
        """
        Execute order with smart routing.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            strategy: 'best_price', 'split', or 'fastest'
            
        Returns:
            ExecutionResult
        """
        start_time = datetime.now()
        slices: List[OrderSlice] = []
        filled_amount = 0
        total_cost = 0
        
        try:
            if strategy == 'best_price':
                # Route to single best exchange
                best = self.get_best_execution_venue(symbol, side, amount)
                if best:
                    order = self.exchange_manager.create_market_order(
                        best['exchange'], symbol, side, amount
                    )
                    
                    if order:
                        filled_amount = amount
                        fill_price = order.get('average', best['expected_price'])
                        total_cost = amount * fill_price
                        
                        slices.append(OrderSlice(
                            slice_id=0,
                            amount=amount,
                            price=None,
                            side=side,
                            status='filled',
                            submitted_at=start_time,
                            filled_at=datetime.now(),
                            fill_price=fill_price,
                            order_id=order.get('id')
                        ))
                        
            elif strategy == 'split':
                # Split across exchanges
                allocation = self.split_order_across_exchanges(symbol, side, amount)
                
                for i, alloc in enumerate(allocation):
                    order = self.exchange_manager.create_market_order(
                        alloc['exchange'], symbol, side, alloc['amount']
                    )
                    
                    if order:
                        fill_price = order.get('average', 0)
                        filled_amount += alloc['amount']
                        total_cost += alloc['amount'] * fill_price
                        
                        slices.append(OrderSlice(
                            slice_id=i,
                            amount=alloc['amount'],
                            price=None,
                            side=side,
                            status='filled',
                            submitted_at=start_time,
                            filled_at=datetime.now(),
                            fill_price=fill_price,
                            order_id=order.get('id')
                        ))
                        
        except Exception as e:
            self.logger.error(f"Smart routing error: {e}")
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        average_price = total_cost / filled_amount if filled_amount > 0 else 0
        
        return ExecutionResult(
            success=filled_amount > 0,
            total_amount=amount,
            filled_amount=filled_amount,
            average_price=average_price,
            total_cost=total_cost,
            slippage=0,  # Would need reference price
            execution_time=execution_time,
            order_slices=slices,
            error=None if filled_amount > 0 else "Execution failed"
        )


class OrderExecutor:
    """
    Main order executor combining all execution strategies.
    """
    
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        
        self.twap = TWAPExecutor(exchange_manager)
        self.vwap = VWAPExecutor(exchange_manager)
        self.iceberg = IcebergExecutor(exchange_manager)
        self.smart_router = SmartOrderRouter(exchange_manager)
        
        self.execution_history = deque(maxlen=1000)
        
    def execute(self, exchange: str, symbol: str, side: str, amount: float,
               strategy: str = 'market', **kwargs) -> ExecutionResult:
        """
        Execute an order using specified strategy.
        
        Args:
            exchange: Exchange to execute on
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order amount
            strategy: Execution strategy:
                - 'market': Simple market order
                - 'twap': Time-weighted average price
                - 'vwap': Volume-weighted average price
                - 'iceberg': Hidden size order
                - 'smart': Smart order routing
                
        Returns:
            ExecutionResult
        """
        self.logger.info(f"Executing {side} {amount} {symbol} using {strategy}")
        
        if strategy == 'market':
            result = self._execute_market(exchange, symbol, side, amount)
        elif strategy == 'twap':
            duration = kwargs.get('duration_seconds', 60)
            num_slices = kwargs.get('num_slices', 10)
            result = self.twap.execute(exchange, symbol, side, amount, 
                                       duration, num_slices)
        elif strategy == 'vwap':
            duration_hours = kwargs.get('duration_hours', 4)
            result = self.vwap.execute(exchange, symbol, side, amount,
                                       duration_hours)
        elif strategy == 'iceberg':
            visible = kwargs.get('visible_amount', amount * 0.1)
            price = kwargs.get('price')
            if not price:
                ticker = self.exchange_manager.get_ticker(exchange, symbol)
                price = ticker['ask'] if side == 'buy' else ticker['bid']
            result = self.iceberg.execute(exchange, symbol, side, amount,
                                         visible, price)
        elif strategy == 'smart':
            routing_strategy = kwargs.get('routing', 'best_price')
            result = self.smart_router.execute_smart(symbol, side, amount,
                                                    routing_strategy)
        else:
            result = ExecutionResult(
                success=False,
                total_amount=amount,
                filled_amount=0,
                average_price=0,
                total_cost=0,
                slippage=0,
                execution_time=0,
                order_slices=[],
                error=f"Unknown strategy: {strategy}"
            )
            
        self.execution_history.append({
            'timestamp': datetime.now(),
            'exchange': exchange,
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'strategy': strategy,
            'result': result
        })
        
        return result
    
    def _execute_market(self, exchange: str, symbol: str, side: str,
                       amount: float) -> ExecutionResult:
        """Execute simple market order."""
        start_time = datetime.now()
        
        order = self.exchange_manager.create_market_order(
            exchange, symbol, side, amount
        )
        
        end_time = datetime.now()
        
        if order:
            fill_price = order.get('average', order.get('price', 0))
            return ExecutionResult(
                success=True,
                total_amount=amount,
                filled_amount=order.get('filled', amount),
                average_price=fill_price,
                total_cost=amount * fill_price,
                slippage=0,
                execution_time=(end_time - start_time).total_seconds(),
                order_slices=[OrderSlice(
                    slice_id=0,
                    amount=amount,
                    price=None,
                    side=side,
                    status='filled',
                    submitted_at=start_time,
                    filled_at=end_time,
                    fill_price=fill_price,
                    order_id=order.get('id')
                )],
                error=None
            )
        else:
            return ExecutionResult(
                success=False,
                total_amount=amount,
                filled_amount=0,
                average_price=0,
                total_cost=0,
                slippage=0,
                execution_time=(end_time - start_time).total_seconds(),
                order_slices=[],
                error="Market order failed"
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {}
            
        total_executions = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e['result'].success)
        
        total_slippage = sum(e['result'].slippage for e in self.execution_history 
                            if e['result'].success)
        avg_slippage = total_slippage / successful if successful > 0 else 0
        
        by_strategy = {}
        for e in self.execution_history:
            strategy = e['strategy']
            if strategy not in by_strategy:
                by_strategy[strategy] = {'count': 0, 'success': 0}
            by_strategy[strategy]['count'] += 1
            if e['result'].success:
                by_strategy[strategy]['success'] += 1
                
        return {
            'total_executions': total_executions,
            'successful': successful,
            'success_rate': successful / total_executions if total_executions > 0 else 0,
            'avg_slippage': avg_slippage,
            'by_strategy': by_strategy
        }
