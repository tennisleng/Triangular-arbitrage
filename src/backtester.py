"""
Comprehensive backtesting framework for arbitrage strategies.
Tests trading strategies against historical data to evaluate performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import os
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    profitable_trades: int
    total_profit_usd: float
    total_profit_percentage: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_profit_per_trade: float
    avg_holding_time: float
    max_consecutive_losses: int
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    alpha: float
    beta: float
    benchmark_return: float


@dataclass
class Trade:
    """Represents a single trade in backtesting."""
    timestamp: datetime
    exchange: str
    base_currency: str
    quote_currency: str
    alt_currency: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_fee: float
    exit_fee: float
    profit_usd: float
    profit_percentage: float
    holding_time: float
    arbitrage_type: str


class ArbitrageBacktester:
    """Backtesting framework for arbitrage strategies."""

    def __init__(self, exchange_manager, data_directory: str = "historical_data"):
        self.exchange_manager = exchange_manager
        self.data_directory = data_directory
        self.logger = logging.getLogger(__name__)

        # Ensure data directory exists
        os.makedirs(data_directory, exist_ok=True)

        # Historical data cache
        self.price_data = {}
        self.orderbook_data = {}

        # Backtesting parameters
        self.transaction_fee = 0.001  # 0.1% default
        self.slippage = 0.0005  # 0.05% default
        self.min_profit_threshold = 0.001  # 0.1% minimum

    def load_historical_data(self, exchange: str, symbol: str, start_date: datetime,
                           end_date: datetime, timeframe: str = '1m') -> pd.DataFrame:
        """
        Load historical price data for backtesting.

        Args:
            exchange: Exchange name
            symbol: Trading symbol (e.g., 'BTC/USDT')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe ('1m', '5m', '1h', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check if data already exists
            data_file = os.path.join(
                self.data_directory,
                f"{exchange}_{symbol.replace('/', '_')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            )

            if os.path.exists(data_file):
                df = pd.read_csv(data_file, parse_dates=['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df

            # Fetch data from exchange (if available)
            if exchange in self.exchange_manager.exchanges:
                exchange_instance = self.exchange_manager.exchanges[exchange]['instance']

                # Convert dates to milliseconds
                since = int(start_date.timestamp() * 1000)

                # Fetch OHLCV data
                ohlcv = exchange_instance.fetch_ohlcv(symbol, timeframe, since)

                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Save to file
                df.to_csv(data_file)

                return df

        except Exception as e:
            self.logger.error(f"Error loading historical data for {exchange} {symbol}: {e}")

        # Return empty DataFrame if data loading fails
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def simulate_triangular_arbitrage(self, exchange: str, base: str, quote: str, alt: str,
                                     start_date: datetime, end_date: datetime,
                                     settings: Dict[str, Any]) -> List[Trade]:
        """
        Simulate triangular arbitrage trading for a specific triangle.

        Args:
            exchange: Exchange name
            base: Base currency
            quote: Quote currency
            alt: Alternative currency
            start_date: Start date for simulation
            end_date: End date for simulation
            settings: Trading settings

        Returns:
            List of simulated trades
        """
        trades = []

        # Load historical data for all three pairs
        pairs = [
            f"{base}/{quote}",
            f"{base}/{alt}",
            f"{alt}/{quote}"
        ]

        price_data = {}
        for pair in pairs:
            df = self.load_historical_data(exchange, pair, start_date, end_date)
            if not df.empty:
                price_data[pair] = df

        if len(price_data) < 3:
            self.logger.warning(f"Insufficient historical data for {base}-{alt}-{quote} triangle")
            return trades

        # Combine timestamps from all pairs
        all_timestamps = set()
        for df in price_data.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(list(all_timestamps))

        # Simulate trading
        for timestamp in tqdm(all_timestamps, desc=f"Simulating {base}-{alt}-{quote}"):
            try:
                # Get prices at this timestamp (forward fill if needed)
                prices = {}
                for pair, df in price_data.items():
                    if timestamp in df.index:
                        prices[pair] = df.loc[timestamp, 'close']
                    else:
                        # Forward fill
                        available_timestamps = df.index[df.index <= timestamp]
                        if not available_timestamps.empty:
                            closest_time = available_timestamps[-1]
                            prices[pair] = df.loc[closest_time, 'close']

                if len(prices) < 3:
                    continue

                # Check for arbitrage opportunities
                forward_profit = self._calculate_simulated_forward_arbitrage(
                    prices[f"{base}/{quote}"],
                    prices[f"{base}/{alt}"],
                    prices[f"{alt}/{quote}"]
                )

                backward_profit = self._calculate_simulated_backward_arbitrage(
                    prices[f"{base}/{quote}"],
                    prices[f"{base}/{alt}"],
                    prices[f"{alt}/{quote}"]
                )

                # Execute trade if profitable
                if forward_profit['profit_percentage'] > settings.get('MIN_DIFFERENCE', 0.15):
                    trade = self._create_simulated_trade(
                        timestamp, exchange, base, quote, alt, 'forward',
                        forward_profit, settings
                    )
                    if trade:
                        trades.append(trade)

                elif backward_profit['profit_percentage'] > settings.get('MIN_DIFFERENCE', 0.15):
                    trade = self._create_simulated_trade(
                        timestamp, exchange, base, quote, alt, 'backward',
                        backward_profit, settings
                    )
                    if trade:
                        trades.append(trade)

            except Exception as e:
                self.logger.debug(f"Error simulating trade at {timestamp}: {e}")
                continue

        return trades

    def _calculate_simulated_forward_arbitrage(self, base_quote_price: float,
                                             base_alt_price: float,
                                             alt_quote_price: float) -> Dict[str, float]:
        """Calculate forward arbitrage with simulated fees and slippage."""
        # Apply slippage
        slippage_factor = 1 + self.slippage

        # Forward: base -> alt -> quote -> base
        # Step 1: base -> alt (sell base for alt)
        base_to_alt = base_alt_price / slippage_factor  # Worse price due to slippage

        # Step 2: alt -> quote (sell alt for quote)
        alt_to_quote = alt_quote_price / slippage_factor

        # Step 3: quote -> base (buy base with quote)
        quote_to_base = 1 / (base_quote_price * slippage_factor)  # Worse price

        # Calculate final amount
        final_amount = 1.0
        final_amount *= (1 - self.transaction_fee) / base_to_alt
        final_amount *= (1 - self.transaction_fee) / alt_to_quote
        final_amount *= (1 - self.transaction_fee) * quote_to_base

        profit_percentage = (final_amount - 1.0) * 100
        profit_usd = profit_percentage * 0.01 * 3000  # Assume $3000 per unit

        return {
            'profit_percentage': profit_percentage,
            'profit_usd': profit_usd,
            'final_amount': final_amount
        }

    def _calculate_simulated_backward_arbitrage(self, base_quote_price: float,
                                              base_alt_price: float,
                                              alt_quote_price: float) -> Dict[str, float]:
        """Calculate backward arbitrage with simulated fees and slippage."""
        # Apply slippage
        slippage_factor = 1 + self.slippage

        # Backward: base -> quote -> alt -> base
        # Step 1: base -> quote (sell base for quote)
        base_to_quote = base_quote_price / slippage_factor

        # Step 2: quote -> alt (buy alt with quote)
        quote_to_alt = 1 / (alt_quote_price * slippage_factor)

        # Step 3: alt -> base (sell alt for base)
        alt_to_base = base_alt_price / slippage_factor

        # Calculate final amount
        final_amount = 1.0
        final_amount *= (1 - self.transaction_fee) / base_to_quote
        final_amount *= (1 - self.transaction_fee) * quote_to_alt
        final_amount *= (1 - self.transaction_fee) * alt_to_base

        profit_percentage = (final_amount - 1.0) * 100
        profit_usd = profit_percentage * 0.01 * 3000

        return {
            'profit_percentage': profit_percentage,
            'profit_usd': profit_usd,
            'final_amount': final_amount
        }

    def _create_simulated_trade(self, timestamp: datetime, exchange: str, base: str,
                               quote: str, alt: str, direction: str,
                               profit_data: Dict[str, float], settings: Dict[str, Any]) -> Optional[Trade]:
        """Create a simulated trade object."""
        try:
            # Calculate position size based on settings
            max_position = settings.get('MAX_POSITION_SIZE', 0.5)
            base_balance = 1.0  # Assume 1 unit starting balance
            position_size = base_balance * max_position

            # Calculate fees
            entry_fee = position_size * self.transaction_fee
            exit_fee = position_size * self.transaction_fee

            # Create trade
            trade = Trade(
                timestamp=timestamp,
                exchange=exchange,
                base_currency=base,
                quote_currency=quote,
                alt_currency=alt,
                direction=direction,
                entry_price=1.0,  # Normalized entry
                exit_price=profit_data['final_amount'],
                quantity=position_size,
                entry_fee=entry_fee,
                exit_fee=exit_fee,
                profit_usd=profit_data['profit_usd'] * position_size,
                profit_percentage=profit_data['profit_percentage'],
                holding_time=60.0,  # Assume 1 minute holding time
                arbitrage_type='triangular'
            )

            return trade

        except Exception as e:
            self.logger.error(f"Error creating simulated trade: {e}")
            return None

    def run_backtest(self, strategy_name: str, exchanges: List[str], base_currencies: List[str],
                    start_date: datetime, end_date: datetime, settings: Dict[str, Any]) -> BacktestResult:
        """
        Run comprehensive backtest for arbitrage strategy.

        Args:
            strategy_name: Name of the strategy being tested
            exchanges: List of exchanges to test
            base_currencies: List of base currencies to test
            start_date: Start date for backtest
            end_date: End date for backtest
            settings: Trading settings

        Returns:
            BacktestResult with comprehensive performance metrics
        """
        all_trades = []

        # Run simulation for each exchange and currency combination
        for exchange in exchanges:
            for base in base_currencies:
                # Get available alt currencies for this exchange
                alt_currencies = self.exchange_manager.get_exchange_tokens(exchange)[:10]  # Limit for performance

                for alt in alt_currencies:
                    if alt == base:
                        continue

                    try:
                        trades = self.simulate_triangular_arbitrage(
                            exchange, base, settings.get('DEFAULT_QUOTE_CURRENCY', 'BTC'),
                            alt, start_date, end_date, settings
                        )
                        all_trades.extend(trades)

                    except Exception as e:
                        self.logger.error(f"Error backtesting {exchange} {base}-{alt}: {e}")
                        continue

        # Calculate performance metrics
        if not all_trades:
            return BacktestResult(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                total_trades=0,
                profitable_trades=0,
                total_profit_usd=0.0,
                total_profit_percentage=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                avg_profit_per_trade=0.0,
                avg_holding_time=0.0,
                max_consecutive_losses=0,
                profit_factor=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                alpha=0.0,
                beta=0.0,
                benchmark_return=0.0
            )

        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([asdict(trade) for trade in all_trades])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # Calculate returns
        trades_df['returns'] = trades_df['profit_percentage'] / 100
        trades_df['cumulative_returns'] = (1 + trades_df['returns']).cumprod() - 1

        # Calculate drawdown
        rolling_max = trades_df['cumulative_returns'].expanding().max()
        drawdown = trades_df['cumulative_returns'] - rolling_max
        max_drawdown = drawdown.min()

        # Calculate Sharpe ratio (assuming daily returns)
        if len(trades_df) > 1:
            daily_returns = trades_df.set_index('timestamp')['returns'].resample('D').sum()
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate Sortino ratio (downside deviation)
        negative_returns = trades_df[trades_df['returns'] < 0]['returns']
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = trades_df['returns'].mean() / downside_std * np.sqrt(365) if downside_std > 0 else 0

        # Calculate win rate and profit factor
        profitable_trades = len(trades_df[trades_df['profit_usd'] > 0])
        losing_trades = len(trades_df[trades_df['profit_usd'] <= 0])

        win_rate = profitable_trades / len(trades_df) if len(trades_df) > 0 else 0

        gross_profit = trades_df[trades_df['profit_usd'] > 0]['profit_usd'].sum()
        gross_loss = abs(trades_df[trades_df['profit_usd'] <= 0]['profit_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for profit in trades_df['profit_usd']:
            if profit <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        # Calculate Calmar ratio
        total_return = trades_df['cumulative_returns'].iloc[-1] if len(trades_df) > 0 else 0
        years = (end_date - start_date).days / 365
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_trades=len(trades_df),
            profitable_trades=profitable_trades,
            total_profit_usd=trades_df['profit_usd'].sum(),
            total_profit_percentage=total_return * 100,
            max_drawdown=max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_profit_per_trade=trades_df['profit_usd'].mean(),
            avg_holding_time=trades_df['holding_time'].mean(),
            max_consecutive_losses=max_consecutive_losses,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            alpha=0.0,  # Would need benchmark comparison
            beta=0.0,   # Would need benchmark comparison
            benchmark_return=0.0  # Would need benchmark data
        )

    def compare_strategies(self, results: List[BacktestResult]) -> pd.DataFrame:
        """Compare multiple backtest results."""
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Total Trades': result.total_trades,
                'Win Rate': f"{result.win_rate:.1%}",
                'Total Profit ($)': f"{result.total_profit_usd:.2f}",
                'Total Return (%)': f"{result.total_profit_percentage:.2f}%",
                'Max Drawdown (%)': f"{result.max_drawdown:.2f}%",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Avg Profit/Trade ($)': f"{result.avg_profit_per_trade:.2f}",
                'Max Consecutive Losses': result.max_consecutive_losses
            })

        return pd.DataFrame(comparison_data)

    def plot_backtest_results(self, result: BacktestResult, trades: List[Trade],
                            save_path: Optional[str] = None):
        """Create comprehensive plots for backtest results."""
        if not trades:
            return

        trades_df = pd.DataFrame([asdict(trade) for trade in trades])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Backtest Results: {result.strategy_name}', fontsize=16)

        # Cumulative returns
        trades_df['cumulative_profit'] = trades_df['profit_usd'].cumsum()
        axes[0, 0].plot(trades_df['timestamp'], trades_df['cumulative_profit'])
        axes[0, 0].set_title('Cumulative Profit Over Time')
        axes[0, 0].set_ylabel('Profit ($)')
        axes[0, 0].grid(True)

        # Daily returns
        daily_returns = trades_df.set_index('timestamp')['profit_usd'].resample('D').sum()
        axes[0, 1].bar(daily_returns.index, daily_returns.values)
        axes[0, 1].set_title('Daily Profit')
        axes[0, 1].set_ylabel('Profit ($)')
        axes[0, 1].grid(True)

        # Profit distribution
        axes[1, 0].hist(trades_df['profit_usd'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Profit Distribution')
        axes[1, 0].set_xlabel('Profit ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # Drawdown
        trades_df['cumulative_max'] = trades_df['cumulative_profit'].expanding().max()
        trades_df['drawdown'] = trades_df['cumulative_profit'] - trades_df['cumulative_max']
        axes[1, 1].fill_between(trades_df['timestamp'], trades_df['drawdown'], 0, alpha=0.3, color='red')
        axes[1, 1].set_title('Drawdown Over Time')
        axes[1, 1].set_ylabel('Drawdown ($)')
        axes[1, 1].grid(True)

        # Win/Loss ratio
        win_trades = len(trades_df[trades_df['profit_usd'] > 0])
        loss_trades = len(trades_df[trades_df['profit_usd'] <= 0])
        axes[2, 0].bar(['Wins', 'Losses'], [win_trades, loss_trades],
                      color=['green', 'red'], alpha=0.7)
        axes[2, 0].set_title('Win/Loss Count')
        axes[2, 0].set_ylabel('Number of Trades')

        # Profit by exchange
        exchange_profits = trades_df.groupby('exchange')['profit_usd'].sum()
        axes[2, 1].bar(exchange_profits.index, exchange_profits.values, alpha=0.7)
        axes[2, 1].set_title('Profit by Exchange')
        axes[2, 1].set_ylabel('Total Profit ($)')
        axes[2, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Backtest plots saved to {save_path}")
        else:
            plt.show()

    def optimize_strategy_parameters(self, base_settings: Dict[str, Any],
                                   parameter_ranges: Dict[str, List[float]],
                                   exchanges: List[str], base_currencies: List[str],
                                   start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.

        Args:
            base_settings: Base strategy settings
            parameter_ranges: Dictionary of parameter names to lists of values to test
            exchanges: Exchanges to test on
            base_currencies: Base currencies to test
            start_date: Start date for optimization
            end_date: End date for optimization

        Returns:
            Best parameter combination and results
        """
        from itertools import product

        # Generate all parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        combinations = list(product(*param_values))

        best_result = None
        best_params = None
        best_score = -float('inf')

        self.logger.info(f"Testing {len(combinations)} parameter combinations...")

        for combo in tqdm(combinations, desc="Optimizing parameters"):
            # Create settings with current parameter combination
            test_settings = base_settings.copy()
            for name, value in zip(param_names, combo):
                test_settings[name] = value

            # Run backtest
            result = self.run_backtest(
                f"optimization_{'_'.join([f'{k}={v}' for k, v in zip(param_names, combo)])}",
                exchanges, base_currencies, start_date, end_date, test_settings
            )

            # Score based on Sharpe ratio and win rate (adjust weights as needed)
            score = result.sharpe_ratio * 0.7 + result.win_rate * 0.3

            if score > best_score:
                best_score = score
                best_result = result
                best_params = dict(zip(param_names, combo))

        self.logger.info(f"Best parameters found: {best_params}")
        self.logger.info(f"Best score: {best_score:.4f}")

        return {
            'best_parameters': best_params,
            'best_result': best_result,
            'best_score': best_score
        }

    def save_backtest_results(self, result: BacktestResult, trades: List[Trade],
                            filename: str):
        """Save backtest results to file."""
        try:
            # Create results directory
            results_dir = os.path.join(self.data_directory, 'backtest_results')
            os.makedirs(results_dir, exist_ok=True)

            # Save result summary
            result_dict = asdict(result)
            result_dict['start_date'] = result.start_date.isoformat()
            result_dict['end_date'] = result.end_date.isoformat()

            with open(os.path.join(results_dir, f"{filename}_summary.json"), 'w') as f:
                json.dump(result_dict, f, indent=2)

            # Save trades
            trades_data = [asdict(trade) for trade in trades]
            for trade in trades_data:
                trade['timestamp'] = trade['timestamp'].isoformat()

            with open(os.path.join(results_dir, f"{filename}_trades.json"), 'w') as f:
                json.dump(trades_data, f, indent=2)

            self.logger.info(f"Backtest results saved to {results_dir}")

        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")

    def load_backtest_results(self, filename: str) -> Tuple[BacktestResult, List[Trade]]:
        """Load backtest results from file."""
        try:
            results_dir = os.path.join(self.data_directory, 'backtest_results')

            # Load summary
            with open(os.path.join(results_dir, f"{filename}_summary.json"), 'r') as f:
                result_dict = json.load(f)

            result_dict['start_date'] = datetime.fromisoformat(result_dict['start_date'])
            result_dict['end_date'] = datetime.fromisoformat(result_dict['end_date'])

            result = BacktestResult(**result_dict)

            # Load trades
            with open(os.path.join(results_dir, f"{filename}_trades.json"), 'r') as f:
                trades_data = json.load(f)

            trades = []
            for trade_dict in trades_data:
                trade_dict['timestamp'] = datetime.fromisoformat(trade_dict['timestamp'])
                trades.append(Trade(**trade_dict))

            return result, trades

        except Exception as e:
            self.logger.error(f"Error loading backtest results: {e}")
            return None, []
