#!/usr/bin/env python3
"""
Vectorbt-powered momentum backtest with professional portfolio statistics.
Falls back gracefully if vectorbt is not installed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    try:
        import vectorbt as vbt
        print("✅ Vectorbt successfully imported")
    except ImportError:
        print("❌ Vectorbt not installed. Please install with:")
        print("   pip install vectorbt")
        print("   or")
        print("   conda install -c conda-forge vectorbt")
        print("\nFalling back to basic analysis...")
        run_fallback_analysis()
        return

    try:
        # Load data
        data_path = Path(__file__).parent.parent / "data" / "dax_daily.csv"
        if not data_path.exists():
            print(f"Error: Data file not found: {data_path}")
            sys.exit(1)
        
        print(f"Loading data from {data_path}")
        # Read CSV with proper column names, skipping the metadata rows
        df = pd.read_csv(data_path, skiprows=3, names=['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        prices = df['Close']
        
        print(f"Loaded {len(prices)} price points from {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate returns for signal generation
        returns = prices.pct_change().fillna(0)
        
        # Generate momentum signal: 3-day rolling sum > 0
        lookback = 3
        momentum_sum = returns.rolling(lookback).sum()
        
        # Create entry signals (when momentum turns positive)
        entries = (momentum_sum > 0) & (momentum_sum.shift(1) <= 0)
        
        # Create exit signals (when momentum turns negative) 
        exits = (momentum_sum <= 0) & (momentum_sum.shift(1) > 0)
        
        # Alternative: simpler approach - always be in when momentum is positive
        # This matches our original strategy more closely
        long_signals = momentum_sum > 0
        
        print(f"\nGenerating signals with {lookback}-day momentum rule...")
        print(f"Total long signals: {long_signals.sum()}")
        print(f"Percentage of time in market: {long_signals.mean():.1%}")
        
        # Build vectorbt portfolio
        print("\nBuilding vectorbt portfolio...")
        
        # Method 1: Using from_signals (entry/exit based)
        # portfolio = vbt.Portfolio.from_signals(
        #     close=prices,
        #     entries=entries,
        #     exits=exits,
        #     fees=0.0001,  # 1 bp
        #     freq='D'
        # )
        
        # Method 2: Using from_signals with cleaner entry/exit logic
        # Create proper entry and exit signals
        # Entry: when we want to be long but aren't already long
        entries = long_signals & ~long_signals.shift(1).fillna(False)
        
        # Exit: when we want to be out but are currently long
        exits = ~long_signals & long_signals.shift(1).fillna(False)
        
        # Create portfolio using signal-based approach
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            fees=0.0001,  # 1 bp
            freq='D'
        )
        
        # Print comprehensive statistics
        print(f"\n{'='*80}")
        print("🚀 VECTORBT PORTFOLIO STATISTICS")
        print(f"{'='*80}")
        
        stats = portfolio.stats()
        print(stats)
        
        print(f"\n{'='*80}")
        print("📊 KEY METRICS SUMMARY")
        print(f"{'='*80}")
        
        # Extract key metrics for cleaner display
        stats_dict = portfolio.stats()
        total_return = stats_dict['Total Return [%]'] / 100
        sharpe_ratio = stats_dict['Sharpe Ratio']
        max_dd = stats_dict['Max Drawdown [%]'] / 100
        win_rate = stats_dict['Win Rate [%]'] / 100
        profit_factor = stats_dict['Profit Factor']
        
        # Calculate CAGR manually from total return and period
        years = (prices.index[-1] - prices.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1/years) - 1
        
        print(f"Total Return:    {total_return:.2%}")
        print(f"CAGR:           {cagr:.2%}")
        print(f"Sharpe Ratio:   {sharpe_ratio:.2f}")
        print(f"Max Drawdown:   {max_dd:.2%}")
        print(f"Win Rate:       {win_rate:.1%}")
        print(f"Profit Factor:  {profit_factor:.2f}")
        
        # Additional vectorbt-specific metrics
        print(f"\n📈 TRADE STATISTICS")
        print(f"Total Trades:   {portfolio.trades.count()}")
        print(f"Avg Trade:      {portfolio.trades.returns.mean():.2%}")
        print(f"Best Trade:     {portfolio.trades.returns.max():.2%}")
        print(f"Worst Trade:    {portfolio.trades.returns.min():.2%}")
        
        # Create and save equity curve plot
        print(f"\n📊 Generating plots...")
        fig_path = Path(__file__).parent.parent / "fig"
        fig_path.mkdir(exist_ok=True)
        
        # Vectorbt has beautiful built-in plotting
        plt.figure(figsize=(14, 8))
        
        # Plot equity curve with benchmark
        portfolio_value = portfolio.value()
        benchmark_value = (1 + returns).cumprod()  # Buy and hold
        
        plt.subplot(2, 1, 1)
        portfolio_value.plot(label='Momentum Strategy', linewidth=2)
        benchmark_value.plot(label='Buy & Hold', alpha=0.7, linewidth=2)
        plt.title('Equity Curves: Momentum Strategy vs Buy & Hold', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        drawdown = portfolio.drawdown()
        drawdown.plot(color='red', alpha=0.7)
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        plt.title('Strategy Drawdown', fontsize=14, fontweight='bold')
        plt.ylabel('Drawdown')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        equity_file = fig_path / "vbt_equity.png"
        plt.savefig(equity_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Equity curve saved: {equity_file}")
        
        # Save additional vectorbt-specific plots
        if hasattr(portfolio.trades, 'plot'):
            try:
                # Plot trade PnL distribution
                plt.figure(figsize=(10, 6))
                portfolio.trades.returns.hist(bins=30, alpha=0.7, edgecolor='black')
                plt.title('Trade Returns Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Trade Return')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                trade_dist_file = fig_path / "vbt_trade_distribution.png"
                plt.savefig(trade_dist_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Trade distribution saved: {trade_dist_file}")
            except Exception as e:
                print(f"Note: Could not generate trade distribution plot: {e}")
        
        print(f"\n🎉 Vectorbt analysis complete! All plots saved to: {fig_path}")
        
    except Exception as e:
        print(f"Error running vectorbt analysis: {e}")
        print("Falling back to basic analysis...")
        run_fallback_analysis()


def run_fallback_analysis():
    """
    Simple fallback analysis when vectorbt is not available.
    Uses our basic pandas implementation.
    """
    try:
        # Import our utility function
        sys.path.append(str(Path(__file__).parent))
        from utils import momentum_backtest
        
        # Load data
        data_path = Path(__file__).parent.parent / "data" / "dax_daily.csv"
        df = pd.read_csv(data_path, skiprows=3, names=['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        prices = df['Close']
        
        print("Running basic momentum backtest (fallback mode)...")
        results = momentum_backtest(prices, lookback=3, fee=0.0001)
        
        print(f"\n{'='*50}")
        print("BASIC BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"CAGR:      {results['cagr']:.2%}")
        print(f"Sharpe:    {results['sharpe']:.2f}")
        print(f"Max DD:    {results['max_dd']:.2%}")
        
        # Simple plot
        fig_path = Path(__file__).parent.parent / "fig"
        fig_path.mkdir(exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        results['equity'].plot(title="Equity Curve (Fallback Mode)")
        plt.grid(True, alpha=0.3)
        equity_file = fig_path / "vbt_equity.png"
        plt.savefig(equity_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Basic equity curve saved: {equity_file}")
        
    except Exception as e:
        print(f"Error in fallback analysis: {e}")


if __name__ == "__main__":
    main() 