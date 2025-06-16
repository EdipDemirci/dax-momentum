#!/usr/bin/env python3
"""
Command-line interface for momentum backtesting with parameter sweep capability.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import momentum_backtest


def main():
    parser = argparse.ArgumentParser(
        description='Run momentum backtest with specified parameters'
    )
    parser.add_argument(
        '--window', 
        type=int, 
        default=3,
        help='Lookback window in days (default: 3)'
    )
    parser.add_argument(
        '--fee', 
        type=float, 
        default=0.0001,
        help='Transaction fee as decimal (default: 0.0001 = 1bp)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.window <= 0:
        print("Error: Window must be positive")
        sys.exit(1)
    
    if args.fee < 0:
        print("Error: Fee cannot be negative")
        sys.exit(1)
    
    try:
        # Load data
        data_path = Path(__file__).parent.parent / "data" / "dax_daily.csv"
        if not data_path.exists():
            print(f"Error: Data file not found: {data_path}")
            sys.exit(1)
        
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        prices = df['Close']
        
        print(f"Loaded {len(prices)} price points from {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
        
        # Run backtest
        print(f"\nRunning momentum backtest with {args.window}-day window and {args.fee:.4f} fee...")
        results = momentum_backtest(prices, args.window, args.fee)
        
        # Print metrics
        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS (Window: {args.window} days)")
        print(f"{'='*50}")
        print(f"CAGR      {results['cagr']:.2%}")
        print(f"Sharpe    {results['sharpe']:.2f}")
        print(f"MaxDD     {results['max_dd']:.2%}")
        print(f"{'='*50}")
        
        # Create figures directory
        fig_path = Path(__file__).parent.parent / "fig"
        fig_path.mkdir(exist_ok=True)
        
        # Generate plots with window in filename
        window_suffix = f"_w{args.window}"
        
        # Equity curve
        plt.figure(figsize=(12, 6))
        results['equity'].plot(title=f"Equity Curve (Window: {args.window} days)")
        plt.grid(True, alpha=0.3)
        plt.ylabel('Equity')
        plt.xlabel('Date')
        equity_file = fig_path / f"equity{window_suffix}.png"
        plt.savefig(equity_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved equity curve: {equity_file}")
        
        # Trade return histogram
        trade_returns = results['returns'][results['returns'] != 0]
        if len(trade_returns) > 0:
            plt.figure(figsize=(10, 6))
            trade_returns.hist(bins=40, alpha=0.7)
            plt.title(f"Trade Return Histogram (Window: {args.window} days)")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            hist_file = fig_path / f"hist{window_suffix}.png"
            plt.savefig(hist_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved histogram: {hist_file}")
        else:
            print("Warning: No trades to plot in histogram")
        
        # Drawdown plot
        plt.figure(figsize=(12, 6))
        drawdown = (results['equity'] / results['equity'].cummax()) - 1
        drawdown.plot(title=f"Drawdown (Window: {args.window} days)")
        plt.ylabel("Drawdown")
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        dd_file = fig_path / f"drawdown{window_suffix}.png"
        plt.savefig(dd_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved drawdown: {dd_file}")
        
        print(f"\nAll plots saved to: {fig_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 