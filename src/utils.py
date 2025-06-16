import pandas as pd
import numpy as np
from pathlib import Path


def momentum_backtest(prices, lookback, fee=0.0001):
    """
    Run momentum backtest with specified lookback window and transaction fee.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series with datetime index
    lookback : int
        Number of days to look back for momentum signal
    fee : float
        Transaction fee as decimal (default 0.0001 = 1bp)
        
    Returns:
    --------
    dict : Dictionary containing CAGR, Sharpe, MaxDD, and returns series
    """
    if lookback <= 0:
        raise ValueError("Lookback window must be positive")
    
    if len(prices) < lookback + 2:
        raise ValueError(f"Not enough data points. Need at least {lookback + 2}, got {len(prices)}")
    
    # Calculate returns
    ret = prices.pct_change().fillna(0)
    
    # Generate signal: positive if last N days had positive cumulative return
    signal = (ret.rolling(lookback).sum() > 0).astype(int)
    
    # Position: trade next day (shift signal by 1)
    pos = signal.shift(1).fillna(0)
    
    # Strategy returns with transaction costs
    strategy_ret = pos * ret - fee * pos.diff().abs().fillna(0)
    
    # Calculate metrics
    ann = 252  # trading days per year
    
    # Sharpe ratio with error handling
    if strategy_ret.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(ann)
    
    # Equity curve
    equity = (1 + strategy_ret).cumprod()
    
    # CAGR
    if len(strategy_ret) == 0 or equity.iloc[-1] <= 0:
        cagr = 0.0
    else:
        cagr = equity.iloc[-1] ** (ann / len(strategy_ret)) - 1
    
    # Maximum drawdown
    mdd = (equity / equity.cummax() - 1).min()
    
    return {
        'cagr': cagr,
        'sharpe': sharpe,
        'max_dd': mdd,
        'returns': strategy_ret,
        'equity': equity
    } 