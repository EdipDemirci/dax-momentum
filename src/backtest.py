import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ---------- load daily data ----------
# Use proper path handling
data_path = Path(__file__).parent.parent / "data" / "dax_daily.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)                       # 1) plain read
df['Date'] = pd.to_datetime(df['Date'])           # 2) force datetime
df.set_index('Date', inplace=True)                # 3) make it the index
prices = df['Close']                              # 4) grab the series

# ---------- strategy ----------
ret = prices.pct_change().fillna(0)
signal = (ret.rolling(3).sum() > 0).astype(int)      # last 3 days up?
pos = signal.shift(1).fillna(0)                      # trade next day
fee = 0.0001                                          # 1 bp
strategy_ret = pos * ret - fee * pos.diff().abs().fillna(0)

# ---------- metrics ----------
ann = 252                                             # trading days/yr

# Add error handling for calculations
if strategy_ret.std() == 0:
    sharpe = 0.0
    print("Warning: Strategy returns have zero volatility")
else:
    sharpe = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(ann)

equity = (1 + strategy_ret).cumprod()
if len(strategy_ret) == 0:
    cagr = 0.0
    print("Warning: No strategy returns available")
else:
    cagr = equity.iloc[-1] ** (ann / len(strategy_ret)) - 1

mdd = (equity / equity.cummax() - 1).min()

print(f"CAGR   {cagr:.2%}")
print(f"Sharpe {sharpe:.2f}")
print(f"MaxDD  {mdd:.2%}")

# ---------- plots ----------
fig_path = Path(__file__).parent.parent / "fig"
fig_path.mkdir(exist_ok=True)

# Equity curve
plt.figure(figsize=(10, 6))
equity.plot(title="Equity Curve")
plt.grid(True, alpha=0.3)
plt.savefig(fig_path / "equity.png", dpi=300, bbox_inches='tight')
plt.close()

# Trade return histogram (only if we have trades)
trade_returns = strategy_ret[strategy_ret != 0]
if len(trade_returns) > 0:
    plt.figure(figsize=(10, 6))
    trade_returns.hist(bins=40)
    plt.title("Trade Return Histogram")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_path / "hist.png", dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("Warning: No trades to plot in histogram")

# Drawdown plot
plt.figure(figsize=(10, 6))
drawdown = (equity / equity.cummax()) - 1
drawdown.plot(title="Drawdown")
plt.ylabel("Drawdown")
plt.grid(True, alpha=0.3)
plt.savefig(fig_path / "drawdown.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Plots saved to: {fig_path}")
