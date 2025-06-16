import yfinance as yf

TICKER = "^GDAXI"
df = yf.download(
        tickers=TICKER,
        start="2020-01-01",
        interval="1d",          # ← change here
        auto_adjust=False       # silence future-warning
)
df.to_csv("data/dax_daily.csv")
print("Saved", len(df), "rows → data/dax_daily.csv")