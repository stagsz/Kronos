import yfinance as yf
import pandas as pd
import os

def download(ticker="AAPL", period="2y", interval="1d", save_dir="./data"):
    print(f"Downloading {ticker} ({interval}, {period})...")
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if raw.empty:
        print("No data returned — check the ticker symbol.")
        return None

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame()
    df["timestamps"] = raw.index
    df["open"]   = raw["Open"].values
    df["high"]   = raw["High"].values
    df["low"]    = raw["Low"].values
    df["close"]  = raw["Close"].values
    df["volume"] = raw["Volume"].values
    df["amount"] = df["volume"] * df["close"]  # yfinance has no turnover amount, approximate it

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"{ticker}_{interval}.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
    print(df.tail(3).to_string(index=False))
    return out

if __name__ == "__main__":
    download(ticker="AAPL", period="2y", interval="1d")
