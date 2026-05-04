import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def make_synthetic_ohlcv(n=600, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    spread = rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0, spread, n)
    low = np.minimum(open_, close) - rng.uniform(0, spread, n)
    volume = rng.uniform(1e6, 5e6, n)
    amount = volume * close
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamps": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    })


def plot_prediction(kline_df, pred_df, lookback, pred_len):
    pred_df.index = kline_df.index[lookback: lookback + pred_len]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(kline_df["close"], label="Ground Truth", color="blue", linewidth=1.2)
    ax1.plot(pred_df["close"], label="Prediction", color="red", linewidth=1.2)
    ax1.set_ylabel("Close Price")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(kline_df["volume"], label="Ground Truth", color="blue", linewidth=1.2)
    ax2.plot(pred_df["volume"], label="Prediction", color="red", linewidth=1.2)
    ax2.set_ylabel("Volume")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("mini_prediction.png", dpi=120)
    print("Plot saved to examples/mini_prediction.png")
    plt.show()


# 1. Load Kronos-mini (4.1M params, 2048 context)
print("Loading Kronos-mini...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
predictor = KronosPredictor(model, tokenizer, max_context=2048)

# 2. Synthetic data
df = make_synthetic_ohlcv(n=600)
lookback = 400
pred_len = 60

x_df = df.loc[:lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df.loc[:lookback - 1, "timestamps"]
y_timestamp = df.loc[lookback: lookback + pred_len - 1, "timestamps"]

# 3. Predict
print(f"Predicting {pred_len} steps from {lookback} context bars...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
)

print("\nForecast head:")
print(pred_df.head())

# 4. Plot
kline_df = df.loc[:lookback + pred_len - 1].set_index("timestamps")
plot_prediction(kline_df, pred_df, lookback, pred_len)
