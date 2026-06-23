import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from model import StockLSTM
from train import FEATURE_COLS, SEQ_LEN, TICKER
from data import fetch_data, add_features, prepare_sequences
import math

def evaluate():
    df = fetch_data(TICKER)
    df = add_features(df)
    _, _, X_test, y_test, scaler = prepare_sequences(df, FEATURE_COLS, seq_len=SEQ_LEN)

    model = StockLSTM(input_size=len(FEATURE_COLS))
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Inverse transform to real prices
    preds_real = scaler.inverse_transform(preds)
    actual_real = scaler.inverse_transform(y_test)

    rmse = math.sqrt(mean_squared_error(actual_real, preds_real))
    print(f"Test RMSE: ${rmse:.2f}")

    plt.figure(figsize=(12, 5))
    plt.plot(actual_real, label="Actual", color="steelblue")
    plt.plot(preds_real, label="Predicted", color="coral", linestyle="--")
    plt.title(f"{TICKER} — Predicted vs Actual Close Price")
    plt.xlabel("Trading days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_chart.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    evaluate()