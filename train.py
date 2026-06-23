import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data import fetch_data, add_features, prepare_sequences
from model import StockLSTM
import pandas as pd

TICKER = "AAPL"
SEQ_LEN = 60
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3



FEATURE_COLS = ["Close", "Volume", "RSI", "SMA_20", "SMA_50",
                "MACD", "MACD_signal", "Returns", "Volatility"]

def train():
    df = fetch_data(TICKER)

    print(df.columns)
    print(type(df["Close"]))
    print(df["Close"].shape)
    
    df = add_features(df)
    X_train, y_train, X_test, y_test, scaler = prepare_sequences(
        df, FEATURE_COLS, seq_len=SEQ_LEN
    )

    # Convert to tensors
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = StockLSTM(input_size=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved.")
    return model, X_test, y_test, scaler

if __name__ == "__main__":
    train()