import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker="AAPL", period="5y"):
    df = yf.download(ticker, period=period)
    df.dropna(inplace=True)
    return df

def add_features(df):
    close = df["Close"]

    # Technical indicators
    df["RSI"] = RSIIndicator(close, window=14).rsi()
    df["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()

    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # Price-based features
    df["Returns"] = close.pct_change()
    df["Volatility"] = df["Returns"].rolling(10).std()

    df.dropna(inplace=True)
    return df

def prepare_sequences(df, feature_cols, target_col="Close", seq_len=60):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[feature_cols])
    target_scaler = MinMaxScaler()
    target = target_scaler.fit_transform(df[[target_col]])

    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y.append(target[i])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    return (X[:split], y[:split],
            X[split:],  y[split:],
            target_scaler)