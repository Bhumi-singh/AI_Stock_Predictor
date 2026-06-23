# 📈 AI Stock Predictor
 
An LSTM-based deep learning application that predicts stock closing prices using historical market data and technical indicators. Built with PyTorch and deployed as an interactive Streamlit web app.
 
🔗 **Live Demo:** [aistockpredictor-xshbdj538sgsxefvc5oiaf.streamlit.app](https://aistockpredictor-xshbdj538sgsxefvc5oiaf.streamlit.app/)
 
---
 
## 🧠 How It Works
 
The model fetches 5 years of historical OHLCV data via `yfinance`, engineers technical indicators, and trains a two-layer LSTM network to predict the next closing price from a 60-day sliding window of features.
 
**Prediction pipeline:**
 
```
Raw OHLCV data → Feature engineering → MinMax scaling → Sequence windows → LSTM → Predicted close price
```
 
---
 
## ✨ Features
 
- **LSTM neural network** with 2 layers, hidden size 128, and dropout regularization
- **9 input features:** Close, Volume, RSI-14, SMA-20, SMA-50, MACD, MACD Signal, Daily Returns, 10-day Volatility
- **60-day sequence length** for temporal context
- **80/20 train-test split** with MinMaxScaler normalization
- **Evaluation metrics:** RMSE, MAE, MAPE, and R²
- **Visualization:** Predicted vs. actual price chart + residuals bar chart + training loss curve
- **Utilities:** GPU/MPS/CPU auto-detection, reproducible seeding, checkpoint save/load, early stopping
- **Interactive Streamlit UI** for live predictions on any ticker
---
 
## 🗂️ Project Structure
 
```
AI_Stock_Predictor/
├── data.py            # Data fetching, feature engineering, sequence preparation
├── model.py           # StockLSTM architecture (PyTorch)
├── train.py           # Training loop, hyperparameters, model saving
├── predict.py         # Model evaluation and chart generation
├── utils.py           # Metrics, plotting, checkpointing, early stopping, device utils
├── model.pth          # Pre-trained model weights
└── prediction_chart.png  # Sample prediction output
```
 
---
 
## ⚙️ Model Architecture
 
```
Input (60 timesteps × 9 features)
    ↓
LSTM (hidden=128, layers=2, dropout=0.2)
    ↓
Linear(128 → 64) → ReLU → Dropout(0.2)
    ↓
Linear(64 → 1)   → Predicted close price
```
 
**Training config (from `train.py`):**
 
| Parameter    | Value  |
|-------------|--------|
| Ticker       | AAPL   |
| Sequence len | 60     |
| Epochs       | 50     |
| Batch size   | 32     |
| Learning rate| 1e-3   |
| Optimizer    | Adam   |
| Scheduler    | ReduceLROnPlateau (patience=5) |
| Loss         | MSELoss |
 
---
 
## 🚀 Getting Started
 
### Prerequisites
 
- Python 3.8+
- pip
### Installation
 
```bash
git clone https://github.com/Bhumi-singh/AI_Stock_Predictor.git
cd AI_Stock_Predictor
pip install torch yfinance pandas numpy scikit-learn ta matplotlib streamlit
```
 
### Train the model
 
```bash
python train.py
```
 
This fetches 5 years of AAPL data, trains for 50 epochs, and saves weights to `model.pth`.
 
### Evaluate & generate charts
 
```bash
python predict.py
```
 
Prints test RMSE and saves `prediction_chart.png` with actual vs. predicted prices.
 
### Run the Streamlit app locally
 
```bash
streamlit run app.py
```
 
---
 
## 📊 Input Features
 
| Feature       | Description                          |
|--------------|--------------------------------------|
| Close        | Adjusted closing price               |
| Volume       | Daily trading volume                 |
| RSI          | Relative Strength Index (14-day)     |
| SMA_20       | Simple Moving Average (20-day)       |
| SMA_50       | Simple Moving Average (50-day)       |
| MACD         | MACD line                            |
| MACD_signal  | MACD signal line                     |
| Returns      | Daily percentage price change        |
| Volatility   | 10-day rolling standard deviation    |
 
---
 
## 🛠️ Utility Highlights (`utils.py`)
 
- **`set_seed()`** — Fixes Python, NumPy, and PyTorch seeds for reproducible runs
- **`get_device()`** — Auto-selects CUDA → MPS (Apple Silicon) → CPU
- **`save_checkpoint()` / `load_checkpoint()`** — Full training state persistence
- **`evaluate_metrics()`** — Computes RMSE, MAE, MAPE, and R² in one call
- **`EarlyStopping`** — Stops training when validation loss plateaus
- **`plot_predictions()`** — Two-panel chart: price overlay + residuals
- **`plot_training_loss()`** — Train/val loss curves over epochs
---
 
## ⚠️ Disclaimer
 
This project is for **educational purposes only**. Stock price predictions made by this model should not be used as financial advice. Past performance is not indicative of future results.
 
---
 
## 👤 Author
 
**Bhumi Singh** — [GitHub](https://github.com/Bhumi-singh)
