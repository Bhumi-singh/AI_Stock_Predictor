"""
AI Stock Predictor — Streamlit Dashboard
Self-contained: fetches data, trains LSTM on-the-fly, shows predictions.
Deploy on Streamlit Cloud with requirements.txt and this file.
"""
 
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1f33 100%);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px;
        border-left: 4px solid #00d4aa;
    }
    .metric-label { color: #8ab4d4; font-size: 13px; font-weight: 500; margin-bottom: 4px; }
    .metric-value { color: #ffffff; font-size: 26px; font-weight: 700; }
    .metric-delta { font-size: 12px; margin-top: 4px; }
    .positive { color: #00d4aa; }
    .negative { color: #ff6b6b; }
    .stProgress > div > div { background-color: #00d4aa; }
    h1 { color: #ffffff; }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #8ab4d4;
        margin: 16px 0 8px 0;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# LSTM Model
# ─────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
 
    def forward(self, x):
        out, _ = self.lstm(x)                          # (B, T, H)
        attn_w = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
        context = (attn_w * out).sum(dim=1)            # (B, H)
        return self.fc(context)
 
 
# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df
 
 
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
 
    # Moving averages
    d["SMA_10"] = c.rolling(10).mean()
    d["SMA_20"] = c.rolling(20).mean()
    d["EMA_12"] = c.ewm(span=12).mean()
    d["EMA_26"] = c.ewm(span=26).mean()
 
    # MACD
    d["MACD"] = d["EMA_12"] - d["EMA_26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9).mean()
 
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI"] = 100 - 100 / (1 + rs)
 
    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d["BB_upper"] = sma20 + 2 * std20
    d["BB_lower"] = sma20 - 2 * std20
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (sma20 + 1e-9)
 
    # OBV
    obv = (np.sign(c.diff()) * d["Volume"]).fillna(0).cumsum()
    d["OBV"] = obv / 1e6   # scale
 
    # Daily return & volatility
    d["Return"] = c.pct_change()
    d["Volatility"] = d["Return"].rolling(10).std()
 
    return d.dropna()
 
 
def prepare_sequences(df: pd.DataFrame, feature_cols: list, target_col: str,
                      seq_len: int, train_ratio: float):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
 
    n = len(df)
    split = int(n * train_ratio)
 
    X_all = scaler_X.fit_transform(df[feature_cols].values)
    y_all = scaler_y.fit_transform(df[[target_col]].values)
 
    def make_seqs(arr_X, arr_y):
        Xs, ys = [], []
        for i in range(len(arr_X) - seq_len):
            Xs.append(arr_X[i: i + seq_len])
            ys.append(arr_y[i + seq_len])
        return np.array(Xs), np.array(ys)
 
    X_train_raw = X_all[:split]
    y_train_raw = y_all[:split]
    X_test_raw  = X_all[split:]
    y_test_raw  = y_all[split:]
 
    X_train, y_train = make_seqs(X_train_raw, y_train_raw)
    X_test,  y_test  = make_seqs(X_test_raw,  y_test_raw)
 
    return X_train, y_train, X_test, y_test, scaler_y, split
 
 
# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val,
                hidden_size, num_layers, dropout,
                epochs, lr, batch_size, patience,
                progress_bar, status_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)
 
    input_size = Xt.shape[2]
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()
 
    best_val = float("inf")
    best_weights = None
    no_improve = 0
    train_losses, val_losses = [], []
 
    dataset = torch.utils.data.TensorDataset(Xt, yt)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
 
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
 
        scheduler.step(val_loss)
        train_losses.append(ep_loss)
        val_losses.append(val_loss)
 
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                status_text.text(f"Early stopping at epoch {epoch}")
                break
 
        progress_bar.progress(epoch / epochs)
        if epoch % 10 == 0 or epoch == 1:
            status_text.text(
                f"Epoch {epoch}/{epochs} — Train Loss: {ep_loss:.5f} | Val Loss: {val_loss:.5f}"
            )
 
    model.load_state_dict(best_weights)
    return model.cpu(), train_losses, val_losses
 
 
def predict(model, X, scaler_y):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X)).numpy()
    return scaler_y.inverse_transform(preds).flatten()
 
 
# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────
def plot_predictions(dates, actual, train_pred, test_pred, split_idx, seq_len, ticker):
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        name="Actual Price",
        line=dict(color="#8ab4d4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates[seq_len:split_idx],
        y=train_pred,
        name="Train Prediction",
        line=dict(color="#00d4aa", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=dates[split_idx + seq_len:],
        y=test_pred,
        name="Test Prediction",
        line=dict(color="#ff9f43", width=2),
    ))
    fig.add_vrect(
        x0=dates[split_idx], x1=dates[-1],
        fillcolor="rgba(255,159,67,0.05)",
        line_width=0,
        annotation_text="Test Period", annotation_position="top left",
        annotation_font_color="#ff9f43",
    )
    fig.update_layout(
        title=f"{ticker} — Predicted vs Actual Close Price",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
 
 
def plot_loss(train_losses, val_losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, name="Train Loss",
                             line=dict(color="#00d4aa", width=2)))
    fig.add_trace(go.Scatter(y=val_losses, name="Val Loss",
                             line=dict(color="#ff9f43", width=2)))
    fig.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch", yaxis_title="MSE Loss",
        template="plotly_dark", height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
 
 
def plot_candlestick(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC", increasing_line_color="#00d4aa",
        decreasing_line_color="#ff6b6b",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color="#8ab4d4", opacity=0.5,
    ), row=2, col=1)
    fig.update_layout(
        title=f"{ticker} — Candlestick Chart",
        template="plotly_dark", height=450,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
 
 
def plot_indicators(df, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.05,
                        subplot_titles=["Price + Bollinger Bands", "MACD", "RSI"])
 
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"],
                             name="Close", line=dict(color="#8ab4d4", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"],
                             name="BB Upper", line=dict(color="#ff9f43", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"],
                             name="BB Lower", line=dict(color="#ff9f43", width=1, dash="dash"),
                             fill="tonexty", fillcolor="rgba(255,159,67,0.05)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"],
                             name="SMA 20", line=dict(color="#00d4aa", width=1, dash="dot")), row=1, col=1)
 
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                             name="MACD", line=dict(color="#00d4aa", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
                             name="Signal", line=dict(color="#ff6b6b", width=1.5)), row=2, col=1)
    fig.add_hline(y=0, line_color="#555", row=2, col=1)
 
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
                             name="RSI", line=dict(color="#ff9f43", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", annotation_text="Overbought", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa", annotation_text="Oversold", row=3, col=1)
 
    fig.update_layout(
        template="plotly_dark", height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=80, b=40),
    )
    return fig
 
 
def metric_card(label, value, delta=None, delta_positive=True):
    delta_html = ""
    if delta is not None:
        cls = "positive" if delta_positive else "negative"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""
 
 
# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 AI Stock Predictor")
    st.markdown("---")
 
    ticker = st.text_input("Ticker Symbol", value="AAPL",
                           placeholder="e.g. AAPL, TSLA, MSFT").upper().strip()
 
    period = st.selectbox("Data Period",
                          ["1y", "2y", "3y", "5y"],
                          index=1)
 
    st.markdown("---")
    st.markdown("#### Model Hyperparameters")
 
    seq_len     = st.slider("Sequence Length (days)", 10, 90, 30, 5)
    hidden_size = st.select_slider("Hidden Size", [64, 128, 256], value=128)
    num_layers  = st.slider("LSTM Layers", 1, 3, 2)
    dropout     = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    epochs      = st.slider("Max Epochs", 20, 200, 80, 10)
    lr          = st.select_slider("Learning Rate", [0.0005, 0.001, 0.002, 0.005], value=0.001)
    batch_size  = st.select_slider("Batch Size", [16, 32, 64], value=32)
    patience    = st.slider("Early Stopping Patience", 5, 30, 10)
    train_ratio = st.slider("Train Split", 0.6, 0.9, 0.8, 0.05)
 
    st.markdown("---")
    run_btn = st.button("🚀 Train & Predict", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")
 
 
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
st.title("📈 AI Stock Price Predictor")
st.caption("LSTM with Attention · Technical Indicators · Live yFinance Data")
 
tab1, tab2, tab3, tab4 = st.tabs(
    ["🤖 Prediction", "📊 Technical Analysis", "📉 Model Training", "ℹ️ About"]
)
 
if run_btn:
    # ── Fetch ──────────────────────────────────
    with st.spinner(f"Fetching {ticker} data…"):
        raw_df = fetch_data(ticker, period)
 
    if raw_df.empty:
        st.error(f"❌ No data found for **{ticker}**. Check the ticker symbol and try again.")
        st.stop()
 
    # ── Feature engineering ────────────────────
    df = add_features(raw_df)
 
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_20", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "RSI",
        "BB_upper", "BB_lower", "BB_width", "OBV",
        "Return", "Volatility",
    ]
 
    X_train, y_train, X_test, y_test, scaler_y, split_idx = prepare_sequences(
        df, feature_cols, "Close", seq_len, train_ratio
    )
 
    # val split from train
    val_cut  = int(len(X_train) * 0.9)
    X_val    = X_train[val_cut:]
    y_val    = y_train[val_cut:]
    X_train  = X_train[:val_cut]
    y_train  = y_train[:val_cut]
 
    # ── Train ──────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Training Progress</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text  = st.empty()
 
    model, train_losses, val_losses = train_model(
        X_train, y_train, X_val, y_val,
        hidden_size, num_layers, dropout,
        epochs, lr, batch_size, patience,
        progress_bar, status_text,
    )
 
    # ── Predict ────────────────────────────────
    train_pred = predict(model, X_train, scaler_y)
    val_pred   = predict(model, X_val,   scaler_y)
    test_pred  = predict(model, X_test,  scaler_y)
 
    actual_close = df["Close"].values
    dates        = df.index
 
    # Train actual (for metrics — use val+train together)
    all_train_pred = np.concatenate([train_pred, val_pred])
    all_train_act  = scaler_y.inverse_transform(
        np.concatenate([y_train, y_val])
    ).flatten()
 
    test_actual = scaler_y.inverse_transform(y_test).flatten()
 
    # ── Metrics ────────────────────────────────
    def compute_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae  = mean_absolute_error(actual, pred)
        r2   = r2_score(actual, pred)
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-9))) * 100
        return rmse, mae, r2, mape
 
    tr_rmse, tr_mae, tr_r2, tr_mape = compute_metrics(all_train_act, all_train_pred)
    te_rmse, te_mae, te_r2, te_mape = compute_metrics(test_actual,   test_pred)
 
    # Next-day forecast
    last_seq   = df[feature_cols].values[-seq_len:]
    last_scaled = MinMaxScaler().fit(df[feature_cols].values).transform(last_seq)
    # Use correct scaler (refit for X only — quick approximation for display)
    from sklearn.preprocessing import MinMaxScaler as MMS
    sx = MMS().fit(df[feature_cols].values)
    last_scaled = sx.transform(last_seq)
    last_tensor = torch.FloatTensor(last_scaled).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        next_scaled = model(last_tensor).numpy()
    next_price = scaler_y.inverse_transform(next_scaled)[0][0]
    last_price = float(df["Close"].iloc[-1])
    price_change = next_price - last_price
    pct_change   = price_change / last_price * 100
 
    # ── Store in session ───────────────────────
    st.session_state["results"] = {
        "df": df, "raw_df": raw_df,
        "dates": dates, "actual_close": actual_close,
        "train_pred": all_train_pred, "test_pred": test_pred,
        "test_actual": test_actual,
        "split_idx": split_idx,
        "train_losses": train_losses, "val_losses": val_losses,
        "metrics": {
            "tr_rmse": tr_rmse, "tr_mae": tr_mae, "tr_r2": tr_r2, "tr_mape": tr_mape,
            "te_rmse": te_rmse, "te_mae": te_mae, "te_r2": te_r2, "te_mape": te_mape,
        },
        "next_price": next_price,
        "last_price": last_price,
        "price_change": price_change,
        "pct_change": pct_change,
        "ticker": ticker,
        "seq_len": seq_len,
    }
 
# ─────────────────────────────────────────────
# Render results
# ─────────────────────────────────────────────
if "results" in st.session_state:
    r = st.session_state["results"]
 
    with tab1:
        # ── KPI row ──────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card(
                "Last Close Price",
                f"${r['last_price']:.2f}",
            ), unsafe_allow_html=True)
        with c2:
            pos = r["price_change"] >= 0
            st.markdown(metric_card(
                "Predicted Next Close",
                f"${r['next_price']:.2f}",
                delta=f"{r['price_change']:+.2f} ({r['pct_change']:+.2f}%)",
                delta_positive=pos,
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card(
                "Test RMSE",
                f"${r['metrics']['te_rmse']:.2f}",
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card(
                "Test R²",
                f"{r['metrics']['te_r2']:.4f}",
                delta=f"MAE ${r['metrics']['te_mae']:.2f}",
                delta_positive=r['metrics']['te_r2'] > 0.9,
            ), unsafe_allow_html=True)
 
        st.plotly_chart(
            plot_predictions(
                r["dates"], r["actual_close"],
                r["train_pred"], r["test_pred"],
                r["split_idx"], r["seq_len"], r["ticker"],
            ),
            use_container_width=True,
        )
 
        # ── Error distribution ────────────────────
        errors = r["test_actual"] - r["test_pred"]
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=errors, nbinsx=40,
            marker_color="#00d4aa", opacity=0.75,
            name="Prediction Error",
        ))
        fig_err.update_layout(
            title="Test Set Prediction Error Distribution",
            xaxis_title="Error (USD)", yaxis_title="Count",
            template="plotly_dark", height=280,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_err, use_container_width=True)
 
        # ── Full metrics table ────────────────────
        st.markdown('<div class="section-header">Full Metrics</div>', unsafe_allow_html=True)
        m = r["metrics"]
        metrics_df = pd.DataFrame({
            "Metric": ["RMSE", "MAE", "R²", "MAPE (%)"],
            "Train": [f"{m['tr_rmse']:.4f}", f"{m['tr_mae']:.4f}",
                      f"{m['tr_r2']:.4f}", f"{m['tr_mape']:.2f}%"],
            "Test":  [f"{m['te_rmse']:.4f}", f"{m['te_mae']:.4f}",
                      f"{m['te_r2']:.4f}", f"{m['te_mape']:.2f}%"],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
 
        # ── Download predictions ──────────────────
        n_test = len(r["test_pred"])
        result_df = pd.DataFrame({
            "Date":   r["dates"][-n_test:],
            "Actual": r["test_actual"],
            "Predicted": r["test_pred"],
            "Error":  r["test_actual"] - r["test_pred"],
        })
        csv_buf = io.StringIO()
        result_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Test Predictions (CSV)",
            csv_buf.getvalue(),
            file_name=f"{r['ticker']}_predictions.csv",
            mime="text/csv",
        )
 
    with tab2:
        st.plotly_chart(
            plot_candlestick(r["raw_df"], r["ticker"]),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_indicators(r["df"], r["ticker"]),
            use_container_width=True,
        )
 
        # Raw data
        with st.expander("📄 Raw Data"):
            st.dataframe(r["raw_df"].tail(60), use_container_width=True)
 
    with tab3:
        st.plotly_chart(
            plot_loss(r["train_losses"], r["val_losses"]),
            use_container_width=True,
        )
 
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Training Summary**
            - Epochs run: `{len(r['train_losses'])}`
            - Best val loss: `{min(r['val_losses']):.6f}`
            - Final train loss: `{r['train_losses'][-1]:.6f}`
            """)
        with col2:
            st.markdown(f"""
            **Architecture**
            - Input features: `{len([
                'Open','High','Low','Close','Volume','SMA_10','SMA_20',
                'EMA_12','EMA_26','MACD','MACD_Signal','RSI',
                'BB_upper','BB_lower','BB_width','OBV','Return','Volatility'
            ])}`
            - Hidden size: `{r.get('hidden_size', hidden_size)}`
            - Layers: `{r.get('num_layers', num_layers)}`
            - Sequence length: `{r['seq_len']} days`
            """)
 
else:
    with tab1:
        st.info("👈 Configure settings in the sidebar and click **Train & Predict** to begin.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🧠 Model**
            - LSTM with Attention
            - Gradient clipping
            - Early stopping
            - LR scheduling
            """)
        with col2:
            st.markdown("""
            **📊 Features (18)**
            - OHLCV
            - SMA / EMA
            - MACD + Signal
            - RSI, Bollinger Bands
            - OBV, Volatility
            """)
        with col3:
            st.markdown("""
            **📈 Outputs**
            - Next-day price forecast
            - RMSE / MAE / R² / MAPE
            - Candlestick + indicators
            - CSV download
            """)
 
with tab4:
    st.markdown("""
    ## About This App
 
    **AI Stock Predictor** is an educational demonstration of using deep learning to model
    historical stock prices. It fetches live data from Yahoo Finance, engineers 18 technical
    features, trains an LSTM + Attention model in-browser, and displays predictions with
    full metrics.
 
    ### ⚠️ Disclaimer
    This tool is **for educational and research purposes only**.  
    Stock prices are inherently unpredictable. Do **not** use these predictions for actual
    trading or investment decisions. Past performance does not guarantee future results.
 
    ### 🛠️ Tech Stack
    | Component | Library |
    |-----------|---------|
    | UI | Streamlit |
    | Data | yfinance |
    | Model | PyTorch (LSTM + Attention) |
    | Features | pandas, numpy |
    | Scaling | scikit-learn |
    | Charts | Plotly |
 
    ### 📁 Project Structure
    ```
    AI_Stock_Predictor/
    ├── app.py              ← This file (Streamlit dashboard)
    ├── requirements.txt    ← Dependencies
    └── .streamlit/
        └── config.toml     ← Theme config
    ```
 
    ### 🚀 How to Run Locally
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)