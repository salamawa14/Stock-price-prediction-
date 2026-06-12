فكرة ممتازة جداً! كتابة التعليقات باللغة الإنجليزية هو المعيار الذهبي في البرمجة، فهو يجعل الكود الخاص بك احترافياً وجاهزاً للمشاركة على منصات مثل GitHub ليقرأه أي مبرمج حول العالم.

لقد قمت بتحويل جميع التعليقات العربية إلى لغة إنجليزية واضحة وموجزة تشرح وظيفة الكود بسلاسة:

```python
# ================================
# Stock LSTM + Streamlit (PyTorch)
# Windows Compatible + SSL Fix + RMSE fallback + Data Leakage Fix
# ================================
import os
import ssl

# --- SSL Fix for all OS ---
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# ================================
# Imports
# ================================
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# ================================
# Page Config (Best practice: place at the top for a professional layout)
# ================================
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")

# ================================
# Helpers
# ================================
def rmse_compat(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def mape_np(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))

# ================================
# Features
# ================================
def add_features(df):
    df = df.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()
    df["TR"] = (df["High"] - df["Low"]).abs()
    df["ATR_14"] = df["TR"].rolling(14).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["Vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-8)
    df = df.dropna()
    return df

# ================================
# Sequence maker
# ================================
def create_sequences(array2d, seq_len, target_col_idx):
    X, y = [], []
    for i in range(len(array2d) - seq_len):
        X.append(array2d[i:i+seq_len])
        y.append(array2d[i+seq_len, target_col_idx])
    return np.array(X), np.array(y).reshape(-1, 1)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ================================
# LSTM Model
# ================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ================================
# Streamlit UI
# ================================
st.title("Stock Price Predictor (Deep Learning)")
st.write("Train a custom LSTM model to forecast the next day's closing price based on historical data.")

st.divider()

# Input section
colA, colB, colC = st.columns(3)
with colA:
    ticker_input = st.text_input("Stock Ticker", "AAPL").upper().strip()
with colB:
    start_date = st.text_input("Start Date", "2018-01-01")
with colC:
    end_date = st.text_input("End Date", "2023-01-01")

# Hide complex settings in an expander for a cleaner UI
with st.expander("Advanced Model Parameters"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        seq_len = st.slider("Sequence Length", 20, 120, 60, step=5)
    with c2:
        epochs = st.slider("Epochs", 5, 40, 15, step=1)
    with c3:
        hidden_size = st.selectbox("Hidden Size", [32, 64, 96, 128], index=1)
    with c4:
        num_layers = st.selectbox("LSTM Layers", [1, 2, 3], index=1)

    c5, c6, c7 = st.columns(3)
    with c5:
        lr = st.selectbox("Learning Rate", [1e-4, 3e-4, 1e-3, 3e-3], index=2)
    with c6:
        batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
    with c7:
        use_extra = st.checkbox("Enable Technical Indicators", value=True)

# Slight spacing before the button
st.write("")
run_clicked = st.button("Start Training", type="primary")

if run_clicked:
    # 1) Download data
    try:
        df = yf.download(ticker_input, start=start_date, end=end_date)
    except Exception as e:
        st.error(f"Error fetching data for {ticker_input}: {e}")
        st.stop()
        
    if df is None or df.empty:
        st.warning("No data found. Please verify the stock ticker and date range.")
        st.stop()

    # 2) Prepare features
    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[base_cols].dropna()
    
    if use_extra:
        df = add_features(df)
        feature_cols = ["Open","High","Low","Close","Volume","Return_1d","MA_7","MA_21","ATR_14","RSI_14","Vol_z"]
    else:
        feature_cols = ["Close"]
        
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    data = df[feature_cols].values.astype(float)
    feature_names = feature_cols
    close_idx = feature_names.index("Close")

    # 3) Split data before scaling
    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data  = data[split:]

    # 4) Scaling
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled  = scaler.transform(test_data)

    # 5) Create sequences
    X_train, y_train = create_sequences(train_scaled, seq_len, target_col_idx=close_idx)
    X_test, y_test   = create_sequences(test_scaled, seq_len, target_col_idx=close_idx)

    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Sequence length is too large for the provided dataset.")
        st.stop()

    # 6) DataLoaders
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(StockDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 7) Model + Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_size=len(feature_names), hidden_size=hidden_size, num_layers=num_layers, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training Loop
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses))
        loss_history.append(avg_loss)
        
        progress_bar.progress(epoch/epochs)
        status_text.text(f"Training in progress... Epoch {epoch}/{epochs} (Loss: {avg_loss:.6f})")
        
    progress_bar.empty()
    status_text.empty()

    # 8) Predictions
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            Xb = Xb.to(device)
            pr = model(Xb).cpu().numpy()
            preds_scaled.append(pr)
    preds_scaled = np.vstack(preds_scaled).reshape(-1,1)

    def invert_close_only(scaled_col_1d):
        tmp = np.zeros((scaled_col_1d.shape[0], len(feature_names)))
        tmp[:, close_idx] = scaled_col_1d.reshape(-1)
        inv = scaler.inverse_transform(tmp)[:, close_idx]
        return inv

    y_test_inv = invert_close_only(y_test.reshape(-1))
    preds_inv  = invert_close_only(preds_scaled.reshape(-1))

    # 9) Metrics (Using professional metric cards instead of plain text)
    st.subheader("Model Performance")
    r2  = r2_score(y_test_inv, preds_inv)
    rmse = rmse_compat(y_test_inv, preds_inv)
    mape = mape_np(y_test_inv, preds_inv)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2*100:.2f}%")
    m2.metric("RMSE", f"{rmse:.2f}")
    m3.metric("MAPE", f"{mape:.2%}")

    st.divider()

    # 10) Plot: Real vs Predicted (Enhanced chart styling for better readability)
    st.subheader(f"{ticker_input} - Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(y_test_inv)), y_test_inv, label="Actual Price", color="#1f77b4", linewidth=1.5)
    ax.plot(range(len(preds_inv)), preds_inv, label="Predicted Price", color="#ff7f0e", linewidth=1.5, linestyle="--")
    
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time Steps (Days)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="upper left")
    
    st.pyplot(fig)

    # Next-day prediction (Polished result display)
    last_window = test_scaled[-seq_len:]
    last_window_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        next_day_pred_scaled = model(last_window_tensor).cpu().numpy()
    
    next_day_pred = invert_close_only(next_day_pred_scaled.reshape(-1))[0]
    
    st.info(f"**Forecast:** The predicted closing price for the next trading day is **${next_day_pred:.2f}**")

```