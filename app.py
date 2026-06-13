# ================================================
# AlphaLSTM — Stock Price Predictor (Final)
# Fix: Rolling-window normalization (no global scaler)
# ================================================
import os, ssl
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error

# ── Page config ──
st.set_page_config(page_title="AlphaLSTM", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ──
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0d1117;color:#e6edf3}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #21262d}
[data-testid="stSidebar"] *{color:#c9d1d9!important}
.hero{background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #21262d;
  border-radius:12px;padding:28px 36px;margin-bottom:20px}
.ticker{font-family:'Courier New',monospace;font-size:3rem;font-weight:900;
  letter-spacing:-2px;color:#58a6ff;line-height:1}
.sub{font-size:.8rem;color:#8b949e;text-transform:uppercase;letter-spacing:2px;margin-top:4px}
.mrow{display:flex;gap:14px;margin-bottom:20px}
.mc{flex:1;background:#161b22;border:1px solid #21262d;border-radius:10px;
  padding:18px 22px;position:relative;overflow:hidden}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.mc.g::before{background:linear-gradient(90deg,#2ea043,#3fb950)}
.mc.b::before{background:linear-gradient(90deg,#1f6feb,#58a6ff)}
.mc.o::before{background:linear-gradient(90deg,#9e6a03,#d29922)}
.mc.p::before{background:linear-gradient(90deg,#6e40c9,#a371f7)}
.ml{font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;color:#8b949e;margin-bottom:5px}
.mv{font-family:'Courier New',monospace;font-size:1.65rem;font-weight:700;color:#e6edf3;line-height:1}
.ms{font-size:.72rem;color:#6e7681;margin-top:3px}
.sec{font-size:.7rem;text-transform:uppercase;letter-spacing:2px;color:#58a6ff;
  margin-bottom:10px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:#21262d}
.fcast{background:linear-gradient(135deg,#0d2137,#0d1117);border:1px solid #1f6feb;
  border-radius:10px;padding:22px 30px;display:flex;align-items:center;
  justify-content:space-between;margin-top:22px}
.fp{font-family:'Courier New',monospace;font-size:2.6rem;font-weight:900;color:#3fb950}
.fl{font-size:.72rem;text-transform:uppercase;letter-spacing:2px;color:#58a6ff}
.fn{font-size:.76rem;color:#6e7681}
.tlog{background:#0d1117;border:1px solid #21262d;border-radius:8px;
  padding:12px 16px;font-family:'Courier New',monospace;font-size:.8rem;
  color:#3fb950;max-height:150px;overflow-y:auto}
.stButton>button{background:#1f6feb!important;color:#fff!important;border:none!important;
  border-radius:8px!important;font-weight:700!important;padding:.6rem 2rem!important}
.stButton>button:hover{background:#388bfd!important}
hr{border-color:#21262d!important}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════
def rmse_val(a, b):
    try:    return mean_squared_error(a, b, squared=False)
    except: return np.sqrt(mean_squared_error(a, b))

def mape_val(a, b, eps=1e-8):
    a, b = np.array(a).reshape(-1), np.array(b).reshape(-1)
    return np.mean(np.abs((a - b) / (np.abs(a) + eps)))

# ════════════════════════════════════════════
# FEATURE ENGINEERING  (only price/volume ratios — scale-invariant)
# ════════════════════════════════════════════
def build_features(df):
    df = df.copy()
    c  = df["Close"].squeeze()
    h  = df["High"].squeeze()
    lo = df["Low"].squeeze()
    v  = df["Volume"].squeeze()

    # Returns (scale-free)
    df["r1"]   = c.pct_change()
    df["r3"]   = c.pct_change(3)
    df["r5"]   = c.pct_change(5)

    # MA ratios (scale-free)
    df["ma7"]  = c / (c.rolling(7).mean()  + 1e-8) - 1
    df["ma21"] = c / (c.rolling(21).mean() + 1e-8) - 1
    df["ma50"] = c / (c.rolling(50).mean() + 1e-8) - 1

    # MACD ratio
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = (e12 - e26) / (c + 1e-8)

    # Bollinger %B (0-1 bounded)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pct"] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-8)

    # RSI (already 0-100, divide to 0-1)
    delta    = c.diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = (100 - 100/(1 + gain/(loss+1e-8))) / 100

    # ATR ratio (scale-free)
    atr       = (h - lo).rolling(14).mean()
    df["atr"] = atr / (c + 1e-8)

    # Volume z-score (clipped)
    vol_z     = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-8)
    df["volz"] = vol_z.clip(-3, 3) / 3

    # Candle body ratio
    df["body"] = (c - df["Open"].squeeze()) / (h - lo + 1e-8)

    df = df.dropna()
    return df

FEAT_COLS = ["r1","r3","r5","ma7","ma21","ma50","macd","bb_pct","rsi","atr","volz","body"]

# ════════════════════════════════════════════
# ROLLING-WINDOW NORMALIZATION  ← the key fix
# Each sample is self-normalized using its own window's close stats
# → no global scaler → no distribution shift between train/test
# ════════════════════════════════════════════
def make_sequences(close_arr, feat_arr, seq_len):
    """
    Returns:
      X      : (N, seq_len, n_features+1)  — normalized close + features
      y      : (N, 1)                       — normalized target (next close)
      mus    : (N,)   window close mean
      stds   : (N,)   window close std
    """
    X, y, mus, stds = [], [], [], []
    for i in range(len(close_arr) - seq_len):
        win_close = close_arr[i : i+seq_len]
        mu  = win_close.mean()
        std = win_close.std() + 1e-8

        # Normalize close window
        close_norm = (win_close - mu) / std                     # (seq_len,)

        # Features for that window (already scale-free, just clip)
        feat_win = feat_arr[i : i+seq_len]                      # (seq_len, F)

        # Stack: normalized close + scale-free features
        sample = np.column_stack([close_norm, feat_win])        # (seq_len, F+1)
        X.append(sample)

        # Target: next close normalized by same window stats
        y_val = (close_arr[i + seq_len] - mu) / std
        y.append(y_val)
        mus.append(mu)
        stds.append(std)

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32).reshape(-1,1),
            np.array(mus), np.array(stds))

# ════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════
class StockDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):    return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ════════════════════════════════════════════
# MODEL  — LSTM (clean, proven architecture)
# ════════════════════════════════════════════
class LSTMPredictor(nn.Module):
    def __init__(self, n_feat, hidden=128, n_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden, 64)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.norm(out[:, -1, :])
        h = self.drop(h)
        return self.fc2(self.act(self.fc1(h)))

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
# ════════════════════════════════════════════
# TICKER RESOLVER  — name → symbol
# ════════════════════════════════════════════
KNOWN_TICKERS = {
    # Tech
    "apple":"AAPL","microsoft":"MSFT","google":"GOOGL","alphabet":"GOOGL",
    "amazon":"AMZN","meta":"META","facebook":"META","nvidia":"NVDA",
    "tesla":"TSLA","netflix":"NFLX","adobe":"ADBE","salesforce":"CRM",
    "intel":"INTC","amd":"AMD","qualcomm":"QCOM","broadcom":"AVGO",
    "oracle":"ORCL","ibm":"IBM","cisco":"CSCO","paypal":"PYPL",
    "uber":"UBER","lyft":"LYFT","airbnb":"ABNB","spotify":"SPOT",
    "twitter":"TWTR","snapchat":"SNAP","pinterest":"PINS","zoom":"ZM",
    "shopify":"SHOP","square":"SQ","block":"SQ","palantir":"PLTR",
    "snowflake":"SNOW","datadog":"DDOG","cloudflare":"NET","twilio":"TWLO",
    # Finance
    "jpmorgan":"JPM","jp morgan":"JPM","goldman sachs":"GS","goldman":"GS",
    "morgan stanley":"MS","bank of america":"BAC","wells fargo":"WFC",
    "citigroup":"C","citi":"C","blackrock":"BLK","visa":"V","mastercard":"MA",
    "american express":"AXP","amex":"AXP",
    # Healthcare
    "johnson & johnson":"JNJ","johnson and johnson":"JNJ","pfizer":"PFE",
    "moderna":"MRNA","abbvie":"ABBV","merck":"MRK","eli lilly":"LLY","lilly":"LLY",
    "unitedhealth":"UNH","cvs":"CVS","walgreens":"WBA",
    # Consumer / Retail
    "walmart":"WMT","target":"TGT","costco":"COST","home depot":"HD",
    "mcdonalds":"MCD","starbucks":"SBUX","coca cola":"KO","pepsi":"PEP",
    "pepsico":"PEP","nike":"NKE","disney":"DIS","comcast":"CMCSA",
    "ford":"F","general motors":"GM","gm":"GM","boeing":"BA",
    # Energy
    "exxon":"XOM","exxonmobil":"XOM","chevron":"CVX","shell":"SHEL",
    "bp":"BP","conocophillips":"COP",
    # Saudi / Gulf (ADX/Tadawul listed as ADRs or regional)
    "aramco":"2222.SR","saudi aramco":"2222.SR","sabic":"2010.SR",
    "stc":"7010.SR","saudi telecom":"7010.SR","alrajhi":"1120.SR",
    "al rajhi":"1120.SR","jarir":"4190.SR","samba":"1090.SR",
}

def resolve_ticker(raw: str) -> tuple[str, str]:
    """
    Returns (ticker_symbol, display_name).
    Accepts: ticker symbol, company name, or partial name.
    """
    raw = raw.strip()
    key = raw.lower()

    # 1. Direct lookup
    if key in KNOWN_TICKERS:
        sym = KNOWN_TICKERS[key]
        return sym, raw.title()

    # 2. Partial match
    for name, sym in KNOWN_TICKERS.items():
        if key in name or name in key:
            return sym, name.title()

    # 3. Assume it's already a valid ticker symbol
    return raw.upper(), raw.upper()

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ AlphaLSTM")
    st.markdown('<div class="sec">Data</div>', unsafe_allow_html=True)

    raw_input = st.text_input(
        "Stock — ticker or company name",
        "AAPL",
        help="Enter a ticker symbol (e.g. AAPL) **or** the company name "
             "(e.g. Apple, Tesla, Saudi Aramco). "
             "The app will automatically find the correct symbol."
    )
    ticker, company_name = resolve_ticker(raw_input)

    # Show resolved ticker if input was a name
    if raw_input.upper() != ticker:
        st.caption(f"✅ Resolved → **{ticker}**")

    c1, c2 = st.columns(2)
    with c1:
        start = st.text_input(
            "From", "2016-01-01",
            help="Start date for historical data. Format: YYYY-MM-DD. "
                 "More data = better training. Minimum ~2 years recommended."
        )
    with c2:
        end = st.text_input(
            "To", "2024-01-01",
            help="End date. The last 14% of this range is used as the "
                 "test set the model never sees during training."
        )

    st.markdown('<div class="sec" style="margin-top:16px">Model</div>', unsafe_allow_html=True)
    seq_len = st.slider(
        "Lookback window (days)", 20, 90, 40, 5,
        help="How many past trading days the model reads before making a prediction. "
             "Think of it as the model's memory. "
             "Larger = more context but slower. 30–50 is a good starting point."
    )
    hidden = st.select_slider(
        "Hidden units", [64, 96, 128, 192, 256], 128,
        help="Number of neurons in each LSTM layer. "
             "More units = more capacity to learn complex patterns, "
             "but also more risk of overfitting on small datasets. "
             "128 works well for most stocks."
    )
    n_layers = st.selectbox(
        "LSTM layers", [1, 2, 3], 1,
        help="Number of stacked LSTM layers. "
             "2 layers capture deeper temporal patterns. "
             "3 layers is rarely better and trains slower."
    )
    dropout = st.slider(
        "Dropout rate", 0.0, 0.5, 0.25, 0.05,
        help="Randomly disables this fraction of neurons during training "
             "to prevent the model from memorising the training data. "
             "0.2–0.3 is a safe default. Increase if val loss >> train loss."
    )

    st.markdown('<div class="sec" style="margin-top:16px">Training</div>', unsafe_allow_html=True)
    epochs = st.slider(
        "Max epochs", 20, 100, 50, 5,
        help="Maximum number of full passes over the training data. "
             "Early stopping will usually stop training before this limit "
             "if the model stops improving."
    )
    lr = st.select_slider(
        "Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], 1e-3,
        help="Controls how large each weight update step is. "
             "Too high → unstable training. Too low → very slow. "
             "1e-3 is a reliable default; try 3e-4 for fine-tuning."
    )
    batch_size = st.selectbox(
        "Batch size", [32, 64, 128], 1,
        help="Number of samples processed together before updating weights. "
             "Larger batches are faster but may generalise slightly worse. "
             "64 is a good default."
    )
    patience = st.slider(
        "Early stopping patience", 5, 20, 10,
        help="Training stops automatically if the validation loss does not "
             "improve for this many consecutive epochs. "
             "Prevents wasting time and overfitting."
    )

    st.markdown("---")
    run = st.button("🚀  Run Analysis", use_container_width=True)

# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
hero_ph    = st.empty()
metrics_ph = st.empty()
chart_ph   = st.empty()
fcast_ph   = st.empty()

hero_ph.markdown(f"""
<div class="hero">
  <div class="ticker">{ticker}</div>
  <div style="font-size:.95rem;color:#c9d1d9;margin-top:4px;font-weight:600">{company_name}</div>
  <div class="sub" style="margin-top:2px">LSTM · Rolling-Norm · Deep Learning Forecast</div>
</div>""", unsafe_allow_html=True)

if not run:
    st.markdown("""
    <div style="text-align:center;padding:60px 0;color:#8b949e">
      <div style="font-size:3rem;margin-bottom:12px">📈</div>
      <div style="font-size:1.05rem;font-weight:600;color:#c9d1d9">Configure & Run Your Analysis</div>
      <div style="font-size:.82rem;margin-top:6px">Set parameters in the sidebar → click <b>Run Analysis</b></div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Fetch ──
with st.spinner("Fetching market data…"):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        st.error(f"Download failed: {e}"); st.stop()

if df is None or df.empty:
    st.warning("No data. Check ticker and date range."); st.stop()

# Flatten MultiIndex if present (yfinance ≥ 0.2)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open","High","Low","Close","Volume"]].dropna()
if len(df) < seq_len + 100:
    st.error(f"Not enough data ({len(df)} rows). Use a longer date range or smaller lookback.")
    st.stop()

# ── Feature engineering ──
df = build_features(df)

close_arr = df["Close"].values.astype(np.float64)
feat_arr  = df[FEAT_COLS].values.astype(np.float64)

# ── Sequences ──
X, y, mus, stds = make_sequences(close_arr, feat_arr, seq_len)
prices_aligned   = close_arr[seq_len:]      # actual prices aligned with targets

n      = len(X)
n_tr   = int(0.72 * n)
n_val  = int(0.86 * n)

X_tr, y_tr = X[:n_tr],       y[:n_tr]
X_va, y_va = X[n_tr:n_val],  y[n_tr:n_val]
X_te, y_te = X[n_val:],      y[n_val:]
mus_te     = mus[n_val:]
stds_te    = stds[n_val:]
prices_te  = prices_aligned[n_val:]

if len(X_tr) < batch_size:
    st.error("Too few training samples. Reduce lookback or use a longer date range."); st.stop()
if len(X_te) == 0:
    st.error("Empty test set. Use a longer date range."); st.stop()

st.info(f"📊 **{ticker}** — {len(df)} trading days loaded  |  "
        f"Train: {len(X_tr)} · Val: {len(X_va)} · Test: {len(X_te)} sequences")

tr_ld = DataLoader(StockDS(X_tr, y_tr), batch_size=batch_size, shuffle=True,  drop_last=True)
va_ld = DataLoader(StockDS(X_va, y_va), batch_size=batch_size, shuffle=False)
te_ld = DataLoader(StockDS(X_te, y_te), batch_size=len(X_te),  shuffle=False)

# ── Model ──
n_feat = X.shape[2]   # close_norm + 12 features = 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = LSTMPredictor(n_feat, hidden, n_layers, dropout).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
sch    = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
crit   = nn.MSELoss()

# ── Training ──
st.markdown('<div class="sec">Training</div>', unsafe_allow_html=True)
prog      = st.progress(0)
ep_txt    = st.empty()
log_lines = []
log_ph    = st.empty()

tr_losses, va_losses     = [], []
best_val, best_w, pat_cnt = 1e9, None, 0

for ep in range(1, epochs+1):
    model.train()
    ep_loss = []
    for xb, yb in tr_ld:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss.append(loss.item())
    tr_l = float(np.mean(ep_loss))

    model.eval()
    with torch.no_grad():
        va_l = float(np.mean([crit(model(xb.to(device)), yb.to(device)).item()
                               for xb, yb in va_ld])) if len(va_ld) > 0 else tr_l
    sch.step(va_l)
    tr_losses.append(tr_l); va_losses.append(va_l)

    marker = ""
    if va_l < best_val:
        best_val = va_l
        best_w   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pat_cnt  = 0; marker = " ★ best"
    else:
        pat_cnt += 1; marker = f" patience {pat_cnt}/{patience}"

    prog.progress(ep / epochs)
    ep_txt.markdown(f"**Epoch {ep}/{epochs}** — train: `{tr_l:.5f}` | val: `{va_l:.5f}`{marker}")
    log_lines.append(f"[{ep:>3}]  tr={tr_l:.5f}  va={va_l:.5f}{marker}")
    log_ph.markdown('<div class="tlog">' + "<br>".join(log_lines[-10:]) + '</div>',
                    unsafe_allow_html=True)

    if pat_cnt >= patience:
        ep_txt.markdown(f"⚡ Early stop at epoch **{ep}** (best val: `{best_val:.5f}`)")
        break

prog.empty()
if best_w: model.load_state_dict(best_w)

# ── Inference ──
model.eval()
with torch.no_grad():
    for xb, _ in te_ld:
        preds_norm = model(xb.to(device)).cpu().numpy().reshape(-1)

# Denormalize: each prediction uses its own window's mu/std
preds_price = preds_norm * stds_te + mus_te

# ── Metrics ──
r2      = r2_score(prices_te, preds_price)
rmse    = rmse_val(prices_te, preds_price)
mape    = mape_val(prices_te, preds_price)
dir_acc = np.mean(np.sign(np.diff(prices_te)) == np.sign(np.diff(preds_price)))

metrics_ph.markdown(f"""
<div class="mrow">
  <div class="mc g"><div class="ml">R² Score</div>
    <div class="mv">{r2*100:.2f}%</div><div class="ms">Variance explained</div></div>
  <div class="mc b"><div class="ml">RMSE</div>
    <div class="mv">${rmse:.2f}</div><div class="ms">Root mean squared error</div></div>
  <div class="mc o"><div class="ml">MAPE</div>
    <div class="mv">{mape*100:.2f}%</div><div class="ms">Mean abs. pct error</div></div>
  <div class="mc p"><div class="ml">Direction Accuracy</div>
    <div class="mv">{dir_acc*100:.1f}%</div><div class="ms">Correct up/down calls</div></div>
</div>""", unsafe_allow_html=True)

# ── Charts ──
plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#21262d","axes.labelcolor":"#8b949e",
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "grid.color":"#21262d","text.color":"#c9d1d9","font.family":"monospace"
})

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3,
                         left=0.07, right=0.97, top=0.93, bottom=0.08)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(prices_te,   color="#58a6ff", lw=1.5, label="Actual",    zorder=3)
ax1.plot(preds_price, color="#3fb950", lw=1.5, label="Predicted", zorder=4, ls="--")
ax1.fill_between(range(len(prices_te)), prices_te, preds_price, alpha=0.07, color="#3fb950")
ax1.set_title(f"{ticker} — Actual vs Predicted (Test Set)", color="#e6edf3", fontsize=11, pad=10)
ax1.set_ylabel("Price (USD)"); ax1.legend(loc="upper left", framealpha=0.3, edgecolor="#30363d")
ax1.grid(True, ls=":", alpha=0.5)
for s in ["top","right"]: ax1.spines[s].set_visible(False)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(tr_losses, color="#58a6ff", lw=1.4, label="Train")
ax2.plot(va_losses, color="#d29922", lw=1.4, label="Val", ls="--")
ax2.set_title("Loss Curve", color="#e6edf3", fontsize=10)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss")
ax2.legend(framealpha=0.3, edgecolor="#30363d"); ax2.grid(True, ls=":", alpha=0.5)
for s in ["top","right"]: ax2.spines[s].set_visible(False)

ax3 = fig.add_subplot(gs[1, 1])
residuals = preds_price - prices_te
ax3.hist(residuals, bins=40, color="#a371f7", edgecolor="#0d1117", alpha=0.85)
ax3.axvline(0, color="#ff7b72", lw=1.5, ls="--")
ax3.set_title("Residuals Distribution", color="#e6edf3", fontsize=10)
ax3.set_xlabel("Predicted − Actual ($)"); ax3.set_ylabel("Count")
ax3.grid(True, ls=":", alpha=0.5)
for s in ["top","right"]: ax3.spines[s].set_visible(False)

chart_ph.pyplot(fig); plt.close(fig)

# ── Next-day forecast ──
last_win   = close_arr[-seq_len:]
last_mu    = last_win.mean(); last_std = last_win.std() + 1e-8
last_cnorm = (last_win - last_mu) / last_std
last_feat  = feat_arr[-seq_len:]
last_samp  = np.column_stack([last_cnorm, last_feat]).astype(np.float32)
last_t     = torch.tensor(last_samp).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    nd_norm = model(last_t).cpu().item()

next_price  = nd_norm * last_std + last_mu
last_actual = close_arr[-1]
delta_pct   = (next_price - last_actual) / last_actual * 100
arrow = "▲" if delta_pct > 0 else "▼"
color = "#3fb950" if delta_pct > 0 else "#ff7b72"

fcast_ph.markdown(f"""
<div class="fcast">
  <div>
    <div class="fl">Next Trading Day Forecast</div>
    <div class="fp">${next_price:.2f}</div>
    <div class="fn">vs last close <b>${last_actual:.2f}</b></div>
  </div>
  <div style="text-align:right">
    <div style="font-size:2.2rem;font-weight:900;color:{color}">{arrow} {abs(delta_pct):.2f}%</div>
    <div style="font-size:.73rem;color:#6e7681;margin-top:4px">
      LSTM · Rolling-Window Norm<br>
      {n_feat} features · {seq_len}d lookback
    </div>
  </div>
</div>""", unsafe_allow_html=True)