import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Athens Stock Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Syne', sans-serif;
      background-color: #0a0f1e;
      color: #e2e8f0;
  }

  .main { background-color: #0a0f1e; }
  .block-container { padding-top: 1.5rem; }

  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

  .signal-buy {
      background: linear-gradient(135deg, #0d4429, #16a34a22);
      border: 1px solid #16a34a;
      border-radius: 8px;
      padding: 12px 16px;
      color: #4ade80;
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      font-size: 0.9rem;
      text-align: center;
  }
  .signal-sell {
      background: linear-gradient(135deg, #4a0d0d, #dc262622);
      border: 1px solid #dc2626;
      border-radius: 8px;
      padding: 12px 16px;
      color: #f87171;
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      font-size: 0.9rem;
      text-align: center;
  }
  .signal-hold {
      background: linear-gradient(135deg, #1a1a2e, #3b3b6022);
      border: 1px solid #6366f1;
      border-radius: 8px;
      padding: 12px 16px;
      color: #a5b4fc;
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      font-size: 0.9rem;
      text-align: center;
  }

  .metric-card {
      background: #111827;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 10px;
  }

  .ticker-header {
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      color: #64748b;
      letter-spacing: 0.12em;
      text-transform: uppercase;
  }

  .stSelectbox label, .stMultiSelect label, .stSlider label {
      color: #94a3b8 !important;
      font-size: 0.8rem;
      font-family: 'Space Mono', monospace;
      letter-spacing: 0.05em;
  }

  div[data-testid="stMetric"] {
      background: #111827;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 12px 16px;
  }
  div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.75rem; }
  div[data-testid="stMetric"] div { color: #e2e8f0 !important; font-family: 'Space Mono', monospace; }

  .stButton>button {
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: white;
      border: none;
      border-radius: 8px;
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      letter-spacing: 0.05em;
  }

  .last-update {
      font-family: 'Space Mono', monospace;
      font-size: 0.7rem;
      color: #475569;
  }
</style>
""", unsafe_allow_html=True)


# ─── Athens Stock Exchange Tickers ─────────────────────────────────────────────
ATHEX_STOCKS = {
    "Alpha Bank":           "ALPHA.AT",
    "Eurobank":             "EUROB.AT",
    "National Bank Greece": "ETE.AT",
    "Piraeus Bank":         "TPEIR.AT",
    "OTE Telecom":          "HTO.AT",
    "OPAP":                 "OPAP.AT",
    "Motor Oil":            "MOH.AT",
    "Hellenic Petroleum":   "ELPE.AT",
    "Mytilineos":           "MYTIL.AT",
    "Jumbo":                "BELA.AT",
    "GEK TERNA":            "GEKTERNA.AT",
    "Lamda Development":    "LAMDA.AT",
    "Aegean Airlines":      "AEGN.AT",
    "Ellaktor":             "ELLAKTOR.AT",
    "ADMIE (IPTO)":         "ADMIE.AT",
}


# ─── Signal Engine ─────────────────────────────────────────────────────────────
def compute_signals(df: pd.DataFrame, fast_ma=20, slow_ma=50, rsi_period=14) -> dict:
    close = df["Close"].squeeze().dropna()
    signals = {}

    # ── Moving Average Signal ──
    ma_fast = close.rolling(fast_ma).mean()
    ma_slow = close.rolling(slow_ma).mean()
    if len(close) >= slow_ma:
        if float(ma_fast.iloc[-1]) > float(ma_slow.iloc[-1]) and \
                float(ma_fast.iloc[-2]) <= float(ma_slow.iloc[-2]):
        #if ma_fast.iloc[-1] > ma_slow.iloc[-1] and ma_fast.iloc[-2] <= ma_slow.iloc[-2]:
            signals["MA"] = ("BUY", f"MA{fast_ma} crossed above MA{slow_ma}")
        elif ma_fast.iloc[-1] < ma_slow.iloc[-1] and ma_fast.iloc[-2] >= ma_slow.iloc[-2]:
            signals["MA"] = ("SELL", f"MA{fast_ma} crossed below MA{slow_ma}")
        elif ma_fast.iloc[-1] > ma_slow.iloc[-1]:
            signals["MA"] = ("HOLD", f"Bullish trend — MA{fast_ma} > MA{slow_ma}")
        else:
            signals["MA"] = ("HOLD", f"Bearish trend — MA{fast_ma} < MA{slow_ma}")
    else:
        signals["MA"] = ("N/A", "Not enough data")

    # ── RSI Signal ──
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1]
    signals["RSI"] = (
        round(rsi_val, 1) if not np.isnan(rsi_val) else None,
        "BUY" if rsi_val < 30 else ("SELL" if rsi_val > 70 else "NEUTRAL")
    )

    # ── MACD Signal ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    if len(close) >= 26:
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            signals["MACD"] = ("BUY", f"MACD crossed above signal")
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            signals["MACD"] = ("SELL", "MACD crossed below signal")
        elif macd_line.iloc[-1] > signal_line.iloc[-1]:
            signals["MACD"] = ("HOLD", "MACD above signal — bullish")
        else:
            signals["MACD"] = ("HOLD", "MACD below signal — bearish")
    else:
        signals["MACD"] = ("N/A", "Not enough data")

    return {
        "signals": signals,
        "ma_fast": ma_fast,
        "ma_slow": ma_slow,
        "rsi": rsi,
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
        "close": close,
    }


def signal_badge(action: str) -> str:
    css = {"BUY": "signal-buy", "SELL": "signal-sell"}.get(action, "signal-hold")
    icon = {"BUY": "▲ BUY", "SELL": "▼ SELL", "HOLD": "◆ HOLD", "N/A": "— N/A"}.get(action, action)
    return f'<div class="{css}">{icon}</div>'


# ─── Chart Builder ─────────────────────────────────────────────────────────────
def build_chart(data: dict, ticker: str, company: str) -> go.Figure:
    close = data["close"]
    dates = close.index

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.04,
        subplot_titles=("Price + Moving Averages", "RSI", "MACD"),
    )

    # Price candlestick (if OHLC available)
    fig.add_trace(go.Scatter(
        x=dates, y=close,
        line=dict(color="#6366f1", width=2),
        name="Price", fill="tozeroy",
        fillcolor="rgba(99,102,241,0.07)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=data["ma_fast"],
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        name=f"MA Fast"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=data["ma_slow"],
        line=dict(color="#ec4899", width=1.5, dash="dash"),
        name=f"MA Slow"
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=dates, y=data["rsi"],
        line=dict(color="#34d399", width=1.5),
        name="RSI"
    ), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#f87171", dash="dot", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#4ade80", dash="dot", width=1), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="#dc2626", opacity=0.05, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="#16a34a", opacity=0.05, row=2, col=1)

    # MACD
    colors = ["#4ade80" if v >= 0 else "#f87171" for v in data["histogram"]]
    fig.add_trace(go.Bar(
        x=dates, y=data["histogram"],
        marker_color=colors, name="Histogram", opacity=0.7
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=data["macd_line"],
        line=dict(color="#6366f1", width=1.5), name="MACD"
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=data["signal_line"],
        line=dict(color="#f59e0b", width=1.5), name="Signal"
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#0d1117",
        font=dict(family="Space Mono", color="#94a3b8", size=11),
        legend=dict(
            bgcolor="#111827", bordercolor="#1e293b",
            borderwidth=1, font=dict(size=10)
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        height=620,
        title=dict(
            text=f"{company}  ·  {ticker}",
            font=dict(family="Syne", size=16, color="#e2e8f0"),
            x=0.01
        ),
        xaxis3=dict(rangeslider=dict(visible=False)),
    )
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
    fig.update_yaxes(gridcolor="#1e293b", showgrid=True)

    return fig


# ─── Fetch Data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str = "3mo", interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Athens Signals")
    st.markdown('<p class="ticker-header">Hellenic Capital Market</p>', unsafe_allow_html=True)
    st.divider()

    selected_names = st.multiselect(
        "Select Stocks",
        options=list(ATHEX_STOCKS.keys()),
        default=["Alpha Bank", "OTE Telecom", "OPAP"],
    )

    st.divider()
    st.markdown('<p class="ticker-header">MA Settings</p>', unsafe_allow_html=True)
    fast_ma = st.slider("Fast MA Period", 5, 50, 20)
    slow_ma = st.slider("Slow MA Period", 20, 200, 50)

    st.markdown('<p class="ticker-header">RSI Settings</p>', unsafe_allow_html=True)
    rsi_period = st.slider("RSI Period", 7, 21, 14)

    st.divider()
    period = st.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

    auto_refresh = st.toggle("Auto-Refresh (60s)", value=True)
    st.divider()
    st.markdown('<p class="last-update">Data via Yahoo Finance<br>Athens Exchange (ATHEX)<br>Tickers end in .AT</p>', unsafe_allow_html=True)


# ─── Header ────────────────────────────────────────────────────────────────────
col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown("# Athens Stock Exchange")
    st.markdown('<p class="ticker-header">Market Signals Dashboard — MA · RSI · MACD</p>', unsafe_allow_html=True)
with col_time:
    st.markdown(f'<p class="last-update" style="text-align:right;padding-top:20px;">Last update<br>{datetime.now().strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)

st.divider()

if not selected_names:
    st.info("👈 Select at least one stock from the sidebar to begin.")
    st.stop()


# ─── Signal Summary Table ──────────────────────────────────────────────────────
st.markdown("### Signal Overview")
summary_cols = st.columns(len(selected_names))

for idx, name in enumerate(selected_names):
    ticker = ATHEX_STOCKS[name]
    df = fetch_data(ticker, period=period, interval=interval)

    with summary_cols[idx]:
        st.markdown(f'<p class="ticker-header">{name}</p>', unsafe_allow_html=True)
        if df is None or len(df) < 30:
            st.warning("Insufficient data")
            continue

        data = compute_signals(df, fast_ma, slow_ma, rsi_period)
        close_price = data["close"].iloc[-1]
        prev_price = data["close"].iloc[-2]
        change_pct = (close_price - prev_price) / prev_price * 100

        st.metric(
            label=ticker,
            value=f"€{close_price:.2f}",
            delta=f"{change_pct:+.2f}%"
        )

        # Signals
        ma_action = data["signals"]["MA"][0]
        macd_action = data["signals"]["MACD"][0]
        rsi_val, rsi_action = data["signals"]["RSI"]

        st.markdown(signal_badge(ma_action), unsafe_allow_html=True)
        st.caption(f"MA{fast_ma}/{slow_ma}")
        st.markdown(signal_badge(rsi_action), unsafe_allow_html=True)
        st.caption(f"RSI: {rsi_val}" if rsi_val else "RSI: N/A")
        st.markdown(signal_badge(macd_action), unsafe_allow_html=True)
        st.caption("MACD")

st.divider()

# ─── Detailed Charts ───────────────────────────────────────────────────────────
st.markdown("### Detailed Analysis")

for name in selected_names:
    ticker = ATHEX_STOCKS[name]
    df = fetch_data(ticker, period=period, interval=interval)

    with st.expander(f"📈  {name}  ({ticker})", expanded=(name == selected_names[0])):
        if df is None or len(df) < 30:
            st.warning(f"Not enough data to compute signals for {name}.")
            continue

        data = compute_signals(df, fast_ma, slow_ma, rsi_period)
        fig = build_chart(data, ticker, name)
        st.plotly_chart(fig, use_container_width=True)

        # Signal detail row
        s = data["signals"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Moving Average**")
            action, desc = s["MA"]
            st.markdown(signal_badge(action), unsafe_allow_html=True)
            st.caption(desc)
        with c2:
            st.markdown("**RSI**")
            rsi_val, rsi_action = s["RSI"]
            st.markdown(signal_badge(rsi_action), unsafe_allow_html=True)
            st.caption(f"Current RSI: {rsi_val}" if rsi_val else "Not enough data")
        with c3:
            st.markdown("**MACD**")
            action, desc = s["MACD"]
            st.markdown(signal_badge(action), unsafe_allow_html=True)
            st.caption(desc)

# ─── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.rerun()
