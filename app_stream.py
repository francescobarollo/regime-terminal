import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import ccxt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

st.set_page_config(page_title="Regime Terminal", layout="wide", page_icon="📈")

st.markdown("""
<style>
body, .stApp { background-color: #0f0f0f; color: #e0e0e0; }
.metric-box {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.metric-label { font-size: 12px; color: #888; margin-bottom: 2px; }
.metric-value { font-size: 22px; font-weight: 600; }
.regime-box {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
}
.regime-name { font-size: 28px; font-weight: 700; margin: 8px 0; }
.regime-action { font-size: 14px; color: #ccc; margin-bottom: 8px; }
.signal-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 16px;
    font-weight: 700;
    margin-top: 8px;
}
.conf-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #222;
    font-size: 13px;
}
div[data-testid="stSidebar"] { background-color: #111; }
</style>
""", unsafe_allow_html=True)

REGIME_NAMES  = {0:"Bull Run",1:"Recovery",2:"Chop",3:"Noise",4:"Crash",5:"Bear",6:"Transition"}
REGIME_COLORS = {0:"#3B8BD4",1:"#9FE1CB",2:"#EF9F27",3:"#888780",4:"#E24B4A",5:"#D85A30",6:"#BA7517"}
REGIME_ACTION = {
    0:"LONG — entra / mantieni posizioni",
    1:"LONG — entrata progressiva",
    2:"HOLD — aspetta conferme",
    3:"HOLD — nessuna azione",
    4:"EXIT — chiudi tutto ora",
    5:"SHORT — regime ribassista",
    6:"WAIT — cambio regime in atto",
}

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configurazione")
    symbol      = st.selectbox("Asset", ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT"])
    timeframe   = st.selectbox("Timeframe", ["1h","4h","15m"], index=0)
    months_back = st.slider("Mesi di storico", 3, 24, 12)
    st.divider()
    st.subheader("Parametri strategia")
    n_regimes   = st.slider("Numero regimi HMM", 3, 9, 7)
    leverage    = st.slider("Leva finanziaria", 1.0, 5.0, 1.0, 0.5)
    confirm_req = st.slider("Conferme richieste (su 8)", 3, 8, 6)
    fee         = st.number_input("Fee per trade (%)", 0.0, 1.0, 0.1, 0.05) / 100
    st.divider()
    st.subheader("Modalità short")
    enable_short   = st.toggle("Abilita short su Bear/Crash", value=True)
    short_leverage = st.slider("Leva short", 1.0, 3.0, 1.0, 0.5, disabled=not enable_short)
    st.divider()
    run = st.button("▶ RUN ANALYSIS", use_container_width=True, type="primary")

# ── Helpers ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def download_crypto(symbol, timeframe, months_back):
    import yfinance as yf
    yf_symbol = symbol.replace("/USDT", "-USD").replace("/BTC", "-BTC")
    # yfinance limiti: 1h max 730 giorni, 15m max 60 giorni
    if timeframe == "15m":
        yf_interval = "15m"
        yf_period   = "60d"
    elif timeframe == "4h":
        yf_interval = "1h"
        yf_period   = "730d"
    else:  # 1h
        yf_interval = "1h"
        yf_period   = "730d"
    df = yf.download(yf_symbol, period=yf_period, interval=yf_interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    # Filtra per months_back
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=months_back)
    df = df[df.index >= cutoff]
    return df

def build_features(df):
    df = df.copy()
    df["returns"]    = df["Close"].pct_change()
    df["log_ret"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility"] = df["returns"].rolling(20).std()
    df["vol_ratio"]  = df["Volume"] / df["Volume"].rolling(48).mean()
    df["range_pct"]  = (df["High"] - df["Low"]) / df["Close"]
    df["ma_diff"]    = (df["Close"].rolling(24).mean() - df["Close"].rolling(96).mean()) / df["Close"]
    df["momentum"]   = df["Close"].pct_change(12)
    return df.dropna()

@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(symbol, timeframe, months_back, n_regimes, leverage, confirm_req, fee, enable_short, short_leverage):
    df_raw = download_crypto(symbol, timeframe, months_back)
    df = build_features(df_raw)

    feats = ["log_ret","volatility","vol_ratio","range_pct","ma_diff","momentum"]
    X = StandardScaler().fit_transform(df[feats].values)
    model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                        n_iter=300, random_state=42, tol=1e-5)
    model.fit(X)
    states = model.predict(X)
    probs  = model.predict_proba(X)

    means = {s: float(df["log_ret"].values[states==s].mean()) if (states==s).sum()>0 else 0.0 for s in range(n_regimes)}
    vols  = {s: float(df["volatility"].values[states==s].mean()) if (states==s).sum()>0 else 0.0 for s in range(n_regimes)}
    sorted_s = sorted(means, key=means.get, reverse=True)
    mean_vol = float(np.mean(list(vols.values())))
    label_map = {}
    for rank, s in enumerate(sorted_s):
        v = vols[s]; r = means[s]
        if rank == 0:                                label_map[s] = 0
        elif rank == 1 and r > 0:                    label_map[s] = 1
        elif rank == len(sorted_s)-1 and r < -0.001: label_map[s] = 4
        elif rank >= len(sorted_s)-2 and r < 0:      label_map[s] = 5
        elif v > mean_vol * 1.4:                     label_map[s] = 2
        elif v < mean_vol * 0.6:                     label_map[s] = 3
        else:                                         label_map[s] = 6
    regimes = np.array([label_map.get(s, 3) for s in states])

    # Backtest con long + short
    # position: 0=flat, 1=long, -1=short
    capital = 100_000; bh_cap = 100_000
    equity = [100_000]; bh_curve = [100_000]
    trades = []
    position = 0; entry_price = 0; entry_bar = -999; last_exit = -999
    min_hold = {"15m":8,"1h":6,"4h":3}.get(timeframe,6)
    cooldown = {"15m":16,"1h":8,"4h":4}.get(timeframe,8)

    for i in range(1, len(df)):
        ret = float(df["returns"].iloc[i])
        bh_cap *= (1 + ret)
        bh_curve.append(bh_cap)
        regime = regimes[i]; held = i - entry_bar

        if position == 1:  # LONG aperto
            capital *= (1 + ret * leverage - fee)
            exit_long = (regime in [4,5] and held >= min_hold) or held >= 200
            if exit_long:
                pnl = (capital - equity[-1]) / equity[-1] * 100
                trades.append({"date": df.index[i], "type":"EXIT LONG",
                               "regime": REGIME_NAMES.get(regime,"?"),
                               "entry": round(entry_price,1), "exit": round(float(df["Close"].iloc[i]),1),
                               "pnl": round(pnl,2), "hold": held})
                position = 0; last_exit = i

        elif position == -1:  # SHORT aperto
            capital *= (1 - ret * short_leverage - fee)
            exit_short = (regime in [0,1] and held >= min_hold) or held >= 200
            if exit_short:
                pnl = (capital - equity[-1]) / equity[-1] * 100
                trades.append({"date": df.index[i], "type":"EXIT SHORT",
                               "regime": REGIME_NAMES.get(regime,"?"),
                               "entry": round(entry_price,1), "exit": round(float(df["Close"].iloc[i]),1),
                               "pnl": round(pnl,2), "hold": held})
                position = 0; last_exit = i
        else:
            capital = equity[-1]

        # Entra LONG
        if position == 0 and regime in [0,1] and i - last_exit >= cooldown:
            position = 1; entry_price = float(df["Close"].iloc[i])
            entry_bar = i; capital *= (1 - fee)
            trades.append({"date": df.index[i], "type":"LONG",
                           "regime": REGIME_NAMES.get(regime,"?"),
                           "entry": round(entry_price,1), "exit":"-", "pnl":"-", "hold":0})

        # Entra SHORT
        elif position == 0 and enable_short and regime in [4,5] and i - last_exit >= cooldown:
            position = -1; entry_price = float(df["Close"].iloc[i])
            entry_bar = i; capital *= (1 - fee)
            trades.append({"date": df.index[i], "type":"SHORT",
                           "regime": REGIME_NAMES.get(regime,"?"),
                           "entry": round(entry_price,1), "exit":"-", "pnl":"-", "hold":0})

        equity.append(capital)

    equity = np.array(equity); bh_curve = np.array(bh_curve)

    # Confirmations
    close = df["Close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0,1e-9)))
    ema12 = close.ewm(span=12).mean(); ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26; sig = macd.ewm(span=9).mean()
    ma20  = close.rolling(20).mean()
    atr   = (df["High"] - df["Low"]).rolling(14).mean()
    atr_avg = atr.rolling(48).mean()
    x = np.arange(24); slope = np.polyfit(x, close.iloc[-24:].values, 1)[0]
    mom4  = float(close.iloc[-1]) / float(close.iloc[-5]) - 1
    mom12 = float(close.iloc[-1]) / float(close.iloc[-13]) - 1
    current_regime = int(regimes[-1])
    confirmations = {
        "RSI > 50":      float(rsi.iloc[-1]) > 50,
        "MACD bullish":  float(macd.iloc[-1]) > float(sig.iloc[-1]),
        "Above MA20":    float(close.iloc[-1]) > float(ma20.iloc[-1]),
        "Volume surge":  float(df["Volume"].iloc[-1]) > float(df["Volume"].rolling(48).mean().iloc[-1]),
        "ATR normale":   float(atr.iloc[-1]) < float(atr_avg.iloc[-1]) * 1.5,
        "Regime HMM":    current_regime in [0,1],
        "Trend slope":   slope > 0,
        "Momentum +":    mom4 > 0 and mom12 > 0,
    }

    # Metrics
    closed = [t for t in trades if t["pnl"] != "-"]
    wins   = [t for t in closed if t["pnl"] > 0]
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak * 100
    bpd    = {"15m":96,"1h":24,"4h":6}.get(timeframe,24)
    daily_r = np.diff(equity) / equity[:-1]
    sharpe = (daily_r.mean() / (daily_r.std()+1e-9)) * np.sqrt(bpd*365)
    metrics = {
        "return":   round((equity[-1]-100000)/100000*100, 1),
        "bh_ret":   round((bh_cap-100000)/100000*100, 1),
        "alpha":    round((equity[-1]-100000)/100000*100 - (bh_cap-100000)/100000*100, 1),
        "max_dd":   round(dd.min(), 1),
        "sharpe":   round(sharpe, 2),
        "win_rate": round(len(wins)/len(closed)*100 if closed else 0, 1),
        "n_trades": len(closed),
        "avg_win":  round(np.mean([t["pnl"] for t in wins]) if wins else 0, 2),
        "avg_loss": round(np.mean([t["pnl"] for t in closed if t["pnl"]<=0]) if any(t["pnl"]<=0 for t in closed) else 0, 2),
        "final":    round(float(equity[-1])),
    }

    return df, regimes, probs, equity, bh_curve, trades, metrics, confirmations, current_regime

# ── Main ─────────────────────────────────────────────────────────
st.title("📈 Regime Terminal")
st.caption(f"Hidden Markov Model · {n_regimes} stati di mercato · Crypto")

if not run:
    st.info("Configura i parametri nella sidebar e premi **▶ RUN ANALYSIS** per iniziare.")
    st.stop()

with st.spinner(f"Scaricando {symbol} {timeframe} da Binance e addestrando HMM..."):
    try:
        df, regimes, probs, equity, bh_curve, trades, metrics, confirmations, current_regime = \
            run_analysis(symbol, timeframe, months_back, n_regimes, leverage, confirm_req, fee, enable_short, short_leverage)
    except Exception as e:
        st.error(f"Errore: {e}")
        st.stop()

# ── Layout ───────────────────────────────────────────────────────
rname  = REGIME_NAMES.get(current_regime, "?")
rcolor = REGIME_COLORS.get(current_regime, "#888")
raction = REGIME_ACTION.get(current_regime, "")
conf_pct = round(float(probs[-1, current_regime]) * 100, 1)
n_pass = sum(confirmations.values())
signal = "LONG ▲" if n_pass >= confirm_req and current_regime in [0,1] else \
         "EXIT ▼" if current_regime in [4,5] else "HOLD ⏸"
sig_col = {"LONG ▲":"#3B8BD4","EXIT ▼":"#E24B4A","HOLD ⏸":"#EF9F27"}.get(signal,"#888")

# Top row: regime + metrics
top_cols = st.columns(5)
with top_cols[0]:
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">REGIME</div><div style="font-size:20px;font-weight:700;color:{rcolor}">{rname}</div><div style="font-size:13px;font-weight:700;color:{sig_col}">{signal}</div><div style="font-size:11px;color:#aaa">conf: {conf_pct}%</div></div>', unsafe_allow_html=True)
with top_cols[1]:
    ret_col = "#3B8BD4" if metrics["return"] >= 0 else "#E24B4A"
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">RENDIMENTO</div><div style="font-size:22px;font-weight:700;color:{ret_col}">{metrics["return"]:+.1f}%</div><div style="font-size:11px;color:#aaa">alpha: {metrics["alpha"]:+.1f}%</div></div>', unsafe_allow_html=True)
with top_cols[2]:
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">MAX DRAWDOWN</div><div style="font-size:22px;font-weight:700;color:#E24B4A">{metrics["max_dd"]:.1f}%</div><div style="font-size:11px;color:#aaa">sharpe: {metrics["sharpe"]}</div></div>', unsafe_allow_html=True)
with top_cols[3]:
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">WIN RATE</div><div style="font-size:22px;font-weight:700;color:#9FE1CB">{metrics["win_rate"]:.1f}%</div><div style="font-size:11px;color:#aaa">{metrics["n_trades"]} trade</div></div>', unsafe_allow_html=True)
with top_cols[4]:
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">CAPITALE FINALE</div><div style="font-size:22px;font-weight:700;color:#3B8BD4">${metrics["final"]:,}</div><div style="font-size:11px;color:#aaa">start: $100,000</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_main, col_side = st.columns([3, 1])

with col_side:
    def mc(label, value, color="#e0e0e0"):
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.subheader("Conferme segnale")
    for name, ok in confirmations.items():
        col = "#9FE1CB" if ok else "#E24B4A"
        sym = "✓" if ok else "✗"
        st.markdown(f'<div class="conf-item"><span>{name}</span><span style="color:{col};font-weight:600">{sym}</span></div>', unsafe_allow_html=True)
    st.markdown(f"<br><b style='color:{'#3B8BD4' if n_pass>=confirm_req else '#EF9F27'}'>{n_pass}/8 conferme</b>", unsafe_allow_html=True)

# Charts — fuori dalle colonne per occupare tutta la larghezza
dates = df.index
n = min(len(equity), len(dates))

# Chart 1: Price + regime
fig1 = go.Figure()
prev_r = regimes[0]; start_i = 0
for i in range(1, len(regimes)):
    if regimes[i] != prev_r or i == len(regimes)-1:
        fig1.add_vrect(x0=dates[start_i], x1=dates[i],
                       fillcolor=REGIME_COLORS.get(prev_r,"#888"),
                       opacity=0.18, line_width=0)
        prev_r = regimes[i]; start_i = i

fig1.add_trace(go.Scatter(x=dates, y=df["Close"], line=dict(color="#e0e0e0", width=1),
                          name=symbol, hovertemplate="%{x}<br>$%{y:,.0f}"))
for t in [t for t in trades if t["type"]=="LONG"]:
    fig1.add_trace(go.Scatter(x=[t["date"]], y=[t["entry"]],
                              mode="markers", marker=dict(symbol="triangle-up", size=8, color="#3B8BD4"),
                              showlegend=False, hovertemplate=f"LONG @ ${t['entry']}"))
for t in [t for t in trades if t["type"]=="SHORT"]:
    fig1.add_trace(go.Scatter(x=[t["date"]], y=[t["entry"]],
                              mode="markers", marker=dict(symbol="triangle-down", size=8, color="#E24B4A"),
                              showlegend=False, hovertemplate=f"SHORT @ ${t['entry']}"))
for t in [t for t in trades if t["type"] in ["EXIT LONG","EXIT SHORT"]]:
    c = "#9FE1CB" if isinstance(t["pnl"],float) and t["pnl"]>0 else "#E24B4A"
    fig1.add_trace(go.Scatter(x=[t["date"]], y=[t["exit"]],
                              mode="markers", marker=dict(symbol="x", size=7, color=c),
                              showlegend=False))
fig1.update_layout(height=320, paper_bgcolor="#0f0f0f", plot_bgcolor="#1a1a1a",
                   font=dict(color="#aaa"), margin=dict(l=10,r=10,t=30,b=10),
                   title=dict(text=f"{symbol} {timeframe} + Regime Overlay", font=dict(size=13)),
                   xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222", tickprefix="$"))
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Equity
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=dates[:n], y=equity[:n], line=dict(color="#3B8BD4", width=2),
                          name="HMM Strategy"), secondary_y=False)
fig2.add_trace(go.Scatter(x=dates[:n], y=bh_curve[:n], line=dict(color="#888780", width=1.5, dash="dash"),
                          name="Buy & Hold"), secondary_y=False)
peak = np.maximum.accumulate(equity[:n])
dd_arr = (equity[:n] - peak) / peak * 100
fig2.add_trace(go.Scatter(x=dates[:n], y=dd_arr, line=dict(color="#E24B4A", width=1),
                          fill="tozeroy", fillcolor="rgba(226,75,74,0.1)", name="Drawdown %"),
               secondary_y=True)
fig2.update_layout(height=280, paper_bgcolor="#0f0f0f", plot_bgcolor="#1a1a1a",
                   font=dict(color="#aaa"), margin=dict(l=10,r=10,t=30,b=10),
                   title=dict(text="Equity Curve + Drawdown", font=dict(size=13)),
                   xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222", tickprefix="$"),
                   legend=dict(bgcolor="#1a1a1a", bordercolor="#333"))
fig2.update_yaxes(ticksuffix="%", secondary_y=True, gridcolor="#222")
st.plotly_chart(fig2, use_container_width=True)

# Chart 3: Regime distribution
total = len(regimes)
pcts_r  = [np.sum(regimes==k)/total*100 for k in range(n_regimes) if k < len(REGIME_NAMES)]
names_r = [REGIME_NAMES[k] for k in range(n_regimes) if k < len(REGIME_NAMES)]
colors_r = [REGIME_COLORS.get(k,"#888") for k in range(n_regimes) if k < len(REGIME_NAMES)]
fig3 = go.Figure(go.Bar(x=names_r, y=pcts_r, marker_color=colors_r,
                        text=[f"{p:.0f}%" for p in pcts_r], textposition="outside"))
fig3.update_layout(height=220, paper_bgcolor="#0f0f0f", plot_bgcolor="#1a1a1a",
                   font=dict(color="#aaa"), margin=dict(l=10,r=10,t=30,b=10),
                   title=dict(text="Distribuzione regimi", font=dict(size=13)),
                   xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222", ticksuffix="%"))
st.plotly_chart(fig3, use_container_width=True)

# Trade log
st.subheader("📋 Trade Log")
closed_trades = [t for t in trades if t["pnl"] != "-"]
if closed_trades:
    df_trades = pd.DataFrame(closed_trades[-50:])
    df_trades["data"] = df_trades["date"].apply(lambda x: x.strftime("%d/%m/%Y %H:%M") if hasattr(x,"strftime") else str(x))
    df_trades["P&L"] = df_trades["pnl"].apply(lambda x: f"{x:+.2f}%" if isinstance(x,float) else x)
    st.dataframe(
        df_trades[["data","type","regime","entry","exit","P&L","hold"]].rename(
            columns={"data":"Data","type":"Tipo","regime":"Regime",
                     "entry":"Entrata","exit":"Uscita","hold":"Hold (barre)"}
        ).style.applymap(lambda v: "color: #9FE1CB" if isinstance(v,str) and "+" in v
                          else ("color: #E24B4A" if isinstance(v,str) and "-" in v else ""),
                          subset=["P&L"]),
        use_container_width=True, height=300
    )
else:
    st.info("Nessun trade chiuso nel periodo selezionato.")
