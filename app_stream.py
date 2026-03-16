import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Regime Terminal", layout="wide", page_icon="📈")

st.markdown("""
<style>
.stApp { background-color: #0f0f0f; color: #e0e0e0; }
div[data-testid="stSidebar"] { background-color: #111; }
.conf-item {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #222;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

REGIME_NAMES  = {0:"Bull Run", 1:"Recovery", 2:"Chop", 3:"Noise", 4:"Crash", 5:"Bear", 6:"Transition"}
REGIME_COLORS = {0:"#3B8BD4", 1:"#9FE1CB", 2:"#EF9F27", 3:"#888780", 4:"#E24B4A", 5:"#D85A30", 6:"#BA7517"}
REGIME_ACTION = {
    0: "LONG — entra / mantieni",
    1: "LONG — entrata progressiva",
    2: "HOLD — aspetta conferme",
    3: "HOLD — nessuna azione",
    4: "EXIT — chiudi tutto ora",
    5: "SHORT — regime ribassista",
    6: "WAIT — cambio regime",
}

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configurazione")
    symbol      = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
    timeframe   = st.selectbox("Timeframe", ["1h", "4h", "15m"])
    months_back = st.slider("Mesi di storico", 3, 18, 12)
    st.divider()
    st.subheader("Strategia")
    n_regimes   = st.slider("Regimi HMM", 3, 9, 7)
    leverage    = st.slider("Leva long", 1.0, 5.0, 1.0, 0.5)
    confirm_req = st.slider("Conferme richieste (su 8)", 3, 8, 6)
    fee         = st.number_input("Fee per trade (%)", 0.0, 1.0, 0.1, 0.05) / 100
    st.divider()
    st.subheader("Short")
    enable_short   = st.toggle("Abilita short su Bear/Crash", value=True)
    short_lev      = st.slider("Leva short", 1.0, 3.0, 1.0, 0.5, disabled=not enable_short)
    st.divider()
    run = st.button("▶ RUN ANALYSIS", use_container_width=True, type="primary")

# ── Download ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def download_data(symbol, timeframe, months_back):
    yf_sym = symbol.replace("/USDT", "-USD").replace("/BTC", "-BTC")
    if timeframe == "15m":
        df = yf.download(yf_sym, period="60d", interval="15m", auto_adjust=True, progress=False)
    else:
        df = yf.download(yf_sym, period="730d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=months_back)
    filtered = df[df.index >= cutoff]
    return filtered if len(filtered) > 200 else df

# ── Features ─────────────────────────────────────────────────────
def make_features(df):
    d = df.copy()
    d["returns"]    = d["Close"].pct_change()
    d["log_ret"]    = np.log(d["Close"] / d["Close"].shift(1))
    d["volatility"] = d["returns"].rolling(20).std()
    d["vol_ratio"]  = d["Volume"] / d["Volume"].rolling(48).mean()
    d["range_pct"]  = (d["High"] - d["Low"]) / d["Close"]
    d["ma_diff"]    = (d["Close"].rolling(24).mean() - d["Close"].rolling(96).mean()) / d["Close"]
    d["momentum"]   = d["Close"].pct_change(12)
    return d.dropna()

# ── HMM ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def full_analysis(symbol, timeframe, months_back, n_regimes, leverage, confirm_req, fee, enable_short, short_lev):
    df_raw = download_data(symbol, timeframe, months_back)
    df = make_features(df_raw)

    feats = ["log_ret", "volatility", "vol_ratio", "range_pct", "ma_diff", "momentum"]
    X = StandardScaler().fit_transform(df[feats].values)
    model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=300, random_state=42)
    model.fit(X)
    states = model.predict(X)
    probs  = model.predict_proba(X)

    means = {s: float(df["log_ret"].values[states==s].mean()) if (states==s).sum()>0 else 0.0 for s in range(n_regimes)}
    vols  = {s: float(df["volatility"].values[states==s].mean()) if (states==s).sum()>0 else 0.0 for s in range(n_regimes)}
    ss = sorted(means, key=means.get, reverse=True)
    mv = float(np.mean(list(vols.values())))
    lmap = {}
    for rank, s in enumerate(ss):
        v, r = vols[s], means[s]
        if rank == 0:                           lmap[s] = 0
        elif rank == 1 and r > 0:               lmap[s] = 1
        elif rank == len(ss)-1 and r < -0.001:  lmap[s] = 4
        elif rank >= len(ss)-2 and r < 0:       lmap[s] = 5
        elif v > mv * 1.4:                      lmap[s] = 2
        elif v < mv * 0.6:                      lmap[s] = 3
        else:                                    lmap[s] = 6
    regimes = np.array([lmap.get(s, 3) for s in states])

    # Backtest
    cap = 100_000; bh = 100_000
    eq = [100_000]; bh_c = [100_000]; trades = []
    pos = 0; ep = 0; eb = -999; le = -999
    mh = {"15m":8, "1h":6, "4h":3}.get(timeframe, 6)
    cd = {"15m":16, "1h":8, "4h":4}.get(timeframe, 8)

    for i in range(1, len(df)):
        ret = float(df["returns"].iloc[i])
        bh *= (1 + ret); bh_c.append(bh)
        rg = regimes[i]; held = i - eb

        if pos == 1:
            cap *= (1 + ret * leverage - fee)
            if (rg in [4,5] and held >= mh) or held >= 200:
                pnl = (cap - eq[-1]) / eq[-1] * 100
                trades.append({"date": df.index[i], "type": "EXIT LONG", "regime": REGIME_NAMES.get(rg,"?"),
                               "entry": round(ep,1), "exit": round(float(df["Close"].iloc[i]),1), "pnl": round(pnl,2), "hold": held})
                pos = 0; le = i
        elif pos == -1:
            cap *= (1 - ret * short_lev - fee)
            if (rg in [0,1] and held >= mh) or held >= 200:
                pnl = (cap - eq[-1]) / eq[-1] * 100
                trades.append({"date": df.index[i], "type": "EXIT SHORT", "regime": REGIME_NAMES.get(rg,"?"),
                               "entry": round(ep,1), "exit": round(float(df["Close"].iloc[i]),1), "pnl": round(pnl,2), "hold": held})
                pos = 0; le = i
        else:
            cap = eq[-1]

        if pos == 0 and rg in [0,1] and i - le >= cd:
            pos = 1; ep = float(df["Close"].iloc[i]); eb = i; cap *= (1-fee)
            trades.append({"date": df.index[i], "type": "LONG", "regime": REGIME_NAMES.get(rg,"?"),
                           "entry": round(ep,1), "exit": "-", "pnl": "-", "hold": 0})
        elif pos == 0 and enable_short and rg in [4,5] and i - le >= cd:
            pos = -1; ep = float(df["Close"].iloc[i]); eb = i; cap *= (1-fee)
            trades.append({"date": df.index[i], "type": "SHORT", "regime": REGIME_NAMES.get(rg,"?"),
                           "entry": round(ep,1), "exit": "-", "pnl": "-", "hold": 0})
        eq.append(cap)

    eq = np.array(eq); bh_c = np.array(bh_c)
    closed = [t for t in trades if t["pnl"] != "-"]
    wins   = [t for t in closed if t["pnl"] > 0]
    peak   = np.maximum.accumulate(eq)
    dd     = (eq - peak) / peak * 100
    dr     = np.diff(eq) / eq[:-1]
    bpd    = {"15m":96,"1h":24,"4h":6}.get(timeframe,24)
    sharpe = (dr.mean() / (dr.std()+1e-9)) * np.sqrt(bpd*365)

    # Confirmations
    cl = df["Close"]
    delta = cl.diff()
    rsi = 100 - 100/(1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0,1e-9))
    macd = cl.ewm(span=12).mean() - cl.ewm(span=26).mean()
    sig  = macd.ewm(span=9).mean()
    ma20 = cl.rolling(20).mean()
    atr  = (df["High"]-df["Low"]).rolling(14).mean()
    slope = np.polyfit(np.arange(24), cl.iloc[-24:].values, 1)[0]
    mom4  = float(cl.iloc[-1])/float(cl.iloc[-5]) - 1
    mom12 = float(cl.iloc[-1])/float(cl.iloc[-13]) - 1
    cur = int(regimes[-1])
    confs = {
        "RSI > 50":     float(rsi.iloc[-1]) > 50,
        "MACD bullish": float(macd.iloc[-1]) > float(sig.iloc[-1]),
        "Above MA20":   float(cl.iloc[-1]) > float(ma20.iloc[-1]),
        "Volume surge": float(df["Volume"].iloc[-1]) > float(df["Volume"].rolling(48).mean().iloc[-1]),
        "ATR normale":  float(atr.iloc[-1]) < float(atr.rolling(48).mean().iloc[-1]) * 1.5,
        "Regime HMM":   cur in [0,1],
        "Trend slope":  slope > 0,
        "Momentum +":   mom4 > 0 and mom12 > 0,
    }

    metrics = {
        "return":   round((eq[-1]-100000)/100000*100, 1),
        "bh_ret":   round((bh-100000)/100000*100, 1),
        "alpha":    round((eq[-1]-100000)/100000*100 - (bh-100000)/100000*100, 1),
        "max_dd":   round(dd.min(), 1),
        "sharpe":   round(sharpe, 2),
        "win_rate": round(len(wins)/len(closed)*100 if closed else 0, 1),
        "n_trades": len(closed),
        "final":    round(float(eq[-1])),
    }

    return df, regimes, probs, eq, bh_c, trades, metrics, confs, cur

# ── Main ─────────────────────────────────────────────────────────
st.title("📈 Regime Terminal")
st.caption("Hidden Markov Model · Crypto · powered by Yahoo Finance")

if not run:
    st.info("Configura i parametri nella sidebar e premi **▶ RUN ANALYSIS**.")
    st.stop()

with st.spinner("Scaricando dati e addestrando HMM..."):
    try:
        df, regimes, probs, eq, bh_c, trades, metrics, confs, cur = \
            full_analysis(symbol, timeframe, months_back, n_regimes, leverage, confirm_req, fee, enable_short, short_lev)
    except Exception as e:
        st.error(f"Errore: {e}")
        st.stop()

# ── Top metrics ──────────────────────────────────────────────────
rname  = REGIME_NAMES.get(cur, "?")
rcolor = REGIME_COLORS.get(cur, "#888")
n_pass = sum(confs.values())
signal = "LONG ▲" if n_pass >= confirm_req and cur in [0,1] else "EXIT ▼" if cur in [4,5] else "HOLD ⏸"
sc     = {"LONG ▲":"#3B8BD4","EXIT ▼":"#E24B4A","HOLD ⏸":"#EF9F27"}.get(signal,"#888")
conf_pct = round(float(probs[-1, cur]) * 100, 1)

c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">REGIME</div><div style="font-size:20px;font-weight:700;color:{rcolor}">{rname}</div><div style="font-size:13px;font-weight:700;color:{sc}">{signal}</div><div style="font-size:11px;color:#aaa">conf: {conf_pct}%</div></div>', unsafe_allow_html=True)
with c2:
    rc = "#3B8BD4" if metrics["return"]>=0 else "#E24B4A"
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">RENDIMENTO</div><div style="font-size:22px;font-weight:700;color:{rc}">{metrics["return"]:+.1f}%</div><div style="font-size:11px;color:#aaa">alpha: {metrics["alpha"]:+.1f}%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">MAX DRAWDOWN</div><div style="font-size:22px;font-weight:700;color:#E24B4A">{metrics["max_dd"]:.1f}%</div><div style="font-size:11px;color:#aaa">sharpe: {metrics["sharpe"]}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">WIN RATE</div><div style="font-size:22px;font-weight:700;color:#9FE1CB">{metrics["win_rate"]:.1f}%</div><div style="font-size:11px;color:#aaa">{metrics["n_trades"]} trade</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:11px;color:#888">CAPITALE</div><div style="font-size:22px;font-weight:700;color:#3B8BD4">${metrics["final"]:,}</div><div style="font-size:11px;color:#aaa">start: $100,000</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ───────────────────────────────────────────────────────
dates = df.index
n = min(len(eq), len(dates))

# Chart 1: Prezzo + regime overlay
fig1 = go.Figure()
prev_r = int(regimes[0]); si = 0
for i in range(1, len(regimes)):
    if int(regimes[i]) != prev_r or i == len(regimes)-1:
        fig1.add_vrect(x0=dates[si], x1=dates[i],
                       fillcolor=REGIME_COLORS.get(prev_r,"#888"), opacity=0.18, line_width=0)
        prev_r = int(regimes[i]); si = i

fig1.add_trace(go.Scatter(x=dates, y=df["Close"], line=dict(color="#e0e0e0", width=1), name=symbol))
for t in [t for t in trades if t["type"]=="LONG"]:
    fig1.add_trace(go.Scatter(x=[t["date"]], y=[t["entry"]], mode="markers",
                              marker=dict(symbol="triangle-up", size=8, color="#3B8BD4"), showlegend=False))
for t in [t for t in trades if t["type"]=="SHORT"]:
    fig1.add_trace(go.Scatter(x=[t["date"]], y=[t["entry"]], mode="markers",
                              marker=dict(symbol="triangle-down", size=8, color="#E24B4A"), showlegend=False))
fig1.update_layout(height=300, paper_bgcolor="#0f0f0f", plot_bgcolor="#1a1a1a",
                   font=dict(color="#aaa"), margin=dict(l=10,r=10,t=30,b=10),
                   title=dict(text=f"{symbol} {timeframe} — Regime Overlay", font=dict(size=13)),
                   xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222", tickprefix="$"))
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Equity curve
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=dates[:n], y=eq[:n], line=dict(color="#3B8BD4", width=2), name="HMM"), secondary_y=False)
fig2.add_trace(go.Scatter(x=dates[:n], y=bh_c[:n], line=dict(color="#888780", width=1.5, dash="dash"), name="Buy&Hold"), secondary_y=False)
peak = np.maximum.accumulate(eq[:n])
dd_arr = (eq[:n] - peak) / peak * 100
fig2.add_trace(go.Scatter(x=dates[:n], y=dd_arr, line=dict(color="#E24B4A", width=1),
                          fill="tozeroy", fillcolor="rgba(226,75,74,0.1)", name="Drawdown"), secondary_y=True)
fig2.update_layout(height=260, paper_bgcolor="#0f0f0f", plot_bgcolor="#1a1a1a",
                   font=dict(color="#aaa"), margin=dict(l=10,r=10,t=30,b=10),
                   title=dict(text="Equity Curve + Drawdown", font=dict(size=13)),
                   xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222", tickprefix="$"),
                   legend=dict(bgcolor="#1a1a1a", bordercolor="#333"))
fig2.update_yaxes(ticksuffix="%", secondary_y=True, gridcolor="#222")
st.plotly_chart(fig2, use_container_width=True)

# Chart 3: Distribuzione regimi
total = len(regimes)
fig3 = go.Figure(go.Bar(
    x=[REGIME_NAMES[k] for k in range(n_regimes) if k < 7],
    y=[np.sum(regimes==k)/total*100 for k in range(n_regimes) if k < 7],
    marker_color=[REGIME_COLORS.get(k,"#888") for k in range(n_regimes) if k < 7],
    text=[f"{np.sum(regimes==k)/total*100:.0f}%" for k in range(n_regimes) if k < 7],
    textposition="outside"
))
fig3.update_layout(height=200, paper_bgcolor="#0f0f0f", plot_bgcolor="#1a1a1a",
                   font=dict(color="#aaa"), margin=dict(l=10,r=10,t=30,b=10),
                   title=dict(text="Distribuzione regimi", font=dict(size=13)),
                   xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222", ticksuffix="%"))
st.plotly_chart(fig3, use_container_width=True)

# ── Conferme ─────────────────────────────────────────────────────
st.subheader("Conferme segnale")
cols = st.columns(4)
for idx, (name, ok) in enumerate(confs.items()):
    with cols[idx % 4]:
        col = "#9FE1CB" if ok else "#E24B4A"
        sym = "✓" if ok else "✗"
        st.markdown(f'<div style="background:#1a1a1a;border-radius:8px;padding:10px;margin-bottom:8px"><span style="color:#888;font-size:12px">{name}</span><br><span style="color:{col};font-size:16px;font-weight:700">{sym}</span></div>', unsafe_allow_html=True)

st.markdown(f"**{n_pass}/8 conferme** — segnale: **{signal}**")

# ── Trade log ────────────────────────────────────────────────────
st.subheader("📋 Trade Log")
closed = [t for t in trades if t["pnl"] != "-"]
if closed:
    df_t = pd.DataFrame(closed[-50:])
    df_t["Data"] = df_t["date"].apply(lambda x: x.strftime("%d/%m/%Y %H:%M") if hasattr(x,"strftime") else str(x))
    df_t["P&L"]  = df_t["pnl"].apply(lambda x: f"{x:+.2f}%" if isinstance(x,float) else x)
    st.dataframe(df_t[["Data","type","regime","entry","exit","P&L","hold"]].rename(
        columns={"type":"Tipo","regime":"Regime","entry":"Entrata","exit":"Uscita","hold":"Hold"}),
        use_container_width=True, height=300)
else:
    st.info("Nessun trade chiuso nel periodo.")
