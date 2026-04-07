"""
Streamlit dashboard for the Energy Trading AI system.

Displays:
  - Forecast metrics and actual-vs-predicted charts (Tab 1)
  - Portfolio performance and trading signals (Tab 2)
  - Data quality report and feature drift (Tab 3)
  - LLM agent single + batch decisions (Tab 4)
  - Full downloadable trade log (Tab 5)

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import run_pipeline, data_quality_report, detect_drift
from src.feature_engineering import build_features
from src.forecasting import train_xgboost, train_lightgbm, evaluate
from src.trading_strategy import generate_signals, simulate_pnl, build_trade_log
from src.llm_agent import agent_decision, batch_agent_analysis
from config.settings import FEATURES, TARGET_COL, TRAIN_SPLIT, DATE_COL

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Energy Trading AI ⚡", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.modebar-container { display: none !important; }
h1 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.5px; }

.flip-card {
    background: transparent;
    min-width: 155px;
    max-width: 200px;
    height: 115px;
    perspective: 1000px;
    display: inline-flex;
    margin: 5px;
    cursor: pointer;
    flex-shrink: 0;
}
.flip-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    text-align: center;
    transition: transform 0.6s cubic-bezier(.4,2,.55,1), box-shadow 0.3s ease;
    transform-style: preserve-3d;
    border-radius: 14px;
}
.flip-card:hover .flip-card-inner {
    transform: rotateY(180deg) translateY(-5px);
    box-shadow: 0 16px 32px rgba(0,0,0,0.35);
}
.flip-card-front, .flip-card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    -webkit-backface-visibility: hidden;
    backface-visibility: hidden;
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 8px 10px;
    box-sizing: border-box;
}
.flip-card-front {
    background: linear-gradient(145deg, #0d1b2a, #1b2a3b);
    border: 1px solid rgba(255,255,255,0.07);
}
.flip-card-back {
    background: linear-gradient(145deg, #1a1060, #3b1f6e);
    transform: rotateY(180deg);
    border: 1px solid rgba(255,255,255,0.12);
}
.metric-label {
    font-size: 9.5px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-bottom: 4px;
    white-space: nowrap;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 700;
    color: #e2eeff;
    line-height: 1.05;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}
.metric-value.wide  { font-size: 15px; }
.metric-value.xwide { font-size: 12px; }
.metric-unit {
    font-size: 9px;
    color: rgba(255,255,255,0.32);
    margin-top: 3px;
    white-space: nowrap;
}
.flip-back-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10px;
    font-weight: 700;
    color: #d4b8ff;
    margin-bottom: 5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.flip-back-desc {
    font-size: 10px;
    color: rgba(255,255,255,0.85);
    line-height: 1.5;
    text-align: center;
}
.metrics-row {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    margin-bottom: 18px;
}
.info-panel {
    background: linear-gradient(135deg, #0d1b2a 0%, #111e2e 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 13px 18px;
    margin-top: 2px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #b8c9e0;
    line-height: 1.7;
}
.info-panel p { margin: 0 0 6px 0; }
.info-panel p:last-child { margin-bottom: 0; }
.cdot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    vertical-align: middle;
    margin: 0 2px 1px 3px;
    flex-shrink: 0;
}
.section-rule {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 20px 0 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Energy Demand Forecasting + AI Trading Assistant")
st.caption("XGBoost · LightGBM · Prophet · Mistral-7B via Hugging Face")


# ── Helper utilities ──────────────────────────────────────────────────────────

def _size_class(value: str) -> str:
    n = len(value)
    if n > 13:
        return "xwide"
    if n > 9:
        return "wide"
    return ""


def flip_card(label: str, value: str, unit: str, title: str, desc: str) -> str:
    cls = _size_class(value)
    return f"""
<div class="flip-card">
  <div class="flip-card-inner">
    <div class="flip-card-front">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
      <div class="metric-unit">{unit}</div>
    </div>
    <div class="flip-card-back">
      <div class="flip-back-title">{title}</div>
      <div class="flip-back-desc">{desc}</div>
    </div>
  </div>
</div>"""


# ── KEY FIX: use st.empty() so the HTML is replaced, not diffed ──────────────
# Streamlit's virtual DOM diffs st.markdown() by position/key and skips
# re-renders when it thinks the element hasn't changed.  Writing into an
# st.empty() container always forces a full replacement of the inner HTML,
# so the new card values actually appear on screen.

def metrics_row(cards: list, slot=None) -> None:
    html = '<div class="metrics-row">'
    for label, value, unit, title, desc in cards:
        html += flip_card(label, value, unit, title, desc)
    html += "</div>"
    if slot is None:
        slot = st.empty()
    slot.markdown(html, unsafe_allow_html=True)


def color_dot(hex_color: str) -> str:
    return f'<span class="cdot" style="background:{hex_color};"></span>'


def info_expander(button_label: str, html_body: str) -> None:
    with st.expander(f"ℹ️  {button_label}", expanded=False):
        st.markdown(f'<div class="info-panel">{html_body}</div>', unsafe_allow_html=True)


def _plotly_cfg() -> dict:
    return {"displayModeBar": False}


# ── Data loading (cached — raw data never changes) ────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = run_pipeline()
    return build_features(df)


@st.cache_data
def get_trained_models(df: pd.DataFrame):
    """Train models once and cache them. Returns models + full train/test splits."""
    n     = int(len(df) * TRAIN_SPLIT)
    train = df.iloc[:n]
    test  = df.iloc[n:]
    xgb   = train_xgboost(train[FEATURES], train["return"])
    lgbm  = train_lightgbm(train[FEATURES], train["return"])
    return xgb, lgbm, test, train


# ── Boot-time data load ───────────────────────────────────────────────────────

with st.spinner("Loading data and training models..."):
    df                      = load_data()
    xgb, lgbm, test, train = get_trained_models(df)

# Dataset-level quality stats — computed once, intentionally not window-specific
dq = data_quality_report(df)

market_stats = {
    "avg_return": float(train["return"].tail(50).mean()),
    "volatility": float(train["return"].tail(50).std()),
}

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Forecasting model", ["XGBoost", "LightGBM"])
n_days       = st.sidebar.slider("Days to display", 30, 365, 90)


st.sidebar.markdown("---")
st.sidebar.markdown("**LLM Agent**")
st.sidebar.caption("Model: `mistralai/Mistral-7B-Instruct-v0.2`")

# ── Windowed computations ─────────────────────────────────────────────────────
# Nothing below is cached — recomputes fresh on every slider/selectbox change.

n_hours  = n_days * 24
test_win = test.tail(n_hours).copy()

model   = xgb if model_choice == "XGBoost" else lgbm
preds   = model.predict(test_win[FEATURES]).tolist()
signals = generate_signals(list(preds))

metrics = evaluate(
    y_true=test_win["return"].values,
    y_pred=np.array(preds)
)

capital_hist, pnl_metrics = simulate_pnl(
    preds.copy(),
    signals.copy(),
    test_win.copy()
)

drift = detect_drift(
    train.copy(),
    test_win.copy(),
    FEATURES
)

_full_preds   = model.predict(test[FEATURES]).tolist()
_full_signals = generate_signals(_full_preds)


# ── Tab layout ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast", "💰 Strategy & PnL", "🔍 Data Quality",
    "🤖 LLM Agent", "📋 Trade Log",
])


# ── Tab 1 — Forecast ──────────────────────────────────────────────────────────

with tab1:
    st.subheader(f"Return Prediction — {model_choice}  ·  last {n_days} days")

    # Each metrics_row gets its own pre-created empty slot so Streamlit
    # always replaces the HTML rather than skipping the diff.
    slot_t1 = st.empty()
    metrics_row([
        (
            "MAE", f"{metrics['MAE']:.4f}", "avg error",
            "MAE",
            "On average, how far off the energy return predictions are. "
            "Smaller = the forecast is closer to what actually happened."
        ),
        (
            "RMSE", f"{metrics['RMSE']:.4f}", "error magnitude",
            "RMSE",
            "Like MAE but big prediction misses count more heavily. "
            "A high RMSE means the model occasionally makes very wrong calls."
        ),
        (
            "MAPE %", f"{metrics['MAPE%']:.2f}%", "% off on avg",
            "MAPE %",
            "The average prediction error as a percentage. "
            "5% means the model is typically 5% away from the true energy return."
        ),
    ], slot=slot_t1)

    # Chart 1: Actual vs Predicted
    dates = pd.to_datetime(
        test_win[DATE_COL].values if DATE_COL in test_win.columns else test_win.index
    )
    viz   = pd.DataFrame(
        {"actual": test_win["return"].values, "predicted": preds}, index=dates
    )
    daily = viz.resample("D").mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=daily.index, y=daily["actual"],
        name="Actual", line=dict(color="#3B8BD4", width=1.8),
    ))
    fig1.add_trace(go.Scatter(
        x=daily.index, y=daily["predicted"],
        name="Predicted", line=dict(color="#F59E0B", width=1.8),
    ))
    fig1.update_layout(
        title=f"Daily Average Return: Actual vs Predicted ({n_days} days)",
        xaxis_title="Date", yaxis_title="Return", height=400,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig1, width="stretch", config=_plotly_cfg())

    info_expander("Info — Daily Avg Return", f"""
<p>This chart compares what energy prices <strong>actually did</strong> versus what the model <strong>predicted</strong> each day.</p>
<p>
  {color_dot('#3B8BD4')} <strong>Blue</strong> — the real energy return for that day, calculated from actual PJM load data.<br>
  {color_dot('#F59E0B')} <strong>Orange</strong> — the model's prediction made before the day played out.
</p>
<p>When both lines move together closely, the model is reading the market well. Large gaps are days where the forecast missed — this usually happens around holidays, heat waves, or cold snaps that the model hadn't seen during training.</p>
""")

    # Chart 2: Feature Importance
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader(f"Feature Importance — {model_choice}")

    importances = model.feature_importances_
    fi = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": importances,
    }).sort_values("Importance")

    color_scale = "teal" if model_choice == "XGBoost" else "Blues"
    fig2 = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=color_scale,
        title=f"{model_choice} Feature Importance",
    )
    fig2.update_layout(
        plot_bgcolor="rgba(13,27,42,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig2, width="stretch", config=_plotly_cfg())

    info_expander("Info — Feature Importance", f"""
<p>Shows which inputs the model leans on most when deciding what energy returns will look like.</p>
<p><strong>Longer bar = more influential.</strong> The model gives that feature more weight when it splits its decision trees.</p>
<p>In most energy datasets:</p>
<p>
  • <strong>lag_24 / lag_168</strong> — yesterday's and last week's demand are usually the top predictors, because energy demand is strongly habitual.<br>
  • <strong>rolling_mean_24</strong> — the recent trend smooths out hourly noise.<br>
  • <strong>hour / dayofweek</strong> — time-of-day and weekday patterns capture commuter and industrial cycles.
</p>
<p>If a single feature dominates with very high importance, that can sometimes indicate overfitting — the model has memorised one pattern too heavily.</p>
<p><em>Note: Feature importance reflects the trained model weights and switches when you change the model selector above.</em></p>
""")


# ── Tab 2 — Strategy & PnL ────────────────────────────────────────────────────

with tab2:
    st.subheader(f"Portfolio Performance  ·  last {n_days} days")

    raw_profit = pnl_metrics["total_profit"]
    if abs(raw_profit) >= 1_000_000:
        profit_str = f"${raw_profit / 1_000_000:.2f}M"
    elif abs(raw_profit) >= 1_000:
        profit_str = f"${raw_profit / 1_000:.1f}K"
    else:
        profit_str = f"${raw_profit:.0f}"

    slot_t2 = st.empty()
    metrics_row([
        (
            "Total Profit", profit_str, "from $10K start",
            "Total Profit",
            "How much the strategy made or lost from a $10,000 starting pot "
            "by following the model's buy/sell signals on energy prices."
        ),
        (
            "Sharpe Ratio", f"{pnl_metrics['sharpe_ratio']:.3f}", "return / risk",
            "Sharpe Ratio",
            "Is the profit worth the risk? Above 1 is healthy, above 2 is excellent. "
            "Below 0 means the strategy lost money vs doing nothing."
        ),
        (
            "Max Drawdown", f"{pnl_metrics['max_drawdown']*100:.1f}%", "worst dip",
            "Max Drawdown",
            "The steepest portfolio drop from a peak before recovery. "
            "-12% means the strategy temporarily lost $1,200 of every $10,000."
        ),
        (
            "Win Rate", f"{pnl_metrics['win_rate']*100:.1f}%", "profitable hrs",
            "Win Rate",
            "Out of every 100 active hours, how many ended with the portfolio "
            "higher than when that hour started."
        ),
    ], slot=slot_t2)

    # Chart 3: Capital Curve
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        y=capital_hist, fill="tozeroy",
        name="Capital ($)",
        line=dict(color="#10b981", width=2),
        fillcolor="rgba(16,185,129,0.12)",
    ))
    fig3.update_layout(
        title=f"Portfolio Capital Curve ({n_days} days)",
        xaxis_title="Hour", yaxis_title="Capital ($)", height=350,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig3, width="stretch", config=_plotly_cfg())

    info_expander("Info — Portfolio Capital Curve", f"""
<p>Tracks the total value of the portfolio <strong>hour-by-hour</strong> across the selected window, starting from <strong>$10,000</strong>.</p>
<p>
  {color_dot('#10b981')} <strong>Green fill</strong> — the portfolio value at each hour. Rising = winning streak. Falling = losing run.
</p>
<p>A healthy strategy shows a <strong>gradual upward slope</strong> with shallow, short dips. A strategy that soars quickly then collapses is likely overfitting the training data.</p>
<p>The deepest trough visible in this chart corresponds to the <strong>Max Drawdown</strong> metric shown in the cards above.</p>
""")

    # Chart 4: Trading Signals
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader("Trading Signals")

    sig_dates = pd.to_datetime(
        test_win[DATE_COL].values if DATE_COL in test_win.columns else test_win.index
    )
    sig_df = pd.DataFrame(
        {"prediction": preds, "signal": signals}, index=sig_dates
    )
    sig_daily = sig_df.resample("D").agg({
        "prediction": "mean",
        "signal":     lambda x: x.value_counts().idxmax(),
    })

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=sig_daily.index, y=sig_daily["prediction"],
        name="Prediction", line=dict(color="#818cf8", width=1.5),
    ))
    buys  = sig_daily[sig_daily["signal"] == 1]
    sells = sig_daily[sig_daily["signal"] == -1]
    fig4.add_trace(go.Scatter(
        x=buys.index, y=buys["prediction"], mode="markers",
        name="Buy",
        marker=dict(symbol="triangle-up", color="#22c55e", size=10),
    ))
    fig4.add_trace(go.Scatter(
        x=sells.index, y=sells["prediction"], mode="markers",
        name="Sell",
        marker=dict(symbol="triangle-down", color="#ef4444", size=10),
    ))
    fig4.update_layout(
        title=f"Daily Signal Chart ({n_days} days)", height=350,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig4, width="stretch", config=_plotly_cfg())

    sig_counts = sig_daily["signal"].value_counts().rename({1: "BUY", -1: "SELL", 0: "HOLD"})
    st.caption(f"Signal distribution (daily, {n_days} days): {sig_counts.to_dict()}")

    info_expander("Info — Trading Signals", f"""
<p>Shows <strong>when and which direction</strong> the model decided to trade each day.</p>
<p>
  {color_dot('#818cf8')} <strong>Purple line</strong> — the model's predicted energy return for that day.<br>
  {color_dot('#22c55e')} <strong>Green ▲</strong> — BUY signal: the model predicted prices would rise above the confidence threshold.<br>
  {color_dot('#ef4444')} <strong>Red ▼</strong> — SELL signal: the model predicted prices would fall below the confidence threshold.
</p>
<p>Days with <strong>no marker</strong> are HOLD periods — the prediction wasn't strong enough in either direction to justify a trade.</p>
<p><em>Tip: if you see almost no BUY signals, lower the THRESHOLD value in config/settings.py.</em></p>
""")


# ── Tab 3 — Data Quality ──────────────────────────────────────────────────────

with tab3:
    st.subheader("Data Quality Report")

    # Data quality cards are dataset-level — intentionally stable, no slot needed
    slot_t3_dq = st.empty()
    metrics_row([
        (
            "Missing", str(dq["missing_values"]), "filled gaps",
            "Missing Values",
            "Hours in the raw PJME dataset where the meter reading was absent. "
            "Filled in automatically — too many gaps means the data may be unreliable."
        ),
        (
            "Anomalies", str(dq["anomalies"]), "unusual hours",
            "Anomalies",
            "Hours where energy demand spiked or crashed far beyond the normal range. "
            "Could be a grid event, sensor fault, or extreme weather."
        ),
        (
            "Duplicates", str(dq["duplicates"]), "duplicate rows",
            "Duplicate Rows",
            "Rows sharing the same timestamp, usually from daylight saving clock changes. "
            "Removed automatically so the model sees a clean hourly sequence."
        ),
        (
            "Volatility", f"{dq['volatility']:,.0f} MW", "std dev",
            "Load Volatility",
            "How much energy demand swings hour-to-hour across the whole dataset. "
            "High volatility makes every forecast harder."
        ),
    ], slot=slot_t3_dq)

    st.subheader(f"Feature Drift (Train → last {n_days} days of test)")

    drift_df = pd.DataFrame({
        "Feature":     list(drift.keys()),
        "Drift Score": list(drift.values()),
    }).sort_values("Drift Score", ascending=False)

    fig5 = px.bar(
        drift_df, x="Feature", y="Drift Score",
        color="Drift Score", color_continuous_scale="Reds",
        title=f"Feature Drift Scores — {n_days}-day window (higher = more drift)",
    )
    fig5.update_layout(
        plot_bgcolor="#000000",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig5, width="stretch", config=_plotly_cfg())

    info_expander("Info — Feature Drift", f"""
<p>Compares how each input variable <strong>behaved during training</strong> versus how it behaves <strong>in the selected window</strong>.</p>
<p>
  {color_dot('#ef4444')} <strong>Tall red bar</strong> — that feature has shifted significantly. The model learned patterns from the training period that may no longer apply.<br>
  {color_dot('#fca5a5')} <strong>Short/light bar</strong> — the feature is stable across both periods. The model's learned patterns still hold.
</p>
<p>If many features show high drift, the model should be <strong>retrained on more recent data</strong>, or a rolling-window retraining schedule should be introduced.</p>
""")

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader(f"Load Distribution  ·  last {n_days} days")

    fig6 = px.histogram(
        test_win, x=TARGET_COL, nbins=80,
        color_discrete_sequence=["#818cf8"],
        title=f"Energy Load Distribution — {n_days}-day window (MW)",
    )
    fig6.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig6, width="stretch", config=_plotly_cfg())

    info_expander("Info — Load Distribution", f"""
<p>Shows <strong>how often each demand level appears</strong> across the selected window.</p>
<p>
  {color_dot('#818cf8')} <strong>Purple bars</strong> — each bar is a demand range (MW). Taller bar = more hours spent at that demand level.
</p>
<p>The <strong>tallest central bars</strong> are the most common demand range — where the model is most accurate, because it has seen plenty of training examples there.</p>
<p>The <strong>thin tails</strong> on either side are rare extreme-demand hours (heat waves, cold snaps). These are where prediction errors tend to be largest.</p>
""")


# ── Tab 4 — LLM Agent ─────────────────────────────────────────────────────────

with tab4:
    st.subheader("🤖 Mistral-7B Trading Agent")
    st.info("Uses Hugging Face Inference API (free). Falls back to rule-based engine if token missing.")

    idx = st.number_input(
        "Select signal index to analyse", 0, len(_full_preds) - 1, 0, step=1
    )
    row = {
        "prediction": _full_preds[idx],
        "signal":     _full_signals[idx],
        "confidence": abs(_full_preds[idx]),
        "price":      float(test["price"].iloc[idx]),
    }
    signal_label = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(row["signal"], "HOLD")

    slot_t4 = st.empty()
    metrics_row([
        (
            "Prediction", f"{row['prediction']:.5f}", "model output",
            "Prediction",
            "The raw number the model produced for this hour. "
            "Positive = energy prices expected to rise; negative = expected to fall."
        ),
        (
            "Signal", signal_label, "",
            "Signal",
            "The trade the model recommends. BUY = prices rising, "
            "SELL = prices falling, HOLD = not confident enough to act."
        ),
        (
            "Confidence", f"{row['confidence']:.5f}", "signal strength",
            "Confidence",
            "How strongly the model feels about this signal. "
            "Higher = more certain. Low confidence usually causes the AI agent to reject the trade."
        ),
        (
            "Price", f"${row['price']:.2f}", "USD/MWh",
            "Energy Price",
            "The estimated electricity price at this hour, derived from demand level. "
            "High-demand hours push the price up; quiet overnight hours push it down."
        ),
    ], slot=slot_t4)

    info_expander("About these metrics — LLM Agent Inputs", f"""
<p>These four values are fed directly into the AI agent as context before it makes its decision.</p>
<p>
  • <strong>Prediction</strong> — the raw ML model output. The agent uses its sign and magnitude to judge direction and conviction.<br>
  • <strong>Signal</strong> — the suggested action derived from the prediction. The agent may override this if other risk factors are too high.<br>
  • <strong>Confidence</strong> — signals with confidence below ~0.001 are typically rejected by the agent regardless of direction.<br>
  • <strong>Price</strong> — gives the agent market context. Very high or low prices relative to normal can flag unusual conditions.
</p>
<p>The agent also receives data quality stats (anomalies, drift) and market volatility before deciding to APPROVE or REJECT the trade.</p>
""")

    if st.button("🤖 Get Agent Decision", type="primary"):
        with st.spinner("Querying agent..."):
            decision = agent_decision(row, market_stats, dq, drift)
        st.success("Agent Response:")
        st.markdown(f"```\n{decision}\n```")

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader("Batch Analysis")
    n_batch = st.slider("Number of signals to analyse", 1, 10, 3)
    if st.button("Run Batch Analysis"):
        log = build_trade_log(test, _full_preds, _full_signals)
        with st.spinner(f"Running agent on {n_batch} signals..."):
            results = batch_agent_analysis(
                log, market_stats, dq, drift, n_samples=n_batch
            )
        st.dataframe(pd.DataFrame(results), width="stretch")


# ── Tab 5 — Trade Log ─────────────────────────────────────────────────────────

with tab5:
    st.subheader(f"Trade Log  ·  last {n_days} days")

    log = build_trade_log(
        test_win.copy(),
        list(preds),
        list(signals)
    )
    st.dataframe(log, width="stretch")

    csv = log.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Trade Log CSV", csv, "trade_log.csv", "text/csv"
    )