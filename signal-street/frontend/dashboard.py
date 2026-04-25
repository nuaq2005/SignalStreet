"""
dashboard.py

Streamlit dashboard — updated for all improvements.

New sections:
  - Observability panel (confidence drift, OOD inputs, latency)
  - Extended signal panel (prob_buy, prob_sell, prob_hold)
  - Extended stress test table (all 10 properties)
  - Version info footer
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from data_layer import get_live_features
from ml_model import predict_from_features, _MODEL
from runner import run
from observability import ObservabilityTracker, get_version_info, benchmark

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropTest",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .big-signal   { font-size: 52px; font-weight: 800; margin: 0; line-height: 1; }
    .buy          { color: #00c853; }
    .sell         { color: #ff1744; }
    .hold         { color: #ff9100; }
    .conf-label   { font-size: 17px; color: #888; margin-top: 4px; }
    .prob-row     { font-size: 13px; color: #aaa; margin-top: 2px; }
    .warning-box  {
        background: #1a1200; border: 1px solid #ff9100;
        border-radius: 8px; padding: 12px 16px; margin-top: 8px;
    }
            
    .clean-box    {
        background: #001a0a; border: 1px solid #00c853;
        border-radius: 8px; padding: 12px 16px; margin-top: 8px;
    }
    .obs-box      {
        background: #0a0a1a; border: 1px solid #4444ff;
        border-radius: 8px; padding: 12px 16px; margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state: observability tracker ─────────────────────────────
if "tracker" not in st.session_state:
    st.session_state.tracker = ObservabilityTracker(window_size=100)

tracker = st.session_state.tracker

# ── Header ────────────────────────────────────────────────────────────
st.title("📈 PropTest")
st.caption("ML-powered stock signal · stress testing · model observability")
st.divider()

# ── Search ────────────────────────────────────────────────────────────
col_search, col_period = st.columns([3, 1])
with col_search:
    ticker = st.text_input(
        "Stock ticker",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, MSFT, NVDA"
    ).upper().strip()

with col_period:
    period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y"], index=0)

run_button = st.button("🔍 Analyse", type="primary")

if not run_button:
    st.info("Enter a stock ticker above and click **Analyse** to get started.")
    st.stop()

# ── Fetch live data ───────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} from Yahoo Finance..."):
    try:
        live = get_live_features(ticker)
    except Exception as e:
        st.error(f"Could not fetch data for **{ticker}**: {e}")
        st.stop()

# ── Predict with latency tracking ────────────────────────────────────
t0       = time.perf_counter()
decision = predict_from_features(live["feature_vector"])
lat_ms   = (time.perf_counter() - t0) * 1000

tracker.log(
    features   = live["feature_vector"],
    signal     = decision.signal,
    confidence = decision.confidence,
    latency_ms = lat_ms,
    ticker     = ticker,
)

# ── Signal + chart ────────────────────────────────────────────────────
col_sig, col_chart = st.columns([1, 2])

with col_sig:
    st.subheader(f"{ticker} — Latest Signal")
    css_cls = decision.signal.lower()
    st.markdown(
        f'<p class="big-signal {css_cls}">{decision.signal}</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p class="conf-label">{decision.confidence:.0%} confidence '
        f'· {lat_ms:.1f} ms inference</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p class="prob-row">BUY {decision.prob_buy:.0%} · '
        f'SELL {decision.prob_sell:.0%} · '
        f'HOLD {decision.prob_hold:.0%}</p>',
        unsafe_allow_html=True
    )
    st.divider()

    col_a, col_b = st.columns(2)
    col_a.metric("Price",        f"${live['price']:.2f}")
    col_a.metric("Volume",       f"{live['volume']:,.0f}")
    col_a.metric("Spread",       f"{live['spread']:.2%}")
    col_b.metric("Change",       f"{live['price_change']:+.2%}")
    col_b.metric("Volatility",   f"{live['volatility']:.2%}")
    col_b.metric("RSI",          f"{live['rsi']:.1f}")

    # trend and vol regime
    trend_label = "📈 Bull" if live["trend"] > 0 else "📉 Bear"
    vol_label   = "🔥 High" if live["vol_regime"] > 1.2 else "😴 Low"
    st.caption(f"Regime: {trend_label}  ·  Vol: {vol_label}")

with col_chart:
    st.subheader("Price History")
    st.line_chart(live["history"], use_container_width=True)

st.divider()

# ── Observability panel ───────────────────────────────────────────────
st.subheader("🔭 Model Observability")
obs_summary = tracker.summary()
drift       = tracker.confidence_drift()
ood_rate    = tracker.unusual_input_rate()

oc1, oc2, oc3, oc4, oc5 = st.columns(5)
oc1.metric("Inferences",    f"{obs_summary['total_inferences']:,}")
oc2.metric("Conf. Drift",   f"{drift:.3f}" if drift else "—",
           help="Std of recent confidence. High = unstable model.")
oc3.metric("OOD Rate",      obs_summary["unusual_input_rate"],
           help="% inputs with >20% features outside training range.")
oc4.metric("p50 Latency",   f"{obs_summary['p50_latency_ms']:.1f} ms")
oc5.metric("p99 Latency",   f"{obs_summary['p99_latency_ms']:.1f} ms")

if ood_rate > 0.20:
    st.markdown(
        '<div class="obs-box">⚠️ High out-of-distribution input rate. '
        'Model may be operating outside its training distribution.</div>',
        unsafe_allow_html=True
    )

st.divider()

# ── Stress test ───────────────────────────────────────────────────────
st.subheader("🧪 Stress Test — How Much Should You Trust This Signal?")
st.caption(
    "Running 10 behavioural rules across 500+ edge-case scenarios. "
    "Violations are shrunk to their minimal failing case."
)

with st.spinner("Running stress tests..."):
    report = run(n_scenarios=500, seed=42)

# summary metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Scenarios",       f"{report.total_scenarios:,}")
m2.metric("Total Checks",    f"{report.total_tests:,}")
m3.metric("Pass Rate",       f"{report.pass_rate:.1%}")
m4.metric("Violations",      f"{report.total_violations}",
          delta=f"{report.total_violations} issues" if report.total_violations else "none",
          delta_color="inverse" if report.total_violations else "normal")
m5.metric("Throughput",      f"{report.scenarios_per_sec:,.0f}/s")

st.divider()

# weaknesses
if report.weaknesses:
    st.markdown("#### ⚠️ Known Model Weaknesses")
    for w in report.weaknesses:
        st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="clean-box">✅ No weaknesses found across all 10 properties.</div>',
        unsafe_allow_html=True
    )

st.divider()

# property breakdown table
st.markdown("#### Stress Test Breakdown (10 Properties)")
rows = []
for name, counts in report.property_summary.items():
    total = counts["passed"] + counts["failed"]
    rate  = counts["passed"] / total if total > 0 else 1.0
    rows.append({
        "Property":  counts["label"],
        "Passed":    counts["passed"],
        "Failed":    counts["failed"],
        "Pass Rate": rate,
        "Status":    "✅" if counts["failed"] == 0 else f"⚠️ {counts['failed']}",
    })

st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Pass Rate": st.column_config.ProgressColumn(
            "Pass Rate", min_value=0, max_value=1, format="%.1f%%"
        )
    }
)

# violation details
if report.violations:
    st.divider()
    st.markdown("#### Violation Details")
    st.caption(
        "Original scenario, what was mutated, and the minimal case "
        "that still triggers the same failure."
    )
    for i, v in enumerate(report.violations[:5]):
        pr = v.property_result
        sr = v.shrink_result
        with st.expander(f"#{i+1} — {pr.property_name}: {pr.violation_description}"):
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**Original**")
                st.json({
                    "price":      pr.original_scenario.price,
                    "volume":     pr.original_scenario.volume,
                    "spread":     pr.original_scenario.spread,
                    "volatility": pr.original_scenario.volatility,
                    "signal":     pr.original_decision.signal,
                    "confidence": f"{pr.original_decision.confidence:.0%}",
                })
            with cb:
                st.markdown("**After mutation**")
                st.json({
                    "price":      pr.mutated_scenario.price,
                    "volume":     pr.mutated_scenario.volume,
                    "spread":     pr.mutated_scenario.spread,
                    "volatility": pr.mutated_scenario.volatility,
                    "signal":     pr.mutated_decision.signal,
                    "confidence": f"{pr.mutated_decision.confidence:.0%}",
                })
            with cc:
                st.markdown(f"**Minimal case** *(shrunk in {sr.steps_taken} steps)*")
                st.json({
                    "price":      sr.minimal_scenario.price,
                    "volume":     sr.minimal_scenario.volume,
                    "spread":     sr.minimal_scenario.spread,
                    "volatility": sr.minimal_scenario.volatility,
                })

st.divider()

# ── Version info ──────────────────────────────────────────────────────
info = get_version_info()
st.caption(
    f"Model v{info['model_version']} · "
    f"Schema v{info['feature_schema_version']} · "
    f"Data hash: {info['training_data_hash']} · "
    f"PropTest is a research tool — not financial advice."
)