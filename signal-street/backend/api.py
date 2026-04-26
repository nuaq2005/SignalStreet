import time
import sys
import os
 
# Make sure sibling modules are importable
sys.path.insert(0, os.path.dirname(__file__))
 
from flask import Flask, request, jsonify
from flask_cors import CORS
 
from data_layer import get_live_features, FEATURE_COLS
from ml_model import predict_from_features, _BUNDLE, CLASSES

app = Flask(__name__)
CORS(app)  # allow React dev-server (localhost:5173 / 3000) to call us
 
 
# ── helpers ───────────────────────────────────────────────────────────
 
def _safe_float(val, fallback=0.0):
    """Convert numpy scalars / NaN to plain Python float."""
    try:
        v = float(val)
        import math
        return fallback if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return fallback
 
 
# ── routes ────────────────────────────────────────────────────────────
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": len(_BUNDLE["models"])})
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Request body: { "ticker": "AAPL" }
 
    Response: all fields the React App.jsx expects, plus extras from the
    richer data_layer (rsi, volatility, spread, price_change, volume).
    """
    body   = request.get_json(force=True, silent=True) or {}
    ticker = str(body.get("ticker", "")).upper().strip()
 
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
 
    # ── fetch live features ───────────────────────────────────────────
    t0 = time.perf_counter()
 
    #try:
    #    live = get_live_features(ticker)
    #except Exception as exc:
    #    return jsonify({"error": f"Could not fetch data for {ticker}: {exc}"}), 502

    live = get_live_features(ticker)
    
    if live is None or "feature_vector" not in live:
        return jsonify({
        "signal": "HOLD",
        "confidence": 0.5,
        "prob_buy": 0.33,
        "prob_sell": 0.33,
        "prob_hold": 0.34,
        "note": "fallback mode"
    }), 200
 
    fv = live["feature_vector"]
 
    # ── predict ───────────────────────────────────────────────────────
    try:
        decision = predict_from_features(fv)
    except Exception as exc:
        return jsonify({"error": f"Model inference failed: {exc}"}), 500
 
    latency_ms = (time.perf_counter() - t0) * 1000
 
    # ── build feature map for transparency ───────────────────────────
    feature_map = {
        col: _safe_float(val)
        for col, val in zip(FEATURE_COLS, fv)
    }
 
    # pull named fields that App.jsx renders
    rsi_raw    = feature_map.get("rsi_14", 50.0) * 1.0   # stored raw in data_layer
    volatility = feature_map.get("volatility", 0.0)
    spread     = feature_map.get("spread", 0.0)
    price_ch   = feature_map.get("price_change", 0.0)
    volume     = feature_map.get("volume", 0.0)
 
    # history: list of {date, close} for chart (last 90 rows)
    history_series = live.get("history")
    history = []
    if history_series is not None:
        df_hist = history_series.reset_index()
        df_hist.columns = ["date", "close"]
        df_hist = df_hist.tail(90)
        history = [
            {"date": str(row["date"])[:10], "close": _safe_float(row["close"])}
            for _, row in df_hist.iterrows()
        ]
 
    return jsonify({
        # ── identity ──────────────────────────────────────────────────
        "ticker":      ticker,
 
        # ── signal ───────────────────────────────────────────────────
        "signal":      decision.signal,
        "confidence":  _safe_float(decision.confidence),
        "prob_buy":    _safe_float(decision.prob_buy),
        "prob_sell":   _safe_float(decision.prob_sell),
        "prob_hold":   _safe_float(decision.prob_hold),
 
        # ── price metrics (App.jsx metrics grid) ──────────────────────
        "price":        _safe_float(live.get("price", 0)),
        "volume":       _safe_float(volume),
        "rsi":          _safe_float(rsi_raw),
        "spread":       _safe_float(spread),
        "volatility":   _safe_float(volatility),
        "price_change": _safe_float(price_ch),
 
        # ── chart ─────────────────────────────────────────────────────
        "history": history,
 
        # ── meta ──────────────────────────────────────────────────────
        "latency_ms":   round(latency_ms, 2),
        "n_models":     len(_BUNDLE["models"]),
        "features":     feature_map,   # full feature vector for debugging
    })
 
 
# ── entry point ───────────────────────────────────────────────────────
 
if __name__ == "__main__":
    # threaded=False keeps numpy thread-safety simple; use gunicorn for prod
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)