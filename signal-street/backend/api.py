import time
import sys
import os
import socket
 
# Make sure sibling modules are importable
sys.path.insert(0, os.path.dirname(__file__))
 
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import logging
 
from data_layer import get_live_features, FEATURE_COLS
from ml_model import get_bundle, predict_from_features, CLASSES

app = Flask(__name__)

# Restrict CORS to known origins (localhost for dev)
CORS(
    app,
    origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=3600,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
 
 
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
 
@app.after_request
def add_security_headers(response):
    """Add security headers to prevent XSS, clickjacking, and other attacks."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'"
    return response


@app.route("/health", methods=["GET"])
def health():
    bundle = get_bundle()
    return jsonify({"status": "ok", "models_loaded": len(bundle["models"])})
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Request body: { "ticker": "AAPL" }
 
    Response: all fields the React App.jsx expects, plus extras from the
    richer data_layer (rsi, volatility, spread, price_change, volume).
    """
    if not request.is_json:
        logger.warning("Non-JSON request received for /analyze")
        return jsonify({"error": "Expected JSON payload"}), 400

    try:
        body = request.get_json(silent=True) or {}
    except Exception as exc:
        logger.warning(f"Invalid JSON in request: {exc}")
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    ticker = str(body.get("ticker", "")).upper().strip()
 
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    
    # Validate ticker: alphanumeric + common symbols (1-10 chars)
    if not re.match(r'^[A-Z0-9\-\.]{1,10}$', ticker):
        logger.warning(f"Invalid ticker format: {ticker}")
        return jsonify({"error": "Invalid ticker format"}), 400
 
    # ── fetch live features ───────────────────────────────────────────
    t0 = time.perf_counter()
 
    try:
        live = get_live_features(ticker)
    except ValueError as exc:
        logger.warning(f"Data fetch failed for {ticker}: {exc}")
        return jsonify({"error": "No data available for this ticker"}), 404
    except Exception as exc:
        logger.error(f"Unexpected error fetching data for {ticker}: {exc}")
        return jsonify({"error": "Service temporarily unavailable"}), 503
    
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
        logger.error(f"Model inference failed for {ticker}: {exc}")
        return jsonify({"error": "Model inference failed"}), 500
 
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
 
    bundle = get_bundle()
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
        "n_models":     len(bundle["models"]),
        "features":     feature_map,   # full feature vector for debugging
    })
 
 
# ── entry point ───────────────────────────────────────────────────────
 
def _port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def run_app():
    preferred_port = int(os.environ.get("BACKEND_PORT", 5000))
    for port in (preferred_port, 5001, 5002):
        if not _port_is_free(port):
            print(f"Port {port} is busy, trying the next available port...")
            continue

        print(f"Starting backend on port {port}...")
        is_production = os.environ.get("ENV", "development") == "production"
        app.run(
            host="0.0.0.0",
            port=port,
            debug=not is_production,
            use_reloader=False,
            threaded=False,
        )
        return

if __name__ == "__main__":
    # threaded=False keeps numpy thread-safety simple; use gunicorn for prod
    run_app()