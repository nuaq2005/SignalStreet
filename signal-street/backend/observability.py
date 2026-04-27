"""
observability.py

Model observability, versioning, and batch inference.

Covers:
  8.  Inference logging: latency, feature ranges, confidence drift, unusual inputs
  9.  Versioning: model version, feature schema version, training data hash
  10. Batch inference: vectorized serving API
  11. Performance benchmarking: latency p50/p99, throughput, memory
"""

import time
import hashlib
import json
import platform
import tracemalloc
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque


# ── Versioning ────────────────────────────────────────────────────────

MODEL_VERSION          = "1.0.0"
FEATURE_SCHEMA_VERSION = "1.0.0"


def hash_training_data(path: str = "data/stocks.csv") -> str:
    """SHA-256 hash of training CSV — detects data changes between runs."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]   # first 12 chars — enough to detect changes
    except FileNotFoundError:
        return "file-not-found"


def get_version_info(data_path: str = "data/stocks.csv") -> dict:
    return {
        "model_version":          MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_data_hash":     hash_training_data(data_path),
        "python":                 platform.python_version(),
        "numpy":                  np.__version__,
    }


# ── Inference log entry ───────────────────────────────────────────────

@dataclass
class InferenceLog:
    timestamp:         float          # unix time
    latency_ms:        float          # inference time in milliseconds
    signal:            str
    confidence:        float
    feature_min:       float          # min value across all features
    feature_max:       float          # max value across all features
    feature_mean:      float
    n_unusual_features:int            # features outside training range
    is_unusual_input:  bool           # flag: likely OOD input
    ticker:            Optional[str]  # if known


@dataclass
class ObservabilityTracker:
    """
    Tracks confidence drift and unusual input rate over a sliding window.
    Attach one of these to your serving layer.
    """
    window_size:        int   = 100
    _recent_confidence: deque = field(default_factory=lambda: deque(maxlen=100))
    _unusual_count:     int   = 0
    _total_count:       int   = 0
    _logs:              list  = field(default_factory=list)

    # Feature ranges learned from training — set these after training
    feature_min: Optional[np.ndarray] = None
    feature_max: Optional[np.ndarray] = None

    def set_feature_ranges(self, X_train: np.ndarray):
        """Call once after training with the training feature matrix."""
        self.feature_min = X_train.min(axis=0)
        self.feature_max = X_train.max(axis=0)

    def log(
        self,
        features:   np.ndarray,
        signal:     str,
        confidence: float,
        latency_ms: float,
        ticker:     Optional[str] = None,
    ) -> InferenceLog:
        """Record one inference. Returns the log entry."""
        n_unusual   = 0
        is_unusual  = False

        if self.feature_min is not None:
            # count features outside training range
            below = np.sum(features < self.feature_min)
            above = np.sum(features > self.feature_max)
            n_unusual  = int(below + above)
            is_unusual = n_unusual > (len(features) * 0.20)  # >20% OOD = flag

        log = InferenceLog(
            timestamp          = time.time(),
            latency_ms         = latency_ms,
            signal             = signal,
            confidence         = confidence,
            feature_min        = float(features.min()),
            feature_max        = float(features.max()),
            feature_mean       = float(features.mean()),
            n_unusual_features = n_unusual,
            is_unusual_input   = is_unusual,
            ticker             = ticker,
        )

        self._recent_confidence.append(confidence)
        self._total_count += 1
        if is_unusual:
            self._unusual_count += 1
        self._logs.append(log)
        return log

    def confidence_drift(self) -> Optional[float]:
        """
        Drift = std of recent confidence scores.
        High drift means model is switching between high/low conviction — unstable.
        """
        if len(self._recent_confidence) < 10:
            return None
        return float(np.std(list(self._recent_confidence)))

    def unusual_input_rate(self) -> float:
        """Fraction of recent inputs flagged as out-of-distribution."""
        if self._total_count == 0:
            return 0.0
        return self._unusual_count / self._total_count

    def summary(self) -> dict:
        confs = list(self._recent_confidence)
        return {
            "total_inferences":    self._total_count,
            "unusual_input_rate":  f"{self.unusual_input_rate():.1%}",
            "confidence_drift":    round(self.confidence_drift() or 0, 4),
            "mean_confidence":     round(float(np.mean(confs)), 4) if confs else 0,
            "p50_latency_ms":      self._percentile_latency(50),
            "p99_latency_ms":      self._percentile_latency(99),
        }

    def _percentile_latency(self, pct: int) -> float:
        latencies = [l.latency_ms for l in self._logs]
        if not latencies:
            return 0.0
        return float(np.percentile(latencies, pct))

    def print_summary(self):
        s = self.summary()
        print("\n── Observability Summary ────────────────────────────────")
        print(f"  Total inferences:   {s['total_inferences']:,}")
        print(f"  Unusual input rate: {s['unusual_input_rate']}")
        print(f"  Confidence drift:   {s['confidence_drift']:.4f}  "
              f"(std of recent confidence scores)")
        print(f"  Mean confidence:    {s['mean_confidence']:.4f}")
        print(f"  Latency p50:        {s['p50_latency_ms']:.2f} ms")
        print(f"  Latency p99:        {s['p99_latency_ms']:.2f} ms")
        print("─────────────────────────────────────────────────────────")


# ── Batch inference ───────────────────────────────────────────────────

def batch_predict(
    feature_matrix: np.ndarray,
    model,                          # NeuralNetwork instance from ml_model.py
    tracker: Optional[ObservabilityTracker] = None,
) -> pd.DataFrame:
    """
    Vectorized batch inference — much faster than calling predict() in a loop.

    Parameters
    ----------
    feature_matrix : shape (N, N_FEATURES) — raw (un-normalised) features
    model          : trained NeuralNetwork from ml_model.py
    tracker        : optional ObservabilityTracker for logging

    Returns
    -------
    DataFrame with columns: signal, confidence, prob_buy, prob_sell, prob_hold
    """
    t0 = time.perf_counter()

    # normalize entire batch at once — vectorized, no Python loop
    X_norm = (feature_matrix - model.mean) / (model.std + 1e-8)

    # forward pass on full batch simultaneously
    probs = model.predict_proba(X_norm)   # shape: (N, 3)

    t1         = time.perf_counter()
    total_ms   = (t1 - t0) * 1000
    per_item   = total_ms / len(feature_matrix) if len(feature_matrix) > 0 else 0

    class_to_idx = {c: i for i, c in enumerate(model.classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    predicted_idx = np.argmax(probs, axis=1)
    signals       = [idx_to_class[i] for i in predicted_idx]
    confidences   = probs.max(axis=1)

    result = pd.DataFrame({
        "signal":     signals,
        "confidence": confidences.round(4),
        "prob_buy":   probs[:, class_to_idx.get("BUY",  0)].round(4),
        "prob_sell":  probs[:, class_to_idx.get("SELL", 1)].round(4),
        "prob_hold":  probs[:, class_to_idx.get("HOLD", 2)].round(4),
    })

    if tracker is not None:
        for i, (feat, sig, conf) in enumerate(
            zip(feature_matrix, signals, confidences)
        ):
            tracker.log(feat, sig, float(conf), per_item)

    return result


# ── Performance benchmark ─────────────────────────────────────────────

def benchmark(
    model,
    n_scenarios: int = 1_000,
    seed:        int = 42,
) -> dict:
    """
    Measures:
      - Startup / warm-up time
      - Throughput (scenarios/sec)
      - Latency p50, p99
      - Memory usage (peak RSS)
    """
    from data_layer import N_FEATURES

    rng      = np.random.default_rng(seed)
    X        = rng.standard_normal((n_scenarios, N_FEATURES)).astype(np.float64)
    X_norm   = (X - model.mean) / (model.std + 1e-8)

    # warm-up — JIT, cache effects
    model.predict_proba(X_norm[:10])

    # memory tracking
    tracemalloc.start()

    latencies = []
    for i in range(n_scenarios):
        t0 = time.perf_counter()
        model.predict_proba(X_norm[i : i + 1])
        latencies.append((time.perf_counter() - t0) * 1000)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # batch throughput
    t0 = time.perf_counter()
    model.predict_proba(X_norm)
    batch_ms = (time.perf_counter() - t0) * 1000

    lat = np.array(latencies)
    result = {
        "n_scenarios":       n_scenarios,
        "throughput_per_sec":round(n_scenarios / (batch_ms / 1000), 0),
        "batch_latency_ms":  round(batch_ms, 2),
        "single_p50_ms":     round(float(np.percentile(lat, 50)), 3),
        "single_p99_ms":     round(float(np.percentile(lat, 99)), 3),
        "peak_memory_kb":    round(peak_mem / 1024, 1),
    }

    print("\n── Performance Benchmark ────────────────────────────────")
    print(f"  Scenarios:      {result['n_scenarios']:,}")
    print(f"  Throughput:     {result['throughput_per_sec']:,.0f} scenarios/sec")
    print(f"  Batch latency:  {result['batch_latency_ms']:.2f} ms  "
          f"(all {n_scenarios:,} at once)")
    print(f"  Single p50:     {result['single_p50_ms']:.3f} ms")
    print(f"  Single p99:     {result['single_p99_ms']:.3f} ms")
    print(f"  Peak memory:    {result['peak_memory_kb']:.1f} KB")
    print("─────────────────────────────────────────────────────────")

    return result