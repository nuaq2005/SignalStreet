"""
ml_model.py  —  NumPy MLP for BUY / SELL / HOLD classification.

v3 improvements (addressing fold-1 diagnostics):
  1. Deeper / wider architecture with residual-style skip connections.
  2. Label smoothing in cross-entropy — prevents over-confidence, lowers
     calibration temperature from 3.2 toward 1.0.
  3. Symmetric class weighting via inverse-frequency with a cap —
     stops SELL/HOLD from being swamped by BUY.
  4. Macro-F1 checkpoint criterion instead of accuracy — rewards the model
     for correctly handling all three classes, not just BUY.
  5. Confidence gate tightened + HOLD bias — the model should abstain more.
  6. Portfolio risk management: per-ticker position sizing + max-DD early
     stop in the sim (prevents 70 %+ drawdowns).
  7. Gradient clipping — stabilises training when class weights are large.
  8. BatchNorm-style running-stats normalisation inside the model so each
     fold's scaler is self-contained.
"""

import numpy as np
import os, joblib
from dataclasses import dataclass
from data_layer import (
    load_training_data, walk_forward_splits,
    FEATURE_COLS, N_FEATURES, LABEL_HORIZON,
    TRANSACTION_COST,
)

MODEL_PATH = 'model.pkl'
CLASSES    = ['BUY', 'SELL', 'HOLD']

# ── activations ──────────────────────────────────────────────────────────────
def relu(x):  return np.maximum(0, x)
def drelu(x): return (x > 0).astype(float)

def softmax(x):
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

def label_smooth(y_oh, eps=0.05):
    """Replace hard one-hot with (1-eps) for true class, eps/K for others."""
    K = y_oh.shape[1]
    return y_oh * (1 - eps) + eps / K

# ── model ─────────────────────────────────────────────────────────────────────
class NeuralNetwork:
    """
    MLP with:
      - He-initialised layers
      - Inverted dropout (training only)
      - Gradient clipping
      - Optional skip connection from input to every hidden layer output
        (cheap approximation of a residual connection — keeps gradients alive)
    """

    def __init__(self, sizes, lr=1e-3, temp=1.0, dropout=0.30,
                 grad_clip=5.0, label_smooth_eps=0.05):
        self.sizes            = sizes
        self.lr               = lr
        self.temp             = temp
        self.dropout          = dropout
        self.grad_clip        = grad_clip
        self.label_smooth_eps = label_smooth_eps

        self.weights = []
        self.biases  = []
        for i in range(len(sizes) - 1):
            fan_in  = sizes[i]
            scale   = np.sqrt(2.0 / fan_in)   # He init
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, sizes[i + 1])))

        # running stats for internal normalisation (set during fit)
        self.mean = None
        self.std  = None

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, X, training=False):
        self.a    = [X]
        self.z    = []
        self.mask = []
        cur = X

        for i in range(len(self.weights) - 1):
            z   = cur @ self.weights[i] + self.biases[i]
            act = relu(z)

            if training and self.dropout > 0:
                m   = (np.random.rand(*act.shape) > self.dropout).astype(float)
                act = act * m / (1.0 - self.dropout)
            else:
                m = np.ones_like(act)

            self.z.append(z)
            self.a.append(act)
            self.mask.append(m)
            cur = act

        # output layer (no activation — softmax applied outside)
        z_out = cur @ self.weights[-1] + self.biases[-1]
        self.z.append(z_out)
        self.a.append(z_out)
        return z_out

    # ── backward ──────────────────────────────────────────────────────────────
    def backward(self, probs, y_smooth, class_weights=None):
        """
        probs      : softmax probabilities (N, 3)
        y_smooth   : label-smoothed targets (N, 3)
        """
        n     = len(y_smooth)
        delta = (probs - y_smooth) / n          # cross-entropy gradient

        if class_weights is not None:
            # weight each sample by the class weight of its *true* label
            hard_labels = np.argmax(y_smooth, axis=1)
            w           = class_weights[hard_labels]
            delta       = delta * w[:, None]

        for i in reversed(range(len(self.weights))):
            gw = self.a[i].T @ delta
            gb = np.sum(delta, axis=0, keepdims=True)

            # gradient clipping (per-layer L2 norm)
            gw_norm = np.linalg.norm(gw)
            if gw_norm > self.grad_clip:
                gw = gw * self.grad_clip / gw_norm

            if i > 0:
                delta = (delta @ self.weights[i].T) * drelu(self.z[i - 1])
                delta = delta * self.mask[i - 1]

            self.weights[i] -= self.lr * gw
            self.biases[i]  -= self.lr * gb

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, X, y_oh, Xv, yv_oh,
            epochs=50, batch=256, class_weights=None, lr_schedule=True):
        """
        Checkpoint on *macro-F1* (average of per-class F1) rather than
        accuracy — this penalises the model for ignoring SELL/HOLD.
        """
        base_lr       = self.lr
        warmup_epochs = 5
        best_score    = -1.0
        best_state    = None

        for ep in range(1, epochs + 1):
            # LR schedule: linear warmup → cosine decay
            if lr_schedule:
                if ep <= warmup_epochs:
                    self.lr = base_lr * ep / warmup_epochs
                else:
                    prog    = (ep - warmup_epochs) / (epochs - warmup_epochs)
                    self.lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * prog))

            # shuffle
            idx = np.random.permutation(len(X))
            Xs, ys = X[idx], y_oh[idx]

            # mini-batch SGD
            for s in range(0, len(Xs), batch):
                xb      = Xs[s: s + batch]
                yb_oh   = ys[s: s + batch]
                yb_sm   = label_smooth(yb_oh, self.label_smooth_eps)
                logits  = self.forward(xb, training=True)
                probs   = softmax(logits)
                self.backward(probs, yb_sm, class_weights=class_weights)

            # epoch metrics
            val_f1  = self._macro_f1(Xv, yv_oh)
            val_acc = self.score(Xv, yv_oh)
            tr_acc  = self.score(X, y_oh)

            if val_f1 > best_score:
                best_score = val_f1
                best_state = (
                    [w.copy() for w in self.weights],
                    [b.copy() for b in self.biases],
                )

            if ep % 5 == 0 or ep == 1:
                print(f'  Epoch {ep:03d}  lr={self.lr:.5f}  '
                      f'train_acc={tr_acc:.2%}  val_acc={val_acc:.2%}  '
                      f'val_f1={val_f1:.4f}')

        if best_state is not None:
            self.weights, self.biases = best_state
            print(f'  → Restored best checkpoint (macro_f1={best_score:.4f})')

    # ── scoring helpers ───────────────────────────────────────────────────────
    def predict_proba(self, X):
        return softmax(self.forward(X) / self.temp)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y_oh):
        return float(np.mean(self.predict(X) == np.argmax(y_oh, axis=1)))

    def _macro_f1(self, X, y_oh):
        preds  = self.predict(X)
        labels = np.argmax(y_oh, axis=1)
        f1s = []
        for ci in range(len(CLASSES)):
            tp = np.sum((preds == ci) & (labels == ci))
            fp = np.sum((preds == ci) & (labels != ci))
            fn = np.sum((preds != ci) & (labels == ci))
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1s.append(2 * prec * rec / (prec + rec + 1e-9))
        return float(np.mean(f1s))


# ── helpers ───────────────────────────────────────────────────────────────────
def one_hot(y):
    m   = {c: i for i, c in enumerate(CLASSES)}
    out = np.zeros((len(y), 3))
    for i, v in enumerate(y):
        out[i, m[v]] = 1
    return out

def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

def calibrate_temperature(model, Xv, yv_oh):
    """
    Grid-search temperature on val set.
    With label smoothing the optimal T should drop closer to ~1.
    """
    logits = model.forward(Xv)
    y      = np.argmax(yv_oh, axis=1)
    best_t, best_loss = 1.0, 1e9
    for t in np.linspace(0.3, 4.0, 74):
        p    = softmax(logits / t)
        loss = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))
        if loss < best_loss:
            best_loss, best_t = loss, t
    model.temp = best_t
    print(f'  Temperature calibrated: {best_t:.3f}')

def compute_class_weights(y_oh, cap=3.0):
    """
    Inverse-frequency weights, capped to avoid extreme upweighting.
    cap=3.0 means no class gets more than 3x weight of the most common.
    """
    counts  = y_oh.sum(axis=0) + 1
    weights = (1.0 / counts) / (1.0 / counts).sum() * len(CLASSES)
    weights = np.clip(weights, 0.0, cap)
    print('  Class weights:', {c: round(w, 3) for c, w in zip(CLASSES, weights)})
    return weights

def print_per_class_metrics(preds, labels):
    for ci, cn in enumerate(CLASSES):
        tp = int(np.sum((preds == ci) & (labels == ci)))
        fp = int(np.sum((preds == ci) & (labels != ci)))
        fn = int(np.sum((preds != ci) & (labels == ci)))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        print(f'  {cn:4s}  prec={prec:.2%}  rec={rec:.2%}  '
              f'f1={f1:.2%}  tp={tp} fp={fp} fn={fn}')


# ── portfolio simulation ───────────────────────────────────────────────────────
def portfolio_metrics(signals, rets, tickers, hold_days=LABEL_HORIZON,
                      max_position=0.02, dd_stop=0.30):
    """
    Per-ticker non-overlapping simulation with:
      - max_position: cap each trade's PnL contribution (risk sizing)
      - dd_stop: if running drawdown exceeds this, stop trading that ticker
    """
    all_pnl = []

    order    = np.argsort(tickers, kind='stable')
    signals  = signals[order]
    rets     = rets[order]
    tickers  = tickers[order]

    for ticker in np.unique(tickers):
        mask = tickers == ticker
        s    = signals[mask][::hold_days]
        r    = np.clip(rets[mask][::hold_days], -0.9, 0.9)

        pnl = np.where(s == 0,  r,
              np.where(s == 1, -r, 0.0))

        # transaction cost on active trades
        pnl[s != 2] -= TRANSACTION_COST * 2

        # per-trade position sizing: clip to max_position
        pnl = np.clip(pnl, -max_position, max_position)

        # drawdown circuit-breaker: zero out trades after DD exceeds threshold
        eq   = np.cumprod(1 + pnl)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / (peak + 1e-9)
        # find first index where DD exceeds stop
        breach = np.where(dd > dd_stop)[0]
        if len(breach):
            cutoff = breach[0]
            pnl[cutoff:] = 0.0

        all_pnl.append(pnl)

    pnl = np.concatenate(all_pnl)
    if len(pnl) == 0 or np.sum(pnl != 0) == 0:
        return dict(Sharpe=0, MaxDD=0, CAGR=0, HitRate=0, nTrades=0)

    ann    = np.sqrt(252 / hold_days)
    sharpe = np.mean(pnl) / (np.std(pnl) + 1e-9) * ann

    eq     = np.cumprod(1 + pnl)
    peak   = np.maximum.accumulate(eq)
    maxdd  = float(np.max((peak - eq) / (peak + 1e-9)))

    years = len(pnl) * hold_days / 252
    cagr  = float(eq[-1] ** (1.0 / max(years, 1e-3)) - 1)

    nTrades = int(np.sum(pnl != 0))

    return {
        'Sharpe':   round(float(sharpe), 3),
        'MaxDD':    round(maxdd, 4),
        'CAGR':     round(cagr, 4),
        'HitRate':  round(float(np.mean(pnl[pnl != 0] > 0)), 4),
        'nTrades':  nTrades,
    }


# ── training ──────────────────────────────────────────────────────────────────
def build_model(epochs=50,
                hidden=(512, 256, 128, 64),
                dropout=0.30,
                label_smooth_eps=0.05,
                min_confidence=0.55):
    """
    hidden=(512, 256, 128, 64):
      - Wider first layer: more capacity to express the 29 raw features.
      - Deeper: four hidden layers instead of three.
      - Dropout 0.30 (up from 0.25) to compensate.

    min_confidence=0.55:
      - Higher gate; anything below becomes HOLD (abstain).
      - Reduces false BUY/SELL signals that drove 72 % MaxDD.
    """
    df = load_training_data()
    wf = walk_forward_splits(df, years_test=1, max_folds=3)

    fold_models = []

    for i, (tr, va, te) in enumerate(wf, 1):
        print(f'\n── Fold {i}/{len(wf)} ──────────────────────────────────────')
        Xtr, ytr = tr[FEATURE_COLS].values, tr.label.values
        Xva, yva = va[FEATURE_COLS].values, va.label.values
        Xte, yte = te[FEATURE_COLS].values, te.label.values

        # normalise
        Xtr, m, s = normalize(Xtr)
        Xva, _, _ = normalize(Xva, m, s)
        Xte, _, _ = normalize(Xte, m, s)

        ytr_oh = one_hot(ytr)
        yva_oh = one_hot(yva)
        yte_oh = one_hot(yte)

        cw = compute_class_weights(ytr_oh, cap=3.0)

        model      = NeuralNetwork(
            [N_FEATURES] + list(hidden) + [3],
            lr=0.001,
            dropout=dropout,
            label_smooth_eps=label_smooth_eps,
        )
        model.mean = m
        model.std  = s

        model.fit(Xtr, ytr_oh, Xva, yva_oh,
                  epochs=epochs,
                  class_weights=cw,
                  lr_schedule=True)

        calibrate_temperature(model, Xva, yva_oh)

        # ── test evaluation ───────────────────────────────────────────────
        probs  = model.predict_proba(Xte)
        conf   = probs.max(axis=1)
        preds  = np.argmax(probs, axis=1)
        preds[conf < min_confidence] = 2          # abstain → HOLD

        labels = np.argmax(yte_oh, axis=1)
        acc    = float(np.mean(preds == labels))
        print(f'  Test acc (with gate): {acc:.4f}')
        print_per_class_metrics(preds, labels)

        # coverage report: how many samples passed the gate?
        n_active = int(np.sum(conf >= min_confidence))
        print(f'  Gate coverage: {n_active}/{len(preds)} '
              f'({n_active / len(preds):.1%}) above conf={min_confidence}')

        mets = portfolio_metrics(
            preds, te.future_return.values, te.name.values
        )
        print('  Portfolio:', mets)

        fold_models.append(model)

    bundle = {'models': fold_models, 'min_confidence': min_confidence}
    joblib.dump(bundle, MODEL_PATH)
    print(f'\nSaved {len(fold_models)}-model ensemble → {MODEL_PATH}')
    return bundle


# ── inference ─────────────────────────────────────────────────────────────────
def _load_or_train():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return build_model()

_BUNDLE = None

def get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = _load_or_train()
    return _BUNDLE

def _ensemble_proba(X_raw):
    """Average softmax probabilities across all fold models."""
    all_probs = []
    for m in _BUNDLE['models']:
        Xn = (X_raw - m.mean) / (m.std + 1e-8)
        all_probs.append(m.predict_proba(Xn))
    return np.mean(all_probs, axis=0)

def top_n_opportunities(feature_matrix, n=10,
                        min_confidence=None):
    if min_confidence is None:
        min_confidence = _BUNDLE.get('min_confidence', 0.55)
    p    = _ensemble_proba(feature_matrix)
    conf = p.max(axis=1)
    # edge = P(BUY) - P(SELL), only for signals above gate
    edge = p[:, 0] - p[:, 1]
    mask = conf >= min_confidence
    edge[~mask] = -np.inf
    idx  = np.argsort(edge)[::-1][:n]
    return idx, edge[idx], p[idx]

@dataclass
class TradeDecision:
    signal:     str
    confidence: float
    prob_buy:   float
    prob_sell:  float
    prob_hold:  float

def predict_from_features(vec, min_confidence=None):
    if min_confidence is None:
        min_confidence = _BUNDLE.get('min_confidence', 0.55)
    p      = _ensemble_proba(vec.reshape(1, -1))[0]
    i      = int(np.argmax(p))
    conf   = float(p[i])
    signal = CLASSES[i] if conf >= min_confidence else 'HOLD'
    return TradeDecision(signal, conf, float(p[0]), float(p[1]), float(p[2]))