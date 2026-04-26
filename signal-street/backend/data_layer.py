import pandas as pd
import numpy as np

TRANSACTION_COST = 0.001

# ---------- labels ----------
def label_from_future_return(r, vol):
    th = max(0.01, float(vol)) + TRANSACTION_COST
    if r > th:  return 'BUY'
    if r < -th: return 'SELL'
    return 'HOLD'

# ---------- indicators ----------
def rsi(close, n=14):
    d  = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn + 1e-8)
    return 100 - (100 / (1 + rs))

def macd(close):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    m   = e12 - e26
    s   = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s

# FEATURE_COLS must exactly match what the training loop uses.
# Keep this in sync with add_features() — every column added there
# that is meant for training must appear here, and nothing else.
FEATURE_COLS = [
    'price_change', 'gap_return', 'volume', 'spread', 'volatility', 'close',
    'price_lag_1', 'price_lag_5', 'price_lag_10',
    'volume_lag_1', 'volume_lag_5',
    'rolling_mean_5', 'rolling_mean_20', 'rolling_std_5', 'rolling_std_20',
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'bb_bandwidth',
    'trend_strength', 'vol_regime', 'volume_ratio',
    'spy_trend', 'vix_proxy',
    'rel_strength_spy', 'beta_60', 'vol_percentile',
]
N_FEATURES = len(FEATURE_COLS)   # should be 29

# LABEL HORIZON — used both for labeling and for how long we hold trades.
# Keep this constant shared so the trading sim always matches the labels.
LABEL_HORIZON = 10   # trading days forward

# ---------- market maps ----------
def compute_market_maps(df):
    t = df.copy()
    t['_ret'] = t.groupby('name')['close'].pct_change()
    t['_vol'] = t.groupby('name')['_ret'].transform(lambda x: x.rolling(20).std())
    daily = (
        t.groupby('date')
         .agg(mkt_ret=('_ret', 'mean'), vix_proxy=('_vol', 'mean'))
         .reset_index()
    )
    daily['spy_trend'] = daily['mkt_ret'].rolling(20, min_periods=1).mean()
    return (
        dict(zip(daily.date, daily.spy_trend)),
        dict(zip(daily.date, daily.vix_proxy)),
        dict(zip(daily.date, daily.mkt_ret)),
    )

# ---------- features ----------
def add_features(g, spy_map, vix_map, mkt_ret_map):
    g = g.copy()
    c = g.close
    v = g.volume

    g['price_change'] = c.pct_change()
    g['gap_return']   = (g.open - c.shift(1)) / (c.shift(1) + 1e-8)
    g['spread']       = (g.high - g.low) / (c + 1e-8)
    g['volatility']   = g['price_change'].rolling(20).std()

    g['price_lag_1']  = c.shift(1)
    g['price_lag_5']  = c.shift(5)
    g['price_lag_10'] = c.shift(10)
    g['volume_lag_1'] = v.shift(1)
    g['volume_lag_5'] = v.shift(5)

    avgv = v.rolling(20).mean()
    g['volume_ratio'] = v / (avgv + 1e-8)

    g['rolling_mean_5']  = c.rolling(5).mean()
    g['rolling_mean_20'] = c.rolling(20).mean()
    g['rolling_std_5']   = c.rolling(5).std()
    g['rolling_std_20']  = c.rolling(20).std()

    g['rsi_14'] = rsi(c)
    m, s, h = macd(c)
    g['macd'] = m; g['macd_signal'] = s; g['macd_hist'] = h

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    up   = ma20 + 2 * sd20
    lo   = ma20 - 2 * sd20
    br   = (up - lo) + 1e-8
    g['bb_position']  = (c - lo) / br
    g['bb_bandwidth'] = br / (ma20 + 1e-8)

    ma50 = c.rolling(50).mean()
    g['trend_strength'] = (c - ma50) / (ma50 + 1e-8)

    lv = g['price_change'].rolling(60).std()
    g['vol_regime'] = g['volatility'] / (lv + 1e-8)

    g['spy_trend'] = g.date.map(spy_map).fillna(0)
    g['vix_proxy'] = g.date.map(vix_map).fillna(0)

    mr = g.date.map(mkt_ret_map).fillna(0)
    g['rel_strength_spy'] = g['price_change'] - mr

    cov = g['price_change'].rolling(60).cov(mr)
    var = mr.rolling(60).var()
    g['beta_60'] = cov / (var + 1e-8)

    g['vol_percentile'] = (
        g['volatility']
        .rolling(window=120, min_periods=20)
        .rank(pct=True)
    )

    # --- label (uses LABEL_HORIZON) ---
    fut = c.shift(-LABEL_HORIZON)
    g['future_return'] = (fut - c) / (c + 1e-8)
    g['label'] = np.vectorize(label_from_future_return)(
        g['future_return'].fillna(0),
        g['volatility'].fillna(0.01),
    )
    return g

# ---------- load ----------
def load_training_data(path='data/stocks.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    df.columns = df.columns.str.lower()
    df = df.sort_values(['name', 'date']).reset_index(drop=True)

    spy_map, vix_map, mkt_ret_map = compute_market_maps(df)

    rows = []
    for _, grp in df.groupby('name'):
        if len(grp) < 180:
            continue
        x = add_features(grp, spy_map, vix_map, mkt_ret_map).dropna()
        rows.append(x)

    out = pd.concat(rows, ignore_index=True)

    # Sanity-check: confirm FEATURE_COLS are actually present
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        raise RuntimeError(f'FEATURE_COLS references columns not in DataFrame: {missing}')

    print(f'Rows:    {len(out):,}')
    print(f'Tickers: {out.name.nunique()}')
    print(f'Classes: {out.label.value_counts().to_dict()}')
    print(f'N_FEATURES={N_FEATURES}  LABEL_HORIZON={LABEL_HORIZON}d')
    return out

# ---------- walk forward ----------
def walk_forward_splits(df, years_test=1, max_folds=3):
    df   = df.sort_values('date').reset_index(drop=True)
    yrs  = sorted(df.date.dt.year.unique())
    folds = []
    for end in range(len(yrs) - years_test - max_folds + 1,
                     len(yrs) - years_test + 1):
        if end < 2:
            continue
        train_years = yrs[:end]
        test_years  = yrs[end: end + years_test]
        val_year    = train_years[-1]
        train_years = train_years[:-1]
        tr = df[df.date.dt.year.isin(train_years)]
        va = df[df.date.dt.year == val_year]
        te = df[df.date.dt.year.isin(test_years)]
        if len(tr) and len(va) and len(te):
            folds.append((tr, va, te))
    return folds

# ---------- live ----------

def get_live_features(ticker):
    import yfinance as yf
    raw = yf.download(ticker, period='220d', interval='1d', progress=False)
    if raw.empty:
        raise ValueError('No data')
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in raw.columns]
    raw = raw.reset_index().rename(columns={'Date': 'date'})
    raw['name'] = ticker.upper()

    ret = raw.close.pct_change()
    spy = float(ret.rolling(20).mean().iloc[-1]) if len(ret) >= 20 else 0
    vix = float(ret.rolling(20).std().iloc[-1])  if len(ret) >= 20 else 0
    maps = (
        {d: spy for d in raw.date},
        {d: vix for d in raw.date},
        {d: 0   for d in raw.date},
    )
    g = add_features(raw, *maps).dropna()
    r = g.iloc[-1]
    return {
        'feature_vector': r[FEATURE_COLS].values.astype(float),
        'price':          float(r.close),
        'history':        g.set_index('date').close,
    }