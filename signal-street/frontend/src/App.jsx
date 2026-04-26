import { useState, useEffect, useRef } from "react";
import "./App.css";

// ── static watchlist loaded from your stocks.csv names ───────────────────────
// You can replace this array with a dynamic fetch if you serve the CSV too.


const API = "http://localhost:5000";

// ── tiny sparkline via SVG ────────────────────────────────────────────────────
function Sparkline({ data, color }) {
  if (!data || data.length < 2) return null;
  const prices = data.map((d) => d.close);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const W = 260, H = 72;
  const pts = prices.map((p, i) => {
    const x = (i / (prices.length - 1)) * W;
    const y = H - ((p - min) / range) * H;
    return `${x},${y}`;
  });
  const polyline = pts.join(" ");
  const fill = `${pts.join(" ")} ${W},${H} 0,${H}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} style={{ display: "block" }}>
      <defs>
        <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={fill} fill="url(#sg)" />
      <polyline points={polyline} fill="none" stroke={color} strokeWidth="1.5"
        strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// ── probability bar ───────────────────────────────────────────────────────────
function ProbBar({ label, value, color }) {
  return (
    <div className="prob-bar-row">
      <span className="prob-bar-label">{label}</span>
      <div className="prob-bar-track">
        <div className="prob-bar-fill" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span className="prob-bar-pct">{Math.round(value * 100)}%</span>
    </div>
  );
}

// ── metric cell ───────────────────────────────────────────────────────────────
function Metric({ label, value }) {
  return (
    <div className="metric-cell">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

// ── main app ──────────────────────────────────────────────────────────────────
export default function App() {
  const [watchlist, setWatchlist] = useState([]);
  const [selected, setSelected]   = useState(null);
  const [analysis, setAnalysis]   = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [searchTicker, setSearch] = useState("");
  const [blinkKey, setBlinkKey]   = useState(0);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/watchlist`)
      .then((r) => r.json())
      .then((data) => setWatchlist(data))
      .catch(() => {
        // fallback static list if server not yet ready
        setWatchlist([
          { id: 1,  ticker: "AAPL",  name: "Apple Inc." },
          { id: 2,  ticker: "MSFT",  name: "Microsoft" },
          { id: 3,  ticker: "NVDA",  name: "NVIDIA" },
          { id: 4,  ticker: "TSLA",  name: "Tesla" },
          { id: 5,  ticker: "AMZN",  name: "Amazon" },
          { id: 6,  ticker: "GOOGL", name: "Alphabet" },
          { id: 7,  ticker: "META",  name: "Meta" },
          { id: 8,  ticker: "JPM",   name: "JPMorgan" },
          { id: 9,  ticker: "BAC",   name: "Bank of America" },
          { id: 10, ticker: "GS",    name: "Goldman Sachs" },
          { id: 11, ticker: "XOM",   name: "ExxonMobil" },
          { id: 12, ticker: "CVX",   name: "Chevron" },
        ]);
      });
  }, []);
  
  async function analyze(ticker) {
    setLoading(true);
    setError(null);
    setAnalysis(null);
    try {
      const res = await fetch(`${API}/analyze`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ ticker }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "API error");
      setAnalysis(data);
      setBlinkKey((k) => k + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleWatchlistClick(stock) {
    setSelected(stock.ticker);
    analyze(stock.ticker);
  }

  function handleSearch(e) {
    e.preventDefault();
    const t = searchTicker.trim().toUpperCase();
    if (!t) return;
    setSelected(t);
    analyze(t);
    setSearch("");
  }

  // keyboard shortcut: "/" to focus search
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const sig       = analysis?.signal ?? null;
  const sigColor  = sig === "BUY" ? "#00e676" : sig === "SELL" ? "#ff1744" : "#ff9100";
  const histColor = analysis
    ? analysis.price_change >= 0 ? "#00e676" : "#ff1744"
    : "#00e676";

  return (
    <div className="app">

      {/* ── sidebar ──────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <span className="logo">SIGNALSTREET</span>
          <span className="logo-sub">ML SIGNAL ENGINE</span>
        </div>

        <form className="search-form" onSubmit={handleSearch}>
          <input
            ref={inputRef}
            className="search-input"
            value={searchTicker}
            onChange={(e) => setSearch(e.target.value.toUpperCase())}
            placeholder="TICKER  [ / ]"
            spellCheck={false}
            autoComplete="off"
          />
          <button className="search-btn" type="submit">→</button>
        </form>

        <div className="watchlist-label">WATCHLIST</div>
        <ul className="watchlist">
          {watchlist.map((s) => (
            <li
              key={s.id}
              className={`watchlist-item ${selected === s.ticker ? "active" : ""}`}
              onClick={() => handleWatchlistClick(s)}
            >
              <span className="wl-ticker">{s.ticker}</span>
              <span className="wl-name">{s.name}</span>
            </li>
          ))}
        </ul>

        <div className="sidebar-footer">
          <span>NOT FINANCIAL ADVICE</span>
        </div>
      </aside>

      {/* ── main panel ───────────────────────────────────────────── */}
      <main className="main">

        {/* empty state */}
        {!selected && !loading && (
          <div className="empty-state">
            <div className="empty-icon">▲</div>
            <div className="empty-title">SELECT A TICKER</div>
            <div className="empty-sub">Click a watchlist item or type a ticker and press enter</div>
          </div>
        )}

        {/* loading */}
        {loading && (
          <div className="empty-state">
            <div className="spinner" />
            <div className="empty-title">FETCHING {selected}</div>
            <div className="empty-sub">Pulling market data · running ensemble inference…</div>
          </div>
        )}

        {/* error */}
        {error && !loading && (
          <div className="empty-state">
            <div className="empty-icon" style={{ color: "#ff1744" }}>✕</div>
            <div className="empty-title" style={{ color: "#ff1744" }}>ERROR</div>
            <div className="empty-sub">{error}</div>
          </div>
        )}

        {/* result */}
        {analysis && !loading && (
          <div className="result" key={blinkKey}>

            {/* ── top bar ──────────────────────────────────────── */}
            <div className="result-topbar">
              <div className="result-ticker">{analysis.ticker}</div>
              <div className="result-price">
                ${analysis.price.toFixed(2)}
                <span
                  className="result-change"
                  style={{ color: analysis.price_change >= 0 ? "#00e676" : "#ff1744" }}
                >
                  {analysis.price_change >= 0 ? "▲" : "▼"}{" "}
                  {Math.abs(analysis.price_change * 100).toFixed(2)}%
                </span>
              </div>
              <div className="result-meta">
                {analysis.n_models}-model ensemble · {analysis.latency_ms.toFixed(0)} ms
              </div>
            </div>

            {/* ── signal block ─────────────────────────────────── */}
            <div className="signal-block">
              <div className="signal-word" style={{ color: sigColor }}>
                {analysis.signal}
              </div>
              <div className="signal-conf" style={{ borderColor: sigColor }}>
                {Math.round(analysis.confidence * 100)}% CONFIDENCE
              </div>
            </div>

            {/* ── probability bars ─────────────────────────────── */}
            <div className="prob-section">
              <ProbBar label="BUY"  value={analysis.prob_buy}  color="#00e676" />
              <ProbBar label="SELL" value={analysis.prob_sell} color="#ff1744" />
              <ProbBar label="HOLD" value={analysis.prob_hold} color="#ff9100" />
            </div>

            {/* ── chart ────────────────────────────────────────── */}
            <div className="chart-section">
              <div className="section-label">PRICE HISTORY (90d)</div>
              <Sparkline data={analysis.history} color={histColor} />
              {analysis.history.length > 0 && (
                <div className="chart-dates">
                  <span>{analysis.history[0]?.date}</span>
                  <span>{analysis.history[analysis.history.length - 1]?.date}</span>
                </div>
              )}
            </div>

            {/* ── metrics grid ─────────────────────────────────── */}
            <div className="metrics-grid">
              <Metric label="VOLUME"     value={Math.round(analysis.volume).toLocaleString()} />
              <Metric label="RSI (14)"   value={analysis.rsi.toFixed(1)} />
              <Metric label="SPREAD"     value={`${(analysis.spread * 100).toFixed(3)}%`} />
              <Metric label="VOLATILITY" value={`${(analysis.volatility * 100).toFixed(2)}%`} />
            </div>

            {/* ── action strip ─────────────────────────────────── */}
            <div className="action-strip">
              <button
                className="action-btn buy-btn"
                onClick={() => alert(`Simulated BUY order for ${analysis.ticker}`)}
              >
                BUY
              </button>
              <button
                className="action-btn hold-btn"
                onClick={() => alert(`Simulated HOLD for ${analysis.ticker}`)}
              >
                HOLD
              </button>
              <button
                className="action-btn sell-btn"
                onClick={() => alert(`Simulated SELL order for ${analysis.ticker}`)}
              >
                SELL
              </button>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}