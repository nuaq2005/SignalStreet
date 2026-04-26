import { useState } from "react";
import { stocks } from "./data/stocks";
import "./App.css";

export default function App() {
  const [selectedStock, setSelectedStock] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  // call Flask API
  async function analyzeStock(stock) {
    setLoading(true);
    setSelectedStock(stock);

    try {
      const res = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: stock.ticker }),
      });

      const data = await res.json();
      setAnalysis(data);
    } catch (err) {
      console.error("API error:", err);
    }

    setLoading(false);
  }

  return (
    <div className="container">

      {/* NAV BAR */}
      <nav className="navbar">
        <p>📈 PropTest</p>
      </nav>

      {/* LEFT SIDE */}
      <div className="list">
        {stocks.map((stock) => {
          const isUp = stock.change >= 0;

          return (
            <div
              key={stock.id}
              className="card"
              onClick={() => analyzeStock(stock)}
            >
              <h3>{stock.name}</h3>
              <p>{stock.ticker}</p>
              <p>${stock.price.toFixed(2)}</p>

              <p style={{ color: isUp ? "green" : "red" }}>
                {isUp ? "▲" : "▼"} {stock.change}%
              </p>
            </div>
          );
        })}
      </div>

      {/* RIGHT SIDE */}
      <div className="details">

        {!selectedStock && (
          <div className="details-content">
            <h2>Select a stock</h2>
          </div>
        )}

        {loading && (
          <div className="details-content">
            <h2>Loading analysis...</h2>
          </div>
        )}

        {analysis && !loading && (
          <div className="details-content show">

            {/* TITLE */}
            <h1>{analysis.ticker}</h1>

            {/* STREAMLIT-LIKE BIG SIGNAL */}
            <div className={`big-signal ${analysis.signal.toLowerCase()}`}>
              {analysis.signal}
            </div>

            <p className="conf-label">
              {Math.round(analysis.confidence * 100)}% confidence ·{" "}
              {analysis.latency_ms.toFixed(1)} ms
            </p>

            <p className="prob-row">
              BUY {Math.round(analysis.prob_buy * 100)}% ·{" "}
              SELL {Math.round(analysis.prob_sell * 100)}% ·{" "}
              HOLD {Math.round(analysis.prob_hold * 100)}%
            </p>

            {/* METRICS GRID */}
            <div className="metrics">

              <div className="metric">
                <p>Price</p>
                <h3>${analysis.price.toFixed(2)}</h3>
              </div>

              <div className="metric">
                <p>Volume</p>
                <h3>{analysis.volume.toLocaleString()}</h3>
              </div>

              <div className="metric">
                <p>RSI</p>
                <h3>{analysis.rsi.toFixed(1)}</h3>
              </div>

              <div className="metric">
                <p>Spread</p>
                <h3>{(analysis.spread * 100).toFixed(2)}%</h3>
              </div>

              <div className="metric">
                <p>Volatility</p>
                <h3>{(analysis.volatility * 100).toFixed(2)}%</h3>
              </div>

              <div className="metric">
                <p>Change</p>
                <h3>{analysis.price_change.toFixed(2)}%</h3>
              </div>

            </div>

            {/* ACTION BUTTONS */}
            <div className="actions">
              <button className="buy">Buy</button>
              <button className="sell">Sell</button>
              <button className="hold">Hold</button>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}