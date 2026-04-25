import { useState } from "react";
import { stocks } from "./data/stocks";
import "./App.css";

export default function App() {
  const [selectedStock, setSelectedStock] = useState(stocks[0]);

  return (
    <div className="container">
      
      {/* LEFT SIDE */}
      <div className="list">
        <h2>📈 Signal Street</h2>

        {stocks.map((stock) => {
          const isUp = stock.change >= 0;

          return (
            <div
              key={stock.id}
              className="card"
              onClick={() => setSelectedStock(stock)}
              style={{
                border:
                  selectedStock.id === stock.id
                    ? "2px solid black"
                    : "1px solid #ddd",
              }}
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
        <h1>{selectedStock.name}</h1>
        <h2>{selectedStock.ticker}</h2>

        <h2>${selectedStock.price}</h2>

        <p style={{ color: selectedStock.change >= 0 ? "green" : "red" }}>
          {selectedStock.change >= 0 ? "Up" : "Down"} {selectedStock.change}%
        </p>

        <p style={{ marginTop: "20px" }}>
          {selectedStock.description}
        </p>

        <div className="box">
          <p>📊 More analytics coming soon</p>
          <p>🧠 ML signals will go here later</p>
        </div>
      </div>
    </div>
  );
}