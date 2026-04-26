import { useState } from "react";
import { stocks } from "./data/stocks";
import "./App.css";

export default function App() {
  const [selectedStock, setSelectedStock] = useState(null);

  return (
    <div className="container">
      {/* NAV BAR */}
      <nav className="navbar">
        <p>Signal Street</p>
      </nav>

      {/* LEFT SIDE */}
      <div className="list">

        {stocks.map((stock) => {
          const isUp = stock.change >= 0;

          return (
            <div
              key={stock.id}
              className="card"
              onClick={() => setSelectedStock(prev => 
                prev && prev.id === stock.id ? null : stock
              )}
            >
              <h3>{stock.name}</h3>
              <p>{stock.ticker}</p>
              <p>${stock.price.toFixed(2)}</p>

              <p style={{ color: isUp ? "green" : "red" }}>
                {isUp ? "▲" : "▼"} {stock.change}%
              </p>
              
              <div className="arrow">
                {selectedStock?.id === stock.id ? "◀" : ""}
              </div>
            </div>
          );
        })}
      </div>

      {/* RIGHT SIDE */}
        <div className= "details">
        {selectedStock ? (
          <div 
            key = {selectedStock.id}
            className="details-content show"
          >

          <h1>{selectedStock.name}</h1>
          <h2>{selectedStock.ticker}</h2>
          <h2>${selectedStock.price}</h2>

        <p style={{ color: selectedStock.change >= 0 ? "green" : "red" }}>
          {selectedStock.change >= 0 ? "Up" : "Down"} {selectedStock.change}%
          </p>
        <p style={{ marginTop: "20px" }}>
          {selectedStock.description}
        </p>

        <div>
          <button className="buy">Buy</button>
          <button className="sell">Sell</button>
          <button className="hold">Hold</button>
        </div>

        </div>
          ) : (
          <div className="details-content show">
            <h2>Select a stock</h2>
          </div>
          )}
      </div>
    </div>
  );
}