"""
generator.py
Creates random but realistic market scenarios for training and stress testing.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class MarketScenario:
    price:        float
    bid:          float
    ask:          float
    volume:       int
    price_change: float
    volatility:   float
    spread:       float
    avg_volume:   float 
    label:        str = "NORMAL"   # tag so you know what kind of scenario it is


def generate_scenario(seed: int = None) -> MarketScenario:
    rng = np.random.default_rng(seed)
    price        = round(float(rng.uniform(20, 300)), 2)
    spread       = round(float(rng.uniform(0.01, 0.09)), 2)
    bid          = round(price - spread / 2, 2)
    ask          = round(price + spread / 2, 2)
    volume       = int(rng.integers(5_000, 100_000))
    price_change = round(float(rng.uniform(-0.12, 0.12)), 4)
    volatility   = round(float(rng.uniform(0.0, 0.08)), 4)
    avg_volume = round(float(rng.uniform(volume * 0.8, volume * 1.5)))
    return MarketScenario(
        price=price, bid=bid, ask=ask,
        volume=volume, price_change=price_change,
        volatility=volatility, spread=spread,
        avg_volume=avg_volume,
        label="NORMAL"
    )


def generate_batch(n: int = 1000, seed: int = 42) -> List[MarketScenario]:
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 999999, size=n)
    return [generate_scenario(int(s)) for s in seeds]


# ── Stress scenarios ─────────────────────────────────────────────────

def stress_scenarios() -> List[MarketScenario]:
    return [
        # price, bid, ask, volume, price_change, volatility, spread, avg_volume, label

        # Boundary: absolute extremes
        MarketScenario(100, 99.75,  100.25, 100,   -4.99, 0.049,  0.50,  5000.0,  "CRASH"),
        MarketScenario(300, 299.75, 300.25, 49999,  4.99, 0.001,  0.01,  25000.0, "MOON"),
        MarketScenario(100, 99.75,  100.25, 1,      0.0,  0.0,    0.01,  1000.0,  "DEAD"),

        # Contradictions
        MarketScenario(200, 199.75, 200.25, 101,    4.99, 0.001,  0.01,  25000.0, "SPIKE_NO_VOLUME"),
        MarketScenario(150, 149.75, 150.25, 25000, -4.99, 0.0001, 0.30,  25000.0, "CRASH_NO_VOL"),
        MarketScenario(100, 99.50,  100.50, 200,    0.01, 0.001,  0.99,  200.0,   "WIDE_SPREAD"),

        # Edge cases
        MarketScenario(1,   0.50,   1.50,   500,    0.50, 0.04,   0.99,  500.0,   "PENNY"),
        MarketScenario(200, 199.75, 200.25, 99999,  0.001,0.001,  0.01,  10000.0, "VOLUME_SPIKE"),
        MarketScenario(200, 200.25, 199.75, 10000,  0.5,  0.02,  -0.01,  10000.0, "BAD_DATA"),
    ] 


def scenario_to_dict(s: MarketScenario) -> dict:
    return {
        "price":        s.price,
        "bid":          s.bid,
        "ask":          s.ask,
        "volume":       s.volume,
        "avg_volume":   s.avg_volume,
        "price_change": s.price_change,
        "volatility":   s.volatility,
        "spread":       s.spread,
        "label":        s.label
    }