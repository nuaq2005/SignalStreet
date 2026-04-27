"""
shrinker.py

Given a failing scenario, finds the MINIMAL version that still fails.
Completely separate from the ML model — pure search loop.

Improvements:
  - price_change strategies always move toward zero (sign-aware)
  - avg_volume preserved during mutation
  - is_realistic guard prevents misleading minimal cases
  - deterministic (no randomness in shrinking)
"""

import copy
from dataclasses import dataclass
from .generator import MarketScenario


@dataclass
class ShrinkResult:
    original_scenario: MarketScenario
    minimal_scenario:  MarketScenario
    steps_taken:       int
    property_name:     str


def _mutate(scenario: MarketScenario, **kwargs) -> MarketScenario:
    """Copy scenario and apply field overrides. Recomputes bid/ask."""
    s = copy.copy(scenario)
    for key, val in kwargs.items():
        setattr(s, key, val)
    s.bid = round(s.price - s.spread / 2, 2)
    s.ask = round(s.price + s.spread / 2, 2)
    return s


def _toward_zero(x: float, factor: float = 0.5) -> float:
    """Move x toward zero by factor — works for both positive and negative."""
    return round(x * factor, 4)


def shrink(
    failing_scenario: MarketScenario,
    property_fn,
    max_steps: int = 100,
) -> ShrinkResult:
    """
    Greedy shrinking: try each simplification strategy in order.
    If simplified version still fails → keep it and continue.
    Repeat until nothing can be simplified further.

    Goal: find simplest possible failing case, not smallest number.
    'Simple' means: closer to zero, fewer decimal places, rounder numbers.
    """
    current = copy.copy(failing_scenario)
    steps   = 0

    strategies = [
        # ── price: round toward whole number ─────────────────────────
        lambda s: _mutate(s, price=round(s.price)),
        lambda s: _mutate(s, price=round(s.price * 10) / 10),

        # ── volume: halve toward minimum liquid ──────────────────────
        lambda s: _mutate(s, volume=max(100, s.volume // 2)),
        lambda s: _mutate(s, volume=max(100, round(s.volume / 1000) * 1000)),

        # ── price_change: toward zero (sign-aware) ────────────────────
        # Halving always moves toward zero regardless of sign:
        #   +3.72 → +1.86 → +0.93 → +0.1
        #   -3.72 → -1.86 → -0.93 → -0.1
        lambda s: _mutate(s, price_change=_toward_zero(s.price_change, 0.5)),
        lambda s: _mutate(s, price_change=_toward_zero(s.price_change, 0.1)),

        # ── spread: toward minimum realistic ─────────────────────────
        lambda s: _mutate(s, spread=round(s.spread, 2)),
        lambda s: _mutate(s, spread=max(0.001, round(s.spread / 2, 4))),

        # ── volatility: toward near-zero ─────────────────────────────
        lambda s: _mutate(s, volatility=round(s.volatility / 2, 4)),
        lambda s: _mutate(s, volatility=1e-4),   # near-zero, not exactly zero

        # ── avg_volume: align toward volume ──────────────────────────
        # Keeps volume_ratio meaningful during shrinking
        lambda s: _mutate(s, avg_volume=max(s.volume, round(s.avg_volume / 2))),
    ]

    improved = True
    while improved and steps < max_steps:
        improved = False
        for strategy in strategies:
            candidate = strategy(current)
            result    = property_fn(candidate)
            if not result.passed:
                current  = candidate
                steps   += 1
                improved = True
                break   # restart from first strategy with simplified scenario

    return ShrinkResult(
        original_scenario = failing_scenario,
        minimal_scenario  = current,
        steps_taken       = steps,
        property_name     = property_fn(current).property_name,
    )


def is_realistic(s: MarketScenario) -> bool:
    """
    Guard: is this scenario a realistic market snapshot?
    Used to filter shrunken scenarios before showing them to users.
    We want minimal failing cases, not market nonsense.

    Note: this does NOT prevent the model from being called on these inputs
    (that's the whole point of stress testing). It only affects
    what gets displayed in the dashboard weakness panel.
    """
    return (
        s.volume >= 5_000                  # minimum liquid
        and s.spread < 0.10                # spread < 10% (not a penny stock)
        and s.price >= 1.0                 # not sub-dollar
        and abs(s.price_change) < 0.15    # no 15%+ moves in minimal cases
        and s.volatility >= 0.0           # non-negative vol
    )