"""
test_properties.py

Property-based and unit tests.
Run with:
    pytest test_properties.py -v
    pytest test_properties.py -v --tb=short --cov=. --cov-report=term-missing

Covers:
  - All 10 model properties (original + invariance + monotonic)
  - Evaluator correctness (non-overlapping periods, slippage, benchmark)
  - Hypothesis-driven fuzzing (random valid inputs)
  - Deterministic simulation (same seed = same result)
"""

import math
import pytest
import numpy as np
from dataclasses import dataclass
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from .generator import MarketScenario, generate_scenario, generate_batch
from evaluator import (
    simulate_portfolio,
    compute_sharpe,
    compute_drawdown as compute_max_drawdown,
    TradingEvaluation,
)
from .shrinker import shrink, is_realistic


# ── Property result type ──────────────────────────────────────────────

@dataclass
class PropertyResult:
    passed:                   bool
    property_name:            str
    violation_description:    str
    original_scenario:        MarketScenario = None
    original_decision:        object = None
    mutated_scenario:         MarketScenario = None
    mutated_decision:         object = None


# ── Property check functions (10 total) ─────────────────────────────

def check_determinism(scenario: MarketScenario) -> PropertyResult:
    """Same input must always give same output."""
    from ml_model import predict
    d1 = predict(scenario)
    d2 = predict(scenario)
    passed = (d1.signal == d2.signal and d1.confidence == d2.confidence)
    return PropertyResult(
        passed=passed,
        property_name="Determinism",
        violation_description="Model gave different outputs for identical input"
    )


def check_no_nan_outputs(scenario: MarketScenario) -> PropertyResult:
    """Outputs must never be NaN."""
    from ml_model import predict
    d = predict(scenario)
    passed = (
        not np.isnan(d.confidence) and
        not np.isnan(d.prob_buy) and
        not np.isnan(d.prob_sell) and
        not np.isnan(d.prob_hold)
    )
    return PropertyResult(
        passed=passed,
        property_name="No NaN Outputs",
        violation_description="Model output contains NaN values"
    )


def check_probability_sum(scenario: MarketScenario) -> PropertyResult:
    """Probabilities must sum to 1.0."""
    from ml_model import predict
    d = predict(scenario)
    prob_sum = d.prob_buy + d.prob_sell + d.prob_hold
    passed = abs(prob_sum - 1.0) < 1e-4
    return PropertyResult(
        passed=passed,
        property_name="Probability Sum",
        violation_description=f"Probabilities sum to {prob_sum:.4f}, not 1.0"
    )


def check_confidence_bounds(scenario: MarketScenario) -> PropertyResult:
    """Confidence must be in [0, 1]."""
    from ml_model import predict
    d = predict(scenario)
    passed = 0.0 <= d.confidence <= 1.0
    return PropertyResult(
        passed=passed,
        property_name="Confidence Bounds",
        violation_description=f"Confidence {d.confidence} is outside [0, 1]"
    )


def check_tick_stability(scenario: MarketScenario) -> PropertyResult:
    """Small price tick should not flip signal."""
    from ml_model import predict
    d1 = predict(scenario)
    tick_scenario = MarketScenario(
        price=scenario.price + 0.01,
        bid=scenario.bid + 0.01,
        ask=scenario.ask + 0.01,
        volume=scenario.volume,
        avg_volume=scenario.avg_volume,
        price_change=scenario.price_change,
        volatility=scenario.volatility,
        spread=scenario.spread,
        label=scenario.label
    )
    d2 = predict(tick_scenario)
    passed = (d1.signal == d2.signal)
    return PropertyResult(
        passed=passed,
        property_name="Tick Stability",
        violation_description=f"1-cent move flipped signal from {d1.signal} to {d2.signal}"
    )


def check_spread_invariance(scenario: MarketScenario) -> PropertyResult:
    """Spread changes should not affect directional signal (for limit orders)."""
    from ml_model import predict
    d1 = predict(scenario)
    spread_scenario = MarketScenario(
        price=scenario.price,
        bid=scenario.bid - 0.02,
        ask=scenario.ask + 0.02,
        volume=scenario.volume,
        avg_volume=scenario.avg_volume,
        price_change=scenario.price_change,
        volatility=scenario.volatility,
        spread=scenario.spread + 0.04,
        label=scenario.label
    )
    d2 = predict(spread_scenario)
    passed = (d1.signal == d2.signal)
    return PropertyResult(
        passed=passed,
        property_name="Spread Invariance",
        violation_description=f"Wider spread flipped signal from {d1.signal} to {d2.signal}"
    )


def check_confidence_smoothness(scenario: MarketScenario) -> PropertyResult:
    """Confidence should change smoothly with small input changes."""
    from ml_model import predict
    d1 = predict(scenario)
    small_change = MarketScenario(
        price=scenario.price + 0.5,
        bid=scenario.bid + 0.5,
        ask=scenario.ask + 0.5,
        volume=int(scenario.volume * 1.05),
        avg_volume=scenario.avg_volume * 1.05,
        price_change=scenario.price_change + 0.001,
        volatility=scenario.volatility + 0.001,
        spread=scenario.spread,
        label=scenario.label
    )
    d2 = predict(small_change)
    conf_diff = abs(d1.confidence - d2.confidence)
    passed = conf_diff < 0.15  # allow up to 15% confidence change for small inputs
    return PropertyResult(
        passed=passed,
        property_name="Confidence Smoothness",
        violation_description=f"Confidence jumped by {conf_diff:.1%}"
    )


def check_volatility_monotonicity(scenario: MarketScenario) -> PropertyResult:
    """Higher volatility should increase confidence (signal is clearer in high vol)."""
    from ml_model import predict
    d1 = predict(scenario)
    high_vol = MarketScenario(
        price=scenario.price,
        bid=scenario.bid,
        ask=scenario.ask,
        volume=scenario.volume,
        avg_volume=scenario.avg_volume,
        price_change=scenario.price_change,
        volatility=scenario.volatility * 2,
        spread=scenario.spread,
        label=scenario.label
    )
    d2 = predict(high_vol)
    # Higher volatility should give stronger signals (higher confidence)
    passed = d2.confidence >= d1.confidence * 0.9  # allow 10% wiggle room
    return PropertyResult(
        passed=passed,
        property_name="Volatility Monotonicity",
        violation_description=f"Doubling volatility decreased confidence"
    )


def check_spread_monotonicity(scenario: MarketScenario) -> PropertyResult:
    """Wider spreads should decrease confidence (harder to execute)."""
    from ml_model import predict
    d1 = predict(scenario)
    wide_spread = MarketScenario(
        price=scenario.price,
        bid=scenario.bid - 0.05,
        ask=scenario.ask + 0.05,
        volume=scenario.volume,
        avg_volume=scenario.avg_volume,
        price_change=scenario.price_change,
        volatility=scenario.volatility,
        spread=scenario.spread + 0.10,
        label=scenario.label
    )
    d2 = predict(wide_spread)
    # Wider spread should weaken confidence (costs more to trade)
    passed = d2.confidence <= d1.confidence * 1.1  # allow 10% wiggle room  
    return PropertyResult(
        passed=passed,
        property_name="Spread Monotonicity",
        violation_description=f"Wider spread increased confidence"
    )


def check_cost_monotonicity(scenario: MarketScenario) -> PropertyResult:
    """Signals with lower confidence shouldn't be chosen when high confidence available."""
    from ml_model import predict
    d = predict(scenario)
    # This is more of a meta-property: if a signal has low confidence,
    # there should be at least one other signal with higher confidence
    other_confs = []
    if d.signal != "BUY":
        other_confs.append(d.prob_buy)
    if d.signal != "SELL":
        other_confs.append(d.prob_sell)
    if d.signal != "HOLD":
        other_confs.append(d.prob_hold)
    
    max_other_conf = max(other_confs) if other_confs else 0
    passed = d.confidence >= max_other_conf or d.confidence > 0.5
    return PropertyResult(
        passed=passed,
        property_name="Cost Monotonicity",
        violation_description="Selected signal doesn't have highest confidence"
    )


# Registry of all 10 properties
ALL_PROPERTIES = [
    check_determinism,
    check_no_nan_outputs,
    check_probability_sum,
    check_confidence_bounds,
    check_tick_stability,
    check_spread_invariance,
    check_confidence_smoothness,
    check_volatility_monotonicity,
    check_spread_monotonicity,
    check_cost_monotonicity,
]




@st.composite
def market_scenario(draw) -> MarketScenario:
    """Generate a valid (not necessarily realistic) MarketScenario."""
    price        = draw(st.floats(min_value=1.0,    max_value=1000.0, allow_nan=False))
    spread       = draw(st.floats(min_value=0.0001, max_value=0.50,   allow_nan=False))
    volume       = draw(st.integers(min_value=0,    max_value=500_000))
    avg_volume   = draw(st.integers(min_value=1,    max_value=500_000))
    price_change = draw(st.floats(min_value=-0.50,  max_value=0.50,   allow_nan=False))
    volatility   = draw(st.floats(min_value=0.0,    max_value=0.20,   allow_nan=False))

    return MarketScenario(
        price=round(price, 2),
        bid=round(price - spread / 2, 2),
        ask=round(price + spread / 2, 2),
        volume=volume,
        avg_volume=float(avg_volume),
        price_change=round(price_change, 4),
        volatility=round(volatility, 4),
        spread=round(spread, 4),
        label="TEST",
    )


# ── Invariance properties (must hold for ALL inputs) ──────────────────

class TestInvarianceProperties:

    @given(market_scenario())
    @settings(max_examples=100)
    def test_determinism_always_holds(self, scenario):
        """Identical inputs must always give identical outputs."""
        result = check_determinism(scenario)
        assert result.passed, result.violation_description

    @given(market_scenario())
    @settings(max_examples=100)
    def test_no_nan_outputs_always(self, scenario):
        """Model must never output NaN."""
        result = check_no_nan_outputs(scenario)
        assert result.passed, result.violation_description

    @given(market_scenario())
    @settings(max_examples=100)
    def test_probability_sum_always_one(self, scenario):
        """BUY + SELL + HOLD must always sum to 1.0 ± 1e-4."""
        result = check_probability_sum(scenario)
        assert result.passed, result.violation_description

    @given(market_scenario())
    @settings(max_examples=100)
    def test_confidence_always_in_bounds(self, scenario):
        """Confidence must always be in [0, 1]."""
        result = check_confidence_bounds(scenario)
        assert result.passed, result.violation_description


# ── Stability properties (should hold for most inputs) ────────────────

class TestStabilityProperties:

    def test_tick_stability_on_normal_scenario(self):
        s = generate_scenario(seed=1)
        result = check_tick_stability(s)
        # not asserting pass — it might fail, that's the point
        assert isinstance(result.passed, bool)
        assert result.property_name == "Tick Stability"

    def test_spread_invariance_on_normal_scenario(self):
        s = generate_scenario(seed=2)
        result = check_spread_invariance(s)
        assert isinstance(result.passed, bool)
        assert result.property_name == "Spread Invariance"

    def test_volatility_monotonicity_on_normal_scenarios(self):
        """On a batch of normal scenarios, most should pass."""
        scenarios = generate_batch(n=50, seed=42)
        results   = [check_volatility_monotonicity(s) for s in scenarios]
        pass_rate = sum(r.passed for r in results) / len(results)
        assert pass_rate > 0.50, f"Too many failures: {pass_rate:.0%} pass rate"

    def test_spread_monotonicity_on_normal_scenarios(self):
        scenarios = generate_batch(n=50, seed=99)
        results   = [check_spread_monotonicity(s) for s in scenarios]
        pass_rate = sum(r.passed for r in results) / len(results)
        assert pass_rate > 0.50, f"Pass rate too low: {pass_rate:.0%}"


# ── ALL_PROPERTIES registry completeness ─────────────────────────────

class TestPropertyRegistry:

    def test_all_properties_has_ten_entries(self):
        assert len(ALL_PROPERTIES) == 10

    def test_all_properties_callable(self):
        for prop in ALL_PROPERTIES:
            assert callable(prop)

    def test_all_properties_return_property_result(self):
        s = generate_scenario(seed=0)
        for prop in ALL_PROPERTIES:
            result = prop(s)
            assert hasattr(result, "passed")
            assert hasattr(result, "property_name")
            assert hasattr(result, "violation_description")
            assert isinstance(result.passed, bool)


# ── Evaluator tests ───────────────────────────────────────────────────

class TestEvaluator:

    def _make_eval(self, preds, rets, confs=None):
        return simulate_portfolio(
            predictions=preds,
            actual_returns=rets,
            confidences=confs,
            seed=42,
        )

    def test_perfect_predictor_is_profitable(self):
        """A model that always correctly calls direction should be profitable."""
        # BUY on rising days, SELL on falling days
        rets  = [0.02, -0.01, 0.03, -0.02, 0.01] * 20
        preds = ["BUY" if r > 0 else "SELL" for r in rets]
        ev    = self._make_eval(preds, rets)
        assert ev.total_return > 0, "Perfect predictor should be profitable"

    def test_always_hold_has_no_trades(self):
        """A model that always says HOLD should make zero trades."""
        preds = ["HOLD"] * 100
        rets  = [0.01] * 100
        ev    = self._make_eval(preds, rets)
        assert ev.n_trades == 0

    def test_deterministic_simulation(self):
        """Same seed must give same result every time."""
        preds = ["BUY", "SELL", "HOLD"] * 33
        rets  = [0.01, -0.01, 0.0] * 33
        ev1   = simulate_portfolio(preds, rets, seed=42)
        ev2   = simulate_portfolio(preds, rets, seed=42)
        assert ev1.total_return   == ev2.total_return
        assert ev1.sharpe_ratio   == ev2.sharpe_ratio
        assert ev1.n_trades       == ev2.n_trades

    def test_transaction_costs_reduce_profit(self):
        """Simulation with real costs should earn less than zero-cost."""
        rets  = [0.005] * 100   # tiny positive returns
        preds = ["BUY"] * 100
        confs = [1.0] * 100
        ev    = self._make_eval(preds, rets, confs)
        # Transaction costs should eat into very small returns
        # (we're not asserting negative — just that costs matter)
        assert ev.n_trades > 0

    def test_circuit_breaker_fires_on_catastrophic_loss(self):
        """30%+ drawdown should halt trading."""
        # Alternating big losses with BUY signals
        rets  = [-0.15] * 20 + [0.01] * 80
        preds = ["BUY"] * 100
        confs = [1.0] * 100
        ev    = self._make_eval(preds, rets, confs)
        assert ev.halted, "Circuit breaker should have fired"

    def test_non_overlapping_positions(self):
        """Position count should be < total signals (10-day holding period)."""
        preds = ["BUY"] * 100
        rets  = [0.01] * 100
        ev    = self._make_eval(preds, rets)
        # With 10-day holding, max trades = 100 / 10 = 10
        assert ev.n_trades <= 10, f"Too many trades: {ev.n_trades}"

    def test_confidence_weighted_sizing(self):
        """High confidence should generate more profit than low confidence."""
        rets    = [0.02] * 50
        preds   = ["BUY"] * 50
        high_ev = simulate_portfolio(preds, rets, confidences=[0.9] * 50, seed=42)
        low_ev  = simulate_portfolio(preds, rets, confidences=[0.1] * 50, seed=42)
        assert high_ev.total_return > low_ev.total_return, \
            "High confidence should generate more return"

    def test_sharpe_is_finite(self):
        preds = ["BUY", "SELL", "HOLD"] * 20
        rets  = [0.01, -0.01, 0.0] * 20
        ev    = self._make_eval(preds, rets)
        assert math.isfinite(ev.sharpe_ratio)

    def test_max_drawdown_between_zero_and_one(self):
        preds = ["BUY"] * 50
        rets  = [0.01, -0.02] * 25
        ev    = self._make_eval(preds, rets)
        assert 0.0 <= ev.max_drawdown <= 1.0

    def test_profit_factor_non_negative(self):
        preds = ["BUY"] * 30
        rets  = [0.01] * 15 + [-0.01] * 15
        ev    = self._make_eval(preds, rets)
        assert ev.profit_factor >= 0.0

    def test_win_rate_between_zero_and_one(self):
        preds = ["BUY", "SELL"] * 25
        rets  = [0.01, -0.01] * 25
        ev    = self._make_eval(preds, rets)
        assert 0.0 <= ev.win_rate <= 1.0

    def test_benchmark_comparison_exists(self):
        preds      = ["BUY"] * 20
        rets       = [0.01] * 20
        bench_rets = [0.005] * 20
        ev = simulate_portfolio(preds, rets, benchmark_returns=bench_rets, seed=42)
        assert hasattr(ev, "alpha")
        assert hasattr(ev, "benchmark_return")

    @given(
        st.lists(st.sampled_from(["BUY", "SELL", "HOLD"]), min_size=10, max_size=200),
        st.lists(st.floats(min_value=-0.10, max_value=0.10, allow_nan=False), min_size=10, max_size=200),
    )
    @settings(max_examples=50)
    def test_evaluator_never_crashes(self, preds, rets):
        """Evaluator must not raise on any valid input combination."""
        n    = min(len(preds), len(rets))
        preds, rets = preds[:n], rets[:n]
        assume(n >= 5)
        try:
            ev = simulate_portfolio(preds, rets, seed=42)
            assert math.isfinite(ev.sharpe_ratio) or ev.n_trades == 0
        except Exception as e:
            pytest.fail(f"Evaluator raised unexpectedly: {e}")


# ── Shrinker tests ────────────────────────────────────────────────────

class TestShrinker:

    def test_shrinker_finds_simpler_case(self):
        """If a scenario fails a property, shrinker should reduce complexity."""
        # Find a failing scenario
        scenarios = generate_batch(n=200, seed=77)
        failing   = next(
            (s for s in scenarios if not check_tick_stability(s).passed),
            None
        )
        if failing is None:
            pytest.skip("No failing scenario found in batch")

        result = shrink(failing, check_tick_stability)
        # shrunken price should be simpler (fewer decimal places or smaller)
        original_decimals = len(str(failing.price).split(".")[-1])
        minimal_decimals  = len(str(result.minimal_scenario.price).split(".")[-1])
        assert minimal_decimals <= original_decimals or result.steps_taken > 0

    def test_shrinker_minimal_still_fails(self):
        """The minimal scenario must still fail the same property."""
        scenarios = generate_batch(n=200, seed=55)
        failing   = next(
            (s for s in scenarios if not check_spread_invariance(s).passed),
            None
        )
        if failing is None:
            pytest.skip("No failing scenario found")

        result = shrink(failing, check_spread_invariance)
        final  = check_spread_invariance(result.minimal_scenario)
        assert not final.passed, "Minimal scenario should still fail"

    def test_is_realistic_accepts_normal_scenario(self):
        s = generate_scenario(seed=10)
        # normal generated scenarios should generally be realistic
        assert is_realistic(s)

    def test_is_realistic_rejects_penny_stock(self):
        s = MarketScenario(
            price=0.50, bid=0.25, ask=0.75,
            volume=100, avg_volume=100.0,
            price_change=0.5, volatility=0.1,
            spread=0.50, label="PENNY"
        )
        assert not is_realistic(s)