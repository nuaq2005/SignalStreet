"""
runner.py

Orchestrates the full stress test pipeline.

Returns a StressReport consumed by dashboard.py.

Covers:
  - All 10 properties (original + invariance + monotonic)
  - Shrinking violations to minimal cases
  - Performance stats (scenarios/sec)
  - Deterministic (same seed = same report always)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict

from .generator import generate_batch, stress_scenarios
from .properties import ALL_PROPERTIES, PropertyResult
from .shrinker import shrink, ShrinkResult, is_realistic


# ── Result types ──────────────────────────────────────────────────────

@dataclass
class Violation:
    property_result: PropertyResult
    shrink_result:   ShrinkResult


@dataclass
class StressReport:
    total_scenarios:  int
    total_tests:      int
    total_violations: int
    pass_rate:        float
    weaknesses:       List[str]          # plain English, one per violation
    violations:       List[Violation]    # full detail for dashboard expanders
    property_summary: Dict               # per-property pass/fail counts
    scenarios_per_sec:float              # throughput
    elapsed_ms:       float


# ── Runner ────────────────────────────────────────────────────────────

def run(
    n_scenarios: int = 500,
    seed:        int = 42,
    shrink_violations: bool = True,
) -> StressReport:
    """
    1. Generate n_scenarios random scenarios + fixed stress scenarios
    2. Run all 10 properties on each scenario
    3. Shrink each violation to its minimal failing case
    4. Return StressReport for the dashboard
    """
    t0 = time.perf_counter()

    all_scenarios = generate_batch(n=n_scenarios, seed=seed) + stress_scenarios()

    violations:       List[Violation] = []
    property_summary: Dict            = {}

    for prop_fn in ALL_PROPERTIES:
        property_summary[prop_fn.__name__] = {
            "label":  prop_fn.__name__.replace("check_", "").replace("_", " ").title(),
            "passed": 0,
            "failed": 0,
        }

    total_tests = 0

    for scenario in all_scenarios:
        for prop_fn in ALL_PROPERTIES:
            result      = prop_fn(scenario)
            total_tests += 1

            if result.passed:
                property_summary[prop_fn.__name__]["passed"] += 1
            else:
                property_summary[prop_fn.__name__]["failed"] += 1

                # shrink to minimal failing case
                if shrink_violations:
                    sr = shrink(scenario, prop_fn)
                else:
                    from shrinker import ShrinkResult
                    sr = ShrinkResult(
                        original_scenario=scenario,
                        minimal_scenario=scenario,
                        steps_taken=0,
                        property_name=result.property_name,
                    )

                violations.append(Violation(
                    property_result=result,
                    shrink_result=sr,
                ))

    elapsed_ms        = (time.perf_counter() - t0) * 1000
    n_scenarios_total = len(all_scenarios)
    pass_rate         = (total_tests - len(violations)) / total_tests if total_tests > 0 else 1.0

    # plain English weakness descriptions (deduplicated by property)
    seen_properties   = set()
    weaknesses        = []
    for v in violations:
        prop = v.property_result.property_name
        if prop not in seen_properties:
            seen_properties.add(prop)
            minimal = v.shrink_result.minimal_scenario
            if is_realistic(minimal):
                weaknesses.append(
                    f"{prop}: {v.property_result.violation_description} "
                    f"(minimal case: price=${minimal.price:.2f}, "
                    f"vol={minimal.volume:,}, "
                    f"spread={minimal.spread:.4f})"
                )
            else:
                weaknesses.append(
                    f"{prop}: {v.property_result.violation_description}"
                )

    return StressReport(
        total_scenarios   = n_scenarios_total,
        total_tests       = total_tests,
        total_violations  = len(violations),
        pass_rate         = pass_rate,
        weaknesses        = weaknesses,
        violations        = violations,
        property_summary  = property_summary,
        scenarios_per_sec = round(n_scenarios_total / (elapsed_ms / 1000), 1),
        elapsed_ms        = round(elapsed_ms, 1),
    )