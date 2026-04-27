"""
evaluator.py

Final upgraded evaluator tuned for your ml_model.py + data_layer.py stack.

Compatible with:
- BUY / SELL / HOLD classification outputs
- confidence probabilities
- volume_ratio feature (slippage realism)
- spy_trend benchmark proxy
- 10-day forward return labels

Features:
1. Non-overlapping holding periods
2. Confidence-weighted sizing
3. Liquidity-aware slippage
4. Commission + borrow cost
5. Benchmark comparison
6. Drawdown circuit breaker
7. Sharpe / Alpha / Profit Factor
8. BUY precision / turnover / expectancy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


# ── Constants ───────────────────────────────────────────────

COMMISSION_BPS    = 5       # per side
SLIPPAGE_BPS      = 3
BORROW_BPS        = 10
HOLDING_PERIOD    = 10

MAX_POSITION_SIZE = 0.20
MAX_DRAWDOWN_HALT = 0.30

TRADING_DAYS      = 252


# ── Result Dataclass ────────────────────────────────────────

@dataclass
class TradingEvaluation:
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    annual_return: float

    benchmark_return: float
    alpha: float

    n_trades: int
    win_rate: float
    buy_precision: float
    expectancy: float
    profit_factor: float
    turnover: float

    halted: bool
    halt_reason: str
    verdict: str

    equity_curve: pd.Series


# ── Helpers ─────────────────────────────────────────────────

def compute_sharpe(returns: list, rf: float = 0.05):
    if len(returns) < 2:
        return 0.0

    r = np.array(returns, dtype=float)
    vol = r.std()

    if vol < 1e-8:
        return 0.0

    mu = r.mean() * TRADING_DAYS
    sigma = vol * np.sqrt(TRADING_DAYS)

    return float((mu - rf) / sigma)


def compute_drawdown(equity: pd.Series):
    peak = equity.cummax()
    dd = (equity - peak) / (peak + 1e-8)
    return float(abs(dd.min()))


def annualize(returns: list):
    if len(returns) < 2:
        return 0.0

    r = np.array(returns)
    total = np.prod(1 + r)

    years = len(r) / TRADING_DAYS
    if years <= 0:
        return 0.0

    return float(total ** (1 / years) - 1)


def verdict(sharpe, dd, pf, alpha):
    if sharpe > 2 and dd < 0.15 and pf > 1.5 and alpha > 0:
        return "Excellent"
    elif sharpe > 1 and dd < 0.25 and pf > 1.2:
        return "Good"
    elif sharpe > 0.5:
        return "Moderate"
    elif sharpe > 0:
        return "Weak"
    return "Poor"


def slippage_cost(position_frac, volume_ratio):
    impact = position_frac / (volume_ratio + 1e-8) * 0.1
    return (SLIPPAGE_BPS + impact) / 10000


# ── Main Simulator ──────────────────────────────────────────

def simulate_portfolio(
    predictions: List[str],
    actual_returns: List[float],
    confidences: Optional[List[float]] = None,
    volume_ratios: Optional[List[float]] = None,
    benchmark_returns: Optional[List[float]] = None,
    initial_capital: float = 100000,
):

    n = len(predictions)

    if confidences is None:
        confidences = [1.0] * n

    if volume_ratios is None:
        volume_ratios = [1.0] * n

    if benchmark_returns is None:
        benchmark_returns = [0.0] * n

    capital = initial_capital
    bench   = initial_capital

    equity_curve = [capital]
    daily_returns = []

    trades = 0
    wins = 0
    buy_total = 0
    buy_correct = 0

    gross_profit = 0.0
    gross_loss   = 0.0
    turnover     = 0.0

    peak = capital
    halted = False
    halt_reason = ""

    next_trade_index = 0

    i = 0
    while i < n:

        # benchmark
        bench *= (1 + benchmark_returns[i])

        # drawdown breaker
        dd = (peak - capital) / (peak + 1e-8)
        if dd > MAX_DRAWDOWN_HALT and not halted:
            halted = True
            halt_reason = f"Drawdown exceeded {MAX_DRAWDOWN_HALT:.0%}"

        if halted:
            equity_curve.append(capital)
            daily_returns.append(0.0)
            i += 1
            continue

        signal = predictions[i]

        # skip hold or locked period
        if i < next_trade_index or signal == "HOLD":
            equity_curve.append(capital)
            daily_returns.append(0.0)
            i += 1
            continue

        conf = float(confidences[i])
        ret  = float(actual_returns[i])
        vr   = max(float(volume_ratios[i]), 0.1)

        size = min(conf, MAX_POSITION_SIZE)
        notional = capital * size

        commission_entry = notional * COMMISSION_BPS / 10000
        commission_exit  = notional * COMMISSION_BPS / 10000

        slip = notional * slippage_cost(size, vr)

        pnl = 0.0

        if signal == "BUY":

            buy_total += 1

            pnl = (
                notional * ret
                - commission_entry
                - commission_exit
                - slip
            )

            if ret > 0:
                wins += 1
                buy_correct += 1

        elif signal == "SELL":

            pnl = (
                notional * (-ret)
                - commission_entry
                - commission_exit
                - slip
                - notional * BORROW_BPS / 10000
            )

            if ret < 0:
                wins += 1

        trades += 1

        if pnl >= 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

        capital += pnl
        capital = max(capital, 1)

        peak = max(peak, capital)

        turnover += notional

        equity_curve.append(capital)
        daily_returns.append(
            pnl / (equity_curve[-2] + 1e-8)
        )

        next_trade_index = i + HOLDING_PERIOD
        i += 1

    # metrics
    eq = pd.Series(equity_curve)

    total_return = (capital - initial_capital) / initial_capital
    bench_return = (bench - initial_capital) / initial_capital

    alpha = total_return - bench_return

    wr = wins / trades if trades else 0
    bp = buy_correct / buy_total if buy_total else 0

    expectancy = (
        (capital - initial_capital) / trades
        if trades else 0
    )

    pf = gross_profit / (gross_loss + 1e-8)

    sr = compute_sharpe(daily_returns)
    dd = compute_drawdown(eq)
    ar = annualize(daily_returns)

    return TradingEvaluation(
        sharpe_ratio     = sr,
        max_drawdown     = dd,
        total_return     = total_return,
        annual_return    = ar,
        benchmark_return = bench_return,
        alpha            = alpha,
        n_trades         = trades,
        win_rate         = wr,
        buy_precision    = bp,
        expectancy       = expectancy,
        profit_factor    = pf,
        turnover         = turnover / initial_capital,
        halted           = halted,
        halt_reason      = halt_reason,
        verdict          = verdict(sr, dd, pf, alpha),
        equity_curve     = eq,
    )


# ── Pretty Print ────────────────────────────────────────────

def print_evaluation(ev: TradingEvaluation):

    print("\n── Trading Evaluation ─────────────────────────────")
    print(f"Sharpe Ratio:       {ev.sharpe_ratio:>8.2f}")
    print(f"Max Drawdown:       {ev.max_drawdown:>8.1%}")
    print(f"Total Return:       {ev.total_return:>8.1%}")
    print(f"Annual Return:      {ev.annual_return:>8.1%}")
    print()
    print(f"Benchmark Return:   {ev.benchmark_return:>8.1%}")
    print(f"Alpha:              {ev.alpha:>8.1%}")
    print()
    print(f"Trades:             {ev.n_trades:,}")
    print(f"Win Rate:           {ev.win_rate:>8.1%}")
    print(f"BUY Precision:      {ev.buy_precision:>8.1%}")
    print(f"Expectancy:         ${ev.expectancy:>8.2f}")
    print(f"Profit Factor:      {ev.profit_factor:>8.2f}")
    print(f"Turnover:           {ev.turnover:>8.1f}x")

    if ev.halted:
        print(f"\nCircuit Breaker:    {ev.halt_reason}")

    print(f"\nVerdict: {ev.verdict}")
    print("──────────────────────────────────────────────────")