"""Confidence intervals for evaluation results (standard library only).

Win rates use exact Clopper-Pearson intervals: with no failures in ``n`` games,
the one-sided 95% upper bound on the failure rate is ``1 - 0.05 ** (1 / n)``,
below 1% from 299 games on. Mean move counts use the normal approximation,
which is adequate for the hundreds of games an evaluation plays.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def _log_pmf(i: int, n: int, p: float) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * math.log(p)
        + (n - i) * math.log1p(-p)
    )


def binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = [_log_pmf(i, n, p) for i in range(k + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * math.fsum(math.exp(v - top) for v in logs))


def _solve(f, target: float) -> float:
    """``p`` in [0, 1] with ``f(p) == target`` for a decreasing ``f``, by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2


def binomial_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on a success probability."""
    _check(successes, trials, confidence)
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    if successes == trials:
        return alpha ** (1.0 / trials)
    # P(X >= successes | p) = 1 - cdf(successes - 1) rises with p; solve it equal to alpha.
    return _solve(lambda p: binomial_cdf(successes - 1, trials, p), 1.0 - alpha)


def binomial_upper_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a success probability."""
    _check(successes, trials, confidence)
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)
    return _solve(lambda p: binomial_cdf(successes, trials, p), alpha)


def binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval: each bound is one-sided at ``(1 + confidence) / 2``."""
    one_sided = (1.0 + confidence) / 2
    return (
        binomial_lower_bound(successes, trials, one_sided),
        binomial_upper_bound(successes, trials, one_sided),
    )


def mean_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Normal-approximation interval for the mean of ``values`` (a point for a single value)."""
    if not values:
        raise ValueError("no values")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return (mean, mean)
    z = statistics.NormalDist().inv_cdf((1.0 + confidence) / 2)
    half = z * statistics.stdev(values) / math.sqrt(len(values))
    return (mean - half, mean + half)


def _check(successes: int, trials: int, confidence: float) -> None:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(f"need 0 <= successes <= trials and trials >= 1, got {successes}/{trials}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
