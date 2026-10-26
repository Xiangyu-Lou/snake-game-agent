import math
import random

import pytest

from snake_agent.stats import (
    binomial_cdf,
    binomial_interval,
    binomial_lower_bound,
    binomial_upper_bound,
    mean_interval,
)


def test_binomial_cdf_matches_the_definition():
    for n, p in ((10, 0.3), (25, 0.9), (7, 0.01)):
        for k in range(-1, n + 1):
            exact = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
            assert binomial_cdf(k, n, p) == pytest.approx(exact, abs=1e-12)


def test_clopper_pearson_known_values():
    low, high = binomial_interval(5, 10)
    assert low == pytest.approx(0.187086, abs=1e-6) and high == pytest.approx(0.812914, abs=1e-6)
    assert binomial_interval(0, 10) == (0.0, pytest.approx(1 - 0.025**0.1))
    assert binomial_interval(10, 10) == (pytest.approx(0.025**0.1), 1.0)


def test_three_hundred_failure_free_games_bound_failures_near_one_percent():
    # The architecture report's gate: a 95% lower bound above 99% completion.
    assert binomial_upper_bound(0, 299) < 0.01 < binomial_upper_bound(0, 298)
    assert binomial_lower_bound(299, 299) > 0.99 > binomial_lower_bound(298, 298)


def test_bounds_contain_the_estimate_and_shrink_with_more_games():
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randrange(1, 400)
        k = rng.randrange(0, n + 1)
        low, high = binomial_interval(k, n)
        assert 0.0 <= low <= k / n <= high <= 1.0
        assert binomial_lower_bound(k, n) >= low and binomial_upper_bound(k, n) <= high
    assert (
        binomial_interval(50, 100)[1] - binomial_interval(50, 100)[0]
        > binomial_interval(500, 1000)[1] - (binomial_interval(500, 1000)[0])
    )


def test_invalid_counts_are_rejected():
    for args in ((5, 4), (-1, 4), (0, 0)):
        with pytest.raises(ValueError):
            binomial_interval(*args)
    with pytest.raises(ValueError):
        binomial_lower_bound(1, 2, confidence=1.0)


def test_mean_interval():
    assert mean_interval([3.0]) == (3.0, 3.0)
    low, high = mean_interval([1.0, 2.0, 3.0, 4.0])
    assert low < 2.5 < high and high - 2.5 == pytest.approx(2.5 - low)
    with pytest.raises(ValueError):
        mean_interval([])
