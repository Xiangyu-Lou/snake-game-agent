import numpy as np
import pytest

pytest.importorskip("torch")

from snake_agent.assess import calibration, uniform_free_states  # noqa: E402
from snake_agent.contract import TRAINING_RULES  # noqa: E402
from snake_agent.exact import solve  # noqa: E402
from snake_agent.search import envelope_certain  # noqa: E402


def test_calibration_separates_bias_from_spread():
    exact = np.array([0.0, 1.0] * 50)
    perfect = calibration(exact, exact)
    assert perfect["ece"] == perfect["mae"] == perfect["brier"] == 0.0
    # Always 0.5 is calibrated on average, but far from each state's value.
    blind = calibration(np.full(100, 0.5), exact)
    assert blind["ece"] == 0.0 and blind["mae"] == 0.5 and blind["brier"] == 0.25
    assert blind["bins"] == [{"bin": [0.5, 0.6], "states": 100, "predicted": 0.5, "exact": 0.5}]
    # Overconfident by 0.1 everywhere.
    biased = calibration(np.full(10, 0.9), np.full(10, 0.8))
    assert biased["ece"] == pytest.approx(0.1) and biased["bins"][0]["bin"] == [0.9, 1.0]


def test_uniform_states_lie_outside_the_envelope():
    solution = solve(3, 4)
    states = uniform_free_states(solution, 200, seed=1)
    assert len(states) == 200 and len(set(states)) > 150
    for state in states:
        assert not envelope_certain(state, TRAINING_RULES)
        assert 0.0 <= solution.value(state) <= 1.0
