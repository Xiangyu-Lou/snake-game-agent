import dataclasses

import numpy as np
import pytest

pytest.importorskip("torch")

from snake_agent.assess import calibration, sibling_ranking, uniform_free_states  # noqa: E402
from snake_agent.contract import TRAINING_RULES  # noqa: E402
from snake_agent.encoding import moves_scale  # noqa: E402
from snake_agent.evaluation import run_game  # noqa: E402
from snake_agent.exact import solve, solve_envelope  # noqa: E402
from snake_agent.search import Prediction, envelope_certain  # noqa: E402
from snake_game import GameConfig, SnakeGame  # noqa: E402


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


def test_exact_values_order_every_pair_of_moves():
    solution = solve_envelope(4, 4)

    def exact_moves(queries):
        moves = []
        for query in queries:
            state = query.state
            if query.afterstate:
                snake = set(state.snake)
                cells = [(x, y) for y in range(4) for x in range(4) if (x, y) not in snake]
                value = np.mean([solution.moves(dataclasses.replace(state, fruit=cell)) for cell in cells])
            else:
                value = solution.moves(state)
            moves.append(value / moves_scale(state.free_cells))
        n = len(queries)
        return Prediction(np.zeros((n, 3)), np.ones(n), np.array(moves))

    decisions = []
    agent = solution.agent()
    for seed in range(20):

        def record(state):
            decisions.append((state, agent(state)))
            return decisions[-1][1]

        run_game(SnakeGame(GameConfig(4, 4), seed=seed), record, TRAINING_RULES)
    report = sibling_ranking(exact_moves, solution, decisions)
    for phase in ("first_half", "second_half"):
        assert report[phase]["decisions"] > 0
        assert report[phase]["difference_error"] == pytest.approx(0.0, abs=1e-9)
        assert report[phase]["order_agreement"] in (None, 1.0)
