import dataclasses
import zlib

import numpy as np
import pytest

pytest.importorskip("torch")

from snake_agent.assess import (  # noqa: E402
    AGENTS,
    _Lockstep,
    _runs,
    calibration,
    play_lockstep,
    sibling_ranking,
    uniform_free_states,
)
from snake_agent.contract import TRAINING_RULES  # noqa: E402
from snake_agent.cycles import CycleEnvelope  # noqa: E402
from snake_agent.encoding import moves_scale  # noqa: E402
from snake_agent.evaluation import run_game  # noqa: E402
from snake_agent.exact import solve, solve_envelope  # noqa: E402
from snake_agent.recovery import recovery_starts  # noqa: E402
from snake_agent.search import EVALUATION, PolicyAgent, Prediction, SearchAgent, envelope_certain  # noqa: E402
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


def hashed(queries):
    """An evaluator whose output for a query depends on nothing else, so batching cannot change it."""
    n = len(queries)
    logits, win, moves = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    for i, q in enumerate(queries):
        key = repr((q.state.snake, q.state.direction, q.state.fruit, q.afterstate)).encode()
        rng = np.random.default_rng(zlib.crc32(key))
        logits[i], win[i], moves[i] = rng.normal(size=3), rng.uniform(0.2, 0.9), rng.uniform(0.5, 1.5)
    return Prediction(logits, win, moves)


@pytest.mark.parametrize("kind", AGENTS)
def test_batched_games_play_as_the_agents_do_one_at_a_time(kind):
    def agent(seed):
        return (
            SearchAgent(hashed, TRAINING_RULES, EVALUATION, seed) if kind == "search" else PolicyAgent(hashed)
        )

    games = [SnakeGame(GameConfig(6, 6), seed=s) for s in range(3)]
    expected = [
        run_game(
            SnakeGame(GameConfig(6, 6), seed=s), agent(i), envelope=CycleEnvelope.for_state(g.state()), seed=s
        )
        for i, (s, g) in enumerate(zip(range(3), games, strict=True))
    ]
    runs = _runs(kind, games, TRAINING_RULES, [{"seed": s} for s in range(3)], envelope=True)
    decisions = []
    lockstep = [_Lockstep(run, kind, TRAINING_RULES, EVALUATION, i, decisions) for i, run in enumerate(runs)]
    assert play_lockstep(hashed, lockstep) == expected
    assert len(decisions) == sum(r.moves for r in expected)
    # Free play from recovery starts, with the T2 labels.
    starts = recovery_starts(6, 6, range(4))[:3]
    games = [SnakeGame.from_state(s.state, seed=100 + i) for i, s in enumerate(starts)]
    expected = [
        run_game(
            SnakeGame.from_state(s.state, seed=100 + i), agent(i), start=s.name, seed=100 + i, band=s.band
        )
        for i, s in enumerate(starts)
    ]
    labels = [{"start": s.name, "seed": 100 + i, "band": s.band} for i, s in enumerate(starts)]
    runs = _runs(kind, games, TRAINING_RULES, labels, envelope=False)
    lockstep = [_Lockstep(run, kind, TRAINING_RULES, EVALUATION, i) for i, run in enumerate(runs)]
    assert play_lockstep(hashed, lockstep) == expected
