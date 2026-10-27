import dataclasses
import random
import statistics

import numpy as np
import pytest

from snake_agent.contract import TRAINING_RULES
from snake_agent.encoding import canonical_envelope, moves_scale
from snake_agent.evaluation import AgentSpec, evaluate_standard
from snake_agent.exact import solve
from snake_agent.search import (
    CERTAIN,
    EVALUATION,
    Mode,
    PolicyAgent,
    Prediction,
    SearchAgent,
    SearchConfig,
    considered_visits,
    envelope_certain,
    position_key,
    run_search,
    search,
)
from snake_agent.survival import MoveClass, classify_moves
from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame

ONE_VISIT = SearchConfig(simulations=3, gumbel_scale=0.0)


@pytest.fixture(scope="module")
def exact():
    return {"free": solve(4, 4), "envelope": solve(4, 4, envelope=canonical_envelope(4, 4))}


def exact_evaluator(solution, informed=False, calls=None):
    """An evaluator with the exact values of ``solution``.

    The logits are uniform, or with ``informed`` point to the best move as an
    ideal network would. ``calls`` collects the batch sizes.
    """

    def outcomes(state):
        if state.fruit is not None:
            return [state]
        snake = set(state.snake)
        cells = [(x, y) for y in range(state.height) for x in range(state.width) if (x, y) not in snake]
        return [dataclasses.replace(state, fruit=cell) for cell in cells]

    def evaluate(queries):
        n = len(queries)
        logits, win, moves = np.zeros((n, 3)), np.zeros(n), np.ones(n)  # moves matter only for certain wins
        for i, query in enumerate(queries):
            states = outcomes(query.state)
            win[i] = statistics.fmean(solution.value(s) for s in states)
            if win[i] == 1.0:
                moves[i] = statistics.fmean(solution.moves(s) for s in states)
                moves[i] /= moves_scale(query.state.free_cells)
            if informed and not query.afterstate:
                values, costs = solution.action_values(query.state), solution.action_moves(query.state)
                logits[i] = [
                    10.0 * v - (c if np.isfinite(c) else 1e3) / 10.0
                    for v, c in zip(values, costs, strict=True)
                ]
        if calls is not None:
            calls.append(n)
        return Prediction(logits, win, moves)

    return evaluate


def uniform_evaluator(calls=None):
    def evaluate(queries):
        if calls is not None:
            calls.append(len(queries))
        n = len(queries)
        return Prediction(np.zeros((n, 3)), np.full(n, 0.5), np.ones(n))

    return evaluate


def states_of(solution, stride, keep):
    """Every ``stride``-th running state of a solved board (move counters at zero) that ``keep`` accepts."""
    states, i = [], 0
    for level in solution._levels.values():
        for body in range(len(level.keys)):
            snake = tuple((c % solution.width, c // solution.width) for c in level.bodies[body])
            heading = Direction(int(level.heading[body]))
            for fruit in level.free[body]:
                i += 1
                if i % stride:
                    continue
                cell = int(fruit)
                state = GameState(4, 4, snake, heading, (cell % 4, cell // 4))
                if keep(state):
                    states.append(state)
    return states


def free_states(solution, stride=199):
    return states_of(solution, stride, lambda s: not envelope_certain(s, TRAINING_RULES))


def envelope_states(solution, stride=7):
    return states_of(solution, stride, lambda s: s.length >= 3 and envelope_certain(s, TRAINING_RULES))


def one_visit(evaluator, state):
    """The search with one simulation per valid root move, so each move's child is evaluated once."""
    probe = run_search(uniform_evaluator(), state, TRAINING_RULES, SearchConfig(simulations=0))
    config = SearchConfig(simulations=max(sum(probe.valid), 1), gumbel_scale=0.0)
    return run_search(evaluator, state, TRAINING_RULES, config)


def test_sequential_halving_schedule_matches_mctx():
    assert considered_visits(3, 16) == (0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
    assert considered_visits(2, 16) == tuple(v for v in range(8) for _ in range(2))
    assert considered_visits(1, 4) == (0, 1, 2, 3)
    assert considered_visits(3, 3) == (0, 0, 0)


def test_one_visit_per_move_gives_the_exact_action_values(exact):
    """With exact leaves, a search that visits each root move once sees the exact action values.

    Learned values stay below 1 (``CERTAIN``), so only exact wins (the envelope,
    a filled board) tie at 1 and are compared on moves.
    """
    solution = exact["free"]
    checked = 0
    for state in free_states(solution):
        result = one_visit(exact_evaluator(solution), state)
        assert result.mode is Mode.FREE
        values, costs = solution.action_values(state), solution.action_moves(state)
        if not any(result.valid):
            assert result.lost and solution.value(state) == 0.0
            continue
        for a in RelativeAction:
            if not result.valid[a]:
                assert values[a] == 0.0  # pruning is sound: fatal, proven lost or timed out
            elif not result.forced:
                assert result.q_win[a] == pytest.approx(min(values[a], CERTAIN), abs=1e-12) or (
                    result.q_win[a] == values[a] == 1.0
                )
                if result.q_win[a] == 1.0:
                    assert result.q_moves[a] == pytest.approx(costs[a], rel=1e-9)
        best = max(values[a] for a in RelativeAction if result.valid[a])
        assert best == solution.value(state)
        if result.forced:
            continue
        assert values[result.action] >= best - 1e-6
        exact_wins = [a for a in RelativeAction if result.valid[a] and result.q_win[a] == 1.0]
        if exact_wins:
            assert costs[result.action] == pytest.approx(min(costs[a] for a in exact_wins), rel=1e-9)
        checked += 1
    assert checked > 150


def test_envelope_search_compares_exact_moves(exact):
    solution = exact["envelope"]
    envelope = canonical_envelope(4, 4)
    checked = 0
    for state in envelope_states(solution):
        result = one_visit(exact_evaluator(solution), state)
        assert result.mode is Mode.ENVELOPE and not result.lost
        assert result.valid == envelope.mask(state)
        assert envelope.allows(state, result.action)
        if result.forced:
            continue
        costs = solution.action_moves(state)
        valid = [a for a in RelativeAction if result.valid[a]]
        assert (result.q_win[valid] == 1.0).all()
        for a in valid:
            assert result.q_moves[a] == pytest.approx(costs[a], rel=1e-9)
        assert costs[result.action] == pytest.approx(min(costs), rel=1e-9)
        checked += 1
    assert checked > 100


def test_deeper_search_with_an_ideal_network_plays_optimally(exact):
    solution = exact["free"]
    config = SearchConfig(simulations=16, gumbel_scale=0.0)
    rng = np.random.default_rng(0)
    regrets = []
    for state in free_states(solution, stride=1601):
        result = run_search(exact_evaluator(solution, informed=True), state, TRAINING_RULES, config, rng)
        regrets.append(solution.regret(state, result.action))
        if not result.forced:
            assert ((0.0 <= result.q_win) | np.isnan(result.q_win)).all()
            assert result.policy.sum() == pytest.approx(1.0) and result.visits.sum() == 16
    assert len(regrets) > 40 and max(regrets) == 0.0


def test_forced_moves_need_no_evaluation():
    # Heading up along the left wall with the body to the right: only straight is open.
    snake = ((0, 1), (0, 2), (1, 2), (1, 1), (1, 0))
    state = GameState(4, 4, snake, Direction.UP, (3, 3))
    calls = []
    result = run_search(uniform_evaluator(calls), state)
    assert result.forced and result.valid == (False, True, False) and not calls
    assert result.action is RelativeAction.STRAIGHT and result.policy is None and result.simulations == 0
    searched = run_search(uniform_evaluator(calls), state, config=SearchConfig(search_forced=True))
    assert not searched.forced and searched.visits.tolist() == [0, 16, 0] and calls


def test_a_state_without_a_safe_move_is_lost():
    # In the corner heading left: the wall ahead and above, the body below.
    state = GameState(4, 4, ((0, 0), (1, 0), (1, 1), (0, 1), (0, 2)), Direction.LEFT, (3, 3))
    result = run_search(uniform_evaluator(), state)
    assert result.lost and result.forced and result.valid == (False, False, False)


def test_proven_lost_moves_are_pruned_at_the_root(exact):
    solution = exact["free"]
    found = 0
    for state in free_states(solution, stride=53):
        classes = classify_moves(state, 10_000)
        if MoveClass.LOST not in classes or all(c in (MoveClass.LOST, MoveClass.FATAL) for c in classes):
            continue
        result = run_search(uniform_evaluator(), state, config=ONE_VISIT)
        for a in RelativeAction:
            if classes[a] in (MoveClass.LOST, MoveClass.FATAL):
                assert not result.valid[a]
        assert not result.lost
        found += 1
    assert found > 10


def test_every_move_proven_lost_or_fatal_marks_the_state_lost(exact):
    solution = exact["free"]
    found = 0
    for state in free_states(solution, stride=53):
        classes = classify_moves(state, 10_000)
        if MoveClass.LOST in classes and all(c in (MoveClass.LOST, MoveClass.FATAL) for c in classes):
            result = run_search(uniform_evaluator(), state, config=ONE_VISIT)
            assert result.lost and solution.value(state) == 0.0
            assert any(result.valid)  # it still plays a move that is not fatal
            assert classes[result.action] is MoveClass.LOST
            found += 1
    assert found > 10


def free_6x6_state():
    """The 6x6 start late in its fruit epoch, the fruit too far along the cycle: free mode."""
    start = SnakeGame(GameConfig(6, 6), seed=0).state()
    envelope = canonical_envelope(6, 6)
    snake = set(start.snake)
    cells = [(x, y) for y in range(6) for x in range(6) if (x, y) not in snake]
    far = max(cells, key=lambda cell: envelope.ahead(start, cell))
    return dataclasses.replace(start, fruit=far, steps_since_fruit=30)


def test_moves_that_time_out_are_pruned():
    start = dataclasses.replace(SnakeGame(GameConfig(4, 4), seed=0).state(), fruit=(0, 0))
    late = dataclasses.replace(start, steps_since_fruit=TRAINING_RULES.timeout(start) - 1)
    assert not envelope_certain(late, TRAINING_RULES)
    result = run_search(uniform_evaluator(), late)
    assert result.mode is Mode.FREE and result.lost  # every move times out: the fruit is not next to the head
    # With the fruit next to the head, eating is the only move left.
    head = start.head
    fed = dataclasses.replace(late, fruit=(head[0], head[1] - 1))
    result = run_search(uniform_evaluator(), fed)
    assert result.forced and not result.lost and result.action is RelativeAction.LEFT


def test_moves_back_to_a_position_of_the_epoch_are_pruned():
    state = free_6x6_state()
    game = SnakeGame.from_state(state, seed=0)
    game.step_relative(RelativeAction.STRAIGHT)
    history = frozenset({position_key(game.state())})
    free = run_search(uniform_evaluator(), state, config=ONE_VISIT)
    pruned = run_search(uniform_evaluator(), state, config=ONE_VISIT, history=history)
    assert free.mode is Mode.FREE and free.valid[RelativeAction.STRAIGHT]
    assert not pruned.valid[RelativeAction.STRAIGHT] and any(pruned.valid)


def test_queries_and_afterstates():
    state = GameState(4, 4, ((1, 1), (0, 1), (0, 2)), Direction.RIGHT, (2, 1))
    queries = []
    generator = search(state, TRAINING_RULES, SearchConfig(simulations=16), np.random.default_rng(1))
    evaluate = uniform_evaluator()
    try:
        query = next(generator)
        while True:
            queries.append(query)
            query = generator.send(evaluate([query]).row(0))
    except StopIteration as stop:
        result = stop.value
    assert result.mode is Mode.FREE
    assert queries[0].state == state and not queries[0].afterstate and queries[0].classes is not None
    afterstates = [q for q in queries if q.afterstate]
    assert afterstates and all(q.state.fruit is None and q.state.length == 4 for q in afterstates)
    assert result.visits.sum() == 16 and not result.forced
    assert len(queries) <= 17  # the root and at most one new leaf per simulation


def test_the_search_is_reproducible():
    state = free_6x6_state()
    a = run_search(uniform_evaluator(), state, rng=np.random.default_rng(7))
    b = run_search(uniform_evaluator(), state, rng=np.random.default_rng(7))
    assert a.mode is Mode.FREE and not a.forced
    assert a.action == b.action and np.array_equal(a.visits, b.visits) and np.allclose(a.policy, b.policy)


def play_t1(spec, games=40):
    return evaluate_standard(spec, 4, 4, range(games), TRAINING_RULES)


def test_the_search_agent_wins_inside_the_envelope_at_the_optimal_pace(exact):
    solution = exact["envelope"]
    spec = AgentSpec(
        "exact-search",
        lambda seed: SearchAgent(exact_evaluator(solution, informed=True), config=EVALUATION, seed=seed),
    )
    evaluation = play_t1(spec, games=200)
    summary = evaluation.summary()
    assert summary.won == 200 and summary.envelope_exits == 0
    moves = [r.moves for r in evaluation.records]
    error = statistics.stdev(moves) / len(moves) ** 0.5
    assert abs(summary.mean_moves - solution.start_moves()) < 4 * error


def test_the_policy_agent_follows_the_logits_inside_the_envelope(exact):
    solution = exact["envelope"]
    spec = AgentSpec("exact-policy", lambda seed: PolicyAgent(exact_evaluator(solution, informed=True)))
    summary = play_t1(spec).summary()
    assert summary.won == 40 and summary.envelope_exits == 0


def test_agents_remember_the_positions_since_the_last_fruit():
    calls = []
    agent = SearchAgent(uniform_evaluator(calls), seed=0)
    rng = random.Random(0)
    game = SnakeGame(GameConfig(6, 6), seed=2)
    seen = 0
    while not game.done and seen < 30:
        state = game.state()
        agent(state)
        if state.steps_since_fruit == 0:
            assert agent.epoch.positions == {position_key(state)}
        seen += 1
        game.step(rng.choice(game.safe_directions() or [game.direction]))
    assert calls
