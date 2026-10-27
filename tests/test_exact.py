import dataclasses
import functools
import math
import statistics

import numpy as np
import pytest

from snake_agent.contract import NO_LIMIT, TRAINING_RULES
from snake_agent.cycles import CycleEnvelope, ShortcutCyclePolicy
from snake_agent.evaluation import AgentSpec, evaluate_standard
from snake_agent.exact import evaluate_policy, solve, standard_starts
from snake_agent.survival import Board, prove_lost
from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame, Status


def every_state(solution):
    """Each running state of a solved board, with the move counters at zero."""
    for level in solution._levels.values():
        for body in range(len(level.keys)):
            snake = tuple((c % solution.width, c // solution.width) for c in level.bodies[body])
            heading = Direction(int(level.heading[body]))
            for fruit in level.free[body]:
                cell = int(fruit)
                yield GameState(
                    solution.width,
                    solution.height,
                    snake,
                    heading,
                    (cell % solution.width, cell // solution.width),
                )


def brute_force(width, height, limit):
    """Expectimax through the real engine: (value, certain-win moves, action values).

    ``limit(state)`` is the number of moves allowed without fruit; the whole game
    tree is finite because every allowance is.
    """

    @functools.cache
    def solve_state(state):
        values, moves = [], []
        for action in RelativeAction:
            game = SnakeGame.from_state(state, seed=0)
            result = game.step_relative(action)
            if result.lost:
                value, cost = 0.0, math.inf
            elif result.won:
                value, cost = 1.0, 1.0
            elif result.ate_fruit:  # the next fruit is uniform over the free cells
                after = game.state()
                free = [(x, y) for y in range(height) for x in range(width) if (x, y) not in after.snake]
                outcomes = [solve_state(dataclasses.replace(after, fruit=f, score=0, steps=0)) for f in free]
                value = sum(o[0] for o in outcomes) / len(free)
                cost = 1.0 + sum(o[1] for o in outcomes) / len(free) if value == 1.0 else math.inf
            else:
                after = dataclasses.replace(game.state(), score=0, steps=0)
                if after.steps_since_fruit >= limit(after):
                    value, cost = 0.0, math.inf
                else:
                    value, rest, _ = solve_state(after)
                    cost = 1.0 + rest if value == 1.0 else math.inf
            values.append(value)
            moves.append(cost)
        best = max(values)
        certain = min((m for v, m in zip(values, moves, strict=True) if v == 1.0), default=math.inf)
        return best, certain, tuple(values)

    return solve_state


@pytest.mark.parametrize("width,height,timeout", [(2, 3, True), (3, 2, False), (2, 4, True), (3, 3, True)])
def test_the_solution_agrees_with_brute_force_through_the_engine(width, height, timeout):
    cells = width * height
    solution = solve(width, height, fruit_timeout=timeout)
    # Without the timeout, an allowance of 6 N moves is already unlimited on these boards.
    brute = brute_force(width, height, (lambda s: cells + s.length) if timeout else (lambda s: 6 * cells))
    checked = 0
    for state in every_state(solution):
        budgets = (0, 1, 2, cells + state.length - 1) if timeout else (0,)
        for since in sorted(set(budgets)):
            if since >= cells + state.length:
                continue
            state = dataclasses.replace(state, steps_since_fruit=since)
            value, moves, actions = brute(state)
            assert solution.value(state) == pytest.approx(value, abs=1e-12)
            assert solution.action_values(state) == pytest.approx(actions, abs=1e-12)
            if moves == math.inf:
                assert math.isnan(solution.moves(state))
            else:
                assert solution.moves(state) == pytest.approx(moves, abs=1e-9)
            checked += 1
    assert checked >= solution.states


@pytest.fixture(scope="module")
def four():
    start = SnakeGame(GameConfig(4, 4), seed=0).state()
    envelope = CycleEnvelope.for_state(start)
    return {
        "rules": solve(4, 4),
        "no limit": solve(4, 4, fruit_timeout=False),
        "envelope": solve(4, 4, envelope=envelope),
        "cycle": envelope,
    }


def test_4x4_size(four):
    assert (
        four["rules"].states == 144_088
    )  # 143,368 in the probe, which counts one heading for one-cell bodies


def test_4x4_standard_start_is_a_certain_win(four):
    for solution in four.values():
        if isinstance(solution, CycleEnvelope):
            continue
        assert solution.start_value() == 1.0


def test_4x4_moves_from_the_standard_start(four):
    cycle = four["cycle"].cycle

    def plain(state):
        x, y = cycle[state.head]
        return Direction.from_delta(x - state.head[0], y - state.head[1])

    plain_cycle = evaluate_policy(4, 4, plain)
    assert (plain_cycle.win, plain_cycle.moves) == (
        1.0,
        pytest.approx(52.0, abs=1e-9),
    )  # F(F + 3) / 4, F = 13
    shortcut = ShortcutCyclePolicy()
    shortcut(standard_starts(4, 4)[0])
    shortcut_cycle = evaluate_policy(4, 4, shortcut)
    assert shortcut_cycle.win == 1.0
    inside = four["envelope"].start_moves()
    best = four["rules"].start_moves()
    assert best < inside < shortcut_cycle.moves < plain_cycle.moves
    assert (best, inside, shortcut_cycle.moves) == (
        pytest.approx(35.469204, abs=1e-6),
        pytest.approx(43.084786, abs=1e-6),
        pytest.approx(43.887971, abs=1e-6),
    )
    assert four["no limit"].start_moves() == pytest.approx(35.469201, abs=1e-6)


def test_the_timeout_costs_win_probability_in_few_4x4_states(four):
    lowered = 0
    for length in four["rules"]._levels:
        with_rule = four["rules"]._values[length][-1]
        without = four["no limit"]._values[length][-1]
        assert (with_rule <= without).all()
        lowered += int((with_rule < without).sum())
    assert lowered == 184


def test_values_never_fall_with_more_moves_left(four):
    for stack in four["rules"]._values.values():
        assert (np.diff(stack, axis=0) >= 0).all()


def test_values_moves_and_best_actions_are_consistent(four):
    solution = four["rules"]
    for i, state in enumerate(every_state(solution)):
        if i % 37:
            continue
        values = solution.action_values(state)
        assert solution.value(state) == max(values)
        best = solution.best_action(state)
        assert solution.regret(state, best) == 0.0
        assert all(solution.regret(state, a) >= 0.0 for a in RelativeAction)
        if solution.value(state) == 1.0:
            assert solution.moves(state) == min(solution.action_moves(state))


def test_proven_lost_states_have_value_zero(four):
    for solution in (solve(3, 3), solve(3, 4)):
        for state in every_state(solution):
            if prove_lost(Board(state), 10_000).lost:
                assert solution.value(state) == 0.0
    proven = 0
    for state in every_state(four["rules"]):
        if prove_lost(Board(state), 10_000).lost:
            assert four["rules"].value(state) == four["no limit"].value(state) == 0.0
            proven += 1
    assert proven == 33_960


def test_queries_check_the_state(four):
    solution = four["rules"]
    start = standard_starts(4, 4)[0]
    with pytest.raises(ValueError):
        solution.value(SnakeGame(GameConfig(5, 4), seed=0).state())
    with pytest.raises(ValueError):  # the heading must point from the neck to the head
        solution.value(dataclasses.replace(start, direction=Direction.UP))
    assert solution.value(dataclasses.replace(start, steps_since_fruit=19)) == 0.0  # timed out
    won = GameState(2, 2, ((0, 1), (0, 0), (1, 0), (1, 1)), Direction.DOWN, None, status=Status.WON)
    assert solve(2, 2).value(won) == 1.0 and solve(2, 2).moves(won) == 0.0
    with pytest.raises(ValueError):
        solve(4, 4, min_length=0)


def test_the_exact_agents_play_their_expected_moves(four):
    def close_to_expected(evaluation, expected):
        moves = [r.moves for r in evaluation.records]
        return abs(statistics.fmean(moves) - expected) < 4 * statistics.stdev(moves) / math.sqrt(len(moves))

    envelope_agent = AgentSpec("exact-envelope", lambda seed: four["envelope"].agent(), deterministic=True)
    inside = evaluate_standard(envelope_agent, 4, 4, range(300), TRAINING_RULES)
    assert inside.summary().won == 300 and inside.summary().envelope_exits == 0
    assert close_to_expected(inside, four["envelope"].start_moves())
    free_agent = AgentSpec("exact", lambda seed: four["rules"].agent(), deterministic=True)
    free = evaluate_standard(free_agent, 4, 4, range(300), NO_LIMIT)
    assert free.summary().won == 300
    assert free.summary().envelope_exits > 0  # the unconstrained optimum leaves the envelope
    assert close_to_expected(free, four["rules"].start_moves())
