import dataclasses
import random
import zlib

import numpy as np
import pytest

from snake_agent.contract import TRAINING_RULES, Outcome
from snake_agent.encoding import (
    GLOBALS,
    PLANES,
    encode,
    horizon_ceiling,
    horizon_scale,
    moves_ceiling,
    moves_scale,
)
from snake_agent.evaluation import run_game
from snake_agent.plans import (
    PlanAgent,
    PlanConfig,
    Planner,
    _direction,
    afterstate,
    candidates,
    choose,
    decide,
    extend,
    horizon_returns,
    next_trip,
    play,
    trust,
    waits,
)
from snake_agent.search import Prediction
from snake_agent.selfplay import Label, moves_returns, standard_starts
from snake_agent.survival import Board, arrival_times, opening_times
from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame


def follow(state, cells):
    """Play the flat ``cells`` on a copy of ``state`` with the engine; returns the results and the game."""
    game = SnakeGame.from_state(state, seed=0)
    results = []
    for cell in cells:
        head = game.head[1] * state.width + game.head[0]
        results.append(game.step(Direction(_direction(state.width, head, cell))))
    return results, game


@pytest.mark.parametrize("side", [6, 8])
@pytest.mark.parametrize("config", [PlanConfig(paths=6), PlanConfig(paths=6, hugs=2, gaps=6)])
def test_every_plan_is_safe_eats_at_its_end_and_reaches_its_afterstate(side, config):
    rng = random.Random(side)
    checked = looped = 0
    for seed in range(4):
        game = SnakeGame(GameConfig(side, side), seed=seed)
        while not game.done and not TRAINING_RULES.timed_out(game.state()):
            state = game.state()
            plans = candidates(state, TRAINING_RULES, config, rng)
            assert len({p.after.snake for p in plans}) == len(plans)  # one plan per afterstate
            for plan in plans:
                results, after = follow(state, plan.cells)
                assert not any(r.lost for r in results)
                assert [r.ate_fruit for r in results] == [False] * (plan.moves - 1) + [True]
                expected = after.state()
                if not expected.won:
                    expected = dataclasses.replace(expected, fruit=None)
                assert plan.after == expected
                assert plan.moves <= TRAINING_RULES.moves_left(state)
                checked += 1
                looped += plan.wait > 0
            if plans:  # a random plan, so that the games reach many kinds of position
                follow_on = rng.choice(plans)
                for cell in follow_on.cells:
                    head = game.head[1] * side + game.head[0]
                    game.step(Direction(_direction(side, head, cell)))
            else:
                options = waits(state, TRAINING_RULES)
                game.step_relative(options[0].action if options else RelativeAction.STRAIGHT)
    assert checked > 100 and looped > 0


def test_paths_enter_body_cells_only_once_vacated():
    # A 4x4 snake heading up along the left column; its tail (3, 3) is vacated after one move.
    snake = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3))
    state = GameState(4, 4, snake, Direction.UP, (3, 0))
    arrival = arrival_times(Board(state))
    flat = {(x, y): y * 4 + x for y in range(4) for x in range(4)}
    assert arrival[flat[(1, 0)]] == 1 and arrival[flat[(3, 0)]] == 3
    assert arrival[flat[(0, 1)]] == 7  # the neck, reached by following the body round as it moves away
    # (2, 3) is vacated after two moves; the head gets next to it after four, so it can enter at five.
    assert arrival[flat[(2, 3)]] == 5
    assert all(a <= 0 or Board(state).release(c) <= a for c, a in enumerate(arrival))


def test_afterstate_keeps_the_last_cells_entered():
    state = GameState(
        5, 5, ((2, 2), (2, 3), (2, 4)), Direction.UP, (4, 0), score=0, steps=7, steps_since_fruit=3
    )
    path = [7, 8, 9, 4]  # (2, 1), (3, 1), (4, 1), (4, 0)
    after = afterstate(state, path)
    assert after.snake == ((4, 0), (4, 1), (3, 1), (2, 1))  # four cells entered; the old body has moved on
    assert after.direction == Direction.UP and after.fruit is None
    assert (after.score, after.steps, after.steps_since_fruit) == (1, 11, 0)
    _, game = follow(state, path)
    assert dataclasses.replace(game.state(), fruit=None) == after


def test_plans_respect_the_fruit_timeout():
    game = SnakeGame(GameConfig(8, 8), seed=3)
    state = game.state()
    plans = candidates(state, TRAINING_RULES, PlanConfig(paths=8))
    shortest = min(p.moves for p in plans)
    limit = TRAINING_RULES.timeout(state)
    late = dataclasses.replace(state, steps_since_fruit=limit - shortest)
    assert all(p.moves <= shortest for p in candidates(late, TRAINING_RULES, PlanConfig(paths=8)))
    too_late = dataclasses.replace(state, steps_since_fruit=limit - shortest + 1)
    assert candidates(too_late, TRAINING_RULES, PlanConfig(paths=8)) == []


def test_the_fastest_acceptable_plan_is_chosen():
    wins = np.array([0.95, 0.99, 0.975, 0.5])
    moves = np.array([100.0, 300.0, 200.0, 10.0])
    assert choose(wins, moves, config=PlanConfig(tolerance=0.0)) == (True, 1)
    assert choose(wins, moves, config=PlanConfig(tolerance=0.02)) == (True, 2)  # within 0.02 of 0.99
    assert choose(wins, moves, config=PlanConfig(tolerance=0.05)) == (True, 0)
    rng = random.Random(0)
    explored = {
        choose(wins, moves, config=PlanConfig(tolerance=0.05, explore=1.0), rng=rng) for _ in range(50)
    }
    assert explored == {(True, 0), (True, 1), (True, 2)}
    # A plan is played unless waiting is safer by more than the tolerance.
    assert choose(wins, moves, [0.995], [400.0], PlanConfig(tolerance=0.02)) == (True, 2)
    assert choose(wins, moves, [0.995, 0.999], [400.0, 900.0], PlanConfig(tolerance=0.001)) == (False, 1)
    assert choose([], [], [0.3, 0.6], [50.0, 70.0], PlanConfig(tolerance=0.1)) == (False, 1)


def test_waiting_moves_neither_collide_nor_eat_nor_time_out():
    # Heading up along the left edge of a 5x5 board, the fruit just to the right of the head.
    snake = ((0, 2), (0, 3), (0, 4))
    state = GameState(5, 5, snake, Direction.UP, (1, 2), steps_since_fruit=4)
    options = waits(state, TRAINING_RULES)
    assert [w.action for w in options] == [RelativeAction.STRAIGHT]  # left is the wall, right eats
    assert options[0].state.snake == ((0, 1), (0, 2), (0, 3)) and options[0].state.fruit == (1, 2)
    assert options[0].state.steps_since_fruit == 5 and options[0].certified
    limit = TRAINING_RULES.timeout(state)
    assert waits(dataclasses.replace(state, steps_since_fruit=limit - 1), TRAINING_RULES) == []


def hashed(queries):
    """An evaluator whose output for a query depends on nothing else, so batching cannot change it."""
    n = len(queries)
    logits, win, moves = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    for i, q in enumerate(queries):
        key = repr((q.state.snake, q.state.direction, q.afterstate)).encode()
        rng = np.random.default_rng(zlib.crc32(key))
        win[i], moves[i] = rng.uniform(0.5, 1.0), rng.uniform(0.2, 1.0)
    return Prediction(logits, win, moves)


@pytest.mark.parametrize(
    "config",
    [
        PlanConfig(paths=4, tolerance=0.05),
        PlanConfig(paths=4, tolerance=0.05, hugs=2, gaps=4, persistence=3.0, certify=0.5, stall=True),
    ],
)
def test_batched_games_play_as_the_plan_agent_does_one_at_a_time(config):
    pytest.importorskip("torch")  # the lockstep driver lives in assess, which imports PyTorch
    from snake_agent.assess import _Lockstep, _runs, play_lockstep

    expected = [
        run_game(SnakeGame(GameConfig(6, 6), seed=s), PlanAgent(hashed, TRAINING_RULES, config, i), seed=s)
        for i, s in enumerate(range(3))
    ]
    games = [SnakeGame(GameConfig(6, 6), seed=s) for s in range(3)]
    runs = _runs("plans", games, TRAINING_RULES, [{"seed": s} for s in range(3)], envelope=False)
    lockstep = [_Lockstep(run, "plans", TRAINING_RULES, config, i) for i, run in enumerate(runs)]
    assert play_lockstep(hashed, lockstep) == expected
    assert any(r.outcome is Outcome.WON for r in expected) or any(r.fruits > 5 for r in expected)


def test_a_decision_asks_only_for_the_afterstates_that_are_not_won():
    # One free cell: the only plan fills the board, an exact win that needs no network call.
    snake = ((0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (2, 0))
    state = GameState(3, 3, snake + ((2, 2), (1, 2)), Direction.DOWN, (0, 2))
    generator = decide(state, TRAINING_RULES, PlanConfig())
    with pytest.raises(StopIteration) as stop:
        next(generator)
    decision = stop.value.value
    assert decision.plan.after.won and decision.win == 1.0 and decision.moves == 1


def test_self_play_examples_are_decision_states_and_afterstates_with_blended_targets():
    config = PlanConfig(paths=4, tolerance=0.0, certify=None)

    def constant(queries):
        n = len(queries)
        return Prediction(np.zeros((n, 3)), np.full(n, 0.8), np.full(n, 0.5))

    starts = standard_starts(4, 4, range(30))
    summaries, examples = play(constant, starts, TRAINING_RULES, config, seed=0, beta=0.5, moves_lambda=0.5)
    afterstates = examples.globals[:, GLOBALS.index("afterstate")] == 1.0
    assert afterstates.sum() == sum(s.fruits - (s.outcome is Outcome.WON) for s in summaries)
    assert (~afterstates).sum() == sum(s.searched for s in summaries)
    assert not examples.has_policy.any() and not examples.actions[afterstates].any()
    # Win targets blend the outcome with the value of a chosen option: 0.8, or 1 for a filled board.
    assert set(examples.label[afterstates]) <= {Label.AFTERSTATE}
    assert set(examples.label[~afterstates]) <= {Label.BLEND}
    for won in (True, False):
        rows = examples.has_moves == won
        z = float(won)
        blends = {0.5 * z + 0.5 * 0.8, 0.5 * z + 0.5}
        assert all(min(abs(w - b) for b in blends | {z}) < 1e-6 for w in examples.win[rows])
        assert all(min(abs(w - b) for b in blends) < 1e-6 for w in examples.win[rows & ~afterstates])
    assert any(s.outcome is Outcome.WON for s in summaries)


def test_long_games_keep_a_uniform_sample_of_their_examples_and_states():
    config = PlanConfig(paths=2, tolerance=0.0, certify=None)

    def constant(queries):
        n = len(queries)
        return Prediction(np.zeros((n, 3)), np.full(n, 0.8), np.full(n, 0.5))

    starts = standard_starts(8, 8, range(4))
    full, every = play(constant, starts, TRAINING_RULES, config, seed=0, samples=3)
    capped, sample = play(constant, starts, TRAINING_RULES, config, seed=0, samples=3, max_examples=7)
    assert [s.moves for s in full] == [s.moves for s in capped]  # sampling does not change the games
    assert len(sample) == sum(min(7, s.searched + s.fruits - (s.outcome is Outcome.WON)) for s in full)
    assert len(every) > len(sample)
    rows = {bytes(p) for p in every.planes.reshape(len(every), -1)}
    assert all(bytes(p) in rows for p in sample.planes.reshape(len(sample), -1))  # a subset of the examples
    for summary in capped:
        assert len(summary.samples) == min(3, summary.moves)
        assert [s.steps for s in summary.samples] == sorted(s.steps for s in summary.samples)


@pytest.mark.parametrize("certify", [None, 0.3])
def test_self_play_moves_targets_match_the_td_returns(certify):
    # Replay one won game by hand: the moves targets are moves_returns over its fruits. Certified
    # afterstates are no examples, and the moves played from them stand for their values.
    config = PlanConfig(paths=2, tolerance=0.0, certify=certify)

    def constant(queries):
        n = len(queries)
        return Prediction(np.zeros((n, 3)), np.full(n, 0.9), np.full(n, 0.25))

    for seed in range(20):
        summaries, examples = play(constant, standard_starts(4, 4, [seed]), TRAINING_RULES, config, seed=seed)
        if summaries[0].outcome is Outcome.WON:
            break
    else:
        pytest.fail("no 4x4 game was won")
    # Rebuild the game's eating moves and afterstates by replaying the same plans.
    game = standard_starts(4, 4, [seed])[0].game()
    rng = np.random.default_rng(seed)
    plan_rng = random.Random(int(rng.integers(1 << 62)))
    planner, eats, afters, certain, moves = Planner(), [], [], [], 0
    decision = None
    while not game.done:
        state = game.state()
        action = planner.next_action(state)
        if action is None:
            generator = decide(state, TRAINING_RULES, config, plan_rng)
            try:
                queries = next(generator)
                generator.send(constant(queries))
            except StopIteration as stop:
                decision = stop.value
            if decision.plan is None:
                game.step_relative(decision.wait)
                moves += 1
                continue
            planner.follow(decision.plan)
            action = planner.next_action(state)
        step = game.step_relative(action)
        moves += 1
        if step.ate_fruit and not step.won:
            eats.append(moves - 1)
            afters.append(game.state())
            certain.append(decision.certain)
    assert moves == summaries[0].moves
    assert any(certain) == (certify is not None)
    values = [
        moves - 1 - eat if sure else 0.25 * moves_scale(a.free_cells)
        for eat, a, sure in zip(eats, afters, certain, strict=True)
    ]
    _, expected = moves_returns(moves, eats, values, 0.5)
    kept = [(m, a) for m, a, sure in zip(expected, afters, certain, strict=True) if not sure]
    rows = examples.globals[:, GLOBALS.index("afterstate")] == 1.0
    assert np.allclose(examples.moves[rows], [m / moves_scale(a.free_cells) for m, a in kept], atol=1e-5)


def test_without_the_cycle_the_encoding_leaves_its_inputs_at_zero():
    state = SnakeGame(GameConfig(6, 6), seed=0).state()
    full, bare = encode(state, TRAINING_RULES), encode(state, TRAINING_RULES, cycle=False)
    cycle_planes = [
        i for i, name in enumerate(PLANES) if name.startswith("cycle") or name in ("envelope_free", "hole")
    ]
    assert full.planes[cycle_planes].any() and not bare.planes[cycle_planes].any()
    for name in ("in_cycle_order", "fruit_ahead", "tail_ahead"):
        assert full.globals[GLOBALS.index(name)] != 0 and bare.globals[GLOBALS.index(name)] == 0
    assert full.actions[:, 6:].any() and not bare.actions[:, 6:].any()  # envelope and jump
    others = [i for i in range(len(PLANES)) if i not in cycle_planes]
    assert np.array_equal(full.planes[others], bare.planes[others])


def test_a_lengthened_path_sweeps_free_cells_and_stays_safe():
    state = GameState(6, 6, ((1, 4), (1, 5), (0, 5)), Direction.UP, (1, 0))
    board = Board(state)
    path = [y * 6 + 1 for y in (3, 2, 1, 0)]  # straight up to the fruit
    longer = extend(board, path, 6 * 0 + 1, 20, random.Random(0))
    assert len(longer) == len(path) + 20 and longer[-1] == path[-1]
    assert len(set(longer)) == len(longer) and set(path) <= set(longer)
    results, _ = follow(state, longer)
    assert not any(r.lost for r in results) and [r.ate_fruit for r in results][-1]
    assert extend(board, path, 1, 0, random.Random(0)) == path


def test_the_next_trip_is_the_earliest_arrival_at_the_fruit_or_its_mean_over_free_cells():
    # A one-cell snake at the left end of a 3x1 board, heading right.
    state = GameState(3, 1, ((0, 0),), Direction.RIGHT, (2, 0))
    assert next_trip(state) == 2.0
    assert next_trip(dataclasses.replace(state, fruit=None)) == 1.5  # cells 1 and 2 moves away
    # Heading up the left column of a 4x4 board (as in the test above): the fruit (3, 0) is 3 moves away,
    # and the nine free cells 1, 2, 3, 2, 3, 4, 3, 4 and 5 moves away, 3 on average.
    snake = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3))
    column = GameState(4, 4, snake, Direction.UP, (3, 0))
    assert next_trip(column) == 3.0 and next_trip(dataclasses.replace(column, fruit=None)) == 3.0
    # Waiting can only help: every cell opens no later than the head arrives without waiting, and the
    # body cells it never reaches on time, such as the neck, open once they move away.
    board = Board(column)
    arrival, opening = arrival_times(board), opening_times(board)
    assert all(0 < o <= a for a, o in zip(arrival, opening, strict=True) if a > 0)
    assert arrival[4] == 7 and opening[4] == 6  # (0, 1), the neck, released after 6 moves


def test_a_horizon_counts_the_moves_to_eat_that_many_more_fruits():
    # Ten moves; fruits eaten at moves 2 and 5, and the board filled at move 9.
    moves, afters = horizon_returns(10, [2, 5], True, 2)
    assert moves[0] == 6 and moves[3] == 7 and moves[6] == 4  # 0..5, 3..9, then to the full board
    assert afters == [7, 4]
    lost_moves, lost_afters = horizon_returns(10, [2, 5], False, 2)
    assert lost_moves[0] == 6 and lost_moves[3] is None and lost_afters == [None, None]
    assert horizon_returns(10, [2, 5], True, 1) == ([3, 2, 1, 3, 2, 1, 4, 3, 2, 1], [3, 4])
    for free in (1, 7, 100):
        assert horizon_scale(free) == moves_scale(free) == horizon_scale(free, free + 3)
        assert horizon_ceiling(400, free) == pytest.approx(moves_ceiling(400, free))
    assert horizon_scale(100, 8) == 8 * (200 - 8 + 3) / 4
    assert trust(300, PlanConfig(), 8) > 0.3 > 0.01 > trust(300, PlanConfig())


def test_self_play_with_a_horizon_targets_the_moves_to_the_next_fruits():
    config = PlanConfig(paths=2, tolerance=0.0, certify=None)

    def constant(queries):
        n = len(queries)
        return Prediction(np.zeros((n, 3)), np.full(n, 0.8), np.full(n, 0.5), horizon=1)

    summaries, examples = play(constant, standard_starts(6, 6, range(4)), TRAINING_RULES, config, horizon=1)
    rows = examples.has_moves
    assert rows.any() and any(s.outcome is not Outcome.WON for s in summaries) or rows.all()
    moves = examples.moves[rows] * np.array([horizon_scale(f, 1) for f in examples.free[rows]])
    assert np.allclose(moves, np.round(moves), atol=1e-3) and (moves >= 1 - 1e-6).all()
    after = examples.globals[rows, GLOBALS.index("afterstate")] == 1.0
    assert moves[after].min() > 1 - 1e-6  # from an afterstate the next fruit is at least a move away
