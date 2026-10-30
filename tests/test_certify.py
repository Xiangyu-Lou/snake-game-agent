import itertools
import random

import numpy as np
import pytest

from snake_agent.certify import (
    after_body,
    certified,
    certified_paths,
    certify,
    extends,
    field_arrival,
    flat_body,
    follow,
    greedy_walk,
    layout,
    prefix,
    recent_run,
    requirements,
    shortcuts,
    spanning,
    stall,
    successor,
    timely,
    transition_paths,
)
from snake_agent.contract import TRAINING_RULES, play_move
from snake_agent.plans import PlanAgent, PlanConfig, _direction, _View, _waits, decide, run_decision
from snake_agent.search import Prediction
from snake_agent.survival import Board
from snake_game import Direction, GameConfig, GameState, SnakeGame
from snake_game.policies import GreedyPolicy


def steps(game, cells):
    """Play the flat ``cells`` with the engine; returns the step results."""
    width = game.config.width
    results = []
    for cell in cells:
        head = game.head[1] * width + game.head[0]
        results.append(game.step(Direction(_direction(width, head, cell))))
    return results


@pytest.mark.parametrize("size", [(2, 2), (4, 4), (6, 4), (4, 10), (20, 20)])
@pytest.mark.parametrize("orientation", [0, 1])
def test_the_walk_around_a_spanning_tree_is_a_hamiltonian_cycle(size, orientation):
    width, height = size
    lay = layout(width, height, orientation)
    rng = random.Random(0)
    for _ in range(10):
        tree = spanning(lay, {}, rng)
        cell, seen = 0, set()
        for _ in range(width * height):
            seen.add(cell)
            nxt = successor(lay, tree, cell)
            assert abs(nxt % width - cell % width) + abs(nxt // width - cell // width) == 1
            assert nxt in (lay.cross[cell], lay.side[cell])
            cell = nxt
        assert cell == 0 and len(seen) == width * height
    with pytest.raises(ValueError):
        layout(5, 4, orientation)


def all_trees(lay):
    """Every spanning tree of the blocks, as membership per edge."""
    trees = []
    for chosen in itertools.combinations(range(len(lay.ends)), lay.blocks - 1):
        parent = list(range(lay.blocks))

        def find(a, parent=parent):
            while parent[a] != a:
                a = parent[a]
            return a

        for e in chosen:
            a, b = find(lay.ends[e][0]), find(lay.ends[e][1])
            if a == b:
                break
            parent[a] = b
        else:
            trees.append([e in chosen for e in range(len(lay.ends))])
    return trees


@pytest.mark.parametrize("orientation", [0, 1])
def test_a_body_is_certified_exactly_when_the_walk_around_some_tree_holds_it(orientation):
    width, height = 6, 4
    lay = layout(width, height, orientation)
    walks = [[successor(lay, tree, c) for c in range(width * height)] for tree in all_trees(lay)]
    assert len(walks) == 15
    rng = random.Random(orientation)
    held = 0
    for _ in range(2000):
        body = [rng.randrange(width * height)]
        for _ in range(rng.randrange(width * height)):
            x, y = body[-1] % width, body[-1] // width
            options = [
                (y + dy) * width + x + dx
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= x + dx < width and 0 <= y + dy < height and (y + dy) * width + x + dx not in body
            ]
            if not options:
                break
            body.append(rng.choice(options))
        truth = any(all(nxt[body[k]] == body[k - 1] for k in range(1, len(body))) for nxt in walks)
        assert certified(lay, body) == truth
        held += truth
    assert held > 100


@pytest.mark.parametrize("side", [6, 10])
def test_the_standard_start_is_certified_and_following_the_walk_wins(side):
    game = SnakeGame(GameConfig(side, side), seed=1)
    lay = certify(game.state())
    assert lay is not None
    rng = random.Random(0)
    while not game.done:
        state = game.state()
        path = follow(state, lay, rng)
        assert len(path) <= state.free_cells
        results = steps(game, path)
        assert not any(r.lost for r in results) and results[-1].ate_fruit
        assert game.won or certify(game.state()) == lay
    assert game.won


def certified_states(side, games, rng):
    """States of games that follow random walks from the standard start, with their layouts."""
    for seed in range(games):
        game = SnakeGame(GameConfig(side, side), seed=seed)
        lay = certify(game.state())
        while not game.done:
            state = game.state()
            yield state, lay
            steps(game, follow(state, lay, rng))


def test_certified_paths_are_safe_and_keep_the_certificate():
    rng = random.Random(1)
    checked = shorter = 0
    for state, lay in certified_states(8, 3, rng):
        paths = certified_paths(state, lay, rng, trees=8, count=4, budget=2000)
        walk_moves = len(paths[0])
        for path in paths:
            game = SnakeGame.from_state(state, seed=0)
            results = steps(game, path)
            assert not any(r.lost for r in results)
            assert [r.ate_fruit for r in results] == [False] * (len(path) - 1) + [True]
            assert timely(state, path)
            assert game.won or certified(lay, flat_body(game.state()))
            assert flat_body(game.state()) == after_body(flat_body(state), path)
            checked += 1
            shorter += len(path) < walk_moves
    assert checked > 200 and shorter > 10


def test_shortcuts_come_shortest_first_and_end_below_the_bound():
    rng = random.Random(2)
    found = 0
    for state, lay in certified_states(10, 2, rng):
        width = state.width
        fruit = state.fruit[1] * width + state.fruit[0]
        bound = len(follow(state, lay, rng))
        paths = shortcuts(state, lay, bound, rng, count=4, budget=5000)
        lengths = [len(p) for p in paths]
        assert lengths == sorted(lengths) and all(n < bound for n in lengths)
        assert all(n >= field_arrival(state, lay)[fruit] for n in lengths)
        assert all(certified(lay, after_body(flat_body(state), p)) and timely(state, p) for p in paths)
        found += len(paths)
        greedy = greedy_walk(lay, requirements(lay, flat_body(state)), flat_body(state)[0], fruit)
        assert greedy is None or certified(lay, after_body(flat_body(state), greedy))
    assert found > 50


def test_a_transition_lengthens_the_certifiable_run_or_certifies():
    rng = random.Random(3)
    exact = partial = 0
    for seed in range(6):
        game = SnakeGame(GameConfig(8, 8), seed=seed)
        while not game.done and game.length < 30:
            state = game.state()
            body = flat_body(state)
            certain, toward = transition_paths(state, rng, trees=16, count=4)
            for path in certain + toward:
                assert timely(state, path)
                after = after_body(body, path)
                lay = max((layout(8, 8, o) for o in (0, 1)), key=lambda lay: prefix(lay, after))
                kept = prefix(lay, after)
                assert certified(lay, after[:kept])
                assert (kept == len(after)) == (path in certain)
            exact += len(certain)
            partial += len(toward)
            # Play on at random, so that the bodies come from free play.
            moves = [d for d in Direction if not SnakeGame.from_state(state, seed=0).step(d).lost]
            if not moves:
                break
            game.step(rng.choice(moves))
    assert exact > 0 and partial > 0


def free_play_states(side, games, longest, rng):
    """States of games that mostly chase the fruit and sometimes turn at random, without colliding."""
    for seed in range(games):
        game = SnakeGame(GameConfig(side, side), seed=seed)
        greedy = GreedyPolicy()
        while not game.done and game.length < longest:
            state = game.state()
            yield state
            safe = game.safe_directions()
            if not safe:
                break
            move = greedy(state)
            game.step(move if move in safe and rng.random() < 0.7 else rng.choice(safe))


def test_the_recent_run_is_the_longer_certifiable_prefix_and_lengthens():
    rng = random.Random(4)
    grew = 0
    for state in free_play_states(8, 4, 30, rng):
        lay, run = recent_run(state)
        body = flat_body(state)
        assert 1 <= run <= prefix(lay, body) and requirements(lay, body[:run]) is not None
        other = layout(8, 8, 1 - lay.orientation)
        assert requirements(other, body[: run + 1]) is None or run == len(body)
        for wait in waits_of(state):
            if extends(lay, run, wait.state):
                after = flat_body(wait.state)
                assert prefix(lay, after) > run and certified(lay, after[: run + 1])
                grew += 1
    assert grew > 50


def waits_of(state):
    return [wait for wait, _ in _waits(state, _View(state), TRAINING_RULES)]


def test_a_stalling_transition_waits_only_with_moves_that_lengthen_the_run():
    rng = random.Random(5)
    config = PlanConfig(certify=0.0, stall=True, paths=4, trees=8, shortcut_budget=2000)
    stalled = 0
    for state in free_play_states(8, 30, 40, rng):
        if state.fruit is None or certify(state) is not None:
            continue
        if TRAINING_RULES.moves_left(state) <= state.num_cells:  # too close to the fruit timeout to wait
            continue
        lay, run = recent_run(state)
        longer = [w for w in waits_of(state) if w.certified and extends(lay, run, w.state)]
        decision = run_decision(constant, state, TRAINING_RULES, config, random.Random(0))
        if decision.plan is None and longer:
            assert decision.wait in [w.action for w in longer]
            stalled += 1
    assert stalled > 10


def test_a_waiting_path_certifies_the_body_without_eating():
    rng = random.Random(6)
    config = PlanConfig(certify=0.0, stall=True, paths=4, trees=8, shortcut_budget=2000)
    played = 0
    for state in free_play_states(8, 12, 40, rng):
        if state.fruit is None or certify(state) is not None:
            continue
        path, goal = stall(state, random.Random(0))
        lay, run = recent_run(state)
        assert goal == state.length - run and len(path) <= goal
        if len(path) < goal or TRAINING_RULES.moves_left(state) - goal <= state.free_cells:
            continue
        decision = run_decision(constant, state, TRAINING_RULES, config, random.Random(0))
        if decision.plan is None or decision.plan.eats:
            continue  # a path to the fruit that certified the body came first
        assert decision.certain and decision.win == 1.0
        game = SnakeGame.from_state(state, seed=0)
        results = steps(game, decision.plan.cells)
        assert not any(r.lost or r.ate_fruit for r in results)
        assert game.state() == decision.plan.after
        assert certify(game.state()) is not None
        played += 1
    assert played > 10


def test_part_of_a_waiting_path_lengthens_the_run_and_keeps_a_survival_certificate():
    rng = random.Random(7)
    config = PlanConfig(
        certify=0.0, stall=True, paths=4, trees=8, shortcut_budget=2000, stall_budget=60, stall_tries=1
    )
    walked = 0
    for state in free_play_states(10, 30, 70, rng):
        if state.fruit is None or certify(state) is not None:
            continue
        decision = run_decision(constant, state, TRAINING_RULES, config, random.Random(0))
        if decision.plan is None or decision.plan.eats or decision.certain:
            continue
        lay, run = recent_run(state)
        cells = decision.plan.cells
        game = SnakeGame.from_state(state, seed=0)
        results = steps(game, cells)
        assert not any(r.lost or r.ate_fruit for r in results)
        after = game.state()
        assert after == decision.plan.after and decision.win is None
        assert recent_run(after) == (lay, run + len(cells))
        assert TRAINING_RULES.moves_left(after) >= state.num_cells
        assert Board(after).has_survival_certificate()  # the snake is not walked into a dead end
        walked += 1
    assert walked > 5


def constant(queries):
    n = len(queries)
    return Prediction(np.zeros((n, 3)), np.full(n, 0.5), np.zeros(n))


@pytest.mark.parametrize("side", [6, 8])
def test_the_plan_agent_finishes_on_the_certificate(side):
    config = PlanConfig(paths=4, trees=16, shortcut_budget=5000)
    for seed in range(3):
        game = SnakeGame(GameConfig(side, side), seed=seed)
        agent = PlanAgent(constant, TRAINING_RULES, config, seed)
        certain_from = None
        while not game.done and not TRAINING_RULES.timed_out(game.state()):
            state = game.state()
            if certain_from is None and state.length >= config.certify * state.num_cells and certify(state):
                certain_from = game.steps
            if certain_from is not None and state.steps_since_fruit == 0:
                assert certify(state) is not None  # once certified, every fruit is eaten certified
            play_move(game, agent(state))
        assert game.won and certain_from is not None


def test_a_certified_decision_asks_the_network_nothing():
    game = SnakeGame(GameConfig(8, 8), seed=0)
    state = game.state()
    assert certify(state) is not None
    generator = decide(state, TRAINING_RULES, PlanConfig(certify=0.0, trees=8, shortcut_budget=2000))
    with pytest.raises(StopIteration) as stop:
        next(generator)
    decision = stop.value.value
    assert decision.certain and decision.win == 1.0 and decision.wait is None
    assert certified(certify(state), after_body(flat_body(state), decision.plan.cells))
    # Below the share of the board that ``certify`` sets, the agent plays free and asks for values.
    queries = next(decide(state, TRAINING_RULES, PlanConfig(certify=0.5)))
    assert queries


def test_timely_rejects_cells_still_held_and_an_early_fruit():
    snake = ((1, 1), (1, 2), (0, 2))
    state = GameState(3, 3, snake, Direction.UP, (2, 0))
    assert timely(state, [1, 2])  # (1, 0), then the fruit (2, 0)
    assert not timely(state, [7, 8, 5, 2])  # the neck (1, 2) is held for two moves
    assert not timely(state, [2, 5])  # the fruit is not last
    assert timely(state, [5, 8, 7, 6, 3, 0, 1, 2])  # round the body as it moves away, then the fruit
    two = GameState(3, 3, ((1, 1), (1, 2)), Direction.UP, (2, 0))
    assert not timely(two, [7, 8, 5, 2])  # into the tail at once: a reversal, which the engine refuses
