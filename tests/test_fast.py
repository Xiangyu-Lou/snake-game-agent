import random

import numpy as np
import pytest

from snake_agent import fast
from snake_agent.contract import TRAINING_RULES
from snake_agent.plans import EVALUATION, _View, candidates
from snake_agent.survival import Board, MoveClass, arrival_times, classify_moves, opening_times
from snake_game import Direction, GameConfig, SnakeGame
from snake_game.policies import GreedyPolicy

CODES = {MoveClass.FATAL: 0, MoveClass.WON: 1, MoveClass.CERTIFIED: 2, MoveClass.UNKNOWN: 3}


def wandering_states(side, games, moves, rng):
    """States of games that mostly chase the fruit and sometimes turn at random."""
    for seed in range(games):
        game = SnakeGame(GameConfig(side, side + seed % 2), seed=seed)
        greedy = GreedyPolicy()
        for _ in range(moves):
            if game.done:
                break
            state = game.state()
            yield state
            safe = game.safe_directions()
            game.step(greedy(state) if rng.random() < 0.7 or not safe else rng.choice(safe))


@pytest.mark.parametrize("side", [4, 7, 12])
def test_the_kernels_agree_with_the_survival_searches(side):
    rng = random.Random(side)
    checked = 0
    for state in wandering_states(side, 4, 400, rng):
        board = Board(state)
        body = fast.flat(state)
        release = fast.release_of(body, state.num_cells)
        assert [board.release(c) if board.release(c) > 0 else 0 for c in range(state.num_cells)] == list(
            release
        )
        nbr = fast.grid(state.width, state.height)
        head, heading = int(body[0]), int(state.direction)
        fruit = -1 if state.fruit is None else state.fruit[1] * state.width + state.fruit[0]
        assert list(fast.arrival(release, nbr, head, heading, -1)) == arrival_times(board)
        assert list(fast.arrival(release, nbr, head, heading, fruit)) == arrival_times(board, fruit)
        assert list(fast.opening(release, nbr, head, heading)) == opening_times(board)
        assert fast.certificate(release, nbr, head, heading) == board.has_survival_certificate()
        loop = board.certificate_loop()
        assert list(fast.certificate_loop(release, body, nbr, head, heading)) == (loop or [])
        if not state.done:
            classes = fast.move_classes(release, nbr, head, heading, fruit, state.length)
            assert list(classes) == [CODES[c] for c in classify_moves(state)]
        checked += 1
    assert checked > 150


def test_compiled_and_plain_kernels_agree():
    if not fast.COMPILED:
        pytest.skip("Numba is not installed: the kernels run plain already")
    rng = random.Random(0)
    for state in wandering_states(8, 2, 150, rng):
        body = fast.flat(state)
        release = fast.release_of(body, state.num_cells)
        nbr = fast.grid(state.width, state.height)
        head, heading = int(body[0]), int(state.direction)
        for kernel in (fast.opening, fast.reach):
            compiled = np.stack(np.broadcast_arrays(kernel(release, nbr, head, heading)))
            plain = np.stack(np.broadcast_arrays(kernel.py_func(release, nbr, head, heading)))
            assert np.array_equal(compiled, plain)
        state_a, state_b = fast.generator(7), fast.generator(7)
        assert [fast.below(state_a, 1000) for _ in range(50)] == [
            fast.below.py_func(state_b, 1000) for _ in range(50)
        ]


def test_the_generator_is_seeded_and_spreads():
    a, b = fast.generator(3), fast.generator(3)
    draws = [fast.uniform(a) for _ in range(2000)]
    assert draws == [fast.uniform(b) for _ in range(2000)]
    assert 0.0 < min(draws) and max(draws) < 1.0 and abs(np.mean(draws) - 0.5) < 0.03
    assert fast.generator(0)[0] != 0  # a zero state would stay zero


@pytest.mark.parametrize("side", [6, 10])
def test_kernel_afterstates_match_the_states_the_plans_reach(side):
    rng = random.Random(side)
    checked = 0
    for state in wandering_states(side, 3, 120, rng):
        if state.done or state.fruit is None:
            continue
        view = _View(state)
        for plan in candidates(state, TRAINING_RULES, EVALUATION, rng):
            release, head, heading = view.after(plan.cells)
            after = plan.after
            assert np.array_equal(release, fast.release_of(fast.flat(after), state.num_cells))
            assert head == after.head[1] * side + after.head[0] and heading == int(after.direction)
            assert view.certified(plan.cells) == (after.won or Board(after).has_survival_certificate())
            checked += 1
    assert checked > 200


def test_shortest_paths_and_extensions_are_safe():
    rng = random.Random(1)
    checked = 0
    for state in wandering_states(9, 3, 200, rng):
        if state.done or state.fruit is None:
            continue
        body = fast.flat(state)
        release = fast.release_of(body, state.num_cells)
        nbr = fast.grid(state.width, state.height)
        head, heading = int(body[0]), int(state.direction)
        fruit = state.fruit[1] * state.width + state.fruit[0]
        arrival = fast.arrival(release, nbr, head, heading, -1)
        if arrival[fruit] <= 0:
            continue
        generator = fast.generator(checked)
        for path in fast.shortest_paths(arrival, nbr, fruit, 4, generator):
            longer = fast.extend(path, release, nbr, state.width, head, heading, fruit, 12, generator)
            for cells in (path, longer):
                game = SnakeGame.from_state(state, seed=0)
                for cell in cells:
                    x, y = game.head
                    step = game.step(Direction(fast.direction(state.width, y * state.width + x, int(cell))))
                    assert not step.lost
                assert game.length == state.length + 1  # ate at the end, not before
            assert len(longer) - len(path) in range(0, 13, 2) and set(path) <= set(longer.tolist())
            checked += 1
    assert checked > 50
