from collections import deque

from snake_agent.references import (
    REFERENCES,
    GuardedGreedyPolicy,
    _fruit_distance,
    _room_after,
)
from snake_agent.survival import Board, MoveClass, classify_moves
from snake_game import DELTAS, GameConfig, SnakeGame
from snake_game.policies import GreedyPolicy, HamiltonianPolicy, RandomPolicy, safe_directions


def fruit_distance(state, start):
    """Shortest path to the fruit with the body minus its tail blocked (the probe's definition)."""
    if start == state.fruit:
        return 0
    blocked, seen, queue = set(state.snake[:-1]), {start}, deque([(start, 0)])
    while queue:
        (x0, y0), dist = queue.popleft()
        for dx, dy in DELTAS:
            cell = (x0 + dx, y0 + dy)
            if 0 <= cell[0] < state.width and 0 <= cell[1] < state.height:
                if cell == state.fruit:
                    return dist + 1
                if cell not in blocked and cell not in seen:
                    seen.add(cell)
                    queue.append((cell, dist + 1))
    return 1 << 30


def states(size, seeds, factory):
    for seed in seeds:
        game, policy = SnakeGame(GameConfig(*size), seed=seed), factory(seed)
        while not game.done and game.steps < 3000:
            state = game.state()
            yield state
            game.step(policy(state))


def test_room_and_fruit_distance_match_the_greedy_policy():
    checked = 0
    for state in states((8, 7), range(6), RandomPolicy):
        board = Board(state)
        snake = [y * 8 + x for x, y in state.snake]
        fruit = state.fruit[1] * 8 + state.fruit[0]
        blocked = bytearray(56)
        for c in snake[:-1]:
            blocked[c] = 1
        for d in safe_directions(state):
            cell = board.neighbours[board.head][d]
            assert _room_after(board, snake, fruit, cell) == GreedyPolicy._room_after(state, d)
            start = (cell % 8, cell // 8)
            assert _fruit_distance(board.neighbours, blocked, cell, fruit) == fruit_distance(state, start)
            checked += 1
    assert checked > 500


def test_guarded_greedy_keeps_a_certificate_whenever_it_can():
    for state in states((8, 8), range(6), lambda seed: GuardedGreedyPolicy()):
        move = GuardedGreedyPolicy()(state)
        safe = safe_directions(state)
        if not safe:
            continue
        assert move in safe
        classes = dict(zip([state.direction.turn(a) for a in range(3)], classify_moves(state), strict=True))
        if any(classes[d] in (MoveClass.CERTIFIED, MoveClass.WON) for d in safe):
            assert classes[move] in (MoveClass.CERTIFIED, MoveClass.WON)


def test_guarded_greedy_rarely_dies():
    outcomes = []
    for seed in range(10):
        game, policy = SnakeGame(GameConfig(8, 8), seed=seed), GuardedGreedyPolicy()
        while not game.done and game.steps < 5000:
            game.step(policy(game.state()))
        outcomes.append("won" if game.won else "lost" if game.done else "running")
    assert outcomes.count("lost") <= 2


def test_reference_registry():
    assert set(REFERENCES) == {"random", "greedy", "guarded-greedy", "hamiltonian", "shortcut"}
    assert not REFERENCES["random"].deterministic
    assert all(spec.deterministic for name, spec in REFERENCES.items() if name != "random")
    hamiltonian = REFERENCES["hamiltonian"]
    policy = hamiltonian.factory(0)
    assert isinstance(policy, HamiltonianPolicy)
    first = hamiltonian.memory(policy)
    policy(SnakeGame(GameConfig(6, 6), seed=0).state())
    assert hamiltonian.memory(policy) != first  # the chosen cycle is part of its memory
