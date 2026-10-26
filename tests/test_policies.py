import pytest

from snake_game import Direction, GameConfig, GameState, SnakeGame
from snake_game.policies import GreedyPolicy, HamiltonianPolicy, RandomPolicy, safe_directions


def play(policy, config, seed, max_steps=200_000):
    game = SnakeGame(config, seed=seed)
    while not game.done and game.steps < max_steps:
        game.step(policy(game.state()))
    return game


@pytest.mark.parametrize("width,height", [(20, 20), (6, 6), (8, 5), (5, 8), (4, 4), (10, 3)])
def test_hamiltonian_policy_always_fills_the_board(width, height):
    game = play(HamiltonianPolicy(), GameConfig(width, height), seed=width * 100 + height)
    assert game.won
    assert game.score == game.config.max_score
    assert game.length == width * height


def test_hamiltonian_policy_needs_an_even_side():
    with pytest.raises(ValueError, match="odd"):
        HamiltonianPolicy()(SnakeGame(GameConfig(5, 5), seed=0).state())


def test_greedy_policy_scores_well_but_is_not_perfect():
    scores = [play(GreedyPolicy(), GameConfig(), seed).score for seed in range(3)]
    assert min(scores) > 40
    assert max(scores) < GameConfig().max_score


def test_random_policy_only_takes_safe_moves():
    policy = RandomPolicy(seed=0)
    for seed in range(5):
        game = SnakeGame(GameConfig(8, 8), seed=seed)
        while not game.done:
            state = game.state()
            move = policy(state)
            if state_safe := safe_directions(state):
                assert move in state_safe
            game.step(move)


def test_safe_directions_match_the_engine():
    game = SnakeGame(GameConfig(7, 7), seed=3)
    policy = RandomPolicy(seed=3)
    while not game.done:
        assert safe_directions(game.state()) == game.safe_directions()
        game.step(policy(game.state()))
    assert safe_directions(game.state()) == []


def test_greedy_policy_does_not_eat_its_way_into_a_trap():
    #  x: 0 1 2 3 4
    #  y0 . . . . .
    #  y1 . o o o .      the fruit is walled in on three sides: eating it now
    #  y2 . o * o .      (straight up) would leave the head with nowhere to go
    #  y3 . t H o .
    #  y4 . . . . .
    snake = ((2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3))
    state = GameState(5, 5, snake, Direction.LEFT, (2, 2))
    assert GreedyPolicy()(state) in (Direction.DOWN, Direction.LEFT)
