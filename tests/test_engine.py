import json
import pickle
from collections import Counter

import numpy as np
import pytest

from snake_game import (
    DeathCause,
    Direction,
    GameConfig,
    GameOverError,
    GameState,
    RelativeAction,
    SnakeGame,
    Status,
    _rng,
)


def make_game(snake, direction, fruit, width=6, height=6, **kwargs):
    state = GameState(width, height, tuple(snake), direction, fruit, **kwargs)
    return SnakeGame.from_state(state, seed=0)


# ------------------------------------------------------------------- set-up
def test_default_game():
    game = SnakeGame(seed=0)
    assert (game.width, game.height) == (20, 20)
    assert game.snake == ((10, 10), (9, 10), (8, 10))
    assert game.length == 3
    assert game.direction is Direction.RIGHT
    assert game.score == game.steps == game.steps_since_fruit == 0
    assert game.status is Status.RUNNING and not game.done
    assert game.fruit is not None and game.fruit not in game.snake
    assert game.config.max_score == 397


@pytest.mark.parametrize("width,height,length", [(4, 4, 3), (5, 3, 3), (3, 1, 2), (9, 7, 9), (2, 2, 1)])
def test_start_snake_is_centred_and_fits(width, height, length):
    config = GameConfig(width, height, length)
    snake = config.start_snake()
    assert len(snake) == length
    assert all(0 <= x < width and y == height // 2 for x, y in snake)
    assert [x for x, _ in snake] == list(range(snake[0][0], snake[0][0] - length, -1))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"width": 0}, ValueError),
        ({"height": 0}, ValueError),
        ({"width": 4, "initial_length": 5}, ValueError),
        ({"width": 3, "height": 1, "initial_length": 3}, ValueError),
        ({"initial_length": 0}, ValueError),
        ({"width": 20.0}, TypeError),
        ({"width": True}, TypeError),
    ],
)
def test_config_validation(kwargs, error):
    with pytest.raises(error):
        GameConfig(**kwargs)


def test_config_accepts_numpy_integers():
    config = GameConfig(np.int64(12), np.int32(8), np.int16(4))
    assert (config.width, config.height, config.initial_length) == (12, 8, 4)
    assert type(config.width) is int


def test_overrides_and_config_are_combined():
    game = SnakeGame(GameConfig(10, 10), height=6, seed=1)
    assert (game.width, game.height) == (10, 6)


# ------------------------------------------------------------------- moving
@pytest.mark.parametrize(
    "direction,head",
    [(Direction.UP, (10, 9)), (Direction.RIGHT, (11, 10)), (Direction.DOWN, (10, 11))],
)
def test_moves(direction, head):
    game = make_game([(10, 10), (9, 10), (8, 10)], Direction.RIGHT, (0, 0), width=20, height=20)
    result = game.step(direction)
    assert not result.ate_fruit and result.status is Status.RUNNING
    assert game.head == head
    assert game.snake[1:] == ((10, 10), (9, 10))
    assert game.direction is direction
    assert game.steps == 1 and game.steps_since_fruit == 1


def test_reversal_is_ignored():
    game = make_game([(2, 2), (1, 2), (0, 2)], Direction.RIGHT, (5, 5))
    result = game.step(Direction.LEFT)
    assert result.direction is Direction.RIGHT
    assert game.head == (3, 2)


def test_actions_are_validated():
    game = SnakeGame(seed=0)
    with pytest.raises(ValueError):
        game.step(4)
    with pytest.raises(TypeError):
        game.step("up")
    game.step(np.int64(0))  # NumPy integers are fine
    assert game.direction is Direction.UP


@pytest.mark.parametrize(
    "snake,direction,action,collision",
    [
        ([(0, 2), (1, 2), (2, 2)], Direction.LEFT, Direction.LEFT, (-1, 2)),
        ([(5, 2), (4, 2), (3, 2)], Direction.RIGHT, Direction.RIGHT, (6, 2)),
        ([(2, 0), (2, 1), (2, 2)], Direction.UP, Direction.UP, (2, -1)),
        ([(2, 5), (2, 4), (2, 3)], Direction.DOWN, Direction.DOWN, (2, 6)),
    ],
)
def test_wall_ends_the_game(snake, direction, action, collision):
    game = make_game(snake, direction, (5, 5))
    before = game.snake
    result = game.step(action)
    assert result.lost and result.done and not result.won
    assert result.death_cause is DeathCause.WALL
    assert result.collision == collision
    assert game.status is Status.LOST and game.death_cause is DeathCause.WALL
    assert game.snake == before  # the board stays at the last valid position
    assert game.steps == 1
    with pytest.raises(GameOverError):
        game.step(Direction.UP)


def test_biting_itself_ends_the_game():
    #  .H o .      head (1,1) heading left; moving down bites segment (1,2)
    #  o o o .
    game = make_game([(1, 1), (2, 1), (2, 2), (1, 2), (0, 2)], Direction.LEFT, (5, 5))
    result = game.step(Direction.DOWN)
    assert result.death_cause is DeathCause.SELF
    assert result.collision == (1, 2)
    assert game.status is Status.LOST


def test_head_may_follow_the_tail():
    game = make_game([(0, 0), (1, 0), (1, 1), (0, 1)], Direction.LEFT, (2, 2), width=3, height=3)
    assert game.is_safe(Direction.DOWN)
    result = game.step(Direction.DOWN)  # (0, 1) is the tail, which moves away
    assert result.status is Status.RUNNING
    assert game.snake == ((0, 1), (0, 0), (1, 0), (1, 1))
    for _ in range(8):  # chase the tail around the square forever
        game.step_relative(RelativeAction.LEFT)
    assert game.status is Status.RUNNING and game.length == 4


def test_eating_grows_the_snake_and_scores():
    game = make_game([(2, 2), (1, 2), (0, 2)], Direction.RIGHT, (3, 2))
    result = game.step(Direction.RIGHT)
    assert result.ate_fruit
    assert game.snake == ((3, 2), (2, 2), (1, 2), (0, 2))
    assert game.score == 1 and game.length == 4
    assert game.steps_since_fruit == 0
    assert game.fruit is not None and game.fruit not in game.snake


def test_filling_the_board_wins():
    game = make_game([(0, 0), (1, 0), (1, 1)], Direction.LEFT, (0, 1), width=2, height=2)
    result = game.step(Direction.DOWN)
    assert result.ate_fruit and result.won and result.done
    assert not result.lost and result.death_cause is None and result.collision is None
    assert game.status is Status.WON and game.won and game.death_cause is None
    assert game.fruit is None and game.length == 4
    with pytest.raises(GameOverError):
        game.step(Direction.UP)


def test_fruit_never_lands_on_the_snake():
    rng = np.random.default_rng(0)
    for seed in range(30):
        game = SnakeGame(GameConfig(6, 5), seed=seed)
        while not game.done:
            safe = game.safe_directions()
            game.step(rng.choice(safe) if safe else game.direction)
            if game.fruit is not None:
                assert game.fruit not in game.snake


def test_fruit_is_uniform_over_free_cells():
    counts = Counter(SnakeGame(GameConfig(4, 3), seed=seed).fruit for seed in range(9000))
    free = 4 * 3 - 3
    assert len(counts) == free
    expected = 9000 / free
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    assert chi2 < 30  # 8 degrees of freedom: p < 0.001 above ~26


# -------------------------------------------------------- reproducibility
def play_random(game, moves, seed=0):
    rng = np.random.default_rng(seed)
    fruits = []
    for _ in range(moves):
        if game.done:
            break
        safe = game.safe_directions()
        game.step(rng.choice(safe) if safe else 0)
        fruits.append(game.fruit)
    return fruits


def test_same_seed_same_game():
    a, b = SnakeGame(seed=42), SnakeGame(seed=42)
    assert play_random(a, 300) == play_random(b, 300)
    assert a.state() == b.state()
    c = SnakeGame(seed=43)
    assert play_random(c, 300) != play_random(SnakeGame(seed=42), 300)


def test_reset_with_and_without_seed():
    game = SnakeGame(seed=7)
    first = game.fruit
    game.reset()
    second_episode = game.fruit
    game.reset(seed=7)
    assert game.fruit == first and game.steps == 0
    other = SnakeGame(seed=7)
    other.reset()
    assert other.fruit == second_episode


def test_clone_is_independent_and_replays_the_same_fruits():
    game = SnakeGame(seed=5)
    play_random(game, 20)
    twin = game.clone()
    assert twin.state() == game.state()
    fruits = play_random(twin, 200, seed=1)
    assert game.steps == 20  # the original did not move
    assert play_random(game, 200, seed=1) == fruits
    other = game.clone()
    other.reseed(123)
    assert other.state() == game.state()


def test_pickle_and_copy_roundtrip():
    import copy

    game = SnakeGame(GameConfig(8, 6), seed=9)
    play_random(game, 50)
    for restored in (pickle.loads(pickle.dumps(game)), copy.deepcopy(game), copy.copy(game)):
        assert restored.state() == game.state()
        assert play_random(restored, 100, seed=3) == play_random(game.clone(), 100, seed=3)


def test_rng_python_and_numpy_agree():
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 2**63, size=500, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    counters = rng.integers(0, 10**6, size=500).astype(np.uint64)
    sizes = rng.integers(1, 10**5, size=500)
    vectorised = _rng.uniform_index_np(keys, counters, sizes)
    scalar = [
        _rng.uniform_index(int(k), int(c), int(n)) for k, c, n in zip(keys, counters, sizes, strict=True)
    ]
    assert vectorised.tolist() == scalar
    assert all(0 <= v < n for v, n in zip(scalar, sizes, strict=True))
    assert _rng.next_key_np(keys).tolist() == [_rng.next_key(int(k)) for k in keys]
    with pytest.raises(ValueError):
        _rng.seed_to_key(-1)


# ------------------------------------------------------------------ states
def test_state_snapshot_and_segment_directions():
    game = make_game([(1, 1), (2, 1), (2, 2), (1, 2), (0, 2)], Direction.LEFT, (5, 5))
    state = game.state()
    assert state.head == (1, 1) and state.tail == (0, 2) and state.length == 5
    assert state.segment_directions == (
        Direction.LEFT,  # head: current heading
        Direction.LEFT,  # (2,1) -> (1,1)
        Direction.UP,  # (2,2) -> (2,1)
        Direction.RIGHT,  # (1,2) -> (2,2)
        Direction.RIGHT,  # (0,2) -> (1,2)
    )
    for k in range(1, state.length):
        dx, dy = state.segment_directions[k].delta
        assert (state.snake[k][0] + dx, state.snake[k][1] + dy) == state.snake[k - 1]
    assert state.free_cells == 36 - 5


def test_state_dict_roundtrip_is_json_serialisable():
    game = SnakeGame(seed=2)
    play_random(game, 40)
    data = json.loads(json.dumps(game.state().to_dict()))
    assert GameState.from_dict(data) == game.state()
    assert data["segment_directions"][0] == game.direction.name


@pytest.mark.parametrize(
    "snake,direction,fruit,message",
    [
        ([(1, 1), (1, 1)], Direction.UP, None, "overlap"),
        ([(1, 1), (3, 1)], Direction.LEFT, None, "adjacent"),
        ([(6, 1), (5, 1)], Direction.RIGHT, None, "outside"),
        ([(1, 1), (2, 1)], Direction.LEFT, (2, 1), "lies on the snake"),
        ([(1, 1), (2, 1)], Direction.RIGHT, None, "neck"),
        ([(1, 1), (2, 1)], Direction.LEFT, (9, 9), "outside"),
    ],
)
def test_invalid_states_are_rejected(snake, direction, fruit, message):
    with pytest.raises(ValueError, match=message):
        GameState(6, 6, tuple(snake), direction, fruit).validate()


def test_status_consistency_is_validated():
    with pytest.raises(ValueError, match="death_cause"):
        GameState(6, 6, ((1, 1),), Direction.UP, (3, 3), status=Status.LOST).validate()
    with pytest.raises(ValueError, match="won"):
        GameState(6, 6, ((1, 1),), Direction.UP, (3, 3), status=Status.WON).validate()


def test_from_state_spawns_a_missing_fruit_and_keeps_counters():
    state = GameState(5, 5, ((2, 2), (1, 2)), Direction.RIGHT, None, score=7, steps=99)
    game = SnakeGame.from_state(state, seed=4)
    assert game.fruit is not None and game.fruit not in game.snake
    assert game.score == 7 and game.steps == 99
    assert game.config.width == 5
    with pytest.raises(ValueError):
        SnakeGame(GameConfig(6, 6)).set_state(state)  # wrong board size


def test_queries():
    game = make_game([(1, 1), (2, 1), (2, 2), (1, 2), (0, 2)], Direction.LEFT, (5, 5))
    assert game.safe_directions() == [Direction.UP, Direction.LEFT]
    assert not game.is_safe(Direction.RIGHT)  # reversal
    assert not game.is_safe(Direction.DOWN)  # body
    assert game.is_occupied((2, 2)) and not game.is_occupied((4, 4)) and not game.is_occupied((-1, 0))
    text = str(game)
    assert text.count("H") == 1 and text.count("*") == 1 and text.count("o") == 4
    assert "length=5" in repr(game)


def test_relative_steps_and_turns():
    assert Direction.UP.turn(RelativeAction.RIGHT) is Direction.RIGHT
    assert Direction.UP.turn(RelativeAction.LEFT) is Direction.LEFT
    assert Direction.LEFT.turn(RelativeAction.STRAIGHT) is Direction.LEFT
    assert Direction.DOWN.opposite is Direction.UP
    assert Direction.from_delta(-1, 0) is Direction.LEFT
    with pytest.raises(ValueError):
        Direction.from_delta(1, 1)
    game = make_game([(2, 2), (1, 2), (0, 2)], Direction.RIGHT, (5, 5))
    game.step_relative(RelativeAction.LEFT)
    assert game.direction is Direction.UP and game.head == (2, 1)
    with pytest.raises(ValueError):
        game.step_relative(3)


def test_play_stops_at_game_over():
    game = make_game([(0, 2), (1, 2), (2, 2)], Direction.LEFT, (5, 5))
    results = game.play([Direction.LEFT, Direction.LEFT, Direction.UP])
    assert len(results) == 1 and results[0].lost
