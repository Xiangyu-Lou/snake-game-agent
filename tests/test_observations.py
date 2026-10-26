import numpy as np
import pytest

from snake_game import FEATURE_NAMES, GRID_CHANNELS, Direction, GameState, SnakeGame, Status, observe
from snake_game.observations import observation_space

#  x: 0 1 2 3 4 5
#  y0 . . . . . .
#  y1 . H o . . .      head (1,1) heading left
#  y2 t o o . . .      tail (0,2)
#  y3 . . . . * .      fruit (4,3)
#  y4 . . . . . .
STATE = GameState(6, 5, ((1, 1), (2, 1), (2, 2), (1, 2), (0, 2)), Direction.LEFT, (4, 3))


def test_grid_channels():
    grid = observe(STATE, "grid")
    assert grid.shape == (len(GRID_CHANNELS), 5, 6) and grid.dtype == np.float32
    head, body, fruit, countdown = grid[0], grid[1], grid[2], grid[3]
    assert np.argwhere(head).tolist() == [[1, 1]]
    assert sorted(np.argwhere(body).tolist()) == [[1, 2], [2, 0], [2, 1], [2, 2]]
    assert np.argwhere(fruit).tolist() == [[3, 4]]
    expected = np.zeros((5, 6))
    for k, (x, y) in enumerate(STATE.snake):
        expected[y, x] = (5 - k) / 30
    np.testing.assert_allclose(countdown, expected, rtol=1e-6)
    heading = {(1, 1): 3, (2, 1): 3, (2, 2): 0, (1, 2): 1, (0, 2): 1}  # LEFT LEFT UP RIGHT RIGHT
    planes = grid[4:]
    assert planes.sum() == 5
    for (x, y), d in heading.items():
        assert planes[d, y, x] == 1


def test_features():
    features = observe(STATE, "features")
    assert features.shape == (len(FEATURE_NAMES),) and features.dtype == np.float32
    values = dict(zip(FEATURE_NAMES, features.tolist(), strict=True))
    assert [values[f"heading_{d}"] for d in ("up", "right", "down", "left")] == [0, 0, 0, 1]
    assert values["head_x"] == pytest.approx(1 / 5) and values["head_y"] == pytest.approx(1 / 4)
    assert values["fruit_dx"] == pytest.approx(3 / 5) and values["fruit_dy"] == pytest.approx(2 / 4)
    assert values["tail_dx"] == pytest.approx(-1 / 5) and values["tail_dy"] == pytest.approx(1 / 4)
    assert values["length"] == pytest.approx(5 / 30)
    assert [values[f"danger_{d}"] for d in ("up", "right", "down", "left")] == [0, 1, 1, 0]
    free = [values[f"free_{d}"] for d in ("up", "right", "down", "left")]
    assert free == pytest.approx([1 / 4, 0, 0, 1 / 5])


def test_tail_does_not_count_as_danger():
    state = GameState(4, 4, ((0, 0), (1, 0), (1, 1), (0, 1)), Direction.LEFT, (3, 3))
    values = dict(zip(FEATURE_NAMES, observe(state, "features").tolist(), strict=True))
    assert values["danger_down"] == 0  # (0, 1) is the tail, which moves away
    assert values["free_down"] == pytest.approx(3 / 3)
    assert values["danger_right"] == 1 and values["danger_up"] == 1 and values["danger_left"] == 1


def test_coords():
    coords = observe(STATE, "coords")
    snake = coords["snake"]
    assert snake.shape == (30, 2) and snake.dtype == np.int32
    assert snake[:5].tolist() == [list(p) for p in STATE.snake]
    assert (snake[5:] == -1).all()
    assert coords["segment_directions"][:6].tolist() == [3, 3, 0, 1, 1, -1]
    assert int(coords["length"]) == 5 and coords["length"].shape == ()
    assert int(coords["direction"]) == 3
    assert coords["fruit"].tolist() == [4, 3]
    won = GameState(2, 1, ((1, 0), (0, 0)), Direction.RIGHT, None, status=Status.WON)
    assert observe(won, "coords")["fruit"].tolist() == [-1, -1]


@pytest.mark.parametrize("obs_type", ["grid", "coords", "features"])
def test_observations_fit_their_space(obs_type):
    space = observation_space(obs_type, 20, 20)
    game = SnakeGame(seed=0)
    rng = np.random.default_rng(0)
    while not game.done:
        assert space.contains(observe(game.state(), obs_type))
        safe = game.safe_directions()
        game.step(rng.choice(safe) if safe else 0)
    assert space.contains(observe(game.state(), obs_type))


def test_unknown_observation_type():
    with pytest.raises(ValueError):
        observe(STATE, "pixels")
    with pytest.raises(ValueError):
        observation_space("pixels", 4, 4)
