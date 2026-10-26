import pickle

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import snake_game
from snake_game import (
    Direction,
    GameOverError,
    GameState,
    RelativeAction,
    Rewards,
    SnakeEnv,
    SnakeGame,
    VectorSnakeEnv,
)

# 5x5 board, snake heading right in the middle row with the fruit just ahead.
AHEAD = GameState(5, 5, ((2, 2), (1, 2), (0, 2)), Direction.RIGHT, (3, 2))


@pytest.mark.parametrize("obs_type", ["grid", "coords", "features"])
@pytest.mark.parametrize("action_mode", ["absolute", "relative"])
def test_passes_gymnasium_env_checker(obs_type, action_mode):
    env = SnakeEnv(8, 6, obs_type=obs_type, action_mode=action_mode)
    check_env(env, skip_render_check=True)


def test_spaces():
    assert SnakeEnv().action_space == gym.spaces.Discrete(4)
    assert SnakeEnv(action_mode="relative").action_space == gym.spaces.Discrete(3)
    assert SnakeEnv(10, 7).observation_space.shape == (8, 7, 10)
    with pytest.raises(ValueError):
        SnakeEnv(action_mode="diagonal")
    with pytest.raises(ValueError):
        SnakeEnv(obs_type="pixels")
    with pytest.raises(ValueError):
        SnakeEnv(render_mode="ascii")


def test_seeded_reset_reproduces_the_game():
    env = SnakeEnv()
    actions = [0, 3, 3, 2, 2, 1, 1, 1, 0, 0] * 3

    def rollout(seed):
        obs, _ = env.reset(seed=seed)
        trace = [obs]
        for a in actions:
            obs, reward, term, trunc, _ = env.step(a)
            trace.append((obs, reward))
            if term or trunc:
                break
        return trace

    first, second = rollout(11), rollout(11)
    assert len(first) == len(second)
    for a, b in zip(first[1:], second[1:], strict=True):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]
    env.reset(seed=11)
    assert env.state() == SnakeGame(seed=11).state()


def test_info():
    env = SnakeEnv()
    _, info = env.reset(seed=0)
    assert set(info) == {
        "score",
        "length",
        "steps",
        "steps_since_fruit",
        "won",
        "hit_wall",
        "hit_self",
        "action_mask",
    }
    assert info["length"] == 3 and info["score"] == 0 and not info["won"]
    assert info["action_mask"].tolist() == [True, True, True, False]


def test_eating_and_dying_rewards():
    env = SnakeEnv(5, 5)
    env.reset(options={"state": AHEAD})
    _, reward, term, trunc, info = env.step(Direction.RIGHT)
    assert reward == 1.0 and not term and not trunc and info["score"] == 1
    env.reset(options={"state": AHEAD})
    env.step(Direction.UP)
    env.step(Direction.UP)
    _, reward, term, _, info = env.step(Direction.UP)  # (2, -1): wall
    assert reward == -1.0 and term and info["hit_wall"] and not info["hit_self"]
    with pytest.raises(GameOverError):
        env.step(Direction.UP)


def test_custom_rewards_and_win_bonus():
    rewards = Rewards(fruit=2.0, death=-5.0, win=10.0, step=-0.01)
    env = SnakeEnv(2, 2, 2, rewards=rewards)
    state = GameState(2, 2, ((0, 0), (1, 0), (1, 1)), Direction.LEFT, (0, 1))
    env.reset(options={"state": state})
    _, reward, term, trunc, info = env.step(Direction.DOWN)
    assert term and not trunc and info["won"]
    assert not info["hit_wall"] and not info["hit_self"]  # a full board is a win, not a death
    assert reward == pytest.approx(2.0 + 10.0 - 0.01)  # no death penalty


def test_truncation_after_steps_without_fruit():
    far = GameState(5, 5, ((2, 2), (1, 2), (0, 2)), Direction.RIGHT, (4, 4))
    loop = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT] * 3
    env = SnakeEnv(5, 5, max_steps_without_fruit=4)
    env.reset(options={"state": far})
    flags = [env.step(a)[3] for a in loop[:4]]
    assert flags == [False, False, False, True]
    assert SnakeEnv(5, 5).max_steps_without_fruit == 25  # "auto": one step per cell
    env = SnakeEnv(5, 5, max_steps_without_fruit=None)
    env.reset(options={"state": far})
    assert not any(env.step(a)[3] for a in loop)
    with pytest.raises(ValueError):
        SnakeEnv(max_steps_without_fruit=0)
    with pytest.raises(ValueError):
        SnakeEnv(max_steps_without_fruit="never")


def test_relative_actions():
    env = SnakeEnv(5, 5, action_mode="relative")
    env.reset(options={"state": AHEAD})
    env.step(RelativeAction.LEFT)
    assert env.game.direction is Direction.UP
    env.step(RelativeAction.RIGHT)
    assert env.game.direction is Direction.RIGHT
    env.step(RelativeAction.STRAIGHT)
    assert env.game.head == (4, 1)


def test_action_masks():
    #  H o .       head (0,0) heading up: up and left are walls,
    #  o o .       right is the body - every move is fatal
    #  . . .
    trapped = GameState(3, 3, ((0, 0), (0, 1), (1, 1), (1, 0), (2, 0)), Direction.UP, (2, 2))
    env = SnakeEnv(3, 3)
    env.reset(options={"state": trapped})
    assert env.action_masks().tolist() == [True, True, False, True]  # never empty, reversal masked
    env = SnakeEnv(3, 3, action_mode="relative")
    env.reset(options={"state": trapped})
    assert env.action_masks().tolist() == [True, True, True]
    env = SnakeEnv(5, 5, action_mode="relative")
    env.reset(options={"state": AHEAD})
    assert env.action_masks().tolist() == [True, True, True]
    env.reset(options={"state": GameState(5, 5, ((4, 2), (3, 2), (2, 2)), Direction.RIGHT, (0, 0))})
    assert env.action_masks().tolist() == [True, False, True]  # straight ahead is the wall


def test_reset_from_state_accepts_dicts_and_seeds():
    env = SnakeEnv(5, 5)
    obs, info = env.reset(seed=3, options={"state": AHEAD.to_dict()})
    assert env.state() == AHEAD
    np.testing.assert_array_equal(obs, snake_game.observe(AHEAD))
    no_fruit = GameState(5, 5, AHEAD.snake, Direction.RIGHT, None)
    env.reset(seed=3, options={"state": no_fruit})
    first = env.state().fruit
    env.reset(seed=3, options={"state": no_fruit})
    assert env.state().fruit == first is not None


def test_pickling_an_environment():
    env = SnakeEnv(6, 6, obs_type="features")
    env.reset(seed=5)
    env.step(0)
    clone = pickle.loads(pickle.dumps(env))
    for action in (1, 1, 2):
        a, b = env.step(action), clone.step(action)
        np.testing.assert_array_equal(a[0], b[0])


def test_registered_with_gymnasium():
    env = gym.make(snake_game.ENV_ID, width=8, height=8, obs_type="features")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (19,)
    env.step(env.action_space.sample())
    env.close()
    venv = gym.make_vec(snake_game.ENV_ID, num_envs=3, width=8, height=8)
    assert isinstance(venv.unwrapped, VectorSnakeEnv) and venv.num_envs == 3
    venv.reset(seed=0)
    venv.step(np.zeros(3, dtype=np.int64))
    venv.close()
    sync = gym.make_vec(snake_game.ENV_ID, num_envs=2, vectorization_mode="sync", width=8, height=8)
    sync.reset(seed=0)
    sync.step(np.zeros(2, dtype=np.int64))
    sync.close()


def _make_small_env():
    return SnakeEnv(6, 6, obs_type="features")


def test_runs_in_subprocesses_with_async_vector_env():
    envs = gym.vector.AsyncVectorEnv([_make_small_env] * 2, context="spawn")
    try:
        obs, _ = envs.reset(seed=1)
        local = VectorSnakeEnv(2, 6, 6, obs_type="features", autoreset_mode="NextStep")
        expected, _ = local.reset(seed=1)
        np.testing.assert_array_equal(obs, expected)
        for _ in range(20):
            actions = np.array([1, 2])
            a, b = envs.step(actions), local.step(actions)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
    finally:
        envs.close()
