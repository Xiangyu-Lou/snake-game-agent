import gymnasium
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers.vector import DictInfoToList, RecordEpisodeStatistics

from snake_game import GameConfig, SnakeEnv, SnakeGame, VectorSnakeEnv

BOARD_8 = GameConfig(8, 8)


def row(obs, i):
    return {k: v[i] for k, v in obs.items()} if isinstance(obs, dict) else obs[i]


def assert_obs_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    else:
        np.testing.assert_array_equal(a, b)


def masked_random_actions(venv, rng, p_random=0.03):
    masks = venv.action_masks()
    n_actions = venv.single_action_space.n
    return np.array(
        [rng.integers(n_actions) if rng.random() < p_random else rng.choice(np.flatnonzero(m)) for m in masks]
    )


@pytest.mark.parametrize(
    "obs_type,action_mode,width,height,length,timeout,steps",
    [
        ("grid", "absolute", 20, 20, 3, "auto", 600),
        ("coords", "relative", 20, 20, 3, "auto", 600),
        ("features", "absolute", 12, 9, 3, "auto", 600),
        ("grid", "absolute", 3, 2, 3, 6, 300),  # tiny board: frequent wins and truncations
        ("coords", "absolute", 2, 2, 2, 6, 300),
        ("features", "relative", 4, 3, 3, None, 300),
    ],
)
def test_matches_independent_single_environments(
    obs_type, action_mode, width, height, length, timeout, steps
):
    n = 8
    kwargs = dict(obs_type=obs_type, action_mode=action_mode, max_steps_without_fruit=timeout)
    venv = VectorSnakeEnv(n, width, height, length, autoreset_mode=AutoresetMode.SAME_STEP, **kwargs)
    envs = [SnakeEnv(width, height, length, **kwargs) for _ in range(n)]
    obs, infos = venv.reset(seed=100)
    for i, env in enumerate(envs):
        single_obs, single_info = env.reset(seed=100 + i)
        assert_obs_equal(row(obs, i), single_obs)
        np.testing.assert_array_equal(infos["action_mask"][i], single_info["action_mask"])
    rng = np.random.default_rng(0)
    episodes = 0
    for t in range(steps):
        actions = masked_random_actions(venv, rng)
        obs, rewards, terms, truncs, infos = venv.step(actions)
        for i, env in enumerate(envs):
            single_obs, reward, term, trunc, info = env.step(int(actions[i]))
            assert (reward, term, trunc) == (rewards[i], terms[i], truncs[i]), (t, i)
            if term or trunc:
                episodes += 1
                assert infos["_final_obs"][i] and infos["_final_info"][i]
                assert_obs_equal(infos["final_obs"][i], single_obs)
                assert venv.single_observation_space.contains(infos["final_obs"][i])
                for key, value in info.items():
                    np.testing.assert_array_equal(infos["final_info"][key][i], value, err_msg=key)
                single_obs, info = env.reset()
            else:
                assert "_final_obs" not in infos or not infos["_final_obs"][i]
            assert_obs_equal(row(obs, i), single_obs)
            for key, value in info.items():
                np.testing.assert_array_equal(infos[key][i], value, err_msg=key)
            assert venv.state(i) == env.state()
    assert episodes > 0


@pytest.mark.parametrize("mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP])
def test_matches_gymnasium_sync_vector_env(mode):
    n = 4
    reference = SyncVectorEnv([lambda: SnakeEnv(4, 3, obs_type="features")] * n, autoreset_mode=mode)
    venv = VectorSnakeEnv(n, 4, 3, obs_type="features", autoreset_mode=mode)
    obs_a, _ = reference.reset(seed=3)
    obs_b, _ = venv.reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)
    rng = np.random.default_rng(1)
    for _ in range(150):
        actions = rng.integers(0, 4, size=n)
        a, b = reference.step(actions), venv.step(actions)
        for x, y in zip(a[:4], b[:4], strict=True):
            np.testing.assert_array_equal(x, y)
        info_a, info_b = a[4], b[4]
        for key in ("score", "length", "steps", "won"):
            np.testing.assert_array_equal(info_a[key], info_b[key])
        if mode is AutoresetMode.SAME_STEP and "final_obs" in info_a:
            np.testing.assert_array_equal(info_a["_final_obs"], info_b["_final_obs"])
            for i in np.flatnonzero(info_a["_final_obs"]):
                np.testing.assert_array_equal(info_a["final_obs"][i], info_b["final_obs"][i])
                assert info_a["final_info"]["score"][i] == info_b["final_info"]["score"][i]
    reference.close()


def test_next_step_autoreset_semantics():
    venv = VectorSnakeEnv(2, 3, 3, obs_type="features", autoreset_mode=AutoresetMode.NEXT_STEP)
    venv.reset(seed=0)
    # On a 3-wide board the snake spans the whole row: moving right hits the wall at once.
    _, rewards, terms, _, _ = venv.step(np.array([1, 0]))
    assert terms.tolist() == [True, False] and rewards[0] == -1
    obs, rewards, terms, truncs, infos = venv.step(np.array([2, 3]))  # game 0 restarts; its action is ignored
    assert rewards[0] == 0 and not terms[0] and not truncs[0]
    assert infos["steps"][0] == 0 and infos["length"][0] == 3
    restarted = SnakeEnv(3, 3, obs_type="features")
    restarted.reset(seed=0)
    expected, _ = restarted.reset()  # the next episode of game 0's stream
    np.testing.assert_array_equal(obs[0], expected)


def test_disabled_autoreset_requires_an_explicit_reset():
    venv = VectorSnakeEnv(2, 3, 3, autoreset_mode="Disabled")
    venv.reset(seed=0)
    _, _, terms, _, _ = venv.step(np.array([1, 0]))  # game 0 hits the right wall at once
    assert terms.tolist() == [True, False]
    with pytest.raises(RuntimeError, match="reset_mask"):
        venv.step(np.array([0, 3]))
    mask = np.array([True, False])
    before = venv.state(1)
    obs, infos = venv.reset(options={"reset_mask": mask})
    assert venv.state(0).steps == 0 and venv.state(1) == before
    venv.step(np.array([0, 3]))
    with pytest.raises(ValueError):
        venv.reset(options={"reset_mask": np.array([1, 0])})


def test_seeds_and_reset_streams():
    venv = VectorSnakeEnv(3, 8, 8)
    venv.reset(seed=[5, None, 7])
    assert venv.state(0) == SnakeGame(BOARD_8, seed=5).state()
    assert venv.state(2) == SnakeGame(BOARD_8, seed=7).state()
    venv.reset(seed=10)
    for i in range(3):
        assert venv.state(i) == SnakeGame(BOARD_8, seed=10 + i).state()
    venv.reset()  # every game moves on to its next episode
    for i in range(3):
        game = SnakeGame(BOARD_8, seed=10 + i)
        game.reset()
        assert venv.state(i) == game.state()
    with pytest.raises(ValueError):
        venv.reset(seed=[1, 2])


@pytest.mark.parametrize("obs_type", ["grid", "coords", "features"])
def test_observations_fit_the_batched_space(obs_type):
    venv = VectorSnakeEnv(5, 7, 6, obs_type=obs_type)
    obs, _ = venv.reset(seed=0)
    assert venv.observation_space.contains(obs)
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, *_ = venv.step(masked_random_actions(venv, rng, 0.2))
        assert venv.observation_space.contains(obs)
    assert venv.action_space.shape == (5,)


def test_invalid_actions_and_arguments():
    venv = VectorSnakeEnv(2)
    venv.reset(seed=0)
    with pytest.raises(ValueError):
        venv.step(np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        venv.step(np.array([0, 4]))
    with pytest.raises(TypeError):
        venv.step(np.array([0.0, 1.0]))
    with pytest.raises(IndexError):
        venv.state(2)
    with pytest.raises(ValueError):
        VectorSnakeEnv(0)
    with pytest.raises(ValueError):
        VectorSnakeEnv(2, render_env=2)


def test_returned_arrays_are_not_reused():
    venv = VectorSnakeEnv(4, obs_type="coords")
    obs, infos = venv.reset(seed=0)
    snapshot = {k: v.copy() for k, v in obs.items()}
    scores = infos["score"]
    venv.step(np.ones(4, dtype=np.int64))
    assert_obs_equal(obs, snapshot)
    assert scores.tolist() == [0, 0, 0, 0]


GYMNASIUM_VERSION = tuple(int(part) for part in gymnasium.__version__.split(".")[:2])


@pytest.mark.parametrize(
    "mode",
    [
        AutoresetMode.NEXT_STEP,
        pytest.param(
            AutoresetMode.SAME_STEP,
            marks=pytest.mark.skipif(
                GYMNASIUM_VERSION < (1, 4),
                reason="RecordEpisodeStatistics supports SAME_STEP from Gymnasium 1.4",
            ),
        ),
    ],
)
def test_gymnasium_vector_wrappers(mode):
    venv = RecordEpisodeStatistics(VectorSnakeEnv(4, 4, 3, autoreset_mode=mode))
    venv.reset(seed=0)
    returns, finished = np.zeros(4), []
    rng = np.random.default_rng(0)
    for _ in range(200):
        _, rewards, terms, truncs, infos = venv.step(rng.integers(0, 4, size=4))
        returns += rewards
        for i in np.flatnonzero(terms | truncs):
            assert infos["_episode"][i]
            assert infos["episode"]["r"][i] == pytest.approx(returns[i])
            finished.append(i)
            returns[i] = 0
    assert finished
    listed = DictInfoToList(VectorSnakeEnv(3, 6, 6))
    _, infos = listed.reset(seed=0)
    assert isinstance(infos, list) and infos[0]["length"] == 3
