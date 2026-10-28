import dataclasses
import json
from collections import Counter, deque

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from snake_agent.encoding import mirror  # noqa: E402
from snake_agent.network import NetworkConfig, load  # noqa: E402
from snake_agent.search import Prediction  # noqa: E402
from snake_agent.seeds import check_training_seed  # noqa: E402
from snake_agent.selfplay import play, standard_starts  # noqa: E402
from snake_agent.train import Replay, TrainConfig, _starts, mirrored, train  # noqa: E402


def uniform(queries):
    n = len(queries)
    return Prediction(np.zeros((n, 3)), np.full(n, 0.5), np.ones(n))


@pytest.fixture(scope="module")
def examples():
    return play(uniform, standard_starts(4, 4, range(3)), seed=0)[1]


def test_the_replay_keeps_the_most_recent_examples(examples):
    replay = Replay(100)
    replay.add(examples.take(slice(0, 60)))
    assert replay.size == 60
    replay.add(examples.take(slice(60, 130)))
    assert replay.size == 100 and replay.cursor == 30
    kept = set(map(bytes, replay.data["planes"].reshape(100, -1)))
    newest = set(map(bytes, examples.planes[30:130].reshape(100, -1)))
    assert kept == newest
    sample = replay.sample(32, np.random.default_rng(0))
    assert len(sample) == 32 and sample.planes.shape[1:] == examples.planes.shape[1:]
    big = Replay(10)
    big.add(examples)
    assert big.size == 10 and np.array_equal(big.data["win"], examples.win[-10:])


def test_mirrored_flips_only_the_chosen_examples(examples):
    batch = examples.take(slice(0, 20))
    flip = np.arange(20) % 2 == 0
    flipped = mirrored(batch, flip)
    planes, actions, head, policy = mirror(
        batch.planes[flip], batch.actions[flip], batch.head[flip], batch.width[flip], batch.policy[flip]
    )
    assert np.array_equal(flipped.planes[flip], planes) and np.array_equal(flipped.head[flip], head)
    assert np.array_equal(flipped.policy[flip], batch.policy[flip][:, ::-1])
    assert np.array_equal(flipped.valid[flip], batch.valid[flip][:, ::-1])
    assert np.array_equal(flipped.planes[~flip], batch.planes[~flip])
    assert np.array_equal(flipped.win, batch.win) and mirrored(batch, np.zeros(20, bool)) is batch


def test_each_generation_has_its_own_training_seeds():
    config = TrainConfig(boards=((6, 6, 100), (10, 10, 200)), generations=400, seed=7)
    seeds = [set(config.seeds(g)) for g in (0, 1, 399)]
    assert all(len(s) == 300 for s in seeds) and not (seeds[0] & seeds[1]) and not (seeds[1] & seeds[2])
    for g in (0, 399):
        for seed in (min(config.seeds(g)), max(config.seeds(g)) + 500_000_000):
            check_training_seed(seed)
    with pytest.raises(ValueError):
        TrainConfig(boards=((6, 6, 600), (8, 8, 401)))
    with pytest.raises(ValueError):
        TrainConfig(seed=500)
    with pytest.raises(ValueError):
        TrainConfig(free_share=0.5, recovery_share=0.6)
    assert config.games == 300 and config.canvas == 10 and config.eval_board == (10, 10)


def test_the_learning_rate_warms_up_then_follows_a_cosine_to_its_floor():
    config = TrainConfig(generations=11, lr=1e-3, lr_floor=0.1, warmup=2)
    rates = [config.learning_rate(g) for g in range(11)]
    assert rates[0] == pytest.approx(0.5e-3) and rates[1] == pytest.approx(
        1e-3 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi / 10)))
    )
    assert rates[5] == pytest.approx(0.55e-3) and rates[10] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(rates[1:-1], rates[2:], strict=True))
    assert TrainConfig().learning_rate(7) == pytest.approx(1e-3)  # constant by default


def test_the_starts_follow_the_shares_board_by_board():
    config = TrainConfig(
        boards=((4, 4, 10), (6, 6, 20)),
        free_share=0.2,
        prefix_share=0.1,
        recovery_share=0.2,
        midgame_share=0.3,
    )
    state = standard_starts(6, 6, [0])[0].state
    specs = _starts(config, 3, {(6, 6): deque([state])})
    assert [s[3] for s in specs] == list(config.seeds(3))
    counts = Counter((s[1], s[0]) for s in specs)
    assert counts == {
        (4, "free"): 2,
        (4, "prefix"): 1,
        (4, "recovery"): 2,
        (4, "standard"): 5,  # no mid-game states on 4x4
        (6, "free"): 4,
        (6, "prefix"): 2,
        (6, "recovery"): 4,
        (6, "midgame"): 6,
        (6, "standard"): 4,
    }
    assert all(s[4] is state for s in specs if s[0] == "midgame")
    assert all(s[4] == config.prefix_fill for s in specs if s[0] == "prefix")


def test_a_short_run_writes_its_log_checkpoints_and_assessment(tmp_path):
    config = TrainConfig(
        boards=((4, 4, 8),),
        generations=2,
        steps=3,
        batch=32,
        replay=2000,
        network=NetworkConfig(channels=8, blocks=1, hidden=16),
        workers=2,
        eval_every=2,
        eval_games=4,
        eval_sources=3,
    )
    lines = []
    train(config, tmp_path, log=lines.append)
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["generation"] for r in records] == [0, 1] and len(lines) == 2
    assert records[0]["games"]["4x4"]["standard"]["games"] == records[0]["games"]["4x4"]["free"]["games"] == 4
    assert records[0]["throughput"]["leaves_per_second"] > 0 and records[1]["lr"] == config.lr
    report = records[1]["assessment"]
    assert report["t1_search"]["summary"]["games"] == 4 and "gate" in report["t1_policy"]
    assert "exact" in report and "uniform" in report["exact"]
    exact = report["t1_search"]["exact"]  # every T1 decision scored against the envelope's optimum
    assert exact["optimum"] == pytest.approx(43.0848, abs=1e-4) and exact["extra_moves"] >= 0.0
    model, meta = load(tmp_path / "model-002.pt")
    assert meta["generation"] == 1 and [list(b) for b in meta["config"]["boards"]] == [[4, 4, 8]]
    assert json.loads((tmp_path / "config.json").read_text())["boards"] == [[4, 4, 8]]
    # A run can start from a checkpoint's weights.
    again = dataclasses.replace(config, generations=1, eval_every=5, seed=1)
    train(again, tmp_path / "again", log=lines.append, init=tmp_path / "model-002.pt")
    assert json.loads((tmp_path / "again" / "config.json").read_text())["init"].endswith("model-002.pt")
    with pytest.raises(ValueError, match="network"):
        train(
            dataclasses.replace(again, network=NetworkConfig(4, 1, 8)),
            tmp_path / "x",
            init=tmp_path / "model-002.pt",
        )
