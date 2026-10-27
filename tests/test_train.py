import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from snake_agent.encoding import mirror  # noqa: E402
from snake_agent.network import NetworkConfig, load  # noqa: E402
from snake_agent.search import Prediction  # noqa: E402
from snake_agent.seeds import check_training_seed  # noqa: E402
from snake_agent.selfplay import play, standard_starts  # noqa: E402
from snake_agent.train import Replay, TrainConfig, mirrored, train  # noqa: E402


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
    config = TrainConfig(games=300, generations=400, seed=7)
    seeds = [set(config.seeds(g)) for g in (0, 1, 399)]
    assert all(len(s) == 300 for s in seeds) and not (seeds[0] & seeds[1]) and not (seeds[1] & seeds[2])
    for g in (0, 399):
        for seed in (min(config.seeds(g)), max(config.seeds(g)) + 500_000_000):
            check_training_seed(seed)
    with pytest.raises(ValueError):
        TrainConfig(games=1001)
    with pytest.raises(ValueError):
        TrainConfig(seed=500)


def test_a_short_run_writes_its_log_checkpoints_and_assessment(tmp_path):
    config = TrainConfig(
        width=4,
        height=4,
        generations=2,
        games=8,
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
    assert records[0]["games"]["standard"]["games"] == records[0]["games"]["free"]["games"] == 4
    report = records[1]["assessment"]
    assert report["t1_search"]["summary"]["games"] == 4 and "gate" in report["t1_policy"]
    assert "exact" in report and "uniform" in report["exact"]
    exact = report["t1_search"]["exact"]  # every T1 decision scored against the envelope's optimum
    assert exact["optimum"] == pytest.approx(43.0848, abs=1e-4) and exact["extra_moves"] >= 0.0
    model, meta = load(tmp_path / "model-002.pt")
    assert meta["generation"] == 1 and meta["config"]["width"] == 4
    assert json.loads((tmp_path / "config.json").read_text())["games"] == 8
