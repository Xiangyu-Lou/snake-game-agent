import dataclasses
import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from snake_agent.contract import TRAINING_RULES  # noqa: E402
from snake_agent.encoding import ARRIVAL_PLANES, encode, stack  # noqa: E402
from snake_agent.network import (  # noqa: E402
    NetworkConfig,
    NetworkEvaluator,
    SnakeNet,
    from_bytes,
    load,
    save,
    tensors,
    to_bytes,
    widen,
)
from snake_agent.search import Query, run_search  # noqa: E402
from snake_agent.selfplay import free_starts, play, standard_starts  # noqa: E402
from snake_game import GameConfig, SnakeGame  # noqa: E402

SMALL = NetworkConfig(channels=16, blocks=2, hidden=32)


def network(seed=0, config=SMALL):
    torch.manual_seed(seed)
    model = SnakeNet(config)
    # Break the identity start of the residual blocks so that every layer matters in the tests.
    with torch.no_grad():
        for block in model.blocks:
            block.conv2.weight.normal_(0.0, 0.1)
    return model.eval()


def states(width, height, count=12, seed=0):
    game = SnakeGame(GameConfig(width, height), seed=seed)
    rng = np.random.default_rng(seed)
    collected = []
    while not game.done and len(collected) < count:
        collected.append(game.state())
        safe = game.safe_directions()
        game.step(safe[int(rng.integers(len(safe)))] if safe else game.direction)
    return collected


def test_outputs_have_one_row_per_state():
    model = network()
    batch = stack([encode(s, TRAINING_RULES) for s in states(6, 6)])
    logits, win, moves = model(*tensors(batch))
    assert logits.shape == (12, 3) and win.shape == (12,) and moves.shape == (12,)
    assert (moves >= 0).all()


def test_the_network_does_not_see_the_canvas_around_a_board():
    """A 4x4 board gives the same outputs on its own canvas and padded into a 6x6 one."""
    model = network(1)
    board = states(4, 4, seed=3)
    alone = model(*tensors(stack([encode(s, TRAINING_RULES) for s in board])))
    padded = model(*tensors(stack([encode(s, TRAINING_RULES, size=6) for s in board])))
    for a, b in zip(alone, padded, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_the_evaluator_answers_the_search():
    evaluator = NetworkEvaluator(network(2))
    queries = [Query(s, TRAINING_RULES) for s in states(4, 4) + states(6, 6)]
    prediction = evaluator(queries)
    assert prediction.logits.shape == (24, 3) and prediction.logits.dtype == np.float64
    assert ((prediction.win > 0) & (prediction.win < 1)).all()
    result = run_search(evaluator, states(6, 6, seed=4)[-1])
    assert result.action in range(3)


def test_checkpoints_keep_the_weights_and_settings():
    model = network(3, NetworkConfig(channels=8, blocks=1, hidden=16))
    buffer = io.BytesIO()
    save(model, buffer, generation=7)
    buffer.seek(0)
    loaded, meta = load(buffer)
    assert meta == {"generation": 7} and loaded.config == model.config
    batch = tensors(stack([encode(s, TRAINING_RULES) for s in states(6, 6)]))
    for a, b in zip(model(*batch), loaded.eval()(*batch), strict=True):
        assert torch.equal(a, b)
    assert from_bytes(to_bytes(model)).config == model.config


def test_a_few_steps_fit_a_batch_of_self_play():
    from snake_agent.train import losses

    model = network(4)
    starts = standard_starts(4, 4, range(2)) + free_starts(4, 4, range(10, 14))
    _, examples = play(NetworkEvaluator(model), starts, seed=0)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    before = {name: value.item() for name, value in losses(model, examples, "cpu").items()}
    for _ in range(60):
        terms = losses(model, examples, "cpu")
        optimizer.zero_grad()
        sum(terms.values()).backward()
        optimizer.step()
    after = {name: value.item() for name, value in losses(model, examples, "cpu").items()}
    assert after["win"] < before["win"] and after["moves"] < before["moves"]
    assert after["policy"] < before["policy"]


def test_the_evaluator_answers_large_batches_in_chunks():
    model = SnakeNet(NetworkConfig(channels=8, blocks=1, hidden=16))
    states = [s.state for s in standard_starts(6, 6, range(4)) + free_starts(6, 6, range(10, 20))]
    queries = [Query(s, TRAINING_RULES) for s in states]
    whole = NetworkEvaluator(model, max_batch=1000)(queries)
    chunked = NetworkEvaluator(model, max_batch=3)(queries)
    assert len(chunked) == len(queries)
    for name in ("logits", "win", "moves"):
        assert np.allclose(getattr(chunked, name), getattr(whole, name), atol=1e-5)
    NetworkEvaluator(model).release()  # a no-op on the CPU


def test_networks_without_the_arrival_planes_do_not_read_them():
    model = network(1)
    encoded = [encode(s, TRAINING_RULES) for s in states(6, 6, 6)]
    planes, actions, globals_, head = stack(encoded)
    noisy = planes.copy()
    noisy[:, -ARRIVAL_PLANES:] = np.random.default_rng(0).random(noisy[:, -ARRIVAL_PLANES:].shape)
    with torch.no_grad():
        clean = model(*tensors((planes, actions, globals_, head)))
        other = model(*tensors((noisy, actions, globals_, head)))
    for a, b in zip(clean, other, strict=True):
        assert torch.equal(a, b)


def test_widening_a_network_for_the_arrival_planes_keeps_its_outputs():
    model = network(2)
    wide = SnakeNet(dataclasses.replace(SMALL, arrival=True)).eval()
    widen(wide, model.state_dict())
    assert wide.stem.weight.shape[1] == model.stem.weight.shape[1] + ARRIVAL_PLANES
    encoded = [encode(s, TRAINING_RULES, arrival=True) for s in states(6, 6, 6, seed=3)]
    batch = tensors(stack(encoded))
    assert batch[0][:, -ARRIVAL_PLANES:].any()
    with torch.no_grad():
        for a, b in zip(model(*batch), wide(*batch), strict=True):
            assert torch.allclose(a, b, atol=1e-6)
