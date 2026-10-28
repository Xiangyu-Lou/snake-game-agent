import json
import math
import zlib

import numpy as np
import pytest

from snake_agent.contract import TRAINING_RULES, planning_game, to_relative
from snake_agent.cycles import CycleEnvelope
from snake_agent.encoding import canonical_envelope
from snake_agent.evaluation import run_game
from snake_agent.habits import (
    BANDS,
    FORCED,
    Comparison,
    Habit,
    HabitEvaluator,
    Window,
    band_entries,
    compare,
    distillation_examples,
    improve,
    planning_seed,
    play_windows,
)
from snake_agent.search import PolicyAgent, Prediction, Query, envelope_certain
from snake_agent.seeds import check_training_seed
from snake_agent.selfplay import Label, free_starts, prefix_starts, standard_starts
from snake_game import Direction, GameConfig, GameState, SnakeGame

INF = math.inf


def hashed(queries):
    """An evaluator whose output for a query depends on nothing else, so batching cannot change it."""
    n = len(queries)
    logits, win, moves = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    for i, q in enumerate(queries):
        key = repr((q.state.snake, q.state.direction, q.state.fruit, q.afterstate)).encode()
        rng = np.random.default_rng(zlib.crc32(key))
        logits[i], win[i], moves[i] = rng.normal(size=3), rng.uniform(0.2, 0.9), rng.uniform(0.5, 1.5)
    return Prediction(logits, win, moves)


def habit(**ranges):
    """A habit from keyword ranges such as ``b0_9=INF`` (bands 0 to 9 inclusive)."""
    biases = [0.0] * BANDS
    for name, bias in ranges.items():
        low, high = (int(x) for x in name[1:].split("_"))
        for k in range(low, high + 1):
            biases[k] = bias
    return Habit(tuple(biases))


def a_choice(width=10, height=10, seed=5):
    """A standard-start game's first state with two envelope moves, and those moves."""
    envelope = canonical_envelope(width, height)
    game = SnakeGame(GameConfig(width, height), seed=seed)
    while True:
        state = game.state()
        jumps = envelope.moves(state)
        if len(jumps) > 1:
            return state, jumps
        game.step(next(iter(jumps)))


def test_bands_cover_the_fill_and_biases_round_trip_through_json():
    assert Habit.band(0, 100) == 0 and Habit.band(4, 100) == 0 and Habit.band(5, 100) == 1
    assert Habit.band(99, 100) == BANDS - 1 and Habit.band(100, 100) == BANDS - 1
    h = habit(b0_8=INF, b9_9=0.15, b19_19=-INF)
    assert not h.neutral and Habit().neutral
    assert Habit.from_json(json.loads(json.dumps(h.to_json()))) == h
    assert str(h) == "0-45%: +inf, 45-50%: +0.15, 50-95%: 0, 95-100%: -inf"
    assert h.replace(9, 0.0).biases[9] == 0.0 and h.biases[9] == 0.15
    with pytest.raises(ValueError):
        Habit((0.0,) * (BANDS - 1))
    with pytest.raises(ValueError):
        Habit((math.nan,) + (0.0,) * (BANDS - 1))


def test_the_bias_favours_longer_jumps_in_its_band():
    state, jumps = a_choice()
    longest = to_relative(state.direction, max(jumps, key=jumps.get))
    shortest = to_relative(state.direction, min(jumps, key=jumps.get))
    band = Habit.band(state.length, state.num_cells)
    logits = np.zeros(3)
    logits[shortest], logits[longest] = 1.0, 0.0  # the raw policy prefers the shorter move
    spread = max(jumps.values()) - min(jumps.values())
    assert Habit().choose(state, jumps, logits) == shortest
    assert Habit().replace(band, INF).choose(state, jumps) == longest  # no logits needed
    assert Habit().replace(band, -INF).choose(state, jumps) == shortest
    assert Habit().replace(band, 1.01 / spread).choose(state, jumps, logits) == longest
    assert Habit().replace(band, 0.99 / spread).choose(state, jumps, logits) == shortest
    assert Habit().replace(band + 1, INF).choose(state, jumps, logits) == shortest  # another band
    assert Habit().replace(band, INF).forced(state, jumps) and not Habit().forced(state, jumps)
    with pytest.raises(ValueError):
        Habit().choose(state, jumps)


def test_targets_are_distributions_over_the_envelope_moves():
    state, jumps = a_choice()
    band = Habit.band(state.length, state.num_cells)
    inside = [to_relative(state.direction, d) for d in jumps]
    longest = to_relative(state.direction, max(jumps, key=jumps.get))
    logits = np.array([0.3, -0.2, 0.5])
    for h in (Habit(), Habit().replace(band, 0.15), Habit().replace(band, INF)):
        target = h.target(state, jumps, logits, smoothing=0.1)
        assert target.sum() == pytest.approx(1.0) and all(target[a] == 0 for a in range(3) if a not in inside)
    raw = Habit().target(state, jumps, logits)
    z = np.exp(logits[inside] - logits[inside].max())
    assert raw[inside] == pytest.approx(z / z.sum())
    assert Habit().replace(band, INF).target(state, jumps, logits, smoothing=0.1)[longest] == pytest.approx(
        0.9
    )
    assert Habit().replace(band, 0.15).target(state, jumps, logits)[longest] > raw[longest]


def test_the_evaluator_turns_the_policy_agent_into_the_habit():
    state, jumps = a_choice()
    band = Habit.band(state.length, state.num_cells)
    longest = to_relative(state.direction, max(jumps, key=jumps.get))
    queries = [Query(state, TRAINING_RULES), Query(state, TRAINING_RULES, afterstate=True)]
    raw = hashed(queries)
    assert (HabitEvaluator(hashed, Habit())(queries).logits == raw.logits).all()
    forced = HabitEvaluator(hashed, Habit().replace(band, INF))(queries)
    inside = [to_relative(state.direction, d) for d in jumps]
    assert forced.logits[0][longest] == pytest.approx(max(raw.logits[0][inside]) + FORCED)
    assert (forced.logits[1] == raw.logits[1]).all()  # afterstates and values are untouched
    assert (forced.win == raw.win).all() and (forced.moves == raw.moves).all()
    assert PolicyAgent(HabitEvaluator(hashed, Habit().replace(band, INF)))(state) == longest
    # Outside the envelope nothing changes.
    free = GameState(10, 10, ((5, 7), (5, 6), (5, 5)), Direction.DOWN, (0, 0))  # against the cycle
    assert not envelope_certain(free, TRAINING_RULES)
    q = [Query(free, TRAINING_RULES)]
    assert (HabitEvaluator(hashed, habit(b0_19=INF))(q).logits == hashed(q).logits).all()


def full_game_moves(state, seed, h):
    """Moves of the habit's agent from ``state`` to the end, fruits from ``seed`` (no merging)."""
    game = planning_game(state, seed)
    agent = PolicyAgent(HabitEvaluator(hashed, h))
    record = run_game(game, agent, envelope=canonical_envelope(state.width, state.height))
    assert record.outcome.value == "won" and record.envelope_exit is None
    return record.moves


def test_windows_stop_at_the_merge_with_the_exact_difference_of_full_games():
    entries = band_entries(hashed, Habit(), 6, 6, [11, 12])
    length = SnakeGame(GameConfig(6, 6), seed=0).length
    start = Habit.band(length, 36)  # the band of the standard start
    assert all(min(e) == start and e[start].length == length and e[start].steps == 0 for e in entries)
    assert all(
        s.steps_since_fruit == 0 and Habit.band(s.length, 36) == k for e in entries for k, s in e.items()
    )
    windows = []
    for seed, found in zip([11, 12], entries, strict=True):
        for band in (2, 5, 9, 14):
            if band in found:
                candidates = tuple(Habit().replace(band, b) for b in (-INF, -0.15, 0.0, 0.15, INF))
                windows.append(Window(found[band], planning_seed(seed, band), band, Habit(), candidates))
    results = play_windows(hashed, windows)
    merged = 0
    for window, row in zip(windows, results, strict=True):
        base = full_game_moves(window.state, window.seed, window.base)
        for candidate, result in zip(window.candidates, row, strict=True):
            assert result.delta == full_game_moves(window.state, window.seed, candidate) - base
            merged += result.merged is not None
            if candidate == window.base:
                assert result.delta == 0 and result.merged == window.state.length + 1
    assert merged > len(windows) * 3  # most branches merge before the end


def test_planning_seeds_are_training_seeds_of_their_own():
    seeds = {planning_seed(s, b) for s in range(100) for b in range(BANDS)}
    assert len(seeds) == 100 * BANDS and all(check_training_seed(s) == s for s in seeds)
    with pytest.raises(ValueError):
        planning_seed(1_200_000_000, 0)


def test_improvements_need_enough_standard_errors():
    comparison = compare(hashed, Habit(), 6, 6, range(20, 26), candidates=(-INF, 0.15, INF))
    assert comparison.windows > 0
    counts = {key: entry[0] for key, entry in comparison.stats.items()}
    assert set(b for _, b in counts) == {-INF, 0.15, INF} and max(counts.values()) <= 6
    report = comparison.to_dict()
    assert report["habit"] == Habit().to_json() and "5-10%" in report["bands"]  # 6x6 starts in band 1
    fake = Comparison(Habit())
    for delta in (-10, -12, -11, -9):  # a clear gain in band 3
        fake.stats.setdefault((3, INF), [0, 0.0, 0.0, 0])
        entry = fake.stats[(3, INF)]
        entry[0], entry[1], entry[2], entry[3] = (
            entry[0] + 1,
            entry[1] + delta,
            entry[2] + delta**2,
            entry[3] + 1,
        )
    fake.stats[(4, INF)] = [4, -8.0, 400.0, 4]  # a noisy one in band 4
    improved, changes = improve(Habit(), fake, z=2.0)
    assert improved == Habit().replace(3, INF) and [c["band"] for c in changes] == ["15-20%"]
    with pytest.raises(ValueError):
        improve(Habit().replace(1, INF), fake)
    merged = Comparison(Habit()).merge(fake).merge(fake)
    assert merged.stats[(3, INF)][0] == 8 and merged.mean(3, INF) == fake.mean(3, INF)


def test_distillation_examples_teach_the_habit_and_keep_the_teacher_elsewhere():
    h = habit(b0_9=INF)
    starts = (
        standard_starts(6, 6, range(3)) + prefix_starts(6, 6, range(3, 5)) + free_starts(6, 6, range(5, 9))
    )
    examples, stats = distillation_examples(hashed, h, starts, smoothing=0.1)
    assert stats["games"]["standard"] == 3 and stats["examples"] == len(examples)
    assert stats["envelope_decisions"] > 0 and stats["free_decisions"] > 0 and stats["afterstates"] > 0
    assert len(stats["standard_moves"]) == 3
    envelope = examples.has_policy & (examples.label == Label.ENVELOPE)
    early = envelope & (examples.free >= 19)  # lengths up to 17 of 36: bands 0 to 9
    assert early.any() and np.allclose(examples.policy[early].max(axis=1), 0.9)
    assert (examples.win[envelope] == 1.0).all()
    teacher = examples.has_policy & (examples.label == Label.TEACHER)
    assert teacher.any() and np.allclose(examples.policy[teacher].sum(axis=1), 1.0)
    assert ((examples.policy[teacher] > 0) == examples.valid[teacher]).all()
    assert ((examples.win[teacher] > 0.2) & (examples.win[teacher] < 0.9)).all()  # the teacher's values
    after = ~examples.has_policy
    assert after.sum() == stats["afterstates"] and not examples.valid[after].any()
    # The standard games are the habit's: the same moves as its agent.
    for start, moves in zip(starts[:3], stats["standard_moves"], strict=True):
        game = start.game()
        agent = PolicyAgent(HabitEvaluator(hashed, h))
        assert run_game(game, agent, envelope=CycleEnvelope.for_state(start.state)).moves == moves


def salted(queries):
    """Another evaluator independent of batching, with win and moves values in their own ranges."""
    n = len(queries)
    logits, win, moves = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    for i, q in enumerate(queries):
        key = repr((q.state.snake, q.state.direction, q.state.fruit, q.afterstate, "salt")).encode()
        rng = np.random.default_rng(zlib.crc32(key))
        logits[i], win[i], moves[i] = rng.normal(size=3), rng.uniform(0.05, 0.15), rng.uniform(2.0, 3.0)
    return Prediction(logits, win, moves)


def test_the_anchor_plays_and_labels_outside_the_envelope():
    starts = standard_starts(6, 6, range(2)) + free_starts(6, 6, range(5, 9))
    examples, stats = distillation_examples(hashed, habit(b0_9=INF), starts, anchor=salted)
    teacher = examples.label == Label.TEACHER
    envelope = examples.label == Label.ENVELOPE
    assert teacher.any() and envelope.any()
    assert ((examples.win[teacher] >= 0.05) & (examples.win[teacher] <= 0.15)).all()  # the anchor's values
    assert ((examples.moves[teacher] >= 2.0) & (examples.moves[teacher] <= 3.0)).all()
    assert ((examples.moves[envelope] >= 0.5) & (examples.moves[envelope] <= 1.5)).all()  # the teacher's
    assert stats["free_decisions"] == (teacher & examples.has_policy).sum()  # the standard games stay inside
    # Without an anchor the teacher answers everywhere.
    plain, _ = distillation_examples(hashed, habit(b0_9=INF), starts)
    assert (plain.win[plain.label == Label.TEACHER] >= 0.2).all()
