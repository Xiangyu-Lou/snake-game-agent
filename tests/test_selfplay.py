import dataclasses

import numpy as np
import pytest

from snake_agent.contract import NO_LIMIT, TRAINING_RULES, Outcome
from snake_agent.cycles import aligned_cycles
from snake_agent.encoding import canonical_envelope, moves_scale
from snake_agent.recovery import recovery_starts
from snake_agent.search import Prediction, envelope_certain
from snake_agent.seeds import VALIDATION
from snake_agent.selfplay import (
    FRUIT_OFFSET,
    Label,
    free_starts,
    make_start,
    midgame_start,
    moves_returns,
    play,
    prefix_starts,
    recovery_start,
    standard_starts,
)
from snake_agent.survival import Board
from snake_game import GameConfig, SnakeGame


def uniform(queries):
    n = len(queries)
    return Prediction(np.zeros((n, 3)), np.full(n, 0.5), np.ones(n))


def test_standard_games_are_won_inside_the_envelope():
    summaries, examples = play(uniform, standard_starts(4, 4, range(6)), seed=0)
    assert all(s.outcome is Outcome.WON and s.standard for s in summaries)
    assert all(s.length == 16 and s.fruits == 13 for s in summaries)
    assert (examples.label == Label.ENVELOPE).all() and (examples.win == 1.0).all()
    # An example per move, and one per eating move that does not fill the board.
    assert len(examples) == sum(s.moves + s.fruits - 1 for s in summaries)
    assert examples.has_moves.all() and (examples.moves > 0).all()
    assert examples.planes.shape[1:] == (25, 4, 4) and examples.planes.dtype == np.float16


def test_policy_targets_cover_the_moves_the_search_considered():
    _, examples = play(uniform, standard_starts(6, 6, range(2)) + free_starts(6, 6, range(10, 14)), seed=1)
    searched = examples.has_policy
    assert searched.any() and not searched.all()
    policy, valid = examples.policy[searched], examples.valid[searched]
    assert np.allclose(policy.sum(axis=1), 1.0)
    assert ((policy > 0) == valid).all()
    assert (valid.sum(axis=1) >= 2).all()  # forced moves give no policy target
    assert not examples.policy[~searched].any()


def test_moves_returns_blend_moves_and_afterstate_values():
    # Ten moves; fruits eaten at moves 2 and 5 (afterstates 0 and 1), the board filled at move 9.
    decisions, afters = moves_returns(10, [2, 5], [20.0, 8.0], 1.0)
    assert decisions == [10 - t for t in range(10)] and afters == [7, 4]  # lambda 1: the moves still taken
    decisions, afters = moves_returns(10, [2, 5], [20.0, 8.0], 0.0)
    assert afters == [3 + 8.0, 4] and decisions[:3] == [3 + 20.0, 2 + 20.0, 1 + 20.0]
    assert decisions[3:6] == [3 + 8.0, 2 + 8.0, 1 + 8.0] and decisions[6:] == [4, 3, 2, 1]
    decisions, afters = moves_returns(10, [2, 5], [20.0, 8.0], 0.5)
    assert afters[0] == 3 + 0.5 * 8.0 + 0.5 * 4 and decisions[0] == 3 + 0.5 * 20.0 + 0.5 * afters[0]


def test_without_bootstrapping_the_targets_are_the_outcomes():
    starts = standard_starts(4, 4, [3]) + free_starts(4, 4, range(20, 40))
    summaries, examples = play(uniform, starts, seed=2, beta=0.0, moves_lambda=1.0)
    offset = 0
    for summary in summaries:
        moves = summary.moves
        rows = slice(offset, offset + moves)
        free = examples.free[rows].astype(int)
        if summary.outcome is Outcome.WON:
            remaining = examples.moves[rows] * np.array([moves_scale(f) for f in free])
            assert np.allclose(remaining, np.arange(moves, 0, -1), rtol=1e-5)
        win = examples.win[rows]
        labels = examples.label[rows]
        assert (win[labels == Label.ENVELOPE] == 1.0).all()
        assert (win[labels == Label.PROVEN_LOST] == 0.0).all()
        resolved = np.isin(labels, (Label.BLEND, Label.OUTCOME))
        assert (win[resolved] == float(summary.outcome is Outcome.WON)).all()
        afterstates = summary.fruits - (summary.outcome is Outcome.WON)
        offset += moves + afterstates
    assert offset == len(examples)


def test_free_starts_are_outside_the_envelope_and_use_training_seeds():
    starts = free_starts(6, 6, range(30))
    assert len(starts) >= 25
    for start in starts:
        state = start.state
        assert not start.standard and not state.done
        assert not envelope_certain(state, TRAINING_RULES) and not TRAINING_RULES.timed_out(state)
        assert start.seed >= 500_000_000 and start.game().state() == state
    summaries, examples = play(uniform, starts[:8], seed=3)
    assert {Label.BLEND, Label.ENVELOPE} <= set(examples.label.tolist())
    assert all(not s.standard for s in summaries)
    with pytest.raises(ValueError):
        standard_starts(6, 6, [VALIDATION.start])
    with pytest.raises(ValueError):
        free_starts(6, 6, [VALIDATION.start])


def test_self_play_needs_the_fruit_timeout_and_is_reproducible():
    starts = standard_starts(4, 4, range(2)) + free_starts(4, 4, range(5, 8))
    with pytest.raises(ValueError):
        play(uniform, starts, NO_LIMIT)
    first, a = play(uniform, starts, seed=(1, 2))
    second, b = play(uniform, starts, seed=(1, 2))
    assert first == second
    assert np.array_equal(a.planes, b.planes) and np.array_equal(a.policy, b.policy)


def test_a_larger_canvas_holds_smaller_boards():
    _, examples = play(uniform, standard_starts(4, 4, [0]), seed=0, size=6)
    assert examples.planes.shape[2:] == (6, 6)
    assert not examples.planes[:, :, 4:, :].any() and not examples.planes[:, :, :, 4:].any()


def test_prefix_starts_hand_won_games_to_the_agent_after_random_envelope_moves():
    starts = prefix_starts(6, 6, range(8), fill=0.5)
    standard = SnakeGame(GameConfig(6, 6), seed=0).state()
    envelope = canonical_envelope(6, 6)
    for start in starts:
        assert start.standard and start.state.snake == standard.snake and 4 <= start.prefix <= 18
        game = start.game()
        assert game.length == start.prefix and envelope.contains(game.state())
        assert start.game().state() == game.state()  # the prefix is reproducible
    assert len({start.prefix for start in starts}) > 1
    summaries, examples = play(uniform, starts, seed=0)
    for summary, start in zip(summaries, starts, strict=True):
        assert summary.outcome is Outcome.WON and summary.prefix > 0
        assert summary.start_length == start.prefix and summary.fruits == 36 - start.prefix
    # Only the agent's moves give examples.
    assert len(examples) == sum(s.moves + s.fruits - 1 for s in summaries)


def test_sibling_examples_take_the_search_value_of_moves_not_played():
    starts = standard_starts(6, 6, range(3))
    plain_summaries, plain = play(uniform, starts, seed=4)
    summaries, examples = play(uniform, starts, seed=4, sibling_visits=1)
    assert summaries == plain_summaries  # the same games: siblings only add examples
    siblings = examples.label == Label.SIBLING
    assert siblings.sum() == len(examples) - len(plain) > 0
    assert (examples.win[siblings] == 1.0).all() and examples.has_moves[siblings].all()
    assert not examples.has_policy[siblings].any() and (examples.moves[siblings] > 0).all()
    assert np.array_equal(examples.planes[~siblings], plain.planes)


def test_recovery_and_midgame_starts_come_from_training_seeds():
    start = next(s for seed in range(20) if (s := recovery_start(6, 6, seed)) is not None)
    source = start.seed - FRUIT_OFFSET
    assert start.kind == "recovery" and not start.standard
    assert start.state in [s.state for s in recovery_starts(6, 6, [source])]
    assert not aligned_cycles(start.state) and Board(start.state).has_survival_certificate()
    with pytest.raises(ValueError):
        recovery_start(6, 6, VALIDATION.start)
    summaries, _ = play(uniform, [start], seed=0, samples=3)
    summary = summaries[0]
    assert (
        summary.kind == "recovery"
        and summary.board == (6, 6)
        and len(summary.samples) == min(3, summary.moves)
    )
    assert summary.leaves >= summary.searched * 16
    state = summary.samples[-1]
    midgame = make_start("midgame", 6, 6, 7, state)
    assert (
        midgame == midgame_start(state, 7) and midgame.kind == "midgame" and midgame.seed == 7 + FRUIT_OFFSET
    )
    assert midgame.game().state() == state
    with pytest.raises(ValueError):
        midgame_start(state, VALIDATION.start - FRUIT_OFFSET)
    assert make_start("prefix", 6, 6, 3, 0.5) == prefix_starts(6, 6, [3], 0.5)[0]
    assert make_start("standard", 6, 6, 3) == standard_starts(6, 6, [3])[0]
    with pytest.raises(ValueError):
        make_start("other", 6, 6, 3)
    with pytest.raises(ValueError):
        dataclasses.replace(start, kind="other")


def test_starts_of_several_boards_share_one_canvas():
    starts = standard_starts(4, 4, [0]) + free_starts(6, 6, range(10, 12))
    summaries, examples = play(uniform, starts, seed=0, size=6)
    assert {s.board for s in summaries} == {(4, 4), (6, 6)}
    cells = examples.planes[:, 0].astype(float).sum((1, 2))  # the valid plane marks the board's cells
    assert examples.planes.shape[2:] == (6, 6) and set(cells.tolist()) == {16.0, 36.0}
