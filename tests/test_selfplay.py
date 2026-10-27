import numpy as np
import pytest

from snake_agent.contract import NO_LIMIT, TRAINING_RULES, Outcome
from snake_agent.encoding import moves_scale
from snake_agent.search import Prediction, envelope_certain
from snake_agent.seeds import VALIDATION
from snake_agent.selfplay import Label, free_starts, moves_returns, play, standard_starts


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
