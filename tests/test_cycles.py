import random

import pytest

from snake_agent.contract import to_relative
from snake_agent.cycles import CycleEnvelope, ShortcutCyclePolicy, aligned, aligned_cycles, board_cycles
from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame


@pytest.mark.parametrize("side", [4, 6, 8, 10, 20])
def test_standard_starts_have_an_envelope(side):
    state = SnakeGame(GameConfig(side, side), seed=0).state()
    envelope = CycleEnvelope.for_state(state)
    assert envelope is not None and envelope.contains(state)
    assert aligned(envelope.cycle, state)
    assert envelope.cycle == dict(aligned_cycles(state)[0])


def test_boards_without_a_cycle_have_no_envelope():
    assert board_cycles(5, 5) == ()
    assert CycleEnvelope.for_state(SnakeGame(GameConfig(5, 5), seed=0).state()) is None


def test_cycle_order_allows_holes_but_not_going_backwards():
    #   0  1  2  3     positions along the cycle of the standard 4x4 start
    #  15 14 13  4     (the start lies on 10, 11, 12)
    #  10 11 12  5
    #   9  8  7  6
    envelope = CycleEnvelope.for_state(SnakeGame(GameConfig(4, 4), seed=0).state())
    assert [
        envelope.ahead(GameState(4, 4, ((0, 0),), Direction.UP, None), c) for c in ((1, 0), (0, 1), (0, 2))
    ] == [
        1,
        15,
        10,
    ]
    # Tail to head: 15, 0, 1, then a jump over 2..13 to 14. In order, with a hole.
    with_hole = GameState(4, 4, ((1, 1), (1, 0), (0, 0), (0, 1)), Direction.DOWN, (3, 3))
    assert envelope.contains(with_hole)
    assert envelope.moves(with_hole) == {Direction.LEFT: 1}  # the only envelope move: onto the tail
    backwards = GameState(4, 4, ((0, 1), (0, 0), (1, 0), (1, 1)), Direction.DOWN, (3, 3))
    assert not envelope.contains(backwards)
    assert envelope.mask(backwards) == (False, False, False)


@pytest.mark.parametrize("width,height", [(4, 4), (6, 6), (6, 4), (8, 6)])
def test_every_envelope_move_keeps_the_win_certain(width, height):
    rng = random.Random(width * 10 + height)
    for seed in range(15):
        game = SnakeGame(GameConfig(width, height), seed=seed)
        envelope = CycleEnvelope.for_state(game.state())
        since = 0
        while not game.done:
            state = game.state()
            assert envelope.contains(state)
            moves = envelope.moves(state)
            successor = envelope.cycle[state.head]
            assert Direction.from_delta(successor[0] - state.head[0], successor[1] - state.head[1]) in moves
            assert all(game.is_safe(d) for d in moves)
            mask = envelope.mask(state)
            assert {RelativeAction(a) for a in range(3) if mask[a]} == {
                to_relative(state.direction, d) for d in moves
            }
            result = game.step(rng.choice(sorted(moves)))
            since = 0 if result.ate_fruit else since + 1
            assert since < width * height  # each fruit is eaten within N moves
        assert game.won


@pytest.mark.parametrize("side,max_fill", [(6, 0.5), (8, 0.5), (10, 0.5), (6, 1.0), (6, 0.0)])
def test_shortcut_cycle_fills_the_board_inside_the_envelope(side, max_fill):
    for seed in range(5):
        game, policy = SnakeGame(GameConfig(side, side), seed=seed), ShortcutCyclePolicy(max_fill)
        envelope = CycleEnvelope.for_state(game.state())
        while not game.done:
            state = game.state()
            move = policy(state)
            jump = envelope.jump(state, move)
            assert jump is not None
            if state.length >= max_fill * side * side:
                assert jump == 1  # past the cutoff it follows the cycle
            game.step(move)
        assert game.won


def test_shortcut_cycle_beats_the_plain_cycle():
    moves = {}
    for max_fill in (0.0, 0.5):
        total = 0
        for seed in range(20):
            game, policy = SnakeGame(GameConfig(6, 6), seed=seed), ShortcutCyclePolicy(max_fill)
            while not game.done:
                game.step(policy(game.state()))
            total += game.steps
        moves[max_fill] = total / 20
    assert moves[0.5] < 0.8 * moves[0.0]


def test_shortcut_cycle_needs_an_aligned_start():
    # A U-turn in the middle of the board: no serpentine turns there.
    state = GameState(6, 6, ((2, 2), (3, 2), (3, 3), (2, 3)), Direction.LEFT, (0, 0))
    assert aligned_cycles(state) == []
    with pytest.raises(ValueError):
        ShortcutCyclePolicy()(state)
