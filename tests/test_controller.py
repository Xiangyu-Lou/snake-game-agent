import math

import pytest

from snake_game import UNLIMITED, Direction, GameController, GameMode, GameState, SnakeGame


def open_game():
    """A 20x20 game whose fruit is out of the way of the first moves."""
    state = GameState(20, 20, ((10, 10), (9, 10), (8, 10)), Direction.RIGHT, (0, 0))
    return SnakeGame.from_state(state, seed=0)


def test_step_mode_is_the_default_and_never_moves_by_itself():
    ctrl = GameController(open_game())
    assert ctrl.mode is GameMode.STEP
    assert ctrl.update(10.0) == []
    assert ctrl.game.steps == 0
    assert ctrl.progress == 1.0


def test_step_mode_moves_once_per_input():
    ctrl = GameController(open_game())
    result = ctrl.steer(Direction.UP)
    assert result is not None and ctrl.game.head == (10, 9)
    assert ctrl.steer(Direction.DOWN) is None  # reversing onto the neck is ignored
    assert ctrl.game.steps == 1
    ctrl.steer(Direction.UP)  # same direction: one more cell
    assert ctrl.game.head == (10, 8)
    ctrl.advance()  # without a policy: straight ahead
    assert ctrl.game.head == (10, 7)


def test_auto_mode_moves_at_the_configured_speed():
    ctrl = GameController(open_game(), GameMode.AUTO, speed=10)
    assert ctrl.update(0.05) == [] and ctrl.progress == pytest.approx(0.5)
    assert len(ctrl.update(0.05)) == 1
    assert ctrl.game.head == (11, 10)  # keeps its heading without input
    moves = sum(len(ctrl.update(0.02)) for _ in range(25))
    assert moves == 5
    assert ctrl.running


def test_auto_mode_catches_up_on_at_most_a_short_stall():
    ctrl = GameController(open_game(), GameMode.AUTO, speed=8)
    assert len(ctrl.update(5.0)) == 2  # 0.25 s of catch-up at most


def test_speed_change_keeps_the_position_within_the_move():
    ctrl = GameController(open_game(), GameMode.AUTO, speed=10)
    ctrl.update(0.05)
    ctrl.speed = 20
    assert ctrl.progress == pytest.approx(0.5)
    assert ctrl.interval == pytest.approx(0.05)
    with pytest.raises(ValueError):
        ctrl.speed = 0
    with pytest.raises(ValueError):
        ctrl.speed = 1e9
    ctrl.speed = UNLIMITED
    assert ctrl.interval == 0.0 and ctrl.progress == 1.0


def test_pause_and_mode_switch():
    ctrl = GameController(open_game(), GameMode.AUTO, speed=10)
    ctrl.toggle_pause()
    assert ctrl.paused and not ctrl.running
    assert ctrl.update(1.0) == []
    ctrl.toggle_pause()
    assert len(ctrl.update(0.1)) == 1
    ctrl.mode = "step"
    assert ctrl.mode is GameMode.STEP and not ctrl.paused
    ctrl.toggle_pause()  # meaningless in step mode
    assert not ctrl.paused


def test_auto_mode_buffers_quick_turns():
    ctrl = GameController(open_game(), GameMode.AUTO, speed=10)
    ctrl.steer(Direction.UP)
    ctrl.steer(Direction.UP)  # duplicate: dropped
    ctrl.steer(Direction.LEFT)  # a U-turn in two moves
    ctrl.steer(Direction.RIGHT)  # reversal of the buffered LEFT: dropped
    ctrl.update(0.1)
    assert ctrl.game.direction is Direction.UP
    ctrl.update(0.1)
    assert ctrl.game.direction is Direction.LEFT
    ctrl.update(0.1)  # no more input: keeps going left
    assert ctrl.game.direction is Direction.LEFT
    assert ctrl.game.head == (8, 9)


def test_input_buffer_is_bounded():
    ctrl = GameController(open_game(), GameMode.AUTO, speed=10, input_buffer=2)
    for d in (Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT):
        ctrl.steer(d)
    directions = []
    for _ in range(3):
        ctrl.update(0.1)
        directions.append(ctrl.game.direction)
    assert directions == [Direction.UP, Direction.LEFT, Direction.LEFT]
    with pytest.raises(ValueError):
        GameController(open_game(), input_buffer=0)


def test_policy_drives_both_modes():
    seen = []

    def policy(state):
        seen.append(state.steps)
        return Direction.DOWN

    ctrl = GameController(open_game(), policy=policy)
    ctrl.advance()
    assert ctrl.game.direction is Direction.DOWN and seen == [0]
    ctrl.mode = GameMode.AUTO
    ctrl.speed = 4
    ctrl.update(0.25)
    assert seen == [0, 1]


def test_unlimited_speed_runs_until_the_game_ends():
    ctrl = GameController(open_game(), GameMode.AUTO, speed=math.inf)
    results = ctrl.update(0.0, time_budget=1.0)
    assert results[-1].lost and ctrl.game.done  # runs straight into the wall
    assert len(results) == 10
    assert ctrl.update(1.0) == [] and not ctrl.running


def test_on_step_callback_and_reset():
    results = []
    ctrl = GameController(open_game(), on_step=results.append)
    ctrl.steer(Direction.UP)
    ctrl.advance()
    assert len(results) == 2 and ctrl.game.steps == 2
    ctrl.reset(seed=1)
    assert ctrl.game.steps == 0 and not ctrl.paused
    assert ctrl.steer(Direction.UP) is not None
    ctrl.game.play([Direction.UP] * 20)
    assert ctrl.steer(Direction.LEFT) is None and ctrl.advance() is None  # game over
