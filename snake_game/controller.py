"""Game modes: turn-based stepping (the default) and real-time auto-movement.

:class:`~snake_game.engine.SnakeGame` itself only moves when ``step()`` is
called. :class:`GameController` adds the notion of time on top of it:

* ``GameMode.STEP`` (default) - the snake never moves by itself. Every move is
  triggered by an input: :meth:`GameController.steer` moves one cell in the given
  direction (manual play), :meth:`GameController.advance` lets the policy pick
  the next move (stepping through an agent's game one move at a time).
* ``GameMode.AUTO`` - the snake moves forward on its own, ``speed`` cells per
  second, while :meth:`GameController.update` is fed the elapsed time. Steering
  inputs are buffered and applied one per move, so a quick double turn is not
  lost; without input the snake keeps its heading. With a policy attached, the
  policy chooses every move.

The controller has no dependency on any UI toolkit, so it is equally usable from
the pygame app, a terminal front-end or a test.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from enum import Enum

from .engine import ActionLike, Direction, GameState, SnakeGame, StepResult

Policy = Callable[[GameState], ActionLike]
"""A policy maps the current :class:`GameState` to an absolute direction."""

MIN_SPEED = 0.5
MAX_SPEED = 1000.0
UNLIMITED = math.inf
"""Pass as ``speed`` to run auto mode as fast as the CPU allows."""

#: Longest stretch of real time a single ``update`` catches up on (avoids a burst
#: of moves after the window was dragged or the process was suspended).
MAX_FRAME_TIME = 0.25


class GameMode(str, Enum):
    STEP = "step"
    AUTO = "auto"


class GameController:
    """Drives a :class:`SnakeGame` in step mode or auto mode.

    Args:
        game: the game to drive.
        mode: ``GameMode.STEP`` (default) or ``GameMode.AUTO``.
        speed: auto-mode speed in moves per second (``UNLIMITED`` for flat out).
        policy: optional agent choosing the moves; ``None`` means manual control.
        input_buffer: how many manual turns auto mode remembers ahead of time.
        on_step: optional callback invoked with every :class:`StepResult`.
    """

    def __init__(
        self,
        game: SnakeGame,
        mode: GameMode | str = GameMode.STEP,
        speed: float = 8.0,
        policy: Policy | None = None,
        *,
        input_buffer: int = 3,
        on_step: Callable[[StepResult], None] | None = None,
    ) -> None:
        if input_buffer < 1:
            raise ValueError("input_buffer must be at least 1")
        self.game = game
        self._mode = GameMode(mode)
        self._speed = self._validate_speed(speed)
        self.policy = policy
        self.on_step = on_step
        self.paused = False
        self._input_buffer = input_buffer
        self._queue: deque[Direction] = deque()
        self._elapsed = 0.0

    # ----------------------------------------------------------------- settings
    @property
    def mode(self) -> GameMode:
        return self._mode

    @mode.setter
    def mode(self, mode: GameMode | str) -> None:
        self._mode = GameMode(mode)
        self._queue.clear()
        self._elapsed = 0.0
        self.paused = False

    @property
    def speed(self) -> float:
        """Auto-mode speed in moves per second."""
        return self._speed

    @speed.setter
    def speed(self, speed: float) -> None:
        old_interval = self.interval
        self._speed = self._validate_speed(speed)
        # Keep the visual position within the current move continuous.
        if old_interval > 0 and self.interval > 0:
            self._elapsed = min(self._elapsed / old_interval, 1.0) * self.interval
        else:
            self._elapsed = 0.0

    @staticmethod
    def _validate_speed(speed: float) -> float:
        speed = float(speed)
        if math.isinf(speed) and speed > 0:
            return UNLIMITED
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be in [{MIN_SPEED}, {MAX_SPEED}] or UNLIMITED, got {speed}")
        return speed

    @property
    def interval(self) -> float:
        """Seconds between two automatic moves (0 at unlimited speed)."""
        return 0.0 if math.isinf(self._speed) else 1.0 / self._speed

    @property
    def progress(self) -> float:
        """Fraction of the current auto-mode move interval that has elapsed.

        Renderers use it to slide the snake smoothly between cells.
        """
        if self._mode is not GameMode.AUTO or self.interval == 0.0:
            return 1.0
        return min(self._elapsed / self.interval, 1.0)

    @property
    def running(self) -> bool:
        """Whether auto mode is currently moving the snake."""
        return self._mode is GameMode.AUTO and not self.paused and not self.game.done

    def toggle_pause(self) -> None:
        if self._mode is GameMode.AUTO:
            self.paused = not self.paused

    def reset(self, seed: int | None = None) -> None:
        self.game.reset(seed)
        self._queue.clear()
        self._elapsed = 0.0
        self.paused = False

    # ------------------------------------------------------------------- inputs
    def steer(self, direction: ActionLike) -> StepResult | None:
        """Manual steering input.

        Step mode: moves the snake one cell right away and returns the result (a
        request to reverse onto the neck is ignored and returns ``None``).
        Auto mode: buffers the turn for the upcoming moves and returns ``None``.
        """
        if self.game.done:
            return None
        d = Direction(direction)
        if self._mode is GameMode.STEP:
            if d == self.game.direction.opposite:
                return None
            return self._step(d)
        last = self._queue[-1] if self._queue else self.game.direction
        if d != last and d != last.opposite and len(self._queue) < self._input_buffer:
            self._queue.append(d)
        return None

    def advance(self) -> StepResult | None:
        """Perform one move right away, chosen by the policy (or the buffered input).

        This is how step mode advances an agent's game; it works in auto mode too.
        """
        if self.game.done:
            return None
        return self._step(self._next_action())

    def update(self, dt: float, *, time_budget: float = 0.012) -> list[StepResult]:
        """Advance the auto-mode clock by ``dt`` seconds and make the moves now due.

        At ``UNLIMITED`` speed, moves are made until ``time_budget`` seconds of
        wall-clock time have been spent. Does nothing in step mode or while paused.
        """
        if not self.running:
            return []
        results: list[StepResult] = []
        if math.isinf(self._speed):
            deadline = time.perf_counter() + time_budget
            while not self.game.done:
                results.append(self._step(self._next_action()))
                if time.perf_counter() >= deadline:
                    break
            return results
        interval = self.interval
        self._elapsed += min(max(dt, 0.0), MAX_FRAME_TIME)
        while self._elapsed >= interval and not self.game.done:
            self._elapsed -= interval
            results.append(self._step(self._next_action()))
        if self.game.done:
            self._elapsed = 0.0
        return results

    # ----------------------------------------------------------------- internals
    def _next_action(self) -> Direction:
        if self.policy is not None:
            return Direction(self.policy(self.game.state()))
        heading = self.game.direction
        while self._queue:
            d = self._queue.popleft()
            if d != heading and d != heading.opposite:
                return d
        return heading

    def _step(self, direction: Direction) -> StepResult:
        result = self.game.step(direction)
        if self.on_step is not None:
            self.on_step(result)
        return result
