"""The Snake game engine: pure Python, deterministic and rendering-free.

Coordinates are ``(x, y)`` with ``x`` the column (``0 .. width-1``, left to right)
and ``y`` the row (``0 .. height-1``, top to bottom), i.e. screen coordinates.
``Direction.UP`` therefore decreases ``y``. NumPy arrays produced by this package
are indexed ``[y, x]`` (row-major), and flat cell indices are ``y * width + x``.

Rules
-----
* The snake starts with ``initial_length`` (default 3) segments, lying
  horizontally in the middle of the board and heading right.
* Every :meth:`SnakeGame.step` moves the head one cell. An action that would
  reverse the snake onto its own neck is ignored: the snake keeps going straight.
* Leaving the board (``DeathCause.WALL``) or running into the body
  (``DeathCause.SELF``) ends the game. The cell the tail is leaving is free, so
  the head may follow the tail closely.
* Eating a fruit grows the snake by one segment and scores one point. A new fruit
  appears on a uniformly random free cell.
* The game is won once the snake covers the whole board.
"""

from __future__ import annotations

import dataclasses
import operator
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from itertools import pairwise
from typing import Any

import numpy as np

from . import _rng

Position = tuple[int, int]

DEFAULT_WIDTH = 20
DEFAULT_HEIGHT = 20
DEFAULT_INITIAL_LENGTH = 3


class Direction(IntEnum):
    """Absolute heading. Values run clockwise, so ``(d + 1) % 4`` turns right."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Position:
        """``(dx, dy)`` of one step in this direction."""
        return DELTAS[self]

    @property
    def opposite(self) -> Direction:
        return _DIRECTIONS[(self + 2) % 4]

    def turn(self, action: RelativeAction | int) -> Direction:
        """Apply a :class:`RelativeAction` to this heading."""
        return _DIRECTIONS[(self + _as_index(action, 3, "relative action") - 1) % 4]

    @classmethod
    def from_delta(cls, dx: int, dy: int) -> Direction:
        """The direction of a unit step ``(dx, dy)``."""
        try:
            return _DIRECTION_FROM_DELTA[(dx, dy)]
        except KeyError:
            raise ValueError(f"({dx}, {dy}) is not a unit step") from None


class RelativeAction(IntEnum):
    """Action relative to the current heading (the 3-action control scheme)."""

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


DELTAS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
_DIRECTIONS: tuple[Direction, ...] = tuple(Direction)
_DIRECTION_FROM_DELTA = {delta: _DIRECTIONS[i] for i, delta in enumerate(DELTAS)}


class Status(str, Enum):
    RUNNING = "running"
    WON = "won"
    LOST = "lost"


class DeathCause(str, Enum):
    WALL = "wall"
    SELF = "self"


class GameOverError(RuntimeError):
    """Raised when :meth:`SnakeGame.step` is called on a finished game."""


def _as_index(value: Any, size: int, what: str) -> int:
    try:
        index = operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {value!r}") from None
    if not 0 <= index < size:
        raise ValueError(f"{what} must be in [0, {size}), got {index}")
    return index


def _as_dimension(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Static parameters of a game."""

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    initial_length: int = DEFAULT_INITIAL_LENGTH

    def __post_init__(self) -> None:
        for name in ("width", "height", "initial_length"):
            object.__setattr__(self, name, _as_dimension(getattr(self, name), name))
        if self.width < 1 or self.height < 1:
            raise ValueError(f"board must be at least 1x1, got {self.width}x{self.height}")
        if not 1 <= self.initial_length <= self.width:
            raise ValueError(f"initial_length must be in [1, width={self.width}], got {self.initial_length}")
        if self.initial_length >= self.num_cells:
            raise ValueError("the initial snake must leave at least one free cell for a fruit")

    @property
    def num_cells(self) -> int:
        return self.width * self.height

    @property
    def max_score(self) -> int:
        """Score of a perfect game: the snake ends up covering the whole board."""
        return self.num_cells - self.initial_length

    def start_snake(self) -> tuple[Position, ...]:
        """Initial body (head first): horizontal, centred, heading right."""
        y = self.height // 2
        head_x = (self.width + self.initial_length) // 2 - 1
        return tuple((head_x - i, y) for i in range(self.initial_length))


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable, exact snapshot of a game: everything an agent can know about it.

    ``snake`` lists the segment positions from head (index 0) to tail.
    ``segment_directions[k]`` is the direction in which segment ``k`` is
    travelling: the current heading for the head, and for every other segment
    the direction pointing to the segment in front of it, so that
    ``snake[k - 1] == snake[k] + segment_directions[k].delta``.
    """

    width: int
    height: int
    snake: tuple[Position, ...]
    direction: Direction
    fruit: Position | None
    score: int = 0
    steps: int = 0
    steps_since_fruit: int = 0
    status: Status = Status.RUNNING
    death_cause: DeathCause | None = None

    @property
    def head(self) -> Position:
        return self.snake[0]

    @property
    def tail(self) -> Position:
        return self.snake[-1]

    @property
    def length(self) -> int:
        return len(self.snake)

    @property
    def num_cells(self) -> int:
        return self.width * self.height

    @property
    def free_cells(self) -> int:
        return self.num_cells - len(self.snake)

    @property
    def done(self) -> bool:
        return self.status is not Status.RUNNING

    @property
    def won(self) -> bool:
        return self.status is Status.WON

    @property
    def segment_directions(self) -> tuple[Direction, ...]:
        snake = self.snake
        directions = [self.direction]
        for k in range(1, len(snake)):
            (x0, y0), (x1, y1) = snake[k], snake[k - 1]
            directions.append(Direction.from_delta(x1 - x0, y1 - y0))
        return tuple(directions)

    def validate(self) -> None:
        """Raise ``ValueError`` unless this is a consistent game state.

        ``fruit`` may be ``None`` in a running game; :meth:`SnakeGame.from_state`
        then places one.
        """
        width, height, snake = self.width, self.height, self.snake
        if width < 1 or height < 1:
            raise ValueError(f"invalid board size {width}x{height}")
        if not snake:
            raise ValueError("the snake needs at least one segment")
        if len(set(snake)) != len(snake):
            raise ValueError("snake segments overlap")
        for x, y in snake:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"segment {(x, y)} is outside the {width}x{height} board")
        for (x0, y0), (x1, y1) in pairwise(snake):
            if abs(x0 - x1) + abs(y0 - y1) != 1:
                raise ValueError(f"segments {(x0, y0)} and {(x1, y1)} are not adjacent")
        if not isinstance(self.direction, Direction):
            raise ValueError(f"direction must be a Direction, got {self.direction!r}")
        if min(self.score, self.steps, self.steps_since_fruit) < 0:
            raise ValueError("score and step counters must be non-negative")
        if (self.status is Status.LOST) != (self.death_cause is not None):
            raise ValueError("death_cause must be set exactly when the game is lost")
        full = len(snake) == width * height
        if full != (self.status is Status.WON):
            raise ValueError("a game is won exactly when the snake covers the whole board")
        if self.fruit is not None:
            if full:
                raise ValueError("a full board has no room for a fruit")
            fx, fy = self.fruit
            if not (0 <= fx < width and 0 <= fy < height):
                raise ValueError(f"fruit {self.fruit} is outside the board")
            if self.fruit in snake:
                raise ValueError(f"fruit {self.fruit} lies on the snake")
        if self.status is Status.RUNNING and len(snake) >= 2:
            dx, dy = self.direction.delta
            if (snake[0][0] + dx, snake[0][1] + dy) == snake[1]:
                raise ValueError("direction points back into the snake's neck")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (see :meth:`from_dict`)."""
        return {
            "width": self.width,
            "height": self.height,
            "snake": [list(p) for p in self.snake],
            "direction": self.direction.name,
            "fruit": None if self.fruit is None else list(self.fruit),
            "score": self.score,
            "steps": self.steps,
            "steps_since_fruit": self.steps_since_fruit,
            "status": self.status.value,
            "death_cause": None if self.death_cause is None else self.death_cause.value,
            "segment_directions": [d.name for d in self.segment_directions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GameState:
        direction = data["direction"]
        cause = data.get("death_cause")
        fruit = data.get("fruit")
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            snake=tuple((int(x), int(y)) for x, y in data["snake"]),
            direction=Direction[direction] if isinstance(direction, str) else Direction(direction),
            fruit=None if fruit is None else (int(fruit[0]), int(fruit[1])),
            score=int(data.get("score", 0)),
            steps=int(data.get("steps", 0)),
            steps_since_fruit=int(data.get("steps_since_fruit", 0)),
            status=Status(data.get("status", Status.RUNNING.value)),
            death_cause=None if cause is None else DeathCause(cause),
        )


@dataclass(frozen=True, slots=True)
class StepResult:
    """What happened during one :meth:`SnakeGame.step`."""

    direction: Direction
    """Direction actually taken (a reversal request is replaced by the heading)."""
    ate_fruit: bool
    status: Status
    death_cause: DeathCause | None = None
    collision: Position | None = None
    """On a fatal move: the cell the head tried to enter (may lie off the board)."""

    @property
    def done(self) -> bool:
        return self.status is not Status.RUNNING

    @property
    def won(self) -> bool:
        return self.status is Status.WON

    @property
    def lost(self) -> bool:
        return self.status is Status.LOST


ActionLike = Direction | int


class SnakeGame:
    """A single game of Snake.

    The game only advances when :meth:`step` is called, which makes it the
    natural fit for agents: one call = one action = one frame. Real-time play is
    layered on top by :class:`~snake_game.controller.GameController`.

    ``SnakeGame(seed=0)`` and ``reset(seed=0)`` are fully deterministic: the
    fruit sequence only depends on the seed and on the moves played. Calling
    :meth:`reset` without a seed moves on to the next episode of the same
    deterministic stream. Games are independent objects without global state,
    so any number of them can run in parallel threads or processes; they are
    also cheap to :meth:`clone` (for tree search) and can be pickled.
    """

    __slots__ = (
        "config",
        "_width",
        "_height",
        "_body",
        "_occupied",
        "_direction",
        "_fruit",
        "_score",
        "_steps",
        "_steps_since_fruit",
        "_status",
        "_death_cause",
        "_key",
        "_fruit_draws",
    )

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        seed: int | None = None,
        **overrides: int,
    ) -> None:
        if config is None:
            config = GameConfig(**overrides)
        elif overrides:
            config = dataclasses.replace(config, **overrides)
        self.config = config
        self._width = config.width
        self._height = config.height
        self._key = _rng.random_key() if seed is None else _rng.seed_to_key(seed)
        self._new_episode()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None) -> None:
        """Start a new game.

        With a ``seed`` the new game is reproducible; without one the game moves
        on to the next episode of its current random stream.
        """
        self._key = _rng.next_key(self._key) if seed is None else _rng.seed_to_key(seed)
        self._new_episode()

    def reseed(self, seed: int | None = None) -> None:
        """Change the random stream for *future* fruits without touching the board."""
        self._key = _rng.random_key() if seed is None else _rng.seed_to_key(seed)
        self._fruit_draws = 0

    def _new_episode(self) -> None:
        body = self.config.start_snake()
        self._body: deque[Position] = deque(body)
        self._occupied = bytearray(self._width * self._height)
        for x, y in body:
            self._occupied[y * self._width + x] = 1
        self._direction = int(Direction.RIGHT)
        self._score = 0
        self._steps = 0
        self._steps_since_fruit = 0
        self._status = Status.RUNNING
        self._death_cause: DeathCause | None = None
        self._fruit_draws = 0
        self._fruit = self._spawn_fruit()

    def _spawn_fruit(self) -> Position | None:
        free = self._width * self._height - len(self._body)
        if free == 0:
            return None
        k = _rng.uniform_index(self._key, self._fruit_draws, free)
        self._fruit_draws += 1
        # The k-th free cell in row-major order (the vectorised engine uses the same rule).
        free_cells = np.flatnonzero(np.frombuffer(self._occupied, dtype=np.uint8) == 0)
        cell = int(free_cells[k])
        return (cell % self._width, cell // self._width)

    # ---------------------------------------------------------------------- moves
    def step(self, action: ActionLike) -> StepResult:
        """Move the snake one cell in the absolute ``action`` direction."""
        if self._status is not Status.RUNNING:
            raise GameOverError("the game is over; call reset() to start a new one")
        d = _as_index(action, 4, "action")
        if d == (self._direction + 2) % 4:
            d = self._direction  # cannot reverse onto the neck: keep going straight
        self._direction = d
        self._steps += 1
        self._steps_since_fruit += 1

        width, height = self._width, self._height
        hx, hy = self._body[0]
        dx, dy = DELTAS[d]
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < width and 0 <= ny < height):
            return self._die(DeathCause.WALL, (nx, ny))
        cell = ny * width + nx
        new_head = (nx, ny)
        # The tail cell is free: the tail moves away during this very step (the head
        # can only grow the snake by landing on the fruit, which is never the tail).
        if self._occupied[cell] and new_head != self._body[-1]:
            return self._die(DeathCause.SELF, new_head)

        ate = new_head == self._fruit
        if ate:
            self._score += 1
            self._steps_since_fruit = 0
        else:
            tx, ty = self._body.pop()
            self._occupied[ty * width + tx] = 0
        self._body.appendleft(new_head)
        self._occupied[cell] = 1
        if ate:
            self._fruit = self._spawn_fruit()
            if self._fruit is None:
                self._status = Status.WON
        return StepResult(_DIRECTIONS[d], ate, self._status)

    def step_relative(self, action: RelativeAction | int) -> StepResult:
        """Move using the 3-action scheme (turn left / go straight / turn right)."""
        return self.step(self.direction.turn(action))

    def _die(self, cause: DeathCause, collision: Position) -> StepResult:
        self._status = Status.LOST
        self._death_cause = cause
        return StepResult(_DIRECTIONS[self._direction], False, Status.LOST, cause, collision)

    # ------------------------------------------------------------------- queries
    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def snake(self) -> tuple[Position, ...]:
        """Segment positions from head to tail."""
        return tuple(self._body)

    @property
    def head(self) -> Position:
        return self._body[0]

    @property
    def tail(self) -> Position:
        return self._body[-1]

    @property
    def length(self) -> int:
        return len(self._body)

    @property
    def direction(self) -> Direction:
        return _DIRECTIONS[self._direction]

    @property
    def fruit(self) -> Position | None:
        return self._fruit

    @property
    def score(self) -> int:
        return self._score

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def steps_since_fruit(self) -> int:
        return self._steps_since_fruit

    @property
    def status(self) -> Status:
        return self._status

    @property
    def death_cause(self) -> DeathCause | None:
        return self._death_cause

    @property
    def done(self) -> bool:
        return self._status is not Status.RUNNING

    @property
    def won(self) -> bool:
        return self._status is Status.WON

    def is_occupied(self, position: Position) -> bool:
        """Whether ``position`` is covered by the snake (off-board cells count as not)."""
        x, y = position
        if not (0 <= x < self._width and 0 <= y < self._height):
            return False
        return bool(self._occupied[y * self._width + x])

    def is_safe(self, direction: ActionLike) -> bool:
        """Whether moving in ``direction`` right now would keep the snake alive.

        Reversals count as unsafe, because the engine turns them into "go straight".
        """
        if self._status is not Status.RUNNING:
            return False
        d = _as_index(direction, 4, "direction")
        if d == (self._direction + 2) % 4:
            return False
        hx, hy = self._body[0]
        dx, dy = DELTAS[d]
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < self._width and 0 <= ny < self._height):
            return False
        return not self._occupied[ny * self._width + nx] or (nx, ny) == self._body[-1]

    def safe_directions(self) -> list[Direction]:
        return [d for d in _DIRECTIONS if self.is_safe(d)]

    def state(self) -> GameState:
        """Exact snapshot of the current game."""
        return GameState(
            width=self._width,
            height=self._height,
            snake=tuple(self._body),
            direction=_DIRECTIONS[self._direction],
            fruit=self._fruit,
            score=self._score,
            steps=self._steps,
            steps_since_fruit=self._steps_since_fruit,
            status=self._status,
            death_cause=self._death_cause,
        )

    # ------------------------------------------------------- copies and states
    def clone(self) -> SnakeGame:
        """Independent deep copy, including the random stream.

        A clone replays the same future fruits as the original; call
        :meth:`reseed` on it to sample different ones.
        """
        other = SnakeGame.__new__(SnakeGame)
        other.config = self.config
        other._width = self._width
        other._height = self._height
        other._body = self._body.copy()
        other._occupied = self._occupied[:]
        other._direction = self._direction
        other._fruit = self._fruit
        other._score = self._score
        other._steps = self._steps
        other._steps_since_fruit = self._steps_since_fruit
        other._status = self._status
        other._death_cause = self._death_cause
        other._key = self._key
        other._fruit_draws = self._fruit_draws
        return other

    __copy__ = clone

    def __deepcopy__(self, memo: dict[int, Any]) -> SnakeGame:
        return self.clone()

    def set_state(self, state: GameState) -> None:
        """Replace the board with ``state`` (validated); the random stream is kept.

        If ``state.fruit`` is ``None`` while free cells remain, a fruit is spawned.
        """
        state.validate()
        if (state.width, state.height) != (self._width, self._height):
            raise ValueError(
                f"state is for a {state.width}x{state.height} board, "
                f"this game is {self._width}x{self._height}"
            )
        self._body = deque(state.snake)
        self._occupied = bytearray(self._width * self._height)
        for x, y in state.snake:
            self._occupied[y * self._width + x] = 1
        self._direction = int(state.direction)
        self._score = state.score
        self._steps = state.steps
        self._steps_since_fruit = state.steps_since_fruit
        self._status = state.status
        self._death_cause = state.death_cause
        self._fruit = state.fruit
        if self._fruit is None and self._status is Status.RUNNING:
            self._fruit = self._spawn_fruit()

    @classmethod
    def from_state(
        cls,
        state: GameState,
        config: GameConfig | None = None,
        *,
        seed: int | None = None,
    ) -> SnakeGame:
        """Create a game positioned at ``state`` (e.g. a curriculum start or a puzzle).

        ``config`` (used by later :meth:`reset` calls) defaults to the state's board
        size with the default initial length, shortened if the board is tiny.
        """
        if config is None:
            length = max(1, min(DEFAULT_INITIAL_LENGTH, state.width, state.width * state.height - 1))
            config = GameConfig(state.width, state.height, length)
        game = cls(config, seed=seed)
        game.set_state(state)
        return game

    def play(self, actions: Iterable[ActionLike]) -> list[StepResult]:
        """Apply several absolute actions, stopping early if the game ends."""
        results = []
        for action in actions:
            results.append(self.step(action))
            if self._status is not Status.RUNNING:
                break
        return results

    def __repr__(self) -> str:
        return (
            f"SnakeGame({self._width}x{self._height}, length={len(self._body)}, "
            f"score={self._score}, steps={self._steps}, status={self._status.value})"
        )

    def __str__(self) -> str:
        """ASCII picture of the board (``H`` head, ``o`` body, ``*`` fruit)."""
        rows = [["." for _ in range(self._width)] for _ in range(self._height)]
        for x, y in self._body:
            rows[y][x] = "o"
        hx, hy = self._body[0]
        rows[hy][hx] = "H"
        if self._fruit is not None:
            fx, fy = self._fruit
            rows[fy][fx] = "*"
        return "\n".join("".join(row) for row in rows)
