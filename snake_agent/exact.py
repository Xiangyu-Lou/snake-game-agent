"""Exact solutions of small boards: the best win probability and moves of every state.

:func:`solve` enumerates every running state of a board (body, heading and
fruit; under the training rules also the moves left before the fruit timeout)
and computes by backward induction, with fruits uniform over the free cells:

* the highest probability of filling the board, for the state and for each
  relative action (:meth:`ExactSolution.value`, :meth:`~ExactSolution.action_values`);
* among the ways to win with certainty, the fewest expected moves to fill the
  board (:meth:`~ExactSolution.moves`), the second goal of the contract;
* with an ``envelope``, the same restricted to the cycle-order envelope: the
  best a standard-start (T1) agent can do under the contract.

:func:`evaluate_policy` gives a fixed policy's exact win probability and
expected moves from the standard start. Together they provide ground truth
for value calibration, policy regret and optimal moves. 4x4 has 27,960 bodies,
1.4e5 states with a fruit and 3.7e6 with the timeout budget, and solves in
about a second; larger boards grow quickly out of reach.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from snake_game import Direction, GameConfig, GameState, RelativeAction

from .contract import Agent, choose_safely, to_relative
from .cycles import CycleEnvelope
from .survival import neighbour_table

_WIN = -2
"""``grow`` entry of a move that eats the last fruit."""


class _Level:
    """Every body of one length, its free cells, and where each relative action leads.

    Arrays over states are ``(body, j)`` with ``j`` indexing the body's free
    cells, the possible fruit positions: ``free[body, j]`` is the cell and
    ``column[body, cell]`` its index (-1 on the body).
    """

    def __init__(self, width: int, height: int, length: int, keys: list[tuple[int, ...]]) -> None:
        cells = width * height
        self.length = length
        self.keys = keys
        self.index = {key: i for i, key in enumerate(keys)}
        if length == 1:  # a single segment: the key is (cell, heading)
            self.heading = np.array([key[1] for key in keys], dtype=np.int64)
            self.bodies = [key[:1] for key in keys]
        else:
            neighbours = neighbour_table(width, height)
            self.heading = np.array([neighbours[key[1]].index(key[0]) for key in keys], dtype=np.int64)
            self.bodies = keys
        on_body = np.zeros((len(keys), cells), dtype=bool)
        for i, body in enumerate(self.bodies):
            on_body[i, list(body)] = True
        self.free = np.nonzero(~on_body)[1].reshape(len(keys), cells - length)
        self.column = np.full((len(keys), cells), -1, dtype=np.int64)
        np.put_along_axis(self.column, self.free, np.arange(cells - length)[None, :], axis=1)

    def link(self, width: int, height: int, longer: _Level | None) -> None:
        """Successor tables; ``longer`` is the level one segment longer (None if that fills the board)."""
        neighbours = neighbour_table(width, height)
        count = len(self.keys)
        self.target = np.full((count, 3), -1, dtype=np.int64)
        self.succ = np.full((count, 3), -1, dtype=np.int64)
        self.grow = np.full((count, 3), -1, dtype=np.int64)
        for i, body in enumerate(self.bodies):
            head, heading = body[0], int(self.heading[i])
            for a in range(3):
                d = (heading + a - 1) % 4
                cell = neighbours[head][d]
                if cell < 0 or cell in body[:-1]:
                    continue
                self.target[i, a] = cell
                self.succ[i, a] = self.index[(cell, d) if self.length == 1 else (cell,) + body[:-1]]
                if cell != body[-1]:  # the fruit is never on the tail
                    self.grow[i, a] = _WIN if longer is None else longer.index[(cell,) + body]
        # For each state and action: does it eat, and which column the fruit has after a move that does not.
        self.eats = self.target[:, :, None] == self.free[:, None, :]
        succ = np.maximum(self.succ, 0)
        column = self.column[succ[:, :, None], self.free[:, None, :]]
        self.next_column = np.where((self.target[:, :, None] >= 0) & ~self.eats, column, 0)


def _bodies(width: int, height: int) -> dict[int, list[tuple[int, ...]]]:
    """Every head-to-tail body of 2 to ``N - 1`` cells, by length."""
    cells = width * height
    neighbours = neighbour_table(width, height)
    by_length: dict[int, list[tuple[int, ...]]] = {length: [] for length in range(2, cells)}
    path: list[int] = []

    def extend(used: int) -> None:
        if len(path) >= 2:
            by_length[len(path)].append(tuple(path))
        if len(path) == cells - 1:
            return
        for nxt in neighbours[path[-1]]:
            if nxt >= 0 and not used >> nxt & 1:
                path.append(nxt)
                extend(used | 1 << nxt)
                path.pop()

    for start in range(cells):
        path.append(start)
        extend(1 << start)
        path.pop()
    return by_length


def _levels(width: int, height: int, min_length: int) -> dict[int, _Level]:
    cells = width * height
    bodies = _bodies(width, height)
    levels = {}
    for length in range(min_length, cells):
        keys = [(c, d) for c in range(cells) for d in range(4)] if length == 1 else bodies[length]
        levels[length] = _Level(width, height, length, keys)
    for length, level in levels.items():
        level.link(width, height, levels.get(length + 1))
    return levels


def _envelope_mask(level: _Level, position: np.ndarray, cells: int) -> np.ndarray:
    """``(body, action, j)``: whether each action is inside the cycle-order envelope."""
    head = np.array([body[0] for body in level.bodies])
    tail = np.array([body[-1] for body in level.bodies])
    in_order = np.array(
        [int(np.sum(np.diff(position[list(body[::-1])]) % cells)) < cells for body in level.bodies]
    )
    ahead_tail = np.full(len(head), cells) if level.length == 1 else (position[tail] - position[head]) % cells
    target = level.target
    jump = np.where(target >= 0, (position[np.maximum(target, 0)] - position[head][:, None]) % cells, 0)
    ahead_fruit = (position[level.free] - position[head][:, None]) % cells
    return (
        in_order[:, None, None]
        & (jump[:, :, None] >= 1)
        & (jump[:, :, None] <= ahead_tail[:, None, None])
        & (jump[:, :, None] <= ahead_fruit[:, None, :])
    )


def _induct(
    level: _Level,
    eat_value: np.ndarray,
    eat_moves: np.ndarray,
    allowed: np.ndarray | None,
    budget: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Values and certain-win moves of a level with 0, 1, 2, ... moves left before the timeout.

    Stops after ``budget`` moves, or earlier once another move changes nothing:
    the last entry then holds for every larger budget.
    """
    blocked = (level.target < 0)[:, :, None]
    if allowed is not None:
        blocked = blocked | ~allowed
    blocked = np.broadcast_to(blocked, level.eats.shape)
    succ = np.maximum(level.succ, 0)[:, :, None]
    shape = level.free.shape
    values, moves = [np.zeros(shape)], [np.full(shape, np.inf)]
    while budget is None or len(values) <= budget:
        q = np.where(level.eats, eat_value[:, :, None], values[-1][succ, level.next_column])
        cost = 1.0 + np.where(level.eats, eat_moves[:, :, None], moves[-1][succ, level.next_column])
        q[blocked] = 0.0
        cost[q < 1.0] = np.inf  # moves count only among certain wins
        value, move = q.max(axis=1), cost.min(axis=1)
        if np.array_equal(value, values[-1]) and np.array_equal(move, moves[-1]):
            break
        values.append(value)
        moves.append(move)
    return np.stack(values), np.stack(moves)


def _afterstates(value: np.ndarray, moves: np.ndarray, certain_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mean value and moves of each body just after eating, over the new fruit."""
    mean_value = value.mean(axis=1)
    if not certain_only:
        return mean_value, moves.mean(axis=1)
    mean_moves = np.full(len(value), np.inf)
    certain = mean_value == 1.0
    mean_moves[certain] = moves[certain].mean(axis=1)
    return mean_value, mean_moves


def _eating(
    level: _Level, after: tuple[np.ndarray, np.ndarray] | None, win_moves: float
) -> tuple[np.ndarray, np.ndarray]:
    """Value and moves after each eating move: the next level's afterstates, or the win."""
    grow = level.grow
    if after is None:  # eating fills the board
        return np.where(grow == _WIN, 1.0, 0.0), np.where(grow == _WIN, 0.0, win_moves)
    safe = np.maximum(grow, 0)
    return np.where(grow >= 0, after[0][safe], 0.0), np.where(grow >= 0, after[1][safe], win_moves)


@dataclass(frozen=True)
class _Located:
    level: _Level
    body: int
    column: int
    budget: int  # moves left before the fruit timeout (a large number without it)


class ExactSolution:
    """A solved board; see :func:`solve`."""

    def __init__(
        self,
        width: int,
        height: int,
        fruit_timeout: bool,
        envelope: CycleEnvelope | None,
        levels: dict[int, _Level],
        values: dict[int, np.ndarray],
        moves: dict[int, np.ndarray],
        eating: dict[int, tuple[np.ndarray, np.ndarray]],
        allowed: dict[int, np.ndarray] | None,
    ) -> None:
        self.width, self.height = width, height
        self.cells = width * height
        self.fruit_timeout = fruit_timeout
        self.envelope = envelope
        self._levels = levels
        self._values = values
        self._moves = moves
        self._eating = eating
        self._allowed = allowed

    @property
    def states(self) -> int:
        """Running states with a fruit, budgets not counted."""
        return sum(level.free.size for level in self._levels.values())

    def _locate(self, state: GameState) -> _Located:
        if (state.width, state.height) != (self.width, self.height):
            raise ValueError(f"solved {self.width}x{self.height}, not {state.width}x{state.height}")
        level = self._levels.get(state.length)
        if level is None:
            raise ValueError(f"lengths below {min(self._levels)} were not solved")
        cells = tuple(y * self.width + x for x, y in state.snake)
        body = level.index.get((cells[0], int(state.direction)) if state.length == 1 else cells)
        if body is None or int(level.heading[body]) != int(state.direction):
            raise ValueError("the heading must point from the neck to the head")
        if state.fruit is None:
            raise ValueError("a running game has a fruit")
        column = int(level.column[body, state.fruit[1] * self.width + state.fruit[0]])
        budget = self.cells + state.length - state.steps_since_fruit if self.fruit_timeout else 1 << 30
        return _Located(level, body, column, budget)

    def _lookup(
        self, table: dict[int, np.ndarray], length: int, body: int, column: int, budget: int
    ) -> float:
        stack = table[length]
        return float(stack[min(budget, len(stack) - 1), body, column])

    def _actions(self, state: GameState) -> tuple[list[float], list[float]]:
        loc = self._locate(state)
        level, length = loc.level, loc.level.length
        eat_value, eat_moves = self._eating[length]
        values, moves = [], []
        for a in range(3):
            open_ = level.target[loc.body, a] >= 0 and (
                self._allowed is None or bool(self._allowed[length][loc.body, a, loc.column])
            )
            if not open_:
                value, cost = 0.0, math.inf
            elif level.eats[loc.body, a, loc.column]:
                value, cost = float(eat_value[loc.body, a]), 1.0 + float(eat_moves[loc.body, a])
            elif loc.budget <= 1:  # this move would time out
                value, cost = 0.0, math.inf
            else:
                succ, column = int(level.succ[loc.body, a]), int(level.next_column[loc.body, a, loc.column])
                value = self._lookup(self._values, length, succ, column, loc.budget - 1)
                cost = 1.0 + self._lookup(self._moves, length, succ, column, loc.budget - 1)
            values.append(value)
            moves.append(cost if value == 1.0 else math.inf)
        return values, moves

    def value(self, state: GameState) -> float:
        """Highest probability of filling the board from ``state`` (0 if the timeout has ended it)."""
        if state.done:
            return 1.0 if state.won else 0.0
        loc = self._locate(state)
        if loc.budget <= 0:
            return 0.0
        return self._lookup(self._values, loc.level.length, loc.body, loc.column, loc.budget)

    def moves(self, state: GameState) -> float:
        """Fewest expected moves to fill the board while keeping the win certain; NaN if it is not certain."""
        if state.done:
            return 0.0 if state.won else math.nan
        loc = self._locate(state)
        moves = (
            self._lookup(self._moves, loc.level.length, loc.body, loc.column, loc.budget)
            if loc.budget > 0
            else math.inf
        )
        return moves if math.isfinite(moves) else math.nan

    def action_values(self, state: GameState) -> tuple[float, float, float]:
        """Win probability of each relative action (``LEFT``, ``STRAIGHT``, ``RIGHT``), then best play."""
        return tuple(self._actions(state)[0])

    def action_moves(self, state: GameState) -> tuple[float, float, float]:
        """Expected moves after each action, among certain wins; inf where the win is not certain."""
        return tuple(self._actions(state)[1])

    def best_action(self, state: GameState) -> RelativeAction:
        """The contract's choice: the highest win probability, then the fewest moves."""
        values, moves = self._actions(state)
        return RelativeAction(choose_safely(values, moves))

    def regret(self, state: GameState, action: RelativeAction) -> float:
        """Win probability lost by playing ``action`` instead of the best move."""
        return self.value(state) - self.action_values(state)[action]

    def agent(self) -> Agent:
        """The optimal agent, playing :meth:`best_action`."""
        return self.best_action

    def start_value(self, initial_length: int = 3) -> float:
        """Win probability from the standard start, averaged over the first fruit."""
        return float(
            np.mean([self.value(s) for s in standard_starts(self.width, self.height, initial_length)])
        )

    def start_moves(self, initial_length: int = 3) -> float:
        """Expected moves from the standard start under :meth:`best_action`, averaged over the first fruit."""
        return float(
            np.mean([self.moves(s) for s in standard_starts(self.width, self.height, initial_length)])
        )


def standard_starts(width: int, height: int, initial_length: int = 3) -> list[GameState]:
    """The standard start with each possible first fruit."""
    snake = GameConfig(width, height, initial_length).start_snake()
    return [
        GameState(width, height, snake, Direction.RIGHT, (x, y))
        for y in range(height)
        for x in range(width)
        if (x, y) not in snake
    ]


def _cycle_positions(envelope: CycleEnvelope, width: int) -> np.ndarray:
    position = np.zeros(envelope.cells, dtype=np.int64)
    for (x, y), i in envelope.position.items():
        position[y * width + x] = i
    return position


def solve(
    width: int,
    height: int,
    *,
    fruit_timeout: bool = True,
    envelope: CycleEnvelope | None = None,
    min_length: int = 1,
) -> ExactSolution:
    """Solve a small board exactly.

    ``fruit_timeout`` applies the training rules: lost after ``N + L`` moves
    without fruit. Without it a game that never eats is simply never won. With
    an ``envelope`` only envelope moves are allowed, so states out of cycle
    order count as lost and the moves are the envelope's optimum. Lengths below
    ``min_length`` are skipped; no state of a game is shorter than its start.
    """
    cells = width * height
    if not 1 <= min_length < cells:
        raise ValueError(f"min_length must be in [1, {cells}), got {min_length}")
    if envelope is not None and envelope.cells != cells:
        raise ValueError("the envelope is for another board")
    levels = _levels(width, height, min_length)
    position = None if envelope is None else _cycle_positions(envelope, width)
    values, moves, eating = {}, {}, {}
    allowed: dict[int, np.ndarray] | None = None if envelope is None else {}
    after = None
    for length in range(cells - 1, min_length - 1, -1):
        level = levels[length]
        eating[length] = _eating(level, after, math.inf)
        mask = None if position is None else _envelope_mask(level, position, cells)
        if allowed is not None:
            allowed[length] = mask
        values[length], moves[length] = _induct(
            level, *eating[length], mask, cells + length if fruit_timeout else None
        )
        after = _afterstates(values[length][-1], moves[length][-1], certain_only=True)
    return ExactSolution(width, height, fruit_timeout, envelope, levels, values, moves, eating, allowed)


@dataclass(frozen=True)
class PolicyValue:
    """Exact results of a policy from the standard start, averaged over the first fruit."""

    win: float
    """Probability of filling the board."""
    moves: float
    """Expected moves until the game ends, won or lost."""
    states: int
    """States (body and fruit) the policy can reach."""


def evaluate_policy(
    width: int,
    height: int,
    policy: Callable[[GameState], RelativeAction | Direction],
    *,
    initial_length: int = 3,
) -> PolicyValue:
    """Exact win probability and expected moves of ``policy`` from the standard start.

    The training rules apply (lost after ``N + L`` moves without fruit). The
    policy must depend on the board alone: it is asked once for every body and
    fruit it can reach from the start, with the move counters at zero.
    """
    cells = width * height
    levels = _levels(width, height, initial_length)
    chosen = {length: np.ones(level.free.shape, dtype=np.int64) for length, level in levels.items()}
    start = tuple(y * width + x for x, y in GameConfig(width, height, initial_length).start_snake())
    first = levels[initial_length]
    start_body = first.index[(start[0], int(Direction.RIGHT)) if initial_length == 1 else start]
    frontier = [(initial_length, start_body, j) for j in range(cells - initial_length)]
    seen = set(frontier)
    while frontier:
        length, body, j = frontier.pop()
        level = levels[length]
        fruit = int(level.free[body, j])
        heading = Direction(int(level.heading[body]))
        snake = tuple((c % width, c // width) for c in level.bodies[body])
        action = policy(GameState(width, height, snake, heading, (fruit % width, fruit // width)))
        a = int(to_relative(heading, action) if isinstance(action, Direction) else action)
        chosen[length][body, j] = a
        if level.target[body, a] < 0 or level.grow[body, a] == _WIN and level.eats[body, a, j]:
            continue
        if level.eats[body, a, j]:
            grown = int(level.grow[body, a])
            nexts = [(length + 1, grown, k) for k in range(cells - length - 1)]
        else:
            nexts = [(length, int(level.succ[body, a]), int(level.next_column[body, a, j]))]
        for nxt in nexts:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)

    after = None
    for length in range(cells - 1, initial_length - 1, -1):
        level = levels[length]
        eat_value, eat_moves = _eating(level, after, 0.0)
        rows = np.arange(len(level.keys))[:, None]
        a = chosen[length]
        fatal = level.target[rows, a] < 0
        eats = np.take_along_axis(level.eats, a[:, None, :], axis=1)[:, 0]
        column = np.take_along_axis(level.next_column, a[:, None, :], axis=1)[:, 0]
        succ = np.maximum(level.succ[rows, a], 0)
        value = np.zeros(level.free.shape)
        moves = np.zeros(level.free.shape)
        for _ in range(cells + length):  # 1, 2, ..., N + L moves left before the timeout
            value, moves = (
                np.where(fatal, 0.0, np.where(eats, eat_value[rows, a], value[succ, column])),
                1.0 + np.where(fatal, 0.0, np.where(eats, eat_moves[rows, a], moves[succ, column])),
            )
        after = _afterstates(value, moves, certain_only=False)
    return PolicyValue(
        win=float(value[start_body].mean()), moves=float(moves[start_body].mean()), states=len(seen)
    )
