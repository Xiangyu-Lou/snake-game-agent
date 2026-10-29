"""Exact survival analysis: the survival certificate and the lost-proof search.

Both reason about the snake moving *without eating*, on the timestamped body of
:class:`Board`. That relaxation makes them exact and cheap, and it is sound in
one direction:

* **Eating only lengthens the body.** Along the same moves, a snake that eats
  always covers the cells of one that does not, so it can only collide sooner.
  If every continuation without eating collides, the real game is lost, unless
  the snake fills the board first. :func:`prove_lost` excludes that case, so its
  "lost" verdict is a proof for the real game.
* **A certificate is a hint, not a guarantee.**
  :meth:`Board.has_survival_certificate` is a sufficient condition for the
  snake to survive forever *without eating*. It treats the fruit as a free
  cell: if the certified path runs through the fruit, the real snake eats it,
  and its longer body may then clear the path too late. Use certificates to
  guide search and as inputs, never to prune.

Cells are flat indices ``y * width + x`` and directions are absolute, ``0..3``
in :class:`~snake_game.Direction` order.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum

from snake_game import DELTAS, Direction, GameState, RelativeAction

NEVER = -(1 << 40)
"""``Board.entered`` value of a cell that is not part of the body."""


@functools.cache
def neighbour_table(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    """``table[cell][d]``: the cell next to ``cell`` in direction ``d``, or -1 off the board."""
    table = []
    for cell in range(width * height):
        x, y = cell % width, cell // width
        row = []
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            row.append(ny * width + nx if 0 <= nx < width and 0 <= ny < height else -1)
        table.append(tuple(row))
    return tuple(table)


class Board:
    """The body as entry times, like :class:`~snake_game.VectorSnakeEnv`.

    ``t`` counts moves and ``entered[c]`` is when the head last entered cell
    ``c``: the cell is occupied while ``t - entered[c] < length``. A move writes
    one timestamp and can be undone in O(1), which makes exhaustive search cheap.
    """

    __slots__ = ("width", "height", "cells", "neighbours", "length", "t", "entered", "head", "heading")

    def __init__(self, state: GameState) -> None:
        width = state.width
        self.width, self.height = width, state.height
        self.cells = width * state.height
        self.neighbours = neighbour_table(width, state.height)
        self.length = state.length
        self.t = 0
        entered = [NEVER] * self.cells
        for k, (x, y) in enumerate(state.snake):
            entered[y * width + x] = -k
        self.entered = entered
        x, y = state.head
        self.head = y * width + x
        self.heading = int(state.direction)

    def release(self, cell: int) -> int:
        """Moves until ``cell`` is vacated if nothing is eaten: 1 for the tail, <= 0 if free."""
        return self.length - (self.t - self.entered[cell])

    def target(self, direction: int) -> int:
        """The cell entered by moving in ``direction``; -1 if that is fatal or a reversal."""
        if direction == (self.heading + 2) % 4:
            return -1
        cell = self.neighbours[self.head][direction]
        if cell < 0 or self.release(cell) > 1:
            return -1
        return cell

    def move(self, direction: int, cell: int, grow: bool = False) -> tuple[int, int, int, int]:
        """Move the head into ``cell`` (a :meth:`target`); returns the record for :meth:`unmove`."""
        undo = (self.head, self.heading, self.entered[cell], self.length)
        self.t += 1
        self.length += grow
        self.entered[cell] = self.t
        self.head, self.heading = cell, direction
        return undo

    def unmove(self, cell: int, undo: tuple[int, int, int, int]) -> None:
        """Undo the :meth:`move` into ``cell``."""
        self.head, self.heading, self.entered[cell], self.length = undo
        self.t -= 1

    def body(self) -> list[int]:
        """Cells of the body, head first."""
        free_before = self.t - self.length
        cells = [c for c in range(self.cells) if self.entered[c] > free_before]
        return sorted(cells, key=self.entered.__getitem__, reverse=True)

    def has_survival_certificate(self) -> bool:
        """Sufficient condition for surviving forever without eating.

        True if a shortest path through free cells brings the head next to a body
        segment that has moved away by the time the head arrives. The head can
        then enter that cell and follow its own body around a closed loop forever
        (:meth:`certificate_loop`). The fruit counts as a free cell, and False
        does not mean the snake is lost.
        """
        entered, neighbours, head = self.entered, self.neighbours, self.head
        free_before = self.t - self.length  # release(c) == entered[c] - free_before
        reverse = (self.heading + 2) % 4
        seen = bytearray(self.cells)
        seen[head] = 1
        frontier = [head]
        arrival = 1  # when the head would enter a neighbour of the frontier
        while frontier:
            layer = []
            for cell in frontier:
                for d, nxt in enumerate(neighbours[cell]):
                    if nxt < 0 or (cell == head and d == reverse):
                        continue
                    release = entered[nxt] - free_before
                    if release > 0:
                        if release <= arrival:
                            return True
                    elif not seen[nxt]:
                        seen[nxt] = 1
                        layer.append(nxt)
            frontier = layer
            arrival += 1
        return False

    def certificate_loop(self) -> list[int] | None:
        """The closed loop behind the certificate, as the cells to enter in order; None without one.

        The loop runs from the head through free cells into a vacated body cell,
        then along the body back to the head's cell. It has at least ``length``
        cells, so the snake, not growing, can circle it forever.
        """
        entered, neighbours, head = self.entered, self.neighbours, self.head
        free_before = self.t - self.length
        reverse = (self.heading + 2) % 4
        parent = {head: -1}
        frontier = [head]
        arrival = 1
        while frontier:
            layer = []
            for cell in frontier:
                for d, nxt in enumerate(neighbours[cell]):
                    if nxt < 0 or (cell == head and d == reverse):
                        continue
                    release = entered[nxt] - free_before
                    if release > 0:
                        if release <= arrival:
                            path = []
                            while cell != head:
                                path.append(cell)
                                cell = parent[cell]
                            segments = [c for c in self.body() if entered[c] >= entered[nxt]]
                            return path[::-1] + segments[::-1]
                    elif nxt not in parent:
                        parent[nxt] = cell
                        layer.append(nxt)
            frontier = layer
            arrival += 1
        return None


def arrival_times(board: Board, avoid: int = -1) -> list[int]:
    """The earliest move at which the head can enter each cell, without eating; -1 if never.

    A breadth-first search from the head through free cells and through body
    cells that have moved away by the time the head arrives: a cell entered at
    move ``t`` must have ``release <= t``. The head cannot wait, so a cell that
    is reached only by circling first counts as never. ``avoid`` (for example
    the fruit, on a path that must not eat) is never entered.
    """
    entered, neighbours, head = board.entered, board.neighbours, board.head
    free_before = board.t - board.length  # release(c) == entered[c] - free_before
    arrival = [-1] * board.cells
    arrival[head] = 0
    reverse = (board.heading + 2) % 4
    frontier, t = [head], 1
    while frontier:
        layer = []
        for cell in frontier:
            for d, nxt in enumerate(neighbours[cell]):
                if nxt < 0 or arrival[nxt] >= 0 or nxt == avoid or (cell == head and d == reverse):
                    continue
                if entered[nxt] - free_before <= t:
                    arrival[nxt] = t
                    layer.append(nxt)
        frontier = layer
        t += 1
    return arrival


def opening_times(board: Board) -> list[int]:
    """When the head can first enter each cell, without eating, if it could wait where it stands.

    As :func:`arrival_times`, but a body cell is entered at ``max(t, release)``: a lower bound on the
    moves to any cell, reached or not, that counts a hole by when the body around it moves away. The
    head cannot really wait, so a cell is no sooner reachable than this. Off-board cells are skipped;
    the head's own cell is 0.
    """
    entered, neighbours, head = board.entered, board.neighbours, board.head
    free_before = board.t - board.length
    cells = board.cells
    time = [-1] * cells
    time[head] = 0
    reverse = (board.heading + 2) % 4
    buckets: list[list[int]] = [[head]]
    t = 0
    while t < len(buckets):
        for cell in buckets[t]:
            if time[cell] != t:
                continue
            for d, nxt in enumerate(neighbours[cell]):
                if nxt < 0 or (cell == head and d == reverse):
                    continue
                when = max(t + 1, entered[nxt] - free_before)
                if time[nxt] < 0 or when < time[nxt]:
                    time[nxt] = when
                    while len(buckets) <= when:
                        buckets.append([])
                    buckets[when].append(nxt)
        t += 1
    return time


@dataclass(frozen=True)
class LostProof:
    """The verdict of :func:`prove_lost`."""

    lost: bool | None
    """True: every continuation collides (a proof for the real game). False: some
    continuation without eating survives forever, which does *not* prove that
    the real game survives. None: unknown."""
    nodes: int
    """States visited, the root included."""
    depth: int
    """Most moves before a forced collision, among the continuations searched."""


def prove_lost(board: Board, budget: int) -> LostProof:
    """Exact search over the continuations without eating, visiting at most ``budget`` states.

    A continuation stops at a state without safe moves or at a state with a
    survival certificate. The game is proven lost when every continuation is
    trapped after fewer moves than there are free cells: eating only lengthens
    the body, so the real snake is trapped no later, and it cannot have filled
    the board first. A proof that needs as many moves as there are free cells,
    or more states than ``budget``, gives ``lost=None``. The board is left
    unchanged.
    """
    free = board.cells - board.length
    nodes = 1
    if nodes > budget:
        return LostProof(None, nodes, 0)
    if board.has_survival_certificate():
        return LostProof(False, nodes, 0)
    # Depth-first, one frame per state: [next direction to try, cell entered, undo record, had a move].
    stack: list[list] = [[0, -1, None, False]]
    deepest = 0
    while stack:
        frame = stack[-1]
        d = frame[0]
        cell = -1
        while d < 4:
            cell = board.target(d)
            if cell >= 0:
                break
            d += 1
        if d == 4:  # every move from here collides, now or later
            if not frame[3]:
                deepest = max(deepest, len(stack) - 1)
            stack.pop()
            if frame[1] >= 0:
                board.unmove(frame[1], frame[2])
            continue
        frame[0], frame[3] = d + 1, True
        undo = board.move(d, cell)
        nodes += 1
        if nodes > budget or board.has_survival_certificate():
            board.unmove(cell, undo)
            for frame in reversed(stack[1:]):
                board.unmove(frame[1], frame[2])
            return LostProof(None if nodes > budget else False, nodes, deepest)
        stack.append([0, cell, undo, False])
    return LostProof(True if deepest < free else None, nodes, deepest)


class MoveClass(str, Enum):
    """What the exact instruments know about a move."""

    FATAL = "fatal"  # collides at once
    WON = "won"  # eats the last fruit
    CERTIFIED = "certified"  # the next state has a survival certificate
    LOST = "lost"  # the next state is proven lost
    UNKNOWN = "unknown"  # none of the above: never prune such a move


def classify_moves(state: GameState, budget: int = 0) -> tuple[MoveClass, MoveClass, MoveClass]:
    """The class of each relative action (``LEFT``, ``STRAIGHT``, ``RIGHT``) in a running game.

    Moves without a certificate are searched with :func:`prove_lost` within
    ``budget`` states; with ``budget=0`` they stay ``UNKNOWN``. Search should
    prune only ``FATAL`` and ``LOST`` moves.
    """
    if state.done:
        raise ValueError("the game is over")
    board = Board(state)
    fruit = -1 if state.fruit is None else state.fruit[1] * state.width + state.fruit[0]
    heading = int(state.direction)
    classes = []
    for action in RelativeAction:
        d = (heading + action - 1) % 4
        cell = board.target(d)
        if cell < 0:
            classes.append(MoveClass.FATAL)
            continue
        undo = board.move(d, cell, cell == fruit)
        if board.length == board.cells:
            classes.append(MoveClass.WON)
        elif board.has_survival_certificate():
            classes.append(MoveClass.CERTIFIED)
        elif budget > 0 and prove_lost(board, budget).lost:
            classes.append(MoveClass.LOST)
        else:
            classes.append(MoveClass.UNKNOWN)
        board.unmove(cell, undo)
    return tuple(classes)


def direction_of(board: Board, cell: int) -> Direction:
    """The direction from the head to the neighbouring ``cell``."""
    return Direction(board.neighbours[board.head].index(cell))
