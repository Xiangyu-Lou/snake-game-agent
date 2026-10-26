"""Hamiltonian cycles: cycle order, the cycle-order envelope and the shortcut cycle.

The body is *in cycle order* when, read from tail to head, it advances around a
Hamiltonian cycle by less than one lap. Every cell from just ahead of the head
round to the tail is then free, so the head can always move on along the
cycle. The cycles are those of :func:`snake_game.policies.hamiltonian_cycles`;
a board needs an even side to have one.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from types import MappingProxyType

from snake_game import Direction, GameState, Position, RelativeAction
from snake_game.policies import hamiltonian_cycles

from .contract import to_relative


@functools.cache
def board_cycles(width: int, height: int) -> tuple[Mapping[Position, Position], ...]:
    """The cycles of :func:`hamiltonian_cycles` as read-only maps (cached); empty if there are none."""
    try:
        return tuple(MappingProxyType(cycle) for cycle in hamiltonian_cycles(width, height))
    except ValueError:
        return ()


def aligned(cycle: Mapping[Position, Position], state: GameState) -> bool:
    """Whether the body occupies consecutive cells of ``cycle``, as when following it."""
    snake = state.snake
    return all(cycle[snake[k]] == snake[k - 1] for k in range(1, len(snake)))


def aligned_cycles(state: GameState) -> list[Mapping[Position, Position]]:
    """The cycles of :func:`board_cycles`, in order, that the body is aligned with."""
    return [cycle for cycle in board_cycles(state.width, state.height) if aligned(cycle, state)]


class CycleEnvelope:
    """The cycle-order envelope of one Hamiltonian cycle: the contract's safety envelope.

    With the body in cycle order, a move is *inside the envelope* when it enters
    a cell further along the cycle without passing the tail or the fruit:
    ``0 < ahead(target) <= min(ahead(tail), ahead(fruit))``, distances counted
    along the cycle from the head. Such a move is safe and keeps the body in
    cycle order, and it brings the head at least one step closer to the fruit,
    which is therefore eaten within ``N`` moves. Playing only envelope moves
    fills the board with certainty, so every envelope move is equally safe, and
    the cycle's next cell is always one of them.
    """

    def __init__(self, cycle: Mapping[Position, Position]) -> None:
        start = next(iter(cycle))
        position: dict[Position, int] = {}
        cell = start
        for i in range(len(cycle)):
            position[cell] = i
            cell = cycle[cell]
        if cell != start or len(position) != len(cycle):
            raise ValueError("not a single cycle through every cell")
        self.cycle = dict(cycle)
        self.position = position
        self.cells = len(cycle)

    @classmethod
    def for_state(cls, state: GameState) -> CycleEnvelope | None:
        """The envelope of the first cycle the body is aligned with; None if there is none.

        On a standard start this is the envelope a game must stay in.
        """
        cycles = aligned_cycles(state)
        return cls(cycles[0]) if cycles else None

    def ahead(self, state: GameState, cell: Position) -> int:
        """Steps along the cycle from the head to ``cell``."""
        return (self.position[cell] - self.position[state.head]) % self.cells

    def contains(self, state: GameState) -> bool:
        """Whether the body is in cycle order, i.e. the envelope is defined for ``state``."""
        order = [self.position[cell] for cell in reversed(state.snake)]
        return sum((b - a) % self.cells for a, b in zip(order, order[1:], strict=False)) < self.cells

    def limit(self, state: GameState) -> int:
        """Longest jump inside the envelope: the distance to the tail or to the fruit, if nearer."""
        tail = self.ahead(state, state.tail) if state.length > 1 else self.cells
        return tail if state.fruit is None else min(tail, self.ahead(state, state.fruit))

    def jump(self, state: GameState, direction: Direction) -> int | None:
        """Steps along the cycle that the move in ``direction`` jumps; None if it leaves the envelope.

        Assumes the body is in cycle order (see :meth:`contains`).
        """
        if direction == state.direction.opposite:
            return None
        dx, dy = Direction(direction).delta
        cell = (state.head[0] + dx, state.head[1] + dy)
        if cell not in self.position:
            return None
        jump = self.ahead(state, cell)
        return jump if 0 < jump <= self.limit(state) else None

    def moves(self, state: GameState) -> dict[Direction, int]:
        """The envelope's moves and the jump of each. Assumes the body is in cycle order."""
        jumps = {}
        for d in Direction:
            jump = self.jump(state, d)
            if jump is not None:
                jumps[d] = jump
        return jumps

    def mask(self, state: GameState) -> tuple[bool, bool, bool]:
        """Which relative actions (``LEFT``, ``STRAIGHT``, ``RIGHT``) are inside the envelope."""
        inside = (
            {to_relative(state.direction, d) for d in self.moves(state)} if self.contains(state) else set()
        )
        return tuple(action in inside for action in RelativeAction)

    def allows(self, state: GameState, action: RelativeAction) -> bool:
        """Whether ``action`` is inside the envelope (False if the body is out of cycle order)."""
        return self.mask(state)[action]


class ShortcutCyclePolicy:
    """The certified shortcut cycle: the speed reference for standard starts.

    It stays inside the envelope of the cycle the start is aligned with, taking
    the longest jump while the snake fills less than ``max_fill`` of the board,
    then following the cycle (late jumps leave holes, and a fruit that appears
    in a hole costs most of a lap). It therefore fills the board in every game,
    eating each fruit within ``N`` moves. With the default cutoff it needs about
    72%, 60% and 52% of the plain cycle's moves on 6x6, 10x10 and 20x20.
    """

    def __init__(self, max_fill: float = 0.5) -> None:
        self.max_fill = max_fill
        self.envelope: CycleEnvelope | None = None

    def __call__(self, state: GameState) -> Direction:
        if self.envelope is None:
            self.envelope = CycleEnvelope.for_state(state)
            if self.envelope is None:
                raise ValueError("the body is not aligned with a Hamiltonian cycle of the board")
        moves = self.envelope.moves(state)
        if not moves:
            raise RuntimeError("the body left cycle order")
        if state.length >= self.max_fill * state.num_cells:
            return min(moves, key=moves.__getitem__)
        return max(moves, key=moves.__getitem__)
