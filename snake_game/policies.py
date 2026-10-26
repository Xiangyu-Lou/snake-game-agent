"""Built-in reference policies: demos, test fixtures and baselines.

They are *not* the project's AI; they exist to drive the UI's autopilot, to
exercise every rule (including winning) in the tests, and as yardsticks for
learned agents. A policy is any callable ``policy(state: GameState) ->
Direction``; plug your own into the UI with ``python -m snake_game --agent
module:attribute``.
"""

from __future__ import annotations

import random
from collections import deque

from .engine import DELTAS, Direction, GameState, Position

__all__ = [
    "RandomPolicy",
    "GreedyPolicy",
    "HamiltonianPolicy",
    "POLICIES",
    "hamiltonian_cycles",
    "safe_directions",
]


def _neighbour(position: Position, direction: Direction) -> Position:
    dx, dy = DELTAS[direction]
    return (position[0] + dx, position[1] + dy)


def safe_directions(state: GameState) -> list[Direction]:
    """Directions that do not end the game on the next move (reversal excluded)."""
    if state.done:
        return []
    blocked = set(state.snake[:-1])  # the tail moves out of the way
    head, reverse = state.head, state.direction.opposite
    safe = []
    for d in Direction:
        x, y = _neighbour(head, d)
        if d != reverse and 0 <= x < state.width and 0 <= y < state.height and (x, y) not in blocked:
            safe.append(d)
    return safe


class RandomPolicy:
    """Uniformly random among the moves that do not lose immediately."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def __call__(self, state: GameState) -> Direction:
        safe = safe_directions(state)
        return self._rng.choice(safe) if safe else state.direction


class GreedyPolicy:
    """Shortest path to the fruit (breadth-first search), with a little foresight.

    The first step of the path is only taken if the snake still has room to move
    afterwards; otherwise (or without a path) it picks the safe move that keeps
    the largest reachable area. Strong early, it eventually traps itself.
    """

    def __call__(self, state: GameState) -> Direction:
        safe = safe_directions(state)
        if not safe:
            return state.direction
        step = self._first_step_to_fruit(state)
        if step is not None and self._room_after(state, step) >= state.length:
            return step
        return max(safe, key=lambda d: self._room_after(state, d))

    @staticmethod
    def _first_step_to_fruit(state: GameState) -> Direction | None:
        if state.fruit is None:
            return None
        blocked = set(state.snake[:-1])
        start, goal = state.head, state.fruit
        first: dict[Position, Direction | None] = {start: None}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell == goal:
                return first[cell]
            for d in Direction:
                nxt = _neighbour(cell, d)
                if (
                    nxt not in first
                    and nxt not in blocked
                    and 0 <= nxt[0] < state.width
                    and 0 <= nxt[1] < state.height
                ):
                    first[nxt] = d if cell == start else first[cell]
                    queue.append(nxt)
        return None

    @staticmethod
    def _room_after(state: GameState, direction: Direction) -> int:
        """Cells reachable from the head after moving in ``direction``."""
        head = _neighbour(state.head, direction)
        body = state.snake if head == state.fruit else state.snake[:-1]
        blocked = set(body[:-1])  # the new tail keeps moving
        blocked.add(head)
        seen = {head}
        queue = deque([head])
        while queue:
            cell = queue.popleft()
            for d in Direction:
                nxt = _neighbour(cell, d)
                if (
                    nxt not in seen
                    and nxt not in blocked
                    and 0 <= nxt[0] < state.width
                    and 0 <= nxt[1] < state.height
                ):
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen)


class HamiltonianPolicy:
    """Follow a Hamiltonian cycle of the board: slow, but it always fills the board.

    Once the snake's body lies along the cycle, following it can never collide,
    so every game ends in a win (from the standard start this holds from the
    first move). Requires an even width or height - an odd x odd grid has no
    Hamiltonian cycle.
    """

    def __init__(self) -> None:
        self._cycles: dict[tuple[int, int], list[dict[Position, Position]]] = {}
        self._chosen: dict[Position, Position] | None = None

    @property
    def cycle(self) -> dict[Position, Position] | None:
        """The cycle being followed (cell -> next cell); ``None`` before the first move."""
        return self._chosen

    def __call__(self, state: GameState) -> Direction:
        cycle = self._cycle(state)
        target = cycle[state.head]
        if target not in self._unsafe_targets(state):
            return Direction.from_delta(target[0] - state.head[0], target[1] - state.head[1])
        safe = safe_directions(state)  # not on the cycle yet: avoid crashing
        return safe[0] if safe else state.direction

    @staticmethod
    def _unsafe_targets(state: GameState) -> set[Position]:
        head, blocked = state.head, set(state.snake[:-1])
        return {
            p
            for p in (_neighbour(head, d) for d in Direction)
            if p in blocked or not (0 <= p[0] < state.width and 0 <= p[1] < state.height)
        }

    def _cycle(self, state: GameState) -> dict[Position, Position]:
        if self._chosen is not None and self._aligned(self._chosen, state):
            return self._chosen
        candidates = self._candidates(state.width, state.height)
        neck = state.snake[1] if state.length > 1 else None
        aligned = [c for c in candidates if self._aligned(c, state)]
        usable = aligned or [c for c in candidates if c[state.head] != neck] or candidates
        self._chosen = usable[0]
        return self._chosen

    @staticmethod
    def _aligned(cycle: dict[Position, Position], state: GameState) -> bool:
        snake = state.snake
        return all(cycle[snake[k]] == snake[k - 1] for k in range(1, len(snake)))

    def _candidates(self, width: int, height: int) -> list[dict[Position, Position]]:
        key = (width, height)
        if key not in self._cycles:
            self._cycles[key] = hamiltonian_cycles(width, height)
        return self._cycles[key]


def hamiltonian_cycles(width: int, height: int) -> list[dict[Position, Position]]:
    """The cycles :class:`HamiltonianPolicy` chooses from, in its order of preference.

    Each cycle maps every cell to the next one. They are serpentines along rows
    (even ``height``) or columns (even ``width``), mirrored and in both directions.
    Raises ``ValueError`` if both sides are odd (no Hamiltonian cycle exists) or
    the board is a single row or column.
    """
    if width % 2 and height % 2:
        raise ValueError(f"a {width}x{height} board has no Hamiltonian cycle (both sides odd)")
    paths = []
    if height % 2 == 0 and width >= 2:
        paths.append(_serpentine(width, height))
    if width % 2 == 0 and height >= 2:
        paths.append([(x, y) for y, x in _serpentine(height, width)])
    variants = []
    for path in paths:
        for mirrored in (path, [(width - 1 - x, y) for x, y in path]):
            for oriented in (mirrored, mirrored[::-1]):
                variants.append({cell: oriented[(i + 1) % len(oriented)] for i, cell in enumerate(oriented)})
    if not variants:
        raise ValueError(f"no Hamiltonian cycle construction for a {width}x{height} board")
    return variants


def _serpentine(width: int, height: int) -> list[Position]:
    """Cycle for an even ``height``: rows snake through columns 1.., column 0 returns."""
    path: list[Position] = []
    for y in range(height):
        xs = range(1, width) if y % 2 == 0 else range(width - 1, 0, -1)
        path.extend((x, y) for x in xs)
    path.extend((0, y) for y in range(height - 1, -1, -1))
    return path


POLICIES = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
    "hamiltonian": HamiltonianPolicy,
}
"""Built-in policies by name (as accepted by ``python -m snake_game --agent``)."""
