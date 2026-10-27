"""Reference policies for evaluation: the yardsticks a learned agent must beat.

Besides the game's built-in policies (:mod:`snake_game.policies`), this adds the
two references of the Architecture B review: the certified shortcut cycle
(:class:`~snake_agent.cycles.ShortcutCyclePolicy`, the speed reference on
standard starts) and :class:`GuardedGreedyPolicy` (a recovery reference, and
the source of the recovery starts). :data:`REFERENCES` describes them all for
the evaluation harness.
"""

from __future__ import annotations

from snake_game import Direction, GameState
from snake_game.policies import GreedyPolicy, HamiltonianPolicy, RandomPolicy

from .cycles import ShortcutCyclePolicy
from .evaluation import AgentSpec
from .survival import Board

_UNREACHABLE = 1 << 30


class GuardedGreedyPolicy:
    """Greedy, restricted to moves that keep a survival certificate.

    Among the safe moves that win or whose next state has a survival certificate
    (:meth:`~snake_agent.survival.Board.has_survival_certificate`), it takes one
    on a shortest path to the fruit, then the one leaving the most room. With no
    such move, it takes the safe move leaving the most room. This is "eat as
    fast as possible while staying alive", one fruit at a time: it rarely dies,
    but it often loops. A reference policy, not a proposed agent.
    """

    def __call__(self, state: GameState) -> Direction:
        board = Board(state)
        safe = [d for d in range(4) if board.target(d) >= 0]
        if not safe:
            return state.direction
        width = state.width
        fruit = -1 if state.fruit is None else state.fruit[1] * width + state.fruit[0]
        good = []
        for d in safe:
            cell = board.target(d)
            undo = board.move(d, cell, cell == fruit)
            if board.length == board.cells or board.has_survival_certificate():
                good.append(d)
            board.unmove(cell, undo)
        snake = [y * width + x for x, y in state.snake]
        if not good:
            return Direction(_most_room(board, snake, fruit, safe))
        if len(good) == 1:
            return Direction(good[0])
        blocked = bytearray(board.cells)
        for cell in snake[:-1]:
            blocked[cell] = 1
        ahead = board.neighbours[board.head]
        distances = [_fruit_distance(board.neighbours, blocked, ahead[d], fruit) for d in good]
        nearest = min(distances)
        tied = [d for d, distance in zip(good, distances, strict=True) if distance == nearest]
        return Direction(tied[0] if len(tied) == 1 else _most_room(board, snake, fruit, tied))


def _fruit_distance(neighbours, blocked: bytearray, start: int, fruit: int) -> int:
    """Shortest path from ``start`` to the fruit avoiding ``blocked`` cells (the body minus its tail)."""
    if start == fruit:
        return 0
    seen = bytearray(blocked)
    seen[start] = 1
    frontier, distance = [start], 0
    while frontier:
        distance += 1
        layer = []
        for cell in frontier:
            for nxt in neighbours[cell]:
                if nxt < 0:
                    continue
                if nxt == fruit:
                    return distance
                if not seen[nxt]:
                    seen[nxt] = 1
                    layer.append(nxt)
        frontier = layer
    return _UNREACHABLE


def _room_after(board: Board, snake: list[int], fruit: int, cell: int) -> int:
    """Cells reachable from the head after it moves into ``cell``, as in :class:`GreedyPolicy`."""
    body = snake if cell == fruit else snake[:-1]
    blocked = bytearray(board.cells)
    for c in body[:-1]:  # the new tail keeps moving
        blocked[c] = 1
    blocked[cell] = 1
    neighbours = board.neighbours
    frontier, count = [cell], 1
    while frontier:
        layer = []
        for c in frontier:
            for nxt in neighbours[c]:
                if nxt >= 0 and not blocked[nxt]:
                    blocked[nxt] = 1
                    layer.append(nxt)
        count += len(layer)
        frontier = layer
    return count


def _most_room(board: Board, snake: list[int], fruit: int, directions: list[int]) -> int:
    """The first of ``directions`` that leaves the most room."""
    if len(directions) == 1:
        return directions[0]
    ahead = board.neighbours[board.head]
    return max(directions, key=lambda d: _room_after(board, snake, fruit, ahead[d]))


REFERENCES: dict[str, AgentSpec] = {
    spec.name: spec
    for spec in (
        AgentSpec("random", lambda seed: RandomPolicy(seed)),
        AgentSpec("greedy", lambda seed: GreedyPolicy(), deterministic=True),
        AgentSpec("guarded-greedy", lambda seed: GuardedGreedyPolicy(), deterministic=True),
        # The cycle a HamiltonianPolicy follows is part of its memory: it can switch cycles.
        AgentSpec(
            "hamiltonian",
            lambda seed: HamiltonianPolicy(),
            deterministic=True,
            memory=lambda policy: id(policy.cycle),
        ),
        AgentSpec("shortcut", lambda seed: ShortcutCyclePolicy(), deterministic=True),
    )
}
"""The reference policies by name: ``random``, ``greedy``, ``guarded-greedy``,
``hamiltonian`` (the plain cycle) and ``shortcut`` (the shortcut cycle)."""
