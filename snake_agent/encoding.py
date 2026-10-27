"""Network inputs for milestone M1: one state per evaluation, in the head's frame.

The board is rotated so that the snake heads up: relative actions then always
point to the same three cells (left, up, right of the head), and the network
needs to learn only one orientation. Mirroring left and right (which swaps the
``LEFT`` and ``RIGHT`` actions) is the remaining symmetry; :func:`mirror` applies
it for augmentation.

Everything is exact preprocessing of the public state, measured in cells rather
than in fractions of the board where it matters for transfer across sizes:

* cell planes (:data:`PLANES`): the body with its release times, the fruit, the
  free cells the head can reach and how far, the slack of the body cells at the
  edge of that region (the survival certificate's ingredients), and the
  canonical Hamiltonian cycle: each cell's distance ahead of the head along it,
  its next cell, the free arc inside the envelope and the holes behind the tail;
* per-action features (:data:`ACTION_FEATURES`): the move classes of
  :func:`~snake_agent.survival.classify_moves`, whether the move eats, and
  whether it is inside the envelope and by how much it jumps;
* global features (:data:`GLOBALS`): occupancy, the fruit-timeout budget of
  decision 3, whether this is an *afterstate* (just after eating, fruit not yet
  placed), whether the body is in cycle order, and a few distances.

Geometry is size-independent: no plane holds absolute or normalized coordinates,
and the walls enter as distances in cells. Boards are placed in the top-left
corner of a ``size`` x ``size`` canvas, so 4x4 and 6x6 states can share a batch;
the ``valid`` plane marks real cells.

The moves head works in units of :func:`moves_scale`, the plain cycle's expected
moves to fill the remaining free cells.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from snake_game import Direction, GameState, RelativeAction

from .contract import Rules
from .cycles import CycleEnvelope, aligned_cycles
from .exact import standard_starts
from .survival import Board, MoveClass, classify_moves, neighbour_table

PLANES = (
    "valid",
    "head",
    "body",
    "tail",
    "fruit",
    "release",  # moves until the segment moves away (no eating), / N
    "release_cells",  # the same in cells, saturating at 16
    "reach",  # free cells the head can reach through free cells
    "distance",  # their shortest distance from the head, / N
    "slack",  # body cells next to that region: arrival minus release, / N, clipped to [-1, 1]
    "toward_up",  # each segment's direction to the one in front of it, in the head's frame
    "toward_right",
    "toward_down",
    "toward_left",
    "cycle_ahead",  # steps along the canonical cycle from the head, / N
    "cycle_up",  # each cell's next cell along the cycle, in the head's frame
    "cycle_right",
    "cycle_down",
    "cycle_left",
    "envelope_free",  # free cells the envelope lets the head jump to (in cycle order only)
    "hole",  # free cells behind the tail along the cycle (in cycle order only)
    "wall_up",  # distance in cells to the wall ahead of the head (in its frame), saturating at 16, / 16
    "wall_right",
    "wall_down",
    "wall_left",
)
GLOBALS = (
    "occupancy",
    "free",
    "side",  # board side / 20
    "timeout_enabled",
    "timeout_configured",
    "timeout_left",
    "afterstate",
    "in_cycle_order",
    "fruit_ahead",  # steps along the cycle to the fruit, / N (in cycle order only)
    "tail_ahead",
    "certified",  # the state itself has a survival certificate
)
ACTION_FEATURES = ("fatal", "won", "certified", "lost", "unknown", "eats", "envelope", "jump")
MOVE_CLASSES = (MoveClass.FATAL, MoveClass.WON, MoveClass.CERTIFIED, MoveClass.LOST, MoveClass.UNKNOWN)
SATURATION = 16
"""Cell counts saturate here in the ``release_cells`` and ``wall_*`` planes."""
_P = {name: i for i, name in enumerate(PLANES)}
_G = {name: i for i, name in enumerate(GLOBALS)}
_A = {name: i for i, name in enumerate(ACTION_FEATURES)}
MAX_MOVES = 2.0
"""The moves head's ceiling, in units of :func:`moves_scale`: the target of lost games."""


def moves_scale(free: int) -> float:
    """``F(F + 3) / 4``: the plain cycle's expected moves to fill ``free`` cells, the moves head's unit.

    Following the cycle, the next fruit is on average ``(F + 1) / 2`` cells
    ahead when ``F`` cells are free; summing over ``F, F - 1, ..., 1`` gives
    ``F(F + 3) / 4``, which keeps the head's targets of order 1 through a game.
    """
    return free * (free + 3) / 4


@dataclass(frozen=True)
class Encoded:
    """One encoded state: ``planes (C, size, size)``, ``actions (3, A)``, ``globals (G,)``.

    ``head`` is the head's flat index ``y * size + x`` in the canvas; the cells
    of the three relative actions are left of it, above it and right of it.
    """

    planes: np.ndarray
    actions: np.ndarray
    globals: np.ndarray
    head: int


@functools.cache
def canonical_cycle(width: int, height: int) -> np.ndarray | None:
    """Position along the board's canonical cycle of each flat cell; None if the board has none.

    The canonical cycle is the one the standard start is aligned with, which
    defines the T1 envelope (:meth:`CycleEnvelope.for_state`).
    """
    try:
        start = standard_starts(width, height)[0]
    except ValueError:  # too small for the standard start
        return None
    cycles = aligned_cycles(start)
    if not cycles:
        return None
    envelope = CycleEnvelope(cycles[0])
    position = np.zeros(width * height, dtype=np.int64)
    for (x, y), i in envelope.position.items():
        position[y * width + x] = i
    position.flags.writeable = False
    return position


@functools.cache
def _successor_directions(width: int, height: int) -> np.ndarray | None:
    """Direction from each cell to its next cell along the canonical cycle."""
    position = canonical_cycle(width, height)
    if position is None:
        return None
    cells = width * height
    at = np.empty(cells, dtype=np.int64)
    at[position] = np.arange(cells)
    neighbours = neighbour_table(width, height)
    directions = np.array([neighbours[c].index(int(at[(position[c] + 1) % cells])) for c in range(cells)])
    directions.flags.writeable = False
    return directions


def _rotated(cell_x: np.ndarray, cell_y: np.ndarray, k: int, width: int, height: int) -> tuple:
    """Coordinates after ``np.rot90(plane, k)`` of a ``height`` x ``width`` plane."""
    if k == 0:
        return cell_x, cell_y
    if k == 1:
        return cell_y, width - 1 - cell_x
    if k == 2:
        return width - 1 - cell_x, height - 1 - cell_y
    return height - 1 - cell_y, cell_x


def encode(
    state: GameState,
    rules: Rules,
    size: int | None = None,
    *,
    afterstate: bool = False,
    classes: tuple[MoveClass, MoveClass, MoveClass] | None = None,
) -> Encoded:
    """Encode a running ``state``; with ``afterstate`` its fruit is ignored (not yet placed).

    ``classes`` overrides the move classes, for instance with ``LOST`` moves
    found by a proof search at the root; by default they come from
    :func:`classify_moves` without one.
    """
    width, height = state.width, state.height
    cells = width * height
    size = max(width, height) if size is None else size
    if size < max(width, height):
        raise ValueError(f"a {width}x{height} board does not fit a {size}x{size} canvas")
    length = state.length
    flat = np.array([y * width + x for x, y in state.snake], dtype=np.int64)
    head, tail = int(flat[0]), int(flat[-1])
    fruit = -1 if afterstate or state.fruit is None else state.fruit[1] * width + state.fruit[0]

    planes = np.zeros((len(PLANES), cells), dtype=np.float32)
    planes[_P["valid"]] = 1.0
    planes[_P["head"], head] = 1.0
    planes[_P["body"], flat[1:]] = 1.0
    planes[_P["tail"], tail] = 1.0
    if fruit >= 0:
        planes[_P["fruit"], fruit] = 1.0
    release = np.arange(length, 0, -1)
    planes[_P["release"], flat] = release / cells
    planes[_P["release_cells"], flat] = np.minimum(release, SATURATION) / SATURATION

    # Free cells reachable from the head, their distance, and the slack of the body cells around them.
    board = Board(state)
    occupied = np.zeros(cells, dtype=bool)
    occupied[flat] = True
    release_of = np.zeros(cells, dtype=np.int64)
    release_of[flat] = release
    neighbours = board.neighbours
    distance = np.full(cells, -1, dtype=np.int64)
    slack = np.full(cells, -(1 << 30), dtype=np.int64)
    frontier, d = [head], 0
    distance[head] = 0
    reverse = (int(state.direction) + 2) % 4
    while frontier:
        layer = []
        for cell in frontier:
            for direction, nxt in enumerate(neighbours[cell]):
                if nxt < 0 or (cell == head and direction == reverse):
                    continue
                if occupied[nxt]:
                    slack[nxt] = max(slack[nxt], d + 1 - release_of[nxt])
                elif distance[nxt] < 0:
                    distance[nxt] = d + 1
                    layer.append(nxt)
        frontier = layer
        d += 1
    reach = (distance > 0) & ~occupied
    planes[_P["reach"]] = reach
    planes[_P["distance"]] = np.where(reach, distance, 0) / cells
    edge = slack > -(1 << 30)
    planes[_P["slack"]] = np.where(edge, np.clip(slack / cells, -1.0, 1.0), 0.0)

    k = int(state.direction)  # quarter turns that bring the heading up
    toward = np.full(cells, -1, dtype=np.int64)
    if length > 1:
        toward[flat[1:]] = [int(d) for d in state.segment_directions[1:]]
    for d in range(4):  # absolute direction d points (d - k) % 4 in the head's frame
        planes[_P["toward_up"] + (d - k) % 4] = toward == d

    globals_ = np.zeros(len(GLOBALS), dtype=np.float32)
    position = canonical_cycle(width, height)
    in_order = False
    jumps: dict[Direction, int] = {}
    if position is not None:
        ahead = (position - position[head]) % cells
        planes[_P["cycle_ahead"]] = ahead / cells
        successor = _successor_directions(width, height)
        for d in range(4):
            planes[_P["cycle_up"] + (d - k) % 4] = successor == d
        order = position[flat[::-1]]
        in_order = int(np.sum(np.diff(order) % cells)) < cells
        if in_order:
            tail_ahead = int(ahead[tail]) if length > 1 else cells
            limit = tail_ahead if fruit < 0 else min(tail_ahead, int(ahead[fruit]))
            free = ~occupied
            planes[_P["envelope_free"]] = free & (ahead > 0) & (ahead <= limit)
            planes[_P["hole"]] = free & (ahead > tail_ahead)
            globals_[_G["fruit_ahead"]] = 0.0 if fruit < 0 else ahead[fruit] / cells
            globals_[_G["tail_ahead"]] = tail_ahead / cells
            if not afterstate:
                envelope = _envelope(width, height)
                jumps = envelope.moves(state)

    globals_[_G["occupancy"]] = length / cells
    globals_[_G["free"]] = (cells - length) / cells
    globals_[_G["side"]] = max(width, height) / 20
    budget = _G["timeout_enabled"]
    globals_[budget : budget + 3] = rules.budget_features(state)
    globals_[_G["afterstate"]] = float(afterstate)
    globals_[_G["in_cycle_order"]] = float(in_order)
    globals_[_G["certified"]] = float(board.has_survival_certificate())

    actions = np.zeros((3, len(ACTION_FEATURES)), dtype=np.float32)
    if not afterstate:
        if classes is None:
            classes = classify_moves(state)
        heading = state.direction
        for a in RelativeAction:
            actions[a, MOVE_CLASSES.index(classes[a])] = 1.0
            target = neighbours[head][heading.turn(a)]
            actions[a, _A["eats"]] = float(target >= 0 and target == fruit)
            jump = jumps.get(heading.turn(a))
            if jump is not None:
                actions[a, _A["envelope"]] = 1.0
                actions[a, _A["jump"]] = jump / cells

    # Rotate into the head's frame and place on the canvas.
    grid = planes.reshape(len(PLANES), height, width)
    grid = np.rot90(grid, k, axes=(1, 2))
    canvas = np.zeros((len(PLANES), size, size), dtype=np.float32)
    rows, columns = grid.shape[1], grid.shape[2]
    canvas[:, :rows, :columns] = grid
    canvas[_WALLS, :rows, :columns] = _walls(rows, columns)
    hx, hy = _rotated(np.array(head % width), np.array(head // width), k, width, height)
    return Encoded(canvas, actions, globals_, int(hy) * size + int(hx))


_WALLS = slice(_P["wall_up"], _P["wall_left"] + 1)


@functools.cache
def _walls(rows: int, columns: int) -> np.ndarray:
    """The four ``wall_*`` planes (up, right, down, left) of a ``rows`` x ``columns`` board."""
    y, x = np.mgrid[0:rows, 0:columns]
    distances = np.stack([y, columns - 1 - x, rows - 1 - y, x])
    walls = (np.minimum(distances, SATURATION) / SATURATION).astype(np.float32)
    walls.flags.writeable = False
    return walls


@functools.cache
def _envelope(width: int, height: int) -> CycleEnvelope:
    return CycleEnvelope.for_state(standard_starts(width, height)[0])


def stack(encoded: list[Encoded]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch arrays ``(planes, actions, globals, head)`` of encodings on the same canvas."""
    return (
        np.stack([e.planes for e in encoded]),
        np.stack([e.actions for e in encoded]),
        np.stack([e.globals for e in encoded]),
        np.array([e.head for e in encoded], dtype=np.int64),
    )


_MIRRORED_PLANES = np.arange(len(PLANES))
for _prefix in ("toward", "cycle", "wall"):
    _i, _j = _P[f"{_prefix}_right"], _P[f"{_prefix}_left"]
    _MIRRORED_PLANES[_i], _MIRRORED_PLANES[_j] = _j, _i


def mirror(
    planes: np.ndarray,
    actions: np.ndarray,
    head: np.ndarray,
    width: np.ndarray,
    policy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Left-right mirror images of a batch: planes, per-action features, head indices and policies.

    ``width`` is each board's width in the head's frame (its canvas columns in
    use). Directions right and left swap, and so do the ``LEFT`` and ``RIGHT``
    actions. Global features are unchanged.
    """
    planes = planes[:, _MIRRORED_PLANES].copy()
    size = planes.shape[-1]
    for i, w in enumerate(width):
        planes[i, :, :, :w] = planes[i, :, :, :w][:, :, ::-1].copy()
    head_y, head_x = np.divmod(head, size)
    head = head_y * size + (width - 1 - head_x)
    actions = actions[:, ::-1].copy()
    if policy is not None:
        policy = policy[:, ::-1].copy()
    return planes, actions, head, policy
