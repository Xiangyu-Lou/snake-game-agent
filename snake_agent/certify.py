"""The late-game certificate: Hamiltonian cycles around spanning trees of 2x2 blocks.

Cut a board with even sides into 2x2 blocks. A spanning tree of the blocks'
grid gives a Hamiltonian cycle of the board, the walk around the tree: inside a
block the walk runs along a side unless the tree crosses that side, and then it
crosses into the next block and comes back along the parallel lane. Walking
with the tree on its left (orientation 0, anticlockwise on the screen), every
such cycle goes left along even rows, right along odd rows, down even columns
and up odd columns; orientation 1 is the same walk backwards. So each cell has
two possible moves, and one edge of the blocks' grid, the cell's *gate*,
decides between them: the walk crosses into the next block if the gate is in
the tree, and moves along the side of the cell's block if not.

A body is *certified* when the walk around some spanning tree holds it in
order. That is the case exactly when every move of the body is one of its
cell's two moves, the gates its crossings need form a forest, no gate is
needed both in and out, and the gates its side moves keep out leave the blocks
connected: the needed gates then extend to a spanning tree without the others
(:func:`spanning`). A snake on such a walk wins for certain. Every cell from
just ahead of the head round to the tail is free, so the next fruit is among
them, and eating it on the way leaves the body on the same walk. The check is
one pass over the body and one union-find over the blocks (:func:`certify`).

Following one walk is a fixed cycle's pace. Speed comes from changing the walk
from fruit to fruit: a path to the fruit is as safe as the walk when its
afterstate is certified too, and :func:`shortcuts` searches such paths,
shortest first.

The searches run in the compiled kernels of :mod:`snake_agent.fast`, over the
layout's tables (:func:`tables`); the functions here take and return plain
lists.
"""

from __future__ import annotations

import functools
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from snake_game import GameState

from . import fast


@dataclass(frozen=True)
class Layout:
    """The blocks of a board with even sides and the gates of its cells, for one orientation."""

    width: int
    height: int
    orientation: int
    """0: the walk keeps the tree on its left (anticlockwise on the screen); 1: on its right."""
    blocks: int
    ends: tuple[tuple[int, int], ...]
    """The two blocks of each edge of the blocks' grid."""
    gate: tuple[int, ...]
    """Each cell's gate, the edge that decides its move; -1 on the board's edge, where it moves along the
    side."""
    cross: tuple[int, ...]
    """The cell the walk enters when the gate is in the tree; -1 if the gate is the board's edge."""
    side: tuple[int, ...]
    """The cell the walk enters when it is not."""
    neighbours: tuple[tuple[tuple[int, int], ...], ...]
    """Each block's neighbouring blocks, with the edge to each."""
    entries: tuple[tuple[int, ...], ...]
    """The cells whose two moves can enter each cell."""
    block: tuple[int, ...]
    """Each cell's block."""


@functools.cache
def layout(width: int, height: int, orientation: int) -> Layout:
    """The layout of a ``width`` x ``height`` board for an orientation (cached); both sides must be even."""
    if width % 2 or height % 2:
        raise ValueError(f"a {width}x{height} board cannot be cut into 2x2 blocks")
    if orientation not in (0, 1):
        raise ValueError(f"orientation must be 0 or 1, got {orientation}")
    bw, bh = width // 2, height // 2
    horizontal = bh * (bw - 1)
    ends: list[tuple[int, int]] = []
    for j in range(bh):
        ends.extend((j * bw + i, j * bw + i + 1) for i in range(bw - 1))
    for j in range(bh - 1):
        ends.extend((j * bw + i, (j + 1) * bw + i) for i in range(bw))

    def left(i: int, j: int) -> int:
        return j * (bw - 1) + i - 1 if i > 0 else -1

    def right(i: int, j: int) -> int:
        return j * (bw - 1) + i if i < bw - 1 else -1

    def up(i: int, j: int) -> int:
        return horizontal + (j - 1) * bw + i if j > 0 else -1

    def down(i: int, j: int) -> int:
        return horizontal + j * bw + i if j < bh - 1 else -1

    cells = width * height
    gate, cross, side = [-1] * cells, [-1] * cells, [-1] * cells
    for y in range(height):
        for x in range(width):
            i, j = x // 2, y // 2
            top, west = y % 2 == 0, x % 2 == 0
            if orientation == 0:  # inside a block: top-left, bottom-left, bottom-right, top-right
                if top and west:
                    g, over, along = left(i, j), (x - 1, y), (x, y + 1)
                elif west:
                    g, over, along = down(i, j), (x, y + 1), (x + 1, y)
                elif not top:
                    g, over, along = right(i, j), (x + 1, y), (x, y - 1)
                else:
                    g, over, along = up(i, j), (x, y - 1), (x - 1, y)
            elif top and west:  # top-left, top-right, bottom-right, bottom-left
                g, over, along = up(i, j), (x, y - 1), (x + 1, y)
            elif top:
                g, over, along = right(i, j), (x + 1, y), (x, y + 1)
            elif not west:
                g, over, along = down(i, j), (x, y + 1), (x - 1, y)
            else:
                g, over, along = left(i, j), (x - 1, y), (x, y - 1)
            cell = y * width + x
            gate[cell] = g
            cross[cell] = over[1] * width + over[0] if g >= 0 else -1
            side[cell] = along[1] * width + along[0]
    neighbours: list[list[tuple[int, int]]] = [[] for _ in range(bw * bh)]
    for e, (a, b) in enumerate(ends):
        neighbours[a].append((b, e))
        neighbours[b].append((a, e))
    entries: list[list[int]] = [[] for _ in range(cells)]
    for cell in range(cells):
        for nxt in (cross[cell], side[cell]):
            if nxt >= 0:
                entries[nxt].append(cell)
    return Layout(
        width,
        height,
        orientation,
        bw * bh,
        tuple(ends),
        tuple(gate),
        tuple(cross),
        tuple(side),
        tuple(tuple(n) for n in neighbours),
        tuple(tuple(e) for e in entries),
        tuple((c // width // 2) * bw + c % width // 2 for c in range(cells)),
    )


class Tables(NamedTuple):
    """A layout as arrays, for the kernels (:mod:`snake_agent.fast`)."""

    gate: np.ndarray
    cross: np.ndarray
    side: np.ndarray
    block: np.ndarray
    ends: np.ndarray
    """``(edges, 2)``."""
    bnbr: np.ndarray
    """``(blocks, 4)``: each block's neighbouring blocks, -1 where there are fewer."""
    bedge: np.ndarray
    """``(blocks, 4)``: the edge to each of them."""
    entries: np.ndarray
    """``(cells, 2)``: the cells whose moves can enter each cell, -1 where there are fewer."""
    ring: np.ndarray
    """``(cells, 8)``: each cell's eight neighbours round the ring from the one above (:func:`_ring`)."""
    nbr: np.ndarray
    blocks: int


@functools.cache
def _tables(width: int, height: int, orientation: int) -> Tables:
    lay = layout(width, height, orientation)

    def padded(rows, size):
        return np.array([list(row) + [-1] * (size - len(row)) for row in rows], dtype=np.int64)

    return Tables(
        np.array(lay.gate, dtype=np.int64),
        np.array(lay.cross, dtype=np.int64),
        np.array(lay.side, dtype=np.int64),
        np.array(lay.block, dtype=np.int64),
        np.array(lay.ends, dtype=np.int64).reshape(len(lay.ends), 2),
        padded([[b for b, _ in n] for n in lay.neighbours], 4),
        padded([[e for _, e in n] for n in lay.neighbours], 4),
        padded(lay.entries, 2),
        np.array(_ring(width, height), dtype=np.int64).reshape(width * height, 8),
        fast.grid(width, height),
        lay.blocks,
    )


def tables(lay: Layout) -> Tables:
    """The layout's tables (cached)."""
    return _tables(lay.width, lay.height, lay.orientation)


def _cells(cells: Sequence[int]) -> np.ndarray:
    return np.asarray(cells, dtype=np.int64)


def _need(lay: Layout, need: dict[int, bool]) -> np.ndarray:
    """``need`` as the kernels' decisions per gate."""
    array = np.full(len(lay.ends), -1, dtype=np.int8)
    for e, inside in need.items():
        array[e] = inside
    return array


def _generator(rng: random.Random) -> np.ndarray:
    return fast.generator(rng.getrandbits(32))


def flat_body(state: GameState) -> list[int]:
    """The body's cells as flat indices, head first."""
    width = state.width
    return [y * width + x for x, y in state.snake]


def requirements(lay: Layout, body: Sequence[int]) -> dict[int, bool] | None:
    """The gates that the moves of ``body`` (flat cells, head first) decide: in the tree (True) or out.

    None if a move is not one of its cell's two moves, or two moves disagree about a gate.
    """
    t = tables(lay)
    ok, need = fast.requirements(t.gate, t.cross, t.side, _cells(body), len(lay.ends))
    return {int(e): bool(need[e]) for e in np.flatnonzero(need >= 0)} if ok else None


def spanning(lay: Layout, need: dict[int, bool], rng: random.Random | None = None) -> list[bool] | None:
    """A spanning tree of the blocks, as membership per edge, with the gates ``need`` puts in and without
    the ones it keeps out; None if there is none. The other edges join in order, or at random with
    ``rng``."""
    t = tables(lay)
    state = fast.generator(1) if rng is None else _generator(rng)
    ok, tree = fast.spanning(t.ends, t.blocks, _need(lay, need), rng is not None, state)
    return tree.tolist() if ok else None


def certified(lay: Layout, body: Sequence[int]) -> bool:
    """Whether the walk around some spanning tree of ``lay`` holds ``body`` (flat cells, head first)."""
    t = tables(lay)
    return bool(fast.certified(t.gate, t.cross, t.side, t.ends, t.blocks, _cells(body)))


def certify(state: GameState) -> Layout | None:
    """The layout whose walks can hold the body in order, or None (also for a board with an odd side).

    A body that has moved fits at most one orientation, since the two run every row and column opposite
    ways; a body of one cell fits both, and orientation 0 is returned.
    """
    if state.width % 2 or state.height % 2:
        return None
    body = fast.flat(state)
    for orientation in (0, 1):
        lay = layout(state.width, state.height, orientation)
        t = tables(lay)
        if fast.certified(t.gate, t.cross, t.side, t.ends, t.blocks, body):
            return lay
    return None


def successor(lay: Layout, tree: Sequence[bool], cell: int) -> int:
    """The cell after ``cell`` on the walk around ``tree``."""
    e = lay.gate[cell]
    return lay.cross[cell] if e >= 0 and tree[e] else lay.side[cell]


def walk(lay: Layout, tree: Sequence[bool], start: int, stop: int) -> list[int]:
    """The cells the walk around ``tree`` enters from ``start`` until it enters ``stop``."""
    t = tables(lay)
    path = fast.walk(t.gate, t.cross, t.side, np.asarray(tree, dtype=np.bool_), start, stop)
    if not len(path):
        raise ValueError("the walk does not reach the stop: the tree does not span the blocks")
    return path.tolist()


def after_body(body: Sequence[int], path: Sequence[int]) -> list[int]:
    """The body (flat, head first) after following ``path`` from ``body`` and eating at its last cell."""
    length = len(body) + 1
    entered = list(reversed(path))
    return entered[:length] if len(entered) >= length else entered + list(body[: length - len(entered)])


def follow(state: GameState, lay: Layout, rng: random.Random | None = None) -> list[int]:
    """The path to the fruit along the walk around a spanning tree that holds the certified body."""
    body = flat_body(state)
    need = requirements(lay, body)
    tree = None if need is None else spanning(lay, need, rng)
    if tree is None:
        raise ValueError("the body is not certified for this layout")
    width = state.width
    return walk(lay, tree, body[0], state.fruit[1] * width + state.fruit[0])


def field_arrival(state: GameState, lay: Layout) -> list[int]:
    """When the head can first enter each cell by the layout's moves, without eating; -1 if never.

    As :func:`~snake_agent.survival.arrival_times`, but each cell offers only its two moves.
    """
    t = tables(lay)
    body = fast.flat(state)
    return fast.field_arrival(fast.release_of(body, state.num_cells), t.cross, t.side, body[0]).tolist()


@functools.cache
def _ring(width: int, height: int) -> tuple[tuple[int, ...], ...]:
    """Each cell's eight neighbours round the ring from the one above, clockwise; -1 off the board."""
    steps = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
    return tuple(
        tuple(
            (y + dy) * width + x + dx if 0 <= x + dx < width and 0 <= y + dy < height else -1
            for dx, dy in steps
        )
        for y in range(height)
        for x in range(width)
    )


def shortcuts(
    state: GameState,
    lay: Layout,
    bound: int,
    rng: random.Random,
    count: int = 4,
    budget: int = 20_000,
) -> list[list[int]]:
    """Paths to the fruit of fewer than ``bound`` moves whose afterstates are certified, shortest first.

    An iterative-deepening search over the layout's moves, through free cells and body cells vacated in
    time, never over the fruit before the end. A path of a given length leaves a known part of the body
    in place, so the search keeps the gates decided by that part and by the path, and cuts a branch as
    soon as they contradict each other, close a loop of crossings or cut the blocks apart, or as soon as
    the path splits the cells that will be free after eating into pieces it cannot finish (a certified
    body leaves them in one run). Each complete path is checked exactly. The search stops after ``count``
    paths or ``budget`` steps.
    """
    t = tables(lay)
    width = state.width
    body = fast.flat(state)
    paths, lengths, _, found = fast.shortcuts(
        t.gate,
        t.cross,
        t.side,
        t.ends,
        t.bnbr,
        t.bedge,
        t.entries,
        t.ring,
        t.nbr,
        t.blocks,
        body,
        state.fruit[1] * width + state.fruit[0],
        bound,
        count,
        budget,
        len(body),
        False,
        _generator(rng),
    )
    return [paths[i, : lengths[i]].tolist() for i in range(found)]


def greedy_walk(
    lay: Layout, need: dict[int, bool], start: int, stop: int, rng: random.Random | None = None
) -> list[int] | None:
    """A walk from ``start`` to ``stop`` around a spanning tree that keeps the gates of ``need``, built
    gate by gate as the walk meets them; None if the gates left no way.

    At a gate not decided yet the walk crosses into the next block if that block is nearer the stop's
    block and crossing closes no loop, and otherwise moves along its block's side, unless keeping the
    gate out would cut the blocks apart. With ``rng``, distances between blocks are blurred at random,
    so repeated calls try other routes.
    """
    t = tables(lay)
    state = fast.generator(1) if rng is None else _generator(rng)
    path = fast.greedy_walk(
        t.gate,
        t.cross,
        t.side,
        t.ends,
        t.bnbr,
        t.bedge,
        t.block,
        t.blocks,
        _need(lay, need),
        start,
        stop,
        rng is not None,
        state,
    )
    return path.tolist() if len(path) else None


def distance(lay: Layout, tree: Sequence[bool], start: int, stop: int, cap: int) -> int:
    """The moves of the walk around ``tree`` from ``start`` to ``stop``, or ``cap`` if that is no fewer."""
    t = tables(lay)
    return int(fast.distance(t.gate, t.cross, t.side, np.asarray(tree, dtype=np.bool_), start, stop, cap))


def prefix(lay: Layout, body: Sequence[int]) -> int:
    """How many cells at the head end of ``body`` (flat, head first) moved by the layout's moves."""
    t = tables(lay)
    return int(fast.prefix(t.cross, t.side, _cells(body)))


def field_paths(state: GameState, lay: Layout, rng: random.Random, count: int) -> list[list[int]]:
    """Up to ``count`` shortest paths to the fruit by the layout's moves (:func:`field_arrival`), their
    ties broken at random; none if the fruit cannot be reached that way."""
    t = tables(lay)
    body = fast.flat(state)
    arrival = fast.field_arrival(fast.release_of(body, state.num_cells), t.cross, t.side, body[0])
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    if arrival[fruit] <= 0:
        return []
    return fast.field_paths(arrival, t.entries, fruit, count, _generator(rng)).tolist()


def timely(state: GameState, path: Sequence[int]) -> bool:
    """Whether following ``path`` is safe: each cell is free when the head enters it, the first move is no
    reversal, and the fruit comes last."""
    body = fast.flat(state)
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    return bool(fast.timely(fast.release_of(body, state.num_cells), body, _cells(path), fruit))


def certified_paths(
    state: GameState,
    lay: Layout,
    rng: random.Random,
    trees: int = 64,
    count: int = 8,
    budget: int = 20_000,
) -> list[list[int]]:
    """Paths to the fruit that keep a certified body certified: the shortest walk around ``trees`` random
    spanning trees that hold the body, the greedy walk (:func:`greedy_walk`), and up to ``count``
    shortcuts shorter than both (:func:`shortcuts`, at most ``budget`` steps of search)."""
    t = tables(lay)
    body = fast.flat(state)
    ok, need = fast.requirements(t.gate, t.cross, t.side, body, len(lay.ends))
    if not ok:
        raise ValueError("the body is not certified for this layout")
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    head = int(body[0])
    ok, best = fast.best_tree(
        t.gate, t.cross, t.side, t.ends, t.blocks, need, head, fruit, trees, _generator(rng)
    )
    if not ok:
        raise ValueError("the body is not certified for this layout")
    paths = [fast.walk(t.gate, t.cross, t.side, best, head, fruit).tolist()]
    greedy = fast.greedy_walk(
        t.gate,
        t.cross,
        t.side,
        t.ends,
        t.bnbr,
        t.bedge,
        t.block,
        t.blocks,
        need,
        head,
        fruit,
        False,
        _generator(rng),
    )
    if len(greedy):
        paths.append(greedy.tolist())
    found, lengths, _, n = fast.shortcuts(
        t.gate,
        t.cross,
        t.side,
        t.ends,
        t.bnbr,
        t.bedge,
        t.entries,
        t.ring,
        t.nbr,
        t.blocks,
        body,
        fruit,
        min(len(p) for p in paths),
        count,
        budget,
        len(body),
        False,
        _generator(rng),
    )
    return paths + [found[i, : lengths[i]].tolist() for i in range(n)]


def transition_paths(
    state: GameState,
    rng: random.Random,
    trees: int = 64,
    count: int = 8,
    searched: int = 8,
    budget: int = 20_000,
) -> tuple[list[list[int]], list[list[int]]]:
    """Safe paths to the fruit toward a certified body, from a body that is not: ``(exact, partial)``.

    The orientation is the one that the most recent moves followed. The paths are walks around ``trees``
    random spanning trees that hold those moves, ``count`` shortest paths by the layout's moves
    (:func:`field_paths`), and up to ``searched`` of the shortest paths by the layout's moves that keep
    the gates of those moves (the search of :func:`shortcuts`, at most ``budget`` steps, for paths up to
    twice as long as the shortest). A path is *exact* when its afterstate is certified, which needs it to
    outlast the rest of the body; it is *partial* when it only lengthens the run of recent moves that a
    spanning tree can hold, so that the body is certified once the older part has gone.
    """
    if state.width % 2 or state.height % 2:
        return [], []
    body = fast.flat(state)
    runs = []  # the run of recent moves that each orientation's gates can hold
    for orientation in (0, 1):
        t = tables(layout(state.width, state.height, orientation))
        runs.append(int(fast.consistent(t.gate, t.cross, t.side, body, len(t.ends))))
    lay = layout(state.width, state.height, int(runs[1] > runs[0]))
    t = tables(lay)
    _, need = fast.requirements(t.gate, t.cross, t.side, body[: max(runs)], len(lay.ends))
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    head = int(body[0])
    release = fast.release_of(body, state.num_cells)
    generator = _generator(rng)
    tried = []
    for _ in range(trees):
        ok, tree = fast.spanning(t.ends, t.blocks, need, True, generator)
        if not ok:
            break
        tried.append(fast.walk(t.gate, t.cross, t.side, tree, head, fruit))
    arrival = fast.field_arrival(release, t.cross, t.side, head)
    if arrival[fruit] > 0:
        tried += list(fast.field_paths(arrival, t.entries, fruit, count, generator))
        if searched:
            found, lengths, _, n = fast.shortcuts(
                t.gate,
                t.cross,
                t.side,
                t.ends,
                t.bnbr,
                t.bedge,
                t.entries,
                t.ring,
                t.nbr,
                t.blocks,
                body,
                fruit,
                2 * int(arrival[fruit]) + 2,
                searched,
                budget,
                max(runs),
                True,
                generator,
            )
            tried += [found[i, : lengths[i]] for i in range(n)]
    exact, partial, seen = [], [], set()
    for path in tried:
        key = path.tobytes()
        if key in seen or not fast.timely(release, body, path, fruit):
            continue
        seen.add(key)
        after = fast.after_body(body, path)
        kept = int(fast.prefix(t.cross, t.side, after))
        if fast.certified(t.gate, t.cross, t.side, t.ends, t.blocks, after[:kept]):
            (exact if kept == len(after) else partial).append(path.tolist())
    return exact, partial
