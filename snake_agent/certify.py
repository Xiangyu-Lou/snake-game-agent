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
"""

from __future__ import annotations

import functools
import random
from collections.abc import Sequence
from dataclasses import dataclass

from snake_game import GameState

from .survival import neighbour_table


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


def flat_body(state: GameState) -> list[int]:
    """The body's cells as flat indices, head first."""
    width = state.width
    return [y * width + x for x, y in state.snake]


def requirements(lay: Layout, body: Sequence[int]) -> dict[int, bool] | None:
    """The gates that the moves of ``body`` (flat cells, head first) decide: in the tree (True) or out.

    None if a move is not one of its cell's two moves, or two moves disagree about a gate.
    """
    gate, cross, side = lay.gate, lay.cross, lay.side
    need: dict[int, bool] = {}
    for k in range(len(body) - 1, 0, -1):
        cell, nxt = body[k], body[k - 1]
        if nxt == side[cell]:
            inside = False
        elif nxt == cross[cell]:
            inside = True
        else:
            return None
        e = gate[cell]
        if e >= 0 and need.setdefault(e, inside) != inside:
            return None
    return need


def spanning(lay: Layout, need: dict[int, bool], rng: random.Random | None = None) -> list[bool] | None:
    """A spanning tree of the blocks, as membership per edge, with the gates ``need`` puts in and without
    the ones it keeps out; None if there is none. The other edges join in order, or at random with
    ``rng``."""
    parent = list(range(lay.blocks))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ends = lay.ends
    tree = [False] * len(ends)
    joined = 0
    for e, inside in need.items():
        if inside:
            a, b = find(ends[e][0]), find(ends[e][1])
            if a == b:
                return None
            parent[a] = b
            tree[e] = True
            joined += 1
    rest = [e for e in range(len(ends)) if e not in need]
    if rng is not None:
        rng.shuffle(rest)
    for e in rest:
        a, b = find(ends[e][0]), find(ends[e][1])
        if a != b:
            parent[a] = b
            tree[e] = True
            joined += 1
    return tree if joined == lay.blocks - 1 else None


def certified(lay: Layout, body: Sequence[int]) -> bool:
    """Whether the walk around some spanning tree of ``lay`` holds ``body`` (flat cells, head first)."""
    need = requirements(lay, body)
    return need is not None and spanning(lay, need) is not None


def certify(state: GameState) -> Layout | None:
    """The layout whose walks can hold the body in order, or None (also for a board with an odd side).

    A body that has moved fits at most one orientation, since the two run every row and column opposite
    ways; a body of one cell fits both, and orientation 0 is returned.
    """
    if state.width % 2 or state.height % 2:
        return None
    body = flat_body(state)
    for orientation in (0, 1):
        lay = layout(state.width, state.height, orientation)
        if certified(lay, body):
            return lay
    return None


def successor(lay: Layout, tree: Sequence[bool], cell: int) -> int:
    """The cell after ``cell`` on the walk around ``tree``."""
    e = lay.gate[cell]
    return lay.cross[cell] if e >= 0 and tree[e] else lay.side[cell]


def walk(lay: Layout, tree: Sequence[bool], start: int, stop: int) -> list[int]:
    """The cells the walk around ``tree`` enters from ``start`` until it enters ``stop``."""
    path = []
    cell = start
    for _ in range(lay.width * lay.height):
        cell = successor(lay, tree, cell)
        path.append(cell)
        if cell == stop:
            return path
    raise ValueError("the walk does not reach the stop: the tree does not span the blocks")


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
    body = flat_body(state)
    length = len(body)
    release = [0] * (lay.width * lay.height)
    for k, cell in enumerate(body):
        release[cell] = length - k
    cross, side = lay.cross, lay.side
    arrival = [-1] * len(release)
    arrival[body[0]] = 0
    frontier, t = [body[0]], 1
    while frontier:
        layer = []
        for cell in frontier:
            for nxt in (cross[cell], side[cell]):
                if nxt >= 0 and arrival[nxt] < 0 and release[nxt] <= t:
                    arrival[nxt] = t
                    layer.append(nxt)
        frontier = layer
        t += 1
    return arrival


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
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    body = flat_body(state)
    length = len(body)
    cells = width * state.height
    gate, cross, side, ends = lay.gate, lay.cross, lay.side, lay.ends
    release = [0] * cells
    for k, cell in enumerate(body):
        release[cell] = length - k
    far = cells + 1
    dist = [far] * cells
    entries = lay.entries

    def distances(limit: int) -> None:
        """Moves from each cell to the fruit for a path of ``limit`` moves, a bound for the search: a cell
        counts only if it is free by the latest move at which such a path can enter it, and a move only if
        the gates decided by the body that stays allow it."""
        for c in range(cells):
            dist[c] = far
        dist[fruit] = 0
        frontier, d = [fruit], 1
        while frontier:
            layer = []
            for cell in frontier:
                for prev in entries[cell]:
                    if dist[prev] != far or release[prev] > limit - d:
                        continue
                    e = gate[prev]
                    if e >= 0:
                        if cell == side[prev]:
                            if inside[e]:
                                continue
                        elif outside[e] or (not inside[e] and find(ends[e][0]) == find(ends[e][1])):
                            continue
                    dist[prev] = d
                    layer.append(prev)
            frontier = layer
            d += 1

    start = field_arrival(state, lay)[fruit]
    if start < 0 or start >= bound:
        return []

    edges = len(ends)
    inside = [0] * edges  # decisions per gate, from the moves kept so far
    outside = [0] * edges
    parent = list(range(lay.blocks))
    size = [1] * lay.blocks

    def find(a: int) -> int:
        while parent[a] != a:
            a = parent[a]
        return a

    def connected() -> bool:
        seen = bytearray(lay.blocks)
        seen[0] = 1
        stack, reached = [0], 1
        while stack:
            a = stack.pop()
            for b, e in lay.neighbours[a]:
                if not seen[b] and not outside[e]:
                    seen[b] = 1
                    reached += 1
                    stack.append(b)
        return reached == lay.blocks

    def decide(e: int, want: bool):
        """Record a gate decision; returns its undo record, or None (and records nothing) if it fails."""
        if want:
            if outside[e]:
                return None
            if inside[e]:
                inside[e] += 1
                return (e, True, -1)
            a, b = find(ends[e][0]), find(ends[e][1])
            if a == b:
                return None
            if size[a] > size[b]:
                a, b = b, a
            parent[a] = b
            size[b] += size[a]
            inside[e] = 1
            return (e, True, a)
        if inside[e]:
            return None
        outside[e] += 1
        if outside[e] == 1 and find(ends[e][0]) != find(ends[e][1]) and not connected():
            outside[e] -= 1
            return None
        return (e, False, -1)

    def undo(record) -> None:
        e, want, root = record
        if want:
            inside[e] -= 1
            if root >= 0:
                b = parent[root]
                parent[root] = root
                size[b] -= size[root]
        else:
            outside[e] -= 1

    found: list[list[int]] = []
    path: list[int] = []
    on = bytearray(cells)  # the cells that stay out of the free cells after eating: the body kept, the path
    steps = 0
    ring = _ring(width, state.height)
    grid = neighbour_table(width, state.height)

    def split(cell: int, left: int) -> bool:
        """Whether taking ``cell`` out of the free cells after eating leaves them unable to be one region:
        the rest of the path, at most ``left`` cells ending at the fruit, stays in the fruit's region, so
        any other region stays apart, and a second one, or one beside an unfinished fruit region, is
        fatal to the certificate. A cell whose free neighbours stay linked around it splits nothing."""
        around = [c >= 0 and not on[c] for c in ring[cell]]
        runs = 0
        for k in (0, 2, 4, 6):  # the four neighbours; neighbours linked through a corner form one run
            if around[k] and not (around[(k + 7) % 8] and around[(k + 6) % 8]):
                runs += 1
        if runs <= 1:
            return False
        seen = bytearray(cells)
        seen[cell] = 1
        regions = []
        for start_cell in grid[cell]:
            if start_cell < 0 or on[start_cell] or seen[start_cell]:
                continue
            seen[start_cell] = 1
            stack, size_, fruity = [start_cell], 1, start_cell == fruit
            while stack:
                c = stack.pop()
                for n in grid[c]:
                    if n >= 0 and not on[n] and not seen[n]:
                        seen[n] = 1
                        size_ += 1
                        fruity |= n == fruit
                        stack.append(n)
            regions.append((fruity, size_))
        fruit_region = sum(s for fruity, s in regions if fruity)
        others = sum(1 for fruity, _ in regions if not fruity)
        if not fruit_region:
            return True  # the fruit is out of reach
        return others >= 2 or (others == 1 and fruit_region > left)

    def separate(limit: int) -> bool:
        """Whether the free cells after eating are split before the path starts, as in :func:`split`."""
        seen = bytearray(cells)
        regions = []
        for first in range(cells):
            if on[first] or seen[first]:
                continue
            seen[first] = 1
            stack, size_, fruity = [first], 1, first == fruit
            while stack:
                c = stack.pop()
                for n in grid[c]:
                    if n >= 0 and not on[n] and not seen[n]:
                        seen[n] = 1
                        size_ += 1
                        fruity |= n == fruit
                        stack.append(n)
            regions.append((fruity, size_))
        fruit_region = sum(s for fruity, s in regions if fruity)
        others = sum(1 for fruity, _ in regions if not fruity)
        return others >= 2 or (others == 1 and fruit_region > limit)

    def search(cell: int, t: int, limit: int) -> None:
        nonlocal steps
        options = []
        for nxt, want in ((cross[cell], True), (side[cell], False)):
            if nxt < 0 or on[nxt] or release[nxt] > t + 1:
                continue
            if nxt == fruit:
                if t + 1 == limit:
                    full = path + [fruit]
                    if certified(lay, after_body(body, full)):
                        found.append(full)
                continue
            if t + 1 + dist[nxt] > limit:
                continue
            options.append((dist[nxt], rng.random(), nxt, want))
        options.sort()
        for _, _, nxt, want in options:
            if len(found) >= count or steps >= budget:
                return
            steps += 1
            e = gate[cell]
            record = decide(e, want) if e >= 0 else (-1, False, -1)
            if record is None:
                continue
            if not split(nxt, limit - t - 1):
                path.append(nxt)
                on[nxt] = 1
                search(nxt, t + 1, limit)
                on[nxt] = 0
                path.pop()
            if e >= 0:
                undo(record)

    # A path of exactly ``limit`` moves leaves body cells 0 .. length - limit in place, with the moves of
    # cells 1 .. length - limit. They are decided from the head back, so the longer limits undo the last.
    kept: list = []
    for k in range(1, length - start + 1):
        cell = body[k]
        e = gate[cell]
        record = decide(e, body[k - 1] == cross[cell]) if e >= 0 else (-1, False, -1)
        if record is None:
            return found  # not certified to begin with
        kept.append(record)
    for limit in range(start, bound, 2):
        while len(kept) > max(length - limit, 0):
            record = kept.pop()
            if record[0] >= 0:
                undo(record)
        for k, cell in enumerate(body):
            on[cell] = k <= length - limit
        if separate(limit):
            continue
        distances(limit)
        search(body[0], 0, limit)
        if len(found) >= count or steps >= budget:
            break
    return found


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
    gate, cross, side, ends, neighbours = lay.gate, lay.cross, lay.side, lay.ends, lay.neighbours
    decided = dict(need)
    blocks = lay.blocks
    parent = list(range(blocks))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e, inside in decided.items():
        if inside:
            a, b = find(ends[e][0]), find(ends[e][1])
            if a == b:
                return None
            parent[a] = b
    # Distances between blocks to the stop's block, over the edges that may join the tree.
    goal = lay.block[stop]
    far = 1 << 30
    toward = [far] * blocks
    toward[goal] = 0
    frontier = [goal]
    while frontier:
        layer = []
        for a in frontier:
            for b, e in neighbours[a]:
                if toward[b] == far and decided.get(e) is not False:
                    toward[b] = toward[a] + 1
                    layer.append(b)
        frontier = layer
    blur = [rng.random() for _ in range(blocks)] if rng is not None else [0.0] * blocks

    def joined_without(e: int) -> bool:
        """Whether the blocks stay connected when ``e`` is kept out too."""
        seen = bytearray(blocks)
        seen[0] = 1
        stack, reached = [0], 1
        while stack:
            a = stack.pop()
            for b, f in neighbours[a]:
                if not seen[b] and f != e and decided.get(f) is not False:
                    seen[b] = 1
                    reached += 1
                    stack.append(b)
        return reached == blocks

    path = []
    cell = start
    for _ in range(lay.width * lay.height):
        e = gate[cell]
        if e >= 0 and e not in decided:
            here, there = lay.block[cell], lay.block[cross[cell]]
            loops = find(here) == find(there)
            if not loops and toward[there] + blur[there] < toward[here] + blur[here]:
                decided[e] = True
            elif joined_without(e):
                decided[e] = False
            elif not loops:
                decided[e] = True
            else:
                return None
            if decided[e]:
                a, b = find(here), find(there)
                parent[a] = b
        cell = cross[cell] if e >= 0 and decided[e] else side[cell]
        path.append(cell)
        if cell == stop:
            return path
    return None


def distance(lay: Layout, tree: Sequence[bool], start: int, stop: int, cap: int) -> int:
    """The moves of the walk around ``tree`` from ``start`` to ``stop``, or ``cap`` if that is no fewer."""
    gate, cross, side = lay.gate, lay.cross, lay.side
    cell = start
    for moves in range(1, cap):
        e = gate[cell]
        cell = cross[cell] if e >= 0 and tree[e] else side[cell]
        if cell == stop:
            return moves
    return cap


def prefix(lay: Layout, body: Sequence[int]) -> int:
    """How many cells at the head end of ``body`` (flat, head first) moved by the layout's moves."""
    cross, side = lay.cross, lay.side
    m = 1
    while m < len(body) and body[m - 1] in (cross[body[m]], side[body[m]]):
        m += 1
    return m


def field_paths(state: GameState, lay: Layout, rng: random.Random, count: int) -> list[list[int]]:
    """Up to ``count`` shortest paths to the fruit by the layout's moves (:func:`field_arrival`), their
    ties broken at random; none if the fruit cannot be reached that way."""
    arrival = field_arrival(state, lay)
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    if arrival[fruit] <= 0:
        return []
    paths = []
    for _ in range(count):
        cell, path = fruit, [fruit]
        while arrival[cell] > 1:
            cell = rng.choice([c for c in lay.entries[cell] if arrival[c] == arrival[cell] - 1])
            path.append(cell)
        paths.append(path[::-1])
    return paths


def timely(state: GameState, path: Sequence[int]) -> bool:
    """Whether following ``path`` is safe: each cell is free when the head enters it, the first move is no
    reversal, and the fruit comes last."""
    body = flat_body(state)
    length = len(body)
    if length > 1 and path and path[0] == body[1]:
        return False  # the engine would go straight on
    release = {cell: length - k for k, cell in enumerate(body)}
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    seen = set()
    for t, cell in enumerate(path, 1):
        if release.get(cell, 0) > t or cell in seen or (cell == fruit) != (t == len(path)):
            return False
        seen.add(cell)
    return True


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
    body = flat_body(state)
    need = requirements(lay, body)
    if need is None:
        raise ValueError("the body is not certified for this layout")
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    best, cap = None, state.num_cells + 1
    for _ in range(max(trees, 1)):
        tree = spanning(lay, need, rng)
        if tree is None:
            raise ValueError("the body is not certified for this layout")
        moves = distance(lay, tree, body[0], fruit, cap)
        if moves < cap:
            best, cap = tree, moves
    paths = [walk(lay, best, body[0], fruit)]
    greedy = greedy_walk(lay, need, body[0], fruit)
    if greedy is not None:
        paths.append(greedy)
    return paths + shortcuts(state, lay, min(len(p) for p in paths), rng, count, budget)


def transition_paths(
    state: GameState, rng: random.Random, trees: int = 64, count: int = 8
) -> tuple[list[list[int]], list[list[int]]]:
    """Safe paths to the fruit toward a certified body, from a body that is not: ``(exact, partial)``.

    The orientation is the one that the most recent moves followed. The paths are walks around ``trees``
    random spanning trees that hold those moves, and ``count`` shortest paths by the layout's moves
    (:func:`field_paths`). A path is *exact* when its afterstate is certified, which needs it to outlast
    the rest of the body; it is *partial* when it only lengthens the run of recent moves that a spanning
    tree can hold, so that the body is certified once the older part has gone.
    """
    if state.width % 2 or state.height % 2:
        return [], []
    body = flat_body(state)
    lay = max((layout(state.width, state.height, o) for o in (0, 1)), key=lambda lay: prefix(lay, body))
    need = requirements(lay, body[: prefix(lay, body)])
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    tried = []
    for _ in range(trees):
        tree = spanning(lay, need, rng)
        if tree is None:
            break
        tried.append(walk(lay, tree, body[0], fruit))
    tried += field_paths(state, lay, rng, count)
    exact, partial, seen = [], [], set()
    for path in tried:
        key = tuple(path)
        if key in seen or not timely(state, path):
            continue
        seen.add(key)
        after = after_body(body, path)
        kept = prefix(lay, after)
        if certified(lay, after[:kept]):
            (exact if kept == len(after) else partial).append(path)
    return exact, partial
