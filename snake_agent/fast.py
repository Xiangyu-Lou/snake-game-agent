"""Compiled kernels: the plan agent's searches and its certificate, over flat arrays, for large boards.

On a 100x100 board the plan agent spends a quarter to half a second per decision in Python loops: the
searches for when the head can reach each cell, survival certificates, the encoder's planes, and the
spanning trees of the certificate (``snake_agent/README.md``, 100x100). These kernels do the same work
over NumPy arrays. With Numba (``pip install -e ".[fast]"``) they are compiled on first use and cached
on disk; without it they run as plain Python, the same code giving the same results, only slower.

Conventions:

* cells are flat indices ``y * width + x``; ``nbr[cell, d]`` is the neighbour in direction ``d``
  (``UP, RIGHT, DOWN, LEFT``), or -1 off the board (:func:`grid`);
* ``release[cell]`` is the number of moves until the body leaves ``cell`` if nothing is eaten:
  ``L - k`` on segment ``k`` (``L`` on the head, 1 on the tail) and 0 on a free cell, which is
  :meth:`~snake_agent.survival.Board.release` of every cell at once (:func:`release_of`);
* random choices come from a small xorshift generator held in a one-element array (:func:`generator`),
  so that compiled and plain runs draw the same numbers;
* the certificate's kernels take the layout's tables (:func:`snake_agent.certify.tables`): each cell's
  ``gate``, ``cross`` and ``side``, the two blocks at the ``ends`` of each edge, each block's
  neighbours ``bnbr`` across the edges ``bedge``, and the ``entries`` of each cell; ``need`` holds a
  decision per gate, 1 in the tree, 0 out of it, -1 undecided.
"""

from __future__ import annotations

import functools

import numpy as np

from snake_game import GameState

from .survival import neighbour_table

try:
    import numba
except ImportError:  # the plain-Python fallback: the same code, interpreted
    numba = None

COMPILED = numba is not None and not numba.config.DISABLE_JIT
"""Whether the kernels are compiled (Numba is installed and not disabled with ``NUMBA_DISABLE_JIT``)."""

UNSEEN = -(1 << 30)
""":func:`reach`'s slack of a body cell next to no reachable cell."""


def kernel(function):
    """``function`` compiled by Numba (cached on disk), or unchanged without Numba."""
    if numba is None:
        return function
    return numba.njit(cache=True, nogil=True)(function)


@functools.cache
def grid(width: int, height: int) -> np.ndarray:
    """``nbr[cell, d]``: the cell next to ``cell`` in direction ``d``, or -1 off the board."""
    return np.array(neighbour_table(width, height), dtype=np.int64).reshape(width * height, 4)


def flat(state: GameState) -> np.ndarray:
    """The body as flat cells, head first."""
    width = state.width
    return np.array([y * width + x for x, y in state.snake], dtype=np.int64)


def release_of(body: np.ndarray, cells: int) -> np.ndarray:
    """``release`` of a body given as flat cells, head first."""
    release = np.zeros(cells, dtype=np.int64)
    release[body] = np.arange(len(body), 0, -1)
    return release


def generator(seed: int) -> np.ndarray:
    """The kernels' random generator, seeded with ``seed`` (any integer)."""
    return np.array([(seed & 0xFFFFFFFF) or 1], dtype=np.int64)


# ------------------------------------------------------------------ randomness
@kernel
def _next(state):
    """The next number of the xorshift generator, in ``[1, 2^32)``."""
    x = state[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    state[0] = x
    return x


@kernel
def below(state, n):
    """A random integer in ``[0, n)``."""
    return _next(state) % n


@kernel
def uniform(state):
    """A random number in ``(0, 1)``."""
    return _next(state) / 4294967296.0


@kernel
def direction(width, cell, nxt):
    """The absolute direction from ``cell`` to the neighbouring ``nxt``."""
    if nxt == cell - width:
        return 0
    if nxt == cell + 1:
        return 1
    if nxt == cell + width:
        return 2
    return 3


# --------------------------------------------------------- searches over cells
@kernel
def arrival(release, nbr, head, heading, avoid):
    """:func:`~snake_agent.survival.arrival_times`: the earliest move at which the head can enter each
    cell, through free cells and cells vacated in time, never entering ``avoid``; -1 if never."""
    cells = release.shape[0]
    time = np.full(cells, -1, np.int64)
    time[head] = 0
    reverse = (heading + 2) % 4
    frontier = np.empty(cells, np.int64)
    layer = np.empty(cells, np.int64)
    frontier[0] = head
    size, t = 1, 1
    while size:
        count = 0
        for i in range(size):
            cell = frontier[i]
            for d in range(4):
                nxt = nbr[cell, d]
                if nxt < 0 or time[nxt] >= 0 or nxt == avoid or (cell == head and d == reverse):
                    continue
                if release[nxt] <= t:
                    time[nxt] = t
                    layer[count] = nxt
                    count += 1
        frontier, layer = layer, frontier
        size = count
        t += 1
    return time


@kernel
def opening(release, nbr, head, heading):
    """:func:`~snake_agent.survival.opening_times`: when the head could first enter each cell if it could
    wait where it stands, a body cell entered at ``max(t, release)``."""
    cells = release.shape[0]
    time = np.full(cells, -1, np.int64)
    time[head] = 0
    reverse = (heading + 2) % 4
    heap = np.empty(4 * cells + 1, np.int64)  # keys time * cells + cell
    heap[0] = head
    size = 1
    while size:
        key = heap[0]
        size -= 1
        if size:
            last, i = heap[size], 0
            while True:
                child = 2 * i + 1
                if child >= size:
                    break
                if child + 1 < size and heap[child + 1] < heap[child]:
                    child += 1
                if heap[child] >= last:
                    break
                heap[i] = heap[child]
                i = child
            heap[i] = last
        t, cell = key // cells, key % cells
        if time[cell] != t:
            continue
        for d in range(4):
            nxt = nbr[cell, d]
            if nxt < 0 or (cell == head and d == reverse):
                continue
            when = max(t + 1, release[nxt])
            if time[nxt] < 0 or when < time[nxt]:
                time[nxt] = when
                j, entry = size, when * cells + nxt
                size += 1
                while j > 0:
                    parent = (j - 1) // 2
                    if heap[parent] <= entry:
                        break
                    heap[j] = heap[parent]
                    j = parent
                heap[j] = entry
    return time


@kernel
def free_mean(times, release):
    """The mean of ``times`` over the free cells (0 if there are none)."""
    total, count = 0.0, 0
    for c in range(times.shape[0]):
        if release[c] == 0:
            total += times[c]
            count += 1
    return total / count if count else 0.0


@kernel
def persistence(times, release, unit):
    """The mean over the free cells of the waits that the fruits after the next would spend on them, if
    the fruits came every ``unit`` moves and each cell opened at ``times``: ``sum over j >= 1 of
    (times - j * unit)`` while positive, the wait of a fruit that appears there ``j`` fruits later."""
    total, count = 0.0, 0
    for c in range(times.shape[0]):
        if release[c] == 0:
            count += 1
            x = float(times[c])
            if x > unit:
                j = np.ceil(x / unit) - 1.0
                total += j * x - unit * j * (j + 1.0) / 2.0
    return total / count if count else 0.0


@kernel
def certificate(release, nbr, head, heading):
    """:meth:`~snake_agent.survival.Board.has_survival_certificate`: whether a shortest path through free
    cells brings the head next to a segment that has moved away by the time it arrives."""
    cells = release.shape[0]
    seen = np.zeros(cells, np.bool_)
    seen[head] = True
    reverse = (heading + 2) % 4
    frontier = np.empty(cells, np.int64)
    layer = np.empty(cells, np.int64)
    frontier[0] = head
    size, t = 1, 1
    while size:
        count = 0
        for i in range(size):
            cell = frontier[i]
            for d in range(4):
                nxt = nbr[cell, d]
                if nxt < 0 or (cell == head and d == reverse):
                    continue
                r = release[nxt]
                if r > 0:
                    if r <= t:
                        return True
                elif not seen[nxt]:
                    seen[nxt] = True
                    layer[count] = nxt
                    count += 1
        frontier, layer = layer, frontier
        size = count
        t += 1
    return False


@kernel
def certificate_loop(release, body, nbr, head, heading):
    """:meth:`~snake_agent.survival.Board.certificate_loop`: the closed loop behind the certificate as
    the cells to enter in order (through free cells into a vacated segment, then along the body back to
    the head's cell); empty without a certificate."""
    cells = release.shape[0]
    parent = np.full(cells, -2, np.int64)
    parent[head] = -1
    reverse = (heading + 2) % 4
    frontier = np.empty(cells, np.int64)
    layer = np.empty(cells, np.int64)
    frontier[0] = head
    size, t = 1, 1
    length = body.shape[0]
    while size:
        count = 0
        for i in range(size):
            cell = frontier[i]
            for d in range(4):
                nxt = nbr[cell, d]
                if nxt < 0 or (cell == head and d == reverse):
                    continue
                r = release[nxt]
                if r > 0:
                    if r <= t:
                        k = length - r  # the segment the loop enters
                        steps = 0
                        c = cell
                        while c != head:
                            steps += 1
                            c = parent[c]
                        loop = np.empty(steps + k + 1, np.int64)
                        c = cell
                        for j in range(steps - 1, -1, -1):
                            loop[j] = c
                            c = parent[c]
                        for j in range(k + 1):
                            loop[steps + j] = body[k - j]
                        return loop
                elif parent[nxt] == -2:
                    parent[nxt] = cell
                    layer[count] = nxt
                    count += 1
        frontier, layer = layer, frontier
        size = count
        t += 1
    return np.empty(0, np.int64)


@kernel
def reach(release, nbr, head, heading):
    """The encoder's search: the distance from the head of each free cell it can reach through free
    cells (-1 if none), and for each body cell next to that region its slack, the latest arrival beside
    it minus its release (:data:`UNSEEN` elsewhere)."""
    cells = release.shape[0]
    distance = np.full(cells, -1, np.int64)
    slack = np.full(cells, UNSEEN, np.int64)
    distance[head] = 0
    reverse = (heading + 2) % 4
    frontier = np.empty(cells, np.int64)
    layer = np.empty(cells, np.int64)
    frontier[0] = head
    size, step = 1, 1
    while size:
        count = 0
        for i in range(size):
            cell = frontier[i]
            for d in range(4):
                nxt = nbr[cell, d]
                if nxt < 0 or (cell == head and d == reverse):
                    continue
                r = release[nxt]
                if r:
                    if step - r > slack[nxt]:
                        slack[nxt] = step - r
                elif distance[nxt] < 0:
                    distance[nxt] = step
                    layer[count] = nxt
                    count += 1
        frontier, layer = layer, frontier
        size = count
        step += 1
    return distance, slack


@kernel
def move_classes(release, nbr, head, heading, fruit, length):
    """:func:`~snake_agent.survival.classify_moves` without a proof search: per relative action (``LEFT``,
    ``STRAIGHT``, ``RIGHT``) 0 if it collides, 1 if it fills the board, 2 if the next state has a
    survival certificate, 3 otherwise."""
    cells = release.shape[0]
    classes = np.empty(3, np.int64)
    moved = np.empty(cells, np.int64)
    for a in range(3):
        d = (heading + a + 3) % 4
        cell = nbr[head, d]
        if cell < 0 or release[cell] > 1:
            classes[a] = 0
            continue
        if cell == fruit:
            if length + 1 == cells:
                classes[a] = 1
                continue
            for c in range(cells):
                moved[c] = release[c]
            moved[cell] = length + 1
        else:
            for c in range(cells):
                moved[c] = release[c] - 1 if release[c] > 1 else 0
            moved[cell] = length
        classes[a] = 2 if certificate(moved, nbr, cell, d) else 3
    return classes


# ------------------------------------------------------------------ the plans
@kernel
def shortest_paths(arrival, nbr, goal, count, state):
    """``count`` shortest paths from the head to ``goal`` by ``arrival`` (the cells entered, one path per
    row), ties between predecessors broken at random."""
    n = arrival[goal]
    paths = np.empty((count, n), np.int64)
    options = np.empty(4, np.int64)
    for p in range(count):
        cell = goal
        paths[p, n - 1] = goal
        for i in range(n - 2, -1, -1):
            t = arrival[cell] - 1
            m = 0
            for d in range(4):
                c = nbr[cell, d]
                if c >= 0 and arrival[c] == t:
                    options[m] = c
                    m += 1
            cell = options[0] if m == 1 else options[below(state, m)]
            paths[p, i] = cell
    return paths


@kernel
def hugging_path(arrival, release, nbr, goal, state):
    """A shortest path from the head to ``goal`` by ``arrival`` that runs beside the body and the board's
    edge as much as it can: the most sides of its cells blocked while the head is there (by the board's
    edge, or by a body cell not yet vacated), ties broken at random."""
    cells = arrival.shape[0]
    n = arrival[goal]
    order = np.argsort(arrival)
    best = np.full(cells, -1, np.int64)
    for i in range(cells):
        cell = order[i]
        t = arrival[cell]
        if t < 0:
            continue
        if t > n:
            break
        if t == 0:
            best[cell] = 0
            continue
        top = -1
        for d in range(4):
            c = nbr[cell, d]
            if c >= 0 and arrival[c] == t - 1 and best[c] > top:
                top = best[c]
        if top < 0:
            continue
        contact = 0
        for d in range(4):
            c = nbr[cell, d]
            if c < 0 or release[c] > t:
                contact += 1
        best[cell] = top + contact
    path = np.empty(n, np.int64)
    options = np.empty(4, np.int64)
    cell = goal
    path[n - 1] = goal
    for i in range(n - 2, -1, -1):
        t = arrival[cell] - 1
        top, m = -1, 0
        for d in range(4):
            c = nbr[cell, d]
            if c >= 0 and arrival[c] == t and best[c] >= top:
                if best[c] > top:
                    top, m = best[c], 0
                options[m] = c
                m += 1
        cell = options[0] if m == 1 else options[below(state, m)]
        path[i] = cell
    return path


@kernel
def extend(path, release, nbr, width, head, heading, fruit, extra, state):
    """:func:`~snake_agent.plans.extend`: ``path`` lengthened by up to ``extra`` cells of detours through
    free cells, two at a time: a step ``u -> v`` becomes ``u -> u' -> v' -> v``."""
    cells = release.shape[0]
    following = np.full(cells, -1, np.int64)  # the path as a linked list from the head
    on = np.zeros(cells, np.bool_)
    on[head] = True
    previous = head
    for c in path:
        following[previous] = c
        previous = c
        on[c] = True
    reverse = (heading + 2) % 4
    added, changed = 0, True
    while changed and added + 2 <= extra:
        changed = False
        u = head
        while following[u] >= 0 and added + 2 <= extra:
            v = following[u]
            d = direction(width, u, v)
            first, second = (d + 1) % 4, (d + 3) % 4
            if _next(state) & 1:
                first, second = second, first
            for k in range(2):
                side = first if k == 0 else second
                if u == head and side == reverse:
                    continue
                a, b = nbr[u, side], nbr[v, side]
                if (
                    a >= 0
                    and b >= 0
                    and not on[a]
                    and not on[b]
                    and a != fruit
                    and b != fruit
                    and release[a] == 0
                    and release[b] == 0
                ):
                    following[u] = a
                    following[a] = b
                    following[b] = v
                    on[a] = True
                    on[b] = True
                    added += 2
                    changed = True
                    break
            u = following[u]
    out = np.empty(path.shape[0] + added, np.int64)
    u, i = head, 0
    while following[u] >= 0:
        u = following[u]
        out[i] = u
        i += 1
    return out


@kernel
def fill_gaps(path, release, nbr, width, head, heading, fruit, state):
    """``path`` with detours into the gaps beside it: a step ``u -> v`` becomes ``u -> u' -> v' -> v`` when
    ``u'`` and ``v'`` are free cells beside the step whose own next cells on that side are blocked (the
    board's edge, the body or the path), the free cells of a strip one cell wide between the path and an
    obstacle, which would otherwise be left as a hole."""
    cells = release.shape[0]
    following = np.full(cells, -1, np.int64)
    on = np.zeros(cells, np.bool_)
    on[head] = True
    previous = head
    for c in path:
        following[previous] = c
        previous = c
        on[c] = True
    reverse = (heading + 2) % 4
    added, changed = 0, True
    while changed:
        changed = False
        u = head
        while following[u] >= 0:
            v = following[u]
            d = direction(width, u, v)
            first, second = (d + 1) % 4, (d + 3) % 4
            if _next(state) & 1:
                first, second = second, first
            for k in range(2):
                side = first if k == 0 else second
                if u == head and side == reverse:
                    continue
                a, b = nbr[u, side], nbr[v, side]
                if a < 0 or b < 0 or on[a] or on[b] or a == fruit or b == fruit or release[a] or release[b]:
                    continue
                aa, bb = nbr[a, side], nbr[b, side]
                if (aa < 0 or on[aa] or release[aa] > 0) and (bb < 0 or on[bb] or release[bb] > 0):
                    following[u] = a
                    following[a] = b
                    following[b] = v
                    on[a] = True
                    on[b] = True
                    added += 2
                    changed = True
                    break
            u = following[u]
    out = np.empty(path.shape[0] + added, np.int64)
    u, i = head, 0
    while following[u] >= 0:
        u = following[u]
        out[i] = u
        i += 1
    return out


@kernel
def after_release(release, path, length):
    """``release`` after entering the cells of ``path`` and eating at its last (the afterstate's body,
    ``length + 1`` long)."""
    d = path.shape[0]
    out = np.empty_like(release)
    for c in range(release.shape[0]):
        r = release[c] + 1 - d
        out[c] = r if r > 0 else 0
    for i in range(d):
        r = length + 2 - d + i
        out[path[i]] = r if r > 0 else 0
    return out


@kernel
def moved_release(release, entered, length):
    """``release`` after entering the cells of ``entered`` without eating."""
    w = entered.shape[0]
    out = np.empty_like(release)
    for c in range(release.shape[0]):
        r = release[c] - w
        out[c] = r if r > 0 else 0
    for j in range(w):
        r = length - w + 1 + j
        out[entered[j]] = r if r > 0 else 0
    return out


# -------------------------------------------------------------- the certificate
@kernel
def requirements(gate, cross, side, body, edges):
    """:func:`~snake_agent.certify.requirements`: ``(ok, need)``, the gates the moves of ``body`` decide;
    not ok if a move is not one of its cell's two moves or two moves disagree about a gate."""
    need = np.full(edges, -1, np.int8)
    for k in range(body.shape[0] - 1, 0, -1):
        cell, nxt = body[k], body[k - 1]
        if nxt == side[cell]:
            inside = 0
        elif nxt == cross[cell]:
            inside = 1
        else:
            return False, need
        e = gate[cell]
        if e >= 0:
            if need[e] < 0:
                need[e] = inside
            elif need[e] != inside:
                return False, need
    return True, need


@kernel
def _root(parent, a):
    """The root of ``a``, halving the path on the way."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@kernel
def spanning(ends, blocks, need, shuffle, state):
    """:func:`~snake_agent.certify.spanning`: ``(ok, tree)``, a spanning tree of the blocks with the gates
    ``need`` puts in and without those it keeps out; the other edges join in order, or at random with
    ``shuffle``."""
    edges = ends.shape[0]
    parent = np.arange(blocks)
    tree = np.zeros(edges, np.bool_)
    joined = 0
    for e in range(edges):
        if need[e] == 1:
            a, b = _root(parent, ends[e, 0]), _root(parent, ends[e, 1])
            if a == b:
                return False, tree
            parent[a] = b
            tree[e] = True
            joined += 1
    rest = np.empty(edges, np.int64)
    n = 0
    for e in range(edges):
        if need[e] < 0:
            rest[n] = e
            n += 1
    if shuffle:
        for i in range(n - 1, 0, -1):
            j = below(state, i + 1)
            rest[i], rest[j] = rest[j], rest[i]
    for i in range(n):
        e = rest[i]
        a, b = _root(parent, ends[e, 0]), _root(parent, ends[e, 1])
        if a != b:
            parent[a] = b
            tree[e] = True
            joined += 1
    return joined == blocks - 1, tree


@kernel
def certified(gate, cross, side, ends, blocks, body):
    """:func:`~snake_agent.certify.certified`: whether the walk around some spanning tree holds ``body``."""
    ok, need = requirements(gate, cross, side, body, ends.shape[0])
    if not ok:
        return False
    ok, _ = spanning(ends, blocks, need, False, np.ones(1, np.int64))
    return ok


@kernel
def walk(gate, cross, side, tree, start, stop):
    """The cells the walk around ``tree`` enters from ``start`` until it enters ``stop`` (empty if it
    never does: the tree does not span the blocks)."""
    cells = gate.shape[0]
    path = np.empty(cells, np.int64)
    cell = start
    for i in range(cells):
        e = gate[cell]
        cell = cross[cell] if e >= 0 and tree[e] else side[cell]
        path[i] = cell
        if cell == stop:
            return path[: i + 1].copy()
    return path[:0].copy()


@kernel
def distance(gate, cross, side, tree, start, stop, cap):
    """The moves of the walk around ``tree`` from ``start`` to ``stop``, or ``cap`` if that is no fewer."""
    cell = start
    for moves in range(1, cap):
        e = gate[cell]
        cell = cross[cell] if e >= 0 and tree[e] else side[cell]
        if cell == stop:
            return moves
    return cap


@kernel
def best_tree(gate, cross, side, ends, blocks, need, start, stop, trees, state):
    """``(ok, tree)``: of ``trees`` random spanning trees that keep ``need``, the one whose walk from
    ``start`` reaches ``stop`` soonest; not ok if there is none."""
    cells = gate.shape[0]
    best = np.zeros(ends.shape[0], np.bool_)
    cap = cells + 1
    for _ in range(max(trees, 1)):
        ok, tree = spanning(ends, blocks, need, True, state)
        if not ok:
            return False, best
        moves = distance(gate, cross, side, tree, start, stop, cap)
        if moves < cap:
            best, cap = tree, moves
    return True, best


@kernel
def _joined_without(bnbr, bedge, decided, blocks, skip, seen, queue):
    """Whether the blocks stay connected over the edges not kept out by ``decided``, without ``skip``."""
    seen[:] = False
    seen[0] = True
    queue[0] = 0
    size, reached = 1, 1
    while size:
        size -= 1
        a = queue[size]
        for k in range(4):
            b, f = bnbr[a, k], bedge[a, k]
            if b >= 0 and not seen[b] and f != skip and decided[f] != 0:
                seen[b] = True
                reached += 1
                queue[size] = b
                size += 1
    return reached == blocks


@kernel
def greedy_walk(gate, cross, side, ends, bnbr, bedge, block, blocks, need, start, stop, blur, state):
    """:func:`~snake_agent.certify.greedy_walk`: a walk from ``start`` to ``stop`` around a spanning tree
    that keeps ``need``, built gate by gate as the walk meets them (empty if the gates leave no way);
    with ``blur``, distances between blocks are blurred at random."""
    cells = gate.shape[0]
    decided = need.copy()
    parent = np.arange(blocks)
    for e in range(ends.shape[0]):
        if decided[e] == 1:
            a, b = _root(parent, ends[e, 0]), _root(parent, ends[e, 1])
            if a == b:
                return np.empty(0, np.int64)
            parent[a] = b
    far = 1 << 30
    toward = np.full(blocks, far, np.int64)
    goal = block[stop]
    toward[goal] = 0
    queue = np.empty(blocks, np.int64)
    queue[0] = goal
    head_, tail_ = 0, 1
    while head_ < tail_:
        a = queue[head_]
        head_ += 1
        for k in range(4):
            b, e = bnbr[a, k], bedge[a, k]
            if b >= 0 and toward[b] == far and decided[e] != 0:
                toward[b] = toward[a] + 1
                queue[tail_] = b
                tail_ += 1
    noise = np.zeros(blocks)
    if blur:
        for a in range(blocks):
            noise[a] = uniform(state)
    seen = np.zeros(blocks, np.bool_)
    path = np.empty(cells, np.int64)
    cell = start
    for i in range(cells):
        e = gate[cell]
        if e >= 0 and decided[e] < 0:
            here, there = block[cell], block[cross[cell]]
            loops = _root(parent, here) == _root(parent, there)
            if not loops and toward[there] + noise[there] < toward[here] + noise[here]:
                decided[e] = 1
            elif _joined_without(bnbr, bedge, decided, blocks, e, seen, queue):
                decided[e] = 0
            elif not loops:
                decided[e] = 1
            else:
                return np.empty(0, np.int64)
            if decided[e] == 1:
                a, b = _root(parent, here), _root(parent, there)
                parent[a] = b
        cell = cross[cell] if e >= 0 and decided[e] == 1 else side[cell]
        path[i] = cell
        if cell == stop:
            return path[: i + 1].copy()
    return np.empty(0, np.int64)


@kernel
def field_arrival(release, cross, side, head):
    """:func:`~snake_agent.certify.field_arrival`: when the head can first enter each cell by the layout's
    moves, without eating; -1 if never."""
    cells = release.shape[0]
    time = np.full(cells, -1, np.int64)
    time[head] = 0
    frontier = np.empty(cells, np.int64)
    layer = np.empty(cells, np.int64)
    frontier[0] = head
    size, t = 1, 1
    while size:
        count = 0
        for i in range(size):
            cell = frontier[i]
            for k in range(2):
                nxt = cross[cell] if k == 0 else side[cell]
                if nxt >= 0 and time[nxt] < 0 and release[nxt] <= t:
                    time[nxt] = t
                    layer[count] = nxt
                    count += 1
        frontier, layer = layer, frontier
        size = count
        t += 1
    return time


@kernel
def field_paths(arrival, entries, goal, count, state):
    """:func:`~snake_agent.certify.field_paths`: ``count`` shortest paths to ``goal`` by the layout's
    moves (one per row), ties broken at random."""
    n = arrival[goal]
    paths = np.empty((count, n), np.int64)
    options = np.empty(2, np.int64)
    for p in range(count):
        cell = goal
        paths[p, n - 1] = goal
        for i in range(n - 2, -1, -1):
            m = 0
            for k in range(2):
                c = entries[cell, k]
                if c >= 0 and arrival[c] == arrival[cell] - 1:
                    options[m] = c
                    m += 1
            cell = options[0] if m == 1 else options[below(state, m)]
            paths[p, i] = cell
    return paths


@kernel
def prefix(cross, side, body):
    """:func:`~snake_agent.certify.prefix`: how many cells at the head end of ``body`` moved by the layout's
    moves."""
    m = 1
    while m < body.shape[0] and (body[m - 1] == cross[body[m]] or body[m - 1] == side[body[m]]):
        m += 1
    return m


@kernel
def consistent(gate, cross, side, body, edges):
    """How many cells at the head end of ``body`` moved by the layout's moves without two of those moves
    disagreeing about a gate."""
    need = np.full(edges, -1, np.int8)
    m = 1
    while m < body.shape[0]:
        cell, nxt = body[m], body[m - 1]
        if nxt == side[cell]:
            inside = 0
        elif nxt == cross[cell]:
            inside = 1
        else:
            break
        e = gate[cell]
        if e >= 0:
            if need[e] < 0:
                need[e] = inside
            elif need[e] != inside:
                break
        m += 1
    return m


@kernel
def timely(release, body, path, fruit):
    """:func:`~snake_agent.certify.timely`: whether each cell of ``path`` is free when the head enters it,
    the first move is no reversal and the fruit comes last."""
    if body.shape[0] > 1 and path.shape[0] and path[0] == body[1]:
        return False
    seen = np.zeros(release.shape[0], np.bool_)
    n = path.shape[0]
    for t in range(1, n + 1):
        cell = path[t - 1]
        if release[cell] > t or seen[cell] or (cell == fruit) != (t == n):
            return False
        seen[cell] = True
    return True


@kernel
def after_body(body, path):
    """:func:`~snake_agent.certify.after_body`: the body after following ``path`` and eating at its end."""
    length = body.shape[0] + 1
    d = path.shape[0]
    out = np.empty(length, np.int64)
    for i in range(min(d, length)):
        out[i] = path[d - 1 - i]
    for i in range(d, length):
        out[i] = body[i - d]
    return out


# ----------------------------------------------------------------- shortcuts
@kernel
def _find(parent, a):
    """The root of ``a`` without compressing the path, so that unions can be undone."""
    while parent[a] != a:
        a = parent[a]
    return a


@kernel
def _reaches(bnbr, bedge, outside, a, b, seen, queue):
    """Whether block ``b`` can be reached from block ``a`` over the edges no decision keeps out: a search
    from ``a`` that stops at ``b``, leaving ``seen`` all False as it found it."""
    seen[a] = True
    queue[0] = a
    head_, tail_ = 0, 1
    found = False
    while head_ < tail_ and not found:
        x = queue[head_]
        head_ += 1
        for k in range(4):
            y, f = bnbr[x, k], bedge[x, k]
            if y >= 0 and not seen[y] and not outside[f]:
                if y == b:
                    found = True
                    break
                seen[y] = True
                queue[tail_] = y
                tail_ += 1
    for i in range(tail_):
        seen[queue[i]] = False
    return found


@kernel
def _decide(e, want, inside, outside, parent, size, ends, bnbr, bedge, blocks, seen, queue):
    """Record a gate decision: ``(ok, root)``, ``root`` the joined block to restore on undo (or -1); not
    ok, and nothing recorded, if the decision contradicts the others."""
    if want:
        if outside[e]:
            return False, -1
        if inside[e]:
            inside[e] += 1
            return True, -1
        a, b = _find(parent, ends[e, 0]), _find(parent, ends[e, 1])
        if a == b:
            return False, -1
        if size[a] > size[b]:
            a, b = b, a
        parent[a] = b
        size[b] += size[a]
        inside[e] = 1
        return True, a
    if inside[e]:
        return False, -1
    outside[e] += 1
    # The blocks were connected over the edges not kept out, so they stay connected without this one
    # exactly when its two blocks still reach each other.
    if (
        outside[e] == 1
        and _find(parent, ends[e, 0]) != _find(parent, ends[e, 1])
        and not _reaches(bnbr, bedge, outside, ends[e, 0], ends[e, 1], seen, queue)
    ):
        outside[e] -= 1
        return False, -1
    return True, -1


@kernel
def _undo(e, want, root, inside, outside, parent, size):
    if want:
        inside[e] -= 1
        if root >= 0:
            b = parent[root]
            parent[root] = root
            size[b] -= size[root]
    else:
        outside[e] -= 1


@kernel
def stall_run(gate, cross, side, ends, bnbr, bedge, blocks, body, run, fruit, goal, budget, tries, state):
    """The longest path found, up to ``goal`` moves, of the layout's moves from the head through free cells
    and body cells vacated in time, never onto the fruit, whose own gates and those of the body's first
    ``run`` cells some spanning tree can hold. ``tries`` depth-first searches of at most ``budget`` steps
    each, trying first the move with more moves after it and breaking ties at random, start from the gates
    of the run, decided once. Without eating, ``length - run`` such moves replace the rest of the body,
    which is then certified."""
    cells = gate.shape[0]
    edges = ends.shape[0]
    length = body.shape[0]
    release = np.zeros(cells, np.int64)
    for k in range(length):
        release[body[k]] = length - k
    kept_inside = np.zeros(edges, np.int64)
    kept_outside = np.zeros(edges, np.int64)
    kept_parent = np.arange(blocks)
    kept_size = np.ones(blocks, np.int64)
    seen = np.zeros(blocks, np.bool_)
    block_queue = np.empty(blocks, np.int64)
    for k in range(1, min(run, length)):
        cell = body[k]
        e = gate[cell]
        if e >= 0:
            ok, _ = _decide(
                e,
                body[k - 1] == cross[cell],
                kept_inside,
                kept_outside,
                kept_parent,
                kept_size,
                ends,
                bnbr,
                bedge,
                blocks,
                seen,
                block_queue,
            )
            if not ok:
                return np.empty(0, np.int64)
    goal = min(goal, cells)
    on = np.zeros(cells, np.bool_)
    path = np.empty(goal + 1, np.int64)
    best = np.empty(goal + 1, np.int64)
    found = 0
    frame_cell = np.empty(goal + 1, np.int64)
    options = np.empty((goal + 1, 2), np.int64)
    wants = np.empty((goal + 1, 2), np.bool_)
    choices = np.zeros(goal + 1, np.int64)
    current = np.zeros(goal + 1, np.int64)
    frame_e = np.empty(goal + 1, np.int64)
    frame_want = np.empty(goal + 1, np.bool_)
    frame_root = np.empty(goal + 1, np.int64)
    keys = np.empty(2)
    for _ in range(max(tries, 1)):
        inside, outside = kept_inside.copy(), kept_outside.copy()
        parent, size = kept_parent.copy(), kept_size.copy()
        on[:] = False
        on[body[0]] = True
        steps, depth = 0, 0
        frame_cell[0] = body[0]
        expand = True
        while depth >= 0:
            if expand:  # the options from the frame's cell, after ``depth`` moves
                expand = False
                if depth > found:
                    found = depth
                    best[:depth] = path[:depth]
                if depth >= goal:
                    break
                cell, t = frame_cell[depth], depth
                n = 0
                for k in range(2):
                    nxt = cross[cell] if k == 0 else side[cell]
                    if nxt < 0 or on[nxt] or nxt == fruit or release[nxt] > t + 1:
                        continue
                    onward = 0
                    for j in range(2):
                        after = cross[nxt] if j == 0 else side[nxt]
                        if after >= 0 and not on[after] and after != fruit and release[after] <= t + 2:
                            onward += 1
                    options[depth, n] = nxt
                    wants[depth, n] = k == 0
                    keys[n] = -onward + uniform(state)
                    n += 1
                if n == 2 and keys[1] < keys[0]:
                    options[depth, 0], options[depth, 1] = options[depth, 1], options[depth, 0]
                    wants[depth, 0], wants[depth, 1] = wants[depth, 1], wants[depth, 0]
                choices[depth] = n
                current[depth] = 0
            if current[depth] >= choices[depth] or steps >= budget:
                depth -= 1  # back to the frame before: undo the move it was exploring
                if depth >= 0:
                    on[options[depth, current[depth]]] = False
                    if frame_e[depth] >= 0:
                        _undo(
                            frame_e[depth],
                            frame_want[depth],
                            frame_root[depth],
                            inside,
                            outside,
                            parent,
                            size,
                        )
                    current[depth] += 1
                continue
            cell = frame_cell[depth]
            nxt = options[depth, current[depth]]
            want = wants[depth, current[depth]]
            steps += 1
            e = gate[cell]
            root = -1
            if e >= 0:
                ok, root = _decide(
                    e, want, inside, outside, parent, size, ends, bnbr, bedge, blocks, seen, block_queue
                )
                if not ok:
                    current[depth] += 1
                    continue
            frame_e[depth], frame_want[depth], frame_root[depth] = e, want, root
            path[depth] = nxt
            on[nxt] = True
            depth += 1
            frame_cell[depth] = nxt
            expand = True
        if found >= goal:
            break
    return best[:found].copy()


@kernel
def _regions(start, on, nbr, fruit, mark, now, stack):
    """The size of the region of cells not ``on`` around ``start`` and whether it holds the fruit,
    marking its cells with ``now``."""
    mark[start] = now
    stack[0] = start
    size, cells, fruity = 1, 1, start == fruit
    while size:
        size -= 1
        c = stack[size]
        for d in range(4):
            n = nbr[c, d]
            if n >= 0 and not on[n] and mark[n] != now:
                mark[n] = now
                cells += 1
                fruity = fruity or n == fruit
                stack[size] = n
                size += 1
    return cells, fruity


@kernel
def _split(cell, left, on, ring, nbr, fruit, mark, clock, stack):
    """Whether taking ``cell`` out of the free cells after eating leaves them unable to be one region (see
    :func:`~snake_agent.certify.shortcuts`)."""
    runs = 0
    for k in range(0, 8, 2):  # the four neighbours; neighbours linked through a corner form one run
        c = ring[cell, k]
        if c >= 0 and not on[c]:
            p, q = ring[cell, (k + 7) % 8], ring[cell, (k + 6) % 8]
            if not (p >= 0 and not on[p] and q >= 0 and not on[q]):
                runs += 1
    if runs <= 1:
        return False
    clock[0] += 1
    now = clock[0]
    mark[cell] = now
    fruit_region, others = 0, 0
    for d in range(4):
        s = nbr[cell, d]
        if s < 0 or on[s] or mark[s] == now:
            continue
        cells, fruity = _regions(s, on, nbr, fruit, mark, now, stack)
        if fruity:
            fruit_region += cells
        else:
            others += 1
    if not fruit_region:
        return True  # the fruit is out of reach
    return others >= 2 or (others == 1 and fruit_region > left)


@kernel
def _separate(limit, on, nbr, fruit, mark, clock, stack):
    """Whether the free cells after eating are split before the path starts, as in :func:`_split`."""
    clock[0] += 1
    now = clock[0]
    fruit_region, others = 0, 0
    for first in range(on.shape[0]):
        if on[first] or mark[first] == now:
            continue
        cells, fruity = _regions(first, on, nbr, fruit, mark, now, stack)
        if fruity:
            fruit_region += cells
        else:
            others += 1
    return others >= 2 or (others == 1 and fruit_region > limit)


@kernel
def _distances(limit, dist, far, fruit, entries, release, gate, side, ends, inside, outside, parent, queue):
    """Moves from each cell to the fruit for a path of ``limit`` moves, a bound for the search: a cell
    counts only if it is free by the latest move at which such a path can enter it, and a move only if
    the gates decided so far allow it."""
    dist[:] = far
    dist[fruit] = 0
    queue[0] = fruit
    head_, tail_ = 0, 1
    while head_ < tail_:
        cell = queue[head_]
        head_ += 1
        d = dist[cell] + 1
        for k in range(2):
            prev = entries[cell, k]
            if prev < 0 or dist[prev] != far or release[prev] > limit - d:
                continue
            e = gate[prev]
            if e >= 0:
                if cell == side[prev]:
                    if inside[e]:
                        continue
                elif outside[e] or (not inside[e] and _find(parent, ends[e, 0]) == _find(parent, ends[e, 1])):
                    continue
            dist[prev] = d
            queue[tail_] = prev
            tail_ += 1


@kernel
def shortcuts(
    gate,
    cross,
    side,
    ends,
    bnbr,
    bedge,
    entries,
    ring,
    nbr,
    blocks,
    body,
    fruit,
    bound,
    count,
    budget,
    run,
    partial,
    state,
):
    """:func:`~snake_agent.certify.shortcuts`: up to ``count`` paths to the fruit of fewer than ``bound``
    moves whose afterstates are certified, shortest first, as ``(paths, lengths, exact, found)`` (a path
    per row).

    An iterative-deepening search over the layout's moves, through free cells and body cells vacated in
    time, with an explicit stack; it stops after ``count`` paths or ``budget`` steps. The gates decided by
    the moves of the first ``run`` cells of the body are kept (the whole body, when it is certified).
    With ``partial`` (the transition, from a body certified only up to ``run``), a path also counts when
    it lengthens the certifiable run of the afterstate without certifying it all (``exact`` False), and
    the split of the free cells prunes nothing, since the older part of the body still divides them."""
    cells = gate.shape[0]
    edges = ends.shape[0]
    length = body.shape[0]
    release = np.zeros(cells, np.int64)
    for k in range(length):
        release[body[k]] = length - k
    paths = np.zeros((count + 1, max(bound, 1)), np.int64)
    lengths = np.zeros(count + 1, np.int64)
    exact = np.zeros(count + 1, np.bool_)
    found = 0
    start = field_arrival(release, cross, side, body[0])[fruit]
    if start < 0 or start >= bound:
        return paths, lengths, exact, found
    far = cells + 1
    dist = np.full(cells, far, np.int64)
    inside = np.zeros(edges, np.int64)  # decisions per gate, from the moves kept so far
    outside = np.zeros(edges, np.int64)
    parent = np.arange(blocks)
    size = np.ones(blocks, np.int64)
    seen = np.zeros(blocks, np.bool_)
    block_queue = np.empty(blocks, np.int64)
    on = np.zeros(cells, np.bool_)  # the cells that stay out of the free cells after eating
    mark = np.zeros(cells, np.int64)
    clock = np.zeros(1, np.int64)
    stack = np.empty(cells, np.int64)
    queue = np.empty(cells, np.int64)
    path = np.empty(cells + 1, np.int64)
    # The moves of the body cells a path of the shortest length leaves in place, decided from the head
    # back, so that longer limits undo the last ones.
    kept_e = np.empty(length, np.int64)
    kept_want = np.empty(length, np.bool_)
    kept_root = np.empty(length, np.int64)
    kept = 0
    for k in range(1, min(run, length - start + 1)):
        cell = body[k]
        e = gate[cell]
        want = body[k - 1] == cross[cell]
        root = -1
        if e >= 0:
            ok, root = _decide(
                e, want, inside, outside, parent, size, ends, bnbr, bedge, blocks, seen, block_queue
            )
            if not ok:
                return paths, lengths, exact, found  # not certified to begin with
        kept_e[kept], kept_want[kept], kept_root[kept] = e, want, root
        kept += 1
    # The search's stack, one frame per move: the cell, its options (at most two) and the gate decision
    # of the option being explored.
    frame_cell = np.empty(cells + 1, np.int64)
    options = np.empty((cells + 1, 2), np.int64)
    wants = np.empty((cells + 1, 2), np.bool_)
    choices = np.zeros(cells + 1, np.int64)
    current = np.zeros(cells + 1, np.int64)
    frame_e = np.empty(cells + 1, np.int64)
    frame_want = np.empty(cells + 1, np.bool_)
    frame_root = np.empty(cells + 1, np.int64)
    keys = np.empty(2)
    steps = 0
    full = np.empty(cells + 1, np.int64)
    for limit in range(start, bound, 2):
        while kept > max(min(run - 1, length - limit), 0):
            kept -= 1
            if kept_e[kept] >= 0:
                _undo(kept_e[kept], kept_want[kept], kept_root[kept], inside, outside, parent, size)
        for k in range(length):
            on[body[k]] = k <= length - limit
        if not partial and _separate(limit, on, nbr, fruit, mark, clock, stack):
            continue
        _distances(
            limit, dist, far, fruit, entries, release, gate, side, ends, inside, outside, parent, queue
        )
        depth, moves = 0, 0
        frame_cell[0] = body[0]
        expand = True
        while depth >= 0:
            if expand:  # the options from the frame's cell, after ``depth`` moves
                expand = False
                cell, t = frame_cell[depth], depth
                n = 0
                for k in range(2):
                    nxt = cross[cell] if k == 0 else side[cell]
                    if nxt < 0 or on[nxt] or release[nxt] > t + 1:
                        continue
                    if nxt == fruit:
                        if t + 1 == limit:
                            for i in range(moves):
                                full[i] = path[i]
                            full[moves] = fruit
                            after = after_body(body, full[: moves + 1])
                            kept_run = prefix(cross, side, after) if partial else after.shape[0]
                            if (
                                certified(gate, cross, side, ends, blocks, after[:kept_run])
                                and found <= count
                            ):
                                paths[found, : moves + 1] = full[: moves + 1]
                                lengths[found] = moves + 1
                                exact[found] = kept_run == after.shape[0]
                                found += 1
                        continue
                    if t + 1 + dist[nxt] > limit:
                        continue
                    options[depth, n] = nxt
                    wants[depth, n] = k == 0
                    keys[n] = dist[nxt] + uniform(state)
                    n += 1
                if n == 2 and keys[1] < keys[0]:
                    options[depth, 0], options[depth, 1] = options[depth, 1], options[depth, 0]
                    wants[depth, 0], wants[depth, 1] = wants[depth, 1], wants[depth, 0]
                choices[depth] = n
                current[depth] = 0
            if current[depth] >= choices[depth] or found >= count or steps >= budget:
                depth -= 1  # back to the frame before: undo the move it was exploring
                if depth >= 0:
                    on[options[depth, current[depth]]] = False
                    moves -= 1
                    if frame_e[depth] >= 0:
                        _undo(
                            frame_e[depth],
                            frame_want[depth],
                            frame_root[depth],
                            inside,
                            outside,
                            parent,
                            size,
                        )
                    current[depth] += 1
                continue
            cell = frame_cell[depth]
            nxt = options[depth, current[depth]]
            want = wants[depth, current[depth]]
            steps += 1
            e = gate[cell]
            root = -1
            if e >= 0:
                ok, root = _decide(
                    e, want, inside, outside, parent, size, ends, bnbr, bedge, blocks, seen, block_queue
                )
                if not ok:
                    current[depth] += 1
                    continue
            if not partial and _split(nxt, limit - depth - 1, on, ring, nbr, fruit, mark, clock, stack):
                if e >= 0:
                    _undo(e, want, root, inside, outside, parent, size)
                current[depth] += 1
                continue
            frame_e[depth], frame_want[depth], frame_root[depth] = e, want, root
            path[moves] = nxt
            moves += 1
            on[nxt] = True
            depth += 1
            frame_cell[depth] = nxt
            expand = True
        if found >= count or steps >= budget:
            break
    return paths, lengths, exact, found
