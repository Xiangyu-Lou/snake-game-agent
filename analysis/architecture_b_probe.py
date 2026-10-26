"""Measurements behind the Architecture B review; no model training.

Run from the repository root:
    .venv/bin/python analysis/architecture_b_probe.py

The script rewrites ``architecture_b_probe_results.json`` next to itself. Game
outcomes and counts are deterministic. Timings depend on the machine, which is
recorded with them. A full run takes a few minutes on a laptop-class CPU.

The survival certificate, the exact doom search and the certificate-guarded
greedy policy below are analysis instruments. They are not application code and
not a proposed agent.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from collections import Counter, deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame, observe
from snake_game.policies import GreedyPolicy, HamiltonianPolicy, RandomPolicy, safe_directions

DX = (0, 1, 0, -1)
DY = (-1, 0, 1, 0)
NEVER = -(1 << 40)
BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0))


# --------------------------------------------------------------- survival analysis
class Board:
    """Timestamped body, as in ``VectorSnakeEnv``: cell ``c`` is occupied while
    ``t - entered[c] < length``. Moves can be made and undone in O(1)."""

    __slots__ = ("w", "h", "length", "t", "entered", "head", "heading")

    def __init__(self, state: GameState) -> None:
        self.w, self.h = state.width, state.height
        self.length = state.length
        self.t = 0
        self.entered = [NEVER] * (self.w * self.h)
        for k, (x, y) in enumerate(state.snake):
            self.entered[y * self.w + x] = -k
        self.head = state.head[1] * self.w + state.head[0]
        self.heading = int(state.direction)

    def release(self, cell: int) -> int:
        """Moves until ``cell`` is vacated if nothing is eaten: 1 = tail, <= 0 = free."""
        return self.length - (self.t - self.entered[cell])

    def target(self, d: int) -> int:
        """Cell entered by absolute direction ``d``; -1 if fatal or a reversal."""
        if d == (self.heading + 2) % 4:
            return -1
        x, y = self.head % self.w + DX[d], self.head // self.w + DY[d]
        if not (0 <= x < self.w and 0 <= y < self.h):
            return -1
        cell = y * self.w + x
        return cell if self.release(cell) <= 1 else -1

    def move(self, d: int, cell: int, grow: bool = False) -> tuple[int, int, int, int]:
        undo = (self.head, self.heading, self.entered[cell], self.length)
        self.t += 1
        self.length += grow
        self.entered[cell] = self.t
        self.head, self.heading = cell, d
        return undo

    def unmove(self, cell: int, undo: tuple[int, int, int, int]) -> None:
        self.head, self.heading, self.entered[cell], self.length = undo
        self.t -= 1

    def has_survival_certificate(self) -> bool:
        """Sufficient condition for surviving forever without eating.

        True if a shortest path through free cells brings the head next to a body
        segment that has moved away by the time the head arrives. The head can then
        enter it and follow its own body around a closed loop indefinitely. The
        fruit is treated as a free cell (no growth); a False result does not prove
        that the snake is lost.
        """
        w, h = self.w, self.h
        seen = {self.head}
        queue: deque[tuple[int, int]] = deque([(self.head, 0)])
        reverse = (self.heading + 2) % 4
        while queue:
            cell, dist = queue.popleft()
            x0, y0 = cell % w, cell // w
            for d in range(4):
                if cell == self.head and d == reverse:
                    continue
                x, y = x0 + DX[d], y0 + DY[d]
                if not (0 <= x < w and 0 <= y < h):
                    continue
                nxt = y * w + x
                release = self.release(nxt)
                if release > 0:
                    if release <= dist + 1:
                        return True
                elif nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return False


class OverBudget(Exception):
    pass


def doomed(board: Board, budget: int) -> tuple[bool | None, int]:
    """Exact search over non-growing continuations: (True if every one collides,
    False if one survives indefinitely, None if ``budget`` nodes were not enough;
    nodes visited). Eating only lengthens the body, so True also means lost,
    unless the last free cells are filled before the collision. The board is
    restored afterwards."""
    nodes = 0

    def search() -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise OverBudget
        if board.has_survival_certificate():
            return False
        for d in range(4):
            cell = board.target(d)
            if cell < 0:
                continue
            undo = board.move(d, cell)
            try:
                lost = search()
            finally:
                board.unmove(cell, undo)
            if not lost:
                return False
        return True

    try:
        return search(), nodes
    except OverBudget:
        return None, nodes


def classify_moves(state: GameState) -> dict[int, str]:
    """``{direction: "won" | "certified" | "uncertified"}`` for every immediately safe move."""
    board = Board(state)
    cells = state.width * state.height
    fruit = -1 if state.fruit is None else state.fruit[1] * state.width + state.fruit[0]
    classes = {}
    for d in range(4):
        cell = board.target(d)
        if cell < 0:
            continue
        grow = cell == fruit
        undo = board.move(d, cell, grow)
        if board.length == cells:
            classes[d] = "won"
        else:
            classes[d] = "certified" if board.has_survival_certificate() else "uncertified"
        board.unmove(cell, undo)
    return classes


def fruit_distance(state: GameState, start: tuple[int, int]) -> int:
    """Shortest path length from ``start`` to the fruit, current body (minus tail) blocked."""
    if start == state.fruit:
        return 0
    blocked = set(state.snake[:-1])
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (x0, y0), dist = queue.popleft()
        for d in range(4):
            cell = (x0 + DX[d], y0 + DY[d])
            if 0 <= cell[0] < state.width and 0 <= cell[1] < state.height:
                if cell == state.fruit:
                    return dist + 1
                if cell not in blocked and cell not in seen:
                    seen.add(cell)
                    queue.append((cell, dist + 1))
    return 1 << 30


class GuardedGreedyPolicy:
    """Greedy restricted to moves that keep a survival certificate (analysis heuristic)."""

    def __call__(self, state: GameState) -> Direction:
        safe = safe_directions(state)
        if not safe:
            return state.direction
        classes = classify_moves(state)
        good = [d for d in safe if classes[int(d)] != "uncertified"]
        if not good:
            return max(safe, key=lambda d: GreedyPolicy._room_after(state, d))
        hx, hy = state.head
        return min(
            good,
            key=lambda d: (
                fruit_distance(state, (hx + DX[d], hy + DY[d])),
                -GreedyPolicy._room_after(state, d),
            ),
        )


# -------------------------------------------------------------------- game helpers
def play(
    game: SnakeGame,
    policy: Callable,
    record: list | None = None,
    *,
    detect_loops: bool = True,
    stall_factor: int = 20,
) -> str:
    """Play to the end: "won", "lost", "looped" or "stalled".

    "looped": a deterministic policy met the same state twice between two fruits,
    so it would repeat the same moves forever. "stalled": more than
    ``stall_factor`` × cells moves without a fruit.
    """
    cells = game.width * game.height
    seen: set = set()
    while not game.done:
        state = game.state()
        if record is not None:
            record.append(state)
        if detect_loops:
            if state.steps_since_fruit == 0:
                seen.clear()
            key = (state.snake, state.direction, id(getattr(policy, "_chosen", None)))
            if key in seen:
                return "looped"
            seen.add(key)
        if state.steps_since_fruit > stall_factor * cells:
            return "stalled"
        game.step(policy(state))
    return "won" if game.won else "lost"


def band_of(state: GameState) -> int:
    occupancy = state.length / (state.width * state.height)
    return next(i for i, (lo, hi) in enumerate(BANDS) if occupancy < hi or i == len(BANDS) - 1)


# --------------------------------------------------------------------- experiments
def certificate_validation() -> dict:
    """Check the instruments against the engine before relying on them."""
    # 1. Board.target agrees with SnakeGame.is_safe.
    mismatches = checked = 0
    for seed in range(30):
        game, policy = SnakeGame(GameConfig(7, 6), seed=seed), RandomPolicy(seed=seed)
        while not game.done:
            board = Board(game.state())
            for d in Direction:
                checked += 1
                mismatches += (board.target(int(d)) >= 0) != game.is_safe(d)
            game.step(policy(game.state()))

    # 2. Replay every certificate: enter the body in time, then circle twice.
    replayed = 0
    for size, seeds, factory in (
        ((6, 6), range(40), lambda s: RandomPolicy(seed=s)),
        ((10, 8), range(10), lambda s: GreedyPolicy()),
    ):
        for seed in seeds:
            game, policy = SnakeGame(GameConfig(*size), seed=seed), factory(seed)
            while not game.done:
                state = game.state()
                board = Board(state)
                if board.has_survival_certificate():
                    loop = certificate_loop(state)
                    assert loop is not None and len(loop) >= state.length
                    for step in range(2 * len(loop)):
                        cell = loop[step % len(loop)]
                        d = Direction.from_delta(
                            cell % board.w - board.head % board.w, cell // board.w - board.head // board.w
                        )
                        assert board.target(int(d)) == cell, "certificate loop collided"
                        board.move(int(d), cell)
                    replayed += 1
                game.step(policy(state))

    # 3. Doomed states really die with the engine (growth and random fruits included).
    rng = random.Random(0)

    def all_die(game: SnakeGame, depth: int) -> bool:
        if game.done:
            return not game.won
        if depth == 0:
            return False
        for d in game.safe_directions():
            child = game.clone()
            child.reseed(rng.randrange(1 << 30))
            child.step(d)
            if not all_die(child, depth - 1):
                return False
        return True

    doomed_checked = doomed_confirmed = 0
    for seed in range(40):
        game, policy, states = SnakeGame(GameConfig(6, 6), seed=seed), RandomPolicy(seed=seed), []
        play(game, policy, states, detect_loops=False)
        for state in states[-12:]:
            if doomed(Board(state), 200_000)[0]:
                doomed_checked += 1
                doomed_confirmed += all_die(SnakeGame.from_state(state, seed=seed), 40)
    return {
        "target_vs_is_safe_checks": checked,
        "target_vs_is_safe_mismatches": mismatches,
        "certificates_replayed_for_two_loop_periods": replayed,
        "doomed_states_checked_with_engine_depth_40": doomed_checked,
        "doomed_states_confirmed": doomed_confirmed,
    }


def certificate_loop(state: GameState) -> list[int] | None:
    """The closed loop promised by the certificate, as cells to enter in order."""
    w, h = state.width, state.height
    body = [y * w + x for x, y in state.snake]
    index = {cell: k for k, cell in enumerate(body)}
    head, length = body[0], len(body)
    reverse = (int(state.direction) + 2) % 4
    parent: dict[int, int | None] = {head: None}
    queue = deque([(head, 0)])
    while queue:
        cell, dist = queue.popleft()
        for d in range(4):
            if cell == head and d == reverse:
                continue
            x, y = cell % w + DX[d], cell // w + DY[d]
            if not (0 <= x < w and 0 <= y < h):
                continue
            nxt = y * w + x
            k = index.get(nxt)
            if k is not None:
                if k > 0 and length - k <= dist + 1:
                    path, node = [nxt], cell
                    while node != head:
                        path.append(node)
                        node = parent[node]
                    return path[::-1] + body[k - 1 :: -1]
            elif nxt not in parent:
                parent[nxt] = cell
                queue.append((nxt, dist + 1))
    return None


def expert_lookup() -> dict:
    """Is the Hamiltonian expert a function of the head cell on standard starts?"""
    table: dict[tuple[int, int], Direction] = {}
    conflicts = decisions = 0
    moves = []
    for seed in range(6):
        game, policy = SnakeGame(GameConfig(20, 20), seed=seed), HamiltonianPolicy()
        while not game.done:
            state = game.state()
            action = policy(state)
            decisions += 1
            conflicts += table.setdefault(state.head, action) != action
            game.step(action)
        moves.append(game.steps)
    return {
        "board": [20, 20],
        "seeds": list(range(6)),
        "decisions": decisions,
        "distinct_head_cells": len(table),
        "conflicting_labels": conflicts,
        "moves_per_game": moves,
    }


def game_lengths() -> list[dict]:
    """Hamiltonian game length versus the aligned-cycle expectation F(F + 3) / 4."""
    rows = []
    for side in (4, 6, 8, 10, 12):
        free = side * side - 3
        steps = []
        for seed in range(20):
            game, policy = SnakeGame(GameConfig(side, side), seed=seed), HamiltonianPolicy()
            assert play(game, policy) == "won"
            steps.append(game.steps)
        rows.append(
            {
                "board": [side, side],
                "episodes": len(steps),
                "mean_moves": statistics.mean(steps),
                "expected_moves": free * (free + 3) / 4,
                "bound_moves": free * (free + 1) // 2,
            }
        )
    return rows


def branching_profile() -> dict:
    """Share of states with at most one immediately safe / certified move, by occupancy."""
    runs = {
        "hamiltonian_20x20_seed0": ((20, 20), lambda: HamiltonianPolicy(), range(1)),
        "hamiltonian_10x10_seeds0-9": ((10, 10), lambda: HamiltonianPolicy(), range(10)),
        "guarded_greedy_10x10_seeds0-9": ((10, 10), lambda: GuardedGreedyPolicy(), range(10)),
    }
    result = {}
    for name, (size, factory, seeds) in runs.items():
        counts = [Counter() for _ in BANDS]
        for seed in seeds:
            game, states = SnakeGame(GameConfig(*size), seed=seed), []
            play(game, factory(), states)
            for state in states:
                classes = classify_moves(state)
                c = counts[band_of(state)]
                c["states"] += 1
                c["at_most_one_safe"] += len(classes) <= 1
                c["at_most_one_certified"] += sum(v != "uncertified" for v in classes.values()) <= 1
        rows = []
        for (lo, hi), c in zip(BANDS, counts, strict=True):
            if c["states"]:
                rows.append(
                    {
                        "occupancy": [lo, hi],
                        "states": c["states"],
                        "at_most_one_safe": round(c["at_most_one_safe"] / c["states"], 4),
                        "at_most_one_certified": round(c["at_most_one_certified"] / c["states"], 4),
                    }
                )
        total = sum(c["states"] for c in counts)
        result[name] = {
            "bands": rows,
            "states": total,
            "at_most_one_safe": round(sum(c["at_most_one_safe"] for c in counts) / total, 4),
            "at_most_one_certified": round(sum(c["at_most_one_certified"] for c in counts) / total, 4),
        }
    return result


def death_horizons() -> dict:
    """How long before a greedy death the game was already lost (exact doom search)."""
    result = {}
    for size, seeds, budget in (((10, 10), range(30), 1_000_000), ((20, 20), range(6), 300_000)):
        rows = []
        for seed in seeds:
            game, states = SnakeGame(GameConfig(*size), seed=seed), []
            assert play(game, GreedyPolicy(), states) == "lost"
            proof_nodes, back = None, 1
            while back <= len(states):
                verdict, nodes = doomed(Board(states[-back]), budget)
                if verdict is not True:
                    break
                proof_nodes = nodes
                back += 1
            assert back > 1, "the state before a greedy death has no safe move"
            first_doomed = states[-(back - 1)]
            rows.append(
                {
                    "seed": seed,
                    "score": game.score,
                    "horizon_moves": back - 1,
                    "lower_bound": verdict is None,
                    "proof_nodes": proof_nodes,
                    "occupancy_when_doomed": round(first_doomed.length / (size[0] * size[1]), 3),
                }
            )
        horizons = [r["horizon_moves"] for r in rows]
        proofs = [r["proof_nodes"] for r in rows if not r["lower_bound"]]
        result[f"{size[0]}x{size[1]}"] = {
            "node_budget": budget,
            "episodes": rows,
            "median_horizon_moves": statistics.median(horizons),
            "min_horizon_moves": min(horizons),
            "max_horizon_moves": max(horizons),
            "budget_exceeded": sum(r["lower_bound"] for r in rows),
            "median_proof_nodes_when_proved": statistics.median(proofs) if proofs else None,
        }
    return result


def uncertified_moves() -> dict:
    """Immediately safe moves without a certificate: how many are actually survivable?"""
    result = {}
    policies = {"greedy": GreedyPolicy, "guarded_greedy": GuardedGreedyPolicy}
    labels = {True: "uncertified_doomed", False: "uncertified_survivable", None: "unknown"}
    for name, factory in policies.items():
        counts = Counter()
        for seed in range(10):
            game, states = SnakeGame(GameConfig(10, 10), seed=seed), []
            play(game, factory(), states)
            for state in states:
                board = Board(state)
                fruit = state.fruit[1] * state.width + state.fruit[0]
                for d, label in classify_moves(state).items():
                    if label != "uncertified":
                        counts[label] += 1
                        continue
                    cell = board.target(d)
                    undo = board.move(d, cell, cell == fruit)
                    verdict = doomed(board, 100_000)[0]
                    board.unmove(cell, undo)
                    counts[labels[verdict]] += 1
        result[f"{name}_10x10_seeds0-9"] = dict(counts)
    return result


def guarded_greedy_outcomes() -> dict:
    result = {}
    for size, seeds in (((10, 10), range(30)), ((20, 20), range(6))):
        rows = []
        for seed in seeds:
            game = SnakeGame(GameConfig(*size), seed=seed)
            outcome = play(game, GuardedGreedyPolicy())
            rows.append({"seed": seed, "outcome": outcome, "score": game.score, "moves": game.steps})
        result[f"{size[0]}x{size[1]}"] = {
            "episodes": rows,
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "mean_score": statistics.mean(r["score"] for r in rows),
            "max_score": size[0] * size[1] - 3,
        }
    return result


def recovery() -> dict:
    """Reference policies started from certified, non-aligned 10x10 mid-game states."""
    size, cells = (10, 10), 100
    hamiltonian = HamiltonianPolicy()
    cycles = hamiltonian._candidates(*size)
    starts = []
    for seed in range(100, 160):
        game, states = SnakeGame(GameConfig(*size), seed=seed), []
        play(game, GuardedGreedyPolicy(), states)
        wanted = [0.3, 0.5, 0.7]
        for state in states:
            while wanted and state.length >= (wanted[0] + 0.1) * cells:
                wanted.pop(0)  # no qualifying state inside this band
            if wanted and state.length >= wanted[0] * cells:
                aligned = any(hamiltonian._aligned(c, state) for c in cycles)
                if not aligned and Board(state).has_survival_certificate():
                    starts.append((wanted.pop(0), state))
    policies = {
        "hamiltonian": HamiltonianPolicy,
        "guarded_greedy": GuardedGreedyPolicy,
        "greedy": GreedyPolicy,
    }
    result: dict = {"start_states": dict(Counter(band for band, _ in starts))}
    for name, factory in policies.items():
        by_band: dict[float, Counter] = {}
        for i, (band, state) in enumerate(starts):
            game = SnakeGame.from_state(state, seed=10_000 + i)
            by_band.setdefault(band, Counter())[play(game, factory())] += 1
        result[name] = {str(band): dict(c) for band, c in sorted(by_band.items())}
    return result


# ------------------------------------------------------------------------ costs
def encode_cells(state: GameState) -> np.ndarray:
    """The 12-feature cell encoding of the Architecture B input specification."""
    w, h = state.width, state.height
    n = w * h
    cells = np.zeros((n, 12), dtype=np.float32)
    cells[:, :8] = observe(state, "grid").reshape(8, n).T
    tx, ty = state.tail
    cells[ty * w + tx, 8] = 1
    yy, xx = np.indices((h, w))
    cells[:, 9] = xx.ravel() / max(w - 1, 1)
    cells[:, 10] = yy.ravel() / max(h - 1, 1)
    cells[:, 11] = 1
    return cells


class Node:
    __slots__ = ("game", "children", "visits", "value_sum")

    def __init__(self, game: SnakeGame) -> None:
        self.game, self.children, self.visits, self.value_sum = game, None, 0, 0.0


FATAL = object()


def puct_without_network(live: SnakeGame, simulations: int, seed: int, c: float = 1.5) -> int:
    """Masked PUCT with uniform priors and a constant 0.5 leaf value.

    Measures tree and simulator overhead only. The root is reseeded, so sampled
    fruits never come from the live game's random stream.
    """
    root_game = live.clone()
    root_game.reseed(seed)
    root = Node(root_game)
    for _ in range(simulations):
        node, path = root, [root]
        while True:
            game = node.game
            if game.done:
                value = 1.0 if game.won else 0.0
                break
            if node.children is None:
                heading = game.direction
                node.children = [None if game.is_safe(heading.turn(a)) else FATAL for a in range(3)]
                value = 0.5 if None in node.children else 0.0
                break
            if all(child is FATAL for child in node.children):
                value = 0.0
                break
            scale = c / 3 * math.sqrt(node.visits)
            best, best_score = 0, -math.inf
            for a, child in enumerate(node.children):
                if child is FATAL:
                    continue
                n = child.visits if child is not None else 0
                q = child.value_sum / n if n else 0.5
                if q + scale / (1 + n) > best_score:
                    best, best_score = a, q + scale / (1 + n)
            child = node.children[best]
            if child is None:
                child_game = game.clone()
                child_game.step_relative(RelativeAction(best))
                child = node.children[best] = Node(child_game)
            node = child
            path.append(node)
        for visited in path:
            visited.visits += 1
            visited.value_sum += value
    return max(range(3), key=lambda a: getattr(root.children[a], "visits", -1))


def timed(fn: Callable[[], object], repeats: int) -> float:
    """Best-of-five mean time per call, in microseconds."""
    best = math.inf
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        best = min(best, (time.perf_counter() - start) / repeats)
    return round(best * 1e6, 2)


def costs() -> dict:
    """Per-operation costs on states taken from Hamiltonian games."""
    rows = []
    for side, occupancies in ((20, (0.1, 0.5, 0.9)), (10, (0.5,))):
        game, policy = SnakeGame(GameConfig(side, side), seed=0), HamiltonianPolicy()
        for occupancy in occupancies:
            while game.length < occupancy * side * side:
                game.step(policy(game.state()))
            state = game.state()
            move = policy(state)

            def clone_and_step(g: SnakeGame = game, m: Direction = move) -> None:
                child = g.clone()
                child.reseed(1)
                child.step(m)

            search = {}
            for simulations in (64, 256):
                start = time.perf_counter()
                for r in range(10):
                    puct_without_network(game, simulations, seed=r)
                search[str(simulations)] = round((time.perf_counter() - start) / (10 * simulations) * 1e6, 2)
            rows.append(
                {
                    "board": [side, side],
                    "occupancy": occupancy,
                    "length": game.length,
                    "us_clone_reseed_step": timed(clone_and_step, 2000),
                    "us_state": timed(game.state, 2000),
                    "us_safe_directions": timed(game.safe_directions, 5000),
                    "us_observe_grid": timed(lambda s=state: observe(s, "grid"), 1000),
                    "us_encode_cells_spec": timed(lambda s=state: encode_cells(s), 1000),
                    "us_from_state_with_seed": timed(lambda s=state: SnakeGame.from_state(s, seed=3), 1000),
                    "us_survival_certificate": timed(
                        lambda s=state: Board(s).has_survival_certificate(), 300
                    ),
                    "us_per_simulation_puct_without_network": search,
                }
            )
    return {"states": rows}


def count_paths(side: int) -> list[int]:
    """Self-avoiding head-to-tail paths of each length (index) on a ``side`` × ``side`` board."""
    cells = side * side
    neighbours = []
    for c in range(cells):
        x, y = c % side, c // side
        steps = [(x + DX[d], y + DY[d]) for d in range(4)]
        neighbours.append([ny * side + nx for nx, ny in steps if 0 <= nx < side and 0 <= ny < side])
    by_length = [0] * (cells + 1)

    def extend(cell: int, used: int, length: int) -> None:
        by_length[length] += 1
        for nxt in neighbours[cell]:
            if not used >> nxt & 1:
                extend(nxt, used | 1 << nxt, length + 1)

    for start in range(cells):
        extend(start, 1 << start, 1)
    return by_length


def exact_state_space() -> dict:
    """Running states of small boards: head-to-tail body paths times fruit cells.

    A length-one snake's heading is not counted. An exact solver needs these
    states (times the remaining fruit-timeout budget, if one is part of the task).
    """
    result = {}
    for side in (4, 5):
        cells = side * side
        by_length = count_paths(side)
        result[f"{side}x{side}"] = {
            "body_paths": sum(by_length[1:cells]),
            "states_with_fruit": sum(by_length[k] * (cells - k) for k in range(1, cells)),
            "board_filling_paths": by_length[cells],
        }
    return result


def network_cost_model() -> dict:
    """Analytic forward-pass cost of the proposed encoders (multiply-adds count as 2 FLOPs)."""

    def transformer(tokens: int, d: int = 128, layers: int = 6, ffn: int = 512, heads: int = 8) -> dict:
        per_layer = 8 * tokens * d * d + 4 * tokens * tokens * d + 4 * tokens * d * ffn
        bias_mlp = 2 * tokens * tokens * (5 * 32 + 32 * heads)  # 5 -> 32 -> heads, shared by all layers
        return {
            "gflop": round((layers * per_layer + bias_mlp + 2 * tokens * 12 * d) / 1e9, 4),
            "attention_matrix_mb_per_layer_fp32": round(heads * tokens * tokens * 4 / 2**20, 2),
        }

    def cnn(cells: int, channels: int = 64, convs: int = 21, planes: int = 11) -> dict:
        flops = 2 * cells * 9 * (planes * channels + (convs - 1) * channels * channels)
        return {"gflop": round(flops / 1e9, 4)}

    rows = []
    for side in (4, 6, 8, 10, 12, 20):
        cells = side * side
        free = cells - 3
        rows.append(
            {
                "board": [side, side],
                "tokens": cells + 1,
                "hamiltonian_expected_moves": free * (free + 3) / 4,
                "transformer_6x128": transformer(cells + 1),
                "cnn_64x21": cnn(cells),
            }
        )
    return {"encoders": rows}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sections = {
        "certificate_validation": certificate_validation,
        "expert_lookup": expert_lookup,
        "game_lengths": game_lengths,
        "branching_profile": branching_profile,
        "death_horizons": death_horizons,
        "uncertified_moves": uncertified_moves,
        "guarded_greedy_outcomes": guarded_greedy_outcomes,
        "recovery": recovery,
        "costs": costs,
        "network_cost_model": network_cost_model,
        "exact_state_space": exact_state_space,
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    result: dict = {
        "repository_commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": {"platform": platform.platform(), "cpus": os.cpu_count()},
        "purpose": "Evidence for the Architecture B review; analysis instruments, no model training",
    }
    for name, fn in sections.items():
        start = time.perf_counter()
        result[name] = fn()
        print(f"{name}: {time.perf_counter() - start:.1f}s", flush=True)
    output = Path(__file__).with_name("architecture_b_probe_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
