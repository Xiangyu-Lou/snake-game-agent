"""Reproduce the Architecture B probe's reference numbers with this package.

``python -m snake_agent reproduce`` replays the probe's reference games through
the evaluation harness and the instruments and compares the numbers with
``analysis/architecture_b_probe_results.json``. ``--full`` adds the slow
sections: the 20x20 games, the death horizons and the move profiles. Outcome
names follow the probe, where a censored game is ``looped`` (a repeated state)
or ``stalled`` (too long without fruit).

One difference is deliberate. :func:`~snake_agent.survival.prove_lost` refuses
a proof whose collisions come only after as many moves as there are free
cells, because the real snake could fill the board first; the probe's search
accepted such proofs. Three of the probe's 565 "doomed" guarded-greedy moves
are of that kind, and the uncertified-moves section accounts for them.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snake_game import Direction, GameConfig, GameState, SnakeGame
from snake_game.policies import GreedyPolicy, HamiltonianPolicy, RandomPolicy

from . import exact
from .contract import NO_LIMIT
from .cycles import ShortcutCyclePolicy
from .evaluation import STALL_FACTOR, AgentSpec, GameRecord, evaluate_recovery, evaluate_standard, run_game
from .recovery import recovery_starts
from .references import REFERENCES
from .survival import Board, MoveClass, classify_moves, neighbour_table, prove_lost

PROBE_RESULTS = Path(__file__).resolve().parents[1] / "analysis" / "architecture_b_probe_results.json"
BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0))
"""The probe's occupancy bands for move profiles."""


@dataclass(frozen=True)
class Check:
    """One reproduced number or table."""

    section: str
    name: str
    expected: Any
    actual: Any

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


def probe_name(record: GameRecord) -> str:
    """The probe's outcome name: ``won``, ``lost``, ``looped`` or ``stalled``."""
    return {"loop": "looped", "stall": "stalled"}.get(record.detail or "", record.outcome.value)


def _games(
    spec: AgentSpec, width: int, height: int, seeds: Sequence[int]
) -> list[tuple[GameRecord, list[GameState]]]:
    """Each game's record and the states before its moves, played as the probe did (no timeout)."""
    games = []
    for seed in seeds:
        agent = spec.factory(0)
        states: list[GameState] = []
        record = run_game(
            SnakeGame(GameConfig(width, height), seed=seed),
            agent,
            NO_LIMIT,
            memory=None if spec.memory is None else (lambda agent=agent: spec.memory(agent)),
            detect_loops=spec.deterministic,
            stall_limit=STALL_FACTOR * width * height,
            record=states,
            seed=seed,
        )
        games.append((record, states))
    return games


# ------------------------------------------------------------------ sections
def shortcut_cycle(probe: dict, full: bool) -> list[Check]:
    """T1 speed reference: the shortcut cycle (6x6 and 10x10 at the 50% cutoff unless ``full``)."""
    checks = []
    for side, games in ((6, 200), (10, 100), (20, 20)) if full else ((6, 200), (10, 100)):
        for row in probe["shortcut_baseline"][f"{side}x{side}"]["cutoffs"]:
            cutoff = row["max_fill"]
            if not full and cutoff != 0.5:
                continue
            spec = AgentSpec(
                f"shortcut {cutoff}", lambda seed, c=cutoff: ShortcutCyclePolicy(c), deterministic=True
            )
            evaluation = evaluate_standard(spec, side, side, range(games), NO_LIMIT)
            moves = [r.moves for r in evaluation.records]
            free = side * side - 3
            actual = {
                "max_fill": cutoff,
                "games": len(moves),
                "won": evaluation.summary().won,
                "mean_moves": round(statistics.mean(moves), 1),
                "median_moves": statistics.median(moves),
                "max_moves": max(moves),
                "share_of_cycle_expectation": round(statistics.mean(moves) / (free * (free + 3) / 4), 3),
                "cycle_order_checks": sum(moves) if side <= 10 else 0,
            }
            name = f"{side}x{side}, jumps below {cutoff:.0%} fill"
            checks.append(Check("shortcut cycle", name, row, actual))
            checks.append(
                Check("shortcut cycle", f"{name}: envelope exits", 0, evaluation.summary().envelope_exits)
            )
    return checks


def plain_cycle(probe: dict, full: bool) -> list[Check]:
    """Hamiltonian game lengths; with ``full`` also the six 20x20 expert games."""
    checks = []
    for row in probe["game_lengths"]:
        side = row["board"][0]
        evaluation = evaluate_standard(
            REFERENCES["hamiltonian"], side, side, range(row["episodes"]), NO_LIMIT
        )
        moves = [r.moves for r in evaluation.records if probe_name(r) == "won"]
        free = side * side - 3
        actual = {
            "board": [side, side],
            "episodes": len(moves),
            "mean_moves": statistics.mean(moves),
            "expected_moves": free * (free + 3) / 4,
            "bound_moves": free * (free + 1) // 2,
        }
        checks.append(Check("plain cycle", f"{side}x{side} games won and mean moves", row, actual))
    if full:  # the expert is a lookup table keyed by the head cell
        expected = probe["expert_lookup"]
        table: dict = {}
        conflicts = 0

        def recording(seed: int) -> Any:
            policy = HamiltonianPolicy()

            def agent(state: GameState) -> Direction:
                nonlocal conflicts
                action = policy(state)
                conflicts += table.setdefault(state.head, action) != action
                return action

            agent.policy = policy
            return agent

        spec = AgentSpec("hamiltonian", recording, deterministic=True, memory=lambda a: id(a.policy.cycle))
        evaluation = evaluate_standard(spec, 20, 20, expected["seeds"], NO_LIMIT)
        moves = [r.moves for r in evaluation.records]
        actual = {
            "board": [20, 20],
            "seeds": list(expected["seeds"]),
            "decisions": sum(moves),
            "distinct_head_cells": len(table),
            "conflicting_labels": conflicts,
            "moves_per_game": moves,
        }
        checks.append(Check("plain cycle", "20x20 expert decisions by head cell", expected, actual))
    return checks


def guarded_greedy(probe: dict, full: bool) -> list[Check]:
    """Guarded-greedy games without a timeout: wins, losses, loops and stalls."""
    checks = []
    for side, games in ((10, 30), (20, 6)) if full else ((10, 30),):
        expected = probe["guarded_greedy_outcomes"][f"{side}x{side}"]
        evaluation = evaluate_standard(REFERENCES["guarded-greedy"], side, side, range(games), NO_LIMIT)
        episodes = [
            {"seed": r.seed, "outcome": probe_name(r), "score": r.fruits, "moves": r.moves}
            for r in evaluation.records
        ]
        summary = (
            dict(Counter(e["outcome"] for e in episodes)),
            statistics.mean(e["score"] for e in episodes),
        )
        checks.append(Check("guarded greedy", f"{side}x{side} games", expected["episodes"], episodes))
        checks.append(
            Check(
                "guarded greedy",
                f"{side}x{side} outcomes and mean score",
                (expected["outcomes"], expected["mean_score"]),
                summary,
            )
        )
    return checks


def recovery(probe: dict, full: bool) -> list[Check]:
    """T2 references: the 10x10 recovery starts and the reference policies' results on them."""
    expected = probe["recovery"]
    starts = recovery_starts(10, 10, range(100, 160))
    counts = Counter(str(start.band) for start in starts)
    checks = [Check("recovery", "starts per band", expected["start_states"], dict(counts))]
    seeds = [10_000 + i for i in range(len(starts))]
    for name in ("hamiltonian", "guarded-greedy", "greedy"):
        evaluation = evaluate_recovery(REFERENCES[name], starts, seeds, NO_LIMIT)
        by_band: dict[str, Counter] = {}
        for r in evaluation.records:
            by_band.setdefault(str(r.band), Counter())[probe_name(r)] += 1
        actual = {band: dict(c) for band, c in sorted(by_band.items())}
        checks.append(
            Check("recovery", f"{name} outcomes per band", expected[name.replace("-", "_")], actual)
        )
    return checks


def certificate_validation(probe: dict, full: bool) -> list[Check]:
    """The instruments against the engine: move safety, certificate loops and lost proofs."""
    expected = probe["certificate_validation"]
    mismatches = checked = 0
    for seed in range(30):
        game, policy = SnakeGame(GameConfig(7, 6), seed=seed), RandomPolicy(seed=seed)
        while not game.done:
            board = Board(game.state())
            for d in Direction:
                checked += 1
                mismatches += (board.target(int(d)) >= 0) != game.is_safe(d)
            game.step(policy(game.state()))

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
                    loop = board.certificate_loop()
                    if loop is None or len(loop) < state.length:
                        raise AssertionError(
                            f"no loop of at least the snake's length behind the certificate of {state}"
                        )
                    for step in range(2 * len(loop)):
                        cell = loop[step % len(loop)]
                        d = board.neighbours[board.head].index(cell)
                        if board.target(d) != cell:
                            raise AssertionError(f"the certificate loop of {state} collides")
                        board.move(d, cell)
                    replayed += 1
                game.step(policy(state))

    rng = random.Random(0)

    def all_die(game: SnakeGame, depth: int) -> bool:
        """Whether every real continuation (growth and random fruit included) dies within ``depth`` moves."""
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

    lost = confirmed = 0
    for seed in range(40):
        states: list[GameState] = []
        game = SnakeGame(GameConfig(6, 6), seed=seed)
        run_game(game, RandomPolicy(seed=seed), NO_LIMIT, stall_limit=STALL_FACTOR * 36, record=states)
        for state in states[-12:]:
            if prove_lost(Board(state), 200_000).lost:
                lost += 1
                confirmed += all_die(SnakeGame.from_state(state, seed=seed), 40)
    return [
        Check(
            "certificate validation",
            "Board.target against SnakeGame.is_safe",
            (expected["target_vs_is_safe_checks"], expected["target_vs_is_safe_mismatches"]),
            (checked, mismatches),
        ),
        Check(
            "certificate validation",
            "certificates replayed for two loop periods",
            expected["certificates_replayed_for_two_loop_periods"],
            replayed,
        ),
        Check(
            "certificate validation",
            "proven-lost states that die in the engine within 40 moves",
            (expected["doomed_states_checked_with_engine_depth_40"], expected["doomed_states_confirmed"]),
            (lost, confirmed),
        ),
    ]


def state_space(probe: dict, full: bool) -> list[Check]:
    """Body and state counts of 4x4, the size of the exact solution."""
    bodies = exact._bodies(4, 4)  # lengths 2 to 15; the probe also counts the 16 one-cell bodies
    neighbours = neighbour_table(4, 4)
    filling = 0
    for body in bodies[15]:
        (free,) = set(range(16)) - set(body)
        filling += free in neighbours[body[-1]]
    actual = {
        "body_paths": 16 + sum(len(b) for b in bodies.values()),
        "states_with_fruit": 16 * 15 + sum(len(b) * (16 - length) for length, b in bodies.items()),
        "board_filling_paths": filling,
    }
    checks = [
        Check(
            "exact state space",
            "4x4 bodies, states and board-filling bodies (from the solver's enumeration)",
            probe["exact_state_space"]["4x4"],
            actual,
        )
    ]
    if full:
        by_length = _count_bodies(5, 5)
        actual = {
            "body_paths": sum(by_length[1:25]),
            "states_with_fruit": sum(count * (25 - k) for k, count in enumerate(by_length[:25])),
            "board_filling_paths": by_length[25],
        }
        checks.append(
            Check("exact state space", "5x5 bodies and states", probe["exact_state_space"]["5x5"], actual)
        )
    return checks


def _count_bodies(width: int, height: int) -> list[int]:
    """Number of head-to-tail bodies of each length (the index), one-cell bodies included."""
    cells = width * height
    neighbours = neighbour_table(width, height)
    by_length = [0] * (cells + 1)

    def extend(cell: int, used: int, length: int) -> None:
        by_length[length] += 1
        for nxt in neighbours[cell]:
            if nxt >= 0 and not used >> nxt & 1:
                extend(nxt, used | 1 << nxt, length + 1)

    for start in range(cells):
        extend(start, 1 << start, 1)
    return by_length


def uncertified_moves(probe: dict, full: bool) -> list[Check]:
    """Safe moves without a certificate: survivable, proven lost or unknown."""
    labels = {True: "uncertified_doomed", False: "uncertified_survivable", None: "unknown"}
    checks = []
    for name, spec in (("greedy", REFERENCES["greedy"]), ("guarded_greedy", REFERENCES["guarded-greedy"])):
        counts: Counter = Counter()
        refused = 0  # proofs refused because the snake might fill the board first
        for _, states in _games(spec, 10, 10, range(10)):
            for state in states:
                board = Board(state)
                fruit = state.fruit[1] * 10 + state.fruit[0]
                for d in range(4):
                    cell = board.target(d)
                    if cell < 0:
                        continue
                    undo = board.move(d, cell, cell == fruit)
                    if board.length == board.cells:
                        counts["won"] += 1
                    elif board.has_survival_certificate():
                        counts["certified"] += 1
                    else:
                        proof = prove_lost(board, 100_000)
                        counts[labels[proof.lost]] += 1
                        refused += proof.lost is None and proof.nodes <= 100_000
                    board.unmove(cell, undo)
        as_probe = dict(counts)  # the probe counted the refused proofs as doomed
        if refused:
            as_probe["uncertified_doomed"] += refused
            as_probe["unknown"] -= refused
        expected = probe["uncertified_moves"][f"{name}_10x10_seeds0-9"]
        checks.append(
            Check("uncertified moves", f"{name} 10x10, refused proofs counted as doomed", expected, as_probe)
        )
        checks.append(
            Check(
                "uncertified moves",
                f"{name} 10x10 proofs refused",
                3 if name == "guarded_greedy" else 0,
                refused,
            )
        )
    return checks


def death_horizons(probe: dict, full: bool) -> list[Check]:
    """How many moves before each greedy death the game was proven lost."""
    checks = []
    for (width, height), seeds, budget in (((10, 10), range(30), 1_000_000), ((20, 20), range(6), 300_000)):
        rows = []
        for record, states in _games(REFERENCES["greedy"], width, height, seeds):
            if probe_name(record) != "lost":
                raise AssertionError(f"greedy game {record.seed} did not end in a collision")
            proof_nodes, back, verdict = None, 1, None
            while back <= len(states):
                proof = prove_lost(Board(states[-back]), budget)
                verdict = proof.lost
                if verdict is not True:
                    break
                proof_nodes = proof.nodes
                back += 1
            rows.append(
                {
                    "seed": record.seed,
                    "score": record.fruits,
                    "horizon_moves": back - 1,
                    "lower_bound": verdict is None,
                    "proof_nodes": proof_nodes,
                    "occupancy_when_doomed": round(states[-(back - 1)].length / (width * height), 3),
                }
            )
        expected = probe["death_horizons"][f"{width}x{height}"]["episodes"]
        checks.append(Check("death horizons", f"{width}x{height} greedy deaths", expected, rows))
    return checks


def branching(probe: dict, full: bool) -> list[Check]:
    """Share of states with at most one safe or certified move, by occupancy."""
    checks = []
    runs = {
        "hamiltonian_20x20_seed0": ((20, 20), REFERENCES["hamiltonian"], range(1)),
        "hamiltonian_10x10_seeds0-9": ((10, 10), REFERENCES["hamiltonian"], range(10)),
        "guarded_greedy_10x10_seeds0-9": ((10, 10), REFERENCES["guarded-greedy"], range(10)),
    }
    for name, ((width, height), spec, seeds) in runs.items():
        counts = [Counter() for _ in BANDS]
        for _, states in _games(spec, width, height, seeds):
            for state in states:
                classes = [c for c in classify_moves(state) if c is not MoveClass.FATAL]
                occupancy = state.length / (width * height)
                c = counts[
                    next(i for i, (_, hi) in enumerate(BANDS) if occupancy < hi or i == len(BANDS) - 1)
                ]
                c["states"] += 1
                c["at_most_one_safe"] += len(classes) <= 1
                c["at_most_one_certified"] += sum(cl is not MoveClass.UNKNOWN for cl in classes) <= 1
        total = sum(c["states"] for c in counts)
        actual = {
            "bands": [
                {
                    "occupancy": [lo, hi],
                    "states": c["states"],
                    "at_most_one_safe": round(c["at_most_one_safe"] / c["states"], 4),
                    "at_most_one_certified": round(c["at_most_one_certified"] / c["states"], 4),
                }
                for (lo, hi), c in zip(BANDS, counts, strict=True)
                if c["states"]
            ],
            "states": total,
            "at_most_one_safe": round(sum(c["at_most_one_safe"] for c in counts) / total, 4),
            "at_most_one_certified": round(sum(c["at_most_one_certified"] for c in counts) / total, 4),
        }
        checks.append(Check("branching", name, probe["branching_profile"][name], actual))
    return checks


SECTIONS: dict[str, tuple[Callable[[dict, bool], list[Check]], bool]] = {
    "shortcut cycle": (shortcut_cycle, False),
    "plain cycle": (plain_cycle, False),
    "guarded greedy": (guarded_greedy, False),
    "recovery": (recovery, False),
    "certificate validation": (certificate_validation, False),
    "exact state space": (state_space, False),
    "uncertified moves": (uncertified_moves, True),
    "death horizons": (death_horizons, True),
    "branching": (branching, True),
}
"""Section name -> (function, only with ``--full``)."""


def load_probe(path: Path = PROBE_RESULTS) -> dict:
    """The committed probe results; they live in the repository, next to the probe."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: reproducing the probe needs a checkout of the repository")
    return json.loads(path.read_text())


def reproduce(
    full: bool = False, probe: dict | None = None, log: Callable[[str], None] | None = None
) -> list[Check]:
    """Run the sections (all of them with ``full``) and return their checks."""
    probe = load_probe() if probe is None else probe
    checks = []
    for name, (section, full_only) in SECTIONS.items():
        if full_only and not full:
            continue
        start = time.perf_counter()
        result = section(probe, full)
        checks += result
        if log is not None:
            for check in result:
                log(f"{'ok  ' if check.ok else 'FAIL'}  {check.section}: {check.name}")
                if not check.ok:
                    log(f"      expected {check.expected!r}\n      got      {check.actual!r}")
            log(f"      {name}: {time.perf_counter() - start:.1f} s")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m snake_agent reproduce", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--full", action="store_true", help="also run the slow sections (several minutes)")
    args = parser.parse_args(argv)
    checks = reproduce(args.full, log=lambda line: print(line, flush=True))
    failed = sum(not c.ok for c in checks)
    print(f"{len(checks) - failed} of {len(checks)} checks match the probe")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
