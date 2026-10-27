"""Assess a trained network against the M1 gates: T1, T2, and the exact 4x4 values.

Both agents are measured, as the M1 gate asks: the search
(:class:`~snake_agent.search.SearchAgent`, no Gumbel noise) and the raw policy
(:class:`~snake_agent.search.PolicyAgent`, the same network without search).
Games run through the evaluation harness on held-out seeds, split across
worker processes:

* **T1**, standard starts: completion and mean moves, gated against the shortcut
  cycle (:data:`T1_REFERENCES`). On boards up to 6x6, the exact envelope
  solution also scores every decision: the expected moves it adds to the
  envelope's optimum (:func:`envelope_regret`).
* **T2**, recovery starts: win rates per occupancy band, against the reference
  policies (and on 4x4 the exact optimum).
* **4x4 values**: the network's win probabilities against the exact solution,
  on the states the search visits in T2 games and on a uniform sample of the
  states outside the envelope (:func:`calibration`), the exact regret of both
  agents' moves, and the raw policy's exact expected moves on T1.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

import numpy as np
import torch

from snake_game import Direction, GameState, RelativeAction, SnakeGame

from .contract import TRAINING_RULES, Rules
from .encoding import canonical_envelope
from .evaluation import (
    AgentSpec,
    Evaluation,
    GameRecord,
    Provenance,
    best_rates,
    evaluate_recovery,
    evaluate_standard,
    run_game,
    standard_gate,
)
from .exact import ExactSolution, evaluate_policy, solve, solve_envelope
from .network import NetworkEvaluator, from_bytes
from .recovery import RecoveryStart, held_out_starts
from .references import REFERENCES
from .search import EVALUATION, PolicyAgent, Query, SearchAgent, SearchConfig, envelope_certain
from .seeds import VALIDATION, SeedRange, recovery_fruit_seeds, standard_seeds

T1_REFERENCES = {(4, 4): 43.89, (6, 6): 214.0, (10, 10): 1445.0, (20, 20): 20737.0}
"""Mean moves of the shortcut cycle on standard starts: the T1 gate. 4x4 is its exact expectation
(43.888); 6x6, 10x10 and 20x20 are the probe's means (exact on 6x6: 212.78)."""

AGENTS = ("search", "policy")
T2_REFERENCES = ("hamiltonian", "guarded-greedy", "greedy")
"""The reference policies of track T2 (the shortcut cycle needs an aligned start)."""


def agent_spec(
    weights: bytes, kind: str, rules: Rules = TRAINING_RULES, search: SearchConfig = EVALUATION
) -> AgentSpec:
    """The network as a harness agent: ``"search"`` (with the ``search`` settings) or ``"policy"`` (raw)."""
    evaluator = NetworkEvaluator(from_bytes(weights))
    if kind == "search":
        factory = lambda seed: SearchAgent(evaluator, rules, search, seed)  # noqa: E731
    elif kind == "policy":
        factory = lambda seed: PolicyAgent(evaluator, rules)  # noqa: E731
    else:
        raise ValueError(f"unknown agent {kind!r}: use {' or '.join(AGENTS)}")
    return AgentSpec(f"m1-{kind}", factory, Provenance.SELF_PLAY)


def _t1_task(args: tuple) -> tuple[list[GameRecord], list[tuple[GameState, RelativeAction]]]:
    weights, kind, width, height, seeds, rules, search, record = args
    torch.set_num_threads(1)
    spec = agent_spec(weights, kind, rules, search)
    recorders: list[_Recorder] = []
    if record:
        make = spec.factory

        def factory(seed: int) -> _Recorder:
            recorders.append(_Recorder(make(seed)))
            return recorders[-1]

        spec = dataclasses.replace(spec, factory=factory)
    records = list(evaluate_standard(spec, width, height, seeds, rules).records)
    return records, [move for recorder in recorders for move in recorder.moves]


def _t2_task(args: tuple) -> list[GameRecord]:
    weights, kind, starts, fruit_seeds, rules, search = args
    torch.set_num_threads(1)
    spec = REFERENCES[kind] if kind in REFERENCES else agent_spec(weights, kind, rules, search)
    return list(evaluate_recovery(spec, starts, fruit_seeds, rules).records)


def _chunks(items: Sequence, parts: int) -> list[Sequence]:
    size = max(1, -(-len(items) // parts))
    return [items[i : i + size] for i in range(0, len(items), size)]


def t1(
    pool: Executor,
    weights: bytes,
    kind: str,
    width: int,
    height: int,
    games: int,
    split: SeedRange = VALIDATION,
    rules: Rules = TRAINING_RULES,
    workers: int = 8,
    search: SearchConfig = EVALUATION,
    decisions: list[tuple[GameState, RelativeAction]] | None = None,
) -> Evaluation:
    """Track T1 for ``games`` held-out standard starts, the games split across ``pool``.

    The agent's moves, with the states they were played in, are appended to ``decisions`` if given.
    """
    seeds = list(standard_seeds(split, games))
    record = decisions is not None
    tasks = [
        (weights, kind, width, height, chunk, rules, search, record) for chunk in _chunks(seeds, workers)
    ]
    records = []
    for part, moves in pool.map(_t1_task, tasks):
        records += part
        if record:
            decisions += moves
    return Evaluation("T1", f"m1-{kind}", Provenance.SELF_PLAY, (width, height), rules, tuple(records))


def envelope_regret(
    solution: ExactSolution, decisions: Sequence[tuple[GameState, RelativeAction]]
) -> dict[str, Any]:
    """The expected moves each decision adds to the envelope's optimum, by the exact envelope ``solution``.

    Only decisions with a choice count (at least two envelope moves); a move outside the envelope or a
    state outside it would be a T1 failure, which the gate reports.
    """
    extra = []
    for state, action in decisions:
        try:
            costs = solution.action_moves(state)
        except ValueError:  # out of cycle order
            continue
        if sum(math.isfinite(c) for c in costs) >= 2:
            extra.append(costs[action] - min(costs))
    extra_ = np.array(extra)
    return {
        "optimum": round(solution.start_moves(), 4),
        "decisions": len(extra),
        "extra_moves": round(float(extra_.mean()), 5) if extra else None,
        "optimal_share": round(float((extra_ < 1e-9).mean()), 4) if extra else None,
    }


def t2(
    pool: Executor,
    weights: bytes | None,
    kind: str,
    starts: Sequence[RecoveryStart],
    split: SeedRange = VALIDATION,
    rules: Rules = TRAINING_RULES,
    workers: int = 8,
    search: SearchConfig = EVALUATION,
) -> Evaluation:
    """Track T2 on ``starts`` for the network (``"search"``, ``"policy"``) or a reference policy."""
    fruit_seeds = list(recovery_fruit_seeds(split, len(starts)))
    pairs = list(zip(starts, fruit_seeds, strict=True))
    tasks = [
        (weights, kind, [s for s, _ in chunk], [f for _, f in chunk], rules, search)
        for chunk in _chunks(pairs, workers)
    ]
    records = [r for part in pool.map(_t2_task, tasks) for r in part]
    name = kind if kind in REFERENCES else f"m1-{kind}"
    provenance = REFERENCES[kind].provenance if kind in REFERENCES else Provenance.SELF_PLAY
    board = (starts[0].state.width, starts[0].state.height)
    return Evaluation("T2", name, provenance, board, rules, tuple(records))


# ------------------------------------------------------------- exact values
def network_values(weights: bytes, states: Sequence[GameState], rules: Rules = TRAINING_RULES) -> np.ndarray:
    """The network's win probabilities of ``states`` (raw, without search)."""
    evaluator = NetworkEvaluator(from_bytes(weights))
    values = []
    for i in range(0, len(states), 1024):
        values.append(evaluator([Query(s, rules) for s in states[i : i + 1024]]).win)
    return np.concatenate(values) if values else np.zeros(0)


def calibration(predicted: np.ndarray, exact: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """How well predicted win probabilities match exact ones.

    ``ece``: the expected calibration error, the mean over equal-width bins of
    predictions of the gap between the mean prediction and the mean exact
    value, weighted by the bin's share; ``mae`` and ``brier`` compare state by
    state (``brier``: mean squared error against the exact probability).
    """
    predicted, exact = np.asarray(predicted, dtype=float), np.asarray(exact, dtype=float)
    which = np.minimum((predicted * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        inside = which == b
        if inside.any():
            gap = abs(predicted[inside].mean() - exact[inside].mean())
            ece += inside.mean() * gap
            table.append(
                {
                    "bin": [b / bins, (b + 1) / bins],
                    "states": int(inside.sum()),
                    "predicted": round(float(predicted[inside].mean()), 4),
                    "exact": round(float(exact[inside].mean()), 4),
                }
            )
    return {
        "states": len(predicted),
        "ece": round(float(ece), 5),
        "mae": round(float(np.abs(predicted - exact).mean()), 5),
        "brier": round(float(((predicted - exact) ** 2).mean()), 6),
        "bins": table,
    }


def uniform_free_states(solution: ExactSolution, count: int, seed: int = 0) -> list[GameState]:
    """``count`` running states drawn uniformly (move counters at zero) among those outside the envelope."""
    rng = np.random.default_rng(seed)
    width = solution.width
    states = []
    levels = list(solution._levels.values())
    sizes = np.array([level.free.size for level in levels], dtype=float)
    while len(states) < count:
        level = levels[rng.choice(len(levels), p=sizes / sizes.sum())]
        body = int(rng.integers(len(level.keys)))
        fruit = int(level.free[body, int(rng.integers(level.free.shape[1]))])
        snake = tuple((c % width, c // width) for c in level.bodies[body])
        heading = Direction(int(level.heading[body]))
        state = GameState(width, solution.height, snake, heading, (fruit % width, fruit // width))
        if not envelope_certain(state, TRAINING_RULES):
            states.append(state)
    return states


class _Recorder:
    """Wraps an agent and keeps the states it saw and the moves it chose."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.moves: list[tuple[GameState, RelativeAction]] = []

    def __call__(self, state: GameState) -> RelativeAction:
        action = self.agent(state)
        self.moves.append((state, action))
        return action


def _visited_task(args: tuple) -> list[tuple[GameState, RelativeAction]]:
    weights, kind, starts, fruit_seeds, rules, search = args
    torch.set_num_threads(1)
    spec = agent_spec(weights, kind, rules, search)
    moves = []
    for i, (start, seed) in enumerate(zip(starts, fruit_seeds, strict=True)):
        recorder = _Recorder(spec.factory(i))
        run_game(SnakeGame.from_state(start.state, seed=seed), recorder, rules)
        moves += recorder.moves
    return moves


def exact_4x4(
    pool: Executor,
    weights: bytes,
    starts: Sequence[RecoveryStart],
    split: SeedRange = VALIDATION,
    rules: Rules = TRAINING_RULES,
    workers: int = 8,
    uniform: int = 5000,
    games: int = 100,
    solution: ExactSolution | None = None,
    search: SearchConfig = EVALUATION,
) -> dict[str, Any]:
    """The network's values and both agents' moves on 4x4 against the exact solution.

    Calibration of the raw win probabilities on the states each agent visits in
    the T2 games (outside the envelope) and on ``uniform`` states drawn
    uniformly outside the envelope; the exact regret (win probability lost) of
    each agent's moves outside the envelope; the exact optimum of the T2
    starts; and the raw policy's exact expected moves from the standard start
    (it is a function of the state), against the envelope's optimum.
    """
    solution = solve(4, 4) if solution is None else solution
    fruit_seeds = list(recovery_fruit_seeds(split, len(starts)))
    pairs = list(zip(starts, fruit_seeds, strict=True))
    report: dict[str, Any] = {
        "t2_optimum": round(statistics.fmean(solution.value(s.state) for s in starts), 5),
    }
    for kind in AGENTS:
        tasks = [
            (weights, kind, [s for s, _ in chunk], [f for _, f in chunk], rules, search)
            for chunk in _chunks(pairs, workers)
        ]
        moves = [m for part in pool.map(_visited_task, tasks) for m in part]
        free = [(s, a) for s, a in moves if not envelope_certain(s, rules)]
        regrets = np.array([solution.regret(s, a) for s, a in free])
        states = [s for s, _ in free]
        report[kind] = {
            "decisions": len(free),
            "mean_regret": round(float(regrets.mean()), 6) if len(free) else None,
            "optimal_share": round(float((regrets == 0.0).mean()), 4) if len(free) else None,
            "calibration": calibration(
                network_values(weights, states, rules), np.array([solution.value(s) for s in states])
            )
            if states
            else None,
        }
    sample = uniform_free_states(solution, uniform)
    report["uniform"] = calibration(
        network_values(weights, sample, rules), np.array([solution.value(s) for s in sample])
    )

    envelope = canonical_envelope(4, 4)
    policy = PolicyAgent(NetworkEvaluator(from_bytes(weights)), rules)
    report["t1_policy_expected_moves"] = round(evaluate_policy(4, 4, policy, envelope=envelope).moves, 4)
    return report


def assess(
    pool: Executor,
    weights: bytes,
    width: int,
    height: int,
    *,
    games: int = 100,
    sources: int = 60,
    split: SeedRange = VALIDATION,
    rules: Rules = TRAINING_RULES,
    workers: int = 8,
    references: dict[float, float] | None = None,
    search: SearchConfig = EVALUATION,
) -> dict[str, Any]:
    """T1 and T2 for both agents (and the exact 4x4 values), as a JSON-ready report with the T1 gate.

    ``search`` holds the search agent's settings (simulations per move, moves unit).
    """
    report: dict[str, Any] = {
        "board": [width, height],
        "split": split.name,
        "games": games,
        "simulations": search.simulations,
        "moves_unit": search.moves_unit,
    }
    reference = T1_REFERENCES.get((width, height))
    exact = width * height <= 36 and canonical_envelope(width, height) is not None
    inside = solve_envelope(width, height) if exact else None
    for kind in AGENTS:
        decisions: list[tuple[GameState, RelativeAction]] | None = [] if exact else None
        evaluation = t1(pool, weights, kind, width, height, games, split, rules, workers, search, decisions)
        entry: dict[str, Any] = {"summary": evaluation.summary().to_dict()}
        if reference is not None:
            gate = standard_gate(evaluation, reference)
            entry["gate"] = {"reference": reference, "passed": gate.passed, "reasons": list(gate.reasons)}
        if inside is not None:
            entry["exact"] = envelope_regret(inside, decisions)
        report[f"t1_{kind}"] = entry
    starts = held_out_starts(width, height, sources, split)
    if starts:
        if references is None:
            references = best_rates(
                t2(pool, None, name, starts, split, rules, workers) for name in T2_REFERENCES
            )
        report["t2_references"] = {str(band): rate for band, rate in sorted(references.items())}
        for kind in AGENTS:
            evaluation = t2(pool, weights, kind, starts, split, rules, workers, search)
            report[f"t2_{kind}"] = {
                str(band): evaluation.summary(band).to_dict() for band in evaluation.bands()
            } | {"all": evaluation.summary().to_dict()}
        if (width, height) == (4, 4):
            report["exact"] = exact_4x4(
                pool, weights, starts, split, rules, workers, games=games, search=search
            )
    return report
