"""Assess a trained network against the M1 and M2 gates: T1, T2, and the exact 4x4 values.

Both agents are measured, as the gates ask: the search
(:class:`~snake_agent.search.SearchAgent`, no Gumbel noise) and the raw policy
(:class:`~snake_agent.search.PolicyAgent`, the same network without search).
Games run on held-out seeds with the evaluation harness's bookkeeping
(:class:`~snake_agent.evaluation.GameRun`), split across worker processes; each
worker plays its games at once and evaluates their leaves in batches
(:func:`play_lockstep`), on the GPU if there is one:

* **T1**, standard starts: completion and mean moves, gated against the shortcut
  cycle (:data:`T1_REFERENCES`). On boards up to 6x6, the exact envelope
  solution also scores every decision: the expected moves it adds to the
  envelope's optimum (:func:`envelope_regret`), and how well the moves head
  orders the moves the search compares (:func:`sibling_ranking`).
* **T2**, recovery starts: win rates per occupancy band, gated against the best
  reference policy on the same starts (and on 4x4 the exact optimum).
* **4x4 values**: the network's win probabilities against the exact solution,
  on the states the search visits in T2 games and on a uniform sample of the
  states outside the envelope (:func:`calibration`), the exact regret of both
  agents' moves, and the raw policy's exact expected moves on T1.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Generator, Sequence
from concurrent.futures import Executor
from typing import Any

import numpy as np
import torch

from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame

from .contract import TRAINING_RULES, Outcome, Rules, planning_game, to_relative
from .cycles import CycleEnvelope
from .encoding import canonical_envelope, moves_ceiling, moves_scale
from .evaluation import (
    AgentSpec,
    Evaluation,
    GameRecord,
    GameRun,
    Provenance,
    best_rates,
    censoring,
    evaluate_recovery,
    evaluate_standard,
    recovery_gate,
    standard_gate,
)
from .exact import ExactSolution, evaluate_policy, solve, solve_envelope
from .network import NetworkEvaluator, from_bytes
from .recovery import RecoveryStart, recovery_starts
from .references import REFERENCES
from .search import (
    EVALUATION,
    Evaluator,
    LeafValue,
    PolicyAgent,
    Query,
    SearchAgent,
    SearchConfig,
    _Epoch,
    envelope_certain,
    policy_move,
)
from .search import search as root_search
from .seeds import VALIDATION, SeedRange, recovery_fruit_seeds, recovery_source_seeds, standard_seeds
from .stats import mean_interval

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
    return AgentSpec(f"network-{kind}", factory, Provenance.SELF_PLAY)


# ------------------------------------------------------------ batched games
class _Lockstep:
    """One harness game of the network agent, advanced a leaf at a time by :func:`play_lockstep`.

    It plays what :func:`agent_spec`'s agent with the same seed plays, move for move, given the same
    network outputs (a batch on another device can change the last bits of those).
    """

    def __init__(
        self,
        run: GameRun,
        kind: str,
        rules: Rules,
        search: SearchConfig,
        seed: int,
        decisions: list[tuple[GameState, RelativeAction]] | None = None,
    ) -> None:
        if kind not in AGENTS:
            raise ValueError(f"unknown agent {kind!r}: use {' or '.join(AGENTS)}")
        self.run, self.kind, self.rules, self.config = run, kind, rules, search
        self.rng = np.random.default_rng(seed)
        self.epoch = _Epoch()
        self.decisions = decisions
        self.moving: Generator[Query, LeafValue, Any] | None = None
        self.state: GameState | None = None
        self.value: LeafValue | None = None

    def advance(self) -> Query | None:
        """Play until the agent needs an evaluation (returned) or the game is over (None)."""
        while True:
            if self.moving is None:
                self.state = self.run.next_state()
                if self.state is None:
                    return None
                history = self.epoch.observe(self.state)
                if self.kind == "search":
                    self.moving = root_search(self.state, self.rules, self.config, self.rng, history)
                else:
                    self.moving = policy_move(self.state, self.rules, history)
            value, self.value = self.value, None
            try:
                return self.moving.send(value)
            except StopIteration as stop:
                self.moving = None
                action = stop.value.action if self.kind == "search" else stop.value
                if self.decisions is not None:
                    self.decisions.append((self.state, action))
                self.run.play(action)


def play_lockstep(evaluator: Evaluator, games: Sequence[_Lockstep]) -> list[GameRecord]:
    """Play ``games`` to the end at once, evaluating all their pending leaves in one batch per round."""
    waiting = list(games)
    while waiting:
        asking, queries = [], []
        for game in waiting:
            query = game.advance()
            if query is not None:
                asking.append(game)
                queries.append(query)
        if queries:
            prediction = evaluator(queries)
            for i, game in enumerate(asking):
                game.value = prediction.row(i)
        waiting = asking
    return [game.run.result for game in games]


def _runs(
    kind: str, games: Sequence[SnakeGame], rules: Rules, labels: Sequence[dict[str, Any]], envelope: bool
) -> list[GameRun]:
    spec = AgentSpec(f"network-{kind}", lambda seed: None, Provenance.SELF_PLAY)  # censoring only
    return [
        GameRun(
            game,
            rules,
            envelope=CycleEnvelope.for_state(game.state()) if envelope else None,
            **censoring(spec, game, rules),
            **label,
        )
        for game, label in zip(games, labels, strict=True)
    ]


def _t1_task(args: tuple) -> tuple[list[GameRecord], list[tuple[GameState, RelativeAction]]]:
    weights, kind, width, height, seeds, rules, search, record, device = args
    torch.set_num_threads(1)
    decisions: list[tuple[GameState, RelativeAction]] | None = [] if record else None
    games = [SnakeGame(GameConfig(width, height), seed=seed) for seed in seeds]
    runs = _runs(kind, games, rules, [{"seed": seed} for seed in seeds], envelope=True)
    lockstep = [_Lockstep(run, kind, rules, search, i, decisions) for i, run in enumerate(runs)]
    evaluator = NetworkEvaluator(from_bytes(weights), device)
    records = play_lockstep(evaluator, lockstep)
    evaluator.release()
    return records, decisions or []


def _t2_task(args: tuple) -> list[GameRecord]:
    weights, kind, starts, fruit_seeds, rules, search, device = args
    torch.set_num_threads(1)
    if kind in REFERENCES:
        return list(evaluate_recovery(REFERENCES[kind], starts, fruit_seeds, rules).records)
    games = [SnakeGame.from_state(s.state, seed=f) for s, f in zip(starts, fruit_seeds, strict=True)]
    labels = [{"start": s.name, "seed": f, "band": s.band} for s, f in zip(starts, fruit_seeds, strict=True)]
    runs = _runs(kind, games, rules, labels, envelope=False)
    lockstep = [_Lockstep(run, kind, rules, search, i) for i, run in enumerate(runs)]
    evaluator = NetworkEvaluator(from_bytes(weights), device)
    records = play_lockstep(evaluator, lockstep)
    evaluator.release()
    return records


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
    device: str = "cpu",
) -> Evaluation:
    """Track T1 for ``games`` held-out standard starts, the games split across ``pool``.

    Each worker plays its games at once, evaluating their leaves in batches on ``device``
    (:func:`play_lockstep`). The agent's moves, with the states they were played in, are appended to
    ``decisions`` if given.
    """
    seeds = list(standard_seeds(split, games))
    record = decisions is not None
    tasks = [
        (weights, kind, width, height, chunk, rules, search, record, device)
        for chunk in _chunks(seeds, workers)
    ]
    records = []
    for part, moves in pool.map(_t1_task, tasks):
        records += part
        if record:
            decisions += moves
    return Evaluation("T1", f"network-{kind}", Provenance.SELF_PLAY, (width, height), rules, tuple(records))


def _shortcut_task(args: tuple) -> list[GameRecord]:
    width, height, seeds, rules = args
    return list(evaluate_standard(REFERENCES["shortcut"], width, height, seeds, rules).records)


def shortcut_t1(
    pool: Executor,
    width: int,
    height: int,
    games: int,
    split: SeedRange = VALIDATION,
    rules: Rules = TRAINING_RULES,
    workers: int = 8,
) -> Evaluation:
    """The shortcut cycle on the same T1 games as :func:`t1` (the same fruit seeds), across ``pool``."""
    seeds = list(standard_seeds(split, games))
    tasks = [(width, height, chunk, rules) for chunk in _chunks(seeds, workers)]
    records = tuple(r for part in pool.map(_shortcut_task, tasks) for r in part)
    return Evaluation("T1", "shortcut", Provenance.REFERENCE, (width, height), rules, records)


def versus(evaluation: Evaluation, reference: Evaluation) -> dict[str, Any]:
    """``evaluation``'s T1 moves against ``reference``'s on the same fruit seeds: the mean of the paired
    differences (negative: fewer moves) and its 95% interval, over the games both won."""
    moves = {r.seed: r.moves for r in reference.records if r.outcome is Outcome.WON}
    pairs = [
        (r.moves, moves[r.seed]) for r in evaluation.records if r.outcome is Outcome.WON and r.seed in moves
    ]
    if not pairs:
        return {"games": 0}
    differences = [a - b for a, b in pairs]
    low, high = mean_interval(differences)
    return {
        "reference": reference.agent,
        "games": len(pairs),
        "reference_mean_moves": round(statistics.fmean(b for _, b in pairs), 2),
        "difference": round(statistics.fmean(differences), 2),
        "interval": [round(low, 2), round(high, 2)],
        "fewer_moves": high < 0.0,
    }


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


def sibling_ranking(
    evaluator: Evaluator,
    solution: ExactSolution,
    decisions: Sequence[tuple[GameState, RelativeAction]],
    rules: Rules = TRAINING_RULES,
) -> dict[str, Any]:
    """How well the moves head orders the envelope moves of T1 decisions, by the exact envelope ``solution``.

    At each decision with at least two envelope moves, each move's one-step value ``1 + V(child)`` (the
    afterstate's value if it eats) is compared with its exact expected moves. ``difference_error``: the
    mean absolute error of each other move's value minus the played move's; ``order_agreement``: the
    share of pairs of moves with different exact values that the head orders as the solution does.
    Reported for the first half of the game (at least half the board free) and the second.
    """
    queries: list[Query] = []
    groups = []
    for state, played in decisions:
        envelope = canonical_envelope(state.width, state.height)
        try:
            costs = solution.action_moves(state)
        except ValueError:  # out of cycle order
            continue
        jumps = envelope.moves(state)
        if len(jumps) < 2:
            continue
        game = planning_game(state, seed=0)
        children = {}
        for direction in jumps:
            action = to_relative(state.direction, direction)
            child = game.clone()
            step = child.step_relative(action)
            if step.won:
                children[action] = None
                continue
            after = dataclasses.replace(child.state(), fruit=None) if step.ate_fruit else child.state()
            children[action] = len(queries)
            queries.append(Query(after, rules, afterstate=step.ate_fruit))
        groups.append((state, played, costs, children))
    values = []
    for i in range(0, len(queries), 1024):
        batch = queries[i : i + 1024]
        for query, moves in zip(batch, evaluator(batch).moves, strict=True):
            after = query.state
            values.append(min(max(float(moves), 0.0), moves_ceiling(after.num_cells, after.free_cells)))
            values[-1] *= moves_scale(after.free_cells)
    report = {}
    for phase, early in (("first_half", True), ("second_half", False)):
        errors, agree, pairs = [], 0, 0
        for state, played, costs, children in groups:
            if (state.free_cells * 2 >= state.num_cells) is not early:
                continue
            q = {a: 1.0 + (0.0 if i is None else values[i]) for a, i in children.items()}
            errors += [(q[a] - q[played]) - (costs[a] - costs[played]) for a in q if a != played]
            moves = sorted(q)
            for j, a in enumerate(moves):
                for b in moves[j + 1 :]:
                    if costs[a] != costs[b]:
                        pairs += 1
                        agree += (q[a] < q[b]) == (costs[a] < costs[b])
        report[phase] = {
            "decisions": sum((s.free_cells * 2 >= s.num_cells) is early for s, *_ in groups),
            "difference_error": round(float(np.mean(np.abs(errors))), 4) if errors else None,
            "order_agreement": round(agree / pairs, 4) if pairs else None,
        }
    return report


def t2(
    pool: Executor,
    weights: bytes | None,
    kind: str,
    starts: Sequence[RecoveryStart],
    split: SeedRange = VALIDATION,
    rules: Rules = TRAINING_RULES,
    workers: int = 8,
    search: SearchConfig = EVALUATION,
    device: str = "cpu",
) -> Evaluation:
    """Track T2 on ``starts`` for the network (``"search"``, ``"policy"``; batched on ``device`` as in
    :func:`t1`) or a reference policy."""
    fruit_seeds = list(recovery_fruit_seeds(split, len(starts)))
    pairs = list(zip(starts, fruit_seeds, strict=True))
    tasks = [
        (weights, kind, [s for s, _ in chunk], [f for _, f in chunk], rules, search, device)
        for chunk in _chunks(pairs, workers)
    ]
    records = [r for part in pool.map(_t2_task, tasks) for r in part]
    name = kind if kind in REFERENCES else f"network-{kind}"
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


def _visited_task(args: tuple) -> list[tuple[GameState, RelativeAction]]:
    weights, kind, starts, fruit_seeds, rules, search, device = args
    torch.set_num_threads(1)
    decisions: list[tuple[GameState, RelativeAction]] = []
    games = [SnakeGame.from_state(s.state, seed=f) for s, f in zip(starts, fruit_seeds, strict=True)]
    runs = _runs(kind, games, rules, [{} for _ in games], envelope=False)
    lockstep = [_Lockstep(run, kind, rules, search, i, decisions) for i, run in enumerate(runs)]
    play_lockstep(NetworkEvaluator(from_bytes(weights), device), lockstep)
    return decisions


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
    device: str = "cpu",
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
            (weights, kind, [s for s, _ in chunk], [f for _, f in chunk], rules, search, device)
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
    device: str = "cpu",
    agents: Sequence[str] = AGENTS,
) -> dict[str, Any]:
    """T1 and T2 for the ``agents`` (and the exact 4x4 values), as a JSON-ready report with the gates.

    ``search`` holds the search agent's settings (simulations per move, moves unit); the network's games
    run in batches on ``device``. The T1 gate compares mean moves with the shortcut cycle
    (:data:`T1_REFERENCES`); the T2 gate asks that in every band the 95% lower bound of the win rate be
    above the best reference policy's rate on the same starts (``references``: measured if not given).
    """
    report: dict[str, Any] = {
        "board": [width, height],
        "split": split.name,
        "games": games,
        "simulations": search.simulations,
        "moves_unit": search.moves_unit,
        "relative_unit": search.relative_unit,
        "win_unit": search.win_unit,
        "confident": search.confident,
        "careful_fill": search.careful_fill,
    }
    reference = T1_REFERENCES.get((width, height))
    exact = width * height <= 36 and canonical_envelope(width, height) is not None
    inside = solve_envelope(width, height) if exact else None
    shortcut = None
    if canonical_envelope(width, height) is not None:
        shortcut = shortcut_t1(pool, width, height, games, split, rules, workers)
        report["t1_shortcut"] = {"summary": shortcut.summary().to_dict()}
    for kind in agents:
        decisions: list[tuple[GameState, RelativeAction]] | None = [] if exact else None
        evaluation = t1(
            pool, weights, kind, width, height, games, split, rules, workers, search, decisions, device
        )
        entry: dict[str, Any] = {"summary": evaluation.summary().to_dict()}
        if reference is not None:
            gate = standard_gate(evaluation, reference)
            entry["gate"] = {"reference": reference, "passed": gate.passed, "reasons": list(gate.reasons)}
        if shortcut is not None:
            entry["versus_shortcut"] = versus(evaluation, shortcut)
        if inside is not None:
            entry["exact"] = envelope_regret(inside, decisions)
            evaluator = NetworkEvaluator(from_bytes(weights))
            entry["exact"]["siblings"] = sibling_ranking(evaluator, inside, decisions, rules)
        report[f"t1_{kind}"] = entry
    starts = recovery_starts_parallel(pool, width, height, sources, split, workers)
    if starts:
        if references is None:
            references = best_rates(
                t2(pool, None, name, starts, split, rules, workers) for name in T2_REFERENCES
            )
        report["t2_references"] = {str(band): rate for band, rate in sorted(references.items())}
        for kind in agents:
            evaluation = t2(pool, weights, kind, starts, split, rules, workers, search, device)
            gate = recovery_gate(evaluation, references)
            report[f"t2_{kind}"] = {
                str(band): evaluation.summary(band).to_dict() for band in evaluation.bands()
            } | {
                "all": evaluation.summary().to_dict(),
                "gate": {"passed": gate.passed, "reasons": list(gate.reasons)},
            }
        if (width, height) == (4, 4):
            report["exact"] = exact_4x4(
                pool, weights, starts, split, rules, workers, games=games, search=search, device=device
            )
    return report


def _recovery_task(args: tuple) -> list[RecoveryStart]:
    width, height, seeds = args
    return recovery_starts(width, height, seeds)


def recovery_starts_parallel(
    pool: Executor, width: int, height: int, sources: int, split: SeedRange = VALIDATION, workers: int = 8
) -> list[RecoveryStart]:
    """:func:`~snake_agent.recovery.held_out_starts`, with the source games split across ``pool``."""
    seeds = list(recovery_source_seeds(split, sources))
    tasks = [(width, height, chunk) for chunk in _chunks(seeds, workers)]
    return [start for part in pool.map(_recovery_task, tasks) for start in part]
