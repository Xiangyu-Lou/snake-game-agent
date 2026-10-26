"""The evaluation harness: standard starts (T1), recovery starts (T2) and track T3.

Every track ranks completion first and moves second:

* **T1, standard starts.** Games from the standard start, which is aligned with
  a Hamiltonian cycle on boards with an even side. The agent must stay inside
  that cycle's envelope (:class:`~snake_agent.cycles.CycleEnvelope`), where
  completion is certain, so any failure is a bug and the gate measures moves
  against the shortcut cycle (:func:`standard_gate`).
* **T2, recovery starts.** Certified mid-game states that no Hamiltonian cycle
  is aligned with (:mod:`snake_agent.recovery`), grouped by occupancy band. The
  gate: beat the best reference in every band (:func:`recovery_gate`).
* **T3, no demonstrations.** T1 and T2 for an agent learned only from its own
  search (:attr:`Provenance.SELF_PLAY`): :func:`evaluate_t3`. Results of agents
  that used demonstrations are reported separately.

Games are played on :class:`~snake_game.SnakeGame` directly. The agent sees each
:class:`~snake_game.GameState`, never the game. Under the training rules a game
ends won, lost or timed out (:class:`~snake_agent.contract.Rules`). Without the
fruit timeout, the check run, endless games are censored instead: a
deterministic agent that repeats a state between two fruits would loop forever,
and any game is stopped after ``STALL_FACTOR * N`` moves without fruit. Seeds
for the final gates come from the held-out ranges of :mod:`snake_agent.seeds`.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame

from .contract import TRAINING_RULES, Agent, Outcome, Rules, play_move, to_direction
from .cycles import CycleEnvelope
from .stats import binomial_interval, binomial_upper_bound, mean_interval

if TYPE_CHECKING:
    from .recovery import RecoveryStart

STALL_FACTOR = 20
"""Without the fruit timeout, a game is censored after more than ``STALL_FACTOR * N`` moves without fruit."""


class Provenance(str, Enum):
    """Where an agent comes from. Results are reported separately for each."""

    REFERENCE = "reference"  # hand-written, no learning
    DEMONSTRATIONS = "demonstrations"  # learned, using demonstrations such as Hamiltonian games
    SELF_PLAY = "self_play"  # learned only from its own search: track T3


@dataclass(frozen=True)
class AgentSpec:
    """How the harness makes and labels an agent.

    ``factory(seed)`` makes a fresh agent for every game. ``seed`` is for the
    agent's own randomness (planning seeds, sampling) and has nothing to do with
    the game's fruits. ``deterministic`` declares that the agent's move depends
    only on the state and on ``memory(agent)``, its internal state if it has
    any: a state repeated between two fruits then means an endless loop.
    """

    name: str
    factory: Callable[[int], Agent]
    provenance: Provenance = Provenance.REFERENCE
    deterministic: bool = False
    memory: Callable[[Any], Hashable] | None = None


@dataclass(frozen=True)
class GameRecord:
    """One evaluated game."""

    start: str
    """``"standard"``, or the name of the recovery start."""
    seed: int
    """The game's fruit seed."""
    outcome: Outcome
    moves: int
    """Moves made from the start."""
    fruits: int
    """Fruits eaten from the start."""
    length: int
    """Final length."""
    detail: str | None = None
    """How the game was lost (``"wall"``, ``"self"``) or why it was censored (``"loop"``, ``"stall"``,
    ``"cap"``)."""
    longest_without_fruit: int = 0
    timeout_margin: float = 0.0
    """Closest approach to the fruit timeout: the largest ``moves since fruit / (N + L)`` (1.0: timed out)."""
    envelope_exit: int | None = None
    """T1: the move at which the agent first left the cycle-order envelope; None if it never did."""
    band: float | None = None
    """T2: the occupancy band of the start."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "seed": self.seed,
            "outcome": self.outcome.value,
            "moves": self.moves,
            "fruits": self.fruits,
            "length": self.length,
            "detail": self.detail,
            "longest_without_fruit": self.longest_without_fruit,
            "timeout_margin": round(self.timeout_margin, 6),
            "envelope_exit": self.envelope_exit,
            "band": self.band,
        }


def run_game(
    game: SnakeGame,
    agent: Agent,
    rules: Rules = TRAINING_RULES,
    *,
    memory: Callable[[], Hashable] | None = None,
    detect_loops: bool = False,
    stall_limit: int | None = None,
    move_cap: int | None = None,
    envelope: CycleEnvelope | None = None,
    record: list[GameState] | None = None,
    start: str = "standard",
    seed: int = -1,
    band: float | None = None,
) -> GameRecord:
    """Play ``game`` to the end with ``agent`` under ``rules``.

    Before each move: the state is appended to ``record``; with
    ``detect_loops`` a state (and ``memory()``) repeated since the last fruit
    censors the game as a ``"loop"``; more than ``stall_limit`` moves without
    fruit censor it as a ``"stall"``, and ``move_cap`` moves as a ``"cap"``.
    With an ``envelope``, the first move outside it is recorded. ``start``,
    ``seed`` and ``band`` only label the record.
    """
    cells = game.width * game.height
    seen: set[Hashable] = set()
    moves = fruits = 0
    longest = game.steps_since_fruit
    margin = longest / (cells + game.length)
    inside = envelope is not None and envelope.contains(game.state())
    exit_move = None if inside or envelope is None else 0

    def finish(outcome: Outcome, detail: str | None = None) -> GameRecord:
        return GameRecord(
            start, seed, outcome, moves, fruits, game.length, detail, longest, margin, exit_move, band
        )

    while not game.done:
        state = game.state()
        if record is not None:
            record.append(state)
        if detect_loops:
            if state.steps_since_fruit == 0:
                seen.clear()
            key = (state.snake, state.direction, None if memory is None else memory())
            if key in seen:
                return finish(Outcome.CENSORED, "loop")
            seen.add(key)
        if stall_limit is not None and state.steps_since_fruit > stall_limit:
            return finish(Outcome.CENSORED, "stall")
        if move_cap is not None and moves >= move_cap:
            return finish(Outcome.CENSORED, "cap")
        action = agent(state)
        if inside:
            direction = (
                to_direction(state.direction, action) if isinstance(action, RelativeAction) else action
            )
            if not isinstance(direction, Direction) or envelope.jump(state, direction) is None:
                inside, exit_move = False, moves
        result = play_move(game, action)
        moves += 1
        if result.ate_fruit:
            fruits += 1
        else:
            since = game.steps_since_fruit
            longest = max(longest, since)
            margin = max(margin, since / (cells + game.length))
            limit = rules.limit(cells, game.length)
            if limit is not None and since >= limit and not game.done:
                return finish(Outcome.TIMED_OUT)
    if game.won:
        return finish(Outcome.WON)
    return finish(Outcome.LOST, game.death_cause.value)


def _play(spec: AgentSpec, agent_seed: int, game: SnakeGame, rules: Rules, **kwargs: Any) -> GameRecord:
    agent = spec.factory(agent_seed)
    memory = None if spec.memory is None else (lambda: spec.memory(agent))
    no_limit = not rules.fruit_timeout
    return run_game(
        game,
        agent,
        rules,
        memory=memory,
        detect_loops=no_limit and spec.deterministic,
        stall_limit=STALL_FACTOR * game.width * game.height if no_limit else None,
        **kwargs,
    )


# --------------------------------------------------------------------- tracks
@dataclass(frozen=True)
class Evaluation:
    """The games of one agent on one track and board."""

    track: str
    """``"T1"`` (standard starts) or ``"T2"`` (recovery starts)."""
    agent: str
    provenance: Provenance
    board: tuple[int, int]
    rules: Rules
    records: tuple[GameRecord, ...]

    @property
    def label(self) -> str:
        """Track name for reports; an agent learned without demonstrations is reported as T3."""
        if self.provenance is Provenance.SELF_PLAY:
            return f"T3 ({self.track} starts)"
        if self.provenance is Provenance.DEMONSTRATIONS:
            return f"{self.track} (with demonstrations)"
        return self.track

    def bands(self) -> list[float]:
        """The occupancy bands of the starts, in increasing order (T2)."""
        return sorted({r.band for r in self.records if r.band is not None})

    def summary(self, band: float | None = None) -> Summary:
        """Summary of all games, or of the games started in ``band``."""
        records = [r for r in self.records if band is None or r.band == band]
        return summarize(records)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready results, with a summary per band."""
        return {
            "track": self.track,
            "label": self.label,
            "agent": self.agent,
            "provenance": self.provenance.value,
            "board": list(self.board),
            "fruit_timeout": self.rules.fruit_timeout,
            "summary": self.summary().to_dict(),
            "bands": {str(b): self.summary(b).to_dict() for b in self.bands()},
            "games": [r.to_dict() for r in self.records],
        }


def evaluate_standard(
    spec: AgentSpec,
    width: int,
    height: int,
    seeds: Iterable[int],
    rules: Rules = TRAINING_RULES,
    *,
    initial_length: int = 3,
    agent_seed: int = 0,
    move_cap: int | None = None,
) -> Evaluation:
    """Track T1: one game from the standard start per fruit seed.

    The envelope is that of the first Hamiltonian cycle the start is aligned
    with; boards without one (both sides odd, for example) are played without.
    """
    config = GameConfig(width, height, initial_length)
    records = []
    for i, seed in enumerate(seeds):
        game = SnakeGame(config, seed=seed)
        envelope = CycleEnvelope.for_state(game.state())
        records.append(
            _play(spec, agent_seed + i, game, rules, envelope=envelope, seed=seed, move_cap=move_cap)
        )
    return Evaluation("T1", spec.name, spec.provenance, (width, height), rules, tuple(records))


def evaluate_recovery(
    spec: AgentSpec,
    starts: Sequence[RecoveryStart],
    fruit_seeds: Sequence[int],
    rules: Rules = TRAINING_RULES,
    *,
    agent_seed: int = 0,
    move_cap: int | None = None,
) -> Evaluation:
    """Track T2: one game from each recovery start, with fruits from the matching seed."""
    if len(fruit_seeds) < len(starts):
        raise ValueError(f"need a fruit seed for each of the {len(starts)} starts, got {len(fruit_seeds)}")
    if not starts:
        raise ValueError("no recovery starts")
    board = (starts[0].state.width, starts[0].state.height)
    records = []
    for i, (start, seed) in enumerate(zip(starts, fruit_seeds[: len(starts)], strict=True)):
        game = SnakeGame.from_state(start.state, seed=seed)
        records.append(
            _play(
                spec,
                agent_seed + i,
                game,
                rules,
                start=start.name,
                seed=seed,
                band=start.band,
                move_cap=move_cap,
            )
        )
    return Evaluation("T2", spec.name, spec.provenance, board, rules, tuple(records))


def evaluate_t3(
    spec: AgentSpec,
    width: int,
    height: int,
    seeds: Iterable[int],
    starts: Sequence[RecoveryStart],
    fruit_seeds: Sequence[int],
    rules: Rules = TRAINING_RULES,
    *,
    agent_seed: int = 0,
    move_cap: int | None = None,
) -> tuple[Evaluation, Evaluation]:
    """Track T3: T1 and T2 for an agent learned without demonstrations."""
    if spec.provenance is not Provenance.SELF_PLAY:
        raise ValueError(
            f"track T3 is for agents learned only from their own search, {spec.name!r} is "
            f"{spec.provenance.value}"
        )
    options = {"agent_seed": agent_seed, "move_cap": move_cap}
    return (
        evaluate_standard(spec, width, height, seeds, rules, **options),
        evaluate_recovery(spec, starts, fruit_seeds, rules, **options),
    )


# ------------------------------------------------------------------ summaries
@dataclass(frozen=True)
class Summary:
    """Outcome counts, win rate with 95% bounds, and moves of the won games."""

    games: int
    won: int
    lost: int
    timed_out: int
    censored: int
    win_interval: tuple[float, float]
    """Two-sided 95% Clopper-Pearson bounds, counting censored games as lost (lower) and as won (upper)."""
    failure_bound: float
    """One-sided 95% upper bound on the share of games not won (censored ones included)."""
    mean_moves: float | None
    moves_interval: tuple[float, float] | None
    median_moves: float | None
    max_moves: int | None
    mean_fruits: float
    envelope_exits: int
    timeout_margin: float
    """The closest any won game came to the fruit timeout (see :attr:`GameRecord.timeout_margin`)."""

    @property
    def win_rate(self) -> float:
        return self.won / self.games

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "won": self.won,
            "lost": self.lost,
            "timed_out": self.timed_out,
            "censored": self.censored,
            "win_rate": round(self.win_rate, 6),
            "win_interval": [round(b, 6) for b in self.win_interval],
            "failure_bound": round(self.failure_bound, 6),
            "mean_moves": None if self.mean_moves is None else round(self.mean_moves, 3),
            "moves_interval": None
            if self.moves_interval is None
            else [round(b, 3) for b in self.moves_interval],
            "median_moves": self.median_moves,
            "max_moves": self.max_moves,
            "mean_fruits": round(self.mean_fruits, 3),
            "envelope_exits": self.envelope_exits,
            "timeout_margin": round(self.timeout_margin, 6),
        }


def summarize(records: Sequence[GameRecord]) -> Summary:
    if not records:
        raise ValueError("no games to summarize")
    games = len(records)
    count = {outcome: 0 for outcome in Outcome}
    for r in records:
        count[r.outcome] += 1
    won, censored = count[Outcome.WON], count[Outcome.CENSORED]
    moves = [r.moves for r in records if r.outcome is Outcome.WON]
    return Summary(
        games=games,
        won=won,
        lost=count[Outcome.LOST],
        timed_out=count[Outcome.TIMED_OUT],
        censored=censored,
        win_interval=(binomial_interval(won, games)[0], binomial_interval(won + censored, games)[1]),
        failure_bound=binomial_upper_bound(games - won, games),
        mean_moves=statistics.fmean(moves) if moves else None,
        moves_interval=mean_interval(moves) if moves else None,
        median_moves=statistics.median(moves) if moves else None,
        max_moves=max(moves) if moves else None,
        mean_fruits=statistics.fmean(r.fruits for r in records),
        envelope_exits=sum(r.envelope_exit is not None for r in records),
        timeout_margin=max((r.timeout_margin for r in records if r.outcome is Outcome.WON), default=0.0),
    )


# ---------------------------------------------------------------------- gates
@dataclass(frozen=True)
class Gate:
    """A gate decision with the reasons it failed (none if it passed)."""

    passed: bool
    reasons: tuple[str, ...] = ()


def standard_gate(evaluation: Evaluation, max_mean_moves: float) -> Gate:
    """T1: every game won inside the envelope, and mean moves below ``max_mean_moves``.

    ``max_mean_moves`` is the reference to beat, for example the shortcut
    cycle's mean (214 on 6x6, 1,445 on 10x10, 20,737 on 20x20).
    """
    s = evaluation.summary()
    reasons = []
    if s.won < s.games:
        reasons.append(f"{s.games - s.won} of {s.games} games not won")
    if s.envelope_exits:
        reasons.append(f"left the envelope in {s.envelope_exits} games")
    if s.mean_moves is not None and s.mean_moves >= max_mean_moves:
        reasons.append(f"mean moves {s.mean_moves:.1f} not below {max_mean_moves:g}")
    return Gate(not reasons, tuple(reasons))


def recovery_gate(evaluation: Evaluation, reference_rates: Mapping[float, float]) -> Gate:
    """T2: in every band, the 95% lower bound of the win rate is above the best reference's win rate."""
    reasons = []
    for band, reference in sorted(reference_rates.items()):
        records = [r for r in evaluation.records if r.band == band]
        if not records:
            reasons.append(f"no games in band {band}")
            continue
        lower = summarize(records).win_interval[0]
        if lower <= reference:
            reasons.append(f"band {band}: lower bound {lower:.3f} not above the reference {reference:.3f}")
    return Gate(not reasons, tuple(reasons))


def best_rates(evaluations: Iterable[Evaluation]) -> dict[float, float]:
    """The best win rate per band among the (reference) ``evaluations`` of the same starts."""
    best: dict[float, float] = {}
    for evaluation in evaluations:
        for band in evaluation.bands():
            best[band] = max(best.get(band, 0.0), evaluation.summary(band).win_rate)
    return best


def format_summary(evaluation: Evaluation) -> str:
    """A small text table: one row per band (T2) or one row (T1)."""
    rows = [("all", evaluation.summary())]
    rows += [(f"{band:.0%}", evaluation.summary(band)) for band in evaluation.bands()]
    width, height = evaluation.board
    header = f"{evaluation.label}  {evaluation.agent}  {width}x{height}  " + (
        "fruit timeout N + L" if evaluation.rules.fruit_timeout else "no fruit timeout"
    )
    lines = [header, "start   games    won   lost  timeout  censored  win 95% interval   mean moves (won)"]
    for name, s in rows:
        low, high = s.win_interval
        mean = "-" if s.mean_moves is None else f"{s.mean_moves:,.1f}"
        lines.append(
            f"{name:<6}{s.games:>7}{s.won:>7}{s.lost:>7}{s.timed_out:>9}{s.censored:>10}"
            f"   {low:6.1%} - {high:6.1%}   {mean:>16}"
        )
    if evaluation.track == "T1":
        lines.append(f"envelope exits: {evaluation.summary().envelope_exits}")
    return "\n".join(lines)
