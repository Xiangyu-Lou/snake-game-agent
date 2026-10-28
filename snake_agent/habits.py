"""Habits: a bias toward longer or shorter jumps in each band of fill, found by paired games (after M2).

M2's agents stop short of the shortcut cycle on 10x10 T1 because what the cycle does better is a habit,
jumping consistently while the snake is short, and no single decision shows it: two envelope moves
differ by a fraction of a move in expectation, while the rest of the game varies by tens of moves, so
neither the moves head nor a few rollouts can rank them (``snake_agent/README.md``). A habit acts on many
decisions at once, so its effect adds up where theirs is lost in the noise.

**Habits.** A :class:`Habit` holds one bias per band of fill, :data:`BANDS` bands of the snake's length
over the board's cells. At an envelope decision with a choice, the agent plays the move that maximizes

    logit(a) + bias * (jump(a) - shortest jump)

with jumps counted in cells along the canonical cycle (:meth:`~snake_agent.cycles.CycleEnvelope.moves`):
bias 0 is the raw policy, ``+inf`` the longest jump and ``-inf`` the cycle's next cell. Habits only act
inside the envelope. :class:`HabitEvaluator` applies one to any evaluator's logits, so that
:class:`~snake_agent.search.PolicyAgent` (or the search's prior) plays it.

**Paired windows** (:func:`compare`). The current habit plays games from the standard start of training
seeds, and the position where each band begins (a fruit just placed) starts a *window*. From it, branches
play other biases in that band only, and the current habit everywhere else, with one planning seed:
common random numbers, every branch draws the same fruit stream. A branch stops when it merges with the
window's base branch (the same position at the same length, so the rest of the game is shared) or when
its game ends. Its difference in moves to the base is a sample of the candidate's effect. Below half
fill nearly every branch merges within a few hundred moves, and the windows of one game are independent
samples of different bands.

**Improvement** (:func:`improve`). In each band, the candidate with the fewest mean moves replaces the
current bias if it is better by more than ``z`` standard errors. :func:`snake_agent.train.distill` then
trains the network to play the improved habit with its raw policy, and the next round starts from the
distilled network with the neutral habit: a policy iteration whose improvement step compares habits
instead of single moves.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame

from .contract import TRAINING_RULES, Rules, planning_game, to_relative
from .cycles import CycleEnvelope
from .encoding import canonical_envelope, encode
from .search import (
    ACTIONS,
    Evaluator,
    LeafValue,
    Mode,
    Prediction,
    Query,
    _Epoch,
    _root,
    envelope_certain,
    position_key,
)
from .seeds import check_training_seed
from .selfplay import FRUIT_OFFSET, Examples, Label, Start, _frame_width, _to_examples

BANDS = 20
"""Bands of fill: band ``k`` holds the lengths ``L`` with ``k <= BANDS * L / N < k + 1``."""

CANDIDATES = (-math.inf, -0.15, 0.0, 0.15, math.inf)
"""The biases :func:`compare` tries in each band by default (the current one is the base)."""

FORCED = 30.0
"""The logit margin :class:`HabitEvaluator` gives the move an infinite bias forces."""


@dataclass(frozen=True)
class Habit:
    """A jump bias for each band of fill (see the module documentation); all zero is the raw policy."""

    biases: tuple[float, ...] = (0.0,) * BANDS

    def __post_init__(self) -> None:
        if len(self.biases) != BANDS:
            raise ValueError(f"a habit has {BANDS} biases, got {len(self.biases)}")
        if any(math.isnan(b) for b in self.biases):
            raise ValueError("biases must not be NaN")

    @staticmethod
    def band(length: int, cells: int) -> int:
        """The band of a snake of ``length`` on ``cells`` cells."""
        return min(BANDS * length // cells, BANDS - 1)

    def bias(self, state: GameState) -> float:
        return self.biases[self.band(state.length, state.num_cells)]

    @property
    def neutral(self) -> bool:
        """Whether the habit is the raw policy everywhere."""
        return not any(self.biases)

    def replace(self, band: int, bias: float) -> Habit:
        """This habit with ``bias`` in ``band``."""
        biases = list(self.biases)
        biases[band] = bias
        return Habit(tuple(biases))

    def forced(self, state: GameState, jumps: Mapping[Direction, int]) -> bool:
        """Whether the move needs no logits: a single envelope move, or an infinite bias."""
        return len(jumps) == 1 or math.isinf(self.bias(state))

    def scores(self, state: GameState, jumps: Mapping[Direction, int], logits: np.ndarray) -> np.ndarray:
        """The habit's scores of ``LEFT``, ``STRAIGHT``, ``RIGHT`` (-inf outside the envelope ``jumps``).

        An infinite bias gives the forced move a margin of :data:`FORCED` over the others' logits.
        """
        scores = np.full(3, -np.inf)
        bias = self.bias(state)
        shortest = min(jumps.values())
        for direction, jump in jumps.items():
            action = to_relative(state.direction, direction)
            scores[action] = logits[action] + (0.0 if math.isinf(bias) else bias * (jump - shortest))
        if math.isinf(bias) and len(jumps) > 1:
            pick = max if bias > 0 else min
            forced = to_relative(state.direction, pick(jumps, key=jumps.__getitem__))
            scores[forced] = max(scores[a] for a in range(3) if np.isfinite(scores[a])) + FORCED
        return scores

    def choose(
        self, state: GameState, jumps: Mapping[Direction, int], logits: np.ndarray | None = None
    ) -> RelativeAction:
        """The habit's move among the envelope moves ``jumps``, given the network's ``logits``.

        ``logits`` may be None when :meth:`forced`. Ties go to the first action, as in
        :class:`~snake_agent.search.PolicyAgent`, which plays the same moves when the habit is neutral.
        """
        bias = self.bias(state)
        if len(jumps) == 1 or math.isinf(bias):
            pick = min if bias < 0 else max
            return to_relative(state.direction, pick(jumps, key=jumps.__getitem__))
        if logits is None:
            raise ValueError("this move needs the network's logits")
        scores = self.scores(state, jumps, logits)
        return RelativeAction(int(np.argmax(scores)))

    def target(
        self,
        state: GameState,
        jumps: Mapping[Direction, int],
        logits: np.ndarray,
        smoothing: float = 0.02,
    ) -> np.ndarray:
        """The habit's policy over ``LEFT``, ``STRAIGHT``, ``RIGHT``: a training target for distillation.

        A finite bias gives the softmax of the habit's scores over the envelope moves; an infinite one
        gives the forced move ``1 - smoothing`` and spreads ``smoothing`` over the other envelope moves.
        """
        policy = np.zeros(3)
        bias = self.bias(state)
        if len(jumps) > 1 and math.isinf(bias):
            pick = max if bias > 0 else min
            forced = to_relative(state.direction, pick(jumps, key=jumps.__getitem__))
            for direction in jumps:
                action = to_relative(state.direction, direction)
                policy[action] = 1.0 - smoothing if action == forced else smoothing / (len(jumps) - 1)
            return policy
        scores = self.scores(state, jumps, logits)
        valid = np.isfinite(scores)
        z = np.exp(scores[valid] - scores[valid].max())
        policy[valid] = z / z.sum()
        return policy

    def to_json(self) -> list[float | str]:
        """The biases for JSON, infinities as ``"inf"`` and ``"-inf"``."""
        return [str(b) if math.isinf(b) else b for b in self.biases]

    @classmethod
    def from_json(cls, data: Sequence[float | str]) -> Habit:
        return cls(tuple(float(b) for b in data))

    def __str__(self) -> str:
        """Runs of equal biases, for example ``0-45%: +inf, 45-100%: 0``."""
        runs, start = [], 0
        for k in range(1, BANDS + 1):
            if k == BANDS or self.biases[k] != self.biases[start]:
                low, high = 100 * start // BANDS, 100 * k // BANDS
                runs.append(f"{low}-{high}%: {_format_bias(self.biases[start])}")
                start = k
        return ", ".join(runs)


def _format_bias(bias: float) -> str:
    return "0" if bias == 0 else f"{bias:+g}"


class HabitEvaluator:
    """An :data:`~snake_agent.search.Evaluator` whose logits play ``habit`` at envelope decisions.

    It adds the habit's jump bias to the logits of ``evaluator`` for every query in envelope mode with a
    choice (:func:`~snake_agent.search.envelope_certain`, not an afterstate); the values and every other
    query are unchanged. Around the network's evaluator it makes :class:`~snake_agent.search.PolicyAgent`
    play the habit, and gives the search the habit's prior.
    """

    def __init__(self, evaluator: Evaluator, habit: Habit) -> None:
        self.evaluator, self.habit = evaluator, habit

    def __call__(self, queries: Sequence[Query]) -> Prediction:
        prediction = self.evaluator(queries)
        if self.habit.neutral:
            return prediction
        logits = np.array(prediction.logits, dtype=np.float64)
        for i, query in enumerate(queries):
            state = query.state
            if query.afterstate or not envelope_certain(state, query.rules):
                continue
            jumps = canonical_envelope(state.width, state.height).moves(state)
            if len(jumps) > 1 and self.habit.bias(state) != 0:
                scores = self.habit.scores(state, jumps, logits[i])
                logits[i] = np.where(np.isfinite(scores), scores, logits[i])
        return Prediction(logits, prediction.win, prediction.moves)


# ------------------------------------------------------------- paired windows
@dataclass(frozen=True)
class Window:
    """A paired comparison: the ``base`` habit and each candidate play from ``state``, with fruits from
    ``seed``."""

    state: GameState
    seed: int
    band: int
    base: Habit
    candidates: tuple[Habit, ...]


@dataclass(frozen=True)
class BranchResult:
    """How a candidate of a window did against its base."""

    delta: int
    """Its moves minus the base's, until they merged or both games ended."""
    merged: int | None
    """The length at which it merged with the base (None if it never did)."""
    moves: int
    """Moves it played."""


class _Branch:
    """One habit playing from a window's start, advanced a decision at a time."""

    def __init__(self, game: SnakeGame, habit: Habit, envelope: CycleEnvelope, rules: Rules) -> None:
        self.game, self.habit, self.envelope, self.rules = game, habit, envelope, rules
        self.moves = 0
        self.fruits: dict[int, tuple[Hashable, int]] = {game.length: (position_key(game.state()), 0)}
        """For each length: the position when its fruit appeared, and the moves played by then."""
        self.stopped = False
        self.result: BranchResult | None = None
        self.pending: tuple[GameState, dict[Direction, int]] | None = None
        self.window: _WindowRun | None = None
        self.entries: dict[int, GameState] | None = None
        """If set, the first position of each band with its fruit just placed is recorded here."""

    def advance(self) -> Query | None:
        """Play until the habit needs the network (returns its query) or the branch stops (None)."""
        while not self.stopped:
            if self.game.done:
                if not self.game.won:
                    raise RuntimeError("an envelope game was lost: a bug")
                self.stopped = True
                if self.window is not None:
                    self.window.finished(self)
                return None
            state = self.game.state()
            jumps = self.envelope.moves(state)
            if not jumps:
                raise RuntimeError("the body left cycle order")
            if self.habit.forced(state, jumps):
                self._move(self.habit.choose(state, jumps))
                continue
            self.pending = (state, jumps)
            return Query(state, self.rules)
        return None

    def play(self, logits: np.ndarray) -> None:
        state, jumps = self.pending
        self.pending = None
        self._move(self.habit.choose(state, jumps, logits))

    def _move(self, action: RelativeAction) -> None:
        step = self.game.step_relative(action)
        self.moves += 1
        if step.ate_fruit and not step.won:
            state = self.game.state()
            self.fruits[state.length] = (position_key(state), self.moves)
            if self.entries is not None:
                self.entries.setdefault(Habit.band(state.length, state.num_cells), state)
            if self.window is not None:
                self.window.ate(self)


class _WindowRun:
    """A window's branches: candidates stop when they merge with the base; the base when all have."""

    def __init__(self, window: Window, envelope: CycleEnvelope, rules: Rules) -> None:
        def branch(habit: Habit) -> _Branch:
            b = _Branch(planning_game(window.state, window.seed), habit, envelope, rules)
            b.window = self
            return b

        self.base = branch(window.base)
        self.candidates = [branch(habit) for habit in window.candidates]

    @property
    def branches(self) -> list[_Branch]:
        return [self.base, *self.candidates]

    def ate(self, branch: _Branch) -> None:
        length = branch.game.length
        for candidate in self.candidates if branch is self.base else [branch]:
            self._try_merge(candidate, length)
        if branch is self.base and all(c.result is not None for c in self.candidates):
            self.base.stopped = True  # every candidate has merged: the rest of the game is shared

    def _try_merge(self, candidate: _Branch, length: int) -> None:
        """Stop ``candidate`` if it reached the base's position at ``length`` and their habits agree from
        there on: the rest of their games is then the same."""
        if candidate.result is not None or length not in candidate.fruits or length not in self.base.fruits:
            return
        band = Habit.band(length, candidate.game.width * candidate.game.height)
        if candidate.habit.biases[band:] != self.base.habit.biases[band:]:
            return
        (key, moves), (base_key, base_moves) = candidate.fruits[length], self.base.fruits[length]
        if key == base_key:
            candidate.stopped = True
            candidate.result = BranchResult(moves - base_moves, length, candidate.moves)

    def finished(self, branch: _Branch) -> None:
        """A branch's game ended: settle the candidates whose games have both ended."""
        if not self.base.game.done:
            return
        for candidate in self.candidates:
            if candidate.result is None and candidate.game.done:
                candidate.result = BranchResult(candidate.moves - self.base.moves, None, candidate.moves)

    def results(self) -> list[BranchResult]:
        return [c.result for c in self.candidates]


def _lockstep(evaluator: Evaluator, branches: Iterable[_Branch]) -> None:
    """Advance ``branches`` until they stop, evaluating their pending queries in one batch per round."""
    waiting = list(branches)
    while waiting:
        asking, queries = [], []
        for branch in waiting:
            query = branch.advance()
            if query is not None:
                asking.append(branch)
                queries.append(query)
        if queries:
            logits = evaluator(queries).logits
            for branch, row in zip(asking, logits, strict=True):
                branch.play(row)
        waiting = asking


def play_windows(
    evaluator: Evaluator, windows: Sequence[Window], rules: Rules = TRAINING_RULES
) -> list[list[BranchResult]]:
    """Play every window's branches at once; for each window, the result of each candidate.

    Each window's start must be inside the envelope of its board's canonical cycle.
    """
    runs = []
    for window in windows:
        envelope = canonical_envelope(window.state.width, window.state.height)
        if envelope is None or not envelope.contains(window.state):
            raise ValueError("a window must start inside the canonical envelope")
        runs.append(_WindowRun(window, envelope, rules))
    _lockstep(evaluator, (b for run in runs for b in run.branches))
    return [run.results() for run in runs]


def band_entries(
    evaluator: Evaluator,
    habit: Habit,
    width: int,
    height: int,
    seeds: Sequence[int],
    rules: Rules = TRAINING_RULES,
) -> list[dict[int, GameState]]:
    """For each training seed, the position where each band begins in the habit's game from the standard
    start: the first position of the band with its fruit just placed."""
    envelope = canonical_envelope(width, height)
    if envelope is None:
        raise ValueError(f"{width}x{height} has no canonical cycle")
    branches = []
    for seed in seeds:
        game = SnakeGame(GameConfig(width, height), seed=check_training_seed(seed))
        branch = _Branch(game, habit, envelope, rules)
        branch.entries = {Habit.band(game.length, game.width * game.height): game.state()}
        branches.append(branch)
    _lockstep(evaluator, branches)
    return [branch.entries for branch in branches]


def planning_seed(seed: int, band: int) -> int:
    """The planning seed of the window of training ``seed``'s game in ``band``: a training seed too."""
    return check_training_seed(FRUIT_OFFSET + (check_training_seed(seed) * BANDS + band) % FRUIT_OFFSET)


# ----------------------------------------------------------------- comparison
@dataclass
class Comparison:
    """Paired differences by band and candidate bias: counts, sums, sums of squares and merges.

    Keys are ``(band, bias)``; a difference is the candidate's moves minus the current habit's.
    """

    habit: Habit
    windows: int = 0
    stats: dict[tuple[int, float], list[float]] = field(default_factory=dict)
    """``[count, sum, sum of squares, merged]`` for each (band, candidate bias)."""

    def add(self, band: int, bias: float, result: BranchResult) -> None:
        entry = self.stats.setdefault((band, bias), [0, 0.0, 0.0, 0])
        entry[0] += 1
        entry[1] += result.delta
        entry[2] += result.delta**2
        entry[3] += result.merged is not None

    def merge(self, other: Comparison) -> Comparison:
        if other.habit != self.habit:
            raise ValueError("comparisons of different habits")
        for key, (n, s, s2, m) in other.stats.items():
            entry = self.stats.setdefault(key, [0, 0.0, 0.0, 0])
            entry[0] += n
            entry[1] += s
            entry[2] += s2
            entry[3] += m
        self.windows += other.windows
        return self

    def mean(self, band: int, bias: float) -> float:
        n, s, _, _ = self.stats[(band, bias)]
        return s / n

    def se(self, band: int, bias: float) -> float:
        n, s, s2, _ = self.stats[(band, bias)]
        if n < 2:
            return math.inf
        variance = max(s2 - s * s / n, 0.0) / (n - 1)
        return math.sqrt(variance / n)

    def candidates(self, band: int) -> list[float]:
        return sorted(bias for b, bias in self.stats if b == band)

    def to_dict(self) -> dict[str, Any]:
        bands: dict[str, Any] = defaultdict(dict)
        for (band, bias), (n, _, _, merged) in sorted(self.stats.items()):
            bands[f"{100 * band // BANDS}-{100 * (band + 1) // BANDS}%"][_format_bias(bias)] = {
                "windows": n,
                "mean": round(self.mean(band, bias), 3),
                "se": round(self.se(band, bias), 3),
                "merged": round(merged / n, 3),
            }
        return {"habit": self.habit.to_json(), "windows": self.windows, "bands": dict(bands)}


def compare(
    evaluator: Evaluator,
    habit: Habit,
    width: int,
    height: int,
    seeds: Sequence[int],
    candidates: Sequence[float] = CANDIDATES,
    rules: Rules = TRAINING_RULES,
) -> Comparison:
    """Paired windows for every band and candidate bias, from the games of training ``seeds``.

    See the module documentation. Candidates equal to the habit's bias in a band are skipped there.
    """
    comparison = Comparison(habit)
    windows, labels = [], []
    for seed, entries in zip(seeds, band_entries(evaluator, habit, width, height, seeds, rules), strict=True):
        for band, state in sorted(entries.items()):
            biases = [c for c in candidates if c != habit.biases[band]]
            if not biases:
                continue
            habits = tuple(habit.replace(band, c) for c in biases)
            windows.append(Window(state, planning_seed(seed, band), band, habit, habits))
            labels.append((band, biases))
    comparison.windows = len(windows)
    for (band, biases), results in zip(labels, play_windows(evaluator, windows, rules), strict=True):
        for bias, result in zip(biases, results, strict=True):
            comparison.add(band, bias, result)
    return comparison


def improve(habit: Habit, comparison: Comparison, z: float = 2.0) -> tuple[Habit, list[dict[str, Any]]]:
    """The habit with, in each band, the candidate of fewest mean moves if it is better by ``z`` standard
    errors; and the changes made."""
    if comparison.habit != habit:
        raise ValueError("the comparison is of another habit")
    changes = []
    for band in range(BANDS):
        options = comparison.candidates(band)
        if not options:
            continue
        best = min(options, key=lambda b: comparison.mean(band, b))
        mean, se = comparison.mean(band, best), comparison.se(band, best)
        if mean + z * se < 0.0:
            changes.append(
                {
                    "band": f"{100 * band // BANDS}-{100 * (band + 1) // BANDS}%",
                    "from": _format_bias(habit.biases[band]),
                    "to": _format_bias(best),
                    "mean": round(mean, 3),
                    "se": round(se, 3),
                }
            )
            habit = habit.replace(band, best)
    return habit, changes


# --------------------------------------------------------------- distillation
class _DistillGame:
    """A game that records distillation examples: inside the envelope the habit's move and target, elsewhere
    the raw policy (:class:`~snake_agent.search.PolicyAgent`'s move) of the network that answers its
    queries there, and that network's outputs (see :func:`distillation_examples`)."""

    def __init__(self, start: Start, habit: Habit, rules: Rules, smoothing: float, lost_budget: int) -> None:
        self.start, self.habit, self.rules = start, habit, rules
        self.smoothing, self.lost_budget = smoothing, lost_budget
        self.game = start.game()
        self.epoch = _Epoch()
        self.pending: tuple[str, GameState, Any, Query] | None = None
        self.rows: list[tuple] = []
        """``(query, valid, policy, win, moves, label)`` for each decision with a choice."""
        self.afterstates: list[GameState] = []
        self.changed = 0
        """Envelope decisions where the habit's move is not the raw policy's."""
        self.moves = 0

    def advance(self) -> Query | None:
        """Play until a decision needs the evaluator (returns its query) or the game is over (None)."""
        while self.pending is None:
            game = self.game
            state = game.state()
            if game.done or self.rules.timed_out(state):
                return None
            history = self.epoch.observe(state)
            if envelope_certain(state, self.rules):
                jumps = canonical_envelope(state.width, state.height).moves(state)
                if len(jumps) == 1:
                    self._move(self.habit.choose(state, jumps))
                    continue
                query = Query(state, self.rules)
                self.pending = ("envelope", state, jumps, query)
                return query
            root, _ = _root(state, self.rules, Mode.FREE, history, self.lost_budget)
            if len(root.actions) <= 1:
                self._move(root.actions[0] if root.actions else RelativeAction.STRAIGHT)
                continue
            query = Query(state, self.rules, False, root.classes)
            self.pending = ("free", state, root.actions, query)
            return query
        raise RuntimeError("a decision is waiting for its value")

    def receive(self, value: LeafValue) -> None:
        kind, state, options, query = self.pending
        self.pending = None
        logits = np.asarray(value.logits, dtype=np.float64)
        if kind == "envelope":
            action = self.habit.choose(state, options, logits)
            valid = tuple(np.isfinite(self.habit.scores(state, options, logits)))
            policy = self.habit.target(state, options, logits, self.smoothing)
            raw = max((a for a in ACTIONS if valid[a]), key=lambda a: logits[a])
            self.changed += action != raw
            self.rows.append((query, valid, policy, 1.0, value.moves, Label.ENVELOPE))
        else:
            action = max(options, key=lambda a: logits[a])
            valid = tuple(a in options for a in ACTIONS)
            policy = np.zeros(3)
            index = [int(a) for a in options]
            z = np.exp(logits[index] - logits[index].max())
            policy[index] = z / z.sum()
            self.rows.append((query, valid, policy, value.win, value.moves, Label.TEACHER))
        self._move(action)

    def _move(self, action: RelativeAction) -> None:
        step = self.game.step_relative(action)
        self.moves += 1
        if step.ate_fruit and not step.won:
            self.afterstates.append(dataclasses.replace(self.game.state(), fruit=None))


def distillation_examples(
    evaluator: Evaluator,
    habit: Habit,
    starts: Sequence[Start],
    rules: Rules = TRAINING_RULES,
    *,
    anchor: Evaluator | None = None,
    size: int | None = None,
    smoothing: float = 0.02,
    lost_budget: int = 10_000,
) -> tuple[Examples, dict[str, Any]]:
    """Training examples that teach a network to play ``habit`` with its raw policy, and statistics.

    ``evaluator`` is the network before distillation, the teacher, and ``anchor`` the network whose play
    outside the envelope the distilled one keeps (by default the teacher). A game is played from each
    start: inside the envelope with the habit on the teacher's logits, elsewhere with the anchor's raw
    policy. Each decision with a choice gives an example: inside the envelope, the habit's policy
    (:meth:`Habit.target`) over the envelope moves and the teacher's moves value; elsewhere, the anchor's
    own policy over the moves the raw policy considers and its values (``Label.TEACHER``). Every
    afterstate gives the teacher's moves value inside the envelope and the anchor's values elsewhere. So
    the distilled network changes its policy inside the envelope and keeps the rest; with the network a
    run started from as the anchor of every round, the rest cannot drift from round to round. Encodings
    use a ``size`` canvas (by default the largest board side).
    """
    if not starts:
        raise ValueError("no starts")
    anchor = evaluator if anchor is None else anchor
    games = [_DistillGame(start, habit, rules, smoothing, lost_budget) for start in starts]
    waiting = games
    while waiting:
        asking: dict[bool, list[_DistillGame]] = {True: [], False: []}
        queries: dict[bool, list[Query]] = {True: [], False: []}
        for game in waiting:
            query = game.advance()
            if query is not None:
                inside = game.pending[0] == "envelope"
                asking[inside].append(game)
                queries[inside].append(query)
        for inside, network in ((True, evaluator), (False, anchor)):
            if queries[inside]:
                prediction = network(queries[inside])
                for i, game in enumerate(asking[inside]):
                    game.receive(prediction.row(i))
        waiting = asking[True] + asking[False]
    size = size or max(max(s.state.width, s.state.height) for s in starts)
    rows = []
    for game in games:
        for query, valid, policy, win, moves, label in game.rows:
            state = query.state
            encoded = encode(state, rules, size, classes=query.classes)
            rows.append(
                (encoded, _frame_width(state), valid, policy, True, win, moves, True, label, state.free_cells)
            )
    afterstates = [after for game in games for after in game.afterstates]
    for inside, network in ((True, evaluator), (False, anchor)):
        group = [after for after in afterstates if envelope_certain(after, rules) is inside]
        for i in range(0, len(group), 1024):
            batch = group[i : i + 1024]
            prediction = network([Query(after, rules, afterstate=True) for after in batch])
            for after, win, moves in zip(batch, prediction.win, prediction.moves, strict=True):
                rows.append(
                    (
                        encode(after, rules, size, afterstate=True),
                        _frame_width(after),
                        (False, False, False),
                        np.zeros(3),
                        False,
                        1.0 if inside else float(win),
                        float(moves),
                        True,
                        Label.ENVELOPE if inside else Label.TEACHER,
                        after.free_cells,
                    )
                )
    standard = [
        g for g in games if g.start.kind == "standard" and g.game.won
    ]  # moves of the habit's T1 games
    stats = {
        "games": {
            kind: sum(g.start.kind == kind for g in games) for kind in sorted({g.start.kind for g in games})
        },
        "examples": len(rows),
        "envelope_decisions": sum(r[5] is Label.ENVELOPE for g in games for r in g.rows),
        "free_decisions": sum(r[5] is Label.TEACHER for g in games for r in g.rows),
        "afterstates": len(afterstates),
        "changed": sum(g.changed for g in games),
        "standard_moves": [g.moves for g in standard],
    }
    return _to_examples(rows), stats
