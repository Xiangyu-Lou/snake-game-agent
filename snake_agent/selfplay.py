"""Self-play for milestone M1: games played with the search, and the training examples they give.

Many games run in one process and share each network call: :func:`play`
advances every game's search to its next leaf, evaluates all the leaves in one
batch, and repeats (review §2.4). The live games are :class:`~snake_game.SnakeGame`
objects that the search never sees: it plans on the public state only.

**Starts.** A game starts from the standard start (:func:`standard_starts`),
where the search stays inside the cycle-order envelope and every game is won;
from the standard start after a random prefix (:func:`prefix_starts`): random
envelope moves up to a random length, then the agent; or from a *free start*
(:func:`free_starts`): a running state of a reference policy's game that is not
a certain win inside the envelope, where the search must judge win
probabilities. Random jumps leave holes that the agent's own games rarely show,
so prefixes train the network on the kind of position that lies behind the
moves its policy does not play, which the search must still judge. Seeds must
be training seeds (:func:`~snake_agent.seeds.check_training_seed`).

**Targets.** Every move gives an example of the state before it, and every
eating move that does not fill the board one of its *afterstate*, the board
before the next fruit appears (the search evaluates those too). Each win target
records its provenance (:class:`Label`):

* ``ENVELOPE``: 1, exact. The state is a certain win inside the envelope.
* ``PROVEN_LOST``: 0, exact. Every move is fatal, proven lost or times out.
* ``BLEND``: the game's outcome ``z`` blended with the search's value ``v`` of
  the state, ``(1 - beta) z + beta v`` (review §2.5).
* ``OUTCOME``: the outcome alone, at moves that were not searched.
* ``AFTERSTATE``: the outcome blended with the search's value of the state
  after the next fruit appeared, an unbiased sample of the afterstate's value.
* ``SIBLING``: optionally, in envelope mode, the position after a move the
  search visited but did not play (1, exact). Its moves target is the search's
  value of that move: the search compares exactly those positions, and the
  policy's own games rarely show them.

The moves target, for won games only, is a TD(lambda) return over the game's
fruits, in units of :func:`~snake_agent.encoding.moves_scale`: the moves until
the visible fruit is eaten, then the network's value of the afterstate, blended
with the return from there, ``(1 - lambda) V(afterstate) + lambda G(afterstate)``
(:func:`moves_returns`). ``lambda = 1`` is the moves the game still took. That
count varies by tens of moves with the fruits still to come, while the moves
the search must compare differ by about one, so the network's own afterstate
values stand in for the later fruits; the last fruit's return is exact. The
policy target is the search's improved policy, at moves that were searched and
not lost.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from snake_game import Direction, GameConfig, GameState, RelativeAction, SnakeGame
from snake_game.policies import GreedyPolicy, RandomPolicy

from .contract import TRAINING_RULES, Outcome, Rules, planning_game
from .encoding import canonical_envelope, encode, moves_ceiling, moves_scale
from .references import GuardedGreedyPolicy
from .search import (
    SELF_PLAY,
    Evaluator,
    LeafValue,
    Mode,
    Query,
    SearchConfig,
    SearchResult,
    _Epoch,
    envelope_certain,
    search,
)
from .seeds import check_training_seed


class Label(IntEnum):
    """Where a win target comes from (see the module documentation)."""

    OUTCOME = 0
    BLEND = 1
    ENVELOPE = 2
    PROVEN_LOST = 3
    AFTERSTATE = 4
    SIBLING = 5


# --------------------------------------------------------------------- starts
@dataclass(frozen=True)
class Start:
    """Where a self-play game starts, and the seed of its fruits."""

    state: GameState
    seed: int
    standard: bool
    """The standard start (its first fruit from ``seed`` too), or a free start."""
    prefix: int = 0
    """Standard starts: random envelope moves are played until the snake is this long, then the agent
    takes over (0: no prefix). The prefix gives no examples."""

    def game(self) -> SnakeGame:
        """The live game, positioned where the agent takes over."""
        if not self.standard:
            return SnakeGame.from_state(self.state, seed=self.seed)
        game = SnakeGame(GameConfig(self.state.width, self.state.height), seed=self.seed)
        if self.prefix:
            envelope = canonical_envelope(game.width, game.height)
            rng = random.Random(f"{self.seed}/prefix")
            while game.length < self.prefix and not game.done:
                game.step(rng.choice(sorted(envelope.moves(game.state()))))
        return game


def standard_starts(width: int, height: int, seeds: Sequence[int]) -> list[Start]:
    """One standard start per training seed."""
    config = GameConfig(width, height)
    return [Start(SnakeGame(config, seed=check_training_seed(s)).state(), s, True) for s in seeds]


def prefix_starts(width: int, height: int, seeds: Sequence[int], fill: float = 0.75) -> list[Start]:
    """One standard start per training seed whose first moves are uniformly random envelope moves.

    The prefix lasts until the snake is as long as a target drawn uniformly
    between one segment more than at the start and ``fill`` of the board.
    Envelope moves keep the win certain, so the agent takes over a won game.
    """
    config = GameConfig(width, height)
    cells = width * height
    starts = []
    for seed in seeds:
        state = SnakeGame(config, seed=check_training_seed(seed)).state()
        shortest = state.length + 1
        target = random.Random(seed).randint(shortest, max(shortest, min(int(fill * cells), cells - 2)))
        starts.append(Start(state, seed, True, target))
    return starts


def free_starts(
    width: int,
    height: int,
    seeds: Sequence[int],
    rules: Rules = TRAINING_RULES,
    fruit_offset: int = 500_000_000,
) -> list[Start]:
    """One free start per training seed, where one exists.

    The seed's game is played from the standard start by a reference policy
    (random, greedy or guarded greedy, chosen by the seed) for at most ``4 N``
    moves. The start is one of its running states, drawn uniformly, that is not
    a certain win inside the envelope and has not timed out under ``rules``.
    The self-play game's fruits then come from ``seed + fruit_offset``, so they
    differ from the source game's.
    """
    cells = width * height
    starts = []
    for seed in seeds:
        rng = random.Random(check_training_seed(seed))
        policy = rng.choice([RandomPolicy(seed), GreedyPolicy(), GuardedGreedyPolicy()])
        game = SnakeGame(GameConfig(width, height), seed=seed)
        candidates = []
        while not game.done and game.steps < 4 * cells:
            state = game.state()
            if not rules.timed_out(state) and not envelope_certain(state, rules):
                candidates.append(state)
            game.step(policy(state))
        if candidates:
            starts.append(Start(rng.choice(candidates), check_training_seed(seed + fruit_offset), False))
    return starts


# ------------------------------------------------------------------- examples
@dataclass(frozen=True)
class Examples:
    """Training examples: encoded states and their targets, as arrays over the examples."""

    planes: np.ndarray
    """``(n, C, S, S)`` float16 (exact for the encodings' values on boards up to 20x20)."""
    actions: np.ndarray
    globals: np.ndarray
    head: np.ndarray
    width: np.ndarray
    """Columns of the canvas in use, in the head's frame (for :func:`~snake_agent.encoding.mirror`)."""
    valid: np.ndarray
    """``(n, 3)``: the moves the search considered."""
    policy: np.ndarray
    has_policy: np.ndarray
    win: np.ndarray
    moves: np.ndarray
    """In units of ``moves_scale(free cells)``; meaningful where ``has_moves`` (won games)."""
    has_moves: np.ndarray
    label: np.ndarray
    free: np.ndarray

    def __len__(self) -> int:
        return len(self.win)

    def take(self, index: np.ndarray | slice) -> Examples:
        return Examples(*(getattr(self, f.name)[index] for f in dataclasses.fields(self)))

    @staticmethod
    def concatenate(parts: Sequence[Examples]) -> Examples:
        return Examples(
            *(np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(Examples))
        )


@dataclass(frozen=True)
class GameSummary:
    """One self-play game."""

    standard: bool
    seed: int
    outcome: Outcome
    moves: int
    """Moves the agent played (after the prefix, if any)."""
    fruits: int
    length: int
    start_length: int
    """Length when the agent took over."""
    searched: int
    """Moves chosen by a search (the others had at most one move left)."""
    prefix: int = 0
    """Random envelope moves played before the agent took over."""


class _Game:
    """One self-play game: the live game, its search in progress and its record."""

    def __init__(self, start: Start, rules: Rules, config: SearchConfig) -> None:
        self.start, self.rules, self.config = start, rules, config
        self.game = start.game()
        self.prefix, self.start_length = self.game.steps if start.prefix else 0, self.game.length
        self.epoch = _Epoch()
        self.search = None
        self.state: GameState | None = None
        self.value: LeafValue | None = None
        self.steps: list[tuple[GameState, SearchResult]] = []
        self.afterstates: list[tuple[int, GameState]] = []
        self.outcome: Outcome | None = None

    def advance(self, rng: np.random.Generator) -> Query | None:
        """Play until the search needs an evaluation (returned) or the game ends (None)."""
        while self.outcome is None:
            if self.search is None:
                self.state = self.game.state()
                self.search = search(self.state, self.rules, self.config, rng, self.epoch.observe(self.state))
            value, self.value = self.value, None
            try:
                return self.search.send(value)
            except StopIteration as stop:
                self.search = None
                self._move(stop.value)
        return None

    def _move(self, result: SearchResult) -> None:
        self.steps.append((self.state, result))
        step = self.game.step_relative(result.action)
        if step.ate_fruit and not step.won:
            self.afterstates.append((len(self.steps) - 1, dataclasses.replace(self.game.state(), fruit=None)))
        if self.game.won:
            self.outcome = Outcome.WON
        elif self.game.done:
            self.outcome = Outcome.LOST
        elif self.rules.timed_out(self.game.state()):
            self.outcome = Outcome.TIMED_OUT

    def summary(self) -> GameSummary:
        return GameSummary(
            self.start.standard,
            self.start.seed,
            self.outcome,
            len(self.steps),
            self.game.length - self.start_length,
            self.game.length,
            self.start_length,
            sum(not result.forced for _, result in self.steps),
            self.prefix,
        )

    def examples(
        self, beta: float, moves_lambda: float, values: Sequence[float], size: int, sibling_visits: int = 0
    ) -> list[tuple]:
        """The game's examples as rows: (encoding, width, valid, policy, has_policy, win, moves, has_moves,
        label, free). ``values`` are the network's moves values of the afterstates (won games only);
        with ``sibling_visits``, envelope moves visited that often but not played give examples too."""
        won = self.outcome is Outcome.WON
        z = float(won)
        eats = [t for t, _ in self.afterstates]
        decisions, afters = (
            moves_returns(len(self.steps), eats, values, moves_lambda)
            if won
            else ([0.0] * len(self.steps), [])
        )
        rows = []
        for t, (state, result) in enumerate(self.steps):
            searched = not result.forced and not result.lost
            if result.mode is Mode.ENVELOPE:
                win, label = 1.0, Label.ENVELOPE
            elif result.lost:
                win, label = 0.0, Label.PROVEN_LOST
            elif searched:
                win, label = (1.0 - beta) * z + beta * result.value_win, Label.BLEND
            else:
                win, label = z, Label.OUTCOME
            moves = decisions[t] / moves_scale(state.free_cells)
            policy = result.policy if searched else np.zeros(3)
            rows.append(
                (
                    encode(state, self.rules, size),
                    _frame_width(state),
                    result.valid,
                    policy,
                    searched,
                    win,
                    moves,
                    won,
                    label,
                    state.free_cells,
                )
            )
            if sibling_visits and searched and result.mode is Mode.ENVELOPE:
                rows += self._siblings(state, result, sibling_visits, size)
        for k, (t, after) in enumerate(self.afterstates):
            _, following = self.steps[t + 1]
            moves = afters[k] / moves_scale(after.free_cells) if won else 0.0
            if envelope_certain(after, self.rules) or following.mode is Mode.ENVELOPE:
                win, label = 1.0, Label.ENVELOPE
            else:
                win, label = z, Label.AFTERSTATE
                if not following.forced and not following.lost:
                    win = (1.0 - beta) * z + beta * following.value_win
            rows.append(
                (
                    encode(after, self.rules, size, afterstate=True),
                    _frame_width(after),
                    (False, False, False),
                    np.zeros(3),
                    False,
                    win,
                    moves,
                    won,
                    label,
                    after.free_cells,
                )
            )
        return rows

    def _siblings(self, state: GameState, result: SearchResult, visits: int, size: int) -> list[tuple]:
        """Examples of the positions after the moves the search visited at least ``visits`` times but did
        not play, with the search's moves value of each move as the target (envelope mode)."""
        rows = []
        game = planning_game(state, seed=0)
        for action in RelativeAction:
            if action == result.action or not result.valid[action] or result.visits[action] < visits:
                continue
            child = game.clone()
            step = child.step_relative(action)
            if step.won:
                continue
            after = dataclasses.replace(child.state(), fruit=None) if step.ate_fruit else child.state()
            rows.append(
                (
                    encode(after, self.rules, size, afterstate=step.ate_fruit),
                    _frame_width(after),
                    (False, False, False),
                    np.zeros(3),
                    False,
                    1.0,
                    (result.q_moves[action] - 1.0) / moves_scale(after.free_cells),
                    True,
                    Label.SIBLING,
                    after.free_cells,
                )
            )
        return rows


def moves_returns(
    total: int, eats: Sequence[int], values: Sequence[float], moves_lambda: float
) -> tuple[list[float], list[float]]:
    """TD(lambda) returns of a won game's moves, over its fruits: one per move, one per afterstate.

    ``total`` moves were played, the last of them filling the board; ``eats[k]``
    is the move that ate the ``k``-th fruit before it, leaving afterstate ``k``,
    whose moves value is ``values[k]``. From afterstate ``k`` the next fruit is
    eaten ``d`` moves later, so its return is ``d + (1 - lambda) values[k + 1]
    + lambda G[k + 1]``, and ``d`` alone for the last fruit. A move ``t`` returns
    the moves until its fruit is eaten, then the same blend at that afterstate.
    """
    nexts = list(eats) + [total - 1]
    afters = [0.0] * len(eats)
    for k in range(len(eats) - 1, -1, -1):
        afters[k] = nexts[k + 1] - nexts[k]
        if k + 1 < len(eats):
            afters[k] += (1.0 - moves_lambda) * values[k + 1] + moves_lambda * afters[k + 1]
    decisions, k = [], 0
    for t in range(total):
        while nexts[k] < t:
            k += 1
        ahead = nexts[k] - t + 1
        if k < len(eats):
            ahead += (1.0 - moves_lambda) * values[k] + moves_lambda * afters[k]
        decisions.append(float(ahead))
    return decisions, afters


def _frame_width(state: GameState) -> int:
    """Columns of the board in the head's frame: the width when heading up or down, else the height."""
    return state.width if state.direction in (Direction.UP, Direction.DOWN) else state.height


def _to_examples(rows: list[tuple]) -> Examples:
    encoded = [row[0] for row in rows]
    return Examples(
        planes=np.stack([e.planes for e in encoded]).astype(np.float16),
        actions=np.stack([e.actions for e in encoded]).astype(np.float16),
        globals=np.stack([e.globals for e in encoded]),
        head=np.array([e.head for e in encoded], dtype=np.int64),
        width=np.array([row[1] for row in rows], dtype=np.int64),
        valid=np.array([row[2] for row in rows], dtype=bool),
        policy=np.array([row[3] for row in rows], dtype=np.float32),
        has_policy=np.array([row[4] for row in rows], dtype=bool),
        win=np.array([row[5] for row in rows], dtype=np.float32),
        moves=np.array([row[6] for row in rows], dtype=np.float32),
        has_moves=np.array([row[7] for row in rows], dtype=bool),
        label=np.array([row[8] for row in rows], dtype=np.int8),
        free=np.array([row[9] for row in rows], dtype=np.int16),
    )


# ----------------------------------------------------------------------- play
def play(
    evaluator: Evaluator,
    starts: Sequence[Start],
    rules: Rules = TRAINING_RULES,
    config: SearchConfig = SELF_PLAY,
    *,
    seed: int | Sequence[int] = 0,
    beta: float = 0.5,
    moves_lambda: float = 0.5,
    size: int | None = None,
    sibling_visits: int = 0,
) -> tuple[list[GameSummary], Examples]:
    """Play a game from each start, all at once, and return their summaries and training examples.

    ``seed`` drives the searches' randomness (Gumbel noise and sampled
    fruits), never the games' fruits. ``beta`` is the weight of the search's
    values in the win targets, ``moves_lambda`` the lambda of the moves
    targets (1: the moves the game still took), and ``size`` the canvas of the
    encodings (by default the largest board side). The afterstates of won games
    are evaluated once more at the end, for the moves targets. With
    ``sibling_visits``, envelope moves the search visited that often but did not
    play give examples too (``Label.SIBLING``).
    """
    if not rules.fruit_timeout:
        raise ValueError("self-play needs the fruit timeout, which turns loops into losses")
    if not starts:
        raise ValueError("no starts")
    rng = np.random.default_rng(seed)
    games = [_Game(start, rules, config) for start in starts]
    waiting = games
    while waiting:
        asking, queries = [], []
        for game in waiting:
            query = game.advance(rng)
            if query is not None:
                asking.append(game)
                queries.append(query)
        if queries:
            prediction = evaluator(queries)
            for i, game in enumerate(asking):
                game.value = prediction.row(i)
        waiting = asking
    size = size or max(max(s.state.width, s.state.height) for s in starts)
    afterstates = [after for game in games if game.outcome is Outcome.WON for _, after in game.afterstates]
    values = iter(
        _moves_values(evaluator, afterstates, rules) if moves_lambda < 1.0 else [0.0] * len(afterstates)
    )
    rows = []
    for game in games:
        mine = [next(values) for _ in game.afterstates] if game.outcome is Outcome.WON else []
        rows += game.examples(beta, moves_lambda, mine, size, sibling_visits)
    return [game.summary() for game in games], _to_examples(rows)


def _moves_values(evaluator: Evaluator, afterstates: Sequence[GameState], rules: Rules) -> list[float]:
    """The evaluator's moves values of ``afterstates``, in moves (within the ceiling)."""
    values = []
    for i in range(0, len(afterstates), 1024):
        batch = afterstates[i : i + 1024]
        prediction = evaluator([Query(after, rules, afterstate=True) for after in batch])
        for after, moves in zip(batch, prediction.moves, strict=True):
            ceiling = moves_ceiling(after.num_cells, after.free_cells)
            values.append(min(max(float(moves), 0.0), ceiling) * moves_scale(after.free_cells))
    return values
