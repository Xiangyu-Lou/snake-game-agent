"""Gumbel root search for milestone M1: few simulations, used where they matter.

The search plays moves forward on the public state with
:func:`~snake_agent.contract.planning_game` and asks an evaluator (the network)
for a prior and values at its leaves. It follows Gumbel MuZero (Danihelka et
al., *Policy improvement by planning with Gumbel*, ICLR 2022): sequential
halving over the root's moves, completed Q-values, and the improved policy
``softmax(logits + sigma(q))`` as the policy target. With at most three moves,
16 simulations give a policy-improvement target.

The contract's safety rule picks one of two modes at the root:

* **Envelope mode**: the body is in cycle order on the board's canonical cycle
  and the fruit can be reached in time (:func:`envelope_certain`), as on every
  standard start. Only envelope moves are searched. Each keeps the win certain,
  so every win value is exactly 1 and the utility is the expected number of
  moves, fewer being better.
* **Free mode**: everywhere else. The utility is the win probability, and
  expected moves only order moves whose win values tie exactly. A state in
  cycle order is an exact win, so it becomes a leaf with win value 1.

Leaves and pruning:

* A move that eats ends the simulation at the *afterstate*, the board before
  the next fruit appears, which the evaluator values as such. Later
  simulations through it draw the fruit uniformly from the free cells with the
  search's own random generator, never the game's.
* Fatal moves and moves that end the game by the fruit timeout are pruned. In
  free mode so are moves back to a position of the current fruit's epoch: the
  same position with less budget. At the root of a free search, moves proven
  lost by :func:`~snake_agent.survival.prove_lost` are pruned too. A node left
  without moves is a loss.

Values are pairs: the probability of filling the board, and the expected moves
until the board is full *if it is filled*. Lost games never enter a moves value:
backups weight moves by the win value, so dying cannot look fast, and slow wins
need no ceiling. (A late fruit in a hole behind the tail can cost many times the
plain cycle's pace, so a fixed value for losses would make some wins look slower
than dying.) Moves only ever compare moves whose win values tie exactly, and a
learned win value is never exactly 1: only the exact instruments (the envelope,
a filled board) make a win certain.

The search is a generator, so that a driver can batch the leaves of many games
into one network call: :func:`search` yields one :class:`Query` at a time,
expects a :class:`LeafValue` back, and returns a :class:`SearchResult`.
:func:`run_search` drives it with one evaluator call per leaf.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Generator, Hashable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from snake_game import GameState, RelativeAction, SnakeGame, Status

from .contract import TRAINING_RULES, Rules, planning_game, to_relative
from .encoding import canonical_envelope, moves_ceiling, moves_scale
from .survival import MoveClass, classify_moves

ACTIONS = tuple(RelativeAction)


class Mode(str, Enum):
    """How the search compares moves (see the module documentation)."""

    ENVELOPE = "envelope"  # inside the cycle-order envelope: every move is a certain win, fewer moves first
    FREE = "free"  # win probability first, moves only on exact ties


# ------------------------------------------------------------------ evaluator
@dataclass(frozen=True)
class Query:
    """A state to evaluate.

    ``afterstate``: the state just after eating, before the next fruit appears
    (its ``fruit`` is ``None``). ``classes`` are the move classes the search
    computed, for the encoder to reuse.
    """

    state: GameState
    rules: Rules
    afterstate: bool = False
    classes: tuple[MoveClass, MoveClass, MoveClass] | None = None


@dataclass(frozen=True)
class LeafValue:
    """The evaluation of one query."""

    logits: np.ndarray
    """Policy logits over ``LEFT``, ``STRAIGHT``, ``RIGHT`` (ignored for afterstates)."""
    win: float
    """Probability of filling the board."""
    moves: float
    """Expected moves until the board is full if it is filled, in units of ``moves_scale(free cells)``."""


@dataclass(frozen=True)
class Prediction:
    """The evaluations of a batch of queries: ``logits (n, 3)``, ``win (n,)``, ``moves (n,)``."""

    logits: np.ndarray
    win: np.ndarray
    moves: np.ndarray

    def __len__(self) -> int:
        return len(self.win)

    def row(self, i: int) -> LeafValue:
        return LeafValue(self.logits[i], float(self.win[i]), float(self.moves[i]))


Evaluator = Callable[[Sequence[Query]], Prediction]
"""Evaluates a batch of queries, for example with the network."""


@dataclass(frozen=True)
class SearchConfig:
    """Search settings; the defaults are M1's (16 simulations, Gumbel noise for self-play)."""

    simulations: int = 16
    gumbel_scale: float = 1.0
    """Scale of the root's Gumbel noise: 1 for self-play, 0 for evaluation (deterministic choice)."""
    c_visit: float = 50.0
    c_scale: float = 0.1
    """``sigma(q) = (c_visit + max visits) * c_scale * q`` for ``q`` rescaled to [0, 1], as in the paper."""
    tie_weight: float = 0.25
    """Free mode: share of the rescaled Q given to moves among the moves tied on the best win value."""
    moves_unit: float | None = 1.0
    """Envelope mode: ``q = -moves / moves_unit``, so that moves differing by much less than a unit count
    for little against the prior. None rescales the moves to [0, 1] at each node, as the paper does with
    values, which turns the moves head's errors into full-size differences. The best unit grows with
    those errors: 1 on 4x4, 16 for the 6x6 network at 16 simulations (``snake_agent/README.md``)."""
    lost_budget: int = 10_000
    """States :func:`~snake_agent.survival.prove_lost` may visit per root move of a free search."""
    search_forced: bool = False
    """Also search when only one move is left (otherwise it is played without a network call)."""


SELF_PLAY = SearchConfig()
"""The search used in self-play: Gumbel noise at the root."""
EVALUATION = SearchConfig(gumbel_scale=0.0)
"""The search used to evaluate an agent: no Gumbel noise."""


@dataclass(frozen=True)
class SearchResult:
    """What the search found at the root.

    Arrays are over ``LEFT``, ``STRAIGHT``, ``RIGHT``; moves are absolute
    expected moves until the board is full, if it is filled. The Q-values are
    *completed*: a valid move that was never visited gets the root's mixed value.
    """

    action: RelativeAction
    mode: Mode
    valid: tuple[bool, bool, bool]
    """The moves the search considered (after pruning and fallbacks)."""
    forced: bool
    """At most one move was left, so there was no search and no policy target."""
    lost: bool
    """Every move is fatal, proven lost or times out: the state is lost (an exact value label)."""
    simulations: int
    visits: np.ndarray
    policy: np.ndarray | None
    """The improved policy ``softmax(logits + sigma(q))`` over the valid moves; None if forced."""
    q_win: np.ndarray
    q_moves: np.ndarray
    value_win: float | None
    """The root's value under the improved policy; None if forced."""
    value_moves: float | None
    prior_win: float | None
    """The evaluator's values for the root itself; None if forced."""
    prior_moves: float | None


# ----------------------------------------------------------------- envelope
def envelope_certain(state: GameState, rules: Rules) -> bool:
    """Whether envelope moves alone win ``state`` for certain, under ``rules``.

    The body must be in cycle order on the canonical cycle. Envelope moves
    then eat the fruit within ``ahead(fruit)`` moves, so under the fruit
    timeout the fruit must also be near enough along the cycle. An afterstate
    (no fruit yet) only needs cycle order: every later fruit is eaten in time.
    """
    envelope = canonical_envelope(state.width, state.height)
    if envelope is None or not envelope.contains(state):
        return False
    limit = rules.timeout(state)
    if limit is None or state.fruit is None:
        return True
    return state.steps_since_fruit + envelope.ahead(state, state.fruit) <= limit


def position_key(state: GameState) -> Hashable:
    """What makes two states the same position within one fruit's epoch."""
    return (state.snake, state.direction, state.fruit)


# -------------------------------------------------------------------- nodes
class _Status(Enum):
    OK = 0
    FATAL = 1
    LOST = 2  # proven lost at the root
    TIMEOUT = 3
    REPEAT = 4


class _Node:
    """Common statistics: visits, and sums of the win values and of the win-weighted moves backed up."""

    __slots__ = ("state", "visits", "win_sum", "moves_sum", "raw_win", "raw_moves", "evaluated", "leaf")

    def __init__(self, state: GameState) -> None:
        self.state = state
        self.visits = 0
        self.win_sum = 0.0
        self.moves_sum = 0.0
        self.raw_win = 0.0
        self.raw_moves = 0.0
        self.evaluated = False
        self.leaf = False  # terminal or exact: never expanded

    def set_value(self, win: float, moves: float) -> None:
        self.raw_win, self.raw_moves, self.evaluated = win, moves, True


class _Terminal(_Node):
    __slots__ = ()

    def __init__(self, state: GameState, win: float, moves: float) -> None:
        super().__init__(state)
        self.set_value(win, moves)
        self.leaf = True


class _Chance(_Node):
    """An afterstate: the fruit is not placed yet."""

    __slots__ = ("children", "free", "certain")

    def __init__(self, state: GameState, certain: bool) -> None:
        super().__init__(state)
        self.children: dict[tuple[int, int], _Decision] = {}
        occupied = set(state.snake)
        self.free = [
            (x, y) for y in range(state.height) for x in range(state.width) if (x, y) not in occupied
        ]
        self.certain = certain


class _Decision(_Node):
    """A state with its fruit, where the agent moves."""

    __slots__ = ("actions", "successors", "children", "logits", "history", "certain", "classes", "status")

    def __init__(self, state: GameState) -> None:
        super().__init__(state)
        self.actions: list[RelativeAction] = []
        self.successors: dict[RelativeAction, tuple[GameState, bool]] = {}
        self.children: dict[RelativeAction, _Node] = {}
        self.logits = np.zeros(3)
        self.history: frozenset = frozenset()
        self.certain = False
        self.classes: tuple[MoveClass, MoveClass, MoveClass] | None = None
        self.status: dict[RelativeAction, _Status] = {}


LOSS = (0.0, 0.0)
"""The value of a lost state: no win, and so no moves (they are weighted by the win value)."""

CERTAIN = 1.0 - 1e-6
"""The highest win value a learned evaluation can give: only exact wins tie at 1."""


def _ceiling(state: GameState) -> float:
    """The most moves a won game can still take from ``state`` (see :func:`moves_ceiling`)."""
    return moves_ceiling(state.num_cells, state.free_cells) * moves_scale(state.free_cells)


def _moves(moves_sum: float, win_sum: float, state: GameState) -> float:
    """Expected moves if the board is filled: win-weighted moves over win values (the ceiling if no win)."""
    return moves_sum / win_sum if win_sum > 0.0 else _ceiling(state)


def _successor(game: SnakeGame, action: RelativeAction) -> tuple[GameState, bool]:
    """The state after ``action`` on a clone of ``game``, and whether it ate; afterstates have no fruit."""
    game = game.clone()
    result = game.step_relative(action)
    state = game.state()
    if result.ate_fruit and not result.won:
        state = dataclasses.replace(state, fruit=None)
    return state, result.ate_fruit


def _decision(state: GameState, rules: Rules, mode: Mode, history: frozenset, budget: int = 0) -> _Decision:
    """A new decision node with its moves classified; ``history`` holds the epoch's earlier positions."""
    node = _Decision(state)
    node.history = history | {position_key(state)}
    game = planning_game(state, seed=0)  # the seed is irrelevant: eating moves stop at the afterstate
    if mode is Mode.ENVELOPE:
        node.certain = True
        jumps = canonical_envelope(state.width, state.height).moves(state)
        for direction in jumps:
            action = to_relative(state.direction, direction)
            node.status[action] = _Status.OK
            node.successors[action] = _successor(game, action)
    else:
        if envelope_certain(state, rules):
            node.certain = node.leaf = True  # an exact win: only its expected moves are needed
            return node
        classes = classify_moves(state, budget)
        node.classes = tuple(MoveClass.UNKNOWN if c is MoveClass.LOST else c for c in classes)
        for action in ACTIONS:
            if classes[action] is MoveClass.FATAL:
                node.status[action] = _Status.FATAL
                continue
            successor, ate = _successor(game, action)
            node.successors[action] = successor, ate
            if classes[action] is MoveClass.LOST:
                node.status[action] = _Status.LOST
            elif not ate and rules.timed_out(successor):
                node.status[action] = _Status.TIMEOUT
            elif not ate and position_key(successor) in node.history:
                node.status[action] = _Status.REPEAT
            else:
                node.status[action] = _Status.OK
    node.actions = [a for a in ACTIONS if node.status.get(a) is _Status.OK]
    return node


def _child(node: _Decision, action: RelativeAction, rules: Rules, mode: Mode) -> _Node:
    state, ate = node.successors[action]
    if state.won:
        return _Terminal(state, 1.0, 0.0)
    if ate:
        return _Chance(state, certain=mode is Mode.ENVELOPE or envelope_certain(state, rules))
    child = _decision(state, rules, mode, node.history)
    if not child.leaf and not child.actions:
        return _Terminal(state, *LOSS)
    return child


# ------------------------------------------------------- sequential halving
@functools.cache
def considered_visits(actions: int, simulations: int) -> tuple[int, ...]:
    """The visit count of the moves to consider at each simulation (sequential halving).

    The schedule of ``mctx.seq_halving``: each phase gives every remaining
    move ``max(1, simulations // (ceil(log2(actions)) * remaining))`` more
    visits, and then the better half stays.
    """
    if actions <= 1:
        return tuple(range(simulations))
    log2 = math.ceil(math.log2(actions))
    sequence: list[int] = []
    visits = [0] * actions
    considered = actions
    while len(sequence) < simulations:
        extra = max(1, int(simulations / (log2 * considered)))
        for _ in range(extra):
            sequence.extend(visits[:considered])
            for i in range(considered):
                visits[i] += 1
        considered = max(2, considered // 2)
    return tuple(sequence[:simulations])


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def _rescale(x: np.ndarray) -> np.ndarray:
    low = x.min()
    return (x - low) / max(float(x.max() - low), 1e-8)


class _Stats:
    """Completed Q-values and the improved policy at a decision node (its valid moves only)."""

    __slots__ = ("visits", "win", "moves", "sigma", "logits")

    def __init__(self, node: _Decision, mode: Mode, config: SearchConfig) -> None:
        actions = node.actions
        visits = np.zeros(len(actions))
        q_win = np.zeros(len(actions))
        q_moves = np.zeros(len(actions))
        for i, a in enumerate(actions):
            child = node.children.get(a)
            if child is not None and child.visits:
                visits[i] = child.visits
                q_win[i] = child.win_sum / child.visits
                q_moves[i] = 1.0 + _moves(child.moves_sum, child.win_sum, child.state)
        logits = node.logits[actions]
        visited = visits > 0
        total = visits.sum()
        mixed_win, mixed_moves = node.raw_win, node.raw_moves
        if total:
            # The mixed value of the paper: the raw value and the prior-weighted Q of the visited moves,
            # with the moves weighted by the win values like every backup.
            prior = _softmax(logits)
            weights = np.where(visited, np.maximum(prior, 1e-30), 0.0)
            weights /= weights.sum()
            win_mass = node.raw_win + total * float(weights @ q_win)
            mixed_win = win_mass / (total + 1)
            if win_mass > 0.0:
                moves_mass = node.raw_win * node.raw_moves + total * float((weights * q_win) @ q_moves)
                mixed_moves = moves_mass / win_mass
        self.win = np.where(visited, q_win, mixed_win)
        self.moves = np.where(visited, q_moves, mixed_moves)
        if mode is Mode.ENVELOPE:
            q = (
                _rescale(-self.moves)
                if config.moves_unit is None
                else (self.moves.min() - self.moves) / config.moves_unit
            )
        else:
            q = _lexicographic(self.win, self.moves, config.tie_weight)
        self.sigma = (config.c_visit + visits.max()) * config.c_scale * q
        self.visits = visits
        self.logits = logits - logits.max()

    def improved_policy(self) -> np.ndarray:
        return _softmax(self.logits + self.sigma)


def _lexicographic(win: np.ndarray, moves: np.ndarray, tie_weight: float) -> np.ndarray:
    """Win values rescaled to [0, 1], with fewer moves ordering only the moves tied on the best win."""
    best = win.max()
    tied = win == best
    if tied.all():
        return _rescale(-moves)
    order = np.zeros(len(win))
    if tied.sum() > 1:
        order[tied] = _rescale(-moves[tied])
    return (1.0 - tie_weight) * _rescale(win) + tie_weight * order


# ------------------------------------------------------------------- search
def _evaluate(node: _Node, rules: Rules) -> Generator[Query, LeafValue, None]:
    """Ask for the node's value (and prior); exact wins keep their win value of 1."""
    if node.evaluated:
        return
    afterstate = isinstance(node, _Chance)
    classes = node.classes if isinstance(node, _Decision) else None
    value = yield Query(node.state, rules, afterstate, classes)
    win = 1.0 if node.certain else min(max(value.win, 0.0), CERTAIN)
    state = node.state
    moves = min(max(value.moves, 0.0), moves_ceiling(state.num_cells, state.free_cells))
    moves *= moves_scale(state.free_cells)
    if isinstance(node, _Decision):
        node.logits = np.asarray(value.logits, dtype=np.float64)
    node.set_value(win, moves)


def _backup(path: list[_Node], win: float, moves: float) -> None:
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        node.visits += 1
        node.win_sum += win
        node.moves_sum += win * moves
        if i and isinstance(path[i - 1], _Decision):
            moves += 1.0  # the move from the parent into this node


def _root(
    state: GameState, rules: Rules, mode: Mode, history: frozenset, budget: int
) -> tuple[_Decision, bool]:
    """The root and whether it is lost; prunes like any node, then falls back so that a move is left."""
    root = _decision(state, rules, mode, history, budget)
    if root.leaf:  # cannot happen in free mode, which is only used when the state is not envelope-certain
        raise AssertionError("the root of a free search is an exact win")
    status = root.status
    lost = all(
        status.get(a, _Status.FATAL) in (_Status.FATAL, _Status.LOST, _Status.TIMEOUT) for a in ACTIONS
    )
    for allowed in ((_Status.OK,), (_Status.OK, _Status.REPEAT), tuple(_Status)[2:] + (_Status.OK,)):
        root.actions = [a for a in ACTIONS if status.get(a, _Status.FATAL) in allowed]
        if root.actions:
            break
    return root, lost


def search(
    state: GameState,
    rules: Rules = TRAINING_RULES,
    config: SearchConfig = SELF_PLAY,
    rng: np.random.Generator | None = None,
    history: frozenset = frozenset(),
) -> Generator[Query, LeafValue, SearchResult]:
    """Search a running ``state``; ``history`` holds the positions of the current fruit's epoch so far.

    Yields :class:`Query` objects and expects a :class:`LeafValue` for each.
    """
    if state.status is not Status.RUNNING:
        raise ValueError("the game is over")
    rng = np.random.default_rng() if rng is None else rng
    mode = Mode.ENVELOPE if envelope_certain(state, rules) else Mode.FREE
    budget = config.lost_budget if mode is Mode.FREE else 0
    root, lost = _root(state, rules, mode, history, budget)
    actions = root.actions
    valid = tuple(a in actions for a in ACTIONS)
    nan = np.full(3, np.nan)
    if not actions or (len(actions) == 1 and not config.search_forced):
        action = actions[0] if actions else RelativeAction.STRAIGHT
        return SearchResult(
            action, mode, valid, True, lost, 0, np.zeros(3), None, nan, nan, None, None, None, None
        )

    yield from _evaluate(root, rules)
    prior_win, prior_moves = root.raw_win, root.raw_moves
    root.visits, root.win_sum, root.moves_sum = 1, root.raw_win, root.raw_win * root.raw_moves
    gumbel = config.gumbel_scale * rng.gumbel(size=len(actions))
    schedule = considered_visits(len(actions), config.simulations)
    for simulation in range(config.simulations):
        stats = _Stats(root, mode, config)
        eligible = stats.visits == schedule[simulation]
        if not eligible.any():
            eligible = stats.visits == stats.visits.min()
        scores = np.where(eligible, gumbel + stats.logits + stats.sigma, -np.inf)
        action = actions[int(np.argmax(scores))]
        node: _Node = root
        path: list[_Node] = [root]
        while True:
            if isinstance(node, _Decision):
                child = node.children.get(action)
                if child is None:
                    child = node.children[action] = _child(node, action, rules, mode)
            else:  # an afterstate: draw the fruit
                assert isinstance(node, _Chance)
                fruit = node.free[int(rng.integers(len(node.free)))]
                child = node.children.get(fruit)
                if child is None:
                    after = dataclasses.replace(node.state, fruit=fruit)
                    child = _decision(after, rules, mode, frozenset())
                    if not child.leaf and not child.actions:
                        child = _Terminal(after, *LOSS)
                    node.children[fruit] = child
            path.append(child)
            if not child.evaluated:
                yield from _evaluate(child, rules)
                break
            if child.leaf:
                break
            node = child
            if isinstance(node, _Decision):
                action = _select(node, mode, config)
        _backup(path, child.raw_win, child.raw_moves)

    stats = _Stats(root, mode, config)
    most = stats.visits == stats.visits.max()
    action = actions[int(np.argmax(np.where(most, gumbel + stats.logits + stats.sigma, -np.inf)))]
    improved = stats.improved_policy()
    policy, visits, q_win, q_moves = np.zeros(3), np.zeros(3), nan.copy(), nan.copy()
    index = [int(a) for a in actions]
    policy[index], visits[index] = improved, stats.visits
    q_win[index], q_moves[index] = stats.win, stats.moves
    win_mass = improved * stats.win
    value_win = float(win_mass.sum())
    value_moves = float(win_mass @ stats.moves) / value_win if value_win > 0.0 else _ceiling(state)
    return SearchResult(
        action,
        mode,
        valid,
        False,
        lost,
        config.simulations,
        visits,
        policy,
        q_win,
        q_moves,
        value_win,
        value_moves,
        prior_win,
        prior_moves,
    )


def _select(node: _Decision, mode: Mode, config: SearchConfig) -> RelativeAction:
    """Gumbel MuZero's interior choice: the improved policy minus the visit share."""
    stats = _Stats(node, mode, config)
    score = stats.improved_policy() - stats.visits / (1.0 + stats.visits.sum())
    return node.actions[int(np.argmax(score))]


def run_search(
    evaluator: Evaluator,
    state: GameState,
    rules: Rules = TRAINING_RULES,
    config: SearchConfig = SELF_PLAY,
    rng: np.random.Generator | None = None,
    history: frozenset = frozenset(),
) -> SearchResult:
    """Run :func:`search` with one ``evaluator`` call per leaf."""
    generator = search(state, rules, config, rng, history)
    try:
        query = next(generator)
        while True:
            query = generator.send(evaluator([query]).row(0))
    except StopIteration as stop:
        return stop.value


# ------------------------------------------------------------------- agents
class _Epoch:
    """The positions since the last fruit, as the agent saw them."""

    def __init__(self) -> None:
        self.positions: set[Hashable] = set()
        self.length = -1

    def observe(self, state: GameState) -> frozenset:
        if state.steps_since_fruit == 0 or state.length != self.length:
            self.positions = set()
        self.length = state.length
        before = frozenset(self.positions)
        self.positions.add(position_key(state))
        return before


class SearchAgent:
    """The M1 agent with search: plays the move of :func:`search` (an :data:`~snake_agent.contract.Agent`).

    The search's results are kept in :attr:`last`. The agent remembers the
    positions since the last fruit, to prune moves back to them.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        rules: Rules = TRAINING_RULES,
        config: SearchConfig = EVALUATION,
        seed: int = 0,
    ) -> None:
        self.evaluator, self.rules, self.config = evaluator, rules, config
        self.rng = np.random.default_rng(seed)
        self.epoch = _Epoch()
        self.last: SearchResult | None = None

    def __call__(self, state: GameState) -> RelativeAction:
        history = self.epoch.observe(state)
        self.last = run_search(self.evaluator, state, self.rules, self.config, self.rng, history)
        return self.last.action


class PolicyAgent:
    """The M1 agent without search: the network's most likely move among those the search would consider.

    The same moves are pruned as at the root of :func:`search` (fatal moves,
    timeouts, repeated positions, proven-lost moves, and moves outside the
    envelope where it applies), so it differs from :class:`SearchAgent` only
    by the search.
    """

    def __init__(
        self, evaluator: Evaluator, rules: Rules = TRAINING_RULES, lost_budget: int = 10_000
    ) -> None:
        self.evaluator, self.rules, self.lost_budget = evaluator, rules, lost_budget
        self.epoch = _Epoch()

    def __call__(self, state: GameState) -> RelativeAction:
        generator = policy_move(state, self.rules, self.epoch.observe(state), self.lost_budget)
        try:
            query = next(generator)
            generator.send(self.evaluator([query]).row(0))
        except StopIteration as stop:
            return stop.value
        raise AssertionError("policy_move asks for one evaluation at most")


def policy_move(
    state: GameState,
    rules: Rules = TRAINING_RULES,
    history: frozenset = frozenset(),
    lost_budget: int = 10_000,
) -> Generator[Query, LeafValue, RelativeAction]:
    """:class:`PolicyAgent`'s move as a generator, like :func:`search`: it yields the state's query if more
    than one move is left, and returns the most likely of those moves."""
    mode = Mode.ENVELOPE if envelope_certain(state, rules) else Mode.FREE
    root, _ = _root(state, rules, mode, history, lost_budget if mode is Mode.FREE else 0)
    if len(root.actions) <= 1:
        return root.actions[0] if root.actions else RelativeAction.STRAIGHT
    logits = (yield Query(state, rules, False, root.classes)).logits
    return max(root.actions, key=lambda a: logits[a])
