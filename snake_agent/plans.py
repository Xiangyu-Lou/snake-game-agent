"""Plans: free play that decides once per fruit, judging each exact path to the fruit by the body it leaves.

The search of :mod:`snake_agent.search` decides one move at a time and sees a
few moves ahead, so on a large board it cannot see the fruit, and its values
are too coarse to tell a move toward the fruit from one away from it. Inside
the cycle-order envelope that did not matter: every move kept the win certain,
and the cycle set the pace. Free play has no such pace, and this agent decides
at another grain:

* **Plans.** When a fruit appears, the agent lists paths to it
  (:func:`candidates`): shortest paths with their ties broken at random,
  sweeping paths (shortest paths lengthened through the free cells beside them,
  which lay the body down compactly), and paths that first follow the survival
  certificate's loop for a few moves (a way to wait for the body to clear).
  Paths go through free cells and through body cells already vacated when the
  head arrives, so every plan is safe move by move, and its length is exact.
* **Afterstates.** A plan ends by eating the fruit, at its *afterstate*: the
  longer body, before the next fruit appears. Nothing random happens on the
  way, so a plan's whole future is the afterstate's, and the network values
  afterstates only: the probability of filling the board from there, and the
  expected moves until it is full if it is filled.
* **Waiting** (:func:`waits`). The agent may also put off eating: each move
  that neither collides, eats nor times out is an option too, valued by the
  network as the state it leads to, the fruit still in place. After a waiting
  move the agent decides again.
* **Choice** (:func:`choose`). A plan is played unless waiting is safer by more
  than ``tolerance``: among the plans whose risk (one minus the win value) is
  within the tolerance of the least risky plan's, the agent takes the fewest
  moves in all: the plan's own, the trip to the next fruit as far as it is
  known exactly (:func:`next_trip`), and the rest of the afterstate's value,
  weighted by how far the network's precision can be trusted (:func:`trust`).
  The tolerance is the price of speed: for this agent the owner accepts a
  slightly lower clear rate for far fewer moves (``snake_agent/README.md``, free
  play). It also keeps the noise of the win head from deciding between equally
  safe plans.
* **The guard.** Exact code decides survival where it can (review §2.3): while
  the fruit timeout leaves room to wait, only the options that keep a survival
  certificate are compared, if there are any. After a certified plan the snake
  can circle its own body, whatever comes next.

* **The certified finish** (``config.certify``). Free play leaves holes
  between the blocks it lays down, and late fruits in holes cost about a hundred
  moves each on 20x20. So from a share of the board filled on, by the owner's
  later choice, the agent heads for a body that some Hamiltonian cycle holds in
  order, the walk around a spanning tree of 2x2 blocks
  (:mod:`snake_agent.certify`), and then keeps such a body to the end. It
  follows no fixed cycle: at each fruit it takes the fastest path whose
  afterstate some such cycle holds, so the win stays certain and the network is
  not asked.

The network does not see the canonical cycle's input planes
(:attr:`~snake_agent.network.NetworkConfig.cycle`), and no move is restricted to
the canonical cycle's envelope. It sees the arrival planes instead
(:attr:`~snake_agent.network.NetworkConfig.arrival`). Self-play (:func:`play`)
trains the values by temporal differences over fruits, as the search's moves
head is trained.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Generator, Sequence
from dataclasses import dataclass

import numpy as np

from snake_game import Direction, GameState, RelativeAction, Status

from .certify import certified_paths, certify, transition_paths
from .contract import TRAINING_RULES, Outcome, Rules, to_relative
from .encoding import encode, moves_ceiling, moves_scale
from .search import Evaluator, Prediction, Query
from .selfplay import Examples, GameSummary, Label, Start, _frame_width, _to_examples, moves_returns
from .survival import Board, arrival_times, opening_times


@dataclass(frozen=True)
class PlanConfig:
    """How the plan agent lists and chooses its plans."""

    paths: int = 16
    """Shortest paths drawn at random per decision (fewer if the fruit has fewer)."""
    waits: tuple[int, ...] = (2, 4, 8, 16, 32)
    """Moves along the certificate's loop before a shortest path, for the waiting plans."""
    sweeps: tuple[float, ...] = (0.25, 1.0)
    """Sweeping plans: shortest paths lengthened through free cells (:func:`extend`), each by up to this
    share of the free cells."""
    tolerance: float = 0.02
    """Plans whose risk exceeds the least risky plan's by at most this are acceptable."""
    moves_trust: float = 1000.0
    """How far the network's moves values are trusted when options are compared: their differences
    count ``1 / (1 + (moves_scale(F) / moves_trust)^2)`` of their face value, ``F`` the free cells
    (:func:`trust`). The moves head errs by a share of that scale, so early on a large board its
    differences are noise next to the exact lengths of the plans."""
    explore: float = 0.0
    """Self-play: the chance of playing a random acceptable plan instead of the fastest."""
    next_trip: bool = True
    """Count the trip to the next fruit exactly (:func:`next_trip`), whatever the trust in the network."""
    guard: bool = True
    """Only plans whose afterstate has a survival certificate are compared, if there are any; if there
    are none, the agent waits along the certificate's loop while the fruit timeout leaves room, and plans
    again at the next move."""
    certify: float | None = 0.3
    """From this share of the board filled, on a board with even sides, the agent finishes on a
    Hamiltonian-cycle certificate (:mod:`snake_agent.certify`): it heads for a body that the walk around a
    spanning tree of 2x2 blocks holds in order, then keeps such a body to the end, taking the fastest
    path to each fruit that keeps it. None plays free to the end."""
    trees: int = 64
    """Random spanning trees tried per decision once the certificate is sought."""
    shortcuts: int = 8
    """Certified shortcuts kept per decision (:func:`~snake_agent.certify.shortcuts`)."""
    shortcut_budget: int = 20_000
    """Steps of search for the certified shortcuts per decision."""


SELF_PLAY = PlanConfig(explore=0.1)
EVALUATION = PlanConfig()


# ---------------------------------------------------------------- paths
def shortest_path(board: Board, arrival: Sequence[int], goal: int, rng: random.Random) -> list[int]:
    """A shortest path from the head to ``goal`` (reached in ``arrival``): the cells entered, in order.

    Ties between predecessors are broken at random, so repeated calls draw
    different paths.
    """
    neighbours = board.neighbours
    path = [goal]
    cell = goal
    while arrival[cell] > 1:
        t = arrival[cell] - 1
        options = [n for n in neighbours[cell] if n >= 0 and arrival[n] == t]
        cell = options[0] if len(options) == 1 else rng.choice(options)
        path.append(cell)
    path.reverse()
    return path


def extend(board: Board, path: list[int], fruit: int, extra: int, rng: random.Random) -> list[int]:
    """``path`` lengthened by up to ``extra`` cells of detours through free cells, two at a time.

    A step ``u -> v`` becomes ``u -> u' -> v' -> v``, ``u'`` and ``v'`` being the free cells beside ``u``
    and ``v`` on the same side, off the path and not the fruit (the classic way to lengthen a path on a
    grid). The detours sweep the free cells next to the path, so a fully lengthened path lays the body
    down compactly. Free cells stay free until the head enters them, so the path stays safe, and its body
    cells are entered no sooner than before.
    """
    neighbours, head = board.neighbours, board.head
    width = board.width
    free_before = board.t - board.length
    entered = board.entered
    full = [head] + path
    on = set(full)
    reverse = (board.heading + 2) % 4
    added, changed = 0, True
    while changed and added + 2 <= extra:
        changed = False
        i = 0
        while i < len(full) - 1 and added + 2 <= extra:
            u, v = full[i], full[i + 1]
            d = _direction(width, u, v)
            sides = [(d + 1) % 4, (d + 3) % 4]
            rng.shuffle(sides)
            for side in sides:
                if u == head and side == reverse:
                    continue
                a, b = neighbours[u][side], neighbours[v][side]
                if (
                    a >= 0
                    and b >= 0
                    and a not in on
                    and b not in on
                    and a != fruit
                    and b != fruit
                    and entered[a] <= free_before
                    and entered[b] <= free_before
                ):
                    full[i + 1 : i + 1] = [a, b]
                    on.update((a, b))
                    added += 2
                    changed = True
                    break
            i += 1
    return full[1:]


def _direction(width: int, cell: int, nxt: int) -> int:
    """The absolute direction from ``cell`` to the neighbouring ``nxt``."""
    if nxt == cell - width:
        return 0
    if nxt == cell + 1:
        return 1
    if nxt == cell + width:
        return 2
    return 3


def afterstate(state: GameState, path: Sequence[int]) -> GameState:
    """The state after entering the cells of ``path`` (flat), the last of which holds the fruit.

    The body is the ``L + 1`` cells entered last; the fruit is not placed yet,
    unless the board is full.
    """
    width = state.width
    d = len(path)
    entered = [(c % width, c // width) for c in reversed(path)]
    length = state.length + 1
    body = tuple(entered[:length]) if d >= length else tuple(entered) + state.snake[: length - d]
    before = path[-2] if d > 1 else state.head[1] * width + state.head[0]
    full = length == state.num_cells
    return GameState(
        width,
        state.height,
        body,
        Direction(_direction(width, before, path[-1])),
        None,
        state.score + 1,
        state.steps + d,
        0,
        Status.WON if full else Status.RUNNING,
    )


@dataclass(frozen=True)
class Plan:
    """A path to the fruit and where it leads."""

    cells: tuple[int, ...]
    """Flat cells entered, in order; the last holds the fruit."""
    after: GameState
    """The afterstate: the board after eating, before the next fruit appears."""
    wait: int = 0
    """Moves along the certificate's loop before the shortest path."""

    @property
    def moves(self) -> int:
        return len(self.cells)


def candidates(
    state: GameState, rules: Rules = TRAINING_RULES, config: PlanConfig = EVALUATION, rng=None
) -> list[Plan]:
    """The plans to choose from in ``state``, one per distinct afterstate (empty if none reaches the fruit).

    Shortest paths (``config.paths`` draws), sweeping plans (shortest paths
    lengthened by :func:`extend`, one per ``config.sweeps``), then for each
    ``config.waits`` the certificate's loop for that many moves followed by a
    shortest path. A plan must eat before the fruit timeout and must not pass
    over the fruit before its end.
    """
    if state.done or state.fruit is None:
        raise ValueError("plans need a running game with a visible fruit")
    rng = random.Random(0) if rng is None else rng
    width = state.width
    fruit = state.fruit[1] * width + state.fruit[0]
    budget = rules.moves_left(state)
    board = Board(state)
    plans: dict[tuple, Plan] = {}

    def add(cells: list[int], wait: int) -> None:
        if budget is not None and len(cells) > budget:
            return
        after = afterstate(state, cells)
        if after.snake not in plans:
            plans[after.snake] = Plan(tuple(cells), after, wait)

    arrival = arrival_times(board)
    if arrival[fruit] > 0:
        shortest = [shortest_path(board, arrival, fruit, rng) for _ in range(config.paths)]
        for path in shortest:
            add(path, 0)
        free = state.free_cells
        for k, share in enumerate(config.sweeps):
            extra = int(share * free)
            if budget is not None:
                extra = min(extra, budget - arrival[fruit])
            if extra >= 2:
                add(extend(board, list(shortest[k % len(shortest)]), fruit, extra, rng), 0)
    loop = board.certificate_loop() if config.waits else None
    if loop:
        if fruit in loop:
            loop = loop[: loop.index(fruit) + 1]  # following the loop eats the fruit on the way
            if loop[-1] == fruit:
                add(list(loop), len(loop))
                loop = loop[:-1]
        for wait in config.waits:
            if wait > len(loop) or (budget is not None and wait >= budget):
                break
            undo = []
            for cell in loop[:wait]:
                undo.append((cell, board.move(_direction(width, board.head, cell), cell)))
            later = arrival_times(board)
            if later[fruit] > 0:
                add(list(loop[:wait]) + shortest_path(board, later, fruit, rng), wait)
            for cell, record in reversed(undo):
                board.unmove(cell, record)
    return list(plans.values())


@dataclass(frozen=True)
class Wait:
    """A move that neither collides, eats nor times out: a way to put off eating."""

    action: RelativeAction
    state: GameState
    """The state it leads to, the fruit still in place."""
    certified: bool
    """Whether that state has a survival certificate."""


def waits(state: GameState, rules: Rules = TRAINING_RULES) -> list[Wait]:
    """The waiting moves of ``state``, in ``LEFT``, ``STRAIGHT``, ``RIGHT`` order."""
    board = Board(state)
    width = state.width
    fruit = -1 if state.fruit is None else state.fruit[1] * width + state.fruit[0]
    heading = int(state.direction)
    options = []
    for action in RelativeAction:
        d = (heading + action - 1) % 4
        cell = board.target(d)
        if cell < 0 or cell == fruit:
            continue
        after = GameState(
            width,
            state.height,
            ((cell % width, cell // width),) + state.snake[:-1],
            Direction(d),
            state.fruit,
            state.score,
            state.steps + 1,
            state.steps_since_fruit + 1,
        )
        if rules.timed_out(after):
            continue
        undo = board.move(d, cell)
        options.append(Wait(action, after, board.has_survival_certificate()))
        board.unmove(cell, undo)
    return options


def certified(plan: Plan) -> bool:
    """Whether the plan's afterstate fills the board or has a survival certificate."""
    return plan.after.won or Board(plan.after).has_survival_certificate()


def _comfortable(state: GameState, rules: Rules, plans: Sequence[Plan]) -> bool:
    """Whether the fruit timeout leaves room to wait: after another move, room for a plan twice as long
    as the shortest one now."""
    left = rules.moves_left(state)
    return left is None or not plans or left - 1 >= 2 * min(p.moves for p in plans)


# --------------------------------------------------------------- choice
def _values(states: Sequence[GameState], prediction: Prediction | None, offset: int) -> tuple[list, list]:
    """Win values and expected moves (in moves) of ``states``, read from ``prediction`` from row
    ``offset`` on, skipping filled boards (exact wins, no moves left)."""
    wins, moves = [], []
    row = offset
    for state in states:
        if state.won:
            wins.append(1.0)
            moves.append(0.0)
            continue
        wins.append(min(max(float(prediction.win[row]), 0.0), 1.0))
        units = min(max(float(prediction.moves[row]), 0.0), moves_ceiling(state.num_cells, state.free_cells))
        moves.append(units * moves_scale(state.free_cells))
        row += 1
    return wins, moves


def trust(free: int, config: PlanConfig = EVALUATION) -> float:
    """The weight of the network's moves values in a comparison of options, with ``free`` cells free.

    The moves head's errors grow with :func:`~snake_agent.encoding.moves_scale` (about a hundredth of
    it: ``snake_agent/README.md``, M1), while two plans' futures differ by tens of moves. Shrinking the
    head's differences by the share of their spread that such differences explain keeps its noise from
    choosing between plans early on a large board (39,700 moves to go on 20x20, a weight of 0.0006),
    and leaves its judgement nearly whole late in the game (15 cells free, a weight of 0.997).
    """
    return 1.0 / (1.0 + (moves_scale(free) / config.moves_trust) ** 2)


def next_trip(state: GameState) -> float:
    """The moves to the next fruit, as far as they are known exactly: when the head can first reach the
    visible fruit, or for an afterstate the mean over the free cells where the next fruit may appear. A
    cell the body walls in counts from when the body beside it moves away
    (:func:`~snake_agent.survival.opening_times`)."""
    times = opening_times(Board(state))
    width = state.width
    if state.fruit is not None:
        return float(times[state.fruit[1] * width + state.fruit[0]])
    occupied = set(state.snake)
    free = [times[y * width + x] for y in range(state.height) for x in range(width) if (x, y) not in occupied]
    return float(sum(free)) / len(free) if free else 0.0


def _fastest(wins: np.ndarray, moves: np.ndarray, config: PlanConfig, rng: random.Random | None) -> int:
    """The fewest moves among the options whose risk is within the tolerance of the least (or, with
    ``config.explore``, a random one of them with that probability)."""
    risk = 1.0 - wins
    accepted = np.flatnonzero(risk <= risk.min() + config.tolerance)
    if config.explore and rng is not None and len(accepted) > 1 and rng.random() < config.explore:
        return int(accepted[rng.randrange(len(accepted))])
    return int(accepted[np.argmin(moves[accepted])])


def choose(
    plan_wins: Sequence[float],
    plan_moves: Sequence[float],
    wait_wins: Sequence[float] = (),
    wait_moves: Sequence[float] = (),
    config: PlanConfig = EVALUATION,
    rng: random.Random | None = None,
) -> tuple[bool, int]:
    """Whether to play a plan, and which plan or waiting move.

    A plan is played unless waiting is safer by more than the tolerance: the fastest of the plans whose
    risk is within the tolerance of the least risky plan's. Otherwise the same rule picks the waiting
    move. With ``config.explore`` and an ``rng``, a random acceptable option instead, with that
    probability. Moves are the expected moves until the board is full, the option's own included.
    """
    best_wait = max(wait_wins, default=-1.0)
    if len(plan_wins) and max(plan_wins) >= best_wait - config.tolerance:
        return True, _fastest(np.asarray(plan_wins), np.asarray(plan_moves), config, rng)
    return False, _fastest(np.asarray(wait_wins), np.asarray(wait_moves), config, rng)


@dataclass(frozen=True)
class Decision:
    """What the agent decided in a state: a plan to follow, or a move to wait with."""

    plan: Plan | None
    wait: RelativeAction | None = None
    win: float | None = None
    """The chosen option's win value (exact 1 for a filled board); None if the move was forced."""
    moves: float | None = None
    """The chosen option's expected moves until the board is full, its own moves included."""
    options: int = 0
    """Options compared: plans and waiting moves."""
    certain: bool = False
    """Whether the plan's afterstate is certified by a Hamiltonian cycle (:mod:`snake_agent.certify`), a
    certain win; ``moves`` then counts the plan and the next trip only."""


def seeks_certificate(state: GameState, config: PlanConfig = EVALUATION) -> bool:
    """Whether the agent seeks or keeps the certificate in ``state`` (``config.certify``)."""
    return (
        config.certify is not None
        and state.width % 2 == 0
        and state.height % 2 == 0
        and state.length >= config.certify * state.num_cells
    )


def _plans(state: GameState, paths: Sequence[Sequence[int]], rules: Rules) -> list[Plan]:
    """``paths`` as plans, one per afterstate, without those the fruit timeout cuts short."""
    budget = rules.moves_left(state)
    plans: dict[tuple, Plan] = {}
    for path in paths:
        if budget is None or len(path) <= budget:
            after = afterstate(state, path)
            plans.setdefault(after.snake, Plan(tuple(path), after))
    return list(plans.values())


def _certain(state: GameState, plans: Sequence[Plan]) -> Decision:
    """The fastest of ``plans``, whose afterstates are certified: the fewest moves with the next trip."""
    totals = [p.moves + (0.0 if p.after.won else next_trip(p.after)) for p in plans]
    i = min(range(len(plans)), key=totals.__getitem__)
    return Decision(plans[i], None, 1.0, totals[i], len(plans), True)


def decide(
    state: GameState,
    rules: Rules = TRAINING_RULES,
    config: PlanConfig = EVALUATION,
    rng: random.Random | None = None,
) -> Generator[list[Query], Prediction, Decision]:
    """The agent's decision in ``state``, as a generator: it yields the queries of its options (the plans'
    afterstates and the states after the waiting moves) and expects their
    :class:`~snake_agent.search.Prediction` back.

    With ``config.guard``, while the fruit timeout leaves room to wait, only the options that keep a
    survival certificate are compared, if there are any.

    From ``config.certify`` on (:func:`seeks_certificate`), a certified body keeps its certificate: the
    fastest certified path is played, with no evaluation (:func:`~snake_agent.certify.certified_paths`).
    Before that, a path whose afterstate is certified is played if there is one; otherwise the paths that
    lengthen the body's certifiable run replace the free plans, if any of them keeps a survival
    certificate (:func:`~snake_agent.certify.transition_paths`).
    """
    rng = random.Random(0) if rng is None else rng
    plans = None
    if seeks_certificate(state, config):
        lay = certify(state)
        if lay is not None:
            kept = _plans(
                state,
                certified_paths(state, lay, rng, config.trees, config.shortcuts, config.shortcut_budget),
                rules,
            )
            if kept:
                return _certain(state, kept)
        else:
            exact, partial = transition_paths(state, rng, config.trees, config.paths)
            exact_plans = _plans(state, exact, rules)
            if exact_plans:
                return _certain(state, exact_plans)
            plans = [p for p in _plans(state, partial, rules) if certified(p)] or None
    if plans is None:
        plans = candidates(state, rules, config, rng)
    holds = waits(state, rules)
    if config.guard and _comfortable(state, rules, plans):
        safe_plans = [p for p in plans if certified(p)]
        safe_holds = [w for w in holds if w.certified]
        if safe_plans or safe_holds:
            plans, holds = safe_plans, safe_holds
    if not plans and len(holds) <= 1:  # forced, or every move collides or times out
        return Decision(None, holds[0].action if holds else RelativeAction.STRAIGHT)
    afters = [p.after for p in plans]
    queries = [Query(a, rules, afterstate=True) for a in afters if not a.won]
    queries += [Query(w.state, rules) for w in holds]
    prediction = (yield queries) if queries else None
    plan_wins, after_moves = _values(afters, prediction, 0)
    plan_moves = [p.moves + m for p, m in zip(plans, after_moves, strict=True)]
    wait_wins, state_moves = _values([w.state for w in holds], prediction, len(queries) - len(holds))
    wait_moves = [1.0 + m for m in state_moves]
    weight = trust(state.free_cells, config)
    plan_near = [0.0 if a.won or not config.next_trip else next_trip(a) for a in afters]
    wait_near = [next_trip(w.state) if config.next_trip else 0.0 for w in holds]
    is_plan, i = choose(
        plan_wins,
        [p.moves + n + weight * (m - n) for p, n, m in zip(plans, plan_near, after_moves, strict=True)],
        wait_wins,
        [1.0 + n + weight * (m - n) for n, m in zip(wait_near, state_moves, strict=True)],
        config,
        rng,
    )
    options = len(plans) + len(holds)
    if is_plan:
        return Decision(plans[i], None, plan_wins[i], plan_moves[i], options)
    return Decision(None, holds[i].action, wait_wins[i], wait_moves[i], options)


def run_decision(
    evaluator: Evaluator,
    state: GameState,
    rules: Rules = TRAINING_RULES,
    config: PlanConfig = EVALUATION,
    rng: random.Random | None = None,
) -> Decision:
    """Run :func:`decide` with one evaluator call."""
    generator = decide(state, rules, config, rng)
    try:
        queries = next(generator)
        generator.send(evaluator(queries))
    except StopIteration as stop:
        return stop.value
    raise AssertionError("decide asks for one evaluation at most")


class Planner:
    """Follows a decision's plan move by move: the moves left, and the next move as a relative action."""

    def __init__(self) -> None:
        self.cells: list[int] = []

    def follow(self, plan: Plan) -> None:
        self.cells = list(plan.cells)

    def next_action(self, state: GameState) -> RelativeAction | None:
        """The plan's next move from ``state``, or None if there is no plan left."""
        if not self.cells:
            return None
        width = state.width
        head = state.head[1] * width + state.head[0]
        cell = self.cells.pop(0)
        return to_relative(state.direction, Direction(_direction(width, head, cell)))


class PlanAgent:
    """The plan agent for the harness (an :data:`~snake_agent.contract.Agent`): it decides when a fruit
    appears or its plan runs out, then follows its plan."""

    def __init__(
        self,
        evaluator: Evaluator,
        rules: Rules = TRAINING_RULES,
        config: PlanConfig = EVALUATION,
        seed: int = 0,
    ) -> None:
        self.evaluator, self.rules, self.config = evaluator, rules, config
        self.rng = random.Random(seed)
        self.planner = Planner()
        self.last: Decision | None = None

    def __call__(self, state: GameState) -> RelativeAction:
        action = self.planner.next_action(state)
        if action is not None:
            return action
        self.last = run_decision(self.evaluator, state, self.rules, self.config, self.rng)
        if self.last.plan is None:
            return self.last.wait
        self.planner.follow(self.last.plan)
        return self.planner.next_action(state)


# ------------------------------------------------------------- self-play
class _Game:
    """One self-play game of the plan agent: the live game, the pending decision and the record."""

    def __init__(self, start: Start, rules: Rules, config: PlanConfig, rng: random.Random) -> None:
        self.start, self.rules, self.config, self.rng = start, rules, config, rng
        self.game = start.game()
        self.start_length = self.game.length
        self.planner = Planner()
        self.deciding: Generator[list[Query], Prediction, Decision] | None = None
        self.prediction: Prediction | None = None
        self.decided: list[tuple[int, GameState, float]] = []
        """Decisions that compared options: the move they were made before, the state and the chosen
        option's win value."""
        self.eats: list[int] = []
        """Moves (from the agent's first) that ate a fruit without filling the board."""
        self.afters: list[GameState] = []
        self.after_moves: list[float] = []
        """The moves value (in moves) the network gave each afterstate when its plan was chosen."""
        self.after_certain: list[bool] = []
        """Whether each afterstate was certified by a Hamiltonian cycle (:attr:`Decision.certain`)."""
        self.moves = 0
        self.leaves = 0
        self.states: list[GameState] = []
        self.outcome: Outcome | None = None
        self.pending: Decision | None = None

    def advance(self) -> list[Query] | None:
        """Play until a decision needs the network (its queries are returned) or the game ends (None)."""
        while self.outcome is None:
            state = self.game.state()
            if self.deciding is None:
                action = self.planner.next_action(state)
                if action is not None:
                    self._play(state, action)
                    continue
                self.deciding = decide(state, self.rules, self.config, self.rng)
                self.prediction = None
            try:
                queries = self.deciding.send(self.prediction)
                self.leaves += len(queries)
                return queries
            except StopIteration as stop:
                self.deciding = None
                decision: Decision = stop.value
                if decision.win is not None and not decision.certain:
                    self.decided.append((self.moves, state, decision.win))
                if decision.plan is None:
                    self._play(state, decision.wait)
                    continue
                self.pending = decision
                self.planner.follow(decision.plan)
        return None

    def _play(self, state: GameState, action: RelativeAction) -> None:
        self.states.append(state)
        step = self.game.step_relative(action)
        self.moves += 1
        if step.ate_fruit and not step.won:
            after = dataclasses.replace(self.game.state(), fruit=None)
            self.eats.append(self.moves - 1)
            self.afters.append(after)
            decision = self.pending
            known = (
                decision is not None and decision.plan.after.snake == after.snake and decision.win is not None
            )
            self.after_moves.append(decision.moves - decision.plan.moves if known else 0.0)
            self.after_certain.append(known and decision.certain)
            self.pending = None
        if self.game.won:
            self.outcome = Outcome.WON
        elif self.game.done:
            self.outcome = Outcome.LOST
        elif self.rules.timed_out(self.game.state()):
            self.outcome = Outcome.TIMED_OUT

    def summary(self, samples: int, rng: np.random.Generator) -> GameSummary:
        chosen = rng.choice(len(self.states), size=min(samples, len(self.states)), replace=False)
        return GameSummary(
            self.start.kind,
            self.start.seed,
            self.outcome,
            self.moves,
            self.game.length - self.start_length,
            self.game.length,
            self.start_length,
            len(self.decided),
            0,
            (self.game.width, self.game.height),
            self.leaves,
            tuple(self.states[i] for i in sorted(chosen)),
        )

    def examples(
        self, beta: float, moves_lambda: float, size: int, cycle: bool, arrival: bool
    ) -> list[tuple]:
        """The game's decision states and afterstates as example rows (see
        :func:`snake_agent.selfplay._to_examples`).

        Win targets blend the outcome ``z`` with a value, ``(1 - beta) z + beta w``: for a decision state
        the chosen option's, for an afterstate the one chosen at the next decision (the outcome alone if
        there was none). Moves targets, won games only: TD(lambda) returns over the fruits
        (:func:`~snake_agent.selfplay.moves_returns`) with the network's values of the afterstates.
        Certified decisions and afterstates (:attr:`Decision.certain`) are no examples, since the agent
        never asks the network about them; the moves the game played from such an afterstate stand for
        its value in the returns of the ones before."""
        won = self.outcome is Outcome.WON
        z = float(won)
        moves = [0.0] * self.moves
        afters = [0.0] * len(self.afters)
        if won:
            values = [
                self.moves - 1 - eat if certain else value
                for eat, value, certain in zip(self.eats, self.after_moves, self.after_certain, strict=True)
            ]
            moves, afters = moves_returns(self.moves, self.eats, values, moves_lambda)
        rows = []
        for t, state, win in self.decided:
            rows.append(
                (
                    encode(state, self.rules, size, cycle=cycle, arrival=arrival),
                    _frame_width(state),
                    (False, False, False),
                    np.zeros(3),
                    False,
                    (1.0 - beta) * z + beta * win,
                    moves[t] / moves_scale(state.free_cells) if won else 0.0,
                    won,
                    Label.BLEND,
                    state.free_cells,
                )
            )
        following = {t: win for t, _, win in self.decided}
        for k, after in enumerate(self.afters):
            if self.after_certain[k]:
                continue
            win = following.get(self.eats[k] + 1)
            rows.append(
                (
                    encode(after, self.rules, size, afterstate=True, cycle=cycle, arrival=arrival),
                    _frame_width(after),
                    (False, False, False),
                    np.zeros(3),
                    False,
                    z if win is None else (1.0 - beta) * z + beta * win,
                    afters[k] / moves_scale(after.free_cells) if won else 0.0,
                    won,
                    Label.AFTERSTATE,
                    after.free_cells,
                )
            )
        return rows


def play(
    evaluator: Evaluator,
    starts: Sequence[Start],
    rules: Rules = TRAINING_RULES,
    config: PlanConfig = SELF_PLAY,
    *,
    seed: int | Sequence[int] = 0,
    beta: float = 0.5,
    moves_lambda: float = 0.5,
    size: int | None = None,
    samples: int = 0,
    cycle: bool = False,
    arrival: bool = False,
) -> tuple[list[GameSummary], Examples | None]:
    """Play a game of the plan agent from each start, all at once, and return their summaries and examples.

    Each round, every game plays until its next decision, and the options of all their decisions are
    evaluated in one batch. ``seed`` drives the agents' random choices, never the fruits. The examples are
    the states where the games decided and the afterstates they reached (:meth:`_Game.examples`), encoded
    on a canvas of side ``size`` with or without the canonical cycle's inputs and the arrival planes
    (``cycle``, ``arrival``, as the network expects). Each summary keeps ``samples`` of its game's states
    for mid-game starts.
    """
    if not rules.fruit_timeout:
        raise ValueError("self-play needs the fruit timeout, which turns loops into losses")
    if not starts:
        raise ValueError("no starts")
    rng = np.random.default_rng(seed)
    games = [_Game(s, rules, config, random.Random(int(rng.integers(1 << 62)))) for s in starts]
    waiting = games
    while waiting:
        asking, queries, counts = [], [], []
        for game in waiting:
            batch = game.advance()
            if batch is not None:
                asking.append(game)
                queries += batch
                counts.append(len(batch))
        if queries:
            prediction = evaluator(queries)
            offset = 0
            for game, count in zip(asking, counts, strict=True):
                game.prediction = prediction.part(offset, offset + count)
                offset += count
        waiting = asking
    size = size or max(max(s.state.width, s.state.height) for s in starts)
    rows = [row for game in games for row in game.examples(beta, moves_lambda, size, cycle, arrival)]
    summaries = [game.summary(samples, rng) for game in games]
    return summaries, _to_examples(rows) if rows else None
