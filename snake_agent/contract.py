"""The information contract between the agent and the game (milestone M0).

These rules fix what the agent sees, what it controls and what it is judged on.
Training, search and evaluation all follow them; ``snake_agent/README.md`` gives
the reasons.

1. **Public state only.** The agent sees a :class:`~snake_game.GameState` (board
   size, ordered body, heading, visible fruit, move counters) and the
   :class:`Rules` in force. It never sees the live :class:`~snake_game.SnakeGame`,
   whose clones carry the hidden random stream, nor seeds or future fruits.
   Simulators for planning come from :func:`planning_game`, which draws future
   fruits from an independent seed.
2. **Relative actions.** One :class:`~snake_game.RelativeAction` per move: turn
   left, go straight or turn right. There is no reverse action.
3. **Safety strictly first.** Maximize the probability of filling the board.
   Fewer moves only decide between equally safe moves (:func:`choose_safely`).
   On standard starts the agent stays inside the cycle-order envelope
   (:class:`snake_agent.cycles.CycleEnvelope`): every move there keeps the win
   certain, so the moves are all equally safe and speed decides between them.
   The plan agent (:mod:`snake_agent.plans`) follows later decisions of the
   owner: it plays free on standard starts for far fewer moves, accepting a
   slightly lower clear rate, and finishes on a Hamiltonian-cycle certificate
   that it rebuilds at every fruit (:mod:`snake_agent.certify`), not on the
   canonical cycle's envelope.
4. **Fruit timeout.** Under the training rules a game is lost after ``N + L``
   consecutive moves without fruit, ``N`` being the number of cells and ``L``
   the snake's current length (:class:`Rules`). Evaluations also run without
   the limit, as a check.
5. **Outcomes.** A game ends :attr:`Outcome.WON`, :attr:`~Outcome.LOST` (a
   collision), :attr:`~Outcome.TIMED_OUT` (the fruit timeout, also a loss) or
   :attr:`~Outcome.CENSORED` (stopped unresolved by the evaluation).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from snake_game import Direction, GameState, RelativeAction, SnakeGame, StepResult

Agent = Callable[[GameState], RelativeAction | Direction]
"""An agent maps the current state to its move.

Agents built on this contract return a :class:`~snake_game.RelativeAction`. The
game's reference policies return an absolute :class:`~snake_game.Direction`,
which is accepted too; :func:`to_relative` gives the equivalent relative action.
"""


class Outcome(str, Enum):
    """How a game ended."""

    WON = "won"
    LOST = "lost"  # collision with a wall or the body
    TIMED_OUT = "timed_out"  # N + L consecutive moves without fruit: a loss
    CENSORED = "censored"  # stopped by the evaluation (loop, stall or move cap): neither won nor lost

    @property
    def resolved(self) -> bool:
        """Whether the game was decided (won or lost); censored games were not."""
        return self is not Outcome.CENSORED


@dataclass(frozen=True)
class Rules:
    """Rules on top of the engine, for training and evaluation.

    The engine ends a game only by a collision or by filling the board. With
    ``fruit_timeout`` (the training rules) a game is also lost after ``N + L``
    consecutive moves without fruit, ``L`` being the current length. The
    allowance grows from about ``N`` to ``2N`` with the snake, and it never binds
    for a cycle strategy, which eats within ``N`` moves. It turns loops into
    resolved losses. Without it, evaluations stop endless games by other means
    and report them censored.
    """

    fruit_timeout: bool = True

    def limit(self, cells: int, length: int) -> int | None:
        """Moves allowed without fruit on ``cells`` cells at ``length``; ``None`` without the rule."""
        return cells + length if self.fruit_timeout else None

    def timeout(self, state: GameState) -> int | None:
        """Moves allowed without fruit at the length of ``state``; ``None`` without the rule."""
        return self.limit(state.num_cells, state.length)

    def moves_left(self, state: GameState) -> int | None:
        """Moves before the timeout ends the game (at least 1 while it runs); ``None`` without the rule."""
        limit = self.timeout(state)
        return None if limit is None else limit - state.steps_since_fruit

    def timed_out(self, state: GameState) -> bool:
        """Whether the rule ends ``state``, a game the engine still considers running."""
        limit = self.timeout(state)
        return limit is not None and not state.done and state.steps_since_fruit >= limit

    def budget_features(self, state: GameState) -> tuple[float, float, float]:
        """Remaining-budget inputs (global features 7-9 of the input specification).

        Enabled (1), configured ``(N + L) / N`` and remaining
        ``(N + L - moves since fruit) / N``; all zero without the rule.
        """
        limit = self.timeout(state)
        if limit is None:
            return (0.0, 0.0, 0.0)
        cells = state.num_cells
        return (1.0, limit / cells, max(limit - state.steps_since_fruit, 0) / cells)


TRAINING_RULES = Rules(fruit_timeout=True)
"""The N + L fruit timeout: the rules for training and for the main evaluation."""

NO_LIMIT = Rules(fruit_timeout=False)
"""The engine's own rules: no timeout (evaluation check; loops end censored)."""


# ------------------------------------------------------------------ actions
_TURNS = (RelativeAction.STRAIGHT, RelativeAction.RIGHT, RelativeAction.STRAIGHT, RelativeAction.LEFT)


def to_direction(heading: Direction, action: RelativeAction) -> Direction:
    """The absolute direction of ``action`` for a snake heading ``heading``."""
    return heading.turn(action)


def to_relative(heading: Direction, direction: Direction) -> RelativeAction:
    """The relative action that moves in ``direction``.

    A reversal is ``STRAIGHT``, because the engine ignores reversals and keeps
    going straight.
    """
    return _TURNS[(int(direction) - int(heading)) % 4]


def play_move(game: SnakeGame, action: RelativeAction | Direction) -> StepResult:
    """Make an agent's move: a relative action turns the snake, a direction is absolute."""
    if isinstance(action, RelativeAction):
        return game.step_relative(action)
    if isinstance(action, Direction):
        return game.step(action)
    raise TypeError(f"an agent must return a RelativeAction or a Direction, got {action!r}")


def as_policy(agent: Agent) -> Callable[[GameState], Direction]:
    """Wrap ``agent`` for ``policy(state) -> Direction`` interfaces such as the game window.

    For example ``python -m snake_game --agent my_module:policy`` with
    ``policy = as_policy(MyAgent())`` in ``my_module``.
    """

    def policy(state: GameState) -> Direction:
        action = agent(state)
        return to_direction(state.direction, action) if isinstance(action, RelativeAction) else action

    return policy


# ----------------------------------------------------------------- planning
def planning_game(state: GameState, seed: int) -> SnakeGame:
    """A simulator positioned at ``state`` whose future fruits come from ``seed``.

    Search and rollouts start from this, never from the live game: a clone of
    the live game replays its hidden random stream, so it would reveal where
    the next fruits appear. The visible fruit is kept.
    """
    return SnakeGame.from_state(state, seed=seed)


def choose_safely(
    win_values: Sequence[float],
    moves: Sequence[float] | None = None,
    allowed: Sequence[bool] | None = None,
) -> int:
    """Index of the move to play when safety comes strictly first.

    The highest win probability decides. Expected moves (fewer is better) only
    break exact ties between win probabilities, and remaining ties go to the
    lowest index. ``allowed`` restricts the choice, for example to the
    cycle-order envelope.
    """
    if any(math.isnan(v) for v in win_values):
        raise ValueError("win values must not be NaN")
    candidates = [i for i in range(len(win_values)) if allowed is None or allowed[i]]
    if not candidates:
        raise ValueError("no move is allowed")
    best = max(win_values[i] for i in candidates)
    tied = [i for i in candidates if win_values[i] == best]
    if moves is None or len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda i: moves[i])
