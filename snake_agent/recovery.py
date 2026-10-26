"""Recovery starts (track T2): certified mid-game states that no cycle is aligned with.

The states come from games of :class:`~snake_agent.references.GuardedGreedyPolicy`.
From each game, :func:`recovery_starts` takes the first state at each target
occupancy (by default 30%, 50% and 70% of the board, each band 10% wide) that

* has a survival certificate (:mod:`snake_agent.survival`), so the start is not
  already lost to a trap, and
* is not aligned with any Hamiltonian cycle of the board, so following a cycle
  is not an immediate solution.

A game that skips through a band without such a state contributes nothing to
it. Whether every start can be won is unknown; reference results on them are
lower bounds on what is achievable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from snake_game import GameConfig, GameState, SnakeGame

from .contract import NO_LIMIT
from .cycles import aligned_cycles
from .evaluation import STALL_FACTOR, run_game
from .references import GuardedGreedyPolicy
from .seeds import TEST, SeedRange, recovery_source_seeds
from .survival import Board

BANDS = (0.3, 0.5, 0.7)
"""Default target occupancies of the recovery starts."""
BAND_WIDTH = 0.1


@dataclass(frozen=True)
class RecoveryStart:
    """A recovery start and where it came from."""

    band: float
    """Target occupancy: the state's length is in ``[band, band + BAND_WIDTH)`` of the board."""
    state: GameState
    source_seed: int
    """Seed of the guarded-greedy game the state was taken from."""

    @property
    def name(self) -> str:
        """``"<source seed>@<move>"``: the source game and the move at which the state occurred."""
        return f"{self.source_seed}@{self.state.steps}"


def recovery_starts(
    width: int,
    height: int,
    source_seeds: Iterable[int],
    *,
    bands: Sequence[float] = BANDS,
    band_width: float = BAND_WIDTH,
) -> list[RecoveryStart]:
    """The recovery starts from one guarded-greedy game per seed, in game order.

    The source games run without a fruit timeout until they are won, lost, loop
    or stall (see :func:`~snake_agent.evaluation.run_game`).
    """
    cells = width * height
    starts = []
    for seed in source_seeds:
        states: list[GameState] = []
        game = SnakeGame(GameConfig(width, height), seed=seed)
        run_game(
            game,
            GuardedGreedyPolicy(),
            NO_LIMIT,
            detect_loops=True,
            stall_limit=STALL_FACTOR * cells,
            record=states,
        )
        wanted = list(bands)
        for state in states:
            while wanted and state.length >= (wanted[0] + band_width) * cells:
                wanted.pop(0)  # no qualifying state in this band
            if wanted and state.length >= wanted[0] * cells:
                if not aligned_cycles(state) and Board(state).has_survival_certificate():
                    starts.append(RecoveryStart(wanted.pop(0), state, seed))
    return starts


def held_out_starts(width: int, height: int, sources: int, split: SeedRange = TEST) -> list[RecoveryStart]:
    """Recovery starts from ``sources`` games with seeds of a held-out range."""
    return recovery_starts(width, height, recovery_source_seeds(split, sources))
