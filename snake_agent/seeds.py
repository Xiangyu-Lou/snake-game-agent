"""Seed ranges: training never sees the games used for validation and testing.

Every game's fruits follow from its seed, so keeping the seed ranges apart keeps
evaluation games out of training. :data:`TRAIN` is for self-play and
experiments, :data:`VALIDATION` for development runs and checkpoint selection,
and :data:`TEST` for the final gates only. Training code should pass its seeds
through :func:`check_training_seed`.

Within a held-out range, each kind of evaluation game has its own block, so the
standard-start games, the games that produce recovery starts and the fruit
streams of the recovery games never share a seed.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedRange:
    """The seeds ``start <= seed < stop``."""

    name: str
    start: int
    stop: int

    def __contains__(self, seed: object) -> bool:
        try:
            value = operator.index(seed)
        except TypeError:
            return False
        return self.start <= value < self.stop

    def take(self, count: int, offset: int = 0) -> range:
        """``count`` consecutive seeds from ``start + offset``."""
        first = self.start + offset
        if count < 0 or offset < 0 or first + count > self.stop:
            raise ValueError(f"{count} seeds at offset {offset} do not fit in the {self.name} range")
        return range(first, first + count)


TRAIN = SeedRange("train", 0, 1_000_000_000)
VALIDATION = SeedRange("validation", 1_000_000_000, 1_500_000_000)
TEST = SeedRange("test", 1_500_000_000, 2_000_000_000)
HELD_OUT = (VALIDATION, TEST)

STANDARD_GAMES = 0
"""Block offset of the standard-start (T1) game seeds within a held-out range."""
RECOVERY_SOURCES = 100_000_000
"""Block offset of the seeds of the games that recovery (T2) starts are taken from."""
RECOVERY_FRUITS = 200_000_000
"""Block offset of the fruit seeds of the recovery (T2) games."""


def check_training_seed(seed: int) -> int:
    """``seed``, or ``ValueError`` if it belongs to a held-out range."""
    for held_out in HELD_OUT:
        if seed in held_out:
            raise ValueError(f"seed {seed} is reserved for {held_out.name}")
    if seed < 0:
        raise ValueError(f"seeds are non-negative, got {seed}")
    return seed


def standard_seeds(split: SeedRange, games: int) -> range:
    """Seeds of ``games`` standard-start games of a held-out range."""
    return split.take(games, STANDARD_GAMES)


def recovery_source_seeds(split: SeedRange, games: int) -> range:
    """Seeds of the games that recovery starts are taken from."""
    return split.take(games, RECOVERY_SOURCES)


def recovery_fruit_seeds(split: SeedRange, starts: int) -> range:
    """Fruit seeds of the recovery games, one per start."""
    return split.take(starts, RECOVERY_FRUITS)
