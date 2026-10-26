"""Counter-based random numbers shared by the scalar and the vectorised engine.

Fruit placement has to be reproducible *per game*, no matter how many games run
side by side in a batch. Instead of a stateful generator, every draw hashes an
``(episode key, draw counter)`` pair with SplitMix64: the ``k``-th fruit of an
episode is the ``k``-th output of a SplitMix64 stream seeded with the episode
key. :class:`~snake_game.engine.SnakeGame` evaluates the hash with Python ints,
:class:`~snake_game.vector_env.VectorSnakeEnv` with ``uint64`` arrays, and both
produce bit-identical results.
"""

from __future__ import annotations

import secrets

import numpy as np

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15  # SplitMix64 increment (2**64 / golden ratio)
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_SEED_SALT = 0x5DEECE66D1F2A3B7
_EPISODE_SALT = 0xA0761D6478BD642F


def mix64(z: int) -> int:
    """SplitMix64 finaliser: a bijective, well-avalanching 64-bit mixer."""
    z &= MASK64
    z = ((z ^ (z >> 30)) * _MIX1) & MASK64
    z = ((z ^ (z >> 27)) * _MIX2) & MASK64
    return z ^ (z >> 31)


def seed_to_key(seed: int) -> int:
    """Derive the key of an episode from a user supplied, non-negative seed."""
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    return mix64(mix64(seed) ^ _SEED_SALT)


def next_key(key: int) -> int:
    """Derive the key of the next episode (used by ``reset()`` without a seed)."""
    return mix64(key ^ _EPISODE_SALT)


def random_key() -> int:
    """A fresh key from the operating system's entropy pool."""
    return secrets.randbits(64)


def uniform_index(key: int, counter: int, n: int) -> int:
    """Uniform integer in ``[0, n)`` for draw number ``counter`` of stream ``key``."""
    z = mix64(key + (counter + 1) * GOLDEN)
    return ((z >> 32) * n) >> 32


# --------------------------------------------------------------------------- numpy
_U64 = np.uint64


def mix64_np(z: np.ndarray) -> np.ndarray:
    """Vectorised :func:`mix64` (``uint64`` arithmetic wraps modulo 2**64)."""
    z = (z ^ (z >> _U64(30))) * _U64(_MIX1)
    z = (z ^ (z >> _U64(27))) * _U64(_MIX2)
    return z ^ (z >> _U64(31))


def next_key_np(keys: np.ndarray) -> np.ndarray:
    """Vectorised :func:`next_key`."""
    return mix64_np(keys ^ _U64(_EPISODE_SALT))


def uniform_index_np(keys: np.ndarray, counters: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Vectorised :func:`uniform_index`; all inputs are 1-D arrays of equal length."""
    z = mix64_np(keys + (counters + _U64(1)) * _U64(GOLDEN))
    return (((z >> _U64(32)) * n.astype(_U64)) >> _U64(32)).astype(np.int64)
