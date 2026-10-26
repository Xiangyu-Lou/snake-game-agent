"""Snake for AI research: an exact, deterministic engine, Gymnasium environments
(single and batched) and an animated pygame front-end.

See ``snake_game/README.md`` for the full documentation.
"""

from __future__ import annotations

from .controller import UNLIMITED, GameController, GameMode, Policy
from .engine import (
    DELTAS,
    DeathCause,
    Direction,
    GameConfig,
    GameOverError,
    GameState,
    Position,
    RelativeAction,
    SnakeGame,
    Status,
    StepResult,
)
from .env import Rewards, SnakeEnv
from .observations import FEATURE_NAMES, GRID_CHANNELS, OBS_TYPES, observe
from .vector_env import VectorSnakeEnv

__version__ = "0.1.0"

ENV_ID = "snake_game/Snake-v0"
"""Gymnasium id: ``gym.make(ENV_ID)`` / ``gym.make_vec(ENV_ID, num_envs=...)``."""

__all__ = [
    "DELTAS",
    "ENV_ID",
    "FEATURE_NAMES",
    "GRID_CHANNELS",
    "OBS_TYPES",
    "UNLIMITED",
    "DeathCause",
    "Direction",
    "GameConfig",
    "GameController",
    "GameMode",
    "GameOverError",
    "GameState",
    "Policy",
    "Position",
    "RelativeAction",
    "Rewards",
    "SnakeEnv",
    "SnakeGame",
    "Status",
    "StepResult",
    "VectorSnakeEnv",
    "observe",
]


def _register() -> None:
    from gymnasium.envs.registration import register, registry

    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="snake_game.env:SnakeEnv",
            vector_entry_point="snake_game.vector_env:VectorSnakeEnv",
        )


_register()
