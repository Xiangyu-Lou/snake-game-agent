"""Gymnasium environment wrapping a single :class:`~snake_game.engine.SnakeGame`.

For high-throughput training use :class:`~snake_game.vector_env.VectorSnakeEnv`,
which steps a whole batch of games with NumPy and reproduces this environment
exactly: sub-environment ``i`` of ``VectorSnakeEnv.reset(seed=s)`` plays the same
game as ``SnakeEnv.reset(seed=s + i)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .engine import (
    DeathCause,
    Direction,
    GameConfig,
    GameState,
    RelativeAction,
    SnakeGame,
    StepResult,
)
from .observations import check_obs_type, observation_space, observe

ActionMode = Literal["absolute", "relative"]
ACTION_MODES: tuple[str, ...] = ("absolute", "relative")
FruitTimeout = int | Literal["auto"] | None


@dataclass(frozen=True)
class Rewards:
    """Reward shaping. The game's own score is always one point per fruit."""

    fruit: float = 1.0
    death: float = -1.0
    win: float = 0.0
    """Bonus on top of ``fruit`` for the fruit that completes the board."""
    step: float = 0.0
    """Added on every step (use a small negative value to discourage dawdling)."""


def check_action_mode(action_mode: str) -> str:
    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
    return action_mode


def resolve_fruit_timeout(value: FruitTimeout, config: GameConfig) -> int | None:
    """``"auto"`` = one step per board cell; ``None`` disables truncation."""
    if value is None:
        return None
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"max_steps_without_fruit must be an int, None or 'auto', got {value!r}")
        return config.num_cells
    limit = int(value)
    if limit < 1:
        raise ValueError(f"max_steps_without_fruit must be positive, got {limit}")
    return limit


def make_action_space(action_mode: str) -> spaces.Discrete:
    return spaces.Discrete(4 if check_action_mode(action_mode) == "absolute" else 3)


class SnakeEnv(gym.Env):
    """Single-game Snake environment with the Gymnasium API.

    Args:
        width, height: board size (default 20x20).
        initial_length: starting length of the snake (default 3).
        obs_type: ``"grid"`` (default), ``"coords"`` or ``"features"``; see
            :mod:`snake_game.observations`.
        action_mode: ``"absolute"`` - ``Discrete(4)`` of :class:`Direction`
            (a reversal acts as "straight"), or ``"relative"`` - ``Discrete(3)`` of
            :class:`RelativeAction` (turn left, straight, turn right).
        rewards: :class:`Rewards` shaping (default: +1 fruit, -1 death).
        max_steps_without_fruit: truncate the episode after this many steps
            without eating (``"auto"`` = width x height, ``None`` = never).
        render_mode: ``None`` (no rendering, nothing imported), ``"human"``
            (animated window) or ``"rgb_array"`` (frames for videos).
        render_fps: moves per second shown in ``"human"`` mode.
        cell_size: pixel size of a board cell when rendering.

    ``info`` holds ``score``, ``length``, ``steps``, ``steps_since_fruit``,
    ``won``, ``hit_wall``, ``hit_self`` and ``action_mask`` (see
    :meth:`action_masks`). The underlying game is available as ``env.game`` for
    exact state queries, cloning (tree search) and so on. Pass
    ``options={"state": GameState(...)}`` to :meth:`reset` to start from an
    arbitrary position (e.g. a curriculum of long snakes).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        initial_length: int = 3,
        *,
        obs_type: str = "grid",
        action_mode: str = "absolute",
        rewards: Rewards | None = None,
        max_steps_without_fruit: FruitTimeout = "auto",
        render_mode: str | None = None,
        render_fps: float | None = None,
        cell_size: int | None = None,
    ) -> None:
        self.config = GameConfig(width, height, initial_length)
        self.obs_type = check_obs_type(obs_type)
        self.action_mode = check_action_mode(action_mode)
        self.rewards = rewards if rewards is not None else Rewards()
        self.max_steps_without_fruit = resolve_fruit_timeout(max_steps_without_fruit, self.config)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None")
        self.render_mode = render_mode
        fps = float(render_fps) if render_fps is not None else float(self.metadata["render_fps"])
        if fps <= 0:
            raise ValueError("render_fps must be positive")
        self.metadata = {**type(self).metadata, "render_fps": fps}
        self.cell_size = cell_size

        self.observation_space = observation_space(self.obs_type, self.config.width, self.config.height)
        self.action_space = make_action_space(self.action_mode)
        self.game = SnakeGame(self.config)
        self._viewer: Any = None
        self._render_enabled = True
        self._last_result: StepResult | None = None

    # --------------------------------------------------------------- gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        state = (options or {}).get("state")
        if state is not None:
            if not isinstance(state, GameState):
                state = GameState.from_dict(state)
            if seed is not None:
                self.game.reseed(seed)
            self.game.set_state(state)
        else:
            self.game.reset(seed)
        self._last_result = None
        if self.render_mode == "human":
            self._render_frame()
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.action_mode == "absolute":
            result = self.game.step(action)
        else:
            result = self.game.step_relative(action)
        self._last_result = result
        rewards = self.rewards
        reward = rewards.step
        if result.ate_fruit:
            reward += rewards.fruit
        if result.lost:
            reward += rewards.death
        elif result.won:
            reward += rewards.win
        terminated = result.done
        limit = self.max_steps_without_fruit
        truncated = not terminated and limit is not None and self.game.steps_since_fruit >= limit
        if self.render_mode == "human":
            self._render_frame()
        return self._observation(), float(reward), terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """Boolean mask of the actions that do not end the game immediately.

        In absolute mode the reversal is always masked out (it would merely act as
        "straight"). If every action is fatal, all non-reversal actions are marked
        valid so that the mask is never empty (masked policies stay well-defined).
        """
        game = self.game
        if self.action_mode == "absolute":
            mask = np.array([game.is_safe(d) for d in Direction], dtype=bool)
            if not mask.any():
                mask[:] = True
                mask[game.direction.opposite] = False
        else:
            heading = game.direction
            mask = np.array([game.is_safe(heading.turn(a)) for a in RelativeAction], dtype=bool)
            if not mask.any():
                mask[:] = True
        return mask

    def state(self) -> GameState:
        """Exact snapshot of the current game."""
        return self.game.state()

    # ------------------------------------------------------------- rendering
    @property
    def render_enabled(self) -> bool:
        """Whether ``"human"`` mode currently draws (also toggled with ``R`` in the window)."""
        if self._viewer is not None:
            return self._viewer.enabled
        return self._render_enabled

    @render_enabled.setter
    def render_enabled(self, enabled: bool) -> None:
        self._render_enabled = bool(enabled)
        if self._viewer is not None:
            self._viewer.enabled = self._render_enabled

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            gym.logger.warn(
                "Calling render() without a render_mode; pass render_mode='human' or "
                "'rgb_array' to SnakeEnv to enable rendering."
            )
            return None
        if self.render_mode == "rgb_array":
            return self._get_viewer().frame(
                self.game.state(), self._last_result, dt=1.0 / self.metadata["render_fps"]
            )
        self._render_frame()
        return None

    def _render_frame(self) -> None:
        viewer = self._get_viewer()
        viewer.show(self.game.state(), self._last_result, duration=1.0 / self.metadata["render_fps"])

    def _get_viewer(self) -> Any:
        if self._viewer is None:
            from .ui.viewer import EnvViewer  # imported lazily: headless training needs no pygame

            self._viewer = EnvViewer(
                self.config.width,
                self.config.height,
                window=self.render_mode == "human",
                cell_size=self.cell_size,
                title="Snake - SnakeEnv",
            )
            self._viewer.enabled = self._render_enabled
        return self._viewer

    def close(self) -> None:
        if getattr(self, "_viewer", None) is not None:
            self._viewer.close()
            self._viewer = None
        super().close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_viewer"] = None  # windows and surfaces are not picklable
        return state

    # -------------------------------------------------------------- helpers
    def _observation(self) -> Any:
        return observe(self.game.state(), self.obs_type)

    def _info(self) -> dict[str, Any]:
        game = self.game
        return {
            "score": game.score,
            "length": game.length,
            "steps": game.steps,
            "steps_since_fruit": game.steps_since_fruit,
            "won": game.won,
            "hit_wall": game.death_cause is DeathCause.WALL,
            "hit_self": game.death_cause is DeathCause.SELF,
            "action_mask": self.action_masks(),
        }
