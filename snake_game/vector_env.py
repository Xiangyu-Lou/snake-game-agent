"""Batched Snake: many independent games stepped together with NumPy.

:class:`VectorSnakeEnv` implements Gymnasium's :class:`~gymnasium.vector.VectorEnv`
API without one Python object per game. All games live in a few ``(num_envs,
cells)`` arrays and every step is a handful of vectorised operations, so the
cost per game-step shrinks as the batch grows (see ``python -m
snake_game.benchmark``).

Board representation (per game, ``cells = width * height`` flat cell indices):

* ``t`` counts the moves of the episode; the head entered its cell at time ``t``.
* ``entered[c]`` is the time at which the head last entered cell ``c``. Cell ``c``
  belongs to the snake exactly when ``t - entered[c] < length``; the head has
  age 0 and the tail age ``length - 1``. Moving therefore only writes the new
  head's timestamp: the tail is vacated implicitly, growth is ``length += 1``.
* ``ring[t % cells]`` is the cell entered at time ``t``, which yields the
  segments in head-to-tail order without sorting.
* ``exit_dir[c]`` is the direction in which the head left cell ``c``: the
  heading of the segment now occupying it.

The rules and the random fruit stream are bit-identical to
:class:`~snake_game.engine.SnakeGame`: after ``reset(seed=s)`` sub-environment
``i`` plays exactly the game of ``SnakeGame(seed=s + i)`` (and of
``SnakeEnv.reset(seed=s + i)``), including the episodes started by autoreset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from . import _rng
from .engine import DeathCause, Direction, GameConfig, GameState, Status, StepResult
from .env import FruitTimeout, Rewards, check_action_mode, make_action_space, resolve_fruit_timeout
from .observations import (
    check_obs_type,
    encode_coords,
    encode_features,
    encode_grid,
    observation_space,
)

_FREE = -(1 << 62)  # "entered" timestamp of a cell that was never visited
_DX = np.array([0, 1, 0, -1], dtype=np.int64)
_DY = np.array([-1, 0, 1, 0], dtype=np.int64)
_NO_DEATH, _WALL, _SELF = 0, 1, 2


class VectorSnakeEnv(VectorEnv):
    """``num_envs`` independent Snake games stepped in lock-step with NumPy.

    Takes the same game/observation/reward arguments as
    :class:`~snake_game.env.SnakeEnv`, plus:

    Args:
        num_envs: number of games in the batch.
        autoreset_mode: how finished games restart (see
            https://farama.org/Vector-Autoreset-Mode):

            * ``AutoresetMode.NEXT_STEP`` (default, Gymnasium's convention): the
              ``step`` after a game ended resets it, ignoring its action and
              returning reward 0 with both flags ``False``.
            * ``SAME_STEP``: finished games are reset within the same ``step``;
              their last observation and info are returned in
              ``infos["final_obs"]`` / ``infos["final_info"]``.
            * ``DISABLED``: reset finished games yourself with
              ``reset(options={"reset_mask": mask})``.
        render_env: index of the game shown by ``render_mode="human"`` (a live,
            non-blocking preview of training) or returned by ``render()`` in
            ``"rgb_array"`` mode.
        render_fps: maximum redraw rate of the ``"human"`` preview.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int = 16,
        width: int = 20,
        height: int = 20,
        initial_length: int = 3,
        *,
        obs_type: str = "grid",
        action_mode: str = "absolute",
        rewards: Rewards | None = None,
        max_steps_without_fruit: FruitTimeout = "auto",
        autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
        render_mode: str | None = None,
        render_env: int = 0,
        render_fps: float | None = None,
        cell_size: int | None = None,
    ) -> None:
        num_envs = int(num_envs)
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs = num_envs
        self.config = config = GameConfig(width, height, initial_length)
        self.obs_type = check_obs_type(obs_type)
        self.action_mode = check_action_mode(action_mode)
        self.rewards = rewards if rewards is not None else Rewards()
        self.max_steps_without_fruit = resolve_fruit_timeout(max_steps_without_fruit, config)
        self.autoreset_mode = (
            autoreset_mode if isinstance(autoreset_mode, AutoresetMode) else AutoresetMode(autoreset_mode)
        )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None")
        if not 0 <= render_env < num_envs:
            raise ValueError(f"render_env must be in [0, {num_envs}), got {render_env}")
        self.render_mode = render_mode
        self.render_env = int(render_env)
        fps = float(render_fps) if render_fps is not None else float(self.metadata["render_fps"])
        self.metadata = {
            **type(self).metadata,
            "autoreset_mode": self.autoreset_mode,
            "render_fps": fps,
        }
        self.cell_size = cell_size

        self.single_observation_space = observation_space(self.obs_type, config.width, config.height)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = make_action_space(self.action_mode)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self._num_actions = int(self.single_action_space.n)

        n, cells = num_envs, config.num_cells
        self._width, self._height, self._cells = config.width, config.height, cells
        self._rows = np.arange(n)
        self._entered = np.full((n, cells), _FREE, dtype=np.int64)
        self._ring = np.zeros((n, cells), dtype=np.int64)
        self._exit_dir = np.zeros((n, cells), dtype=np.int8)
        self._t = np.zeros(n, dtype=np.int64)
        self._length = np.zeros(n, dtype=np.int64)
        self._head = np.zeros(n, dtype=np.int64)
        self._dir = np.zeros(n, dtype=np.int64)
        self._fruit = np.zeros(n, dtype=np.int64)
        self._score = np.zeros(n, dtype=np.int64)
        self._steps = np.zeros(n, dtype=np.int64)
        self._since_fruit = np.zeros(n, dtype=np.int64)
        self._won = np.zeros(n, dtype=bool)
        self._death = np.zeros(n, dtype=np.int8)
        self._keys = np.array([_rng.random_key() for _ in range(n)], dtype=np.uint64)
        self._draws = np.zeros(n, dtype=np.uint64)
        # NEXT_STEP: games to reset on the next step. DISABLED: games awaiting reset().
        self._pending_reset = np.zeros(n, dtype=bool)

        start = config.start_snake()
        self._start_cells = np.array([y * config.width + x for x, y in start], dtype=np.int64)
        self._viewer: Any = None
        self._render_enabled = True
        self._reset_games(self._rows, self._keys)

    # --------------------------------------------------------------- gym API
    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset all games, or only those selected by ``options["reset_mask"]``.

        ``seed`` may be ``None`` (continue every game's random stream), an int
        (game ``i`` is seeded with ``seed + i``) or one seed (or ``None``) per game.
        """
        n = self.num_envs
        if isinstance(seed, (int, np.integer)):
            super().reset(seed=int(seed))
            seeds: list[int | None] = [int(seed) + i for i in range(n)]
        elif seed is None:
            seeds = [None] * n
        else:
            seeds = list(seed)
            if len(seeds) != n:
                raise ValueError(f"expected {n} seeds, got {len(seeds)}")

        mask = (options or {}).get("reset_mask")
        if mask is None:
            idx = self._rows
        else:
            mask = np.asarray(mask)
            if mask.shape != (n,) or mask.dtype != np.bool_:
                raise ValueError(f"options['reset_mask'] must be a bool array of shape ({n},)")
            idx = np.flatnonzero(mask)
        keys = _rng.next_key_np(self._keys[idx])
        for j, i in enumerate(idx):
            if seeds[i] is not None:
                keys[j] = _rng.seed_to_key(seeds[i])
        self._reset_games(idx, keys)
        self._pending_reset[idx] = False
        if self.render_mode == "human":
            self._render_frame(None)
        return self._observe(), self._infos()

    def step(self, actions: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        n, width, height, cells = self.num_envs, self._width, self._height, self._cells
        actions = np.asarray(actions)
        if actions.shape != (n,):
            raise ValueError(f"expected actions of shape ({n},), got {actions.shape}")
        if actions.dtype.kind not in "iu":
            raise TypeError(f"actions must be integers, got dtype {actions.dtype}")
        if actions.size and (actions.min() < 0 or actions.max() >= self._num_actions):
            raise ValueError(f"actions must be in [0, {self._num_actions})")
        actions = actions.astype(np.int64, copy=False)

        if self.autoreset_mode is AutoresetMode.NEXT_STEP:
            restarting = self._pending_reset.copy()
            if restarting.any():
                idx = np.flatnonzero(restarting)
                self._reset_games(idx, _rng.next_key_np(self._keys[idx]))
            active = ~restarting
        else:
            if self.autoreset_mode is AutoresetMode.DISABLED and self._pending_reset.any():
                raise RuntimeError(
                    "some games are over; reset them with "
                    "reset(options={'reset_mask': mask}) before stepping (autoreset is disabled)"
                )
            active = np.ones(n, dtype=bool)

        rows = self._rows
        heading = self._dir
        if self.action_mode == "absolute":
            new_dir = np.where(actions == (heading + 2) % 4, heading, actions)
        else:
            new_dir = (heading + actions - 1) % 4
        new_dir = np.where(active, new_dir, heading)
        hx, hy = self._head % width, self._head // width
        nx, ny = hx + _DX[new_dir], hy + _DY[new_dir]
        wall = (nx < 0) | (nx >= width) | (ny < 0) | (ny >= height)
        cell = np.clip(ny, 0, height - 1) * width + np.clip(nx, 0, width - 1)
        age = self._t - self._entered[rows, cell]
        # Cells with age < length - 1 stay occupied after this move; the tail
        # (age length - 1) moves away, and the fruit cell is always free.
        hit_self = ~wall & (age < self._length - 1)
        dead = active & (wall | hit_self)
        moved = active & ~dead
        ate = moved & (cell == self._fruit)

        self._dir = new_dir
        self._steps[active] += 1
        self._since_fruit[active] += 1
        mi = np.flatnonzero(moved)
        if mi.size:
            target = cell[mi]
            self._exit_dir[mi, self._head[mi]] = new_dir[mi]
            t = self._t[mi] + 1
            self._t[mi] = t
            self._entered[mi, target] = t
            self._ring[mi, t % cells] = target
            self._head[mi] = target
        won_now = np.zeros(n, dtype=bool)
        ai = np.flatnonzero(ate)
        if ai.size:
            self._length[ai] += 1
            self._score[ai] += 1
            self._since_fruit[ai] = 0
            full = self._length[ai] == cells
            won_now[ai[full]] = True
            self._won[ai[full]] = True
            self._fruit[ai[full]] = -1
            self._spawn_fruit(ai[~full])
        self._death[dead & wall] = _WALL
        self._death[dead & ~wall] = _SELF

        terminated = dead | won_now
        if self.max_steps_without_fruit is None:
            truncated = np.zeros(n, dtype=bool)
        else:
            truncated = moved & ~terminated & (self._since_fruit >= self.max_steps_without_fruit)
        r = self.rewards
        rewards = (r.step * active + r.fruit * ate + r.death * dead + r.win * won_now).astype(np.float64)
        done = terminated | truncated

        if self.render_mode == "human":
            i = self.render_env
            result = StepResult(
                Direction(int(new_dir[i])),
                bool(ate[i]),
                Status.WON if won_now[i] else Status.LOST if dead[i] else Status.RUNNING,
                DeathCause.WALL if dead[i] and wall[i] else DeathCause.SELF if dead[i] else None,
                (int(nx[i]), int(ny[i])) if dead[i] else None,
            )
            self._render_frame(result)

        final: dict[str, Any] = {}
        if self.autoreset_mode is AutoresetMode.SAME_STEP:
            di = np.flatnonzero(done)
            if di.size:
                final = self._final_entries(di, done)
                self._reset_games(di, _rng.next_key_np(self._keys[di]))
        else:
            self._pending_reset = done
        infos = self._infos()
        infos.update(final)
        return self._observe(), rewards, terminated, truncated, infos

    def action_masks(self) -> np.ndarray:
        """``(num_envs, n_actions)`` mask of the actions that are not immediately fatal.

        Same semantics as :meth:`SnakeEnv.action_masks`: the reversal is masked in
        absolute mode and a game with no safe move gets every non-reversal action.
        """
        return self._action_masks(self._rows)

    def state(self, index: int = 0) -> GameState:
        """Exact :class:`~snake_game.engine.GameState` of game ``index``."""
        i = int(index)
        if not 0 <= i < self.num_envs:
            raise IndexError(f"game index {i} out of range for {self.num_envs} games")
        width, cells = self._width, self._cells
        length, t = int(self._length[i]), int(self._t[i])
        body = self._ring[i, (t - np.arange(length)) % cells]
        fruit = int(self._fruit[i])
        death = int(self._death[i])
        if self._won[i]:
            status = Status.WON
        elif death:
            status = Status.LOST
        else:
            status = Status.RUNNING
        return GameState(
            width=width,
            height=self._height,
            snake=tuple((int(c) % width, int(c) // width) for c in body),
            direction=Direction(int(self._dir[i])),
            fruit=None if fruit < 0 else (fruit % width, fruit // width),
            score=int(self._score[i]),
            steps=int(self._steps[i]),
            steps_since_fruit=int(self._since_fruit[i]),
            status=status,
            death_cause={_WALL: DeathCause.WALL, _SELF: DeathCause.SELF}.get(death),
        )

    # ------------------------------------------------------------- rendering
    @property
    def render_enabled(self) -> bool:
        """Whether the ``"human"`` preview draws (also toggled with ``R`` in the window)."""
        if self._viewer is not None:
            return self._viewer.enabled
        return self._render_enabled

    @render_enabled.setter
    def render_enabled(self, enabled: bool) -> None:
        self._render_enabled = bool(enabled)
        if self._viewer is not None:
            self._viewer.enabled = self._render_enabled

    def render(self) -> np.ndarray | None:
        """``"rgb_array"``: frame of game ``render_env``; ``"human"``: redraw the preview."""
        if self.render_mode is None:
            gym.logger.warn("Calling render() without a render_mode.")
            return None
        if self.render_mode == "rgb_array":
            return self._get_viewer().frame(
                self.state(self.render_env), caption=self._caption(), dt=1.0 / self.metadata["render_fps"]
            )
        self._render_frame(None, force=True)
        return None

    def _render_frame(self, result: StepResult | None, *, force: bool = False) -> None:
        # Non-blocking: redraw at most render_fps times per second; in between (or
        # while rendering is toggled off) only keep the window responsive.
        viewer = self._get_viewer()
        if viewer.enabled and (force or viewer.due()):
            viewer.show(self.state(self.render_env), result, caption=self._caption())
        else:
            viewer.pump()

    def _caption(self) -> str:
        return f"game {self.render_env + 1} of {self.num_envs}"

    def _get_viewer(self) -> Any:
        if self._viewer is None:
            from .ui.viewer import EnvViewer  # imported lazily: headless training needs no pygame

            self._viewer = EnvViewer(
                self._width,
                self._height,
                window=self.render_mode == "human",
                cell_size=self.cell_size,
                title="Snake - VectorSnakeEnv",
                max_fps=self.metadata["render_fps"],
            )
            self._viewer.enabled = self._render_enabled
        return self._viewer

    def close_extras(self, **kwargs: Any) -> None:
        # getattr: VectorEnv.__del__ also closes instances whose __init__ raised.
        if getattr(self, "_viewer", None) is not None:
            self._viewer.close()
            self._viewer = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_viewer"] = None
        return state

    # ------------------------------------------------------------- internals
    def _reset_games(self, idx: np.ndarray, keys: np.ndarray) -> None:
        if idx.size == 0:
            return
        start = self._start_cells
        length = start.size
        cells = self._cells
        self._keys[idx] = keys
        self._draws[idx] = 0
        self._entered[idx] = _FREE
        offsets = np.arange(length)
        self._entered[idx[:, None], start[None, :]] = -offsets[None, :]
        self._ring[idx[:, None], (-offsets[None, :]) % cells] = start[None, :]
        self._exit_dir[idx[:, None], start[None, 1:]] = int(Direction.RIGHT)
        self._t[idx] = 0
        self._length[idx] = length
        self._head[idx] = start[0]
        self._dir[idx] = int(Direction.RIGHT)
        self._score[idx] = 0
        self._steps[idx] = 0
        self._since_fruit[idx] = 0
        self._won[idx] = False
        self._death[idx] = _NO_DEATH
        self._spawn_fruit(idx)

    def _spawn_fruit(self, idx: np.ndarray) -> None:
        """Put a fruit on a uniformly random free cell of each game in ``idx``."""
        if idx.size == 0:
            return
        length = self._length[idx]
        free = (self._t[idx, None] - self._entered[idx]) >= length[:, None]
        k = _rng.uniform_index_np(self._keys[idx], self._draws[idx], self._cells - length)
        self._draws[idx] += np.uint64(1)
        # The k-th free cell in row-major order, exactly like SnakeGame.
        self._fruit[idx] = np.argmax(np.cumsum(free, axis=1) > k[:, None], axis=1)

    def _countdown(self, idx: np.ndarray | slice) -> np.ndarray:
        # length - (t - entered) == entered - (t - length): one pass over the board.
        countdown = self._entered[idx] - (self._t[idx] - self._length[idx])[:, None]
        np.maximum(countdown, 0, out=countdown)
        return countdown

    def _observe(self, idx: np.ndarray | None = None) -> Any:
        sel: np.ndarray | slice = slice(None) if idx is None else idx
        rows = self._rows if idx is None else idx
        width, height = self._width, self._height
        if self.obs_type == "grid":
            return encode_grid(
                width,
                height,
                self._countdown(sel),
                self._exit_dir[sel],
                self._head[sel],
                self._dir[sel],
                self._fruit[sel],
            )
        if self.obs_type == "features":
            length, t = self._length[sel], self._t[sel]
            tail = self._ring[rows, (t - length + 1) % self._cells]
            return encode_features(
                width,
                height,
                self._countdown(sel),
                self._head[sel],
                tail,
                self._dir[sel],
                self._fruit[sel],
                length,
            )
        body, headings = self._ordered_body(rows)
        return encode_coords(
            width, height, body, headings, self._length[sel], self._dir[sel], self._fruit[sel]
        )

    def _ordered_body(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Segment cells and headings of games ``rows``: ``(len(rows), longest snake)``,
        head first, ``-1`` past each snake's length."""
        length, t = self._length[rows], self._t[rows]
        k = np.arange(int(length.max()))
        body = self._ring[rows[:, None], (t[:, None] - k[None, :]) % self._cells]
        headings = self._exit_dir[rows[:, None], body].astype(np.int64)
        headings[:, 0] = self._dir[rows]
        beyond = k[None, :] >= length[:, None]
        body[beyond] = -1
        headings[beyond] = -1
        return body, headings

    def _action_masks(self, rows: np.ndarray) -> np.ndarray:
        width, height = self._width, self._height
        heading = self._dir[rows]
        if self.action_mode == "absolute":
            dirs = np.broadcast_to(np.arange(4), (rows.size, 4))
        else:
            dirs = (heading[:, None] + np.arange(-1, 2)[None, :]) % 4
        head = self._head[rows]
        nx = (head % width)[:, None] + _DX[dirs]
        ny = (head // width)[:, None] + _DY[dirs]
        inside = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
        cell = np.clip(ny, 0, height - 1) * width + np.clip(nx, 0, width - 1)
        age = self._t[rows, None] - self._entered[rows[:, None], cell]
        safe = inside & (age >= self._length[rows, None] - 1)
        safe[self._won[rows] | (self._death[rows] != _NO_DEATH)] = False  # finished game
        if self.action_mode == "absolute":
            not_reverse = dirs != ((heading + 2) % 4)[:, None]
            safe &= not_reverse
            trapped = ~safe.any(axis=1)
            safe[trapped] = not_reverse[trapped]
        else:
            safe[~safe.any(axis=1)] = True
        return safe

    def _infos(self) -> dict[str, np.ndarray]:
        """Info of every game, with Gymnasium's ``_key`` "is present" masks."""
        values = self._info_values(self._rows)
        present = np.ones(self.num_envs, dtype=bool)
        return {**values, **{f"_{key}": present for key in values}}

    def _info_values(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        death = self._death[rows]
        return {
            "score": self._score[rows],
            "length": self._length[rows],
            "steps": self._steps[rows],
            "steps_since_fruit": self._since_fruit[rows],
            "won": self._won[rows],
            "hit_wall": death == _WALL,
            "hit_self": death == _SELF,
            "action_mask": self._action_masks(rows),
        }

    def _final_entries(self, idx: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        """``final_obs`` / ``final_info`` entries in Gymnasium's SAME_STEP layout."""
        n = self.num_envs
        obs = self._observe(idx)
        final_obs = np.full(n, None, dtype=object)
        for j, i in enumerate(idx):  # [j, ...] keeps scalar fields as 0-d arrays, like SnakeEnv
            final_obs[i] = (
                {key: value[j, ...] for key, value in obs.items()} if isinstance(obs, dict) else obs[j]
            )
        final_info: dict[str, np.ndarray] = {}
        for key, value in self._info_values(idx).items():
            full = np.zeros((n, *value.shape[1:]), dtype=value.dtype)
            full[idx] = value
            final_info[key] = full
            final_info[f"_{key}"] = mask
        return {
            "final_obs": final_obs,
            "_final_obs": mask,
            "final_info": final_info,
            "_final_info": mask,
        }
