"""Observation encodings for agents.

Agents never look at pixels: every encoding below is an exact, symbolic
description of the game state. Three encodings are available.

``"grid"`` - ``float32`` array ``(8, height, width)``, one plane per channel of
:data:`GRID_CHANNELS`:

====  ============  =======================================================
  0   head          1 on the head cell
  1   body          1 on every other snake cell (tail included)
  2   fruit         1 on the fruit cell
  3   countdown     moves (without eating) until the cell is vacated, divided
                    by ``width * height``: head = length / cells, tail = 1 / cells
 4-7  dir_*         one-hot heading of the segment on the cell (up, right,
                    down, left): the head's current heading, and for every other
                    segment the direction towards the segment in front of it
====  ============  =======================================================

``"coords"`` - a dict of ``int32`` arrays with the raw coordinates
(``cells = width * height``):

* ``snake``: ``(cells, 2)`` ``(x, y)`` of every segment, head first; rows past
  the snake's length are ``-1``.
* ``segment_directions``: ``(cells,)`` heading of every segment (see
  :attr:`~snake_game.engine.GameState.segment_directions`); ``-1`` padding.
* ``length``, ``direction``: scalars. ``fruit``: ``(2,)`` ``(x, y)``, ``-1`` if none.

``"features"`` - ``float32`` vector of :data:`FEATURE_NAMES` (19 values in
``[-1, 1]``): heading one-hot, head position, fruit and tail offsets relative to
the head, length, immediate danger and free straight-line run in each absolute
direction (wall or a body segment that is still there after the next move).

The ``encode_*`` functions work on batches of boards described by flat NumPy
arrays (cell index ``y * width + x``); they are shared by
:class:`~snake_game.env.SnakeEnv` and :class:`~snake_game.vector_env.VectorSnakeEnv`.
The ``*_observation`` helpers encode a single :class:`~snake_game.engine.GameState`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from .engine import GameState

ObsType = Literal["grid", "coords", "features"]
OBS_TYPES: tuple[str, ...] = ("grid", "coords", "features")

GRID_CHANNELS: tuple[str, ...] = (
    "head",
    "body",
    "fruit",
    "countdown",
    "dir_up",
    "dir_right",
    "dir_down",
    "dir_left",
)

FEATURE_NAMES: tuple[str, ...] = (
    "heading_up",
    "heading_right",
    "heading_down",
    "heading_left",
    "head_x",
    "head_y",
    "fruit_dx",
    "fruit_dy",
    "tail_dx",
    "tail_dy",
    "length",
    "danger_up",
    "danger_right",
    "danger_down",
    "danger_left",
    "free_up",
    "free_right",
    "free_down",
    "free_left",
)

# _HEADING_LUT[dx + 1, dy + 1] -> direction of the unit step (dx, dy), -1 otherwise.
_HEADING_LUT = np.full((3, 3), -1, dtype=np.int64)
for _code, (_dx, _dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
    _HEADING_LUT[_dx + 1, _dy + 1] = _code


def check_obs_type(obs_type: str) -> str:
    if obs_type not in OBS_TYPES:
        raise ValueError(f"obs_type must be one of {OBS_TYPES}, got {obs_type!r}")
    return obs_type


def observation_space(obs_type: str, width: int, height: int) -> spaces.Space:
    """The Gymnasium space of a single observation."""
    check_obs_type(obs_type)
    cells = width * height
    if obs_type == "grid":
        return spaces.Box(0.0, 1.0, (len(GRID_CHANNELS), height, width), np.float32)
    if obs_type == "features":
        return spaces.Box(-1.0, 1.0, (len(FEATURE_NAMES),), np.float32)
    xy_high = np.array([width - 1, height - 1], dtype=np.int32)
    return spaces.Dict(
        {
            "snake": spaces.Box(-1, np.tile(xy_high, (cells, 1)), (cells, 2), np.int32),
            "segment_directions": spaces.Box(-1, 3, (cells,), np.int32),
            "length": spaces.Box(1, cells, (), np.int32),
            "direction": spaces.Box(0, 3, (), np.int32),
            "fruit": spaces.Box(-1, xy_high, (2,), np.int32),
        }
    )


# ------------------------------------------------------------- batched encoders
def encode_grid(
    width: int,
    height: int,
    countdown: np.ndarray,
    cell_heading: np.ndarray,
    head: np.ndarray,
    direction: np.ndarray,
    fruit: np.ndarray,
) -> np.ndarray:
    """Grid planes ``(N, 8, height, width)`` for ``N`` boards.

    Args:
        countdown: ``(N, cells)`` moves until each cell is vacated (0 = free).
        cell_heading: ``(N, cells)`` heading of the segment on each snake cell
            (ignored on free cells and on the head, which uses ``direction``).
        head, direction, fruit: ``(N,)`` head cell, heading, fruit cell (-1 = none).
    """
    n, cells = countdown.shape
    rows = np.arange(n)
    # np.empty + explicit writes: zeroing the whole batch would cost more than encoding it.
    planes = np.empty((n, len(GRID_CHANNELS), cells), dtype=np.float32)
    planes[:, 0] = 0.0
    planes[:, 2] = 0.0
    body = countdown > 0
    planes[:, 1] = body
    planes[rows, 1, head] = 0.0
    planes[rows, 0, head] = 1.0
    has_fruit = fruit >= 0
    planes[rows[has_fruit], 2, fruit[has_fruit]] = 1.0
    np.multiply(countdown, np.float32(1.0 / cells), out=planes[:, 3], casting="unsafe")
    heading = np.where(body, cell_heading, -1)
    for d in range(4):
        planes[:, 4 + d] = heading == d
    planes[rows, 4:8, head] = 0.0
    planes[rows, 4 + direction, head] = 1.0
    return planes.reshape(n, len(GRID_CHANNELS), height, width)


def free_runs(width: int, height: int, blocked: np.ndarray, head: np.ndarray) -> np.ndarray:
    """``(N, 4)`` free cells in a straight line from the head (up, right, down, left)."""
    n = head.shape[0]
    rows = np.arange(n)
    hx, hy = head % width, head // width
    grid = blocked.reshape(n, height, width)
    row = grid[rows, hy, :]
    col = grid[rows, :, hx]
    xs, ys = np.arange(width), np.arange(height)
    right = np.where(row & (xs > hx[:, None]), xs, width).min(axis=1) - hx - 1
    left = hx - np.where(row & (xs < hx[:, None]), xs, -1).max(axis=1) - 1
    down = np.where(col & (ys > hy[:, None]), ys, height).min(axis=1) - hy - 1
    up = hy - np.where(col & (ys < hy[:, None]), ys, -1).max(axis=1) - 1
    return np.stack([up, right, down, left], axis=1)


def encode_features(
    width: int,
    height: int,
    countdown: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    direction: np.ndarray,
    fruit: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    """Feature vectors ``(N, 19)`` (see :data:`FEATURE_NAMES`)."""
    n, cells = countdown.shape
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    out[np.arange(n), direction] = 1.0
    sx, sy = max(width - 1, 1), max(height - 1, 1)
    hx, hy = head % width, head // width
    out[:, 4] = hx / sx
    out[:, 5] = hy / sy
    has_fruit = fruit >= 0
    out[:, 6] = (np.where(has_fruit, fruit % width, hx) - hx) / sx
    out[:, 7] = (np.where(has_fruit, fruit // width, hy) - hy) / sy
    out[:, 8] = (tail % width - hx) / sx
    out[:, 9] = (tail // width - hy) / sy
    out[:, 10] = length / cells
    # The tail (countdown 1) moves away before the head arrives, so it never blocks.
    runs = free_runs(width, height, countdown >= 2, head)
    out[:, 11:15] = runs == 0
    out[:, 15:19] = runs / np.array([sy, sx, sy, sx], dtype=np.float32)
    return out


def encode_coords(
    width: int,
    height: int,
    body: np.ndarray,
    headings: np.ndarray,
    length: np.ndarray,
    direction: np.ndarray,
    fruit: np.ndarray,
) -> dict[str, np.ndarray]:
    """Coordinate dicts for ``N`` boards.

    Args:
        body: ``(N, K)`` segment cells, head first, ``-1`` past each snake's
            length (``K <= cells``; the output is padded to ``cells`` rows).
        headings: ``(N, K)`` segment headings, ``-1`` padded likewise.
    """
    n, k = body.shape
    cells = width * height
    valid = body >= 0
    snake = np.full((n, cells, 2), -1, dtype=np.int32)
    snake[:, :k, 0] = np.where(valid, body % width, -1)
    snake[:, :k, 1] = np.where(valid, body // width, -1)
    directions = np.full((n, cells), -1, dtype=np.int32)
    directions[:, :k] = headings
    has_fruit = fruit >= 0
    fruit_xy = np.stack(
        [np.where(has_fruit, fruit % width, -1), np.where(has_fruit, fruit // width, -1)], axis=-1
    ).astype(np.int32)
    return {
        "snake": snake,
        "segment_directions": directions,
        "length": length.astype(np.int32),
        "direction": direction.astype(np.int32),
        "fruit": fruit_xy,
    }


# ---------------------------------------------------------------- single boards
def _single(state: GameState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat cells and headings of the snake (head first) plus the fruit cell."""
    snake = np.asarray(state.snake, dtype=np.int64).reshape(-1, 2)
    cells = snake[:, 1] * state.width + snake[:, 0]
    headings = np.empty(len(cells), dtype=np.int64)
    headings[0] = int(state.direction)
    if len(cells) > 1:
        delta = snake[:-1] - snake[1:]
        headings[1:] = _HEADING_LUT[delta[:, 0] + 1, delta[:, 1] + 1]
    fruit = -1 if state.fruit is None else state.fruit[1] * state.width + state.fruit[0]
    return cells, headings, np.array([fruit], dtype=np.int64)


def grid_observation(state: GameState) -> np.ndarray:
    """``"grid"`` encoding ``(8, height, width)`` of one state."""
    cells, headings, fruit = _single(state)
    n_cells = state.width * state.height
    countdown = np.zeros((1, n_cells), dtype=np.int64)
    countdown[0, cells] = np.arange(len(cells), 0, -1)
    cell_heading = np.zeros((1, n_cells), dtype=np.int64)
    cell_heading[0, cells] = headings
    direction = np.array([int(state.direction)], dtype=np.int64)
    return encode_grid(state.width, state.height, countdown, cell_heading, cells[:1], direction, fruit)[0]


def features_observation(state: GameState) -> np.ndarray:
    """``"features"`` encoding ``(19,)`` of one state."""
    cells, _, fruit = _single(state)
    countdown = np.zeros((1, state.width * state.height), dtype=np.int64)
    countdown[0, cells] = np.arange(len(cells), 0, -1)
    direction = np.array([int(state.direction)], dtype=np.int64)
    length = np.array([len(cells)], dtype=np.int64)
    return encode_features(
        state.width, state.height, countdown, cells[:1], cells[-1:], direction, fruit, length
    )[0]


def coords_observation(state: GameState) -> dict[str, np.ndarray]:
    """``"coords"`` encoding of one state."""
    cells, headings, fruit = _single(state)
    direction = np.array([int(state.direction)], dtype=np.int64)
    length = np.array([len(cells)], dtype=np.int64)
    batch = encode_coords(
        state.width, state.height, cells[None, :], headings[None, :], length, direction, fruit
    )
    return {key: value[0, ...] for key, value in batch.items()}  # [0, ...] keeps 0-d arrays


def observe(state: GameState, obs_type: str = "grid") -> Any:
    """Encode ``state`` with the given observation type."""
    check_obs_type(obs_type)
    if obs_type == "grid":
        return grid_observation(state)
    if obs_type == "features":
        return features_observation(state)
    return coords_observation(state)
