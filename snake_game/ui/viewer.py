"""Window and frame rendering used by :class:`~snake_game.env.SnakeEnv` and
:class:`~snake_game.vector_env.VectorSnakeEnv`.

In the window, ``R`` toggles rendering (the environment keeps stepping at full
speed while it is off), ``+`` / ``-`` change the animation speed, and closing
the window simply turns rendering off instead of interrupting training.
"""

from __future__ import annotations

import time

import numpy as np
import pygame

from ..engine import GameState, StepResult
from . import hud
from .renderer import BoardRenderer
from .theme import DEFAULT_THEME, Fonts, Theme, vertical_gradient

_TOP, _BOTTOM, _MARGIN = 46, 34, 16


class EnvViewer:
    """Draws environment states into a window (``window=True``) or off-screen frames."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        window: bool = True,
        cell_size: int | None = None,
        title: str = "Snake",
        max_fps: float = 60.0,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        self.cols, self.rows = width, height
        self.window = window
        self.title = title
        self.max_fps = float(max_fps)
        self.theme = theme
        self.enabled = True
        self.speed_factor = 1.0
        self._cell = cell_size
        self._screen: pygame.Surface | None = None
        self._canvas: pygame.Surface | None = None
        self._backgrounds: dict[tuple[int, int], pygame.Surface] = {}
        self.fonts: Fonts | None = None
        self.renderer: BoardRenderer | None = None
        self._last_draw = -1e9
        self._last_pump = -1e9
        self._last_frame = time.perf_counter()
        self._showing_paused = False
        self._caption = ""

    # ----------------------------------------------------------------- set-up
    def _default_cell(self) -> int:
        if not self.window:
            return 24
        try:
            desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
        except (pygame.error, IndexError):
            desktop_w, desktop_h = 1280, 800
        avail_w = desktop_w * 0.8 - 2 * _MARGIN
        avail_h = desktop_h * 0.8 - _TOP - _BOTTOM - _MARGIN
        return int(max(6, min(30, avail_w / (self.cols + 0.6), avail_h / (self.rows + 0.6))))

    def _ensure_renderer(self) -> BoardRenderer:
        if self.window and not pygame.display.get_init():
            pygame.display.init()
        if self.renderer is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self.fonts = Fonts()
            cell = self._cell or self._default_cell()
            self.renderer = BoardRenderer(self.cols, self.rows, cell, self.theme, self.fonts)
        return self.renderer

    def _size(self, window: bool) -> tuple[int, int]:
        panel_w, panel_h = self._ensure_renderer().panel_size
        return (max(panel_w + 2 * _MARGIN, 360), _TOP + panel_h + (_BOTTOM if window else _MARGIN))

    def _open_window(self) -> pygame.Surface:
        if self._screen is None:
            self._ensure_renderer()
            pygame.display.set_caption(self.title)
            self._screen = pygame.display.set_mode(self._size(True))
        return self._screen

    def _background(self, size: tuple[int, int]) -> pygame.Surface:
        if size not in self._backgrounds:
            self._backgrounds[size] = vertical_gradient(
                size, self.theme.background_top, self.theme.background_bottom
            )
        return self._backgrounds[size]

    @property
    def is_open(self) -> bool:
        return self._screen is not None

    # ---------------------------------------------------------------- window
    def due(self) -> bool:
        """Whether enough time passed since the last frame to draw another one."""
        return time.perf_counter() - self._last_draw >= 1.0 / self.max_fps

    def pump(self) -> None:
        """Keep the window responsive and handle its keys (cheap; rate-limited)."""
        if self._screen is None:
            return
        now = time.perf_counter()
        if now - self._last_pump < 0.03:
            return
        self._last_pump = now
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.enabled = False
                self._close_window()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.enabled = not self.enabled
                    self._showing_paused = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed_factor = min(self.speed_factor * 2, 64.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed_factor = max(self.speed_factor / 2, 1 / 8)
        if not self.enabled and not self._showing_paused:
            self._draw_paused()

    def show(
        self,
        state: GameState,
        result: StepResult | None = None,
        *,
        duration: float = 0.0,
        caption: str = "",
    ) -> None:
        """Display ``state`` in the window; with ``duration`` > 0 the move is animated (blocking)."""
        if not self.window:
            return
        self.pump()
        if not self.enabled:
            return
        screen = self._open_window()
        renderer = self._ensure_renderer()
        self._caption = caption
        if state == renderer.state:  # e.g. an explicit render() after step(): just redraw
            duration = 0.0
        else:
            duration /= self.speed_factor
            if result is not None and result.done:
                duration = max(duration, 0.9 / self.speed_factor)  # let the ending play out
            renderer.set_state(state, result, animate=duration > 0, duration=max(duration * 0.9, 1e-3))
        end = time.perf_counter() + duration
        while True:
            now = time.perf_counter()
            renderer.update(min(now - self._last_frame, 0.1))
            self._last_frame = now
            self._compose(screen, state, hints=True)
            pygame.display.flip()
            self._last_draw = now
            if now >= end:
                break
            self.pump()
            if not self.enabled or self._screen is None:
                break
            time.sleep(min(1 / 60, max(end - time.perf_counter(), 0.0)))

    def frame(
        self,
        state: GameState,
        result: StepResult | None = None,
        *,
        caption: str = "",
        dt: float = 0.1,
    ) -> np.ndarray:
        """Render ``state`` off-screen and return it as an ``(H, W, 3)`` ``uint8`` array.

        Each call advances the animation clock by ``dt`` seconds (one video frame),
        so fruit spawns and eat/crash effects play out over consecutive frames.
        """
        renderer = self._ensure_renderer()
        if self._canvas is None:
            self._canvas = pygame.Surface(self._size(False))
        self._caption = caption
        if state != renderer.state:  # rendering the same state twice must not replay effects
            renderer.set_state(state, result, animate=False)
        renderer.update(dt)
        self._compose(self._canvas, state, hints=False)
        return np.ascontiguousarray(pygame.surfarray.array3d(self._canvas).transpose(1, 0, 2))

    def close(self) -> None:
        self._close_window()
        self.renderer = None
        self._canvas = None

    def _close_window(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None

    # ------------------------------------------------------------------ draw
    def _compose(self, surface: pygame.Surface, state: GameState, hints: bool) -> None:
        th, fonts, renderer = self.theme, self.fonts, self.renderer
        assert renderer is not None
        surface.blit(self._background(surface.get_size()), (0, 0))
        width = surface.get_width()
        panel_x = (width - renderer.panel_size[0]) // 2
        renderer.draw(surface, (panel_x, _TOP))
        y = _TOP // 2
        rect = hud.text(surface, fonts, f"Score {state.score}", 18, th.text, bold=True, midleft=(panel_x, y))
        detail = f"Length {state.length}  ·  Steps {state.steps}"
        hud.text(surface, fonts, detail, 13, th.text_dim, midleft=(rect.right + 14, y + 1))
        if self._caption:
            hud.text(
                surface,
                fonts,
                self._caption,
                13,
                th.text_faint,
                midright=(panel_x + renderer.panel_size[0], y + 1),
            )
        if hints:
            hud.key_hints(
                surface,
                fonts,
                th,
                (width // 2, _TOP + renderer.panel_size[1] + _BOTTOM // 2),
                width - 16,
                [("R", "rendering on/off"), ("+/-", f"speed x{self.speed_factor:g}")],
                10,
            )

    def _draw_paused(self) -> None:
        screen, renderer = self._screen, self.renderer
        if screen is None or renderer is None:
            return
        screen.blit(self._background(screen.get_size()), (0, 0))
        panel_x = (screen.get_width() - renderer.panel_size[0]) // 2
        area = pygame.Rect((panel_x, _TOP), renderer.panel_size)
        hud.overlay_card(
            screen,
            self.fonts,
            self.theme,
            area,
            "Rendering off",
            [("The environment keeps running at full speed.", self.theme.text_dim)],
            footer=[("R", "resume rendering")],
            scale=0.85,
        )
        pygame.display.flip()
        self._showing_paused = True
