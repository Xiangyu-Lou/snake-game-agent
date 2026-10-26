"""The interactive window: play Snake yourself or watch an agent play.

Run ``python -m snake_game`` (see ``--help``) or call :func:`play`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import pygame

from ..controller import UNLIMITED, GameController, GameMode, Policy
from ..engine import DeathCause, Direction, GameConfig, SnakeGame, Status, StepResult
from ..policies import GreedyPolicy
from . import hud
from .renderer import BoardRenderer
from .theme import DEFAULT_THEME, Fonts, Theme, aacircle, gradient, radial_glow, vertical_gradient

SPEEDS: tuple[float, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    80,
    100,
    150,
    200,
    300,
    500,
    UNLIMITED,
)
"""Auto-mode speed presets (moves per second) cycled with ``+`` / ``-``."""

KEY_DIRECTIONS = {
    pygame.K_UP: Direction.UP,
    pygame.K_w: Direction.UP,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_d: Direction.RIGHT,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_a: Direction.LEFT,
}

HELP = [
    ("Arrows / WASD", "steer - in step mode every press moves one cell"),
    ("Space", "step once (step mode) - pause / resume (auto mode)"),
    ("M", "switch between step mode and auto mode"),
    ("+ / -", "auto-mode speed (up to MAX)"),
    ("P", "switch player: you / agent"),
    ("R", "rendering on / off (the game keeps running)"),
    ("Enter", "new game"),
    ("H", "show / hide this help"),
    ("Esc", "quit"),
]

_FAST = 120.0  # above this many moves per second, frames skip the slide animation
_MARGIN, _HEADER, _FOOTER = 22, 76, 96  # layout at scale 1, in pixels
_CHROME_H = 2 * _MARGIN + _HEADER + _FOOTER  # window height not used by the board
_MIN_WIDTH = 640


@dataclass(frozen=True)
class _Layout:
    scale: float
    margin: int
    board: pygame.Rect
    left: int  # horizontal extent of the header and footer content
    right: int
    footer_y: int


def speed_label(speed: float) -> str:
    return "max speed" if math.isinf(speed) else f"{speed:g} moves/s"


class SnakeApp:
    """Animated Snake window driving a :class:`GameController`.

    Args:
        config: board size and initial length (default 20x20, length 3).
        mode: ``"step"`` (default; the snake moves only on input) or ``"auto"``.
        speed: auto-mode speed in moves per second (``UNLIMITED`` = max).
        seed: seed of the first game (later games continue the stream).
        agent: policy used when the agent plays (default: greedy demo policy).
        agent_name: label shown in the HUD.
        autopilot: start with the agent in control (default: when ``agent`` is given).
        render: start with rendering enabled (toggle with ``R``).
        cell_size: board cell size in pixels (default: fit the screen).
        auto_restart: start a new game automatically after the agent's game ends.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        mode: GameMode | str = GameMode.STEP,
        speed: float = 8.0,
        seed: int | None = None,
        agent: Policy | None = None,
        agent_name: str | None = None,
        autopilot: bool | None = None,
        render: bool = True,
        cell_size: int | None = None,
        fps: int = 60,
        auto_restart: bool = False,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        self.config = config or GameConfig()
        self.game = SnakeGame(self.config, seed=seed)
        self.agent: Policy = agent if agent is not None else GreedyPolicy()
        if agent_name is None:
            agent_name = "greedy" if agent is None else getattr(agent, "__name__", type(agent).__name__)
        self.agent_name = agent_name
        self.controller = GameController(self.game, mode, speed, on_step=self._on_step)
        self.autopilot = agent is not None if autopilot is None else autopilot
        self.render_enabled = render
        self.auto_restart = auto_restart
        self.fps = fps
        self.theme = theme
        self.cell_size = cell_size
        self.running = False
        self.frames = 0
        self.best = 0
        self.games = 0
        self.total_score = 0
        self.show_help = False
        self._help_paused = False
        self._waiting_start = False
        self._over_since: float | None = None
        self._new_best = False
        self._toast = ("", -10.0)
        self._score_pop = -10.0
        self._step_times: deque[tuple[float, int]] = deque(maxlen=64)
        self._moves = 0
        self._pending: StepResult | None = None
        self._last_status_draw = 0.0
        self._screen: pygame.Surface | None = None
        self.fonts: Fonts | None = None
        self.renderer: BoardRenderer | None = None
        self._layout: _Layout | None = None
        self._background: pygame.Surface | None = None

    # ---------------------------------------------------------------- settings
    @property
    def autopilot(self) -> bool:
        return self.controller.policy is not None

    @autopilot.setter
    def autopilot(self, enabled: bool) -> None:
        self.controller.policy = self.agent if enabled else None

    def set_mode(self, mode: GameMode | str) -> None:
        self.controller.mode = mode
        self._prepare_start()

    def change_speed(self, steps: int) -> None:
        """Move ``steps`` presets up (positive) or down the :data:`SPEEDS` list."""
        current = self.controller.speed
        if math.isinf(current):
            index = len(SPEEDS) - 1
        else:
            index = min(range(len(SPEEDS) - 1), key=lambda i: abs(SPEEDS[i] - current))
        speed = SPEEDS[min(max(index + steps, 0), len(SPEEDS) - 1)]
        self.controller.speed = speed
        self.notify(f"Speed: {speed_label(speed)}")

    def toggle_help(self) -> None:
        """Show or hide the controls; auto mode is paused while they are shown."""
        self.show_help = not self.show_help
        if self.show_help and self.controller.running:
            self.controller.paused = True
            self._help_paused = True
        elif not self.show_help and self._help_paused:
            self.controller.paused = False
            self._help_paused = False

    def toggle_render(self) -> None:
        self.render_enabled = not self.render_enabled
        if self.render_enabled:
            self._sync_renderer()
            self.notify("Rendering on")
        else:
            self._last_status_draw = 0.0

    def new_game(self, seed: int | None = None) -> None:
        self.controller.reset(seed)
        self._over_since = None
        self._new_best = False
        self._prepare_start()

    def notify(self, message: str) -> None:
        self._toast = (message, time.perf_counter())

    def _prepare_start(self) -> None:
        # A human in auto mode starts the clock with the first key press.
        self._waiting_start = (
            self.controller.mode is GameMode.AUTO and not self.autopilot and self.game.steps == 0
        )
        if self._waiting_start:
            self.controller.paused = True
        self._sync_renderer()

    # ------------------------------------------------------------------- loop
    def run(self, max_frames: int | None = None) -> None:
        """Open the window and run until it is closed (or for ``max_frames`` frames)."""
        self.open()
        clock = pygame.time.Clock()
        try:
            while self.running and (max_frames is None or self.frames < max_frames):
                dt = clock.tick(self.fps if self.render_enabled else 30) / 1000.0
                self.frame(dt)
        finally:
            self.close()

    def open(self) -> None:
        if not pygame.display.get_init():
            pygame.display.init()
        pygame.font.init()
        self.fonts = Fonts()
        size = self._initial_window_size()
        pygame.display.set_caption("Snake")
        pygame.display.set_icon(_icon())
        self._screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.key.set_repeat(170, 55)
        self.renderer = BoardRenderer(
            self.config.width, self.config.height, self._cell_for(size), self.theme, self.fonts
        )
        self._relayout()
        self.running = True
        self._prepare_start()

    def close(self) -> None:
        self.running = False
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None

    def frame(self, dt: float) -> None:
        """Process input, advance the game and draw one frame."""
        for event in pygame.event.get():
            self.handle_event(event)
        if not self.running:
            return
        self._simulate(dt)
        if self.render_enabled:
            self._draw(min(dt, 0.1))
        else:
            self._draw_render_off()
        self.frames += 1

    # ------------------------------------------------------------------ input
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self._relayout()
        elif event.type == pygame.KEYDOWN:
            self.handle_key(event.key)

    def handle_key(self, key: int) -> None:
        ctrl = self.controller
        if self.show_help and key not in (pygame.K_ESCAPE, pygame.K_q, pygame.K_h, pygame.K_F1):
            return  # the help card is modal
        if key in (pygame.K_ESCAPE, pygame.K_q):
            if self.show_help:
                self.toggle_help()
            else:
                self.running = False
        elif key in (pygame.K_h, pygame.K_F1):
            self.toggle_help()
        elif key == pygame.K_r:
            self.toggle_render()
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_n):
            self.new_game()
        elif key == pygame.K_m:
            self.set_mode(GameMode.AUTO if ctrl.mode is GameMode.STEP else GameMode.STEP)
            self.notify(
                "Auto mode - the snake moves by itself"
                if ctrl.mode is GameMode.AUTO
                else "Step mode - one move per input"
            )
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_RIGHTBRACKET):
            self.change_speed(+1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_LEFTBRACKET):
            self.change_speed(-1)
        elif key == pygame.K_p:
            self.autopilot = not self.autopilot
            self.notify(f"Agent playing: {self.agent_name}" if self.autopilot else "You are playing")
            if self._waiting_start and self.autopilot:
                self._waiting_start = False
                ctrl.paused = False
        elif key == pygame.K_SPACE:
            if self.game.done:
                self.new_game()
            elif ctrl.mode is GameMode.STEP:
                ctrl.advance()
            elif self._waiting_start:
                self._waiting_start = False
                ctrl.paused = False
            else:
                ctrl.toggle_pause()
        elif key in KEY_DIRECTIONS:
            if self.game.done:
                return
            if self.autopilot:
                self.autopilot = False
                self.notify("You are playing")
            ctrl.steer(KEY_DIRECTIONS[key])
            if self._waiting_start:
                self._waiting_start = False
                ctrl.paused = False

    # ------------------------------------------------------------- simulation
    def _simulate(self, dt: float) -> None:
        if self.game.done:
            if self.auto_restart and self.autopilot and self._over_since is not None:
                delay = 1.6 if self.render_enabled else 0.0
                if time.perf_counter() - self._over_since >= delay:
                    self.new_game()
            return
        if self.controller.mode is GameMode.AUTO:
            self.controller.update(dt, time_budget=0.008 if self.render_enabled else 0.03)

    def _on_step(self, result: StepResult) -> None:
        game = self.game
        self._moves += 1
        if result.ate_fruit:
            if game.score > self.best:
                self.best = game.score
                self._new_best = self.games > 0 or self._new_best
            self._score_pop = time.perf_counter()
        if result.done:
            self.games += 1
            self.total_score += game.score
            self._over_since = time.perf_counter()
        if not self.render_enabled or self.renderer is None:
            return
        speed = self.controller.speed
        fast = self.controller.mode is GameMode.AUTO and speed > _FAST
        if fast:
            # Too fast to animate every move: draw the latest state once per frame.
            if self._pending is None or not self._pending.done:
                self._pending = result
        else:
            auto = self.controller.mode is GameMode.AUTO
            self.renderer.set_state(game.state(), result, duration=self.controller.interval if auto else 0.13)

    def _sync_renderer(self) -> None:
        if self.renderer is not None and self.render_enabled:
            self.renderer.set_state(self.game.state(), animate=False)
            self._pending = None

    # ------------------------------------------------------------------ layout
    def _initial_window_size(self) -> tuple[int, int]:
        cols, rows = self.config.width, self.config.height
        if self.cell_size:
            cell = self.cell_size
        else:
            try:
                desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
            except (pygame.error, IndexError):
                desktop_w, desktop_h = 1280, 800
            avail_w = desktop_w * 0.9 - 2 * _MARGIN
            avail_h = desktop_h * 0.86 - _CHROME_H
            cell = int(max(6, min(36, avail_w / (cols + 0.6), avail_h / (rows + 0.6))))
        width = max(_panel_extent(cols, cell) + 2 * _MARGIN, _MIN_WIDTH)
        return width, _panel_extent(rows, cell) + round(_CHROME_H * self._scale(width))

    def _cell_for(self, size: tuple[int, int]) -> int:
        """Largest cell size whose board fits in a window of ``size``."""
        width, height = size
        s = self._scale(width)
        avail_w = width - 2 * round(_MARGIN * s)
        avail_h = height - round(_CHROME_H * s)
        cols, rows = self.config.width, self.config.height
        cell = int(min(avail_w / (cols + 0.6), avail_h / (rows + 0.6))) + 1
        while cell > 4 and (_panel_extent(cols, cell) > avail_w or _panel_extent(rows, cell) > avail_h):
            cell -= 1
        return max(cell, 4)

    @staticmethod
    def _scale(width: int) -> float:
        """UI scale for text and margins, from the window width."""
        return min(max(width / 720, 0.75), 1.3)

    def _relayout(self) -> None:
        if self._screen is None or self.renderer is None:
            return
        screen = self._screen = pygame.display.get_surface() or self._screen
        width, height = screen.get_size()
        s = self._scale(width)
        cell = self._cell_for((width, height))
        if cell != self.renderer.cell:
            self.renderer.resize(cell)
        panel_w, panel_h = self.renderer.panel_size
        margin = round(_MARGIN * s)
        top = margin + round(_HEADER * s)
        avail_h = height - round(_CHROME_H * s)
        board = pygame.Rect((width - panel_w) // 2, top + max(0, (avail_h - panel_h) // 2), panel_w, panel_h)
        self._layout = _Layout(
            scale=s,
            margin=margin,
            board=board,
            left=min(board.x, 2 * margin),
            right=max(board.right, width - 2 * margin),
            footer_y=board.bottom + round(16 * s),
        )
        bg = vertical_gradient((width, height), self.theme.background_top, self.theme.background_bottom)
        glow = radial_glow(max(width, height) // 2, (40, 60, 140), 0.18)
        bg.blit(glow, glow.get_rect(center=board.center))
        self._background = bg.convert()

    # -------------------------------------------------------------------- draw
    def _draw(self, dt: float) -> None:
        screen, renderer = self._screen, self.renderer
        assert screen is not None and renderer is not None and self._background is not None
        if self._pending is not None:
            renderer.set_state(
                self.game.state(), self._pending if self._pending.done else None, animate=False
            )
            self._pending = None
        screen.blit(self._background, (0, 0))
        renderer.update(dt)
        board = self._layout.board
        auto = self.controller.mode is GameMode.AUTO and self.controller.speed <= _FAST
        renderer.draw(screen, board.topleft, self.controller.progress if auto else None)
        self._draw_header()
        self._draw_overlays(board)
        self._draw_footer()
        pygame.display.flip()

    def _draw_header(self) -> None:
        screen, fonts, th, layout = self._screen, self.fonts, self.theme, self._layout
        s, left, right, top = layout.scale, layout.left, layout.right, layout.margin
        # logo: a tiny snake
        r = round(7 * s)
        for i in range(3):
            aacircle(
                screen,
                gradient(th.snake_body, i / 3),
                (left + r + i * r * 1.5, top + round(19 * s)),
                r * (1 - i * 0.12),
            )
        aacircle(screen, th.snake_head, (left + r, top + round(19 * s)), r)
        title_x = left + round(r * 5.6)
        hud.text(
            screen, fonts, "SNAKE", round(26 * s), th.text, bold=True, midleft=(title_x, top + round(19 * s))
        )
        cfg = self.config
        subtitle = f"{cfg.width} × {cfg.height} board  ·  fill it to win ({cfg.max_score} fruits)"
        hud.text(screen, fonts, subtitle, round(13 * s), th.text_faint, topleft=(left, top + round(40 * s)))

        card_h, card_w = round(58 * s), round(108 * s)
        best_rect = pygame.Rect(right - card_w, top, card_w, card_h)
        score_rect = pygame.Rect(best_rect.x - card_w - round(10 * s), top, card_w, card_h)
        pop = max(0.0, 1 - (time.perf_counter() - self._score_pop) / 0.35)
        score_color = th.accent if pop > 0 else th.text
        hud.stat_card(screen, fonts, th, score_rect, "SCORE", str(self.game.score), score_color)
        hud.stat_card(
            screen, fonts, th, best_rect, "BEST", str(self.best), th.gold if self._new_best else th.text
        )

    def _draw_footer(self) -> None:
        screen, fonts, th, ctrl = self._screen, self.fonts, self.theme, self.controller
        layout = self._layout
        s, y, left, right = layout.scale, layout.footer_y, layout.left, layout.right
        height, size = round(28 * s), round(13 * s)
        if ctrl.mode is GameMode.STEP:
            mode, dot = "Step mode", th.accent
        elif ctrl.paused or self._waiting_start:
            mode, dot = "Auto · paused", th.gold
        else:
            mode, dot = f"Auto · {speed_label(ctrl.speed)}", (96, 165, 250)
        x = left
        rect = hud.chip(screen, fonts, th, (x, y), mode, dot=dot, height=height, size=size)
        player = f"Agent: {self.agent_name}" if self.autopilot else "You"
        rect = hud.chip(
            screen,
            fonts,
            th,
            (rect.right + round(8 * s), y),
            player,
            dot=(192, 132, 252) if self.autopilot else th.accent,
            height=height,
            size=size,
        )
        render = "Render on" if self.render_enabled else "Render off"
        rect = hud.chip(
            screen,
            fonts,
            th,
            (rect.right + round(8 * s), y),
            render,
            dot=th.accent if self.render_enabled else th.text_faint,
            height=height,
            size=size,
        )
        stats = f"Length {self.game.length}  ·  Steps {self.game.steps}"
        if rect.right + fonts.render(stats, size, th.text_dim).get_width() + 16 < right:
            hud.text(screen, fonts, stats, size, th.text_dim, midright=(right, y + height // 2))
        hints = [
            ("Arrows", "move"),
            ("Space", "step" if ctrl.mode is GameMode.STEP else "pause"),
            ("M", "mode"),
            ("+/-", "speed"),
            ("P", "agent"),
            ("R", "render"),
            ("Enter", "new game"),
            ("H", "help"),
        ]
        hud.key_hints(
            screen,
            fonts,
            th,
            (screen.get_width() // 2, y + height + round(26 * s)),
            screen.get_width() - 24,
            hints,
            round(11 * s),
        )

    def _draw_overlays(self, board: pygame.Rect) -> None:
        screen, fonts, th, game = self._screen, self.fonts, self.theme, self.game
        s = self._layout.scale
        now = time.perf_counter()
        if self.show_help:
            self._draw_help(board)
        elif game.done and self._over_since is not None:
            appear = (now - self._over_since - (0.55 if game.status is Status.LOST else 0.9)) / 0.35
            footer = [("Enter", "play again"), ("M", "mode"), ("P", "agent" if not self.autopilot else "you")]
            if game.won:
                hud.overlay_card(
                    screen,
                    fonts,
                    th,
                    board,
                    "Perfect game!",
                    [
                        ("The snake fills the whole board.", th.text_dim),
                        (f"Score {game.score}  ·  {game.steps} moves", th.gold),
                    ],
                    appear=appear,
                    title_color=th.gold,
                    footer=footer,
                    scale=s,
                )
            else:
                cause = "Hit the wall" if game.death_cause is DeathCause.WALL else "Ran into itself"
                lines = [(cause, th.text_dim), (f"Score {game.score}  ·  Length {game.length}", th.text)]
                if self._new_best and game.score == self.best:
                    lines.append(("New best score!", th.gold))
                hud.overlay_card(
                    screen,
                    fonts,
                    th,
                    board,
                    "Game over",
                    lines,
                    appear=appear,
                    title_color=th.danger,
                    footer=footer,
                    scale=s,
                )
        elif self.controller.mode is GameMode.AUTO and self.controller.paused and not self._waiting_start:
            hud.overlay_card(
                screen,
                fonts,
                th,
                board,
                "Paused",
                [("The snake is waiting for you.", th.text_dim)],
                footer=[("Space", "resume")],
                scale=s,
            )
        elif game.steps == 0:
            if self._waiting_start:
                hint = "Press an arrow key to start"
            elif self.controller.mode is GameMode.STEP:
                hint = (
                    "Press Space to let the agent move"
                    if self.autopilot
                    else "Step mode: press an arrow key to move"
                )
            else:
                hint = ""
            if hint:
                pulse = 0.65 + 0.35 * math.sin(now * 3.2)
                hud.toast(
                    screen,
                    fonts,
                    th,
                    (board.centerx, board.bottom - round(64 * s)),
                    hint,
                    pulse,
                    round(15 * s),
                )
        message, since = self._toast
        age = now - since
        if age < 1.6:
            alpha = 1.0 if age < 1.1 else 1 - (age - 1.1) / 0.5
            hud.toast(
                screen, fonts, th, (board.centerx, board.y + round(16 * s)), message, alpha, round(14 * s)
            )

    def _draw_help(self, board: pygame.Rect) -> None:
        screen, fonts, th = self._screen, self.fonts, self.theme
        s = self._layout.scale
        hud.dim(screen, board, 0.55)
        size = round(13 * s)
        row = round(30 * s)
        key_w = max(hud.hint_width(fonts, key, "", size) for key, _ in HELP)
        text_w = max(fonts.render(desc, size + 1, th.text).get_width() for _, desc in HELP)
        card = pygame.Rect(
            0, 0, min(board.width - 16, key_w + text_w + round(64 * s)), row * len(HELP) + round(84 * s)
        )
        card.center = board.center
        screen.blit(
            hud.rounded_panel(card.size, (12, 16, 33, 240), th.panel_border, radius=round(18 * s)), card
        )
        hud.text(
            screen,
            fonts,
            "Controls",
            round(22 * s),
            th.text,
            bold=True,
            midtop=(card.centerx, card.y + round(16 * s)),
        )
        y = card.y + round(60 * s)
        for key, desc in HELP:
            cap = hud.keycap(screen, fonts, th, (card.x + round(24 * s), y), key, size)
            hud.text(
                screen,
                fonts,
                desc,
                size + 1,
                th.text_dim,
                midleft=(card.x + round(32 * s) + key_w, cap.centery),
            )
            y += row

    def _draw_render_off(self) -> None:
        """Minimal status screen, refreshed a few times per second."""
        now = time.perf_counter()
        self._step_times.append((now, self._moves))
        if now - self._last_status_draw < 0.25:
            return
        self._last_status_draw = now
        screen, fonts, th = self._screen, self.fonts, self.theme
        s, board = self._layout.scale, self._layout.board
        screen.blit(self._background, (0, 0))
        self._draw_header()
        screen.blit(
            hud.rounded_panel(board.size, (14, 18, 36, 255), th.board_border, radius=round(18 * s)), board
        )
        t0, m0 = self._step_times[0]
        rate = (self._moves - m0) / (now - t0) if now > t0 else 0.0
        mean = self.total_score / self.games if self.games else 0.0
        lines = [
            ("The game keeps running; nothing is drawn.", th.text_dim),
            (f"{rate:,.0f} moves/s", th.accent),
            (f"Score {self.game.score}  ·  Length {self.game.length}  ·  Steps {self.game.steps}", th.text),
            (f"Games {self.games}  ·  Best {self.best}  ·  Mean {mean:.1f}", th.text_dim),
        ]
        hud.overlay_card(
            screen,
            fonts,
            th,
            board,
            "Rendering off",
            lines,
            footer=[("R", "resume rendering"), ("+/-", "speed")],
            scale=s,
        )
        self._draw_footer()
        pygame.display.flip()


def _panel_extent(cells: int, cell: int) -> int:
    """Pixel size of the board panel along an axis of ``cells`` cells."""
    return cells * cell + 2 * BoardRenderer.padding_for(cell)


def _icon() -> pygame.Surface:
    icon = pygame.Surface((32, 32), pygame.SRCALPHA)
    th = DEFAULT_THEME
    for i, (x, y) in enumerate(((22, 22), (14, 22), (8, 16))):
        pygame.draw.circle(icon, gradient(th.snake_body, i / 3), (x, y), 7)
    pygame.draw.circle(icon, th.snake_head, (22, 22), 8)
    pygame.draw.circle(icon, th.fruit, (24, 8), 5)
    return icon


def play(
    width: int = 20,
    height: int = 20,
    *,
    initial_length: int = 3,
    mode: GameMode | str = GameMode.STEP,
    speed: float = 8.0,
    seed: int | None = None,
    agent: Policy | None = None,
    agent_name: str | None = None,
    render: bool = True,
    cell_size: int | None = None,
    auto_restart: bool = False,
) -> SnakeApp:
    """Open the game window and block until it is closed. Returns the finished app."""
    app = SnakeApp(
        GameConfig(width, height, initial_length),
        mode=mode,
        speed=speed,
        seed=seed,
        agent=agent,
        agent_name=agent_name,
        render=render,
        cell_size=cell_size,
        auto_restart=auto_restart,
    )
    app.run()
    return app
