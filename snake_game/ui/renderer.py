"""Animated drawing of a board: panel, snake, fruit and effects.

:class:`BoardRenderer` is fed :class:`~snake_game.engine.GameState` snapshots
and draws them with smooth motion: between two consecutive states the head
slides into its new cell while the tail slides out of the vacated one, so the
snake moves continuously at any speed. It is used by the interactive app, the
environments' ``"human"`` window and their ``"rgb_array"`` frames.
"""

from __future__ import annotations

import math
import random

import pygame

from ..engine import DELTAS, GameState, Position, Status, StepResult
from .effects import Effects
from .theme import (
    DEFAULT_THEME,
    Color,
    Fonts,
    Theme,
    aacircle,
    brighten,
    ease_out_back,
    ease_out_cubic,
    gradient,
    hue,
    lerp,
    mix,
    radial_glow,
    soft_shadow,
)

_PALETTE_SIZE = 128


def is_successor(prev: GameState, state: GameState) -> bool:
    """Whether ``state`` follows ``prev`` by exactly one move of the snake."""
    if state.steps != prev.steps + 1 or (state.width, state.height) != (prev.width, prev.height):
        return False
    s, p = state.snake, prev.snake
    if len(s) == len(p):
        return s[1:] == p[:-1]
    if len(s) == len(p) + 1:
        return s[1:] == p
    return False


class BoardRenderer:
    """Draws one board. Call :meth:`set_state` on every move, :meth:`update`
    every frame with the elapsed time, then :meth:`draw`."""

    def __init__(
        self,
        width: int,
        height: int,
        cell: int,
        theme: Theme = DEFAULT_THEME,
        fonts: Fonts | None = None,
    ) -> None:
        self.cols, self.rows = width, height
        self.theme = theme
        self.fonts = fonts or Fonts()
        self.effects = Effects(fade_to=theme.board)
        self.time = 0.0
        self.state: GameState | None = None
        self.prev: GameState | None = None
        self.effects_enabled = True
        self._anim = 1.0
        self._anim_duration = 0.12
        self._fruit: Position | None = None
        self._fruit_since = -10.0
        self._death_since: float | None = None
        self._win_since: float | None = None
        self._next_confetti = 0.0
        self._bulges: list[tuple[Position, int]] = []
        self._index: dict[Position, int] = {}
        self._rng = random.Random(7)
        self._next_blink = 2.5
        self._blink_until = 0.0
        self._next_tongue = 1.5
        self._tongue_since = -10.0
        self._palette = [gradient(theme.snake_body, i / (_PALETTE_SIZE - 1)) for i in range(_PALETTE_SIZE)]
        self._fruit_sprites: dict[int, pygame.Surface] = {}
        self.resize(cell)

    # ---------------------------------------------------------------- geometry
    @staticmethod
    def padding_for(cell: int) -> int:
        """Width of the panel's border around the cells, for a given cell size."""
        return max(4, round(cell * 0.3))

    def resize(self, cell: int) -> None:
        self.cell = max(4, int(cell))
        self.pad = self.padding_for(self.cell)
        self.radius = max(6, round(self.cell * 0.45))
        self.board_size = (self.cols * self.cell, self.rows * self.cell)
        self.panel_size = (self.board_size[0] + 2 * self.pad, self.board_size[1] + 2 * self.pad)
        self._glow_spread = max(10, self.cell)
        self._panel = self._build_panel()
        self._glow = soft_shadow(self.panel_size, self.radius, self._glow_spread, self.theme.board_glow, 34)
        self._fruit_glow = radial_glow(round(self.cell * 1.15), self.theme.fruit_glow, 0.5)
        self._shadow_layer = pygame.Surface(self.board_size, pygame.SRCALPHA)
        self._fruit_master = self._build_fruit_master()
        self._fruit_sprites.clear()

    def panel_rect(self, topleft: tuple[int, int]) -> pygame.Rect:
        return pygame.Rect(topleft, self.panel_size)

    def _build_panel(self) -> pygame.Surface:
        th, cell, pad = self.theme, self.cell, self.pad
        panel = pygame.Surface(self.panel_size, pygame.SRCALPHA)
        rect = panel.get_rect()
        pygame.draw.rect(panel, th.board, rect, border_radius=self.radius)
        checker = pygame.Surface(self.panel_size, pygame.SRCALPHA)
        for y in range(self.rows):
            for x in range((y % 2), self.cols, 2):
                checker.fill(th.board_alt, (pad + x * cell, pad + y * cell, cell, cell))
        mask = pygame.Surface(self.panel_size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), rect, border_radius=max(2, self.radius - 2))
        checker.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        panel.blit(checker, (0, 0))
        pygame.draw.rect(panel, th.board_border, rect, width=2, border_radius=self.radius)
        return panel

    def _build_fruit_master(self) -> pygame.Surface:
        """Supersampled fruit sprite (body, highlight, stem, leaf), scaled down when drawn."""
        th = self.theme
        size = 192
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy, r = size * 0.5, size * 0.56, size * 0.36
        pygame.draw.circle(s, mix(th.fruit, (0, 0, 0), 0.25), (cx, cy + r * 0.06), r)
        pygame.draw.circle(s, th.fruit, (cx, cy), r * 0.97)
        pygame.draw.circle(s, brighten(th.fruit, 0.18), (cx - r * 0.12, cy - r * 0.14), r * 0.7)
        pygame.draw.circle(s, th.fruit_highlight, (cx - r * 0.38, cy - r * 0.4), r * 0.2)
        stem = [
            (cx - r * 0.06, cy - r * 0.9),
            (cx + r * 0.08, cy - r * 0.9),
            (cx + r * 0.2, cy - r * 1.32),
            (cx + r * 0.06, cy - r * 1.32),
        ]
        pygame.draw.polygon(s, th.stem, stem)
        leaf = []
        angle = math.radians(-28)
        lx, ly, a, b = cx + r * 0.5, cy - r * 1.1, r * 0.46, r * 0.2
        for i in range(24):
            t = math.tau * i / 24
            ex, ey = a * math.cos(t), b * math.sin(t)
            leaf.append(
                (
                    lx + ex * math.cos(angle) - ey * math.sin(angle),
                    ly + ex * math.sin(angle) + ey * math.cos(angle),
                )
            )
        pygame.draw.polygon(s, th.leaf, leaf)
        return s

    def _fruit_sprite(self, diameter: int) -> pygame.Surface:
        sprite = self._fruit_sprites.get(diameter)
        if sprite is None:
            sprite = pygame.transform.smoothscale(self._fruit_master, (diameter, diameter))
            if len(self._fruit_sprites) > 64:
                self._fruit_sprites.clear()
            self._fruit_sprites[diameter] = sprite
        return sprite

    # ------------------------------------------------------------------ states
    def set_state(
        self,
        state: GameState,
        result: StepResult | None = None,
        *,
        animate: bool = True,
        duration: float = 0.12,
    ) -> None:
        """Show ``state``; with ``animate``, slide from the previous one over ``duration`` s."""
        prev = self.state
        new_game = prev is None or state.steps < prev.steps or state.steps == 0
        if new_game:
            self.effects.clear()
            self._bulges.clear()
            self._death_since = None
            self._win_since = None
        self.prev = prev if animate and prev is not None and is_successor(prev, state) else None
        self.state = state
        self._anim = 0.0 if self.prev is not None else 1.0
        self._anim_duration = max(duration, 1e-3)
        self._index = {pos: i for i, pos in enumerate(state.snake)}
        if state.fruit != self._fruit:
            self._fruit = state.fruit
            self._fruit_since = self.time
        self._bulges = [(pos, step) for pos, step in self._bulges if pos in self._index]
        if result is not None:
            self._on_result(state, result)
        elif state.status is Status.LOST and self._death_since is None:
            self._death_since = self.time - 10.0  # already over: no animation
        elif state.status is Status.WON and self._win_since is None:
            self._win_since = self.time

    def _on_result(self, state: GameState, result: StepResult) -> None:
        th, cell = self.theme, self.cell
        hx, hy = state.head
        cx, cy = (hx + 0.5) * cell, (hy + 0.5) * cell
        if result.ate_fruit:
            self._bulges.append((state.head, state.steps))
            if self.effects_enabled:
                self.effects.burst(
                    cx,
                    cy,
                    (th.fruit, brighten(th.fruit, 0.4), th.gold, (255, 255, 255)),
                    count=18,
                    speed=(cell * 2.0, cell * 6.5),
                )
                self.effects.popup(
                    self.fonts.render("+1", max(12, round(cell * 0.62)), th.accent, bold=True),
                    cx,
                    cy - cell * 0.4,
                )
        if result.lost:
            self._death_since = self.time
            if self.effects_enabled:
                self.effects.shake(cell * 0.32, 0.5)
                self.effects.flash(th.danger, 0.32, 0.45)
                dx, dy = DELTAS[result.direction]
                self.effects.burst(
                    cx + dx * cell * 0.45,
                    cy + dy * cell * 0.45,
                    (th.danger, (255, 255, 255), th.snake_dead),
                    count=22,
                    speed=(cell * 1.5, cell * 5.0),
                )
        if result.won:
            self._win_since = self.time
            self._next_confetti = self.time
            if self.effects_enabled:
                self.effects.flash(th.gold, 0.3, 0.6)

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        self.time += dt
        if self._anim < 1.0:
            self._anim = min(1.0, self._anim + dt / self._anim_duration)
        if self._win_since is not None and self.effects_enabled and self.time >= self._next_confetti:
            if self.time - self._win_since < 4.0:
                colors = tuple(hue(i / 7, 0.6, 1.0) for i in range(7))
                self.effects.confetti(self.board_size[0], 26, colors, self.cell * 0.32)
                self._next_confetti = self.time + 0.35
        self.effects.update(dt)
        if self.time >= self._next_blink:
            self._blink_until = self.time + 0.13
            self._next_blink = self.time + self._rng.uniform(2.5, 5.5)
        if self.time >= self._next_tongue:
            self._tongue_since = self.time
            self._next_tongue = self.time + self._rng.uniform(2.0, 4.5)

    # -------------------------------------------------------------------- draw
    def draw(self, surface: pygame.Surface, topleft: tuple[int, int], progress: float | None = None) -> None:
        """Draw the panel at ``topleft``. ``progress`` (0..1) overrides the move animation."""
        x0, y0 = topleft
        surface.blit(self._glow, (x0 - self._glow_spread, y0 - self._glow_spread + 2))
        sx, sy = self.effects.shake_offset()
        px, py = round(x0 + sx), round(y0 + sy)
        surface.blit(self._panel, (px, py))
        if self.state is None:
            return
        ox, oy = px + self.pad, py + self.pad
        board = pygame.Rect(ox, oy, *self.board_size)
        p = ease_out_cubic(self._anim) if progress is None else min(max(progress, 0.0), 1.0)
        if self.prev is None:
            p = 1.0
        clip = surface.get_clip()
        surface.set_clip(pygame.Rect((px, py), self.panel_size).clip(clip))
        self._draw_fruit(surface, ox, oy)
        self._draw_snake(surface, ox, oy, p)
        if self.effects_enabled:
            self.effects.draw_particles(surface, ox, oy)
            self.effects.draw_popups(surface, ox, oy)
            self.effects.draw_flash(surface, board)
        surface.set_clip(clip)

    def _center(self, ox: float, oy: float, pos: Position) -> list[float]:
        c = self.cell
        return [ox + (pos[0] + 0.5) * c, oy + (pos[1] + 0.5) * c]

    def _draw_fruit(self, surface: pygame.Surface, ox: float, oy: float) -> None:
        fruit = self.state.fruit if self.state is not None else None
        if fruit is None:
            return
        cx, cy = self._center(ox, oy, fruit)
        age = self.time - self._fruit_since
        appear = ease_out_back(age / 0.4)
        pulse = 1.0 + 0.055 * math.sin(self.time * math.tau * 1.1)
        glow = self._fruit_glow
        glow.set_alpha(round(255 * min(1.0, max(appear, 0.0)) * (0.75 + 0.25 * pulse)))
        surface.blit(glow, glow.get_rect(center=(round(cx), round(cy))))
        diameter = round(self.cell * 1.02 * max(appear, 0.0) * pulse)
        if diameter >= 2:
            sprite = self._fruit_sprite(diameter)
            surface.blit(sprite, sprite.get_rect(center=(round(cx), round(cy - self.cell * 0.04))))

    def _snake_points(self, ox: float, oy: float, p: float) -> list[list[float]]:
        state, prev = self.state, self.prev
        pts = [self._center(ox, oy, pos) for pos in state.snake]
        if prev is not None and p < 1.0:
            hx0, hy0 = self._center(ox, oy, prev.snake[0])
            pts[0] = [lerp(hx0, pts[0][0], p), lerp(hy0, pts[0][1], p)]
            if len(state.snake) == len(prev.snake):  # the tail slides out of the vacated cell
                tx0, ty0 = self._center(ox, oy, prev.snake[-1])
                tx1, ty1 = self._center(ox, oy, state.snake[-1])
                pts.append([lerp(tx0, tx1, p), lerp(ty0, ty1, p)])
        if state.status is Status.LOST and self._death_since is not None:
            k = (self.time - self._death_since) / 0.3
            if 0.0 <= k < 1.0:  # bump into the obstacle and recoil
                dx, dy = DELTAS[state.direction]
                bump = math.sin(math.pi * k) * self.cell * 0.34
                pts[0] = [pts[0][0] + dx * bump, pts[0][1] + dy * bump]
        return pts

    def _colors(self, fractions: list[float]) -> list[Color]:
        state, th = self.state, self.theme
        last = _PALETTE_SIZE - 1
        if self._win_since is not None:
            shift = (self.time - self._win_since) * 0.45
            return [hue(0.36 + f * 0.85 - shift, 0.5, 1.0) for f in fractions]
        colors = [self._palette[min(last, int(f * last))] for f in fractions]
        if state.status is Status.LOST and self._death_since is not None:
            elapsed = (self.time - self._death_since) * 2.6
            colors = [
                mix(c, th.snake_dead, min(max(elapsed - f * 1.4, 0.0), 1.0) * 0.85)
                for c, f in zip(colors, fractions, strict=True)
            ]
        return colors

    def _draw_snake(self, surface: pygame.Surface, ox: float, oy: float, p: float) -> None:
        th, cell = self.theme, self.cell
        pts = self._snake_points(ox, oy, p)
        n = len(pts)
        arc = [0.0] * n
        for i in range(1, n):
            arc[i] = arc[i - 1] + abs(pts[i][0] - pts[i - 1][0]) + abs(pts[i][1] - pts[i - 1][1])
        total = max(arc[-1], 1e-6)
        # The palette spans at least 16 cells: a young snake is green, blues appear as it grows.
        span = max(total, cell * 16)
        colors = self._colors([a / span for a in arc])
        r_body = cell * 0.35
        taper = min(cell * 1.8, total * 0.45)
        taper_start = total - taper

        def radius_at(s: float) -> float:
            if s <= taper_start:
                return r_body
            u = min((s - taper_start) / max(taper, 1e-6), 1.0)
            return r_body * (1.0 - 0.64 * u**1.6)  # gradual point

        # Drop shadow on its own layer (alpha does not blend when drawing).
        layer = self._shadow_layer
        layer.fill((0, 0, 0, 0))
        shadow = th.snake_shadow
        self._draw_tube(layer, pts, arc, [shadow] * n, -ox, -oy, radius_at, taper_start, coarse=True)
        aacircle(layer, shadow, (pts[0][0] - ox, pts[0][1] - oy), cell * 0.46)
        surface.blit(layer, (ox + round(cell * 0.05), oy + round(cell * 0.14)))

        # Darker full-width tube, then the body colour slightly narrower and offset towards
        # the light: the shaded rim left on the bottom-right side makes the tube look round.
        shade = [mix(c, (0, 0, 0), 0.3) for c in colors]
        self._draw_tube(surface, pts, arc, shade, 0.0, 0.0, radius_at, taper_start)
        off = -cell * 0.035

        def inner_radius(s: float) -> float:
            return radius_at(s) * 0.84

        self._draw_tube(surface, pts, arc, colors, off, off, inner_radius, taper_start)
        self._draw_bulges(surface, ox, oy, pts, colors, r_body)
        self._draw_head(surface, ox, oy, pts[0], colors[0])

    def _draw_tube(self, surface, pts, arc, colors, dx, dy, radius_at, taper_start, coarse=False) -> None:
        """Straight links are rectangles, corners circles, the tapered tail sampled circles.

        ``coarse`` merges collinear links and uses one colour per run (for the shadow).
        """
        n = len(pts)
        r_body = radius_at(0.0)
        i = 0
        while i < n - 1:
            j = i + 1
            # Merge straight runs that share a colour (always true for the shadow).
            while j < n - 1 and (coarse or colors[j] == colors[i]) and _collinear(pts[i], pts[j], pts[j + 1]):
                j += 1
            a, b = pts[i], pts[j]
            s0, s1 = arc[i], arc[j]
            end = min(s1, taper_start)
            if end - s0 > 0.25:
                t = (end - s0) / (s1 - s0)
                bx, by = lerp(a[0], b[0], t), lerp(a[1], b[1], t)
                _link_rect(surface, colors[i], a[0] + dx, a[1] + dy, bx + dx, by + dy, r_body)
            if s1 > taper_start and s1 - s0 > 1e-6:
                start = max(s0, taper_start)
                step = max(1.25, r_body * (0.6 if coarse else 0.3))
                s = start
                while s <= s1 + 1e-6:
                    t = (s - s0) / (s1 - s0)
                    color = colors[i] if coarse else mix(colors[i], colors[j], t)
                    aacircle(
                        surface, color, (lerp(a[0], b[0], t) + dx, lerp(a[1], b[1], t) + dy), radius_at(s)
                    )
                    s += step
            if 0 < j < n - 1 and arc[j] <= taper_start and not _collinear(pts[j - 1], pts[j], pts[j + 1]):
                aacircle(surface, colors[j], (pts[j][0] + dx, pts[j][1] + dy), r_body)
            i = j

    def _draw_bulges(self, surface, ox, oy, pts, colors, r_body) -> None:
        state = self.state
        for pos, step in self._bulges:
            idx = self._index.get(pos)
            if idx is None or idx == 0 or idx >= len(pts):
                continue
            digested = min((state.steps - step) / max(state.length, 1), 1.0)
            radius = r_body * (1.24 - 0.2 * digested)
            aacircle(surface, brighten(colors[idx], 0.12), pts[idx], radius)

    def _draw_head(self, surface, ox, oy, head: list[float], color: Color) -> None:
        state, th, cell = self.state, self.theme, self.cell
        hx, hy = head
        fx, fy = DELTAS[state.direction]
        nx, ny = -fy, fx
        r = cell * 0.46
        dead = state.status is Status.LOST
        if not dead and self._win_since is None:
            color = mix(th.snake_head, color, 0.25)
        tongue_k = (self.time - self._tongue_since) / 0.32
        if 0.0 <= tongue_k < 1.0 and not dead:
            length = r * 0.95 * math.sin(math.pi * tongue_k)
            base = (hx + fx * r * 0.8, hy + fy * r * 0.8)
            tip = (base[0] + fx * length, base[1] + fy * length)
            width = max(1, round(cell * 0.06))
            pygame.draw.line(surface, th.danger, base, tip, width)
            for side in (-1, 1):
                fork = (
                    tip[0] + (fx * 0.6 + nx * side * 0.5) * r * 0.28,
                    tip[1] + (fy * 0.6 + ny * side * 0.5) * r * 0.28,
                )
                pygame.draw.line(surface, th.danger, tip, fork, width)
        aacircle(surface, mix(color, (0, 0, 0), 0.3), (hx, hy), r)
        aacircle(surface, color, (hx - r * 0.03, hy - r * 0.05), r * 0.9)
        aacircle(surface, brighten(color, 0.3), (hx - r * 0.3, hy - r * 0.34), r * 0.3)

        look_x, look_y = fx, fy
        fruit = state.fruit
        if fruit is not None and not dead:
            tx, ty = self._center(ox, oy, fruit)
            vx, vy = tx - hx, ty - hy
            dist = math.hypot(vx, vy)
            if dist > 1e-6:
                look_x, look_y = vx / dist, vy / dist
        blink = self.time < self._blink_until
        eye_r, pupil_r = cell * 0.135, cell * 0.07
        for side in (-1, 1):
            ex = hx + fx * cell * 0.1 + nx * side * cell * 0.19
            ey = hy + fy * cell * 0.1 + ny * side * cell * 0.19
            if dead:
                arm = eye_r * 0.85
                width = max(1, round(cell * 0.055))
                pygame.draw.line(surface, th.pupil, (ex - arm, ey - arm), (ex + arm, ey + arm), width)
                pygame.draw.line(surface, th.pupil, (ex - arm, ey + arm), (ex + arm, ey - arm), width)
            elif blink:
                width = max(1, round(cell * 0.05))
                pygame.draw.line(
                    surface,
                    th.pupil,
                    (ex - fx * eye_r, ey - fy * eye_r),
                    (ex + fx * eye_r, ey + fy * eye_r),
                    width,
                )
            else:
                aacircle(surface, th.eye_white, (ex, ey), eye_r)
                aacircle(surface, th.pupil, (ex + look_x * eye_r * 0.42, ey + look_y * eye_r * 0.42), pupil_r)
                aacircle(
                    surface,
                    th.eye_white,
                    (
                        ex + look_x * eye_r * 0.42 - pupil_r * 0.35,
                        ey + look_y * eye_r * 0.42 - pupil_r * 0.35,
                    ),
                    pupil_r * 0.32,
                )


def _collinear(a, b, c) -> bool:
    return (abs(a[0] - b[0]) < 1e-6 and abs(b[0] - c[0]) < 1e-6) or (
        abs(a[1] - b[1]) < 1e-6 and abs(b[1] - c[1]) < 1e-6
    )


def _link_rect(surface: pygame.Surface, color, ax: float, ay: float, bx: float, by: float, r: float) -> None:
    """Axis-aligned link of half-width ``r`` from ``a`` to ``b``."""
    if abs(ay - by) < 1e-6:  # horizontal
        left, right = min(ax, bx), max(ax, bx)
        rect = pygame.Rect(round(left), round(ay - r), round(right) - round(left) + 1, round(2 * r))
    else:
        top, bottom = min(ay, by), max(ay, by)
        rect = pygame.Rect(round(ax - r), round(top), round(2 * r), round(bottom) - round(top) + 1)
    surface.fill(color, rect)
