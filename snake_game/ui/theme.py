"""Colours, fonts and small drawing helpers shared by the renderer and the app."""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import pygame

Color = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

FONT_FAMILIES = "inter,segoeui,helveticaneue,helvetica,sfprodisplay,roboto,arial,dejavusans"


@dataclass(frozen=True)
class Theme:
    background_top: Color = (15, 19, 38)
    background_bottom: Color = (7, 9, 20)
    board: Color = (21, 27, 51)
    board_alt: Color = (25, 32, 59)
    board_border: Color = (49, 62, 107)
    board_glow: Color = (72, 108, 255)
    snake_head: Color = (163, 250, 190)
    snake_body: tuple[Color, ...] = ((110, 231, 160), (45, 212, 191), (56, 152, 240), (99, 102, 241))
    snake_shadow: RGBA = (0, 0, 0, 105)
    snake_dead: Color = (100, 110, 132)
    eye_white: Color = (248, 250, 252)
    pupil: Color = (15, 23, 42)
    fruit: Color = (255, 92, 118)
    fruit_highlight: Color = (255, 200, 210)
    fruit_glow: Color = (255, 70, 110)
    leaf: Color = (74, 222, 128)
    stem: Color = (146, 95, 60)
    text: Color = (226, 232, 240)
    text_dim: Color = (148, 163, 184)
    text_faint: Color = (88, 102, 130)
    accent: Color = (134, 239, 172)
    danger: Color = (244, 63, 94)
    gold: Color = (250, 204, 21)
    panel: Color = (24, 31, 58)
    panel_border: Color = (56, 70, 116)
    chip: Color = (28, 36, 66)
    chip_border: Color = (48, 60, 100)


DEFAULT_THEME = Theme()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix(a: Color, b: Color, t: float) -> Color:
    t = min(max(t, 0.0), 1.0)
    return (round(lerp(a[0], b[0], t)), round(lerp(a[1], b[1], t)), round(lerp(a[2], b[2], t)))


def gradient(stops: tuple[Color, ...], t: float) -> Color:
    """Colour at ``t`` in ``[0, 1]`` along evenly spaced ``stops``."""
    if len(stops) == 1:
        return stops[0]
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return mix(stops[i], stops[i + 1], t - i)


def brighten(color: Color, amount: float) -> Color:
    """Move towards white (``amount`` > 0) or black (``amount`` < 0)."""
    return mix(color, (255, 255, 255), amount) if amount >= 0 else mix(color, (0, 0, 0), -amount)


def hue(h: float, s: float = 0.65, v: float = 1.0) -> Color:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))


def ease_out_cubic(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_out_back(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    c1 = 1.70158
    return 1 + (c1 + 1) * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_in_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


if hasattr(pygame.draw, "aacircle"):  # pygame-ce >= 2.5

    def aacircle(surface: pygame.Surface, color, center: tuple[float, float], radius: float) -> None:
        if radius >= 0.5:
            pygame.draw.aacircle(surface, color, center, radius)

else:  # classic pygame
    from pygame import gfxdraw

    def aacircle(surface: pygame.Surface, color, center: tuple[float, float], radius: float) -> None:
        x, y, r = round(center[0]), round(center[1]), round(radius)
        if r >= 1:
            gfxdraw.aacircle(surface, x, y, r, color)
            gfxdraw.filled_circle(surface, x, y, r, color)


class Fonts:
    """Lazily created fonts, with a small cache of rendered text."""

    def __init__(self, families: str = FONT_FAMILIES) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self._families = families
        self._fonts: dict[tuple[int, bool], pygame.font.Font] = {}
        self._cache: dict[tuple[str, int, bool, Color], pygame.Surface] = {}

    def font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._fonts:
            try:
                font = pygame.font.SysFont(self._families, size, bold=bold)
            except Exception:  # pragma: no cover - broken font configuration
                font = pygame.font.Font(None, size)
            self._fonts[key] = font
        return self._fonts[key]

    def render(self, text: str, size: int, color: Color, bold: bool = False) -> pygame.Surface:
        key = (text, size, bold, color)
        surface = self._cache.get(key)
        if surface is None:
            if len(self._cache) > 512:
                self._cache.clear()
            surface = self.font(size, bold).render(text, True, color)
            self._cache[key] = surface
        return surface


def vertical_gradient(size: tuple[int, int], top: Color, bottom: Color) -> pygame.Surface:
    width, height = size
    surface = pygame.Surface((max(width, 1), max(height, 1)))
    for y in range(height):
        surface.fill(mix(top, bottom, y / max(height - 1, 1)), (0, y, width, 1))
    return surface


def radial_glow(radius: int, color: Color, strength: float = 1.0) -> pygame.Surface:
    """Soft round glow: alpha falls off quadratically from the centre.

    Drawing on an SRCALPHA surface overwrites pixels, so the rings are drawn
    from the outside in and each ring's alpha is the final alpha at its radius.
    """
    size = radius * 2 + 1
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    steps = max(radius // 2, 8)
    for i in range(steps, 0, -1):
        alpha = round(255 * strength * (1 - i / steps) ** 2)
        pygame.draw.circle(surface, (*color, min(alpha, 255)), (radius, radius), radius * i / steps)
    return surface


def rounded_panel(
    size: tuple[int, int],
    fill: Color | RGBA,
    border: Color | RGBA | None = None,
    radius: int = 14,
    border_width: int = 1,
) -> pygame.Surface:
    surface = pygame.Surface(size, pygame.SRCALPHA)
    rect = surface.get_rect()
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surface, border, rect, width=border_width, border_radius=radius)
    return surface


def soft_shadow(size: tuple[int, int], radius: int, spread: int, color: Color, alpha: int) -> pygame.Surface:
    """A blurred-looking rounded rectangle (layered outlines) for glows and drop shadows."""
    width, height = size
    surface = pygame.Surface((width + spread * 2, height + spread * 2), pygame.SRCALPHA)
    for i in range(spread, 0, -1):  # outside in; inner layers overwrite outer ones
        a = round(alpha * (1 - i / spread) ** 2)
        rect = pygame.Rect(spread - i, spread - i, width + i * 2, height + i * 2)
        pygame.draw.rect(surface, (*color, min(a, 255)), rect, border_radius=radius + i)
    return surface
