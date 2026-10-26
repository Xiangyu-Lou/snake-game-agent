"""HUD widgets: stat cards, chips, key hints, overlay cards and toasts."""

from __future__ import annotations

import pygame

from .theme import Color, Fonts, Theme, ease_out_cubic, mix, rounded_panel


def text(
    surface: pygame.Surface,
    fonts: Fonts,
    string: str,
    size: int,
    color: Color,
    *,
    bold: bool = False,
    alpha: int = 255,
    **anchor: tuple[float, float] | float,
) -> pygame.Rect:
    """Blit ``string``; ``anchor`` is one ``pygame.Rect`` attribute, e.g. ``midtop=(x, y)``."""
    rendered = fonts.render(string, size, color, bold)
    if alpha < 255:
        rendered = rendered.copy()
        rendered.set_alpha(alpha)
    rect = rendered.get_rect(**anchor)
    surface.blit(rendered, rect)
    return rect


def stat_card(
    surface: pygame.Surface,
    fonts: Fonts,
    theme: Theme,
    rect: pygame.Rect,
    label: str,
    value: str,
    value_color: Color | None = None,
) -> None:
    surface.blit(rounded_panel(rect.size, (*theme.panel, 235), theme.panel_border, radius=12), rect)
    scale = rect.height / 64
    text(
        surface,
        fonts,
        label,
        round(11 * scale),
        theme.text_faint,
        bold=True,
        topleft=(rect.x + round(14 * scale), rect.y + round(9 * scale)),
    )
    text(
        surface,
        fonts,
        value,
        round(27 * scale),
        value_color or theme.text,
        bold=True,
        bottomleft=(rect.x + round(13 * scale), rect.bottom - round(6 * scale)),
    )


def chip(
    surface: pygame.Surface,
    fonts: Fonts,
    theme: Theme,
    topleft: tuple[int, int],
    label: str,
    *,
    dot: Color | None = None,
    height: int = 28,
    size: int = 13,
) -> pygame.Rect:
    """Pill with an optional coloured status dot; returns its rect."""
    rendered = fonts.render(label, size, theme.text, False)
    pad = height // 2
    dot_space = height // 2 if dot else 0
    rect = pygame.Rect(topleft, (rendered.get_width() + pad * 2 + dot_space, height))
    surface.blit(rounded_panel(rect.size, theme.chip, theme.chip_border, radius=height // 2), rect)
    x = rect.x + pad
    if dot:
        pygame.draw.circle(surface, dot, (x + height // 8, rect.centery), max(3, height // 8))
        x += dot_space
    surface.blit(rendered, rendered.get_rect(midleft=(x, rect.centery)))
    return rect


def keycap(
    surface: pygame.Surface, fonts: Fonts, theme: Theme, topleft: tuple[int, int], key: str, size: int = 11
) -> pygame.Rect:
    rendered = fonts.render(key, size, theme.text, True)
    height = size + 10
    rect = pygame.Rect(topleft, (max(rendered.get_width() + 12, height), height))
    surface.blit(
        rounded_panel(rect.size, mix(theme.chip, theme.text_faint, 0.18), theme.chip_border, radius=6), rect
    )
    surface.blit(rendered, rendered.get_rect(center=rect.center))
    return rect


def hint_width(fonts: Fonts, key: str, label: str, size: int) -> int:
    key_w = max(fonts.render(key, size, (0, 0, 0), True).get_width() + 12, size + 10)
    return key_w + 6 + fonts.render(label, size + 1, (0, 0, 0)).get_width()


def key_hints(
    surface: pygame.Surface,
    fonts: Fonts,
    theme: Theme,
    center: tuple[int, int],
    max_width: int,
    hints: list[tuple[str, str]],
    size: int = 11,
) -> None:
    """A centred row of ``[key] label`` pairs; trailing hints that do not fit are dropped."""
    gap = 18
    widths = [hint_width(fonts, key, label, size) for key, label in hints]
    while hints and sum(widths) + gap * (len(widths) - 1) > max_width:
        hints, widths = hints[:-1], widths[:-1]
    if not hints:
        return
    x = center[0] - (sum(widths) + gap * (len(widths) - 1)) // 2
    for (key, label), width in zip(hints, widths, strict=True):
        cap = keycap(surface, fonts, theme, (x, center[1] - (size + 10) // 2), key, size)
        text(surface, fonts, label, size + 1, theme.text_dim, midleft=(cap.right + 6, cap.centery))
        x += width + gap


def dim(surface: pygame.Surface, rect: pygame.Rect, amount: float) -> None:
    """Darken ``rect`` (0 = unchanged, 1 = black)."""
    k = round(255 * (1 - amount))
    surface.fill((k, k, k), rect, special_flags=pygame.BLEND_RGB_MULT)


def overlay_card(
    surface: pygame.Surface,
    fonts: Fonts,
    theme: Theme,
    area: pygame.Rect,
    title: str,
    lines: list[tuple[str, Color]],
    *,
    appear: float = 1.0,
    title_color: Color | None = None,
    footer: list[tuple[str, str]] | None = None,
    scale: float = 1.0,
) -> None:
    """A centred card over ``area`` that fades and slides in as ``appear`` goes 0 -> 1."""
    k = ease_out_cubic(appear)
    if k <= 0.0:
        return
    dim(surface, area, 0.45 * k)
    title_size = round(34 * scale)
    line_size = round(16 * scale)
    line_gap = round(line_size * 1.55)
    width = min(area.width - 24, round(400 * scale))
    height = (
        round(title_size * 1.5)
        + line_gap * len(lines)
        + (round(48 * scale) if footer else 0)
        + round(36 * scale)
    )
    card = pygame.Rect(0, 0, width, height)
    card.center = (area.centerx, area.centery + round((1 - k) * 18))
    panel = rounded_panel(card.size, (12, 16, 33, 238), theme.panel_border, radius=round(18 * scale))
    panel.set_alpha(round(255 * k))
    surface.blit(panel, card)
    alpha = round(255 * k)
    y = card.y + round(22 * scale)
    rect = text(
        surface,
        fonts,
        title,
        title_size,
        title_color or theme.text,
        bold=True,
        alpha=alpha,
        midtop=(card.centerx, y),
    )
    y = rect.bottom + round(12 * scale)
    for string, color in lines:
        text(surface, fonts, string, line_size, color, alpha=alpha, midtop=(card.centerx, y))
        y += line_gap
    if footer and k > 0.6:
        key_hints(
            surface,
            fonts,
            theme,
            (card.centerx, card.bottom - round(28 * scale)),
            card.width - 20,
            footer,
            round(11 * scale),
        )


def toast(
    surface: pygame.Surface,
    fonts: Fonts,
    theme: Theme,
    midtop: tuple[int, int],
    message: str,
    alpha: float,
    size: int = 14,
) -> None:
    if alpha <= 0:
        return
    rendered = fonts.render(message, size, theme.text, True)
    rect = rendered.get_rect()
    rect.inflate_ip(size * 2, round(size * 1.1))
    rect.midtop = midtop
    panel = rounded_panel(rect.size, (14, 19, 38, 235), theme.panel_border, radius=rect.height // 2)
    panel.set_alpha(round(255 * alpha))
    surface.blit(panel, rect)
    shown = rendered.copy()
    shown.set_alpha(round(255 * alpha))
    surface.blit(shown, shown.get_rect(center=rect.center))
