"""Short-lived visual effects: particle bursts, confetti, pop-up text, shake, flash."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pygame

from .theme import Color, aacircle, ease_out_cubic, mix


@dataclass(slots=True)
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    radius: float
    color: Color
    gravity: float = 0.0
    drag: float = 2.5
    square: bool = False
    spin: float = 0.0
    angle: float = 0.0
    age: float = 0.0


@dataclass(slots=True)
class PopupText:
    surface: pygame.Surface
    x: float
    y: float
    life: float = 0.9
    rise: float = 34.0
    age: float = 0.0


@dataclass
class Effects:
    """Effects in board pixel coordinates; ``fade_to`` is the colour particles dissolve into."""

    fade_to: Color = (21, 27, 51)
    particles: list[Particle] = field(default_factory=list)
    popups: list[PopupText] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)
    _shake_amplitude: float = 0.0
    _shake_duration: float = 0.0
    _shake_age: float = 0.0
    _flash_color: Color = (0, 0, 0)
    _flash_strength: float = 0.0
    _flash_duration: float = 0.0
    _flash_age: float = 0.0

    def clear(self) -> None:
        self.particles.clear()
        self.popups.clear()
        self._shake_duration = 0.0
        self._flash_duration = 0.0

    # ---------------------------------------------------------------- spawning
    def burst(
        self,
        x: float,
        y: float,
        colors: tuple[Color, ...],
        count: int = 16,
        speed: tuple[float, float] = (60.0, 200.0),
        life: tuple[float, float] = (0.35, 0.75),
        radius: tuple[float, float] = (1.8, 4.2),
    ) -> None:
        rng = self.rng
        for _ in range(count):
            angle = rng.uniform(0, math.tau)
            v = rng.uniform(*speed)
            self.particles.append(
                Particle(
                    x,
                    y,
                    math.cos(angle) * v,
                    math.sin(angle) * v,
                    rng.uniform(*life),
                    rng.uniform(*radius),
                    rng.choice(colors),
                )
            )

    def confetti(self, width: float, count: int, colors: tuple[Color, ...], size: float) -> None:
        rng = self.rng
        for _ in range(count):
            self.particles.append(
                Particle(
                    rng.uniform(0, width),
                    rng.uniform(-40, -5),
                    rng.uniform(-40, 40),
                    rng.uniform(40, 160),
                    rng.uniform(2.2, 3.6),
                    rng.uniform(0.5, 1.0) * size,
                    rng.choice(colors),
                    gravity=110.0,
                    drag=0.4,
                    square=True,
                    spin=rng.uniform(-8, 8),
                    angle=rng.uniform(0, math.tau),
                )
            )

    def popup(self, surface: pygame.Surface, x: float, y: float) -> None:
        # Own copy: the alpha fade must not leak into a shared text cache.
        self.popups.append(PopupText(surface.copy(), x, y))

    def shake(self, amplitude: float, duration: float = 0.45) -> None:
        self._shake_amplitude = amplitude
        self._shake_duration = duration
        self._shake_age = 0.0

    def flash(self, color: Color, strength: float = 0.35, duration: float = 0.35) -> None:
        self._flash_color = color
        self._flash_strength = strength
        self._flash_duration = duration
        self._flash_age = 0.0

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        alive = []
        for p in self.particles:
            p.age += dt
            if p.age >= p.life:
                continue
            damping = math.exp(-p.drag * dt)
            p.vx *= damping
            p.vy = p.vy * damping + p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            alive.append(p)
        self.particles = alive
        for t in self.popups:
            t.age += dt
        self.popups = [t for t in self.popups if t.age < t.life]
        self._shake_age += dt
        self._flash_age += dt

    def shake_offset(self) -> tuple[float, float]:
        if self._shake_age >= self._shake_duration:
            return (0.0, 0.0)
        decay = (1 - self._shake_age / self._shake_duration) ** 2
        a = self._shake_amplitude * decay
        t = self._shake_age
        return (a * math.sin(t * 61.0), a * math.cos(t * 47.0))

    # -------------------------------------------------------------------- draw
    def draw_particles(self, surface: pygame.Surface, ox: float, oy: float) -> None:
        for p in self.particles:
            k = p.age / p.life
            if p.square:
                # confetti: a spinning square that flickers as it turns
                size = p.radius * (0.35 + 0.65 * abs(math.cos(p.angle)))
                color = mix(p.color, self.fade_to, max(0.0, k - 0.7) / 0.3)
                rect = pygame.Rect(0, 0, max(1, round(size)), max(1, round(p.radius)))
                rect.center = (round(ox + p.x), round(oy + p.y))
                surface.fill(color, rect)
            else:
                radius = p.radius * (1 - ease_out_cubic(k) * 0.85)
                aacircle(surface, mix(p.color, self.fade_to, k * k), (ox + p.x, oy + p.y), radius)

    def draw_popups(self, surface: pygame.Surface, ox: float, oy: float) -> None:
        for t in self.popups:
            k = t.age / t.life
            t.surface.set_alpha(round(255 * (1 - k * k)))
            rect = t.surface.get_rect(center=(round(ox + t.x), round(oy + t.y - t.rise * ease_out_cubic(k))))
            surface.blit(t.surface, rect)

    def draw_flash(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        if self._flash_age >= self._flash_duration:
            return
        k = (1 - self._flash_age / self._flash_duration) ** 2 * self._flash_strength
        r, g, b = self._flash_color
        surface.fill((round(r * k), round(g * k), round(b * k)), rect, special_flags=pygame.BLEND_RGB_ADD)
