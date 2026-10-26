"""Command line entry point: ``python -m snake_game --help``."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from .controller import MAX_SPEED, MIN_SPEED, UNLIMITED, GameMode, Policy
from .engine import GameConfig
from .policies import POLICIES


def load_agent(spec: str) -> tuple[Policy, str]:
    """Resolve ``--agent``: a built-in policy name or ``module:attribute``.

    A class found at ``module:attribute`` is instantiated without arguments; any
    other callable is used as the policy directly.
    """
    if spec in POLICIES:
        return POLICIES[spec](), spec
    module_name, sep, attribute = spec.partition(":")
    if not sep or not module_name or not attribute:
        raise ValueError(f"unknown agent {spec!r}: use one of {sorted(POLICIES)} or module:attribute")
    obj = importlib.import_module(module_name)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    if isinstance(obj, type):
        obj = obj()
    if not callable(obj):
        raise ValueError(f"{spec!r} is not callable")
    return obj, attribute.rsplit(".", 1)[-1]


def parse_speed(text: str) -> float:
    if text.lower() in ("max", "unlimited", "inf"):
        return UNLIMITED
    speed = float(text)
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise argparse.ArgumentTypeError(f"speed must be in [{MIN_SPEED:g}, {MAX_SPEED:g}] or 'max'")
    return speed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m snake_game",
        description="Play Snake, or watch an agent play it.",
        epilog="In the window press H for all controls.",
    )
    parser.add_argument("--width", type=int, default=20, help="board width in cells (default: 20)")
    parser.add_argument("--height", type=int, default=20, help="board height in cells (default: 20)")
    parser.add_argument(
        "--initial-length", type=int, default=3, help="starting length of the snake (default: 3)"
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in GameMode],
        default=GameMode.STEP.value,
        help="step: the snake moves only on input (default); auto: it moves by itself",
    )
    parser.add_argument(
        "--speed", type=parse_speed, default=8.0, help="auto-mode moves per second, or 'max' (default: 8)"
    )
    parser.add_argument("--seed", type=int, help="seed of the first game (fruits are random otherwise)")
    parser.add_argument(
        "--agent",
        help=f"let an agent play: {', '.join(sorted(POLICIES))} or module:attribute (a policy callable)",
    )
    parser.add_argument("--no-render", action="store_true", help="start with rendering off (toggle with R)")
    parser.add_argument(
        "--auto-restart", action="store_true", help="start a new game after the agent's game ends"
    )
    parser.add_argument("--cell-size", type=int, help="cell size in pixels (default: fit the screen)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = GameConfig(args.width, args.height, args.initial_length)
        agent, agent_name = load_agent(args.agent) if args.agent else (None, None)
    except (ValueError, TypeError, ImportError, AttributeError) as error:
        parser.error(str(error))
    try:
        from .ui import SnakeApp
    except ImportError as error:  # pragma: no cover - depends on the environment
        print(
            f"The game window needs pygame-ce ({error}).\n"
            "Install it with:  pip install -e '.[ui]'   (or: pip install pygame-ce)",
            file=sys.stderr,
        )
        return 1
    app = SnakeApp(
        config,
        mode=args.mode,
        speed=args.speed,
        seed=args.seed,
        agent=agent,
        agent_name=agent_name,
        render=not args.no_render,
        cell_size=args.cell_size,
        auto_restart=args.auto_restart,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
