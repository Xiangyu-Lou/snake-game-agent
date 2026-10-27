"""Command line: ``python -m snake_agent {evaluate,solve,reproduce,train,assess} --help``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from snake_game import Direction, GameConfig, SnakeGame

from . import reproduce, seeds
from .contract import NO_LIMIT, TRAINING_RULES
from .cycles import CycleEnvelope, ShortcutCyclePolicy
from .evaluation import (
    AgentSpec,
    best_rates,
    evaluate_recovery,
    evaluate_standard,
    format_summary,
    recovery_gate,
    standard_gate,
)
from .exact import evaluate_policy, solve
from .recovery import held_out_starts
from .references import REFERENCES

SPLITS = {"validation": seeds.VALIDATION, "test": seeds.TEST}


def board_size(text: str) -> tuple[int, int]:
    width, sep, height = text.lower().partition("x")
    try:
        return int(width), int(height if sep else width)
    except ValueError:
        raise argparse.ArgumentTypeError(f"board must look like 10x10, got {text!r}") from None


def load_spec(name: str, width: int, height: int) -> AgentSpec:
    """A reference by name, ``exact`` or ``exact-envelope`` (small boards), or ``module:attribute``.

    The attribute may be an :class:`AgentSpec`, a class (one instance per game)
    or a callable used for every game.
    """
    if name in REFERENCES:
        return REFERENCES[name]
    if name in ("exact", "exact-envelope"):
        if width * height > 16:
            raise ValueError(f"exact agents need a board of at most 16 cells, not {width}x{height}")
        envelope = None
        if name == "exact-envelope":
            envelope = CycleEnvelope.for_state(SnakeGame(GameConfig(width, height), seed=0).state())
            if envelope is None:
                raise ValueError(
                    f"the standard start of {width}x{height} is not aligned with a Hamiltonian cycle"
                )
        agent = solve(width, height, envelope=envelope).agent()
        return AgentSpec(name, lambda seed: agent, deterministic=True)
    module_name, sep, attribute = name.partition(":")
    if not sep:
        raise ValueError(
            f"unknown agent {name!r}: use {', '.join(sorted(REFERENCES))}, exact or module:attribute"
        )
    obj = importlib.import_module(module_name)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    if isinstance(obj, AgentSpec):
        return obj
    if isinstance(obj, type):
        return AgentSpec(attribute, lambda seed: obj())
    if callable(obj):
        return AgentSpec(attribute, lambda seed: obj)
    raise ValueError(f"{name!r} is neither an AgentSpec nor callable")


def evaluate(args: argparse.Namespace) -> int:
    width, height = args.board
    split = SPLITS[args.split]
    rules = NO_LIMIT if args.no_timeout else TRAINING_RULES
    spec = load_spec(args.agent, width, height)
    if args.track == "t1":
        evaluation = evaluate_standard(spec, width, height, seeds.standard_seeds(split, args.games), rules)
        print(format_summary(evaluation))
        gate = standard_gate(evaluation, args.max_mean_moves) if args.max_mean_moves else None
    else:
        starts = held_out_starts(width, height, args.sources, split)
        fruit_seeds = seeds.recovery_fruit_seeds(split, len(starts))
        evaluation = evaluate_recovery(spec, starts, fruit_seeds, rules)
        print(format_summary(evaluation))
        gate = None
        if args.compare:
            references = [
                evaluate_recovery(REFERENCES[name], starts, fruit_seeds, rules)
                for name in ("hamiltonian", "guarded-greedy", "greedy")
            ]
            for reference in references:
                print()
                print(format_summary(reference))
            gate = recovery_gate(evaluation, best_rates(references))
    if gate is not None:
        print()
        print("gate passed" if gate.passed else "gate failed: " + "; ".join(gate.reasons))
    if args.json:
        Path(args.json).write_text(json.dumps(evaluation.to_dict(), indent=2) + "\n")
    return 0 if gate is None or gate.passed else 1


def solve_board(args: argparse.Namespace) -> int:
    width, height = args.board
    if width * height > 16:
        print(f"{width}x{height} is too large to solve exactly here (at most 16 cells)", file=sys.stderr)
        return 2
    rules = "no fruit timeout" if args.no_timeout else "fruit timeout N + L"
    solution = solve(width, height, fruit_timeout=not args.no_timeout)
    print(f"{width}x{height}, {rules}: {solution.states:,} states with a fruit")
    print(f"  standard start: win probability {solution.start_value():.6f}")
    print(f"  fewest expected moves among certain wins: {solution.start_moves():.4f}")
    start = SnakeGame(GameConfig(width, height), seed=0).state()
    envelope = CycleEnvelope.for_state(start)
    if envelope is None:
        print("  no Hamiltonian cycle is aligned with the standard start: no envelope")
        return 0
    inside = solve(width, height, fruit_timeout=not args.no_timeout, envelope=envelope)
    print(f"  fewest expected moves inside the envelope: {inside.start_moves():.4f}")
    shortcut = ShortcutCyclePolicy()
    shortcut(start)
    for name, policy in (("shortcut cycle", shortcut), ("plain cycle", _plain(envelope))):
        result = evaluate_policy(width, height, policy)
        print(f"  {name}: win probability {result.win:.6f}, expected moves {result.moves:.4f}")
    return 0


def _plain(envelope: CycleEnvelope):
    """The plain cycle as a function of the board (for exact evaluation)."""

    def policy(state):
        x, y = envelope.cycle[state.head]
        return Direction.from_delta(x - state.head[0], y - state.head[1])

    return policy


def train_network(args: argparse.Namespace) -> int:
    from .network import NetworkConfig  # needs PyTorch
    from .train import TrainConfig, train

    width, height = args.board
    config = TrainConfig(
        width=width,
        height=height,
        generations=args.generations,
        games=args.games,
        free_share=args.free_share,
        prefix_share=args.prefix_share,
        prefix_fill=args.prefix_fill,
        simulations=args.simulations,
        beta=args.beta,
        moves_lambda=args.moves_lambda,
        sibling_visits=args.sibling_visits,
        replay=args.replay,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        network=NetworkConfig(args.channels, args.blocks, args.hidden),
        workers=args.workers,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_sources=args.eval_sources,
    )
    train(config, args.out, device=args.device or _default_device())
    return 0


def assess_network(args: argparse.Namespace) -> int:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from .assess import assess  # needs PyTorch
    from .network import load, to_bytes
    from .search import SearchConfig
    from .train import _init_worker

    model, meta = load(args.checkpoint)
    config = meta.get("config", {})
    width, height = args.board or (config.get("width", 6), config.get("height", 6))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=_init_worker) as pool:
        report = assess(
            pool,
            to_bytes(model),
            width,
            height,
            games=args.games,
            sources=args.sources,
            split=SPLITS[args.split],
            workers=args.workers,
            search=SearchConfig(simulations=args.simulations, gumbel_scale=0.0, moves_unit=args.moves_unit),
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    gates = [report[f"t1_{kind}"].get("gate") for kind in ("search",)]
    return 0 if all(gate is None or gate["passed"] for gate in gates) else 1


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m snake_agent", description="Snake agent tools (milestones M0 and M1)."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ev = commands.add_parser("evaluate", help="evaluate an agent on held-out games")
    ev.add_argument("track", choices=["t1", "t2"], help="t1: standard starts; t2: recovery starts")
    ev.add_argument(
        "--agent",
        default="shortcut",
        help=f"{', '.join(sorted(REFERENCES))}, exact, exact-envelope or module:attribute",
    )
    ev.add_argument("--board", type=board_size, default=(10, 10), help="board size, e.g. 10x10 (default)")
    ev.add_argument(
        "--split", choices=sorted(SPLITS), default="validation", help="held-out seeds (default: validation)"
    )
    ev.add_argument("--games", type=int, default=100, help="t1: games (default: 100)")
    ev.add_argument("--sources", type=int, default=60, help="t2: source games for the starts (default: 60)")
    ev.add_argument("--no-timeout", action="store_true", help="no fruit timeout (endless games are censored)")
    ev.add_argument("--max-mean-moves", type=float, help="t1: apply the gate with this reference mean")
    ev.add_argument("--compare", action="store_true", help="t2: also run the references and apply the gate")
    ev.add_argument("--json", help="write the results to this JSON file")
    ev.set_defaults(run=evaluate)

    so = commands.add_parser("solve", help="solve a small board exactly")
    so.add_argument("--board", type=board_size, default=(4, 4), help="board size (default: 4x4)")
    so.add_argument("--no-timeout", action="store_true", help="solve without the fruit timeout")
    so.set_defaults(run=solve_board)

    rp = commands.add_parser("reproduce", help="reproduce the Architecture B probe's reference numbers")
    rp.add_argument("--full", action="store_true", help="also run the slow sections (several minutes)")
    rp.set_defaults(run=lambda args: reproduce.main(["--full"] if args.full else []))

    tr = commands.add_parser("train", help="train the M1 network by self-play (needs PyTorch)")
    tr.add_argument("--board", type=board_size, default=(6, 6), help="board size (default: 6x6)")
    tr.add_argument("--out", required=True, help="output directory for the log and checkpoints")
    tr.add_argument("--generations", type=int, default=50)
    tr.add_argument("--games", type=int, default=256, help="self-play games per generation (default: 256)")
    tr.add_argument("--free-share", type=float, default=0.5, help="share of games from free starts")
    tr.add_argument(
        "--prefix-share", type=float, default=0.0, help="share of games after a random envelope prefix"
    )
    tr.add_argument(
        "--prefix-fill", type=float, default=0.75, help="the longest prefix, as a share of the board"
    )
    tr.add_argument("--simulations", type=int, default=16)
    tr.add_argument(
        "--beta", type=float, default=0.5, help="weight of the search's values in the win targets"
    )
    tr.add_argument(
        "--moves-lambda", type=float, default=0.5, help="lambda of the moves targets (1: moves still taken)"
    )
    tr.add_argument(
        "--sibling-visits",
        type=int,
        default=0,
        help="envelope moves the search visited this often but did not play give examples (0: none)",
    )
    tr.add_argument("--replay", type=int, default=300_000, help="examples kept for training")
    tr.add_argument("--steps", type=int, default=200, help="optimizer steps per generation")
    tr.add_argument("--batch", type=int, default=512)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--channels", type=int, default=64)
    tr.add_argument("--blocks", type=int, default=6)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--workers", type=int, default=8, help="self-play and evaluation processes")
    tr.add_argument("--seed", type=int, default=0, help="run seed (selects the training seeds)")
    tr.add_argument("--eval-every", type=int, default=10, help="assess every this many generations")
    tr.add_argument("--eval-games", type=int, default=100, help="T1 games per assessment")
    tr.add_argument("--eval-sources", type=int, default=60, help="T2 source games per assessment")
    tr.add_argument("--device", help="torch device for training (default: cuda if available, else cpu)")
    tr.set_defaults(run=train_network)

    asm = commands.add_parser("assess", help="assess a trained network against the M1 gates (needs PyTorch)")
    asm.add_argument("checkpoint", help="a model.pt written by train")
    asm.add_argument("--board", type=board_size, help="board size (default: the training board)")
    asm.add_argument("--games", type=int, default=300, help="T1 games (default: 300)")
    asm.add_argument("--sources", type=int, default=60, help="T2 source games (default: 60)")
    asm.add_argument(
        "--split", choices=sorted(SPLITS), default="validation", help="held-out seeds (default: validation)"
    )
    asm.add_argument(
        "--simulations", type=int, default=16, help="the search's simulations per move (default: 16)"
    )
    asm.add_argument(
        "--moves-unit", type=float, default=1.0, help="the search's unit for comparing moves (default: 1)"
    )
    asm.add_argument("--workers", type=int, default=8)
    asm.add_argument("--json", help="write the report to this JSON file")
    asm.set_defaults(run=assess_network)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, ImportError, AttributeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
