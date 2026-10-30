"""Command line: ``python -m snake_agent {evaluate,solve,reproduce,train,assess,habits} --help``."""

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
from .habits import CANDIDATES
from .recovery import held_out_starts
from .references import REFERENCES

SPLITS = {"validation": seeds.VALIDATION, "test": seeds.TEST}


def board_size(text: str) -> tuple[int, int]:
    width, sep, height = text.lower().partition("x")
    try:
        return int(width), int(height if sep else width)
    except ValueError:
        raise argparse.ArgumentTypeError(f"board must look like 10x10, got {text!r}") from None


def board_games(text: str) -> tuple[int, int, int]:
    """``10x10`` or ``10x10:128``: a board and its games per generation (0: the default)."""
    board, _, games = text.partition(":")
    width, height = board_size(board)
    try:
        return width, height, int(games) if games else 0
    except ValueError:
        raise argparse.ArgumentTypeError(f"board must look like 10x10 or 10x10:128, got {text!r}") from None


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


def waits(text: str) -> tuple[int, ...]:
    """Comma-separated moves along the certificate's loop, for example ``2,4,8,16,32`` (empty: none)."""
    try:
        return tuple(int(part) for part in text.split(",") if part)
    except ValueError:
        raise argparse.ArgumentTypeError(f"waits must look like 2,4,8,16,32, got {text!r}") from None


def shares(text: str) -> tuple[float, ...]:
    """Comma-separated shares of the free cells, for example ``0.05,0.25,1`` (empty: none)."""
    try:
        return tuple(float(part) for part in text.split(",") if part)
    except ValueError:
        raise argparse.ArgumentTypeError(f"shares must look like 0.05,0.25,1, got {text!r}") from None


def plan_config(args: argparse.Namespace, explore: float = 0.0):
    from .plans import PlanConfig

    return PlanConfig(
        paths=args.paths,
        waits=args.waits,
        sweeps=args.sweeps,
        tolerance=args.tolerance,
        moves_trust=args.moves_trust,
        explore=explore,
        certify=args.certify,
        trees=args.trees,
        shortcut_budget=args.shortcut_budget,
    )


def train_network(args: argparse.Namespace) -> int:
    from .network import NetworkConfig  # needs PyTorch
    from .train import TrainConfig, train

    boards = tuple((width, height, games or args.games) for width, height, games in args.board or [(6, 6, 0)])
    config = TrainConfig(
        boards=boards,
        agent=args.agent,
        plans=plan_config(args, args.explore),
        generations=args.generations,
        free_share=args.free_share,
        prefix_share=args.prefix_share,
        prefix_fill=args.prefix_fill,
        recovery_share=args.recovery_share,
        midgame_share=args.midgame_share,
        midgame_samples=args.midgame_samples,
        simulations=args.simulations,
        moves_unit=args.moves_unit,
        relative_unit=args.relative_unit,
        win_unit=args.win_unit,
        full_share=args.full_share,
        fast_simulations=args.fast_simulations,
        confident=args.confident,
        max_examples=args.max_examples,
        beta=args.beta,
        moves_lambda=args.moves_lambda,
        sibling_visits=args.sibling_visits,
        replay=args.replay,
        replay_cells=args.replay_cells,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        lr_floor=args.lr_floor,
        warmup=args.warmup,
        min_replay=args.min_replay,
        network=NetworkConfig(
            args.channels,
            args.blocks,
            args.hidden,
            cycle=args.agent != "plans",
            arrival=args.agent == "plans",
            kind=args.network,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            patch=args.patch,
            horizon=args.horizon,
        ),
        workers=args.workers,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_sources=args.eval_sources,
        eval_simulations=args.eval_simulations,
        eval_moves_unit=args.eval_moves_unit,
        eval_relative_unit=args.eval_relative_unit,
        eval_confident=args.eval_confident,
        eval_win_unit=args.eval_win_unit,
    )
    train(config, args.out, device=args.device or _default_device(), init=args.init)
    return 0


def assess_network(args: argparse.Namespace) -> int:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from .assess import assess, assess_free  # needs PyTorch
    from .network import load, to_bytes
    from .search import SearchConfig
    from .train import _init_worker

    model, meta = load(args.checkpoint)
    config = meta.get("config", {})
    trained = config.get("boards") or [(config.get("width", 6), config.get("height", 6), 0)]
    width, height = args.board or max(((w, h) for w, h, _ in trained), key=lambda b: (b[0] * b[1], b[0]))
    context = multiprocessing.get_context("spawn")
    if args.agents == ["plans"]:
        with ProcessPoolExecutor(args.workers, mp_context=context, initializer=_init_worker) as pool:
            report = assess_free(
                pool,
                to_bytes(model),
                width,
                height,
                games=args.games,
                sources=args.sources,
                split=SPLITS[args.split],
                workers=args.workers,
                plans=plan_config(args),
                device=args.device or _default_device(),
            )
        text = json.dumps(report, indent=2)
        print(text)
        if args.json:
            Path(args.json).write_text(text + "\n")
        return 0
    if "plans" in args.agents:
        raise ValueError("assess the plan agent on its own: --agents plans")
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
            search=SearchConfig(
                simulations=args.simulations,
                gumbel_scale=0.0,
                moves_unit=args.moves_unit,
                relative_unit=args.relative_unit,
                confident=args.confident,
                careful_fill=args.careful_fill,
                win_unit=args.win_unit,
            ),
            device=args.device or _default_device(),
            agents=args.agents,
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    gates = [report.get(f"{track}_{kind}", {}).get("gate") for track in ("t1", "t2") for kind in args.agents]
    return 0 if all(gate["passed"] for gate in gates if gate is not None) else 1


def biases(text: str) -> tuple[float, ...]:
    """Comma-separated biases, for example ``-inf,-0.15,0.15,inf``."""
    try:
        return tuple(float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"biases must look like -inf,-0.15,0.15,inf, got {text!r}") from None


def improve_network_habits(args: argparse.Namespace) -> int:
    from .train import DistillConfig, HabitConfig, improve_habits  # needs PyTorch

    width, height = args.board
    config = HabitConfig(
        width=width,
        height=height,
        rounds=args.rounds,
        seeds=args.seeds,
        candidates=args.candidates,
        z=args.z,
        distill=DistillConfig(
            standard=args.standard,
            prefix=args.prefix,
            prefix_fill=args.prefix_fill,
            free=args.free,
            recovery=args.recovery,
            steps=args.steps,
            batch=args.batch,
            lr=args.lr,
            smoothing=args.smoothing,
        ),
        workers=args.workers,
        eval_games=args.eval_games,
        seed=args.seed,
    )
    improve_habits(
        config, args.checkpoint, args.out, device=args.device or _default_device(), anchor=args.anchor
    )
    return 0


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m snake_agent", description="Snake agent tools (milestones M0 to M2, and habits)."
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

    tr = commands.add_parser("train", help="train the network by self-play (needs PyTorch)")
    tr.add_argument(
        "--board",
        type=board_games,
        action="append",
        help="a board of the mixture, with its games per generation: 10x10 or 10x10:128 "
        "(repeatable; default: 6x6)",
    )
    tr.add_argument("--out", required=True, help="output directory for the log and checkpoints")
    tr.add_argument(
        "--agent",
        choices=["search", "plans"],
        default="search",
        help="search: self-play with the Gumbel search (default); plans: the plan agent's free play, "
        "with a network that does not see the canonical cycle",
    )
    tr.add_argument("--generations", type=int, default=50)
    tr.add_argument(
        "--games", type=int, default=256, help="self-play games per generation of a board (default: 256)"
    )
    tr.add_argument("--free-share", type=float, default=0.5, help="share of games from free starts")
    tr.add_argument(
        "--prefix-share", type=float, default=0.0, help="share of games after a random envelope prefix"
    )
    tr.add_argument(
        "--prefix-fill", type=float, default=0.75, help="the longest prefix, as a share of the board"
    )
    tr.add_argument(
        "--recovery-share", type=float, default=0.0, help="share of games from recovery starts (T2's kind)"
    )
    tr.add_argument(
        "--midgame-share", type=float, default=0.0, help="share of games from states of earlier games"
    )
    tr.add_argument("--midgame-samples", type=int, default=2, help="states each game leaves for those")
    tr.add_argument("--simulations", type=int, default=16)
    tr.add_argument(
        "--moves-unit", type=float, default=1.0, help="the self-play search's unit for comparing moves"
    )
    tr.add_argument(
        "--relative-unit", type=float, help="instead, that unit as a share of the moves scale (e.g. 0.05)"
    )
    tr.add_argument("--win-unit", type=float, help="the self-play search's free-mode win unit (e.g. 0.1)")
    tr.add_argument(
        "--full-share", type=float, default=1.0, help="share of moves with a full search and policy target"
    )
    tr.add_argument("--fast-simulations", type=int, default=0, help="simulations of the other moves")
    tr.add_argument("--confident", type=float, help="skip the search where the prior is this confident")
    tr.add_argument("--max-examples", type=int, default=0, help="the most examples one game gives (0: all)")
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
    tr.add_argument(
        "--replay-cells",
        type=int,
        default=0,
        help="plans: cap each board's examples at this many canvas cells (0: none)",
    )
    tr.add_argument("--steps", type=int, default=200, help="optimizer steps per generation")
    tr.add_argument("--batch", type=int, default=512)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument(
        "--lr-floor", type=float, default=1.0, help="cosine decay to this share of --lr (1: constant)"
    )
    tr.add_argument("--warmup", type=int, default=0, help="generations of linear learning-rate warm-up")
    tr.add_argument("--min-replay", type=int, default=0, help="no updates before the buffer holds this")
    tr.add_argument("--network", choices=["cnn", "transformer"], default="cnn", help="the network's kind")
    tr.add_argument("--channels", type=int, default=64)
    tr.add_argument("--blocks", type=int, default=6, help="residual blocks (the Transformer's stem)")
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--width", type=int, default=128, help="the Transformer's token width")
    tr.add_argument("--layers", type=int, default=6, help="the Transformer's attention layers")
    tr.add_argument("--heads", type=int, default=4, help="the Transformer's attention heads")
    tr.add_argument("--patch", type=int, default=2, help="the side of the Transformer's tokens, in cells")
    tr.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="plans: the moves head counts the moves to eat this many more fruits (0: to the end)",
    )
    tr.add_argument("--workers", type=int, default=8, help="self-play and evaluation processes")
    tr.add_argument("--seed", type=int, default=0, help="run seed (selects the training seeds)")
    tr.add_argument("--eval-every", type=int, default=10, help="assess every this many generations")
    tr.add_argument("--eval-games", type=int, default=100, help="T1 games per assessment")
    tr.add_argument("--eval-sources", type=int, default=60, help="T2 source games per assessment")
    tr.add_argument("--eval-simulations", type=int, default=16, help="the assessed search's simulations")
    tr.add_argument("--eval-moves-unit", type=float, default=1.0, help="the assessed search's moves unit")
    tr.add_argument("--eval-relative-unit", type=float, help="instead, its relative unit")
    tr.add_argument("--eval-confident", type=float, help="the assessed search's adaptive budget")
    tr.add_argument("--eval-win-unit", type=float, help="the assessed search's free-mode win unit")
    _plan_arguments(tr)
    tr.add_argument(
        "--explore", type=float, default=0.1, help="plans: chance of a random acceptable plan (default: 0.1)"
    )
    tr.add_argument("--device", help="torch device for training (default: cuda if available, else cpu)")
    tr.add_argument("--init", help="start from this checkpoint's weights (a model.pt written by train)")
    tr.set_defaults(run=train_network)

    asm = commands.add_parser("assess", help="assess a trained network against the gates (needs PyTorch)")
    asm.add_argument("checkpoint", help="a model.pt written by train")
    asm.add_argument("--board", type=board_size, help="board size (default: the largest training board)")
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
    asm.add_argument(
        "--relative-unit", type=float, help="instead, that unit as a share of the moves scale (e.g. 0.05)"
    )
    asm.add_argument("--confident", type=float, help="skip the search where the prior is this confident")
    asm.add_argument(
        "--careful-fill",
        type=float,
        default=0.7,
        help="free mode: search above this fill whatever the prior (0: always; default: 0.7)",
    )
    asm.add_argument("--win-unit", type=float, help="free mode: compare win values in this unit (e.g. 0.1)")
    asm.add_argument("--workers", type=int, default=8)
    asm.add_argument(
        "--agents",
        nargs="+",
        choices=["search", "policy", "plans"],
        default=["search", "policy"],
        help="the agents to assess (default: search and policy; plans: the plan agent, on its own)",
    )
    _plan_arguments(asm)
    asm.add_argument("--device", help="torch device for the games (default: cuda if available, else cpu)")
    asm.add_argument("--json", help="write the report to this JSON file")
    asm.set_defaults(run=assess_network)

    hb = commands.add_parser(
        "habits", help="improve a network's jump habits by paired games and distil them (needs PyTorch)"
    )
    hb.add_argument("checkpoint", help="a model.pt written by train (or by habits)")
    hb.add_argument("--out", required=True, help="output directory for the log and checkpoints")
    hb.add_argument("--board", type=board_size, default=(10, 10), help="board size (default: 10x10)")
    hb.add_argument("--rounds", type=int, default=2, help="rounds of comparison and distillation")
    hb.add_argument("--seeds", type=int, default=400, help="games per round whose windows compare habits")
    hb.add_argument(
        "--candidates",
        type=biases,
        default=CANDIDATES,
        help="biases tried in each band, comma-separated (default: -inf,-0.15,0,0.15,inf)",
    )
    hb.add_argument("--z", type=float, default=2.0, help="standard errors a band's improvement needs")
    hb.add_argument("--standard", type=int, default=300, help="distillation games from the standard start")
    hb.add_argument("--prefix", type=int, default=100, help="... after a random envelope prefix")
    hb.add_argument(
        "--prefix-fill", type=float, default=0.75, help="the longest prefix, as a share of the board"
    )
    hb.add_argument("--free", type=int, default=150, help="... from free starts (the anchor's own policy)")
    hb.add_argument("--recovery", type=int, default=150, help="... from recovery starts (the same)")
    hb.add_argument(
        "--anchor", help="the checkpoint whose play outside the envelope is kept (default: the checkpoint)"
    )
    hb.add_argument("--steps", type=int, default=1500, help="distillation optimizer steps")
    hb.add_argument("--batch", type=int, default=512)
    hb.add_argument("--lr", type=float, default=3e-4)
    hb.add_argument("--smoothing", type=float, default=0.02, help="target share left by an infinite bias")
    hb.add_argument("--workers", type=int, default=8)
    hb.add_argument(
        "--eval-games", type=int, default=300, help="validation T1 games after each round (0: none)"
    )
    hb.add_argument("--seed", type=int, default=0, help="run seed (selects the training seeds)")
    hb.add_argument("--device", help="torch device for distillation (default: cuda if available, else cpu)")
    hb.set_defaults(run=improve_network_habits)
    return parser


def _plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paths", type=int, default=16, help="plans: shortest paths drawn per decision")
    parser.add_argument(
        "--waits",
        type=waits,
        default=(2, 4, 8, 16, 32),
        help="plans: moves along the certificate's loop before a waiting plan (default: 2,4,8,16,32)",
    )
    parser.add_argument(
        "--sweeps",
        type=shares,
        default=(0.25, 1.0),
        help="plans: sweeping plans lengthen a shortest path by up to these shares of the free cells "
        "(default: 0.25,1)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="plans: extra risk accepted for a faster plan (default: 0.02)",
    )
    parser.add_argument(
        "--moves-trust",
        type=float,
        default=1000.0,
        help="plans: moves scale up to which the moves head's differences are trusted (default: 1000)",
    )
    parser.add_argument(
        "--certify",
        type=certify_share,
        default=0.4,
        help="plans: share of the board filled from which the agent finishes on a Hamiltonian-cycle "
        "certificate, or 'none' to play free to the end (default: 0.4)",
    )
    parser.add_argument(
        "--trees", type=int, default=64, help="plans: random spanning trees tried per certified decision"
    )
    parser.add_argument(
        "--shortcut-budget",
        type=int,
        default=20_000,
        help="plans: steps of search for certified shortcuts per decision (default: 20000)",
    )


def certify_share(text: str) -> float | None:
    """A share of the board in [0, 1], or ``none``."""
    if text.lower() == "none":
        return None
    share = float(text)
    if not 0.0 <= share <= 1.0:
        raise argparse.ArgumentTypeError(f"a share of the board must be in [0, 1], got {text}")
    return share


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, ImportError, AttributeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
