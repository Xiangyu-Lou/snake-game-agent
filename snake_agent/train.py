"""Training: self-play with the search, then supervised updates, generation after generation.

Each generation:

1. **Self-play.** On each board of the mixture (:attr:`TrainConfig.boards`), that
   board's games with the current network, split across worker processes that
   each play their games at once (:func:`~snake_agent.selfplay.play`). A
   ``free_share`` of each board's games start from free starts, a
   ``prefix_share`` from the standard start after a random prefix, a
   ``recovery_share`` from recovery starts on training seeds (track T2's kind of
   start), a ``midgame_share`` from states of the agent's own earlier games, the
   rest from the standard start. The workers build the starts. Seeds are
   training seeds, a block per generation (:meth:`TrainConfig.seeds`).
2. **Replay.** The examples join a buffer that keeps the most recent ``replay``,
   encoded on the canvas of the largest board. Each game also leaves a few of
   its states for later mid-game starts.
3. **Updates.** ``steps`` optimizer steps on minibatches drawn uniformly from the
   buffer, each example mirrored left to right with probability 1/2, once it
   holds ``min_replay`` examples. The policy head learns the search's improved
   policy (cross-entropy over the moves the search considered), the win head
   the win target (binary cross-entropy), and the moves head the moves target of
   won games (Huber loss). The learning rate warms up linearly over ``warmup``
   generations and then follows a cosine down to ``lr * lr_floor``.
4. **Assessment.** Every ``eval_every`` generations, and after the last, the
   network is assessed on validation seeds on the largest board
   (:func:`~snake_agent.assess.assess`).

Everything is written to the output directory: ``log.jsonl`` (one line per
generation, with the throughput), ``model.pt`` (the latest network) and
``model-<generation>.pt`` at each assessment.
"""

from __future__ import annotations

import dataclasses
import json
import math
import multiprocessing
import random
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from snake_game import GameState

from .assess import assess
from .contract import TRAINING_RULES, Outcome
from .encoding import mirror
from .network import NetworkConfig, NetworkEvaluator, SnakeNet, from_bytes, load, save, tensors, to_bytes
from .search import SELF_PLAY, SearchConfig
from .selfplay import KINDS, Examples, GameSummary, make_start, play

GAMES_PER_GENERATION_LIMIT = 1000
GENERATION_LIMIT = 1000


@dataclass(frozen=True)
class TrainConfig:
    """Settings of a training run."""

    boards: tuple[tuple[int, int, int], ...] = ((6, 6, 256),)
    """The mixture of boards: ``(width, height, games per generation)`` for each."""
    generations: int = 50
    free_share: float = 0.5
    """Share of each board's games from free starts."""
    prefix_share: float = 0.0
    """Share of each board's games from the standard start after a random envelope prefix."""
    prefix_fill: float = 0.75
    """The longest prefix, as a share of the board the snake fills (see :func:`prefix_start`)."""
    recovery_share: float = 0.0
    """Share of each board's games from recovery starts on training seeds (see :func:`recovery_start`)."""
    midgame_share: float = 0.0
    """Share of each board's games from states of the agent's earlier games (see :func:`midgame_start`)."""
    midgame_samples: int = 2
    """States each game leaves for mid-game starts."""
    midgame_pool: int = 20_000
    """Mid-game starts are drawn from the most recent this many states left on each board."""
    simulations: int = 16
    moves_unit: float | None = 1.0
    """The self-play search's moves unit (:attr:`~snake_agent.search.SearchConfig.moves_unit`). Its
    improved policy is the policy target, so it should be no worse than the raw policy: a unit far
    below the moves head's errors makes the targets chase those errors."""
    relative_unit: float | None = None
    """If set, the self-play search's unit is this share of the moves scale instead
    (:attr:`~snake_agent.search.SearchConfig.relative_unit`)."""
    beta: float = 0.5
    """Weight of the search's values in the win targets."""
    moves_lambda: float = 0.5
    """Lambda of the moves targets' TD returns over fruits (1: the moves the game still took)."""
    sibling_visits: int = 0
    """Envelope moves the search visited this often but did not play give examples (0: none)."""
    replay: int = 300_000
    steps: int = 200
    """Optimizer steps per generation."""
    batch: int = 512
    lr: float = 1e-3
    lr_floor: float = 1.0
    """The learning rate follows a cosine over the run from ``lr`` down to ``lr * lr_floor`` (1: constant)."""
    warmup: int = 0
    """Generations over which the learning rate first rises linearly to ``lr``."""
    min_replay: int = 0
    """No updates until the buffer holds this many examples."""
    weight_decay: float = 1e-4
    network: NetworkConfig = field(default_factory=NetworkConfig)
    workers: int = 8
    seed: int = 0
    eval_every: int = 10
    eval_games: int = 100
    eval_sources: int = 60
    eval_simulations: int = 16
    eval_moves_unit: float | None = 1.0
    """The search's moves unit in the assessments (:attr:`~snake_agent.search.SearchConfig.moves_unit`)."""
    eval_relative_unit: float | None = None
    """If set, the assessed search's relative unit instead."""

    def __post_init__(self) -> None:
        if not self.boards or any(games <= 0 for _, _, games in self.boards):
            raise ValueError("every board needs games")
        if self.games > GAMES_PER_GENERATION_LIMIT or not 0 < self.generations <= GENERATION_LIMIT:
            raise ValueError(
                f"at most {GAMES_PER_GENERATION_LIMIT} games per generation "
                f"and {GENERATION_LIMIT} generations"
            )
        shares = self.free_share + self.prefix_share + self.recovery_share + self.midgame_share
        if min(self.free_share, self.prefix_share, self.recovery_share, self.midgame_share) < 0 or shares > 1:
            raise ValueError("the start shares must be non-negative and add up to at most 1")
        if not 0 <= self.seed < 500:
            raise ValueError("the run seed must be in [0, 500)")

    @property
    def games(self) -> int:
        """Self-play games per generation, over all boards."""
        return sum(games for _, _, games in self.boards)

    @property
    def canvas(self) -> int:
        """The side of the canvas the examples are encoded on: the largest board side."""
        return max(max(width, height) for width, height, _ in self.boards)

    @property
    def eval_board(self) -> tuple[int, int]:
        """The board the assessments use: the largest of the mixture."""
        width, height, _ = max(self.boards, key=lambda b: (b[0] * b[1], b[0]))
        return width, height

    def seeds(self, generation: int) -> range:
        """The generation's training seeds: one block of 1,000 per generation, 10^6 per run seed.

        Starts taken from another game draw their fruits from these plus 5 x 10^8, still training seeds.
        """
        first = self.seed * 1_000_000 + generation * 1_000
        return range(first, first + self.games)

    def learning_rate(self, generation: int) -> float:
        """The generation's learning rate: linear warm-up, then a cosine down to ``lr * lr_floor``."""
        progress = generation / max(1, self.generations - 1)
        rate = self.lr * (self.lr_floor + (1.0 - self.lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        if generation < self.warmup:
            rate *= (generation + 1) / self.warmup
        return rate


# --------------------------------------------------------------------- replay
class Replay:
    """The most recent ``capacity`` examples, in a ring buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.size = 0
        self.cursor = 0
        self.data: dict[str, np.ndarray] | None = None

    def add(self, examples: Examples) -> None:
        n = len(examples)
        if n > self.capacity:
            examples, n = examples.take(slice(n - self.capacity, n)), self.capacity
        if self.data is None:
            self.data = {
                f.name: np.zeros(
                    (self.capacity,) + getattr(examples, f.name).shape[1:], getattr(examples, f.name).dtype
                )
                for f in dataclasses.fields(Examples)
            }
        index = (self.cursor + np.arange(n)) % self.capacity
        for name, array in self.data.items():
            array[index] = getattr(examples, name)
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, count: int, rng: np.random.Generator) -> Examples:
        index = rng.integers(self.size, size=count)
        return Examples(**{name: array[index] for name, array in self.data.items()})


# --------------------------------------------------------------------- losses
def losses(model: SnakeNet, batch: Examples, device: torch.device | str) -> dict[str, torch.Tensor]:
    """The three heads' losses on a batch of examples."""
    planes, actions, globals_, head = tensors(
        (batch.planes, batch.actions, batch.globals, batch.head), device
    )
    logits, win_logit, moves = model(planes, actions, globals_, head)
    valid = torch.as_tensor(batch.valid, device=device)
    has_policy = torch.as_tensor(batch.has_policy, dtype=torch.float32, device=device)
    has_moves = torch.as_tensor(batch.has_moves, dtype=torch.float32, device=device)
    log_policy = F.log_softmax(logits.masked_fill(~valid, -1e4), dim=1)
    target = torch.as_tensor(batch.policy, device=device)
    policy = -(target * log_policy).sum(1)
    win = F.binary_cross_entropy_with_logits(win_logit, torch.as_tensor(batch.win, device=device))
    huber = F.smooth_l1_loss(moves, torch.as_tensor(batch.moves, device=device), reduction="none")
    return {
        "policy": (policy * has_policy).sum() / has_policy.sum().clamp(min=1.0),
        "win": win,
        "moves": (huber * has_moves).sum() / has_moves.sum().clamp(min=1.0),
    }


def mirrored(batch: Examples, flip: np.ndarray) -> Examples:
    """``batch`` with the examples where ``flip`` is set mirrored left to right."""
    if not flip.any():
        return batch
    planes, actions, head, policy = (
        batch.planes.copy(),
        batch.actions.copy(),
        batch.head.copy(),
        batch.policy.copy(),
    )
    valid = batch.valid.copy()
    planes[flip], actions[flip], head[flip], policy[flip] = mirror(
        batch.planes[flip], batch.actions[flip], batch.head[flip], batch.width[flip], batch.policy[flip]
    )
    valid[flip] = batch.valid[flip][:, ::-1]
    return dataclasses.replace(batch, planes=planes, actions=actions, head=head, policy=policy, valid=valid)


def optimize(
    model: SnakeNet,
    optimizer: torch.optim.Optimizer,
    replay: Replay,
    steps: int,
    batch: int,
    rng: np.random.Generator,
    device: torch.device | str,
) -> dict[str, float]:
    """``steps`` updates on minibatches from ``replay``; returns the mean losses."""
    model.train()
    totals: Counter[str] = Counter()
    for _ in range(steps):
        sample = replay.sample(batch, rng)
        terms = losses(model, mirrored(sample, rng.random(batch) < 0.5), device)
        loss = terms["policy"] + terms["win"] + terms["moves"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for name, value in terms.items():
            totals[name] += value.item()
    model.eval()
    return {name: round(total / steps, 5) for name, total in totals.items()}


# ------------------------------------------------------------------ workers
def _init_worker() -> None:
    torch.set_num_threads(1)


def _self_play_task(args: tuple) -> tuple[list[GameSummary], Examples | None, float]:
    weights, specs, simulations, units, beta, moves_lambda, sibling_visits, samples, size, seed, device = args
    began = time.perf_counter()
    starts = [start for spec in specs if (start := make_start(*spec)) is not None]
    if not starts:
        return [], None, time.perf_counter() - began
    unit, relative = units
    config = dataclasses.replace(SELF_PLAY, simulations=simulations, moves_unit=unit, relative_unit=relative)
    evaluator = NetworkEvaluator(from_bytes(weights), device)
    summaries, examples = play(
        evaluator,
        starts,
        TRAINING_RULES,
        config,
        seed=seed,
        beta=beta,
        moves_lambda=moves_lambda,
        size=size,
        sibling_visits=sibling_visits,
        samples=samples,
    )
    return summaries, examples, time.perf_counter() - began


def balanced(costs: list[float], parts: int) -> list[list[int]]:
    """Indices of ``costs`` split into ``parts`` groups of similar total cost: each item, the costliest
    first, joins the group with the least cost so far (the longest-processing-time rule)."""
    groups: list[list[int]] = [[] for _ in range(min(parts, len(costs)))]
    loads = [0.0] * len(groups)
    for i in sorted(range(len(costs)), key=lambda i: -costs[i]):
        k = loads.index(min(loads))
        groups[k].append(i)
        loads[k] += costs[i]
    return [sorted(group) for group in groups]


def _costs(specs: list[StartSpec], summaries: list[GameSummary]) -> list[float]:
    """Each start's expected self-play cost: the mean leaves of the last generation's games of its board
    and kind (``cells^1.5`` for a board and kind not seen yet, the scale of a game's moves)."""
    leaves: dict[tuple[tuple[int, int], str], list[int]] = {}
    for s in summaries:
        leaves.setdefault((s.board, s.kind), []).append(max(s.leaves, s.moves))
    mean = {key: float(np.mean(values)) for key, values in leaves.items()}
    return [mean.get(((w, h), kind), float(w * h) ** 1.5) for kind, w, h, _, _ in specs]


StartSpec = tuple[str, int, int, int, Any]
"""What a worker turns into a start: :func:`~snake_agent.selfplay.make_start`'s arguments."""


def _starts(
    config: TrainConfig, generation: int, pools: dict[tuple[int, int], deque[GameState]]
) -> list[StartSpec]:
    """The generation's starts, board by board, each with its own training seed."""
    seeds = iter(config.seeds(generation))
    rng = random.Random(f"{config.seed}/{generation}/midgame")
    specs: list[StartSpec] = []
    for width, height, games in config.boards:
        pool = pools.get((width, height), ())
        counts = {
            "free": round(config.free_share * games),
            "prefix": round(config.prefix_share * games),
            "recovery": round(config.recovery_share * games),
            "midgame": round(config.midgame_share * games) if pool else 0,
        }
        counts["standard"] = games - sum(counts.values())
        if counts["standard"] < 0:  # rounding
            counts["free"] += counts.pop("standard")
            counts["standard"] = 0
        for kind, count in counts.items():
            for _ in range(count):
                extra = config.prefix_fill if kind == "prefix" else None
                if kind == "midgame":
                    extra = pool[rng.randrange(len(pool))]
                specs.append((kind, width, height, next(seeds), extra))
    return specs


def _game_stats(summaries: list[GameSummary]) -> dict[str, Any]:
    """Outcomes, moves and search effort of the games, by board and kind of start."""
    stats: dict[str, Any] = {}
    for board in sorted({s.board for s in summaries}):
        entry: dict[str, Any] = {}
        for kind in KINDS:
            games = [s for s in summaries if s.board == board and s.kind == kind]
            if not games:
                continue
            outcomes = Counter(s.outcome.value for s in games)
            won = [s.moves for s in games if s.outcome is Outcome.WON]
            moves = sum(s.moves for s in games)
            entry[kind] = {
                "games": len(games),
                **{outcome.value: outcomes.get(outcome.value, 0) for outcome in Outcome if outcome.resolved},
                "mean_moves_won": round(float(np.mean(won)), 2) if won else None,
                "moves": moves,
                "searched_share": round(sum(s.searched for s in games) / max(1, moves), 4),
                "leaves": sum(s.leaves for s in games),
            }
        stats[f"{board[0]}x{board[1]}"] = entry
    return stats


def train(
    config: TrainConfig,
    out: str | Path,
    device: str = "cpu",
    log: Callable[[str], None] = print,
    init: str | Path | None = None,
) -> SnakeNet:
    """Run ``config`` and write its log and checkpoints to ``out``; returns the trained network.

    With a CUDA ``device`` the self-play workers evaluate their leaves on it too: batched across their
    games, the network then costs little next to the search's own work. (With many workers on one GPU,
    run NVIDIA's Multi-Process Service, which lets their small batches share the GPU.) ``init`` starts
    from a checkpoint's weights (its network size must match ``config``); the replay buffer and the
    optimizer start empty, and a new run ``seed`` gives new games.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = asdict(config) | ({"init": str(init)} if init is not None else {})
    (out / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    model = SnakeNet(config.network).to(device)
    if init is not None:
        start, _ = load(init, device)
        if start.config != config.network:
            raise ValueError(f"the checkpoint's network {start.config} is not {config.network}")
        model.load_state_dict(start.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    replay = Replay(config.replay)
    pools: dict[tuple[int, int], deque[GameState]] = {}
    inference = "cuda" if str(device).startswith("cuda") else "cpu"
    if inference == "cuda":
        torch.set_num_threads(2)  # leave the CPUs to the self-play workers
    context = multiprocessing.get_context("spawn")
    references = None
    search = SearchConfig(
        simulations=config.eval_simulations,
        gumbel_scale=0.0,
        moves_unit=config.eval_moves_unit,
        relative_unit=config.eval_relative_unit,
    )
    games_played = 0
    summaries: list[GameSummary] = []  # the last generation's games, which price the next one's starts
    with ProcessPoolExecutor(config.workers, mp_context=context, initializer=_init_worker) as pool:
        for generation in range(config.generations):
            started = time.perf_counter()
            weights = to_bytes(model)
            specs = _starts(config, generation, pools)
            tasks = [
                (
                    weights,
                    [specs[i] for i in group],
                    config.simulations,
                    (config.moves_unit, config.relative_unit),
                    config.beta,
                    config.moves_lambda,
                    config.sibling_visits,
                    config.midgame_samples,
                    config.canvas,
                    (config.seed, generation, k),
                    inference,
                )
                for k, group in enumerate(balanced(_costs(specs, summaries), config.workers))
            ]
            summaries, parts, busy = [], [], []
            for task_summaries, examples, seconds in pool.map(_self_play_task, tasks):
                summaries += task_summaries
                busy.append(seconds)
                if examples is not None:
                    parts.append(examples)
            examples = Examples.concatenate(parts)
            replay.add(examples)
            for summary in summaries:
                pools.setdefault(summary.board, deque(maxlen=config.midgame_pool)).extend(summary.samples)
            games_played += len(summaries)
            played = time.perf_counter()
            rate = config.learning_rate(generation)
            for group in optimizer.param_groups:
                group["lr"] = rate
            terms = {}
            if replay.size >= config.min_replay:
                terms = optimize(model, optimizer, replay, config.steps, config.batch, rng, device)
            trained = time.perf_counter()
            moves = sum(s.moves for s in summaries)
            leaves = sum(s.leaves for s in summaries)
            record: dict[str, Any] = {
                "generation": generation,
                "games": _game_stats(summaries),
                "games_played": games_played,
                "examples": len(examples),
                "replay": replay.size,
                "lr": rate,
                "losses": terms,
                "seconds": {"self_play": round(played - started, 1), "train": round(trained - played, 1)},
                "throughput": {
                    "moves_per_second": round(moves / max(played - started, 1e-9), 1),
                    "leaves_per_second": round(leaves / max(played - started, 1e-9), 1),
                    "worker_busy": round(float(np.mean(busy)) / max(max(busy), 1e-9), 3),
                },
            }
            save(model, out / "model.pt", generation=generation, config=asdict(config))
            last = generation + 1 == config.generations
            if (generation + 1) % config.eval_every == 0 or last:
                width, height = config.eval_board
                report = assess(
                    pool,
                    to_bytes(model),
                    width,
                    height,
                    games=config.eval_games,
                    sources=config.eval_sources,
                    workers=config.workers,
                    references=references,
                    search=search,
                    device=inference,
                )
                references = {float(k): v for k, v in report.get("t2_references", {}).items()} or None
                record["assessment"] = report
                record["seconds"]["assessment"] = round(time.perf_counter() - trained, 1)
                save(
                    model,
                    out / f"model-{generation + 1:03d}.pt",
                    generation=generation,
                    config=asdict(config),
                )
            with open(out / "log.jsonl", "a") as handle:
                handle.write(json.dumps(record) + "\n")
            log(_line(record))
    return model


SHORT = {"standard": "std", "prefix": "pre", "free": "free", "recovery": "rec", "midgame": "mid"}


def _line(record: dict[str, Any]) -> str:
    """A one-line summary of a generation's record."""
    parts = [f"gen {record['generation'] + 1:3d}"]
    for board, kinds in record["games"].items():
        games = []
        for kind, g in kinds.items():
            text = f"{SHORT[kind]} {g['won']}/{g['games']}"
            if kind == "standard" and g["mean_moves_won"] is not None:
                text += f" {g['mean_moves_won']:.1f}"
            games.append(text)
        parts.append(f"{board} " + " ".join(games))
    losses_ = record["losses"]
    if losses_:
        parts.append(f"loss p {losses_['policy']:.3f} w {losses_['win']:.3f} m {losses_['moves']:.3f}")
    seconds = record["seconds"]
    parts.append(f"{seconds['self_play']:.0f}+{seconds['train']:.0f}s")
    if "assessment" in record:
        report = record["assessment"]
        for kind in ("search", "policy"):
            if f"t1_{kind}" in report:
                t1 = report[f"t1_{kind}"]["summary"]
                parts.append(f"T1 {kind} {t1['won']}/{t1['games']} {t1['mean_moves']}")
            if f"t2_{kind}" in report:
                t2 = report[f"t2_{kind}"]
                bands = " ".join(f"{t2[b]['won']}/{t2[b]['games']}" for b in t2 if b not in ("all", "gate"))
                parts.append(f"T2 {kind} {bands}")
    return " | ".join(parts)
