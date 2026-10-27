"""Training for milestone M1: self-play with the search, then supervised updates, generation after generation.

Each generation:

1. **Self-play.** ``games`` games with the current network, split across worker
   processes that each play their games at once (:func:`~snake_agent.selfplay.play`).
   A ``free_share`` of them start from free starts, a ``prefix_share`` from the
   standard start after a random prefix, the rest from the standard start. Seeds
   are training seeds, a block per generation (:meth:`TrainConfig.seeds`).
2. **Replay.** The examples join a buffer that keeps the most recent ``replay``.
3. **Updates.** ``steps`` optimizer steps on minibatches drawn uniformly from the
   buffer, each example mirrored left to right with probability 1/2. The policy
   head learns the search's improved policy (cross-entropy over the moves the
   search considered), the win head the win target (binary cross-entropy), and
   the moves head the moves target of won games (Huber loss).
4. **Assessment.** Every ``eval_every`` generations, and after the last, the
   network is assessed on validation seeds (:func:`~snake_agent.assess.assess`).

Everything is written to the output directory: ``log.jsonl`` (one line per
generation), ``model.pt`` (the latest network) and ``model-<generation>.pt`` at
each assessment.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .assess import assess
from .contract import TRAINING_RULES, Outcome
from .encoding import mirror
from .network import NetworkConfig, NetworkEvaluator, SnakeNet, from_bytes, load, save, tensors, to_bytes
from .search import SELF_PLAY
from .selfplay import Examples, GameSummary, Start, free_starts, play, prefix_starts, standard_starts

GAMES_PER_GENERATION_LIMIT = 1000
GENERATION_LIMIT = 1000


@dataclass(frozen=True)
class TrainConfig:
    """Settings of a training run."""

    width: int = 6
    height: int = 6
    generations: int = 50
    games: int = 256
    """Self-play games per generation."""
    free_share: float = 0.5
    """Share of the games from free starts."""
    prefix_share: float = 0.0
    """Share of the games from the standard start after a random envelope prefix (the rest: standard)."""
    prefix_fill: float = 0.75
    """The longest prefix, as a share of the board the snake fills (see :func:`prefix_starts`)."""
    simulations: int = 16
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
    weight_decay: float = 1e-4
    network: NetworkConfig = field(default_factory=NetworkConfig)
    workers: int = 8
    seed: int = 0
    eval_every: int = 10
    eval_games: int = 100
    eval_sources: int = 60

    def __post_init__(self) -> None:
        if not 0 < self.games <= GAMES_PER_GENERATION_LIMIT or not 0 < self.generations <= GENERATION_LIMIT:
            raise ValueError(
                f"at most {GAMES_PER_GENERATION_LIMIT} games per generation "
                f"and {GENERATION_LIMIT} generations"
            )
        if not 0 <= self.seed < 500:
            raise ValueError("the run seed must be in [0, 500)")

    def seeds(self, generation: int) -> range:
        """The generation's training seeds: one block of 1,000 per generation, 10^6 per run seed.

        Free starts take their fruits from these plus 5 x 10^8, still training seeds.
        """
        first = self.seed * 1_000_000 + generation * 1_000
        return range(first, first + self.games)


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


def _self_play_task(args: tuple) -> tuple[list[GameSummary], Examples]:
    weights, starts, simulations, beta, moves_lambda, sibling_visits, seed, device = args
    config = dataclasses.replace(SELF_PLAY, simulations=simulations)
    evaluator = NetworkEvaluator(from_bytes(weights), device)
    return play(
        evaluator,
        starts,
        TRAINING_RULES,
        config,
        seed=seed,
        beta=beta,
        moves_lambda=moves_lambda,
        sibling_visits=sibling_visits,
    )


def _starts(config: TrainConfig, generation: int) -> list[Start]:
    seeds = config.seeds(generation)
    free = round(config.free_share * len(seeds))
    prefix = free + round(config.prefix_share * len(seeds))
    starts = free_starts(config.width, config.height, seeds[:free])
    starts += prefix_starts(config.width, config.height, seeds[free:prefix], config.prefix_fill)
    starts += standard_starts(config.width, config.height, seeds[prefix:])
    return starts


def _game_stats(summaries: list[GameSummary]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    kinds = {"standard": (True, False), "prefix": (True, True), "free": (False, False)}
    for kind, (standard, prefixed) in kinds.items():
        games = [s for s in summaries if s.standard is standard and (s.prefix > 0) is prefixed]
        if not games:
            continue
        outcomes = Counter(s.outcome.value for s in games)
        won = [s.moves for s in games if s.outcome is Outcome.WON]
        stats[kind] = {
            "games": len(games),
            **{outcome.value: outcomes.get(outcome.value, 0) for outcome in Outcome if outcome.resolved},
            "mean_moves_won": round(float(np.mean(won)), 2) if won else None,
            "searched_share": round(sum(s.searched for s in games) / max(1, sum(s.moves for s in games)), 4),
        }
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
    games, the network then costs little next to the search's own work. Assessment games evaluate one
    leaf at a time, on the CPU. ``init`` starts from a checkpoint's weights (its network size must match
    ``config``); the replay buffer and the optimizer start empty, and a new run ``seed`` gives new games.
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
    inference = "cuda" if str(device).startswith("cuda") else "cpu"
    if inference == "cuda":
        torch.set_num_threads(2)  # leave the CPUs to the self-play workers
    context = multiprocessing.get_context("spawn")
    references = None
    with ProcessPoolExecutor(config.workers, mp_context=context, initializer=_init_worker) as pool:
        for generation in range(config.generations):
            started = time.perf_counter()
            weights = to_bytes(model)
            starts = _starts(config, generation)
            order = np.random.default_rng([config.seed, generation]).permutation(len(starts))
            tasks = [
                (
                    weights,
                    [starts[i] for i in order[k :: config.workers]],
                    config.simulations,
                    config.beta,
                    config.moves_lambda,
                    config.sibling_visits,
                    (config.seed, generation, k),
                    inference,
                )
                for k in range(min(config.workers, len(starts)))
            ]
            summaries, parts = [], []
            for task_summaries, examples in pool.map(_self_play_task, tasks):
                summaries += task_summaries
                parts.append(examples)
            examples = Examples.concatenate(parts)
            replay.add(examples)
            played = time.perf_counter()
            terms = optimize(model, optimizer, replay, config.steps, config.batch, rng, device)
            trained = time.perf_counter()
            record: dict[str, Any] = {
                "generation": generation,
                "games": _game_stats(summaries),
                "examples": len(examples),
                "replay": replay.size,
                "losses": terms,
                "seconds": {"self_play": round(played - started, 1), "train": round(trained - played, 1)},
            }
            save(model, out / "model.pt", generation=generation, config=asdict(config))
            last = generation + 1 == config.generations
            if (generation + 1) % config.eval_every == 0 or last:
                report = assess(
                    pool,
                    to_bytes(model),
                    config.width,
                    config.height,
                    games=config.eval_games,
                    sources=config.eval_sources,
                    workers=config.workers,
                    references=references,
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


def _line(record: dict[str, Any]) -> str:
    """A one-line summary of a generation's record."""
    games = record["games"]
    parts = [f"gen {record['generation'] + 1:3d}"]
    if "standard" in games:
        s = games["standard"]
        parts.append(f"std won {s['won']}/{s['games']} moves {s['mean_moves_won']}")
    if "prefix" in games:
        p = games["prefix"]
        parts.append(f"prefix won {p['won']}/{p['games']}")
    if "free" in games:
        f = games["free"]
        parts.append(f"free won {f['won']}/{f['games']}")
    losses_ = record["losses"]
    parts.append(f"loss p {losses_['policy']:.3f} w {losses_['win']:.3f} m {losses_['moves']:.3f}")
    seconds = record["seconds"]
    parts.append(f"{seconds['self_play']:.0f}+{seconds['train']:.0f}s")
    if "assessment" in record:
        report = record["assessment"]
        for kind in ("search", "policy"):
            t1 = report[f"t1_{kind}"]["summary"]
            parts.append(f"T1 {kind} {t1['won']}/{t1['games']} {t1['mean_moves']}")
    return " | ".join(parts)
