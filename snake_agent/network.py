"""The M1 network: a small residual CNN with a policy head, a win-probability head and a moves head.

PyTorch is the network's stack (review, decision 4) and an optional dependency:
``pip install -e ".[train]"``. The rest of the package works without it.

The inputs are the encodings of :mod:`snake_agent.encoding`, in the head's
frame. The trunk is fully convolutional and masked to the board's cells, so the
same weights run on any board size and on boards padded into a larger canvas:

* a 3x3 stem with the global features added as a per-channel bias, then
  residual blocks of two 3x3 convolutions with a global-pooling bias between
  them (as in KataGo), each block starting as the identity;
* a summary of the board: the head cell's features, the mean and maximum over
  the board's cells, and the global features;
* the **policy head** scores each relative action from the features of the cell
  it enters (left of, above or right of the head), its per-action features and
  the summary;
* the **win head** gives the probability of filling the board (a logit), and the
  **moves head** the expected moves until the board is full if it is filled, in
  units of :func:`~snake_agent.encoding.moves_scale` (non-negative).

:class:`NetworkEvaluator` evaluates the search's queries with it.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .encoding import ACTION_FEATURES, ARRIVAL_PLANES, GLOBALS, PLANES, encode, stack
from .search import Prediction, Query

_VALID = PLANES.index("valid")


@dataclass(frozen=True)
class NetworkConfig:
    """The network's size."""

    channels: int = 64
    blocks: int = 6
    hidden: int = 128
    cycle: bool = True
    """Whether the inputs include the canonical cycle (see :func:`~snake_agent.encoding.encode`)."""
    arrival: bool = False
    """Whether the inputs include the arrival planes (see :func:`~snake_agent.encoding.encode`)."""

    @property
    def planes(self) -> int:
        """The input planes the network reads: the first ones of :data:`~snake_agent.encoding.PLANES`."""
        return len(PLANES) if self.arrival else len(PLANES) - ARRIVAL_PLANES


class _Block(nn.Module):
    """Two 3x3 convolutions with a global-pooling bias between them; starts as the identity."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.Linear(2 * channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(F.relu(x))) * mask
        y = y + self.pool(_pooled(y, cells))[:, :, None, None] * mask
        return x + self.conv2(F.relu(y)) * mask


def _pooled(x: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Mean and maximum over the board of non-negative, masked features: ``(batch, 2 channels)``."""
    return torch.cat([x.sum((2, 3)) / cells, x.amax((2, 3))], dim=1)


class SnakeNet(nn.Module):
    """The policy, win and moves heads over a shared convolutional trunk (see the module documentation)."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config = NetworkConfig() if config is None else config
        c, h = config.channels, config.hidden
        self.stem = nn.Conv2d(config.planes, c, 3, padding=1)
        self.stem_globals = nn.Linear(len(GLOBALS), c)
        self.blocks = nn.ModuleList(_Block(c) for _ in range(config.blocks))
        self.summary = nn.Sequential(nn.Linear(3 * c + len(GLOBALS), h), nn.ReLU())
        self.action_embedding = nn.Parameter(torch.zeros(3, 8))
        self.policy = nn.Sequential(
            nn.Linear(c + len(ACTION_FEATURES) + 8 + h, h), nn.ReLU(), nn.Linear(h, 1)
        )
        self.win = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.moves = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(
        self, planes: torch.Tensor, actions: torch.Tensor, globals_: torch.Tensor, head: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy logits ``(n, 3)``, win logits ``(n,)`` and moves ``(n,)`` of a batch of encodings."""
        planes = planes[:, : self.config.planes]
        mask = planes[:, _VALID : _VALID + 1]
        cells = mask.sum((2, 3)).clamp(min=1.0)
        x = (self.stem(planes) + self.stem_globals(globals_)[:, :, None, None]) * mask
        for block in self.blocks:
            x = block(x, mask, cells)
        x = F.relu(x) * mask

        # The head cell and the cells the three actions enter, read from a zero-padded map so that a move
        # off the board reads zeros.
        n, c, size, _ = x.shape
        flat = F.pad(x, (1, 1, 1, 1)).flatten(2)
        row = size + 2
        centre = (head // size + 1) * row + head % size + 1
        index = torch.stack([centre, centre - 1, centre - row, centre + 1], dim=1)
        gathered = flat.gather(2, index[:, None, :].expand(n, c, 4))
        summary = self.summary(torch.cat([gathered[:, :, 0], _pooled(x, cells), globals_], dim=1))
        per_action = torch.cat(
            [
                gathered[:, :, 1:].transpose(1, 2),
                actions,
                self.action_embedding.expand(n, 3, -1),
                summary[:, None, :].expand(n, 3, -1),
            ],
            dim=2,
        )
        logits = self.policy(per_action).squeeze(-1)
        return logits, self.win(summary).squeeze(-1), F.softplus(self.moves(summary).squeeze(-1))


def tensors(
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, ...]:
    """The arrays of :func:`~snake_agent.encoding.stack` as tensors on ``device`` (float32 inputs)."""
    planes, actions, globals_, head = batch
    return (
        torch.as_tensor(planes, dtype=torch.float32, device=device),
        torch.as_tensor(actions, dtype=torch.float32, device=device),
        torch.as_tensor(globals_, dtype=torch.float32, device=device),
        torch.as_tensor(head, dtype=torch.int64, device=device),
    )


class NetworkEvaluator:
    """Evaluates the search's queries with a network (an :data:`~snake_agent.search.Evaluator`).

    Queries are encoded on a canvas as large as the largest board among them, and evaluated at most
    ``max_batch`` at a time: on a GPU shared by many worker processes, each keeps a cache as large as its
    largest batch (a batch of 1,024 on a 20x20 canvas leaves about a gigabyte).
    """

    def __init__(self, model: SnakeNet, device: torch.device | str = "cpu", max_batch: int = 256) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch = max_batch

    @torch.no_grad()
    def __call__(self, queries: Sequence[Query]) -> Prediction:
        size = max(max(q.state.width, q.state.height) for q in queries)
        parts = []
        for i in range(0, len(queries), self.max_batch):
            config = self.model.config
            encoded = [
                encode(
                    q.state,
                    q.rules,
                    size,
                    afterstate=q.afterstate,
                    classes=q.classes,
                    cycle=config.cycle,
                    arrival=config.arrival,
                )
                for q in queries[i : i + self.max_batch]
            ]
            logits, win, moves = self.model(*tensors(stack(encoded), self.device))
            parts.append((logits, torch.sigmoid(win), moves))
        return Prediction(*(torch.cat([part[k] for part in parts]).double().cpu().numpy() for k in range(3)))

    def release(self) -> None:
        """Give the GPU memory this process cached back to the device, for other processes."""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


# --------------------------------------------------------------- checkpoints
def save(model: SnakeNet, path: str | io.BytesIO, **meta: Any) -> None:
    """Save the weights, the network's size and ``meta`` (for example the training settings)."""
    weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"config": asdict(model.config), "state_dict": weights, "meta": meta}, path)


def widen(model: SnakeNet, weights: dict[str, torch.Tensor]) -> None:
    """Load ``weights`` into ``model``, whose stem may read more input planes: the extra planes start with
    zero weights, so the network computes what it did and training can learn to use them."""
    weights = dict(weights)
    stem = weights["stem.weight"]
    width = model.stem.weight.shape[1]
    if stem.shape[1] < width:
        wider = torch.zeros(stem.shape[0], width, *stem.shape[2:], dtype=stem.dtype, device=stem.device)
        wider[:, : stem.shape[1]] = stem
        weights["stem.weight"] = wider
    model.load_state_dict(weights)


def load(path: str | io.BytesIO, device: torch.device | str = "cpu") -> tuple[SnakeNet, dict[str, Any]]:
    """The network saved by :func:`save`, and its ``meta``."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SnakeNet(NetworkConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint["meta"]


def to_bytes(model: SnakeNet) -> bytes:
    """The network as bytes, to send to worker processes."""
    buffer = io.BytesIO()
    save(model, buffer)
    return buffer.getvalue()


def from_bytes(data: bytes) -> SnakeNet:
    return load(io.BytesIO(data))[0]
