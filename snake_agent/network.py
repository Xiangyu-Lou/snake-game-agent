"""The networks: a residual CNN (M1) and a Transformer, each with a policy, a win and a moves head.

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

The **Transformer** (:class:`SnakeTransformer`, ``NetworkConfig(kind="transformer")``) keeps a short
convolutional stem at the cells, then cuts the board into ``patch`` x ``patch`` tokens (2x2, the blocks
of the certificate, by default) and lets every token attend to every other, so that a far bottleneck is
one step away rather than the depth of the trunk. Positions enter attention as rotations of the queries
and keys by the tokens' two coordinates (two-dimensional rotary embeddings), which depend only on the
tokens' offsets and so on no board size. The summary reads the head's token and the mean and maximum
over the tokens; the heads are the CNN's.

:class:`NetworkEvaluator` evaluates the search's queries with either, one board size at a time.
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
    """The network's kind and size."""

    channels: int = 64
    blocks: int = 6
    """Residual blocks: the CNN's trunk, or the Transformer's stem at the cells."""
    hidden: int = 128
    cycle: bool = True
    """Whether the inputs include the canonical cycle (see :func:`~snake_agent.encoding.encode`)."""
    arrival: bool = False
    """Whether the inputs include the arrival planes (see :func:`~snake_agent.encoding.encode`)."""
    kind: str = "cnn"
    """``"cnn"`` (:class:`SnakeNet`) or ``"transformer"`` (:class:`SnakeTransformer`)."""
    width: int = 128
    """The Transformer's token width."""
    layers: int = 6
    """The Transformer's attention layers."""
    heads: int = 4
    patch: int = 2
    """The side of a token, in cells."""
    horizon: int = 0
    """The plan agent's moves head: the moves until the board is full (0), or until this many more fruits
    are eaten (in units of :func:`~snake_agent.encoding.horizon_scale`)."""

    def __post_init__(self) -> None:
        if self.kind not in ("cnn", "transformer"):
            raise ValueError(f"unknown network kind {self.kind!r}: use cnn or transformer")
        if self.kind == "transformer" and (self.width % self.heads or (self.width // self.heads) % 4):
            raise ValueError("the token width must split into heads whose width is a multiple of 4")

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


class _Layer(nn.Module):
    """Self-attention and an MLP, each with a residual connection after layer normalization; starts as
    the identity."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        for layer in (self.out, self.mlp[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        n, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(_rotate(q, cos, sin), _rotate(k, cos, sin), v, attn_mask=mask)
        x = x + self.out(y.transpose(1, 2).reshape(n, t, d))
        return x + self.mlp(self.norm2(x))


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Two-dimensional rotary embedding: the first half of each head's features turned by the tokens'
    column, the second half by their row (``cos`` and ``sin`` from :func:`_rotations`)."""
    quarter = x.shape[-1] // 4
    a, b, c, d = x.split(quarter, dim=-1)
    turned = torch.cat([-b, a, -d, c], dim=-1)
    return x * cos.to(x.dtype) + turned * sin.to(x.dtype)


def _rotations(side: int, dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The rotary tables of a ``side`` x ``side`` grid of tokens: ``(tokens, dim)`` cosines and sines."""
    quarter = dim // 4
    frequencies = 100.0 ** (-torch.arange(quarter, device=device, dtype=torch.float32) / quarter)
    rows, columns = torch.meshgrid(
        torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij"
    )
    x = columns.reshape(-1, 1).float() * frequencies
    y = rows.reshape(-1, 1).float() * frequencies
    angles = torch.cat([x, x, y, y], dim=1)
    return angles.cos(), angles.sin()


class SnakeTransformer(nn.Module):
    """The policy, win and moves heads over a convolutional stem and attention between board patches (see
    the module documentation). Its inputs and outputs are :class:`SnakeNet`'s."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        c, h, d, p = config.channels, config.hidden, config.width, config.patch
        self.stem = nn.Conv2d(config.planes, c, 3, padding=1)
        self.stem_globals = nn.Linear(len(GLOBALS), c)
        self.blocks = nn.ModuleList(_Block(c) for _ in range(config.blocks))
        self.embed = nn.Linear(c * p * p, d)
        self.embed_globals = nn.Linear(len(GLOBALS), d)
        self.layers = nn.ModuleList(_Layer(d, config.heads) for _ in range(config.layers))
        self.norm = nn.LayerNorm(d)
        self.summary = nn.Sequential(nn.Linear(3 * d + len(GLOBALS), h), nn.ReLU())
        self.action_embedding = nn.Parameter(torch.zeros(3, 8))
        self.policy = nn.Sequential(
            nn.Linear(c + len(ACTION_FEATURES) + 8 + h, h), nn.ReLU(), nn.Linear(h, 1)
        )
        self.win = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.moves = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self._tables: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self, planes: torch.Tensor, actions: torch.Tensor, globals_: torch.Tensor, head: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy logits ``(n, 3)``, win logits ``(n,)`` and moves ``(n,)`` of a batch of encodings."""
        config = self.config
        p = config.patch
        size = planes.shape[-1]
        side = -(-size // p)
        planes = F.pad(planes[:, : config.planes], (0, side * p - size, 0, side * p - size))
        mask = planes[:, _VALID : _VALID + 1]
        cells = mask.sum((2, 3)).clamp(min=1.0)
        x = (self.stem(planes) + self.stem_globals(globals_)[:, :, None, None]) * mask
        for block in self.blocks:
            x = block(x, mask, cells)
        x = F.relu(x) * mask
        n, c, canvas, _ = x.shape

        tokens = self.embed(F.pixel_unshuffle(x, p).flatten(2).transpose(1, 2))
        tokens = tokens + self.embed_globals(globals_)[:, None, :]
        valid = F.max_pool2d(mask, p).flatten(1) > 0  # (n, tokens): patches with a cell of the board
        attend = None if bool(valid.all()) else valid[:, None, None, :]
        key = (side, str(x.device))
        if key not in self._tables:
            self._tables[key] = _rotations(side, config.width // config.heads, x.device)
        cos, sin = self._tables[key]
        for layer in self.layers:
            tokens = layer(tokens, cos, sin, attend)
        tokens = self.norm(tokens)

        # The head's token, the mean and maximum over the board's tokens; the cells the actions enter, read
        # from a zero-padded map so that a move off the board reads zeros.
        row, column = head // size, head % size
        own = tokens.gather(1, ((row // p) * side + column // p)[:, None, None].expand(n, 1, tokens.shape[2]))
        weights = valid.to(tokens.dtype)[:, :, None]
        mean = (tokens * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        top = tokens.masked_fill(~valid[:, :, None], torch.finfo(tokens.dtype).min).amax(1)
        summary = self.summary(torch.cat([own[:, 0], mean, top, globals_], dim=1))
        flat = F.pad(x, (1, 1, 1, 1)).flatten(2)
        stride = canvas + 2
        centre = (row + 1) * stride + column + 1
        index = torch.stack([centre - 1, centre - stride, centre + 1], dim=1)
        entered = flat.gather(2, index[:, None, :].expand(n, c, 3)).transpose(1, 2)
        per_action = torch.cat(
            [entered, actions, self.action_embedding.expand(n, 3, -1), summary[:, None, :].expand(n, 3, -1)],
            dim=2,
        )
        logits = self.policy(per_action).squeeze(-1)
        return logits, self.win(summary).squeeze(-1), F.softplus(self.moves(summary).squeeze(-1))


Network = SnakeNet | SnakeTransformer


def build(config: NetworkConfig | None = None) -> Network:
    """The network of ``config``'s kind."""
    config = NetworkConfig() if config is None else config
    return SnakeTransformer(config) if config.kind == "transformer" else SnakeNet(config)


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

    Queries are encoded board size by board size, each on a canvas of its own board's side, and evaluated
    at most ``max_batch`` at a time, or as many as hold ``max_cells`` cells of canvas on a large board: on
    a GPU shared by many worker processes, each keeps a cache as large as its largest batch (a batch of
    1,024 on a 20x20 canvas leaves about a gigabyte, and one of 256 on 100x100 about three). On a GPU the
    Transformer runs in bfloat16.
    """

    def __init__(
        self,
        model: Network,
        device: torch.device | str = "cpu",
        max_batch: int = 256,
        max_cells: int = 256 * 400,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch, self.max_cells = max_batch, max_cells
        self.half = str(device).startswith("cuda") and model.config.kind == "transformer"

    @torch.no_grad()
    def __call__(self, queries: Sequence[Query]) -> Prediction:
        groups: dict[int, list[int]] = {}
        for i, q in enumerate(queries):
            groups.setdefault(max(q.state.width, q.state.height), []).append(i)
        config = self.model.config
        outputs = [np.empty((len(queries), 3)), np.empty(len(queries)), np.empty(len(queries))]
        for size, rows in groups.items():
            batch = max(1, min(self.max_batch, self.max_cells // (size * size)))
            for i in range(0, len(rows), batch):
                chunk = rows[i : i + batch]
                encoded = [
                    encode(
                        queries[r].state,
                        queries[r].rules,
                        size,
                        afterstate=queries[r].afterstate,
                        classes=queries[r].classes,
                        cycle=config.cycle,
                        arrival=config.arrival,
                    )
                    for r in chunk
                ]
                with torch.autocast("cuda", torch.bfloat16, enabled=self.half):
                    logits, win, moves = self.model(*tensors(stack(encoded), self.device))
                for k, value in enumerate((logits, torch.sigmoid(win), moves)):
                    outputs[k][chunk] = value.cpu().double().numpy()  # float64 on the CPU: MPS has none
        return Prediction(*outputs, horizon=config.horizon)

    def release(self) -> None:
        """Give the GPU memory this process cached back to the device, for other processes."""
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


# --------------------------------------------------------------- checkpoints
def save(model: Network, path: str | io.BytesIO, **meta: Any) -> None:
    """Save the weights, the network's size and ``meta`` (for example the training settings)."""
    weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"config": asdict(model.config), "state_dict": weights, "meta": meta}, path)


def widen(model: Network, weights: dict[str, torch.Tensor]) -> None:
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


def load(path: str | io.BytesIO, device: torch.device | str = "cpu") -> tuple[Network, dict[str, Any]]:
    """The network saved by :func:`save`, and its ``meta``."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build(NetworkConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint["meta"]


def to_bytes(model: Network) -> bytes:
    """The network as bytes, to send to worker processes."""
    buffer = io.BytesIO()
    save(model, buffer)
    return buffer.getvalue()


def from_bytes(data: bytes) -> Network:
    return load(io.BytesIO(data))[0]
