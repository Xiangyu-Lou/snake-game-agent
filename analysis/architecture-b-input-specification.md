# Architecture B: Input Specification

## Recommended representation

Represent the current board as **one numerical token per cell, plus one global
token**. For a 20×20 board, this produces 401 Transformer tokens. A token here is
a learned projection of structured numbers, not a word or an image patch.

The input describes a single observed state. The existing ordered snake body
already supplies the information that frame history would otherwise need to
recover. The network predicts an action prior and a value; MCTS obtains future
states from the exact simulator and encodes each evaluated leaf in the same way.

Use [`GameState`](../snake_game/engine.py) and the existing
[`observe(state, "grid")`](../snake_game/observations.py) as the canonical sources.
Keep a separate exact state object for simulation. Neural tensors must not become
the simulator's source of truth.

This specification refines [Architecture B](agent-architecture-design.md).
It is a proposed input contract, not an implemented or trained model.
See [context management and initial rewards](context-and-initial-rewards.md)
for the distinction between one-state context, training trajectories, and
search memory, together with a first-run reward configuration.

## 1. Cell features: 12 numbers per cell

Let `W` and `H` be board dimensions, `N = W × H`, and `L` be snake length. Flatten
cells in row-major order: `i = y × W + x`. Coordinates use the repository's
screen convention: right increases `x`; down increases `y`.

| Index | Feature | Encoding |
| --- | --- | --- |
| 0 | Head | 1 at the head, otherwise 0. |
| 1 | Body excluding head | 1 on other occupied cells, including the tail. |
| 2 | Current fruit | 1 at the visible fruit, otherwise 0. |
| 3 | Body release order | `(L - k) / N` on body segment `k`, where head index is 0; 0 on free cells. |
| 4–7 | Segment direction | One-hot in `UP, RIGHT, DOWN, LEFT` order; all zeros on free cells. |
| 8 | Tail | 1 at the tail, otherwise 0. |
| 9 | Horizontal coordinate | `x / max(W - 1, 1)`. |
| 10 | Vertical coordinate | `y / max(H - 1, 1)`. |
| 11 | Valid cell | 1 for a real board cell, 0 for batch padding. |

Features 0–7 are exactly the existing grid channels. A tail flag, coordinate
planes, and a validity flag are deterministic additions. Flags are not mutually
exclusive: a tail is also a body cell, and a length-one snake is both head and tail.

For non-head segments, direction points toward the preceding segment; it is not
a forecast of future motion after arbitrary turns. At the head it is the current
heading, as defined by `GameState.segment_directions`.

### Why body order is essential

A binary occupied/free map does not generally identify the order of an arbitrary
self-touching body. The model needs to distinguish cells that will vacate soon
from cells that remain occupied for many steps. The release-order channel
preserves that order: occupied cells sorted by decreasing value reconstruct the
head-to-tail sequence. The explicit tail flag makes immediate tail entry easy
to recognize.

For a length-five snake on 20×20, the head-to-tail values are:

```text
segment index:       0       1       2       3       4
release order:   5/400   4/400   3/400   2/400   1/400
```

These are release times **only if no fruit is eaten**. Growth delays tail movement,
and revisiting a cell can occupy it again. Exact collision checks and rollout
transitions remain the simulator's responsibility.

All free cells are included. Ordered body coordinates plus board dimensions and
fruit would also describe the geometry, but cell tokens make available space and
spatial relations directly accessible to the network. There is no need to supply
a second, duplicate sequence of body tokens in the first implementation.

## 2. Global features: 10 numbers

Build a separate vector, then project it into the global token:

| Index | Feature | Encoding |
| --- | --- | --- |
| 0–1 | Board dimensions | `W / 20`, `H / 20`; 20 is a fixed scaling reference, not a size limit. |
| 2 | Occupancy fraction | `L / N`. |
| 3–6 | Head direction | One-hot `UP, RIGHT, DOWN, LEFT`. |
| 7 | Fruit timeout enabled | 0 when disabled, otherwise 1. |
| 8 | Configured timeout | `T / N` when enabled, otherwise 0. |
| 9 | Remaining timeout | `max(T - steps_since_fruit, 0) / N` when enabled, otherwise 0. |

The recommended task disables the fruit timeout, so fields 7–9 are zero. If it is
retained, obtain `T` from the environment configuration; it is not a `GameState`
field. Including both configured and remaining budget avoids conflating different
timeout rules that happen to have the same remaining fraction.

Total elapsed moves and score are not necessary for the proposed win-only,
no-step-penalty objective. If a hard episode horizon becomes part of the game
rules, include its remaining budget too. A collector's computational cutoff is
not automatically such a rule.

Under the intended stochastic-fruit model, this is sufficient public state for
the specified task. It does not encode the hidden PRNG stream needed for bitwise
prediction of future fruits. Planning seeds, live RNG keys, future fruit locations,
and training episode identifiers are excluded from model inputs by the chosen
information contract.

## 3. Tensor shapes and projection

With batch size `B` and a fixed 20×20 board:

```text
Existing symbolic grid       [B,   8, 20, 20]
Flatten and append features  [B, 400, 12]
Cell projection 12 → 128      [B, 400, 128]

Global features              [B,  10]
Global projection 10 → 128    [B,   1, 128]

Prepend global token         [B, 401, 128]
6 Transformer blocks         [B, 401, 128]
```

A linear projection is an adequate starting point. Add learned cell/global type
embeddings to distinguish token roles. The projected coordinate features supply
absolute position; an additional fixed 400-entry position table is unnecessary
and would make board-size transfer less convenient.

For the policy head, concatenate the output global token and output head-cell
token, then use an MLP to produce three relative-action logits. The head token is
at index `1 + head_y × W + head_x`. The global token can also feed sigmoid win
value and auxiliary occupancy heads.

This is bidirectional state attention, not autoregressive sequence generation:
every valid cell may attend to every other valid cell. No causal attention mask
is needed.

## 4. Relational attention

Coordinates and release order are sufficient for an initial encoder. The proposed
relational variant additionally makes geometry and body connections explicit as
soft attention biases. Derive these relations from integer coordinates and the
ordered body, rather than trying to recover them from rounded neural tensors.

For query cell `i` and key cell `j`, construct:

```text
r(i,j) = [
    (x_j - x_i) / max(W - 1, 1),
    (y_j - y_i) / max(H - 1, 1),
    are_grid_neighbors,
    j_is_body_predecessor_of_i,
    j_is_body_successor_of_i,
]
```

If segment `i` has body index `k`, its predecessor has index `k - 1` and successor
has index `k + 1`. Relations are false if either cell is unoccupied or the relevant
endpoint does not exist. Board adjacency refers to physical neighboring cells,
not whether moving there is currently legal.

A small MLP maps these five features to one scalar per attention head:

`attention_score_h(i,j) = Q_h(i) · K_h(j) / sqrt(d_head) + bias_h(r(i,j))`.

Use learned type-specific constants for global↔cell and global↔global biases;
the global token has no board coordinate. These are soft biases: do not prevent
attention to distant cells or to occupied cells. Both can determine a long-term
trap. Begin with coordinate-only attention and compare the relational biases in
an ablation before assuming they improve performance.

For 8 heads, a dense bias tensor has shape `[B, 8, 401, 401]`. Include this and
attention activations in memory budgeting. For substantially larger boards,
consider sparse/local attention with global communication instead of scaling
dense attention indefinitely.

## 5. Masks and variable board sizes

Keep two distinct masks:

- **Padding mask:** `[B, N_max + 1]`, with `True` meaning padding is ignored by
  attention. The global token is always valid. Pack each actual board's `W × H`
  row-major cells first, then append zero-feature padding up to `N_max`.
- **Immediate-safe action mask:** `[B, 3]`, ordered left, straight, right. Compute
  it exactly from the state, separately from learned representations.

Normalize coordinates using each example's own dimensions, not the largest board
in the batch. Set the validity feature to zero only on padding. Exclude padded
keys from attention and ignore padded query outputs in later gathers/pooling.
The validity feature alone does not implement attention masking.

The engine allows entry into the current tail cell because it vacates during a
non-growing move. Use engine legality checks to preserve this rule. The existing
Gymnasium mask has a nonempty fallback when all moves are fatal; that fallback
must not be interpreted as evidence of safety. Detect the empty genuinely-safe
set before applying a masked softmax, avoiding an all-negative-infinity row.

Terminal game states bypass network inference: a full board has known win value
one, and a collision has value zero. Wrapper truncation and search-budget cutoffs
must retain their separately defined semantics.

## 6. Reference construction and worked example

The following numerical construction reuses the repository's existing encoder.
It is a specification example, not a change to application code:

```python
import numpy as np
from snake_game import observe

def encode_cells(state):
    w, h = state.width, state.height
    n = w * h
    cells = np.zeros((n, 12), dtype=np.float32)
    cells[:, :8] = observe(state, "grid").reshape(8, n).T
    tx, ty = state.tail
    cells[ty * w + tx, 8] = 1
    yy, xx = np.indices((h, w))
    cells[:, 9] = xx.ravel() / max(w - 1, 1)
    cells[:, 10] = yy.ravel() / max(h - 1, 1)
    cells[:, 11] = 1
    return cells

def encode_global(state, timeout=None):
    n = state.width * state.height
    features = np.zeros(10, dtype=np.float32)
    features[:3] = (state.width / 20, state.height / 20, state.length / n)
    features[3 + int(state.direction)] = 1
    if timeout is not None:
        assert timeout > 0
        features[7:] = (
            1,
            timeout / n,
            max(timeout - state.steps_since_fruit, 0) / n,
        )
    return features
```

For a valid example with body
`[(10,10), (9,10), (8,10), (8,11), (8,12)]`, heading `RIGHT`, and fruit `(15,4)`:

- The head-cell vector is approximately
  `[1, 0, 0, 0.0125, 0, 1, 0, 0, 0, 0.5263, 0.5263, 1]`.
- The tail-cell vector is approximately
  `[0, 1, 0, 0.0025, 1, 0, 0, 0, 1, 0.4211, 0.6316, 1]`.
- Global features with no timeout are
  `[1, 1, 0.0125, 0, 1, 0, 0, 0, 0, 0]`.

The tail's `UP` direction points to the segment at `(8,11)`. The head's `RIGHT`
direction records its current heading. The example therefore illustrates both
the body order and the local direction conventions.

The field construction was checked against the live repository on this bent-body
example and six additional configurations: 4×4, 8×5, 5×8, 20×20, 1×4, and 4×1.
Checks confirmed feature counts, finite values, and reconstruction of the exact
ordered body, with timeout features enabled and disabled on those six boards.
These are encoding checks, not evidence of trained-model performance. Transformer
projection, attention biases, and batch padding remain proposed implementation.
