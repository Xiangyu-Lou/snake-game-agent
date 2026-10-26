# Context Management and Rewards for the First Training Run

## Recommended starting point

Use **one current state per network call**, with no history stacking or recurrent
memory. For Architecture B, the 20×20 input remains 401 tokens regardless of
episode duration. Begin learning on 4×4, then 6×6 boards, using expert policy
pretraining and a bounded fruit-reward warm-up. Evaluate completion throughout,
then switch to the exact terminal-success objective before final comparisons.

This is an experimental starting configuration, not a trained result. It refines
the [architecture report](agent-architecture-design.md) and
[input specification](architecture-b-input-specification.md).

## 1. Model context is one state, not an episode

The current input contains the ordered body, heading, visible fruit, board size,
and any enabled termination budget. Under the chosen stochastic-fruit model,
these determine the next-state distribution for an action. Historical frames are
not required to infer motion or body order because the game supplies them directly.

With one cell token per board cell and one global token:

`context_length = W × H + 1`.

| Board | Tokens per state |
| --- | ---: |
| 4×4 | 17 |
| 6×6 | 37 |
| 20×20 | 401 |

On a 20×20 board, the first decision and the forty-thousandth decision both use
401 tokens. Increasing snake length changes their contents, not their count.
There is no growing chat transcript and no chronological position index reaching
40,000. Position encodings describe cells on the board.

Do not reuse an autoregressive KV cache by appending successive frames. This is
a bidirectional state encoder: after a move changes the board, recompute the
state representation. Exact repeated states may reuse a versioned evaluation
cache, but stale network values must be invalidated when model weights change.

### Three different lengths

| Quantity | Suggested initial setting | Meaning |
| --- | --- | --- |
| Network context | One state, `N + 1` tokens | Information visible in one policy/value evaluation |
| Collector rollout | 256 transitions per environment | Data collected before an optimizer update |
| Episode | Until completion or collision, subject to separately recorded collection limits | Source of true outcome labels |

With 64 environments and a 256-step collector, there are up to 16,384 transition
samples, not one input containing 16,384 frames. For the stateless network,
training can flatten environment/time axes into independent state examples and
evaluate them in manageable minibatches. Returns and advantages still use each
trajectory's temporal order; they must not connect unrelated episodes.

Long-horizon behavior comes from outcome targets, critic bootstrapping, and
simulator search. Increasing frame history does not by itself improve assignment
of credit to a decision thousands of steps earlier.

### MCTS memory has a separate budget

Each evaluated leaf gets its own current-state encoding. Ancestor nodes and all
simulated paths are not concatenated into the network input. Batch unrelated
leaves along the batch axis.

Control search memory with node/expansion limits, bounded caches, and subtree
reuse after actual observations. Store exact state or reconstructible transitions
for search, not every layer's neural activations. For training replay, prefer
compact state records and re-encode sampled minibatches; retain provenance and
episode boundaries needed for targets.

### When history might be useful

History becomes relevant if the input is deliberately incomplete, hidden game
mechanics must be inferred, or an agent maintains a persistent option/plan whose
identity affects its decisions. In those cases, add the missing state or a compact
plan/memory representation deliberately.

For the current game, start with zero historical frames. If testing history later,
compare one, two, and four frames at matched resources, with frame/action labels
and reset masks. Eight snapshots with one global token each would contain 3,208
tokens on 20×20; dense attention scores grow by about 64 times relative to a
single snapshot, before considering other costs. That expense currently buys
largely duplicated information.

## 2. First-run rewards: bounded progress as a warm-up

For the first policy-learning experiment, use:

`r = (β / N) × ate_fruit + 1 × won - 1 × collided`, with `β = 0.2`.

Ordinary movement receives zero reward. The final fruit earns both the fruit
reward and the win bonus, matching the repository's reward implementation.

| Event | Warm-up reward |
| --- | ---: |
| Move without eating or ending the game | 0 |
| Eat an ordinary fruit | `0.2 / N` |
| Collide with a wall or the body | -1 |
| Eat the final fruit and fill the board | `1 + 0.2 / N` |

For 4×4, each fruit gives `0.0125`; for 6×6, approximately `0.00556`; for 20×20,
`0.0005`. The relatively small per-fruit value is intentional: even all fruits
combined contribute less than 0.2 over an episode. This avoids allowing hundreds
of fruit points to dwarf a unit terminal reward.

Normalize by the board area, not by the length at an arbitrary curriculum start.
Board area is already in the observation contract. Reward scaling by a varying
episode-start length would require that extra context to keep the return critic's
inputs sufficient across mixed curriculum starts.

For a finite episode starting at length `L0`, with `gamma=1`:

- A completed game returns `1 + β × (N - L0) / N`, between 1 and 1.2.
- A collision after eating `k` fruits returns `-1 + β × k / N`, between -1 and -0.8.

For the standard 20×20 start, the winning return is `1.1985`. For a 4×4 start with
length three, it is `1.1625`. Thus no individual failed episode earns more total
return than a completed episode.

**This is still a surrogate objective.** For fully resolved episodes its expected
return is `2 × P(win) - 1 + β × E(fruits eaten) / N`. Progress can therefore alter
the ranking of policies with similar win probabilities. It is not mathematically
equivalent to maximizing completion probability. Unfinished loops must also be
handled separately; negative death reward can encourage postponing failure.

### Why a collision penalty of -1 is not weak

The penalty must be judged against the other rewards, not the literal size of
the number. Here, an ordinary fruit on 20×20 earns `0.0005`, all available fruit
rewards sum to less than `0.2`, and the completion bonus is `+1`. A collision
penalty of `-1` is therefore large relative to the entire progress bonus.

Termination also removes the opportunity to obtain future rewards. A return
critic uses zero future bootstrap at a true terminal transition. For example,
if it previously estimated `V_return(s) = 0.8`, a collision gives the one-step
TD error `-1 + 0 - 0.8 = -1.8`. Do not add another penalty to represent those
forgone rewards; the terminal target already captures the loss of continuation.
This applies to the warm-up return critic, not the separate probability head.

Increasing only the collision penalty to `-100` changes the optimization scale
and its balance with the dense bonus; it does not supply new information about
which earlier decision caused the trap. In an ideal undiscounted task with only
terminal rewards `+1` and `-C`, where all episodes eventually resolve, expected
return is `(1 + C) × P(win) - C`. Any positive `C` gives the same optimal ordering
by win probability. Dense rewards, nonterminating episodes, and approximate
learning complicate that equivalence.

Multiplying every reward by the same positive constant likewise preserves the
ideal return ordering, but can change gradient scales, value losses, and their
balance with entropy regularization in a practical trainer. Start with the
bounded configuration above and inspect real failures before increasing penalties.

## 3. Map the reward to the current repository

The warm-up needs no game-rule change:

```python
from snake_game import SnakeEnv
from snake_game.env import Rewards

w, h = 4, 4
env = SnakeEnv(
    width=w,
    height=h,
    obs_type="grid",
    action_mode="relative",
    max_steps_without_fruit=None,
    render_mode=None,
    rewards=Rewards(fruit=0.2 / (w * h), death=-1.0, win=1.0, step=0.0),
)
```

The same reward configuration is supported by `VectorSnakeEnv`. When changing
board size, construct the appropriate per-board coefficient. Do not change a
live episode's reward definition halfway through collection.

Warm-up returns are not probabilities and may be negative or above one. If using
PPO, its reward-return critic must have an unrestricted scalar output. Keep a
separate sigmoid `V_win` head trained on resolved 0/1 outcomes if estimating win
probabilities during warm-up. A sigmoid cannot correctly represent the warm-up
return target.

For full Architecture B, MCTS values and backups should use the explicitly chosen
win-probability target. Do not quietly add warm-up fruit rewards to those backups
while calling the result a probability. Search visit counts remain policy targets;
actual completed outcomes supply the win-value labels.

## 4. Move to the exact completion objective

After the warm-up establishes small-board completions, compare a strict-objective
checkpoint using:

```python
Rewards(fruit=0.0, death=0.0, win=1.0, step=0.0)
```

With undiscounted success returns, a winning episode returns one and a losing
episode returns zero. The expected return is the probability of completion.
Using zero death reward does not tell the agent that collision is desirable:
collision ends the opportunity to obtain the success reward.

Suggested progression is `β = 0.2 → 0.05 → 0`, followed by the strict configuration
above, which also changes death reward to zero. These are separate recorded
training phases, not an unlogged drifting objective. Clear stale PPO advantages
and adapt/reinitialize the reward-return critic when changing reward semantics.
Keep win labels and evaluation metrics consistent across phases.

If strictly aligned training is required from the first update, use this terminal
success configuration immediately, together with expert policy pretraining and
short, known-solvable curriculum starts. Its sparse feedback is a reason to improve
data and curriculum, not to claim that normalized fruit rewards preserve the same
objective.

## 5. Practical first experiment

1. **Validate the state/action path.** Use one current state, three relative
   actions, and exact immediate-action masking. Treat the all-fatal case explicitly;
   the wrapper's nonempty fallback mask is not a safety certificate.
2. **Pretrain the policy.** A proposed starting dataset is 200 completed 4×4
   Hamiltonian episodes with separate training seeds. Use these for action
   imitation. Expert success does not calibrate the learner's own success value;
   train that value on the appropriate policy's continuations and outcomes.
3. **Run a learning baseline.** Use the proposed encoder without search initially
   so search cannot conceal policy weaknesses. Try 64 environments, 256 collected
   steps per update, PPO clipping 0.2, AdamW learning rate `3e-4`, and the warm-up
   rewards. Minibatch size should fit measured accelerator memory; start at 64
   state examples and adjust. These are hypotheses to test, not measured settings.
4. **Use long-horizon-aware targets.** Start with `gamma=1` and GAE `lambda=0.99`
   on the small board. Lambda still limits credit: with gamma one, its rough
   trace scale is 100 steps, versus 20 at lambda 0.95. Sweep longer traces as board
   size grows; train/calibrate critics using resolved episodes. Gamma one alone
   does not solve sparse outcomes or inaccurate bootstrap values.
5. **Validate and scale.** A proposed curriculum gate is at least 90 completions
   out of 100 validation episodes on the current board, with censored episodes
   counted separately and not silently excluded. Then try 6×6 and later sizes,
   retaining some earlier boards. Keep final test seeds untouched. This gate is
   a development criterion, not evidence of near-perfect reliability.
6. **Introduce search and the strict target.** Compare search/no-search with the
   same encoder and compute accounting. Train Architecture B's prior on search
   distributions and its win value on actual outcomes. Select by held-out
   completion, not the warm-up reward score.

Whenever feasible, add late-game states from verified successful trajectories,
as well as relevant learner failures. Custom-state curriculum currently uses
`SnakeEnv.reset(options={"state": ...})`; the vector environment does not provide
the same arbitrary-state injection interface.

## 6. Loops, truncation, and reward shortcuts

Do not add a positive survival reward: a non-progressing loop could collect it
forever. Do not add a default per-step penalty: it introduces a move-efficiency
objective and can penalize valid long solutions. A head-to-fruit distance bonus
also rewards choices that may enter traps, while safe detours may initially move
away from the fruit. None is needed for the first experiment.

With the proposed zero movement reward, loops do not generate additional reward,
but they can still occur. Track time since fruit and repeated complete states as
diagnostics. Improve exploration, expert/curriculum coverage, and search rather
than assuming a reward table alone eliminates looping.

Set the game's fruit timeout to `None` for the principal task. Use an explicit
collector cap if necessary to keep experiments finite, and record capped episodes
as censored. Bootstrap the return critic from the final observed state at a
collection boundary; do not give it the terminal collision penalty. Keep the
win label unresolved unless the game actually wins or loses. A bootstrapped value
is an estimate, not observed evidence of success.

Vector autoreset also needs care. With `NextStep`, the reset-only transition
ignores the supplied action and should not be treated as a policy decision.
With `SameStep`, the returned main observation may already belong to the next
episode; preserve the original terminal observation/info for targets. Rewards and
advantages must not flow across those episode boundaries.

## Validation performed

Five direct transitions were checked with the existing `SnakeEnv`:

- Ordinary movement receives zero warm-up reward.
- Eating on 6×6 receives `0.2 / 36`.
- Collision receives -1 and terminates.
- The last fruit on 2×2 receives `1 + 0.2 / 4 = 1.05` and wins.
- The same win with the strict reward configuration receives exactly one.

All five checks passed, with fruit timeout disabled. They establish the reward
API's semantics, including addition of fruit and terminal bonuses. They do not
establish that a trained agent will learn successfully with these hyperparameters.
