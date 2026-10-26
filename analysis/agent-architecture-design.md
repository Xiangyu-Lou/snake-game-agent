# Architectures for a Snake Agent That Fills the Board

## 1. Recommendation and scope

Build a **policy/value network guided by exact simulator search** as the main
learning architecture. First establish a smaller **residual CNN with imitation
learning and masked PPO** as a measurable learning baseline. Keep the existing
Hamiltonian controller as a reference and an optional component with explicitly
checked applicability. A graph-based hierarchical planner is a later alternative
if experiments show that primitive-action search cannot handle the long horizon.

The objective is to maximize the probability of filling the board. The game
advances only after an agent action. Neither visual perception nor a real-time
response deadline is required. Move count is a diagnostic, not an optimization
target. Extra thinking time is available, but training cost, search memory, and
finite experiment budgets still matter.

This report proposes four architectures. **No neural model has been trained or
benchmarked here.** The measurements below concern existing repository policies
and observation interfaces. All suggested network sizes and training settings are
starting configurations, not validated optimums.

## 2. What is already solved, and what remains difficult

### The current default game has a constructive solution

The default board is an obstacle-free 20×20 rectangle with a standard horizontal
snake of length three. A rectangular grid with both dimensions at least two has
a Hamiltonian cycle when at least one dimension is even. If the ordered snake
body follows consecutive cells of that directed cycle, continuing along it
avoids collisions and eventually visits every possible fruit location.

The repository already implements this idea in `HamiltonianPolicy`, and tests
cover the default board and several other rectangles. Thus, **reliable completion
of the default game does not require solving an unrestricted hard planning
problem or training a large model**. A learned agent is still a meaningful goal;
its performance should be compared with this constructive solution.

For an aligned fixed cycle, let `N = width × height`, `L` be snake length, and
`F = N - L`. The free cells form the forward arc immediately after the head.
The current fruit is reached in at most `F` moves, after which `F` decreases by
one. Completion therefore needs at most `F(F + 1)/2` further moves. Under ideal
uniform fruit placement, the expected total is `F(F + 3)/4`. At the default
start, these are **79,003** and **39,700** moves respectively. The expectation
uses the intended random-fruit model; the bound does not depend on fruit order.

The guarantee is conditional on alignment and the stated board rules. The
implementation's fallback from a non-aligned body is a heuristic, not a proof of
recovery. Odd-by-odd boards have no Hamiltonian cycle, but that does **not** imply
they can never be filled. Arbitrary valid states may already be impossible to win.

### The learning problem still has a long horizon

Unconstrained policies must account for body order, tail motion, growth, narrow
passages, and future fruit placement. A locally safe move can create a trap many
moves later. A good average score can coexist with zero completed games. A search
with up to three non-reversing actions per step can also grow exponentially with
depth, even though many actual states have fewer choices.

Board-size transfer, odd-sized boards, and recovery from non-aligned states are
useful separate evaluations. Obstacles would require new game functionality and
are outside the present implementation. None of these extensions should silently
replace the requested default-board completion objective.

## 3. Repository findings and measured baselines

| Existing component | Consequence for the design |
| --- | --- |
| [`SnakeGame.step`](../snake_game/engine.py) | One call makes one move; use it directly without the real-time controller. |
| `GameState.snake` | Ordered head-to-tail coordinates expose the complete body geometry. |
| `clone()`, `reseed()`, `from_state()` | Exact rollouts and independent future-fruit sampling are already available. |
| [`SnakeEnv`](../snake_game/env.py) | Supports absolute or relative actions, configurable rewards, and custom starting states. |
| [`VectorSnakeEnv`](../snake_game/vector_env.py) | Batches standard-start games; its reset API currently supports masks/seeds, not arbitrary `GameState` injection. |
| [`observations.py`](../snake_game/observations.py) | Supplies symbolic grids, ordered coordinates, and a compressed feature vector. |
| [`policies.py`](../snake_game/policies.py) | Supplies random, greedy, and Hamiltonian references; these are not learned agents. |
| [`test_policies.py`](../tests/test_policies.py) | Checks cycle completion and illustrates a fruit trap. |

The accompanying [probe](architecture_probe.py) ran each existing policy on
20×20 boards, initial length three, seeds 0–5, using `SnakeGame` directly.
The measurement cap was 160,000 actions per episode, without Gymnasium's fruit
timeout. All 18 episodes terminated before the cap.

| Policy | Completed games | Mean fruit score | Mean moves per episode |
| --- | ---: | ---: | ---: |
| Random, immediate-safe moves | 0/6 | 4.83 | 1,996.67 |
| Greedy | 0/6 | 111.50 | 2,431.83 |
| Hamiltonian | 6/6 | 397.00 | 39,804.00 |

These six seeds are illustrative, not a statistical claim of general superiority
or perfect reliability. The Hamiltonian guarantee comes from its invariant under
the conditions above, not from six successes. Raw episodes, the source commit,
and reproducible interface examples are in
[`architecture_probe_results.json`](architecture_probe_results.json).

The probe also confirmed two important interface limitations:

1. **The 19-dimensional features are lossy.** Two valid 6×6 states with the same
   head, tail, fruit, length, heading, and local feature vector have different
   intermediate body paths. Their full grids differ. These features alone are
   unsuitable for a general planner that needs the actual body configuration.
2. **A nonempty Gymnasium action mask does not prove a safe action exists.** When
   every action is fatal, the wrapper deliberately marks non-reversing actions
   valid to keep policy distributions defined. Search must check
   `game.safe_directions()` or simulate the actions explicitly.

## 4. Shared state, action, and objective contracts

### State representation

Use the ordered body, current fruit, heading, and board dimensions as the canonical
public state. A symbolic grid is structured game data; a convolution over that
grid does not introduce image recognition or rendering into the project.

For a grid encoder, retain all eight existing channels: head, body, fruit,
normalized body countdown, and four segment directions. Add coordinate planes
and a valid-cell plane when padding mixed board sizes. These additions are
deterministic preprocessing. Global features should include dimensions, length
fraction, and, if a fruit timeout remains enabled, the remaining timeout budget.

The countdown is the number of moves until a cell vacates **if no fruit is eaten**.
Growth delays release. Do not treat countdowns as an unconditional future schedule.
The current grid and coordinate observations omit `steps_since_fruit`; it is in
`GameState` and `info`. Omitting it while retaining a timeout hides a part of the
task state from the model.

For a token or graph encoder, use one node per board cell with the same semantic
features plus body rank. Preserve empty cells: they define the available routes.
Never encode only head-to-fruit distance, occupancy without order, or occupied
cells without a representation of the free space.

### Actions and interaction

Use three relative actions: **left, straight, right**. Convert them to the existing
absolute `Direction` at the public `policy(GameState) -> Direction` boundary.
This avoids a fourth reverse action that the engine silently executes as straight.
The UI can already load such a callable via `--agent module:attribute`.

Decision flow: receive state → encode → propose/search → return one action →
advance the game once → observe again. Internal rollouts operate on independent
simulators and do not advance the live game. A recurrent memory is optional for
search bookkeeping, not required to reconstruct the fully observed body.

### Completion-first learning

The primary value is

`V_win(s) = P(eventually fill the board | s, the deployed decision procedure)`.

A matching terminal reward is available through
`Rewards(fruit=0, death=0, win=1, step=0)`. It is sparse, so use expert trajectories,
curriculum starts, and auxiliary prediction tasks to make learning practical.
Default `+1 fruit / -1 death` instead optimizes a score-related return; it does not
directly encode the probability of completion. If used for early exploration,
label it as a different training phase and select final models by completion.
The [initial-training specification](context-and-initial-rewards.md) gives a
bounded progress-reward warm-up and an explicit transition to the strict target.

Use undiscounted episodic success targets where feasible. On a 39,700-move game,
`0.99 ** 39700 ≈ 5.2e-174`; a conventional short-horizon discount effectively erases
the terminal objective. Even `0.999` gives about `5.6e-18`. Setting `gamma=1`
avoids that attenuation but does not solve sparse feedback or estimator variance.
In PPO, a small GAE lambda still shortens credit assignment and depends heavily on
a well-trained critic; sweep it explicitly and anchor the critic with completed
episode returns. Avoid an action penalty because speed is not a user objective.

Keep final occupancy as an auxiliary target and diagnostic. A fixed weighted sum
of win probability and occupancy is not generally equivalent to strict priority
for wins. If occupancy breaks near-ties, document that approximation explicitly.

For the principal task, set `max_steps_without_fruit=None`. The default is 400
steps on 20×20, an additional rule absent from the engine. Resource caps can still
end data collection; distinguish these censored episodes from deaths. Bootstrap
at collector boundaries instead of assigning them a false terminal-loss label.
Track unresolved loops separately rather than inventing evidence of eventual wins.

## 5. Architecture A: residual grid network with imitation and masked PPO

**Purpose:** the simplest serious learned baseline and a reusable encoder for
later search.

```text
8 semantic planes + x/y + valid-cell plane
                 ↓
3×3 convolution, 64 channels
                 ↓
10 residual blocks, 64 channels, two 3×3 convolutions each
                 ↓
head/tail/fruit feature gathers + masked global mean + global scalars
                 ↓
shared MLP, 256 hidden units
                 ↓
3 action logits | win-probability value | auxiliary final occupancy
```

Use GroupNorm and SiLU in the blocks, preserve spatial resolution, and mask padded
cells during pooling. The 21 convolutional layers provide a 43×43 theoretical
receptive field, sufficient to span the initial 20×20 board from any head location.
For larger boards, enlarge the receptive field or add global attention; pooling
alone does not ensure that detailed distant geometry is represented adequately.

**Training:** collect Hamiltonian successes and greedy/random failures; pretrain
the policy on applicable expert actions and initialize outcome prediction from
both outcomes. Expert success shows that a continuation exists, not that the
current learner will execute it successfully. Record the behavior policy for each
trajectory and recalibrate the deployed-policy critic on its own continuations;
do not treat mixed-policy labels as calibrated learner win probabilities.
Then perform masked PPO from learner-generated trajectories. Add expert
relabeling only where the expert's continuation is valid. Learning exclusively
from successful aligned cycles cannot teach recovery from arbitrary states.

A starting PPO setup is AdamW at `3e-4`, clip ratio `0.2`, 64–256 parallel standard
environments, and 256 collected steps per environment before an update. These are
collector chunks, not episode limits. Use recent learner-completed episodes to
improve the win critic, keeping PPO policy updates on fresh on-policy batches;
older/expert episodes can support separate supervised auxiliary losses.
Evaluate longer credit-assignment settings on small boards
before scaling. Keep validation seeds separate from rollout seeds.

**Strengths:** low implementation complexity; uses the current vector environment;
cheap inference and a clear baseline. **Failure mode:** a policy can imitate
ordinary moves while making rare catastrophic mistakes near full occupancy.
For illustration, an independent per-move failure rate of `1e-4` would leave only
about `exp(-4) ≈ 1.8%` survival across 40,000 moves. Independence is an illustrative
assumption, not a model of measured errors. High action accuracy is insufficient.

**Use when:** establishing whether learning works at all and measuring how much
search subsequently contributes. Do not interpret an immediate action mask as a
long-term safety guarantee.

## 6. Architecture B: relational policy/value network with stochastic MCTS

**Purpose:** the recommended main research architecture, using extra computation
per action and the existing exact simulator.

See the [input specification](architecture-b-input-specification.md) for exact
cell features, global features, tensor shapes, masks, and relational encodings.

```text
Public state → one token per cell + global token
                           ↓
6 Transformer blocks; width 128; 8 heads; FFN width 512
                           ↓
policy prior P(a|s), V_win(s), auxiliary final occupancy
                           ↓
MCTS decision nodes ↔ exact SnakeGame transitions ↔ fruit chance nodes
                           ↓
root visit distribution → one real action → observe → re-root
```

Each cell token contains coordinates, occupancy, body rank/countdown, segment
heading, and head/tail/fruit indicators. Add attention biases for displacement,
board adjacency, and predecessor/successor body relations. A 20×20 board uses
401 tokens including the global token. Global attention makes distant bottlenecks
visible immediately, but its attention cost grows quadratically with board area.
Reuse A's encoder initially to isolate the benefit of search before comparing
the Transformer; do not change search and encoding simultaneously in an ablation.

### Search semantics

At a decision node, use a masked PUCT rule such as

`Q(s,a) + c * P(s,a) * sqrt(sum_b visits(s,b)) / (1 + visits(s,a))`.

`Q` is backed up on the same success-probability scale throughout the tree.
This is a single-player task: **do not alternate value signs** as in adversarial
game search. Forced single-safe-action sequences can be simulated without invoking
the network at each step. Stop expansion at a terminal state, a depth budget, or
a leaf to be evaluated. A depth cutoff receives a value estimate, not a loss label.

Movement and growth are known exactly. Fruit spawning creates chance outcomes:
sample them from the game's intended distribution and average their values;
never choose the most favorable fruit as though it were an agent action.
With many free cells, use sampled outcomes/progressive widening. Near the end,
enumerating the few remaining fruit locations may be affordable.

The default information contract excludes the live game's RNG key and future
fruits. `clone()` copies that key, so a planner must not exploit an unmodified
live clone to predict the actual future. Construct a simulator from the public
state with an independent planning seed, or reseed a clone before simulation.
Keep the visible current fruit, sample independent future streams, and aggregate
by observed state/chance outcome. Do not solve each sampled future with advance
knowledge and then pretend that clairvoyant value is achievable. A known-RNG
oracle is a separate optional experiment, not the main evaluation.

Cache using the ordered body, heading, current fruit, board configuration, and
any active termination budget. Body occupancy alone is not a valid key. After a
real fruit spawn, reuse only the subtree matching the actual observation, or
start a new root. Keep the selected chance outcomes and value-model version
consistent when reusing statistics.

### Learning from search

Store `(state, root visits, final outcome, final occupancy, provenance)` from
search-controlled games. Train with

`loss = CE(search policy, network policy) + λv * BCE(z_win, V_win) + λa * auxiliary_loss`.

Use real completed outcomes for `z_win`; mark bootstrapped or budget-limited
targets as estimates. Initially use expert games to obtain positive examples,
then increasingly train on search's own distribution. Include losses and hard
near-full states rather than filling replay only with easy cycle trajectories.
Record which policy/search budget produced each outcome; success under a strong
expert must not be reported as the standalone neural policy's success probability.
Frozen evaluator checkpoints should control selection to reduce self-confirming
optimistic values.

Start with 64, 256, and 1,024 expansions per decision as an evaluation sweep;
increase to 4,096 only if it yields a useful improvement. Batch leaf evaluations,
reuse trees, and report simulator steps separately from neural evaluations.
At roughly 40,000 real moves, even 256 expansions per move imply about ten million
expansions per completed game before reuse or forced-move savings. The absence
of a real-time constraint does not remove this cost.

**Strengths:** detects tactical traps, uses exact dynamics, and can improve with
more computation. **Failure modes:** limited depth misses eventual traps; a
miscalibrated critic overvalues them; fruit outcomes increase branching. Search
and a learned value are not a proof of completion.

## 7. Architecture C: graph network with hierarchical route planning

**Purpose:** reduce the effective decision horizon by reasoning over routes and
regions while retaining exact primitive-action execution.

Use one node per cell and typed edges for four-neighbor board adjacency and
ordered-body predecessor/successor relations. Node features include the common
state features and release order. Start with six message-passing layers of width
128 and a global attention/pooling node; local message passing alone cannot
communicate across a large board in six layers.

A small Transformer decoder, two layers of width 128, scores a bounded set of
proposed options using the graph embedding. Candidate options can include a
time-dependent route toward the fruit, a moving-tail route, or a compatible
cycle continuation. Generate a mix of deterministic search candidates and learned
proposals; otherwise the model cannot select a route that the generator omits.
Outputs are an option distribution, per-option success estimates, and an overall
win value. The low-level controller executes the selected route one cell at a time.

Simulate each option with the actual engine. A static shortest path ignores tail
motion; a static flood-fill area ignores future body release. In particular,
requiring reachable free area to exceed snake length would reject some winnable
near-full cycle states. These quantities can be useful features, not general
proofs. Validate geometry, growth, and fruit-dependent events along the route.

End an option when its waypoint is reached, a fruit is eaten, the plan becomes
invalid, or its budget is exhausted. Re-observe before continuing. Train the
high-level policy with option-level imitation and actor-critic or search targets;
assign credit across the actual number of primitive moves. Store option durations
even when `gamma=1`, so experiments with other discounts remain well-defined.

**Strengths:** explicit body topology, longer decisions, and potentially better
transfer across board sizes. **Failure modes:** valid local options need not
compose into a winning global plan; narrow candidate sets hide solutions;
hierarchical training and data collection add substantial complexity. Graph
structure is an inductive bias, not a guarantee of generalization.

**Use when:** B's primitive search fails primarily because of horizon and repeated
route planning, with evidence that longer validated options improve the tradeoff.
It is not the first implementation to build.

## 8. Architecture D: learned proposal ranking with a cycle certificate

**Purpose:** separate learned preferences from a constructive completion guarantee
on the part of the state space where such a certificate is available.

A small grid or graph encoder scores candidate directed cycles. The proposals
can begin with the repository's constructed orientations. A deterministic verifier
checks that a candidate visits every board cell exactly once, closes using legal
neighbor edges, and contains the current body in directed tail-to-head order.
Also check action legality and the supported board/configuration assumptions.

Select only a verified candidate. **Commit to its successor actions until the
next fruit is eaten**, then reconsider. This provides progress: simply switching
among individually safe cycles at every step could postpone fruit indefinitely.
For a compatible fixed cycle, the next fruit is reached within the free-cell count;
repeating the argument gives a full-board completion certificate.

The proposal scorer can be trained from verified completions or candidate
selection examples. On the current standard task, every certified candidate
already wins, so the win objective supplies little ranking signal. With no
move-efficiency objective, a deterministic tie-break is enough: a neural scorer
adds research capacity rather than improving the guaranteed outcome.

If no compatible certificate exists, classify the state as **uncovered by the
certificate**, not necessarily unwinnable. It may go to B or C in an explicitly
uncertified mode. Do not let a network move off the certified set and assume that
switching back to `HamiltonianPolicy` later restores a guarantee. Future shortcut
schemes need their own invariant and progress argument.

**Strengths:** strongest reliability for the supported default task; clear
separation of empirical performance and proof. **Limitations:** does not certify
general non-aligned or odd-by-odd states; may leave little substantive work for
the learned component. If completion alone is the product objective, the existing
fixed cycle is the simplest valid special case of this architecture.

## 9. Comparison and selection

| Architecture | Learned component | Inference work | Main advantage | Main limitation |
| --- | --- | --- | --- | --- |
| A: residual CNN + PPO | Primitive policy and critic | One forward pass | Clear learning baseline | Rare mistakes compound over long games |
| B: network + stochastic MCTS | Priors and leaf values | Adaptive exact rollouts | Best initial match for deliberate planning | Search cost and value errors |
| C: graph hierarchy | Option proposals/ranking and values | Route generation and simulation | Longer planning units | Extra abstraction and training failure modes |
| D: verified cycle proposals | Optional candidate scorer | Certificate checks and successor moves | Conditional completion guarantee | Restricted applicability; learning may be unnecessary |

For the requested **learned agent**, choose **B**, starting with A's encoder and
training infrastructure. Evaluate a relational Transformer only after obtaining
a search/no-search comparison. Keep D as a distinct certified mode/reference;
do not silently mix its wins into a claim about the standalone learned policy.

Large language models, image encoders, and a learned world model are not justified
as initial components: all actionable state is structured and dynamics are already
implemented. Model capacity should grow in response to measured failures, not the
apparent difficulty of long episodes.

## 10. Training data and curriculum

1. **Establish the information contract.** Fix board rules, RNG visibility, action
   encoding, reward, timeout semantics, and allowed inference computation.
2. **Start small.** Use 4×4, 6×6, and 8×8 boards, then 12×12 and 20×20. Gate progression
   on held-out complete episodes, not fruit reward alone. Preserve a mixture of
   previous sizes to detect forgetting.
3. **Expose late-game decisions.** Sample valid states at several occupancy bands
   from recorded trajectories. Aligned cycle states provide demonstrably solvable
   examples; learner states provide relevant difficult examples. Controlled
   perturbations can be useful but must be validated and need not remain solvable.
4. **Keep expert coverage explicit.** An expert label from a heuristic recovery
   state is not equivalent to a certified cycle label. Use exhaustive small-board
   analysis where feasible, otherwise label unknown outcomes as unknown.
5. **Avoid dataset leakage.** Split by source episode/seed and perturbation family
   before taking snapshots. For augmentation, rotate/reflect coordinates,
   directions, and actions together. Reflections swap relative left/right; square
   rotations and rectangular dimension swaps require consistent metadata.
6. **Collect on-policy failures.** Add near-trap states encountered by the current
   agent, not only hand-selected successful traces. Keep late-game replay balanced
   without evaluating solely on these artificial starting states.

Use `SnakeEnv.reset(options={"state": state})` or `SnakeGame.from_state()` for
custom starts. The current `VectorSnakeEnv` does not provide equivalent arbitrary
state injection; use scalar/independent environments for this curriculum until a
separate implementation task adds a tested interface. Standard-start batching
can use the existing vector implementation immediately.

## 11. Evaluation and decision gates

Report the following separately for each architecture, checkpoint, board size,
starting-state distribution, and search budget:

- Full-board wins, collision deaths, and budget-censored/looping episodes.
- Completion fraction with a binomial interval when outcomes are fully observed;
  with censoring, also show resolved fraction and success bounds rather than
  quietly dropping unfinished episodes.
- Terminal occupancy distribution and failure frequency by occupancy band.
- Real actions, simulated transitions, neural evaluations, time, and peak memory.
- Raw policy, policy with immediate mask, policy plus search, and any certified
  controller use. Record intervention counts and certificate coverage.

Use a shared held-out seed set for fair comparisons. Different policies occupy
different cells, so the same seed does not imply identical fruit coordinates.
Budget both total compute and episode counts; a small policy can play many more
games than a large search procedure in the same time.

Separate three evaluation tracks:

| Track | Purpose | Reference |
| --- | --- | --- |
| Standard 20×20 starts | Primary user objective | Existing Hamiltonian completion |
| Valid non-aligned snapshots | Recovery and robustness | Known-solvable subset plus explicitly unknown states |
| Other dimensions, including odd-by-odd | Generalization | Separate results; no assumed cycle guarantee |

Begin development comparisons with 100 held-out standard episodes and several
training seeds. For a milestone claiming high empirical reliability, a proposed
gate is a 95% lower confidence bound above 99% completion, with a predefined
censoring policy and compute budget. Roughly 300 independent episodes with zero
failures yield an approximately 1% one-sided upper failure bound; that is still
not a mathematical guarantee, nor evidence about other state distributions.

Essential ablations are full grid versus 19 features, imitation versus
imitation-plus-RL, search versus no search with the same encoder, CNN versus
relational encoding at matched resources, and fruit-timeout versus no-timeout
as separately defined tasks. Evaluate search values on held-out continuations;
good calibration on easy cycle states alone is not enough.

## 12. Implementation sequence and reproducibility

**First milestone:** build the evaluation/data adapters, implement A, and show
complete small-board games with correct masks and terminal targets. Reuse existing
game interfaces. Any new training dependencies belong in a separate approved
implementation change; the repository currently has no neural training framework.

**Second milestone:** add exact stochastic search around the same network. Check
that its simulator never consumes or reveals live RNG state. Validate chance
sampling, terminal backups, action conversion, and bounded-search behavior on
small tractable cases. Require a measured improvement before increasing scale.

**Third milestone:** scale curriculum and complete standard 20×20 evaluations.
Compare Transformer encoding if capacity/topology failures justify it. Add C only
if primitive horizon remains the bottleneck. Add D if the system needs a clearly
identified certified execution mode; report its contribution separately.

To reproduce the measurements in this report from the repository root:

```bash
.venv/bin/python analysis/architecture_probe.py
```

The probe uses installed project dependencies, writes only its adjacent JSON
result file, and trains no models. It checks the baseline episodes, an explicit
feature-alias example, and the action-mask fallback/timeout behavior. It does not
prove that the proposed neural architectures will reach their design goals.

This analysis changes only files under `analysis/`. The game engine, dependency
declarations, and training implementation remain subjects for a separate coding
task after choosing an architecture.
