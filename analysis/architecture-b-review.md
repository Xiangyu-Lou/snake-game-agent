# Architecture B Review: What Holds, What Breaks, What to Change

## Verdict

**Keep Architecture B** (a policy/value network guiding exact stochastic search)
as the main design. It matches the task: the agent sees the exact state, the
simulator is exact and cheap to copy, and no move has a deadline. The existing
reports are careful. Every number rechecked here is correct, and the committed
probe reproduces exactly (§1).

**Objective, as clarified by the project owner.** "Does not care about speed" in
the project README refers to computation: the agent may think as long as it
likes before each move. The number of moves does matter. The goal is to fill the
board first and, among equally safe ways to do so, to use as few moves as
possible: eat quickly early in the game, and as quickly as safety allows later.
The earlier reports, and the first version of this review, treated move count as
a diagnostic only. This version ranks it second. That changes the reference
strategy (§2.1), how B combines its two goals (§2.2), and the gates (§3).

Measured against that objective, the B plan as written has six gaps. Some would
stall training; others would make a successful result impossible to interpret.
They are listed below in order of impact. Each is backed by a new reproducible
probe, [`architecture_b_probe.py`](architecture_b_probe.py), whose output is in
[`architecture_b_probe_results.json`](architecture_b_probe_results.json).

| # | Plan as written | Problem found | Change |
| --- | --- | --- | --- |
| 1 | Main gate: ≥ 99% completion on standard 20×20 starts, after pretraining on Hamiltonian games; moves only a diagnostic | Imitating the expert means memorizing a 400-entry table keyed by the head cell, and the plain cycle is the slowest way to finish. A certified shortcut cycle, with no learning, finished all 320 test games and used 52% of the plain cycle's moves on 20×20 | Gate **completion first and moves second**, against the shortcut cycle; add **recovery** and **demonstration-free** tracks (§2.1) |
| 2 | No fruit timeout; win-only values with γ = 1; no step penalty | Once deaths are prevented, loops take over: a survival-guarded greedy policy died in 1 of 36 games but looped or stalled in 28. A per-move penalty folded into the win reward is either large enough that crashing beats winning or too small to learn from | **Separate win-probability and moves heads**, combined by the owner's 8:2 game score (§4); **N + L moves without fruit = loss** as a backstop (§2.2) |
| 3 | Traps found by depth-limited search plus a learned value | Greedy games were already lost 6–30 moves before the collision on 10×10, and at least 32 moves before it in five of six 20×20 games. Proving it took up to 545,518 nodes on 10×10, and more than 300,000 in those five 20×20 games | Use an **exact, cheap survival certificate** as search mask, input feature and label (§2.3) |
| 4 | 64–1,024 (up to 4,096) expansions per move; skip forced moves | At 256 expansions, one 20×20 game needs about 10 million network evaluations (16 PFLOP for the proposed Transformer). Only 9% of expert moves are forced | **Gumbel root search with 8–32 simulations**, adaptive per-move budgets, afterstate leaves, batching across games (§2.4) |
| 5 | Win-probability value trained on final 0/1 outcomes | One label per ~40,000 states. Values near 0 or 1 hide the differences between moves | **Bootstrapped and exact labels**; compare actions on a failure-probability scale (§2.5) |
| 6 | Imitation, then PPO with a reward warm-up, then search; coordinates normalized by board size | PPO is a detour for B. Normalized coordinates change what "adjacent" looks like on every board size, which undermines the small-board curriculum | **Self-play with search targets from 4×4 upward**; size-independent geometry (§2.6) |

§3 turns these changes into milestones with gates. §4 records the owner's
decisions and the specifications they imply.

## 1. What holds up

| Claim in the reports | Check | Result |
| --- | --- | --- |
| Baseline episodes, feature alias, mask fallback | Re-ran `architecture_probe.py` functions against the committed JSON | Identical |
| Aligned cycle: at most `F(F+1)/2` = 79,003 moves, expected `F(F+3)/4` = 39,700 | 20 games per size, Hamiltonian policy | 4×4: 52.4 vs 52; 6×6: 301 vs 297; 8×8: 986 vs 976; 10×10: 2,447 vs 2,425; 12×12: 5,088 vs 5,076; 20×20 (6 games): 39,804 vs 39,700 |
| `0.99^39700 ≈ 5.2e-174`, `0.999^39700 ≈ 5.6e-18` | Arithmetic | Correct |
| 21 3×3 convolutions give a 43×43 receptive field | Arithmetic | Correct |
| About 300 failure-free games bound failures near 1% | Exact one-sided 95% bound | 299 games needed |
| Input-specification worked example | Ran the reference `encode_cells` / `encode_global` | All 12 + 10 values match |
| Reward transitions (move, eat, collide, win, strict win) | Re-ran with `SnakeEnv` | 0, 0.2/36, -1, 1.05, 1.0 |
| `clone()` copies the hidden RNG key | Read `SnakeGame.clone` | Copies `_key` and `_fruit_draws`; `reseed()` or `from_state(seed=...)` is required for planning |
| `VectorSnakeEnv` cannot start from arbitrary states | Read `VectorSnakeEnv.reset` | Only seeds and reset masks |

The plan is also right on the points that are easiest to get wrong: one state
per network call with no history; three relative actions; never choosing the
favourable fruit; never planning with the live RNG stream; cache keys that
include the heading; provenance on every label; censoring instead of false
terminal labels; and accounting for simulator steps separately from network
evaluations.

**One minor correction.** The probe's all-fatal 3×3 state has heading `UP`,
although its last move (neck to head) was `LEFT`. `GameState.validate()`
accepts this, but play cannot produce it. With the consistent heading, moving
`DOWN` into the tail is safe. The API point stands, and a reachable example is
easy to find: `RandomPolicy` dies only when no safe move exists, so the state
before any of its deaths is all-fatal. Hand-built curriculum states should
therefore set the heading to the neck-to-head direction. Keep the heading in
cache keys, as the plan says.

**Superseded by the clarified objective.** These statements in the earlier
reports assume that move count does not matter:

- "Move count is a diagnostic, not an optimization target" (architecture report, §1).
- "Avoid an action penalty because speed is not a user objective" (§4).
- "With no move-efficiency objective, a deterministic tie-break is enough" (§8,
  Architecture D).
- The advice against any move-efficiency objective (rewards report, §6).

Fewer moves is now the second goal. Their warnings still apply to how it is
added: a per-move penalty mixed into the win reward does distort the first goal
(§2.2).

## 2. Issues, evidence and changes

### 2.1 Gates must rank completion first and moves second

The architecture report already separates three evaluation tracks: standard
starts, non-aligned snapshots and other board sizes (its §11). Only the first
has a gate, a 95% lower confidence bound above 99% completion, and moves are not
part of it. Both A and B begin with Hamiltonian demonstrations, and the report
does not separate runs with and without them.

**Evidence: the expert is a lookup table, and a slow one.** In six standard
games the Hamiltonian expert made 238,824 decisions from exactly 400 head cells,
with no cell ever labelled with two different actions. Imitating it means
memorizing a 400-entry table keyed by the head cell. That wins every game, but
in about 39,700 moves on 20×20: the slowest way to finish. Such a network could
pass the completion gate with no search, no useful value function, and no ability
to recover from a state the table does not cover.

**A certified, faster reference needs no learning.** Keep the body in cycle
order, and let the head jump to a neighbouring cell further along the cycle,
never past the tail or the fruit. Every cell between the head and the tail along
the cycle is then free, so the next cell on the cycle is always safe. Each move
brings the head at least one step closer to the fruit along the cycle, so every
fruit is eaten within N moves. Jumps pay off only while the snake is short: late
jumps leave holes, and a fruit that appears in a hole costs most of a lap.

| Board | Plain cycle, expected moves | Jumps while under 50% fill, mean moves | Share | Completed |
| --- | ---: | ---: | ---: | ---: |
| 6×6 | 297 | 214 | 72% | 200 of 200 |
| 10×10 | 2,425 | 1,445 | 60% | 100 of 100 |
| 20×20 | 39,700 | 20,737 | 52% | 20 of 20 |

Cutoffs between 40% and 60% fill were all within 5% of the best. Allowing jumps
at any fill saved only 4% on 20×20. The probe checked the cycle-order invariant
at every move of the 6×6 and 10×10 games, about 1.1 million moves in total. This
is the shortcut variant that the architecture report anticipated for
Architecture D, with its invariant and progress argument stated.

Recovery is where the reference policies struggle. The probe took mid-game
10×10 states from held-out guarded-greedy games, kept those that are
certified-survivable (§2.3) and not aligned with any Hamiltonian cycle, and
played each with an independent fruit seed:

| Start occupancy | States | Hamiltonian (heuristic recovery) | Guarded greedy | Greedy |
| --- | ---: | ---: | ---: | ---: |
| 30% | 59 | 45 won (76%) | 11 won (19%) | 0 |
| 50% | 42 | 19 won (45%) | 9 won (21%) | 0 |
| 70% | 25 | 3 won (12%) | 7 won (28%) | 0 |

Hamiltonian failures here are deaths: its fallback is a heuristic, as the
report warns. Guarded-greedy failures are mostly loops. Whether every one of
these states can be won is unknown, so the table gives lower bounds on what is
achievable, and a planning agent has a lot of room to improve on them.

**Change.** Gate B on three tracks, completion first and moves second:

- **T1, standard starts.** The completion gate as planned, then mean moves
  against the shortcut cycle. Report models trained with and without
  demonstrations separately. Imitating the plain cycle passes the completion gate
  and takes about twice the shortcut cycle's moves.
- **T2, recovery**: certified, non-aligned mid-game states at fixed occupancy
  bands, generated from held-out seeds as in the probe. Gate: beat the best
  reference policy in every band, with 95% intervals.
- **T3, no demonstrations**: B trained only from its own search. T1 and T2 are
  measured on the result.

Odd-by-odd boards remain a separate generalization track. Filling the default
board needs no learning, and the shortcut cycle does it in about half the moves.
B is justified if it beats the shortcut cycle on moves at equal completion,
recovers from states the cycle strategies cannot handle, or both.

### 2.2 Speed and loops: keep the two goals separate

The plan disables the fruit timeout for the principal task, treats unfinished
episodes as censored, trains `V_win` only on resolved outcomes, backs it up with
γ = 1, and rejects step penalties because move count was not a goal. Under the
clarified objective the agent needs pressure toward fewer moves, and it needs
protection against loops. Nothing in the plan supplies either: a safe loop and
real progress both keep the snake alive, and a loop is worth whatever the
network predicts for the states it cycles through.

**Evidence: loops.** The probe's guarded greedy policy takes the greedy move
only when its successor carries a survival certificate (§2.3). That nearly
eliminates deaths, and loops take their place:

| Board | Games | Won | Lost | Looped | Stalled | Mean score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10×10 | 30 | 6 | 1 | 23 | 0 | 64.6 of 97 |
| 20×20 | 6 | 1 | 0 | 4 | 1 | 229.2 of 397 |

"Looped" means a deterministic policy revisited an identical state between two
fruits, so it would repeat forever. "Stalled" means more than 20 × N moves
without fruit. Plain greedy loses all six 20×20 games with a mean score of
111.5. The single 20×20 win took 21,547 moves, about half the Hamiltonian
expectation.

Guarded greedy is "eat as fast as possible while staying alive", applied one
fruit at a time. It got stuck in 28 of 36 games: the fastest safe move now can
lead to positions where no safe move makes progress. Speed has to be judged over
the rest of the game, not per fruit.

**Why one combined reward does not work.** A per-move penalty adds speed
pressure and makes loops costly. One that shrinks as the snake grows matches the
intent: hurry early, be careful late. But a 20×20 game takes 20,000–40,000
moves, so a penalty folded into the win reward has no size that works. The
examples below use a penalty that shrinks linearly with the free cells,
`c0 × (1 − L / N)`, and a game played at the plain-cycle pace:

- **Too large:** with a starting penalty above 7.6 × 10⁻⁵ per move, crashing on
  the first move beats a full win, because the remaining penalty exceeds the
  2-point gap between winning (+1) and dying (−1). A constant penalty crosses
  that line at 5.0 × 10⁻⁵.
- **Small enough to be safe:** a policy that maximizes win probability minus
  penalty loses at most the penalty a win-maximizing policy pays. Keeping that
  penalty below 0.01 (one percentage point of win probability) means a starting
  penalty of 3.8 × 10⁻⁷.
- **Too weak at that size:** 400 moves then cost 1.1 × 10⁻⁴ at 25% fill and
  1.5 × 10⁻⁵ at 90% fill, far below the win-probability differences the value
  network must resolve. It learns neither speed nor loop avoidance from them.

**Change.** Learn the two goals separately, and combine them only when choosing
a move:

1. **Two heads.** `V_win`, the probability of filling the board, as planned.
   `M`, the expected moves until the board is full, divided by the plain-cycle
   expectation `F(F + 3) / 4` so its scale stays stable through the game. Train
   `M` on won games. Count deaths and timeouts as the maximum, so that dying
   never looks fast.
2. **Combine them with the owner's game score** (§4, decision 1). A filled board
   scores `0.8 + 0.2 · S`, where `S` is the share of the plain cycle's moves
   saved, and anything else scores 0. The search maximizes the expected score,
   `P_win · (0.8 + 0.2 · S)`, with `S` computed from the moves so far plus `M`.
   Because the speed bonus only counts for a filled board, dying always scores
   below any win, so the crash problem of per-move penalties cannot arise. The
   weights enter only at decision time, so the 0.2 weight does not shrink what
   the moves head learns.
3. **The owner's schedule falls out of this rule.** Early in the game every safe
   move has nearly the same win probability, so the speed term decides. Late in
   the game a risky move loses more through `P_win` than it gains in speed. On
   20×20 one saved move is worth at most 5.6 × 10⁻⁶ of win probability, so no
   single decision trades much safety for speed.
4. **Timeout as a backstop.** In training, a game is lost after N + L
   consecutive moves without fruit, where L is the snake's current length (the
   owner's choice: the allowance grows with the snake, from N to about 2N). It
   never binds for the plain or shortcut cycle, which eat within N moves. Loops
   then end as resolved losses, so neither head has to learn from games that
   never end. The environment's `max_steps_without_fruit` is a constant, so the
   training loop, which steps `SnakeGame` directly, enforces this rule. The
   input specification's remaining-budget features (global features 7–9) carry
   it.
5. **Search and evaluation details.**
   - In search, a node whose board (body, heading, fruit) repeats an ancestor's,
     with no fruit eaten in between, is the same position with less budget.
     Prune it.
   - Evaluate final models twice: with the rule, and with no limit as a check,
     reporting loops as censored.
   - Log how close winning games come to the limit.

### 2.3 Use exact survival analysis instead of rediscovering it

**When is a game actually lost?** For each greedy death the probe searched
backwards, using exact search over all continuations without eating, for the
first state from which every continuation collides. Eating only lengthens the
body, so such a state is lost (unless the last free cells fill first, which is
impossible at these occupancies).

| Board | Deaths | Moves from "lost" to the collision | Nodes needed to prove "lost" |
| --- | ---: | --- | --- |
| 10×10 | 30 | median 20.5 (range 6–30) | median 2,586; max 545,518 |
| 20×20 | 6 | 7 in one game; at least 32–39 in five | 7 in one game; over 300,000 (the budget) in five |

The fatal decision comes tens of moves before the collision. Confirming it by
enumeration took a median of 2,586 nodes on 10×10 and more than 300,000 in five
20×20 games, typically more than a 64–1,024-simulation search explores per move.
In two of the 20×20 games the
snake entered a 35-cell strip, two cells wide, along a wall. It was lost at
least 36 moves before it died.

**A cheap test exists: a survival certificate.** After a candidate move, run one
breadth-first search from the head through free cells. Look for a path that ends
next to a body segment which has already moved away when the head arrives. If
nothing is eaten, segment `k` of a length-`L` snake moves away after `L - k`
moves. If such a path exists, the snake can enter that cell and then follow its
own body around a closed loop forever.

The probe validated it against the engine:

- 8,613 certificates from random and greedy games were replayed: the snake
  entered the body on time and circled the loop twice, with no collision.
- All 49 states classified as lost were confirmed by brute force with the real
  engine (growth and random fruit included, depth 40).
- The probe's `Board.target` agreed with `SnakeGame.is_safe` on all 23,072 checks.
- Guarded greedy died once in 36 games (§2.2).

**Limits.** The certificate is sufficient, not necessary. Among immediately safe
moves that lack one, exact search found 62% (greedy states) and 23%
(guarded-greedy states) still survivable. Survival also does not imply progress
(§2.2).

**Change.** Split the work: exact code decides survival, and learning decides
progress.

- **Search mask.** Classify every immediately safe move as *certified*,
  *proven lost* (by bounded exact search, for example 10⁴ nodes), or *unknown*.
  Prune only proven-lost moves, with value 0. Never prune unknown moves.
- **Inputs.** Per action: certified, proven lost or unknown. Per cell: shortest
  free-path distance from the head, release time of body cells, and slack
  (arrival minus release) on body cells bordering the reachable region. Like the
  coordinate planes, these are exact preprocessing. Ablate them.
- **Labels.** A proven-lost state has value 0, whatever the episode's outcome.
- **Speed.** In pure Python the certificate costs 34–159 µs per call on 20×20,
  and a search node needs up to three. Implement it in compiled or vectorized
  form on `VectorSnakeEnv`'s timestamp representation, which the probe's
  `Board` mirrors.

### 2.4 Search: fewer simulations, used where they matter

**Evidence: cost per game.** The cost model below assumes every move is
searched, the plan's encoders (Transformer: 6 blocks, width 128, 401 tokens;
CNN: 64 channels, 21 convolutions), and, for illustration, a sustained
20 TFLOP/s:

| Board | Moves per game | Simulations per move | Network evaluations | Transformer | CNN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6×6 | 297 | 256 | 76,000 | 7 TFLOP (0.4 s) | 4 TFLOP (0.2 s) |
| 10×10 | 2,425 | 256 | 621,000 | 173 TFLOP (9 s) | 92 TFLOP (5 s) |
| 20×20 | 39,700 | 16 | 635,000 | 1.0 PFLOP (50 s) | 0.4 PFLOP (19 s) |
| 20×20 | 39,700 | 256 | 10.2 M | 16 PFLOP (13 min) | 6 PFLOP (5 min) |
| 20×20 | 39,700 | 1,024 | 40.7 M | 64 PFLOP (53 min) | 24 PFLOP (20 min) |

The CPU side, measured on this 4-CPU container (timings varied by up to about
50% between runs):

- A masked PUCT tree over `SnakeGame` clones, without a network, costs 11–19 µs
  per simulation (`clone` + `reseed` + `step` costs 3–6 µs). At 256 simulations
  that is 2.5–3.5 minutes per 20×20 game, which is affordable.
- Encoding leaves one at a time with the specification's `encode_cells` costs
  54–105 µs each on 20×20, which is 9–18 minutes per game at 256 simulations.
  Leaves must be encoded in batches; the vectorized `encode_grid` already exists.
- On boards up to 10×10, this per-leaf Python work costs more than the network.

So 1,000 self-play games on 20×20 cost about 220 GPU-hours at 256 simulations
and 14 GPU-hours at 16 simulations (Transformer, 20 TFLOP/s). The table assumes
the plain-cycle pace. At the shortcut cycle's pace (about 20,700 moves on
20×20) the 20×20 costs roughly halve, and a faster agent costs less still.

Skipping forced moves saves little on cycle-like play. Only 9.4% of expert
decisions on 20×20 (18% on 10×10) had at most one safe move. On compact
guarded-greedy play the figure was 38%.

**Change.**

1. **Gumbel root search** (Danihelka et al., *Policy improvement by planning
   with Gumbel*, ICLR 2022). It uses sequential halving at the root and
   completed-Q policy targets. With at most three actions, 8–32 simulations give
   a policy-improvement target; train on that instead of raw visit counts.
2. **Adaptive budgets.** Search only when at least two moves are not proven
   lost and, in addition, the prior is uncertain, a safe move lacks a
   certificate, or occupancy is high. Use playout-cap randomization (Wu,
   *Accelerating Self-Play Learning in Go*, 2019): only a fraction of moves get
   full searches, and only those produce policy targets.
3. **Value scale.** Average values in probability space, which keeps backups
   correct. Compare actions on a scale that spreads out values near 1, such as
   `-log(1 - q)`, or with Gumbel's scaled completed-Q. With `c = 1.25`, a prior
   of 1/3 and 256 parent visits, the PUCT exploration term is 0.6 at 10 child
   visits and 0.08 at 85. A 0.01 win-probability difference between two safe
   moves is invisible until well beyond that.
4. **Afterstate leaves.** When a simulated move eats while many cells are
   free, stop and evaluate the afterstate: the board without a fruit, plus a
   "fruit pending" global flag. This is the decomposition used by Stochastic
   MuZero (Antonoglou et al., ICLR 2022). Enumerate fruit outcomes only when
   few free cells remain, for example 16 or fewer. The plan's rules on
   independent sampling still apply.
5. **Batch across games.** Run 64–256 self-play games in parallel, with one
   pending leaf per game per network call. This is simpler than virtual loss
   within one tree, and it fills the accelerator.

### 2.5 Value targets: more signal per game

A final 0/1 outcome labels about 40,000 states per 20×20 game with the same bit,
including early states that had nothing to do with the result. With the
timeout rule there are no censored games, but the signal is still sparse.

**Change.**

- Blend the outcome with a bootstrapped target: the search value n moves later,
  or the root's completed Q. Record which one was used, as the plan requires for
  all labels.
- Use exact labels wherever they are cheap. Proven-lost states are 0. Small
  boards can be solved exactly: 4×4 has 27,960 head-to-tail body paths, about
  1.4 × 10⁵ states with a fruit. 5×5, an odd-by-odd board with no Hamiltonian
  cycle, has about 2.2 × 10⁷, which is feasible in compiled code. These give
  ground-truth values for calibration, policy regret, and a known optimum for
  the odd-board track.
- Train the moves head (§2.2) on every state of every won game: its target, the
  moves that game still took, is observed everywhere, so it is far denser than
  the win label. Bootstrap it the same way as the win value.
- Sample replay so that long games do not dominate, and oversample states shortly
  before failures. The plan already asks for hard near-full states; §2.3 adds
  the earlier, decisive ones.

### 2.6 Encoder and training path

**Training path.** For a B-first project, skip PPO and the reward warm-up:

- **B does not use them.** B trains with supervised losses on its own games: the
  policy head imitates the search's improved move distribution, the value head
  predicts the outcome, and the moves head predicts the moves still needed.
  Advantages, GAE λ, a return critic and a staged
  reward warm-up would be built, tuned and then discarded. The plan already keeps
  warm-up rewards out of B's backups.
- **PPO is weakest where this task is hard.** It assigns credit to individual
  moves through a policy gradient. The plan's λ = 0.99 gives a credit horizon of
  about 100 moves, while the payoff is a win tens of thousands of moves later.
  Search instead plays candidate moves forward in the exact simulator, so its
  targets already carry their consequences. B's value head still learns from
  outcomes; §2.3 and §2.5 give it denser and exact signals.
- **The no-search baseline comes free.** Evaluating B's raw policy head at every
  checkpoint compares the same network with and without search, a cleaner test
  than a separately trained PPO model.
- **No warm start is needed.** On small boards, early search reaches real
  outcomes (below).

The environment already exposes `action_masks()`, so an off-the-shelf maskable
PPO (for example `MaskablePPO` from sb3-contrib) can run on `SnakeEnv` as a cheap
check of the environment and logging, and as a model-free reference point. It
should not be a stage of B or a gate.

Start on 4×4 and 6×6, where games last about 50 and 300 moves and
late-game searches reach the actual win or loss, so no demonstrations are needed
to get started. Then grow through 8×8, 10×10 and 14×14 to 20×20. Keep earlier sizes in the mixture.
Mix standard starts with mid-game states replayed from the agent's own games,
using `SnakeGame.from_state` (as the plan notes, `VectorSnakeEnv` cannot inject
states). Demonstrations become an optional, separately reported ingredient
(T1 with demonstrations).

**Encoder.** Keep the plan's order: start with A's CNN, because at 0.59 GFLOP it
is 2.6 times cheaper than the Transformer on 20×20 and is naturally size-independent.
Compare the Transformer later. Apply these changes to the input specification
either way:

- **Size-independent geometry.** `x / (W - 1)` makes a neighbour 1/3 away on
  4×4 and 1/19 away on 20×20, so local patterns learned on small boards do not
  transfer. Encode wall distances in cells (`x`, `W-1-x`, `y`, `H-1-y`) through
  embeddings or Fourier features. Give attention biases integer offsets
  (`dx`, `dy`, clipped, with a "far" bucket) rather than normalized ones.
- **Relational biases from the start.** When coordinates enter only through a
  linear projection, a first-block attention score is a smooth function of the
  key's position (affine before layer normalization). No head can sharply pick
  out the four neighbours of every cell; later blocks must build such features
  first. Adjacency and body-link biases belong in the baseline, not in a later
  ablation.
- **Release times as integers.** Consecutive segments differ by 1/400 in
  `(L - k) / N` on 20×20. Embed integer release times (bucketed or Fourier)
  instead of feeding a tiny scalar to a linear projection.
- **Exact survival planes** from §2.3.
- **Symmetry.** Use the 8 symmetries of square boards for augmentation (the plan
  covers the action relabelling).
- **Memory.** Dense attention on 401 tokens stores 4.9 MiB per example per layer
  in fp32. For 256 leaves and 6 layers that is 7.4 GiB of attention
  probabilities during training. Plan for bf16 and attention kernels that accept
  an additive bias.

**Stack.** Start with PyTorch for the network and a Python search over
`SnakeGame`, batched across games, with the certificate compiled (for example
with Numba). That is enough up to about 10×10. If profiling at 20×20 shows
per-leaf Python work dominating, port the timestamp engine to JAX and use
DeepMind's `mctx` library, which implements Gumbel and stochastic MuZero search.
Either choice is a new dependency and belongs in a separate implementation change.

## 3. Revised plan for B

| Milestone | Work | Gate to continue |
| --- | --- | --- |
| **M0. Contracts and instruments** (no learning) | Fix the information contract: relative actions, public state only, the 8:2 game score, and the N + L fruit timeout as a training loss. Turn the probe's certificate, lost-proof search, shortcut cycle and recovery-state generator into tested code. Build an evaluation harness for T1–T3 with held-out seeds, outcome classes (won, lost, timed out, censored) and move counts. Solve 4×4 exactly | Tests pass; the harness reproduces this probe's reference numbers |
| **M1. Self-play on 4×4 and 6×6**, no demonstrations | CNN with certificate planes and size-independent geometry; win-probability and moves heads combined by the game score; Gumbel root search, 16 simulations, at non-forced moves; afterstate leaves; blended value targets | 6×6 T1: no failures in 299 held-out games (lower bound above 99%) and mean moves at or below the shortcut cycle's 214; 4×4 values calibrated against the exact solution; raw-policy and search results both reported |
| **M2. 8×8 and 10×10, recovery** | Mixed sizes; replayed mid-game starts; search budget sweep (8, 16, 32, 64) at equal compute; measured games-to-plateau and throughput, which size the 20×20 rental | 10×10 T1 ≥ 95% in development with mean moves at or below 1,445; T2 beats the best reference in every band (currently 76%, 45%, 28%) with 95% intervals |
| **M3. 20×20** | Fine-tune from M2; adaptive budgets; batched leaves; Transformer only if CNN errors point to capacity or long-range limits | 100-game development run, then the plan's 299-game reliability gate on T1 with mean moves below 20,737, plus T2 at 20×20 against the reference policies |

A demonstration-assisted run can be added at any stage, reported separately.
Architecture A's baseline is B's raw policy head. C and D keep the triggers the
architecture report gives them.

## 4. Decisions

The project owner answered the open questions on 2026-10-26. Each answer and the
specification it implies:

1. **Weights: 8/10 on winning, 2/10 on speed.** A game that fills the board
   scores `0.8 + 0.2 · S`, and any other game scores 0.
   `S = clip(1 − T / T_cycle, 0, 1)` is the share of moves saved relative to the
   plain cycle. `T` is the number of moves taken, and `T_cycle = F(F + 3) / 4` is
   the plain cycle's expectation from the game's start state.
   - **Reference scores:** the plain cycle scores 0.80; the shortcut cycle scores
     0.90 on 20×20 and 0.88 on 10×10.
   - **Exchange rate:** the agent gives up one percentage point of completion
     only to save about 4.5% of `T_cycle`. That is 1,805 moves on 20×20, 110 on
     10×10 and 14 on 6×6.
   - **Gate conflict:** if training settles below the 99% completion gate, the
     gate wins and the win weight goes up.
2. **Safety envelope: evaluate both ways.** On T1 the same network is evaluated
   with and without the cycle-order move mask (§2.1). Recovery (T2) uses free
   choice.
3. **Timeout: yes, growing with the snake.** A game is lost after N + L
   consecutive moves without fruit (§2.2). The remaining-budget global features
   become: enabled = 1, configured = (N + L) / N, and remaining =
   (N + L − moves since fruit) / N.
4. **Stack and hardware: PyTorch, on a rented GPU.**
   - **M0** needs no GPU.
   - **M1** (4×4, 6×6) runs on CPU; a GPU only shortens it.
   - **From M2:** one machine with a single 24 GB GPU (RTX 4090, L4 or A10G
     class), 16–32 CPU cores and 64 GB RAM. The network is small (0.15–0.6
     GFLOP per evaluation) and the search runs on the CPU, so cores matter as
     much as the GPU. A larger GPU such as an A100 or H100 would sit idle.
   - **Budget:** a 20×20 training run of about 3,000 games at about 10 searched
     simulations per move is about 600 M evaluations: 5 GPU-hours and 17 CPU
     core-hours. With 10×10 runs, sweeps and reruns, plan for about 100
     GPU-hours in total.
   - **Largest unknown:** how many games each stage needs. M2 measures it before
     the larger rentals.

## 5. Reproducing the measurements

From the repository root, with development dependencies installed:

```bash
.venv/bin/python analysis/architecture_b_probe.py
```

The script rewrites [`architecture_b_probe_results.json`](architecture_b_probe_results.json)
in about four minutes on this container. Game outcomes, counts and proofs are
deterministic for the fixed seeds; timings vary between runs and machines and
are recorded with the machine description. The survival certificate, the exact
lost-proof search, the guarded greedy policy and the shortcut cycle are analysis
instruments, not application code and not a proposed agent. No model was
trained.
