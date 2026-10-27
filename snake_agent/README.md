# Snake agent

The agent's side of the project, built milestone by milestone after the
[Architecture B plan](../analysis/architecture-b-review.md#3-revised-plan-for-b).
Milestone **M0, contracts and instruments**, holds the rules every agent plays
by, the exact tools that training and search use, the evaluation harness, and
an exact solution of 4×4. Milestone **M1, self-play on 4×4 and 6×6**, adds the
network inputs, the search, the network, self-play and training
([below](#m1-self-play-on-44-and-66)).

M0's gate is met: the tests pass, and the harness reproduces the reference
numbers of the [Architecture B probe](../analysis/architecture_b_probe.py):
every section of its results except the machine-dependent timings and the
analytic network cost model ([below](#reproducing-the-probe)).

- [Quick start](#quick-start)
- [The information contract](#the-information-contract)
- [Exact instruments](#exact-instruments)
- [Evaluation harness](#evaluation-harness)
- [4×4 solved exactly](#44-solved-exactly)
- [M1: self-play on 4×4 and 6×6](#m1-self-play-on-44-and-66)
- [Reproducing the probe](#reproducing-the-probe)
- [Package layout](#package-layout)

## Quick start

```bash
pip install -e ".[dev]"
python -m snake_agent evaluate t1 --agent shortcut --board 10x10 --games 100   # standard starts
python -m snake_agent evaluate t2 --agent guarded-greedy --board 10x10 --compare # recovery starts
python -m snake_agent solve --board 4x4                                        # exact 4x4 numbers
python -m snake_agent reproduce [--full]                                       # the probe's numbers
pytest [--slow]                                                                # --slow: full reproduction

pip install -e ".[train]"                                                      # PyTorch, for M1
python -m snake_agent train --board 6x6 --out runs/m1-6x6 --device cuda        # self-play and training
python -m snake_agent assess runs/m1-6x6/model.pt --games 300                  # the M1 gates
```

```python
from snake_agent import REFERENCES, TRAINING_RULES, evaluate_standard, standard_gate
from snake_agent.seeds import VALIDATION, standard_seeds

evaluation = evaluate_standard(REFERENCES["shortcut"], 10, 10, standard_seeds(VALIDATION, 100), TRAINING_RULES)
summary = evaluation.summary()
print(summary.won, summary.mean_moves)                # 100 wins in about 1,450 moves
print(standard_gate(evaluation, max_mean_moves=2425))  # passes: faster than the plain cycle
```

`--agent` takes a reference name (`random`, `greedy`, `guarded-greedy`,
`hamiltonian`, `shortcut`), `exact` or `exact-envelope` (boards up to 16
cells), or `module:attribute` naming an [`AgentSpec`](evaluation.py), a class
or a callable.

## The information contract

[`contract.py`](contract.py) fixes what an agent sees, what it does and what
counts as success. Training, search and evaluation all follow it.

1. **Public state only.** The agent receives a `GameState`: board size, ordered
   body, heading, visible fruit and move counters, plus the `Rules` in force.
   It never gets the live `SnakeGame`, whose clones replay the hidden fruit
   stream, nor seeds or future fruits. Planning starts from
   `planning_game(state, seed)`, which keeps the visible fruit and draws later
   ones from an independent seed. The harness hands agents `GameState`
   objects only, and their seeds (`AgentSpec.factory(seed)`) are unrelated to
   the fruit seeds.
2. **Relative actions.** One `RelativeAction` per move: left, straight or
   right. The harness also accepts the absolute `Direction` that the game's
   reference policies return; a bare `int` is rejected because it would be
   ambiguous. `as_policy(agent)` adapts an agent for the game window.
3. **Safety strictly first.** The agent maximizes the probability of filling
   the board. Fewer moves only decide between equally safe moves:
   `choose_safely(win_values, moves, allowed)` breaks only exact ties. On
   standard starts every move stays inside the cycle-order envelope, where
   every move keeps the win certain (below), so there speed decides.
4. **Fruit timeout as a loss.** Under `TRAINING_RULES` a game is lost after
   `N + L` consecutive moves without fruit (`N` cells, `L` the current length):
   `Rules.timeout`, `Rules.timed_out`. The budget features of decision 3 are
   `Rules.budget_features(state)`: enabled 1, configured `(N + L) / N`,
   remaining `(N + L - moves since fruit) / N`. `NO_LIMIT` is the engine's own
   rules, for the evaluation check without a limit.
5. **Outcomes.** `Outcome.WON`, `LOST` (collision), `TIMED_OUT` (the timeout,
   also a loss) and `CENSORED` (stopped unresolved by the evaluation: a loop,
   a stall or a move cap). Censored games are never counted as wins or losses.

## Exact instruments

**Survival analysis** ([`survival.py`](survival.py)) works on `Board`, the
timestamped body of `VectorSnakeEnv`, and reasons about the snake moving
without eating.

- `Board.has_survival_certificate()`: a shortest path through free cells
  reaches a body cell that has moved away by the time the head arrives, so the
  snake can circle its own body forever (`Board.certificate_loop()` returns the
  loop). It treats the fruit as a free cell, so it holds for the snake that
  does not eat: a search hint and an input feature, never a reason to prune.
- `prove_lost(board, budget)`: exact search over the continuations without
  eating. Eating only lengthens the body, so if every one of them collides the
  real game is lost, unless the snake fills the board first. The search
  therefore refuses proofs that need as many moves as there are free cells; the
  probe's version did not. Without that check, the 3×3 snake
  `((0,0), (1,0), (1,1), (1,2), (2,2), (2,1), (2,0))` heading left with the
  fruit at (0,1) is "proven lost", yet it wins by eating twice. With it, every
  "lost" verdict on the 159,776 states of 3×3, 3×4 and 4×4 has exact value 0.
- `classify_moves(state, budget)`: each relative action is `FATAL`, `WON`,
  `CERTIFIED`, `LOST` (proven) or `UNKNOWN`. Search prunes only `FATAL` and
  `LOST`.

**Cycles** ([`cycles.py`](cycles.py)). A body is in cycle order when, read from
tail to head, it advances less than one lap around a Hamiltonian cycle.
`CycleEnvelope` holds the moves that enter a cell further along the cycle
without passing the tail or the fruit. They are safe, keep cycle order and
reach every fruit within `N` moves, so a game played inside the envelope is
certainly won, and the next cell of the cycle is always available. The
envelope of a standard start (`CycleEnvelope.for_state`) uses the first cycle of
`snake_game.policies.hamiltonian_cycles` that the start is aligned with.
`ShortcutCyclePolicy` takes the envelope's longest jump while the snake fills
less than half the board, then follows the cycle: the T1 speed reference.

**References** ([`references.py`](references.py)). `REFERENCES` describes the
reference policies for the harness: the game's `random`, `greedy` and
`hamiltonian`, the `shortcut` cycle and `GuardedGreedyPolicy` (greedy among the
moves that keep a certificate). They make the probe's moves: the games
[reproduced below](#reproducing-the-probe) match move for move.

**Recovery starts** ([`recovery.py`](recovery.py)). `recovery_starts` takes,
from each guarded-greedy game, the first certified state at 30%, 50% and 70%
occupancy that no Hamiltonian cycle is aligned with. These are T2's starts.

## Evaluation harness

[`evaluation.py`](evaluation.py) plays games on `SnakeGame` directly, records
an outcome class, move and fruit counts, the longest stretch without fruit and
the closest approach to the timeout (`timeout_margin`), and summarizes them
with exact Clopper-Pearson intervals ([`stats.py`](stats.py)).

| Track | Starts | Rule | Gate |
| --- | --- | --- | --- |
| **T1** `evaluate_standard` | standard start, one game per fruit seed | the agent must stay inside the envelope; leaving it is recorded (`envelope_exit`) | `standard_gate`: every game won, no envelope exit, mean moves below the reference (shortcut cycle: 214 on 6×6, 1,445 on 10×10, 20,737 on 20×20) |
| **T2** `evaluate_recovery` | recovery starts, one fruit seed each | free choice | `recovery_gate`: in every band, the 95% lower bound of the win rate above the best reference's rate (`best_rates`) |
| **T3** `evaluate_t3` | T1 and T2 | only for `Provenance.SELF_PLAY` agents | as T1 and T2 |

- **Rules.** `TRAINING_RULES` (default) ends loops as timeouts. With `NO_LIMIT`
  a game is censored when a deterministic agent repeats a state since the last
  fruit (`"loop"`; `AgentSpec.memory` adds the agent's internal state to the
  comparison) or after more than `20 N` moves without fruit (`"stall"`).
  Evaluate final models both ways.
- **Provenance.** Each result carries the agent's `Provenance`:
  `REFERENCE`, `DEMONSTRATIONS` (reported as "T1 (with demonstrations)") or
  `SELF_PLAY` (reported as T3).
- **Win intervals with censoring.** `Summary.win_interval` counts censored games
  as lost for the lower bound and as won for the upper bound.
  `Summary.failure_bound` is the one-sided 95% upper bound on games not won:
  299 failure-free games bound it below 1%.
- **Held-out seeds** ([`seeds.py`](seeds.py)). `TRAIN` (0 to 10⁹) is for
  training and experiments, `VALIDATION` for development runs, `TEST` for the
  final gates; `check_training_seed` rejects held-out seeds. Inside a held-out
  range, standard games, recovery source games and recovery fruit streams use
  separate blocks (`standard_seeds`, `recovery_source_seeds`,
  `recovery_fruit_seeds`).

## 4×4 solved exactly

[`exact.py`](exact.py) enumerates all 144,088 running states of 4×4 (27,960
bodies with each fruit) and, under the training rules, every number of moves
left before the timeout: 3.7 million states, solved in about a second. It
gives each state's best win probability and the value of each action
(`value`, `action_values`, `regret`), the fewest expected moves among certain
wins (`moves`), and the same restricted to the envelope (`solve(...,
envelope=...)`). `evaluate_policy` computes a policy's exact results. A brute
force through the real engine agrees on every state of 2×3, 3×2, 2×4 and 3×3.

From the standard 4×4 start (expected moves, averaged over the first fruit):

| Strategy | Win probability | Expected moves |
| --- | ---: | ---: |
| Plain cycle | 1 | 52.00 (exactly `F(F+3)/4`) |
| Shortcut cycle (jumps below 50% fill) | 1 | 43.89 |
| Best possible inside the envelope | 1 | 43.08 |
| Best possible with any certain win | 1 | 35.47 |

Two measurements bear on the owner's decisions:

- **The envelope costs moves.** On 4×4 the best certain win needs 18% fewer
  moves than the best one inside the envelope. Decision 2 keeps T1 inside the
  envelope, so 43.08 is the T1 optimum there; the larger certified sets of the
  review, such as switching cycles after a fruit, are the way to recover part
  of the difference.
- **The timeout costs a few states.** Under the `N + L` rule, 184 of the
  144,088 states lose win probability (lengths 11 to 14; 72 certain wins at
  length 14 become losses because every safe route needs more than `N + L`
  moves without fruit). The standard start stays a certain win, and the best
  expected moves change by 3 × 10⁻⁶.

## M1: self-play on 4×4 and 6×6

M1 trains the plan's network from scratch by self-play with search, on 4×4 and
6×6, without demonstrations (track T3). Its gate: on 6×6 T1, no failures and
mean moves below the shortcut cycle's 214; on 4×4, values calibrated against
the exact solution; raw-policy and search results both reported. PyTorch is an
optional dependency: `pip install -e ".[train]"`.

### Network inputs

[`encoding.py`](encoding.py) encodes one state per evaluation, turned so that
the snake heads up: the three relative actions then always enter the cells left
of, above and right of the head, and only the left-right mirror remains as a
symmetry (`mirror`, for augmentation). Everything is exact preprocessing of the
public state, measured in cells where that matters for transfer across sizes:

- 25 cell planes: the body and its release times, the fruit, the free cells the
  head can reach and how far, the slack of the body cells at the edge of that
  region (the survival certificate's ingredients), segment directions, the
  canonical cycle (distance ahead, next cell, the envelope's free arc, the holes
  behind the tail) and the distances to the walls;
- 8 features per action: its move class (`classify_moves`), whether it eats,
  and whether it is inside the envelope and how far it jumps;
- 11 global features: occupancy, the timeout budget of decision 3, the
  afterstate flag, cycle order and a few distances.

No plane holds coordinates, and boards sit in the corner of a larger canvas
when sizes are mixed. The moves head works in units of `moves_scale(F) =
F(F + 3) / 4`, the plain cycle's expected moves to fill `F` free cells.

### Search

[`search.py`](search.py) is Gumbel root search (Danihelka et al., *Policy
improvement by planning with Gumbel*, 2022): sequential halving over the root's
moves, completed Q-values, and the improved policy `softmax(logits + σ(q))` as
the policy target, with 16 simulations at every move that has a choice. The
contract picks one of two modes at the root:

- **Envelope mode**, on every standard start: only envelope moves are searched.
  Each keeps the win certain, so the search compares expected moves, in steps
  of one move (`moves_unit`) rather than rescaled to [0, 1] at each node as the
  paper does with values: rescaling turns differences of a fraction of a move
  into full-size ones, so the search follows the moves head's errors.
- **Free mode** elsewhere: the win probability decides, and expected moves only
  order moves whose win values tie exactly. A learned win value stays below 1
  (`CERTAIN`), so only exact wins tie: the envelope and a filled board. A state
  in cycle order is such an exact win, a leaf with win value 1.

A move that eats ends the simulation at its *afterstate*, the board before the
next fruit appears; later visits draw the fruit from the search's own
generator. Fatal moves and moves that time out are pruned, and in free mode
moves back to a position of the current fruit's epoch, and at the root moves
that `prove_lost` proves lost. The search is a generator, so that a driver can
batch the leaves of many games into one network call. `SearchAgent` plays its
moves; `PolicyAgent` plays the network's most likely move among the same
moves, without search.

**Moves are counted only in won games.** The second value of a node is the
expected moves until the board is full *if it is filled*: backups weight moves
by the win value, and the moves head trains on won games only. The review
counted a loss as a maximum instead, and the first draft of the search clipped
moves at twice the plain cycle's pace. Both distort slow wins: late in a game a
fruit that appears in a hole behind the tail costs most of a lap. On 4×4 with
two cells free, the best certain win takes 3.3 times the plain cycle's pace on
average and up to 12.4 times, so the clip made late moves look alike. Weighting
by the win value keeps dying from ever looking fast, since a loss contributes
no moves, and needs no ceiling (`moves_ceiling` only bounds the network's
output).

The tests check the search against the exact 4×4 solution. With exact leaf
values, a search that visits each root move once sees every move's exact win
probability and, among certain wins, its exact expected moves, and picks the
contract's best move. With an ideal network (exact values, a prior on the best
move), 16 simulations play every sampled state optimally, and the search agent
plays T1 at the envelope's optimum of 43.08 moves.

### Network

[`network.py`](network.py) is a residual CNN (64 channels, 6 blocks of two 3×3
convolutions with a global-pooling bias, 0.6 M parameters by default), masked
to the board's cells: a 4×4 board gives the same outputs on its own canvas and
padded into a 6×6 one. The policy head scores each action from the features of
the cell it enters, its per-action features and a summary of the board; the
win head gives a logit and the moves head a non-negative number of
`moves_scale` units. `NetworkEvaluator` answers the search's queries.

### Self-play and training

[`selfplay.py`](selfplay.py) plays many games in one process, all sharing each
network call. A game starts from the standard start or from a *free start*: a
running state of a reference policy's game that is not a certain win inside
the envelope, so that the network also learns win probabilities. Every move
gives an example of the state before it, and every eating move one of its
afterstate. The targets, each win target with its provenance (`Label`):

| Target | Value |
| --- | --- |
| Policy | the search's improved policy, at moves searched and not lost |
| Win | exact 1 inside the envelope, exact 0 when every move is fatal, proven lost or times out; otherwise the outcome blended with the search's value, `0.5 z + 0.5 v` |
| Moves (won games) | a TD(λ) return over the game's fruits, in `moves_scale` units: the moves until the visible fruit is eaten, then `(1 - λ) V + λ G` at the afterstate, with `V` the network's value and `G` the return from there (`moves_returns`; λ = 1 is the moves the game still took) |

[`train.py`](train.py) runs generations: self-play with the current network
across worker processes (on a GPU, each worker evaluates its games' leaves
there in batches), a replay buffer of the most recent examples, optimizer steps
on minibatches with random left-right mirroring (cross-entropy for the policy,
binary cross-entropy for the win value, a Huber loss for the moves), and an
assessment every few generations. Seeds are training seeds, a block per
generation.

[`assess.py`](assess.py) measures a network against the M1 gates on held-out
seeds: T1 and T2 for the search and the raw policy, with the exact solutions'
view where they exist. On boards up to 6×6, `solve_envelope` scores every T1
decision by the expected moves it adds to the envelope's optimum. On 4×4, it
also measures the calibration of the win head on the states the agents visit
and on states drawn uniformly outside the envelope, and the win probability
each free decision gives up.

### Results

The runs used a rented RTX 4090 server whose container is capped at about six
CPUs, which bound self-play: workers ran the network on the GPU in batches, and
a 6×6 generation took about a minute (45 s of self-play, 15 s of training).
Default network, 480 self-play games per generation, 16 simulations, and 300
(4×4) or 1,000 (6×6) optimizer steps of 512 examples per generation. Numbers
are on validation seeds unless marked *test*. "Extra moves" are exact: the
expected moves a decision adds to the envelope's optimum, averaged over T1
decisions with a choice.

**4×4** (30 generations, half the games from free starts; 1,000 T1 games, 245
T2 starts):

| Agent | T1 mean moves | Extra moves per decision | T2 won | Free decisions optimal |
| --- | ---: | ---: | ---: | ---: |
| Raw policy | 43.52 (exactly 43.63) | 0.098 | 245 of 245 | 99.94% |
| Search, 16 simulations | 43.51 | 0.096 | 245 of 245 | 99.94% |
| Search, 64 simulations | 43.10 | 0.049 | 245 of 245 | 100% |
| Optimum inside the envelope | 43.08 | 0 | 245 of 245 | |
| Shortcut cycle | 43.89 | | | |

Every T2 start is a certain win; the best reference wins 98%, 92% and 64% of
the bands. The win head is calibrated: against the exact values, its expected
calibration error is 0.0017 (mean absolute error 0.0017) on the 1,714 states
outside the envelope that the search visits in T2 games, and 0.018 (mean
absolute error 0.037) on 5,000 states drawn uniformly outside the envelope,
most of which play never reaches.

**6×6** (a quarter of the games from free starts). The first run trained the
moves head on the moves each game still took (λ = 1) and searched with Q
rescaled to [0, 1]; the second used TD returns over fruits (λ = 0.5) and the
one-move unit. The raw policy on T1 (mean moves over 1,000 games, 200 or 300
at generation 10; extra moves over 200 to 300 games):

| Generation | Mean moves, λ = 1 | Extra moves, λ = 1 | Mean moves, λ = 0.5 | Extra moves, λ = 0.5 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 225.9 | 0.38 | 213.3 | 0.26 |
| 20 | 212.5 | 0.24 | **210.5** | **0.18** |
| 30 | 212.9 | 0.22 | 212.6 | 0.20 |
| 40 | (stopped) | | 214.9 | 0.26 |

Every T1 game was won inside the envelope. TD returns cut the moves head's loss
four- to fivefold and reached in 10 generations what the outcomes took 20 for.
Both runs peaked by generation 20; the search is why (below).

**The M1 gate, test seeds** (generation 20 of the λ = 0.5 run and the search
settings chosen on validation; 1,000 T1 games each):

| Agent | Won | T1 mean moves (95% interval) | Extra moves per decision | Gate (below 214) |
| --- | ---: | ---: | ---: | --- |
| Raw policy | 1,000 | 210.37 (209.07–211.67) | 0.166 | passed |
| Search, 16 simulations, 16-move unit | 1,000 | 209.43 (208.11–210.76) | 0.160 | passed |
| Search, 64 simulations, 4-move unit | 1,000 | 208.84 (207.51–210.17) | 0.156 | passed |
| Shortcut cycle (exact) | certain | 212.78 | | |
| Optimum inside the envelope | certain | 201.91 | 0 | |

No game left the envelope. On 6×6 T2 (test seeds, 158 recovery starts), the
search wins 155 (59 of 60, 56 of 57 and 40 of 41 by band) and the raw policy
151, against 98%, 61% and 41% for the best reference (T2 is M2's gate, on
10×10).

**What limits the search.** On 6×6 the search adds little to the raw policy:
with 16 simulations and the one-move unit it plays worse (0.29 extra moves per
decision against 0.18 at generation 20). Against `solve_envelope`, the moves
head misjudges the difference between two sibling moves by 3 to 5 moves early
in the game (17 or more cells free), where the best and second-best moves
differ by 0.6 to 1.5; its one-step ranking agrees with the exact one about 60%
of the time. The search therefore follows the moves head's noise, and the
policy, which learns from the search, stops improving and then gets worse
(generations 30 and 40). Trusting values less helps: with a 16-move unit at 16
simulations, or 64 simulations and a 4-move unit (`assess --moves-unit`), the
search beats the raw policy slightly on the test seeds. On 4×4, where the moves
head is accurate enough, the one-move unit is best and more simulations reach
the optimum. Making the moves head rank close moves is the open problem for M2:
the policy narrows the states it visits, so siblings it does not play are
rarely trained on (the error grows from 3.0 moves at generation 10 to 4.8 at
30).

## Reproducing the probe

`python -m snake_agent reproduce` (about 20 seconds) replays the probe's
reference games through this package and compares each number with
[`architecture_b_probe_results.json`](../analysis/architecture_b_probe_results.json);
`--full` (about 3 minutes) adds every shortcut cutoff, the 20×20 games, 5×5,
the death horizons, the uncertified moves and the move profiles. The tests run
the quick sections; `pytest --slow` runs all of them.

| Section | Reproduced |
| --- | --- |
| Shortcut cycle | games, wins, mean, median and maximum moves at every cutoff on 6×6, 10×10 and 20×20 (214.1, 1,445.3, 20,737.5 at 50%) |
| Plain cycle | mean moves on 4×4 to 12×12; the six 20×20 games: moves, and the expert as a 400-entry table keyed by the head cell |
| Guarded greedy | every 10×10 and 20×20 game: outcome, score, moves |
| Recovery | 59, 42 and 25 starts; the results of the Hamiltonian, guarded-greedy and greedy policies in each band |
| Certificate validation | 23,072 safety checks, 8,613 certificate loops replayed, 49 of 49 lost states dying in the engine |
| State space | 27,960 bodies, 143,368 states and 552 board-filling bodies on 4×4; 3,051,769 bodies and 22,064,488 states on 5×5 |
| Death horizons | horizon, proof size and occupancy of every greedy death on 10×10 and 20×20 |
| Uncertified moves, branching | the move classes and the forced-move shares by occupancy |

The one deliberate difference is the board-fill check of `prove_lost`: 3 of
the probe's 565 "doomed" guarded-greedy moves rest on proofs it refuses. The
review's conclusion is unchanged: 23% of the uncertified moves there are
survivable.

## Package layout

| Module | Contents |
| --- | --- |
| [`contract.py`](contract.py) | `Rules`, `TRAINING_RULES`, `NO_LIMIT`, `Outcome`, action conversion, `planning_game`, `choose_safely` |
| [`survival.py`](survival.py) | `Board`, the survival certificate, `prove_lost`, `classify_moves` |
| [`cycles.py`](cycles.py) | cycle order, `CycleEnvelope`, `ShortcutCyclePolicy` |
| [`references.py`](references.py) | `GuardedGreedyPolicy`, the `REFERENCES` registry |
| [`recovery.py`](recovery.py) | recovery starts for T2 |
| [`evaluation.py`](evaluation.py) | `run_game`, `AgentSpec`, the tracks, summaries and gates |
| [`exact.py`](exact.py) | `solve`, `solve_envelope`, `ExactSolution`, `evaluate_policy` |
| [`seeds.py`](seeds.py), [`stats.py`](stats.py) | held-out seed ranges; binomial and mean intervals |
| [`reproduce.py`](reproduce.py) | the probe's reference numbers |
| [`encoding.py`](encoding.py) | M1: the network inputs, `mirror`, `moves_scale`, `canonical_envelope` |
| [`search.py`](search.py) | M1: Gumbel root search, `SearchAgent`, `PolicyAgent` |
| [`network.py`](network.py) | M1: `SnakeNet`, `NetworkEvaluator`, checkpoints (PyTorch) |
| [`selfplay.py`](selfplay.py) | M1: batched self-play, starts and training examples |
| [`train.py`](train.py), [`assess.py`](assess.py) | M1: the training loop; assessment against the gates (PyTorch) |
| [`__main__.py`](__main__.py) | `python -m snake_agent evaluate | solve | reproduce | train | assess` |
