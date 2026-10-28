# Snake agent

The agent's side of the project, built milestone by milestone after the
[Architecture B plan](../analysis/architecture-b-review.md#3-revised-plan-for-b).
Milestone **M0, contracts and instruments**, holds the rules every agent plays
by, the exact tools that training and search use, the evaluation harness, and
an exact solution of 4×4. Milestone **M1, self-play on 4×4 and 6×6**, adds the
network inputs, the search, the network, self-play and training
([below](#m1-self-play-on-44-and-66)). Milestone **M2, 8×8 and 10×10 with
recovery**, adds board mixtures, recovery and mid-game starts, batched
assessment and a moves unit that scales with the board
([below](#m2-88-and-1010-recovery)).

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
- [M2: 8×8 and 10×10, recovery](#m2-88-and-1010-recovery)
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

pip install -e ".[train]"                                                      # PyTorch, for M1 and M2
python -m snake_agent train --board 6x6 --out runs/m1-6x6 --device cuda        # self-play and training
python -m snake_agent assess runs/m1-6x6/model.pt --games 300                  # the M1 gates
python -m snake_agent train --board 10x10:160 --board 8x8:48 --board 6x6:48 \
    --relative-unit 0.3 --eval-relative-unit 0.3 --out runs/m2 --device cuda   # a board mixture
python -m snake_agent assess runs/m2/model.pt --relative-unit 0.3 --sources 100  # the M2 gates (10x10)
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

### Training on the moves the policy does not play

Two options give the moves head positions that the policy's own games rarely
show, which the search must still judge:

- **Random prefixes** (`--prefix-share`, `prefix_starts`): a game starts from the
  standard start, plays uniformly random envelope moves until the snake reaches
  a length drawn up to `--prefix-fill` of the board, then hands over to the
  agent; only the agent's moves give examples. Random jumps leave holes in
  places the policy would not.
- **Sibling examples** (`--sibling-visits`): every envelope move the search
  visited at least that often but did not play gives an example of the position
  it leads to, with the search's value of the move as its moves target
  (`Label.SIBLING`).

`assess` measures the problem directly (`sibling_ranking`): at each T1 decision
with a choice, the error of each move's one-step value relative to the played
move's, against `solve_envelope`, and the share of pairs of moves ordered as
the exact solution orders them, for each half of the game (the first: at least
half the board free, where moves differ by little). Runs as in M1 (λ = 0.5);
the prefix runs start 37.5% of the games after a prefix (fill up to 75%) and
25% from free starts:

| Run | Generation | Raw policy: T1 moves | Raw policy: extra moves | Search: extra moves | First half: error | First half: agreement | Second half: agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 10 | 213.3 | 0.26 | 0.34 | 3.0 | 0.61 | 0.73 |
| M1 | 20 | 210.3 | 0.17 | 0.29 | 4.7 | 0.58 | 0.80 |
| M1 | 30 | 211.5 | 0.20 | 0.45 | 4.9 | 0.56 | 0.83 |
| M1 | 40 | 213.7 | 0.26 | 0.39 | 4.5 | 0.54 | 0.83 |
| Prefixes | 10 | 211.1 | 0.19 | 0.22 | 3.2 | 0.65 | 0.74 |
| Prefixes | 20 | **208.3** | **0.11** | 0.29 | 3.8 | 0.55 | 0.81 |
| Prefixes | 30 | 210.5 | 0.17 | 0.34 | 4.6 | 0.57 | 0.85 |
| Prefixes and siblings, first run | 10 | 213.6 | 0.29 | 0.26 | 3.6 | 0.56 | 0.78 |
| Prefixes and siblings, first run | 20 | 213.2 | 0.22 | 0.27 | 4.1 | 0.63 | 0.93 |
| Prefixes and siblings | 10 | 213.7 | 0.25 | 0.23 | 3.0 | 0.61 | 0.75 |
| Prefixes and siblings | 20 | 213.7 | 0.21 | 0.29 | 3.5 | 0.53 | 0.88 |
| Prefixes and siblings | 30 | 218.6 | 0.28 | 0.32 | 4.0 | 0.51 | **0.94** |
| Prefixes and siblings | 40 | 217.7 | 0.29 | 0.37 | 3.7 | 0.52 | **0.94** |

The raw policy's numbers are over 300 validation games, the search's (16
simulations, one-move unit) over 200; "error" is in moves, "agreement" the
share of pairs ordered as the exact solution orders them.

- **Random prefixes make a better policy.** Its best checkpoint (generation 20)
  averages 209.0 moves on 1,000 validation games against 210.5 for M1's, and
  the 16-simulation search with a 16-move unit costs 0.13 extra moves per
  decision against 0.17 (300 games each). They do not stop the ranking in the
  first half from degrading, nor the decline after generation 20.
- **Sibling examples make a better moves head late in the game, not a better
  agent.** Two runs (the first stopped at generation 20, the second run to 40
  with other seeds) order 88% to 94% of second-half pairs as the exact solution
  does, against 80% to 85% without them. The first half does not improve: 63%
  and 53% at generation 20 in the two runs, which also shows how much this
  measure varies between runs. The policy is worse throughout (the runs' best
  checkpoints average 213.2 and 213.7 moves, against 208.3 with prefixes
  alone) and declines after generation 20 as well (217.7 at generation 40).
  Continuing the first run from generation 20 with `--init`, so with an empty
  replay buffer and a fresh optimizer, collapsed within ten generations (220.9
  moves): a restart needs the replay buffer too.
- **Open.** In every run the policy peaks around generation 20 and then
  declines while the losses keep falling, whatever the values late in the game.
  Early in the game no variant orders close moves much better than chance:
  they differ by 0.6 to 1.5 moves out of about 170 still to come. Next steps: a
  decaying learning rate or fewer updates per generation against the late
  decline, and value targets precise enough for the first half. (M2 traced the
  decline: self-play searched with the one-move unit, whose improved policy, the
  policy target, is worse than the raw policy from generation 10 on; see
  [the moves unit](#the-moves-unit-a-share-of-the-scale).)

## M2: 8×8 and 10×10, recovery

M2 grows the network to 8×8 and 10×10 and to recovery starts. Its gate, on
10×10: T1 with no failures and mean moves below the shortcut cycle's 1,445; T2
beating the best reference policy in every occupancy band, the 95% lower bound
of the win rate above that policy's rate on the same starts. The plan also asks
for mixed sizes, replayed mid-game starts, a sweep of the search budget (8, 16,
32 and 64 simulations) at equal compute, and the measured games to a plateau
and throughput, which size the 20×20 run of M3.

**Status:** T2 is met, on the test seeds by both agents. T1 is met only by the
letter: on 1,000 test games the raw policy of the checkpoint chosen on
validation averages 1,444.5 moves, under the gate's 1,445 by half a move, but
the shortcut cycle averages 1,435.1 on the same games
([the gate](#the-m2-gate-test-seeds)). The agents plateau near the shortcut
cycle's pace, and the cause is pinned down
([below](#why-t1-stops-short-of-the-shortcut-cycle)).

### What M2 adds

- **Board mixtures** (`--board 10x10:160 --board 8x8:48 --board 6x6:48`): each
  generation plays that many games on each board. Their examples share one
  canvas, the largest board's, and the assessments use the largest board.
- **Two more kinds of start** (`selfplay.KINDS`), besides M1's standard, prefix
  and free starts: *recovery starts* (`--recovery-share`, `recovery_start`),
  track T2's kind of start (a certified state that no cycle is aligned with)
  taken from a guarded-greedy game of a training seed; and *mid-game starts*
  (`--midgame-share`, `midgame_start`), a state of one of the agent's own
  earlier games replayed with new fruits (review §2.6). Each game leaves a few
  states for later mid-game starts (`--midgame-samples`). The workers build the
  starts, so the guarded-greedy games behind recovery starts run in parallel.
- **A relative moves unit** (`--relative-unit`, `SearchConfig.relative_unit`):
  the envelope search compares expected moves in a share of `moves_scale` at the
  node instead of a fixed number of moves ([below](#the-moves-unit-a-share-of-the-scale)).
- **Training settings** for runs that start from a checkpoint (`--init`): a
  learning-rate warm-up (`--warmup`), a cosine decay (`--lr-floor`), and no
  updates before the buffer holds `--min-replay` examples.
- **Throughput.** The starts go to the workers by the longest-processing-time
  rule, priced by the last generation's network evaluations per game of their
  board and kind, and the log records moves and evaluations per second and how
  busy the workers were. With many workers on one GPU, run NVIDIA's
  Multi-Process Service (`nvidia-cuda-mps-control -d`): without it the workers'
  small batches take turns on the GPU, and 10×10 self-play runs 2.5 to 3.5
  times slower.
- **Batched assessment.** The harness's game loop is a stepper
  (`evaluation.GameRun`), so `assess` plays a worker's games at once and
  evaluates their pending leaves in one batch on the GPU (`play_lockstep`). The
  batched games play exactly what `SearchAgent` and `PolicyAgent` play one game
  at a time (the tests check this move for move). A 10×10 assessment (100 T1
  games and about 220 T2 starts, both agents and the references) takes 3 to 4
  minutes. `assess` also applies the T2 gate (`recovery_gate`).

### The moves unit: a share of the scale

The moves head predicts in units of `moves_scale(F)`, so its errors grow with
it: about 300 moves at the 6×6 start, 2,425 at the 10×10 start, 39,700 on 20×20.
A fixed unit suited to one board is wrong for the next. The M1 network on 10×10
T1 (100 validation games, 16 simulations):

| Search unit | Mean moves | | Relative unit | Mean moves |
| --- | ---: | --- | --- | ---: |
| 4 moves | 1,562.6 | | 0.01 of the scale | 1,567.3 |
| 16 moves | 1,535.2 | | 0.03 | 1,510.9 |
| 64 moves | 1,516.7 | | 0.1 | 1,471.9 |
| 256 moves | 1,474.6 | | 0.3 | **1,452.3** |
| Raw policy | 1,494.6 | | Raw policy | 1,494.6 |

On 6×6 (300 games) the same shares give 215.9, 213.2, 209.6 and 207.3 moves
against 208.3 for the raw policy: 0.3 is the only share where the search beats
the policy on both boards. The improved policy is the policy target, so a unit
too small for the moves head makes the targets worse than the policy itself.
The first 10×10 run, self-play with the one-move unit, made the raw policy
worse within five generations (1,494.6 to 1,732.9 moves; its search at a 4-move
unit, 1,645.7, was still better than its policy). With a 0.3 share in self-play
and assessment, the policy improved instead (1,472.2 at generation 5).

### Why T1 stops short of the shortcut cycle

The main run (`runs/m2-main`): the three boards above, 10% free, 15% prefix
(fill up to 75%), 25% recovery and 20% mid-game starts, the rest standard; 16
simulations at a 0.3 share; 400 steps of 512 examples per generation, a buffer
of a million, warm-up over 3 generations, cosine decay to a tenth; starting
from M1's best checkpoint (`runs/m1-6x6-prefix/model-020.pt`). Validation seeds:
the raw policy over 300 T1 games (95% intervals about ±10 moves), the search
over 100 (±16), and the T2 starts of 100 source games (98, 73 and 46 by band):

| Generation | 10×10 games | T1 raw policy | T1 search | T2 raw policy | T2 search |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 (M1's network) | 0 | 1,482.3 | 1,452.3 | 67, 46, 41: fails | 98, 65, 45: passes |
| 5 | 795 | **1,463.2** | 1,471.9 | 65, 56, 43: fails | 97, 62, 46: passes |
| 10 | 1,588 | 1,470.6 | 1,479.5 | 80, 58, 44: passes | 98, 69, 44: passes |
| 15 | 2,381 | 1,472.3 | 1,487.7 | 84, 56, 39: passes | 97, 68, 44: passes |
| 20 | 3,180 | 1,476.4 | 1,487.7 | 87, 69, 45: passes | 98, 72, 46: passes |

T2 counts are games won per band; the best reference wins 63%, 36% and 24% of
these starts. Every T1 game was won inside the envelope. A generation (256
games, 160 of them on 10×10) took about 100 seconds of self-play and 10 of
training, with 13,600 network evaluations a second and the workers 77% to 83%
busy; an assessment took 200 seconds.

T1 peaks after about 800 10×10 games and then drifts back, as M1's runs did
once the search stopped improving on the policy: by generation 10 it no longer
does at any setting (1,476.7 to 1,512.1 at shares from 1.0 down to 0.03, and
1,484.8 and 1,493.0 with 64 simulations, against 1,479.0 for the policy, 100
games each). Hybrids of the best policy (generation 5) and the shortcut
cycle's rule locate the loss (300 games each, ±10 moves):

| Agent | Mean moves |
| --- | ---: |
| Raw policy | 1,463.2 |
| Shortcut cycle | 1,442.2 |
| The policy to 25% fill, then the shortcut rule | 1,436.5 |
| The shortcut rule to 25% fill, then the policy | 1,462.4 |
| The shortcut rule to 50% fill, then the policy | **1,429.5** |

The policy plays the first quarter and the second half better than the
shortcut cycle, and gives away about 33 moves between 25% and 50% fill, where
it jumps less than the cycle's longest-jump rule. Over the 150 or so decisions
in that range that is a fifth of a move each, and single decisions are
near-ties: at 300 of them (generation 20), each move followed by the shortcut
rule to the end, averaged over 16 fruit seeds shared by the moves, the
policy's move costs 0.07 moves more than the longest jump on average, and the
longest jump is better in 51% of them. The moves head's sibling differences do
not correlate with these rollout differences (r = −0.09). Trusting the head
more in the first half only makes the search worse (shares of 0.003 to 0.03
below 50% fill: 1,514.8 to 1,560.2 moves). The shortcut cycle wins by a habit,
jumping consistently while the snake is short, whose value shows over a whole
trajectory and not in any one decision a one-step search compares. Closing the
gap needs a training signal at that level, or values far more precise than
these.

### The search budget at equal compute

Four runs side by side (`runs/m2-sweep-*`), 5 workers each, with the main run's
mixture and settings except the budget, all from M1's network. Each budget's
games per generation follow its measured cost per 10×10 game (below), and its
steps, buffer and minimum buffer follow the examples those games give, so that
every generation of every run spends about 0.7 million network evaluations.
Validation seeds; the raw policy over 300 T1 games (±10 moves):

| Simulations | Games per generation (10×10, 8×8, 6×6) | Self-play per generation | T1 raw policy, generation 10 | T1 raw policy, generation 20 |
| ---: | --- | ---: | ---: | ---: |
| 8 | 178, 59, 59 | 123 s | 1,464.4 | 1,463.8 |
| 16 | 96, 32, 32 | 146 s | 1,461.0 | 1,472.8 |
| 32 | 47, 16, 16 | 217 s | 1,475.1 | 1,496.2 |
| 64 | 22, 7, 7 | 297 s | **1,450.7** | 1,496.0 |
| 64, another seed, twice the games | 44, 14, 14 | 300 s (12 workers) | 1,474.6 | |

Both agents pass T2 at every assessment of every run (100 source games each),
and no run's search beats its raw policy on T1 (the 64-simulation run's
generation 10: 1,458.1 at 16 simulations, 1,459.7 at 64, against 1,453.2 for
its policy over the same 100 games). At equal evaluations the budget matters
little for what the network learns: generation 10 of the 64-simulation run came
within 6 moves of the gate, but its replication on another seed did not
(1,474.6), and both of its later checkpoints fell back, as do the 16- and
32-simulation runs. The high budgets fall back hardest: they play the fewest
games per generation and keep the smallest buffers. In wall-clock time low
budgets win clearly, because their many games fill the workers' batches: the
same evaluations took 2.4 times longer at 64 simulations (7 games per worker)
than at 8 (59 per worker).

### The M2 gate, test seeds

The checkpoint is the one with the lowest raw-policy T1 mean on validation (300
games) among those whose agents both pass T2 there: generation 10 of the
64-simulation run (1,450.7). Choosing the best of 13 checkpoints favors a lucky
one, and this one is: its replication reached 1,474.6 and its own run 1,496.0
ten generations later. The test seeds give it an unbiased measure: 1,000 T1
games, and the T2 starts of 300 source games (290, 217 and 138 by band); the
search uses 16 simulations and a 0.3 share.

| Agent | T1 won | T1 mean moves (95% interval) | T1 gate (below 1,445) | T2 won by band (95% lower bounds) | T2 gate |
| --- | ---: | ---: | --- | --- | --- |
| Raw policy | 1,000 | 1,444.52 (1,438.81–1,450.23) | passed | 288, 193, 126 (0.975, 0.840, 0.853) | passed |
| Search | 1,000 | 1,448.57 (1,443.11–1,454.03) | failed | 280, 184, 121 (0.938, 0.793, 0.810) | passed |
| Shortcut cycle | 1,000 | 1,435.1 | | | |
| Best reference policy | | | | 67.6%, 46.5%, 31.9% | |

No game left the envelope. The gate's 1,445 is the probe's shortcut-cycle mean
on other seeds; on these 1,000 games the cycle is 9.4 moves faster than the
policy (about two standard errors), so the raw policy passes the T1 gate as
written without beating the cycle. T2 is a clear pass: in every band the 95%
lower bound of both agents' win rates is at least 0.79, where the best
reference wins at most 68%.

### Throughput, and the size of a 20×20 run

On the server (an RTX 4090 and a 24.5-CPU container quota, 22 workers, MPS),
the cost of self-play is the cost of a network evaluation, the leaf's expansion
(move classes, the survival certificate, the encoding) and the batched network
call together: games per hour fall in proportion to the simulations. M1's
network on 10×10 standard starts, 6 games per worker (`runs/m2-bench`):

| Simulations per searched move | Games per hour | Mean moves | Evaluations per second |
| ---: | ---: | ---: | ---: |
| 8 | 11,900 | 1,677 | 13,900 |
| 16 | 6,410 | 1,603 | 15,000 |
| 32 | 3,130 | 1,556 | 14,500 |
| 64 | 1,460 | 1,534 | 14,200 |

(Self-play keeps its Gumbel noise, so these games are slower than T1's.) Free
starts cost the same per evaluation (5,350 games an hour at 16 simulations).
Larger batches help: with 48 positions per worker, an evaluation costs a
worker 0.8 to 1.1 ms on 10×10, 0.9 to 1.5 ms on 14×14 and 1.3 to 2.0 ms on
20×20 (free and envelope positions), so 20×20 needs only 1.5 to 2 times 10×10's
work per evaluation for four times the cells. A 20×20 game at the shortcut cycle's
pace is about 20,700 moves, a third of them searched, about 110,000
evaluations at 16 simulations: 7 to 10 seconds of this server, 360 to 540
games an hour. The review's 3,000-game 20×20 run would take 6 to 8 hours here.
On 10×10, T1 peaked after about 800 games from M1's network, while T2 kept
improving for at least 3,200.

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
| [`evaluation.py`](evaluation.py) | `run_game`, `GameRun`, `AgentSpec`, the tracks, summaries and gates |
| [`exact.py`](exact.py) | `solve`, `solve_envelope`, `ExactSolution`, `evaluate_policy` |
| [`seeds.py`](seeds.py), [`stats.py`](stats.py) | held-out seed ranges; binomial and mean intervals |
| [`reproduce.py`](reproduce.py) | the probe's reference numbers |
| [`encoding.py`](encoding.py) | M1: the network inputs, `mirror`, `moves_scale`, `canonical_envelope` |
| [`search.py`](search.py) | M1: Gumbel root search, `SearchAgent`, `PolicyAgent`, `policy_move`; M2: the relative unit |
| [`network.py`](network.py) | M1: `SnakeNet`, `NetworkEvaluator`, checkpoints (PyTorch) |
| [`selfplay.py`](selfplay.py) | M1 and M2: batched self-play, the kinds of start, training examples |
| [`train.py`](train.py) | M1 and M2: the training loop over board mixtures, worker balancing (PyTorch) |
| [`assess.py`](assess.py) | M1 and M2: batched assessment against the gates (PyTorch) |
| [`__main__.py`](__main__.py) | `python -m snake_agent evaluate | solve | reproduce | train | assess` |
