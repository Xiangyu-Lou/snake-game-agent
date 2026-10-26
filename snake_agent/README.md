# Snake agent

The agent's side of the project. This package holds milestone **M0, contracts
and instruments**, of the [Architecture B plan](../analysis/architecture-b-review.md#3-revised-plan-for-b):
the rules every agent plays by, the exact tools that training and search will
use, the evaluation harness, and an exact solution of 4×4. Nothing here
learns; M1 adds the network and search on top.

M0's gate is met: the tests pass, and the harness reproduces the reference
numbers of the [Architecture B probe](../analysis/architecture_b_probe.py):
every section of its results except the machine-dependent timings and the
analytic network cost model ([below](#reproducing-the-probe)).

- [Quick start](#quick-start)
- [The information contract](#the-information-contract)
- [Exact instruments](#exact-instruments)
- [Evaluation harness](#evaluation-harness)
- [4×4 solved exactly](#44-solved-exactly)
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
| [`exact.py`](exact.py) | `solve`, `ExactSolution`, `evaluate_policy` |
| [`seeds.py`](seeds.py), [`stats.py`](stats.py) | held-out seed ranges; binomial and mean intervals |
| [`reproduce.py`](reproduce.py) | the probe's reference numbers |
| [`__main__.py`](__main__.py) | `python -m snake_agent evaluate | solve | reproduce` |
