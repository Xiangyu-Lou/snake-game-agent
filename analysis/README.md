# Project Analysis

This directory contains analysis reports, proposed solutions, and supporting
technical notes for the project. All documents in this directory must be written
in English.

The assistant's role is to analyze problems and propose solutions. Pull requests
for this work should include only content under `analysis/`. Changes to application
code or other files require a separate explicit request.

## Design Summary

The task is to fill the board using exact game state and one agent action per
move, without a visual-perception or real-time requirement. The current standard
20×20 game has an applicable Hamiltonian reference strategy; learned-agent results
must be compared against it.

The proposed learning path starts with an imitation/RL baseline and adds exact
simulator search guided by a policy/value network. The reports also consider
graph-based hierarchical planning and certified cycle execution.

- Architecture B uses one current-state snapshot: `W × H + 1` tokens, with
  12 features per cell and 10 global features. History stacking is unnecessary
  for the current complete-state contract.
- The first-run warm-up uses `0.2 / (W × H)` per fruit, `-1` for collision, a
  `+1` completion bonus, and zero movement reward. The final objective uses only
  a terminal success reward. Reward scales, timeout handling, and separate value
  targets are specified explicitly.
- Existing-policy rollouts and interface checks have been executed. Neural
  architectures, training settings, and evaluation gates are proposals; no
  learned model has been trained as part of this analysis.
- The Architecture B review keeps B as the main design and recommends six
  changes: gate B on recovery and demonstration-free tracks, because imitation
  alone can pass the standard-start gate; count N moves without fruit as a
  training loss, because loops become the main failure once deaths are
  prevented; use an exact survival certificate in search, inputs, and labels;
  use Gumbel root search with 8–32 simulations and adaptive budgets; add
  bootstrapped and exact value labels; and replace PPO with self-play from 4×4
  upward, using size-independent geometry.

## Reports

- [Architectures for a Snake Agent That Fills the Board](agent-architecture-design.md):
  four model/system designs, a recommended implementation sequence, and an
  evaluation protocol grounded in the existing game interfaces.
- [Architecture probe](architecture_probe.py) and
  [measured results](architecture_probe_results.json): reproducible baseline
  rollouts and observation/interface checks supporting the report.
- [Architecture B input specification](architecture-b-input-specification.md):
  cell/global features, tensor shapes, attention relations, and construction
  examples using the existing game-state encoder.
- [Context management and initial rewards](context-and-initial-rewards.md):
  single-state context, trajectory/search memory, a concrete reward warm-up,
  reward/penalty scales, and transition to the terminal-completion objective.
- [Architecture B review](architecture-b-review.md): verification of the reports,
  six gaps in the B plan with measured evidence, a milestone plan with gates, and
  the decisions left to the project owner.
- [Architecture B probe](architecture_b_probe.py) and
  [measured results](architecture_b_probe_results.json): survival certificate,
  death horizons, loop outcomes, recovery baselines, branching, and search and
  network costs supporting the review.

## Reproduce the Baseline Evidence

From the repository root, with development dependencies installed:

```bash
.venv/bin/python analysis/architecture_probe.py
```

The script runs 18 standard-board episodes and checks feature aliasing and action
mask behavior. It rewrites `architecture_probe_results.json` with the current
source commit and measurements; the committed result records the revision used
for the report. Review generated differences before committing another run.

The review's measurements are reproduced the same way:

```bash
.venv/bin/python analysis/architecture_b_probe.py
```

It takes about three minutes and rewrites `architecture_b_probe_results.json`.
Outcomes and counts are deterministic; timings depend on the machine.
