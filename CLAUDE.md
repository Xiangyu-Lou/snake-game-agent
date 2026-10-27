# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An AI that learns to beat Snake. The goal is to fill the whole board, and then to do it in as few moves as possible. Thinking time per move does not matter. The agent reads the exact game state, never pixels.

- `snake_game/`: the environment. An exact, deterministic engine (`engine.py`), Gymnasium environments for one game (`env.py`) or thousands stepped together in NumPy (`vector_env.py`), and an optional pygame UI (`ui/`). Its documentation is `snake_game/README.md`.
- `snake_agent/`: the agent, built milestone by milestone after the Architecture B plan in `analysis/architecture-b-review.md`. §3 lists the milestones and their gates; §4 the owner's decisions, which the code implements. Milestone M0 is done: the contract, the exact instruments, the evaluation harness and an exact 4×4 solution. M1, self-play on 4×4 and 6×6, is in progress: `encoding.py` holds the network inputs and `search.py` the Gumbel search (a draft without unit tests). Its documentation is `snake_agent/README.md`.
- `analysis/`: the design reports and the probe scripts behind them.

The two package READMEs document the API and the measured numbers. Update them when behaviour changes.

## Commands

```bash
pip install -e ".[dev]"                                       # Python >= 3.10; includes pytest and pygame-ce
python -m pytest -W error                                     # the suite as CI runs it: warnings are errors
python -m pytest -W error --slow                              # adds the slow tests (full probe reproduction, exact 6x6)
python -m pytest -W error tests/test_exact.py::test_4x4_size  # one test (or -k <pattern>)
ruff check . && ruff format --check snake_game snake_agent tests  # ruff is not in the dev extras: install it separately

python -m snake_game --agent hamiltonian --mode auto --speed max  # play or watch a game (needs pygame-ce)
python -m snake_agent evaluate t1 --agent shortcut --board 10x10  # evaluate an agent on held-out games
python -m snake_agent solve --board 4x4                           # exact numbers for a small board
python -m snake_agent reproduce [--full]                          # check the probe's reference numbers
```

CI runs only pytest (`.github/workflows/tests.yml`), on Python 3.10 to 3.14. One more job runs the oldest supported dependencies: numpy 1.24, gymnasium 1.1 and pygame-ce 2.5.0. Code must work with those. Ruff's settings, including the 110-character line length, are in `pyproject.toml`. The UI tests run headless: `tests/conftest.py` sets SDL's dummy drivers.

## Conventions that span files

- **Coordinates.** Positions are `(x, y)`, and `y` grows downward. Arrays are indexed `[y, x]`. Flat cell indices are `y * width + x`.
- **Directions and actions.** `Direction` is `UP, RIGHT, DOWN, LEFT`, numbered 0–3 clockwise. `RelativeAction` is `LEFT, STRAIGHT, RIGHT`. A reversal request means "go straight".
- **Fruit is the only randomness.** It comes from a counter-based SplitMix64 stream per episode (`snake_game/_rng.py`), shared by `SnakeGame` and `VectorSnakeEnv`.
- **The two engines must play identical games.** After `reset(seed=s)`, game `i` of `VectorSnakeEnv` is `SnakeEnv` with seed `s + i`, and the tests check this move for move. Change the rules in both engines.
- **Reference players.** `snake_game.policies` holds reference players, not the project's AI, and they return absolute `Direction`s. Agents in `snake_agent` return `RelativeAction`s.

## The agent's contract (`snake_agent/contract.py`)

Training, search and evaluation all follow these rules:

- **Public state only.**
  - Agents receive a `GameState`, never the live `SnakeGame`: clones of the live game replay its hidden fruit stream.
  - Simulate with `planning_game(state, seed)`, which draws future fruits from an independent seed.
- **Safety strictly first.**
  - Maximize the probability of filling the board.
  - Fewer moves only decide between moves whose win values tie exactly (`choose_safely`).
  - Never trade death risk for speed.
- **The envelope on standard starts.** Games from the standard start stay inside the cycle-order envelope (`cycles.CycleEnvelope`).
  - There every move keeps the win certain, so any failure is a bug, and the T1 gate measures moves.
  - The board's canonical cycle is the first cycle from `snake_game.policies.hamiltonian_cycles` that the standard start is aligned with (`encoding.canonical_envelope`).
- **The fruit timeout.**
  - Under `TRAINING_RULES`, a game is lost after `N + L` moves without fruit (`N` cells, `L` the snake's length).
  - `NO_LIMIT` is the engine's own rules; the harness ends endless games as censored.
- **Pruning.**
  - The survival certificate (`survival.Board.has_survival_certificate`) is a hint and an input feature, never a reason to prune.
  - Search prunes only `FATAL` moves and moves that `prove_lost` proves lost.
- **Seeds.**
  - `seeds.TRAIN` is for training and experiments.
  - `VALIDATION` and `TEST` are held out, and `TEST` is for the final gates only.
  - Training code passes its seeds through `check_training_seed`.

## Reference numbers

Results are judged against exact or reproduced numbers, so keep these consistent when you change the instruments:

- **The probe's results.** `analysis/architecture_b_probe_results.json` is the committed output of the Architecture B probe.
  - `python -m snake_agent reproduce --full` checks the package against it. The quick sections run in the test suite.
  - Running a probe script rewrites its JSON. Review the diff before committing a rerun.
- **Exact solutions.** `snake_agent/exact.py` solves 4×4 exactly and solves the 6×6 envelope (`solve_envelope`). Its tests pin the results:
  - From the 6×6 standard start, the shortcut cycle (the T1 reference) needs 212.78 expected moves.
  - Optimal play inside the envelope needs 201.91.

## Repository rules

- **`analysis/`.**
  - Documents there are written in English.
  - A pull request for analysis work contains only `analysis/`. Changes to code or other files need a separate, explicit request.
- **Model files.** `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`, `*.h5`, `*.keras`, `*.zip` and `*.npz` are stored with Git LFS (`.gitattributes`). Run `git lfs install` before committing such files.
- **PyTorch.** It is the chosen stack for the network (review §4), but it is not a dependency yet.
