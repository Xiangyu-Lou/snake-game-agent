# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An AI that learns to beat Snake. The goal is to fill the whole board, and then to do it in as few moves as possible. Thinking time per move does not matter. The agent reads the exact game state, never pixels.

- `snake_game/`: the environment. An exact, deterministic engine (`engine.py`), Gymnasium environments for one game (`env.py`) or thousands stepped together in NumPy (`vector_env.py`), and an optional pygame UI (`ui/`). Its documentation is `snake_game/README.md`.
- `snake_agent/`: the agent, built milestone by milestone after the Architecture B plan in `analysis/architecture-b-review.md`. §3 lists the milestones and their gates; §4 the owner's decisions, which the code implements. Milestone M0 is done: the contract, the exact instruments, the evaluation harness and an exact 4×4 solution. M1, self-play on 4×4 and 6×6, adds `encoding.py` (the network inputs), `search.py` (the Gumbel search), `network.py` (the PyTorch network), `selfplay.py` and `train.py` (self-play and training) and `assess.py` (the gates); its gate is met on the test seeds. M2, 8×8 and 10×10 with recovery, adds board mixtures, recovery and mid-game starts, worker balancing, batched assessment (`evaluation.GameRun`, `assess.play_lockstep`) and a moves unit that scales with the board (`SearchConfig.relative_unit`). Its T2 gate is met on the test seeds. Its T1 gate (10×10 below 1,445 moves) was first met only by the letter: the M2 checkpoint's raw policy averages 1,444.5 on the test seeds, while the shortcut cycle averages 1,435.1 on the same games, because the cycle's advantage is a habit that one-step comparisons by the moves head cannot see. `habits.py` closes the gap with a training signal at that level: a jump bias per band of fill, compared on paired windows that share their fruits, then distilled into the network (`train.distill`, `train.improve_habits`, `python -m snake_agent habits`); `assess` compares T1 with the shortcut cycle on the same seeds (`versus_shortcut`). After four distillations the raw policy averages 1,406.2 moves on the test seeds, 28.9 fewer than the shortcut cycle on the same games (the search 30.1 fewer), and T2 still passes. M3, 20×20, adds adaptive search budgets (`SearchConfig.confident`, `careful_fill`; playout-cap randomization in self-play), a free-mode win unit (`SearchConfig.win_unit`), an example cap per game and bounded GPU memory (`NetworkEvaluator.max_batch`, `release`). Its gate is met on the test seeds by the search with the adaptive budget (the prior inside the envelope, a search with 0.3 units in free play) on a network given one round of 20×20 habits: all 299 T1 games won in 20,581.4 moves on average, 301.1 fewer than the shortcut cycle on the same games, and the best reference beaten in every T2 band. After M3, by later owner decisions, `plans.py` adds the plan agent, which clears the board far faster, a slightly lower clear rate accepted. It decides once per fruit among exact paths to the fruit (shortest, sweeping and waiting plans) and waiting moves, valued by a network of afterstates with arrival-time input planes (`NetworkConfig.arrival`; no canonical-cycle inputs, `NetworkConfig.cycle=False`), trained by its own self-play (`train --agent plans`), and assessed as free T1 (`assess --agents plans`, `assess.assess_free`). Pure free play (`--certify none`) left holes late in the game and was slower than M3 on 20×20, so the agent plays free until 30% of the board is filled and then finishes on a Hamiltonian-cycle certificate that it rebuilds at every fruit (`certify.py`: cycles around spanning trees of 2×2 blocks, an exact check, certified paths and shortcuts, and the transition from free play; `PlanConfig.certify`). On the test seeds it wins 298 of 299 T1 games of 20×20 in 11,046.7 moves on average, 46% fewer than M3's agent at its gate, and 999 of 1,000 of 10×10 in 925.3, 34% fewer than the habits network, and it passes the T2 gate on both boards. After that, by the owner's next targets, the plan agent gets a Transformer (`NetworkConfig(kind="transformer")`, `network.SnakeTransformer`: a short convolutional stem, then attention between 2×2 tokens with two-dimensional rotary positions), plays 100×100, and switches to the certified finish later. 100×100 needed compiled kernels (`fast.py`, Numba, an optional dependency: the searches of `plans.py`, `certify.py` and the encoder over flat arrays, 10 to 25 times faster), sampled self-play records (`plans.play(max_examples=...)`), a replay per board canvas (`train.Replays`) and evaluation board by board (`NetworkEvaluator(max_cells=...)`). Free play now runs to 40% of the board (`PlanConfig.certify` 0.4); later than that it costs more per fruit than the certified finish with every network, also with `NetworkConfig.horizon` (a moves head over the next K fruits, which makes late switches safe, not faster). The Transformer plays level with the CNN. Trained on boards up to 20×20 it clears 100×100 in 45% of the shortcut cycle's moves; so does a CNN trained the same way, while the plan agent's CNN, trained in pure free play, times out there. On the test seeds the Transformer with an 8-fruit horizon wins all 299 T1 games of 20×20 in 10,978.4 moves and passes T2, and all 8 games of 100×100 in 5,693,532.6. Then, by the owner's next target, free play can run to 80% of the board (`--certify 0.8`). From a dense free-play body the old transition rarely completed (4 of 100 games of 20×20 won), so the transition now waits for its certificate along a path of the cycles' moves that replaces the rest of the body without eating (`PlanConfig.stall`, `certify.stall`, `fast.stall_run`: restarted depth-first searches; it follows a full path, or the part of the longest path after which the snake keeps a survival certificate). Free play gets paths beside the body and into the strips one cell wide beside them, and a cost for holes that stay closed (`PlanConfig.hugs`, `gaps`, `persistence`), and the Transformer is retrained switching at 80% (`runs/t-f80b`; 10×10 left out, where self-play at 80% collapsed). Free play to 80% then wins 96 of 100 validation games of 20×20 in 15,210 moves, 40% more than switching at 40%, and all 8 test games of 100×100 in 8,672,857.1, 52% more than switching at 40% and 69% of the shortcut cycle's moves: beyond 50% free play still costs more per fruit than the certified finish (switching at 70% costs 19% more on 20×20). The waiting path and the three options are defaults; the default switch stays at 40%. Its documentation is `snake_agent/README.md`.
- `analysis/`: the design reports and the probe scripts behind them.

The two package READMEs document the API and the measured numbers. Update them when behaviour changes.

## Commands

```bash
pip install -e ".[dev]"                                       # Python >= 3.10; includes pytest, pygame-ce and Numba
python -m pytest -W error                                     # the suite as CI runs it: warnings are errors
python -m pytest -W error --slow                              # adds the slow tests (full probe reproduction, exact 6x6; several minutes)
python -m pytest -W error tests/test_exact.py::test_4x4_size  # one test (or -k <pattern>)
ruff check . && ruff format --check snake_game snake_agent tests  # ruff is not in the dev extras: install it separately

python -m snake_game --agent hamiltonian --mode auto --speed max  # play or watch a game (needs pygame-ce)
python -m snake_agent evaluate t1 --agent shortcut --board 10x10  # evaluate on held-out games (t2: recovery starts)
python -m snake_agent evaluate t1 --agent my_module:my_agent      # evaluate your own agent; --split test is for final gates only
python -m snake_agent solve --board 4x4                           # exact numbers for a small board
python -m snake_agent reproduce [--full]                          # check the probe's reference numbers
python -m snake_agent train --board 6x6 --out runs/m1-6x6 --device cuda  # M1 self-play training (needs the train extra)
python -m snake_agent assess runs/m1-6x6/model.pt --games 300    # a trained network against the gates
python -m snake_agent train --board 10x10:160 --board 8x8:48 --relative-unit 0.3 --out runs/m2 --device cuda  # a mixture
python -m snake_agent habits runs/m2/model-010.pt --out runs/habits --rounds 3 --device cuda  # habit rounds, then distillation
python -m snake_agent assess runs/m3-habits/model-h01.pt --board 20x20 --games 299 --sources 500 --relative-unit 0.3 --win-unit 0.3 --confident 0 --careful-fill 0 --split test  # the M3 gate
python -m snake_agent train --agent plans --board 20x20:32 --board 14x14:16 --board 10x10:32 --free-share 0 --midgame-share 0.4 --sweeps 0.05,0.25,1 --out runs/plans --device cuda  # the plan agent
python -m snake_agent assess runs/plans/model.pt --agents plans --board 20x20 --games 100 --sources 100  # free T1 and T2 (--certify none: pure free play)
python -m snake_agent train --agent plans --network transformer --blocks 2 --init runs/t-small/model-040.pt --board 20x20:32 --board 14x14:16 --board 10x10:32 --free-share 0 --midgame-share 0.4 --certify 0.5 --out runs/t-20 --device cuda  # the Transformer (stage B; --horizon 8: the moves head looks 8 fruits ahead)
python -m snake_agent assess runs/t-h8/model-020.pt --agents plans --board 100x100 --games 8 --sources 0  # 100x100 (about 30 minutes a game; no T2 starts)
python -m snake_agent assess runs/t-f80b/model-020.pt --agents plans --board 100x100 --games 8 --sources 0 --certify 0.8  # free play to 80% (about 2 hours a game)
```

CI runs only pytest (`.github/workflows/tests.yml`), on Python 3.10 to 3.14. One more job runs the oldest supported dependencies: numpy 1.24, gymnasium 1.1 and pygame-ce 2.5.0. Code must work with those. A third job adds PyTorch's CPU build, so that the network tests run (they skip without PyTorch). Ruff's settings, including the 110-character line length, are in `pyproject.toml`. The UI tests run headless: `tests/conftest.py` sets SDL's dummy drivers.

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
- **The envelope on standard starts.** Games from the standard start stay inside the cycle-order envelope (`cycles.CycleEnvelope`). The plan agent (`plans.py`) is the one exception, by the owner's later decisions: it plays free, then finishes on its own certificate (`certify.py`), a Hamiltonian cycle it rebuilds at every fruit, never the canonical cycle or its envelope.
  - There every move keeps the win certain, so any failure is a bug, and the T1 gate measures moves.
  - The board's canonical cycle is the first cycle from `snake_game.policies.hamiltonian_cycles` that the standard start is aligned with (`encoding.canonical_envelope`).
- **The fruit timeout.**
  - Under `TRAINING_RULES`, a game is lost after `N + L` moves without fruit (`N` cells, `L` the snake's length).
  - `NO_LIMIT` is the engine's own rules; the harness ends endless games as censored.
- **Pruning.**
  - The survival certificate (`survival.Board.has_survival_certificate`) is a hint and an input feature, never a reason to prune.
  - Search prunes only `FATAL` moves, moves that `prove_lost` proves lost, moves that time out, and in free mode moves back to a position of the current fruit's epoch (the same position with less budget).
  - The moves value is the expected moves until the board is full if it is filled: losses never enter it, and only exact wins (the envelope, a filled board) tie at a win value of 1.
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
- **Model files and videos.** `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`, `*.h5`, `*.keras`, `*.zip`, `*.npz` and `*.mp4` are stored with Git LFS (`.gitattributes`): the repository keeps only a pointer to each. Run `git lfs install` before committing such files.
- **PyTorch.** The network's stack (review §4), an optional dependency: `pip install -e ".[train]"`. Only `network.py`, `train.py` and `assess.py` import it.
- **Numba.** Compiles the kernels of `fast.py`, an optional dependency (`pip install -e ".[fast]"`, also in `dev` and `train`). Without it the kernels run as plain Python, the same code and results, only slower. Kernels take flat NumPy arrays, never Python lists, and draw random numbers from their own generator (`fast.generator`), so compiled and plain runs agree.
- **Training runs.** `train` writes logs and checkpoints to `runs/`, which git ignores. Commit a checkpoint only on request (Git LFS). With many self-play workers on one GPU, start NVIDIA MPS first (`nvidia-cuda-mps-control -d`), or their small batches take turns on the GPU.
