# snake-game-agent

An AI that learns to beat Snake: to fill the entire board with the snake's body.

The project chases the highest possible score and does not care about speed: the
agent may take as long as it likes to choose each move. It does not look at the
screen either; it reads the exact positions of the snake and the fruit.

<p align="center"><img src="snake_game/docs/gameplay.gif" alt="Snake gameplay" width="320"></p>

## Roadmap

1. **A Snake game to train on** (done): an exact, deterministic engine, Gymnasium
   environments for one game or thousands at once, and an animated UI that can be
   switched off during training.
   **Documentation: [`snake_game/README.md`](snake_game/README.md)**
2. **The agent** (in progress): a model trained on the game's exact state to play
   a perfect game, following the [Architecture B plan](analysis/architecture-b-review.md).
   Milestone M0 is done: the information contract, the exact instruments
   (survival certificate, lost-proof search, cycle-order envelope, shortcut
   cycle), the evaluation harness for standard and recovery starts, and an exact
   solution of 4×4. It reproduces the reference numbers of the plan's probe.
   Milestone M1 adds the network, the Gumbel search, self-play and training,
   learned from scratch on 4×4 and 6×6 (`pip install -e ".[train]"` for PyTorch).
   Milestones M2 and M3 take it to 10×10 and 20×20, where it wins every test
   game. After M3 the **plan agent** clears the board far faster: it plays free
   while the snake is short, then finishes on a Hamiltonian-cycle certificate
   that it rebuilds at every fruit. With its current network (a residual CNN of
   0.6 M parameters trained by its own self-play; an interim result, training
   continues) it wins 298 of 299 test games of 20×20 in 11,047 moves on
   average, 46% fewer than M3's agent.
   **Documentation: [`snake_agent/README.md`](snake_agent/README.md)**

<p align="center"><a href="snake_agent/docs/plan-agent-20x20.mp4"><img src="snake_agent/docs/plan-agent-20x20.png" alt="The plan agent on 20×20" width="320"></a><br>
<a href="snake_agent/docs/plan-agent-20x20.mp4">Watch the plan agent fill a 20×20 board</a> (100 s video)</p>

## Quick start

```bash
pip install -e ".[ui]"
python -m snake_game                                             # play: one move per key press
python -m snake_game --mode auto                                 # classic real-time Snake
python -m snake_game --agent hamiltonian --mode auto --speed max # watch a perfect game
python -m snake_agent evaluate t1 --agent shortcut --board 10x10 # evaluate on held-out games
```

```python
from snake_game import VectorSnakeEnv

envs = VectorSnakeEnv(num_envs=256)  # 256 games stepped together with NumPy
obs, infos = envs.reset(seed=0)      # obs: (256, 8, 20, 20) exact board encoding
```

## Model files

Trained models are stored with [Git LFS](https://git-lfs.com). Install it once
(`git lfs install`) before cloning or committing models; files matching the
patterns in [`.gitattributes`](.gitattributes) (`*.pt`, `*.pth`, `*.ckpt`,
`*.safetensors`, `*.onnx`, `*.h5`, `*.keras`, `*.zip`, `*.npz`) are then kept in LFS
automatically.

## Repository layout

| Path | Contents |
| --- | --- |
| [`snake_game/`](snake_game/) | the game: engine, Gymnasium environments, pygame UI ([README](snake_game/README.md)) |
| [`snake_agent/`](snake_agent/) | the agent: contract, exact instruments, evaluation harness, exact solutions, network, search, self-play and training ([README](snake_agent/README.md)) |
| [`analysis/`](analysis/) | design reports and the probes behind them ([README](analysis/README.md)) |
| [`tests/`](tests/) | test suite: `pip install -e ".[dev]"`, then `pytest` |
| [`.github/workflows/`](.github/workflows/) | continuous integration |

## License

[MIT](LICENSE)
