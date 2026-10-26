"""Throughput benchmark: ``python -m snake_game.benchmark [--steps N]``.

Measures environment steps per second (random actions, observations included)
for the single-game :class:`SnakeEnv` and the batched :class:`VectorSnakeEnv`.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .env import SnakeEnv
from .vector_env import VectorSnakeEnv


def single_env_rate(obs_type: str, steps: int, seed: int = 0) -> float:
    env = SnakeEnv(obs_type=obs_type)
    env.reset(seed=seed)
    actions = np.random.default_rng(seed).integers(0, 4, size=steps)
    start = time.perf_counter()
    for action in actions:
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            env.reset()
    return steps / (time.perf_counter() - start)


def vector_env_rate(obs_type: str, num_envs: int, steps: int, seed: int = 0) -> float:
    env = VectorSnakeEnv(num_envs, obs_type=obs_type)
    env.reset(seed=seed)
    actions = np.random.default_rng(seed).integers(0, 4, size=(steps, num_envs))
    start = time.perf_counter()
    for batch in actions:
        env.step(batch)
    return steps * num_envs / (time.perf_counter() - start)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m snake_game.benchmark", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--steps", type=int, default=200, help="batched steps per measurement (default: 200)")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[64, 256, 1024])
    args = parser.parse_args(argv)
    print(f"{'environment':<28}{'obs':>10}{'steps/s':>14}")
    for obs_type in ("grid", "coords", "features"):
        rate = single_env_rate(obs_type, args.steps * 10)
        print(f"{'SnakeEnv':<28}{obs_type:>10}{rate:>14,.0f}")
        for n in args.num_envs:
            rate = vector_env_rate(obs_type, n, args.steps)
            print(f"{f'VectorSnakeEnv({n})':<28}{obs_type:>10}{rate:>14,.0f}")


if __name__ == "__main__":
    main()
