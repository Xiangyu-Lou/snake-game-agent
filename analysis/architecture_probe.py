"""Reproduce observations supporting the architecture report; no model training.

Run from the repository root:
    .venv/bin/python analysis/architecture_probe.py
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
from pathlib import Path

import numpy as np

from snake_game import Direction, GameConfig, GameState, SnakeEnv, SnakeGame, observe
from snake_game.policies import GreedyPolicy, HamiltonianPolicy, RandomPolicy


def baseline_results() -> list[dict]:
    config = GameConfig(20, 20)
    results = []
    for name, policy_class in (
        ("random", RandomPolicy),
        ("greedy", GreedyPolicy),
        ("hamiltonian", HamiltonianPolicy),
    ):
        episodes = []
        for seed in range(6):
            policy = policy_class(seed=seed) if name == "random" else policy_class()
            game = SnakeGame(config, seed=seed)
            # A measurement cap, not a game rule or a failure label.
            while not game.done and game.steps < 160_000:
                game.step(policy(game.state()))
            episodes.append(
                {
                    "seed": seed,
                    "score": game.score,
                    "steps": game.steps,
                    "outcome": "won" if game.won else "lost" if game.done else "censored",
                }
            )
        result = {
            "policy": name,
            "episodes": episodes,
            "wins": sum(e["outcome"] == "won" for e in episodes),
            "censored": sum(e["outcome"] == "censored" for e in episodes),
            "mean_score": statistics.mean(e["score"] for e in episodes),
            "mean_steps": statistics.mean(e["steps"] for e in episodes),
        }
        print(json.dumps(result), flush=True)
        results.append(result)
    return results


def feature_alias() -> dict:
    """Find two valid, different bodies with identical 19-feature observations."""
    seen = {}

    def search(body):
        if len(body) == 8:
            state = GameState(6, 6, tuple(body), Direction.RIGHT, (0, 0), score=5)
            state.validate()
            key = tuple(observe(state, "features").tolist())
            if key in seen and seen[key].snake != state.snake:
                other = seen[key]
                assert not np.array_equal(observe(other, "grid"), observe(state, "grid"))
                return {"first": other.to_dict(), "second": state.to_dict(), "features": list(key)}
            seen[key] = state
            return None
        x, y = body[-1]
        for direction in Direction:
            dx, dy = direction.delta
            cell = (x + dx, y + dy)
            if not (0 <= cell[0] < 6 and 0 <= cell[1] < 6):
                continue
            if cell == (0, 0) or cell in body:
                continue
            if len(body) == 1 and cell == (4, 3):
                continue  # Heading cannot point into the neck.
            found = search([*body, cell])
            if found is not None:
                return found
        return None

    result = search([(3, 3)])
    assert result is not None, "No feature collision found in the selected state set"
    return result


def mask_and_timeout_observations() -> dict:
    trapped = GameState(
        3, 3, ((1, 1), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2)),
        Direction.UP, (2, 2),
    )
    env = SnakeEnv(3, 3)
    env.reset(seed=0, options={"state": trapped})
    safe = [int(d) for d in env.game.safe_directions()]
    mask = env.action_masks().tolist()
    assert not safe and any(mask)
    default_limit = SnakeEnv().max_steps_without_fruit
    env.close()
    return {
        "trapped_state": trapped.to_dict(),
        "safe_actions": safe,
        "gym_mask": mask,
        "default_20x20_fruit_timeout": default_limit,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "board": [20, 20], "initial_length": 3, "seeds": list(range(6)),
            "max_episode_steps": 160_000, "runner": "SnakeGame (no Gymnasium truncation)",
            "purpose": "Illustrative baseline and interface checks, not a model comparison",
        },
        "baselines": baseline_results(),
        "feature_alias": feature_alias(),
        "interface_checks": mask_and_timeout_observations(),
    }
    output = Path(__file__).with_name("architecture_probe_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
