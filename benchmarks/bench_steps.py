"""Throughput benchmark: random-legal-action rollouts, steps/sec.

Usage: .venv/bin/python benchmarks/bench_steps.py [--steps 20000]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Allow running without an editable install (e.g. system python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def bench_engine(n_steps: int) -> float:
    """Engine-only: Game.legal_actions + apply, no obs encoding."""
    from balatro_gym.content import load_all
    from balatro_gym.engine import Game, Phase

    load_all()
    rng = random.Random(0)
    done_steps = 0
    t0 = time.perf_counter()
    episode = 0
    while done_steps < n_steps:
        game = Game(seed=f"bench-{episode}")
        episode += 1
        while game.run.phase not in (Phase.WON, Phase.GAME_OVER) and done_steps < n_steps:
            acts = game.legal_actions()
            game.apply(acts[rng.randrange(len(acts))])
            done_steps += 1
    return done_steps / (time.perf_counter() - t0)


def bench_env(n_steps: int) -> float:
    """Full env: mask + step + obs encoding."""
    import numpy as np

    from balatro_gym.env.envs import BalatroEnv

    env = BalatroEnv()
    rng = np.random.default_rng(0)
    done_steps = 0
    seed = 0
    t0 = time.perf_counter()
    env.reset(seed=seed)
    while done_steps < n_steps:
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            seed += 1
            env.reset(seed=seed)
            continue
        _, _, term, trunc, _ = env.step(int(rng.choice(legal)))
        done_steps += 1
        if term or trunc:
            seed += 1
            env.reset(seed=seed)
    return done_steps / (time.perf_counter() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args()

    engine_sps = bench_engine(args.steps)
    print(f"engine-only: {engine_sps:,.0f} steps/sec")
    env_sps = bench_env(args.steps)
    print(f"full env:    {env_sps:,.0f} steps/sec")
