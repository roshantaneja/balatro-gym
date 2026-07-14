"""Human/debug REPL: play the environment by hand.

Usage: .venv/bin/python scripts/play_repl.py [--env blind|full] [--seed 0]
Commands: a number (action index), a legal-action list number like #3,
'legal' to relist, 'state' for raw state, 'q' to quit.
"""

from __future__ import annotations

import argparse

import numpy as np

from balatro_gym.env import action_codec as codec
from balatro_gym.env.envs import BalatroBlindEnv, BalatroEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["full", "blind"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = (BalatroEnv if args.env == "full" else BalatroBlindEnv)(render_mode="ansi")
    obs, info = env.reset(seed=args.seed)
    total = 0.0

    while True:
        print("\n" + env.render())
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            print("terminal state.")
            break
        strs = [f"#{i} {codec.to_string(int(a))} ({a})" for i, a in enumerate(legal)]
        print("legal:", "  ".join(strs[:40]) + (" ..." if len(strs) > 40 else ""))

        raw = input("> ").strip()
        if raw in ("q", "quit", "exit"):
            break
        if raw == "legal":
            continue
        if raw == "state":
            print(env.run.to_dict())
            continue
        try:
            action = int(legal[int(raw[1:])]) if raw.startswith("#") else int(raw)
        except (ValueError, IndexError):
            print("bad input")
            continue
        obs, r, terminated, truncated, info = env.step(action)
        total += r
        print(f"reward {r:+.3f} (total {total:+.3f})")
        for e in info.get("events", []):
            print(f"  event: {e}")
        if terminated or truncated:
            print(f"episode over. stats: {info.get('episode_stats')}")
            break


if __name__ == "__main__":
    main()
