"""Collect planner-labelled decisions for the typed decision model.

    python scripts/typed/collect.py --seeds 50 --start 0 --out data/typed/part0.npz

Every decision of a full planner run is one record: the env observation,
the legal-action mask, the planner's action (the label), the phase, the
ante and, once the run ends, whether it was won (the outcome label every
record of that run shares). Records are saved as an .npz of stacked arrays.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from balatro_gym.engine import HandType, Phase
from balatro_gym.env.envs import BalatroEnv
from balatro_gym.env.planner import PlannerAgent

PHASES = {"blind_select": 0, "playing": 1, "shop": 2, "pack_open": 3, "target": 4}


def collect(seed: int, deck: str, stake: str) -> dict:
    agent = PlannerAgent(commit=HandType.PAIR)
    env = BalatroEnv(deck=deck, stake=stake)
    obs, info = env.reset(seed=seed)
    records: dict[str, list] = {k: [] for k in obs}
    masks, actions, phases, antes, blinds = [], [], [], [], []
    done, steps = False, 0
    while not done and steps < 4000:
        mask = info["action_mask"]
        a = agent.act(env, obs, mask)
        for k, v in obs.items():
            records[k].append(np.asarray(v))
        masks.append(mask.copy())
        actions.append(a)
        phases.append(PHASES.get(env._env_phase(), 5))
        antes.append(env.run.ante)
        blinds.append(env.run.blind_index)
        obs, _r, term, _trunc, info = env.step(a)
        done, steps = term, steps + 1
    won = env.run.phase == Phase.WON
    n = len(actions)
    out = {k: np.stack(v) for k, v in records.items()}
    out.update(
        mask=np.stack(masks),
        action=np.asarray(actions, dtype=np.int64),
        phase=np.asarray(phases, dtype=np.int64),
        ante=np.asarray(antes, dtype=np.int64),
        blind=np.asarray(blinds, dtype=np.int64),
        won=np.full(n, won, dtype=bool),
        seed=np.full(n, seed, dtype=np.int64),
        final_ante=np.full(n, env.run.ante, dtype=np.int64),
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--deck", default="d_red")
    p.add_argument("--stake", default="white")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    parts: list[dict] = []
    for seed in range(args.start, args.start + args.seeds):
        t = time.time()
        rec = collect(seed, args.deck, args.stake)
        parts.append(rec)
        print(f"seed {seed}: {len(rec['action'])} decisions, won={bool(rec['won'][0])}, "
              f"ante {int(rec['final_ante'][0])}, {time.time() - t:.0f}s", flush=True)
    merged = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    np.savez_compressed(args.out, **merged)
    print(f"saved {len(merged['action'])} records to {args.out}")


if __name__ == "__main__":
    main()
