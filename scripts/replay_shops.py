"""Measure the headroom in the planner's shop decisions.

    python scripts/replay_shops.py --seeds 24 --workers 4

Each seed is one planner run; the engine state at the entry of every shop
is snapshotted. For every lost run, every shop from ante 2 on is replayed
from its snapshot in two ways: (a) the same planner again (a control that
measures how much of the outcome is rollout variance), and (b) one forced
first action — buy each affordable joker on offer, reroll once, or leave
at once — after which the planner plays the rest of the run. A lost run
with a winning forced alternative was losable to a shop decision; the gap
between (b) and (a) is what a better shop policy could recover.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

from balatro_gym.engine import Phase
from balatro_gym.env import action_codec as codec
from balatro_gym.env.envs import BalatroEnv
from balatro_gym.env.planner import PlannerAgent


def restore(env: BalatroEnv, clone):
    env.game.run = clone.clone()
    env._mask_cache = None
    env._pending = None
    env._refresh_hand_order()
    return env._obs(), env._info()


def finish(env: BalatroEnv, agent, obs, info, steps: int = 0) -> tuple[bool, int]:
    done = env.run.phase in (Phase.WON, Phase.GAME_OVER)
    while not done and steps < 4000:
        a = agent.act(env, obs, info["action_mask"])
        obs, _r, done, _trunc, info = env.step(a)
        steps += 1
    return env.run.phase == Phase.WON, env.run.ante


def run_seed(args: tuple) -> dict:
    seed, min_ante, controls = args
    env = BalatroEnv()
    agent = PlannerAgent(seed=seed)
    obs, info = env.reset(seed=seed)
    snaps = []  # (ante, blind_index, run clone) at every shop entry
    prev, steps, done = None, 0, False
    while not done and steps < 4000:
        phase = env._env_phase()
        if phase == "shop" and prev != "shop":
            snaps.append((env.run.ante, env.run.blind_index, env.run.clone()))
        prev = phase
        a = agent.act(env, obs, info["action_mask"])
        obs, _r, done, _trunc, info = env.step(a)
        steps += 1
    out = {"seed": seed, "won": env.run.phase == Phase.WON, "ante": env.run.ante, "shops": []}
    if out["won"]:
        return out
    for k, (ante, blind, clone) in enumerate(snaps):
        if ante < min_ante:
            continue
        rec = {"ante": ante, "blind": blind, "control": [], "alternatives": {}}
        for c in range(controls):
            obs, info = restore(env, clone)
            rec["control"].append(finish(env, PlannerAgent(seed=1000 * (c + 1) + seed), obs, info)[0])
        obs, info = restore(env, clone)
        mask = info["action_mask"]
        forced: dict[str, int] = {}
        shop = env.run.shop
        if shop is not None:
            for i, it in enumerate(shop.card_slots[: codec.N_BUY_CARD_SLOTS]):
                idx = codec.BUY_CARD_BASE + i
                if mask[idx] and it.kind == "joker":
                    forced[f"buy {it.item_id}"] = idx
        if mask[codec.REROLL_IDX]:
            forced["reroll"] = codec.REROLL_IDX
        if mask[codec.LEAVE_SHOP_IDX]:
            forced["leave"] = codec.LEAVE_SHOP_IDX
        for name, idx in forced.items():
            obs, info = restore(env, clone)
            obs, _r, _term, _trunc, info = env.step(idx)
            won, _ante = finish(env, PlannerAgent(seed=7000 + seed + k), obs, info, steps=1)
            rec["alternatives"][name] = won
        out["shops"].append(rec)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--min-ante", type=int, default=2)
    p.add_argument("--controls", type=int, default=1, help="plain re-runs per shop snapshot")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    jobs = [(s, args.min_ante, args.controls) for s in range(args.start, args.start + args.seeds)]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(run_seed, jobs))
    else:
        results = [run_seed(j) for j in jobs]
    results.sort(key=lambda r: r["seed"])
    lost = [r for r in results if not r["won"]]
    ctrl_n = ctrl_w = alt_n = alt_w = 0
    recoverable = recoverable_ctrl = 0
    by_alt: dict[str, list] = {}
    for r in lost:
        any_ctrl = any_alt = False
        for s in r["shops"]:
            ctrl_n += len(s["control"])
            ctrl_w += sum(s["control"])
            any_ctrl |= any(s["control"])
            for name, won in s["alternatives"].items():
                alt_n += 1
                alt_w += won
                any_alt |= won
                key = name.split(" ")[0]
                by_alt.setdefault(key, []).append(won)
        recoverable_ctrl += any_ctrl
        recoverable += any_alt
        print(f"seed {r['seed']}: lost at ante {r['ante']}; shops replayed {len(r['shops'])}; "
              f"control wins {sum(sum(s['control']) for s in r['shops'])}; "
              f"alternative wins {sum(sum(s['alternatives'].values()) for s in r['shops'])}"
              f"{' <- recoverable' if any_alt and not any_ctrl else ''}")
    print(json.dumps({
        "runs": len(results), "lost": len(lost),
        "control_win_rate_from_shop": round(ctrl_w / max(ctrl_n, 1), 3),
        "forced_alternative_win_rate": round(alt_w / max(alt_n, 1), 3),
        "by_alternative": {k: round(sum(v) / len(v), 3) for k, v in by_alt.items()},
        "lost_runs_with_a_winning_control_replay": recoverable_ctrl,
        "lost_runs_with_a_winning_alternative": recoverable,
    }, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
