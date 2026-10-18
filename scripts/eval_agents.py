"""Reproduce the baseline table: full-game win rate and mean ante per agent.

    python scripts/eval_agents.py --agent planner --seeds 60 --workers 4
    python scripts/eval_agents.py --agent strategist --seeds 100 --deck d_plasma --stake red

Each seed is one full run from ante 1. Per-seed lines list the outcome, the
money held at the start of every ante, the final board, hand levels, vouchers
and (for losses) the chips scored against the requirement of the lost blind.
The planner takes ~60 s per run on one core, the lookahead planner ~8 min; the others take a second.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool


def make_agent(name: str, samples: int, top_k: int):
    from balatro_gym.env.baselines import HeuristicAgent, RandomLegalAgent, StrategistAgent
    from balatro_gym.env.planner import PlannerAgent

    if name == "random":
        return RandomLegalAgent()
    if name == "heuristic":
        return HeuristicAgent()
    if name == "strategist":
        return StrategistAgent()
    if name == "planner":
        return PlannerAgent(samples=samples, top_k=top_k)
    if name == "planner-adaptive":
        return PlannerAgent(samples=samples, top_k=top_k, adaptive=True)
    if name == "lookahead":
        from balatro_gym.env.lookahead import LookaheadPlannerAgent

        return LookaheadPlannerAgent(samples=samples, top_k=top_k, samples_ahead=3)
    raise SystemExit(f"unknown agent {name!r}")


def run_seed(args: tuple) -> dict:
    name, seed, deck, stake, samples, top_k = args
    from balatro_gym.engine import RunLost
    from balatro_gym.env.envs import BalatroEnv

    agent = make_agent(name, samples, top_k)
    env = BalatroEnv(deck=deck, stake=stake)
    t = time.time()
    obs, info = env.reset(seed=seed)
    done, steps, loss, loss_boss = False, 0, None, None
    money_trace: dict[int, int] = {}
    while not done and steps < 4000:
        if env.run.phase.value == "blind_select":
            money_trace.setdefault(env.run.ante, env.run.money)
        a = agent.act(env, obs, info["action_mask"])
        obs, _r, term, _trunc, info = env.step(a)
        done, steps = term, steps + 1
        for e in info.get("events", []):
            if isinstance(e, RunLost):
                loss, loss_boss = e, (env.run.round.boss_id if env.run.round else None)
    run = env.run
    return {
        "seed": seed,
        "won": run.phase.value == "won",
        "ante": run.ante,
        "blind": run.blind_index,
        "money": list(money_trace.values()),
        "jokers": [j.id[2:] for j in run.jokers],
        "levels": {ht.name: lvl for ht, lvl in run.hand_levels.items() if lvl > 1},
        "vouchers": sorted(v[2:] for v in run.vouchers),
        "seconds": round(time.time() - t, 1),
        "lost": None if loss is None else (loss.chips_scored, loss.chips_needed, loss_boss),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="planner",
                   choices=["random", "heuristic", "strategist", "planner", "planner-adaptive", "lookahead"])
    p.add_argument("--seeds", type=int, default=60, help="number of seeds (0..N-1)")
    p.add_argument("--start", type=int, default=0, help="first seed")
    p.add_argument("--deck", default="d_red")
    p.add_argument("--stake", default="white")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--samples", type=int, default=5, help="planner: hands sampled per valuation")
    p.add_argument("--top-k", type=int, default=10, help="planner: plays scored per sampled hand")
    p.add_argument("--json", default=None, help="write per-seed records to this file")
    args = p.parse_args()

    jobs = [(args.agent, s, args.deck, args.stake, args.samples, args.top_k)
            for s in range(args.start, args.start + args.seeds)]
    results = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for rec in pool.imap_unordered(run_seed, jobs):
                results.append(rec)
                print(_fmt(rec), flush=True)
    else:
        for job in jobs:
            rec = run_seed(job)
            results.append(rec)
            print(_fmt(rec), flush=True)
    results.sort(key=lambda r: r["seed"])
    wins = sum(r["won"] for r in results)
    antes = [r["ante"] for r in results]
    summary = {
        "agent": args.agent, "deck": args.deck, "stake": args.stake, "episodes": len(results),
        "win_rate": wins / max(len(results), 1), "mean_ante": sum(antes) / max(len(antes), 1),
        "ante_hist": dict(sorted(Counter(antes).items())),
        "mean_seconds": sum(r["seconds"] for r in results) / max(len(results), 1),
    }
    print(json.dumps(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "runs": results}, f, indent=1)


def _fmt(r: dict) -> str:
    outcome = "WON" if r["won"] else f"ante {r['ante']}.{r['blind']}"
    line = (f"seed {r['seed']}: {outcome} money@ante {r['money']} {r['jokers']} "
            f"lv {r['levels']} v={r['vouchers']} {r['seconds']}s")
    if r["lost"]:
        scored, needed, boss = r["lost"]
        line += f" lost {scored:.0f}/{needed:.0f} boss={boss}"
    return line


if __name__ == "__main__":
    main()
