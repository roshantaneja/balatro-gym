"""Check that an agent plays complete games under every initial condition.

    python scripts/validate_agent.py --agent strategist --seeds 3 --workers 4
    python scripts/validate_agent.py --agent planner --decks d_red --seeds 2
    python scripts/validate_agent.py --agent planner --stakes white --seeds 2

Every registered deck and stake (or the subsets given) is played for the
given seeds. A run counts as complete when it ends in WON or GAME_OVER
without an exception and within the step cap, with every action legal.
The report lists completion, wins and mean ante per deck and per stake,
and any failure with its traceback, so a gap in the engine or the agent
under some deck, stake or boss shows up by name.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool


def run_one(args: tuple) -> dict:
    name, deck, stake, seed, samples = args
    from balatro_gym.content import load_all
    from balatro_gym.engine import Phase
    from balatro_gym.env.envs import BalatroEnv

    load_all()
    out = {"deck": deck, "stake": stake, "seed": seed, "complete": False, "won": False, "ante": 0}
    t = time.time()
    try:
        if name == "planner":
            from balatro_gym.env.planner import PlannerAgent

            agent = PlannerAgent(samples=samples)
        elif name == "strategist":
            from balatro_gym.env.baselines import StrategistAgent

            agent = StrategistAgent(seed=seed)
        elif name == "heuristic":
            from balatro_gym.env.baselines import HeuristicAgent

            agent = HeuristicAgent()
        elif name.endswith(".pt"):
            from balatro_gym.typed.agent import TypedModelAgent

            agent = TypedModelAgent(name, seed=seed)
        else:
            raise SystemExit(f"unknown agent {name!r}")
        env = BalatroEnv(deck=deck, stake=stake)
        obs, info = env.reset(seed=seed)
        done, steps = False, 0
        while not done and steps < 4000:
            mask = info["action_mask"]
            a = int(agent.act(env, obs, mask))
            if not mask[a]:
                raise RuntimeError(f"illegal action {a} in phase {env._env_phase()}")
            obs, _r, term, _trunc, info = env.step(a)
            done, steps = term, steps + 1
        out["ante"] = env.run.ante
        out["won"] = env.run.phase == Phase.WON
        out["complete"] = done and env.run.phase in (Phase.WON, Phase.GAME_OVER)
        if not done:
            out["error"] = f"step cap reached in phase {env._env_phase()} ante {env.run.ante}"
        out["steps"] = steps
    except Exception:  # noqa: BLE001 - every failure is a finding
        out["error"] = traceback.format_exc()
    out["seconds"] = round(time.time() - t, 1)
    return out


def main() -> None:
    from balatro_gym.content import load_all
    from balatro_gym.engine.effects import DECKS, STAKES

    load_all()
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="strategist", help="strategist | heuristic | planner | <typed .pt>")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--decks", default="all", help="comma-separated deck ids or 'all'")
    p.add_argument("--stakes", default="all", help="comma-separated stake names or 'all'")
    p.add_argument("--samples", type=int, default=40, help="planner rollouts")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--json", default=None, help="write every run's record here")
    args = p.parse_args()
    decks = list(DECKS) if args.decks == "all" else args.decks.split(",")
    stakes = [s.value if hasattr(s, "value") else str(s) for s in STAKES] if args.stakes == "all" else args.stakes.split(",")
    jobs = [(args.agent, d, s, seed, args.samples)
            for d in decks for s in stakes for seed in range(args.start, args.start + args.seeds)]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(run_one, jobs))
    else:
        results = [run_one(j) for j in jobs]
    results.sort(key=lambda r: (r["deck"], r["stake"], r["seed"]))
    by_deck: dict[str, list] = defaultdict(list)
    by_stake: dict[str, list] = defaultdict(list)
    for r in results:
        by_deck[r["deck"]].append(r)
        by_stake[r["stake"]].append(r)

    def line(name: str, rs: list) -> str:
        n = len(rs)
        return (f"{name:14s} complete {sum(r['complete'] for r in rs):3d}/{n:<3d} "
                f"wins {sum(r['won'] for r in rs):3d} mean ante {sum(r['ante'] for r in rs) / n:.2f}")

    print(f"agent {args.agent}: {len(results)} runs over {len(decks)} decks x {len(stakes)} stakes x {args.seeds} seeds")
    for d in decks:
        print(line(d, by_deck[d]))
    print()
    for s in stakes:
        print(line(s, by_stake[s]))
    failures = [r for r in results if not r["complete"]]
    print(f"\ncomplete {len(results) - len(failures)}/{len(results)}, "
          f"wins {sum(r['won'] for r in results)}, failures {len(failures)}")
    for r in failures:
        print(f"--- {r['deck']} {r['stake']} seed {r['seed']}:\n{r.get('error', 'incomplete')}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
