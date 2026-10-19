"""Evaluate a typed decision model as a player and as a forecaster.

    python scripts/typed/eval.py --model models/typed_v1.pt --seeds 60 --start 1000 --workers 4

Plays full games with `TypedModelAgent` (argmax of the calibrated action
distribution) and reports win rate and mean ante next to the planner's
numbers, plus the calibration of its `will_win` answers: the win probability
it states at the start of every ante against what happened to the run.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np


def run_seed(args: tuple) -> dict:
    model, seed, deck, stake, sample = args
    from balatro_gym.engine import Phase
    from balatro_gym.env.envs import BalatroEnv
    from balatro_gym.typed.agent import TypedModelAgent

    agent = TypedModelAgent(model, sample=sample, seed=seed)
    env = BalatroEnv(deck=deck, stake=stake)
    obs, info = env.reset(seed=seed)
    forecasts: dict[int, float] = {}
    done, steps = False, 0
    while not done and steps < 4000:
        if env.run.phase.value == "blind_select" and env.run.ante not in forecasts:
            answers = agent.ask(env, obs, info["action_mask"])
            a = answers["will_win"]
            forecasts[env.run.ante] = a.confidence if a.value else 1.0 - a.confidence
        action = agent.act(env, obs, info["action_mask"])
        obs, _r, term, _trunc, info = env.step(action)
        done, steps = term, steps + 1
    return {"seed": seed, "won": env.run.phase == Phase.WON, "ante": env.run.ante,
            "forecasts": forecasts, "steps": steps}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", type=int, default=60)
    p.add_argument("--start", type=int, default=1000, help="first seed (keep clear of the training seeds)")
    p.add_argument("--deck", default="d_red")
    p.add_argument("--stake", default="white")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    args = p.parse_args()
    jobs = [(args.model, s, args.deck, args.stake, args.sample) for s in range(args.start, args.start + args.seeds)]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(run_seed, jobs))
    else:
        results = [run_seed(j) for j in jobs]
    results.sort(key=lambda r: r["seed"])
    for r in results:
        print(f"seed {r['seed']}: {'WON' if r['won'] else 'ante %d' % r['ante']} "
              f"P(win)@ante {[round(r['forecasts'].get(a, float('nan')), 2) for a in range(1, 9) if a in r['forecasts']]}")
    wins = sum(r["won"] for r in results)
    # Forecast calibration: stated P(win) at each ante start vs the outcome.
    by_ante: dict[int, list] = {}
    for r in results:
        for ante, pw in r["forecasts"].items():
            by_ante.setdefault(ante, []).append((pw, float(r["won"])))
    calib = {a: {"mean_forecast": round(float(np.mean([x[0] for x in v])), 3),
                 "win_rate": round(float(np.mean([x[1] for x in v])), 3), "n": len(v)}
             for a, v in sorted(by_ante.items())}
    print(json.dumps({"episodes": len(results), "win_rate": wins / len(results),
                      "mean_ante": sum(r["ante"] for r in results) / len(results),
                      "forecast_calibration_by_ante": calib}, indent=1))


if __name__ == "__main__":
    main()
