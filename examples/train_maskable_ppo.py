"""Train MaskablePPO on a Balatro env.

Requires the train extra:  uv pip install -e ".[train]"

Usage:
  python examples/train_maskable_ppo.py --env balatro_gym/BalatroBlind-v0 --steps 200000
  python examples/train_maskable_ppo.py --env balatro_gym/BalatroScriptedShop-v0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without an editable install (e.g. system python).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.evaluation import evaluate_policy
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        'sb3-contrib not installed. Run: uv pip install -e ".[train]"'
    ) from exc

import gymnasium as gym

import balatro_gym.env  # noqa: F401  (registers envs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="balatro_gym/BalatroBlind-v0")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="ppo_balatro")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--deck", default="d_red", help="deck id (full/scripted envs)")
    parser.add_argument("--stake", default="white", help="stake name (full/scripted envs)")
    args = parser.parse_args()

    env_kwargs = {} if "Blind" in args.env else {"deck": args.deck, "stake": args.stake}
    env = gym.make(args.env, **env_kwargs)

    try:
        import tensorboard  # noqa: F401

        tb_log = "./tb"
    except ImportError:
        tb_log = None
        print("tensorboard not installed - training without TB logging "
              '(uv pip install tensorboard to enable)')

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tb_log,
        n_steps=1024,
        batch_size=256,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(args.out)
    print(f"saved model to {args.out}.zip")

    mean_r, std_r = evaluate_policy(
        model, env, n_eval_episodes=args.eval_episodes, warn=False
    )
    print(f"eval over {args.eval_episodes} episodes: reward {mean_r:.2f} +/- {std_r:.2f}")

    # Compare against baselines on the same env.
    from balatro_gym.env.baselines import HeuristicAgent, RandomLegalAgent, StrategistAgent
    from balatro_gym.env.eval import evaluate

    raw = env.unwrapped
    print("random baseline:   ", evaluate(raw, RandomLegalAgent(), seeds=20).summary())
    print("heuristic baseline:", evaluate(raw, HeuristicAgent(), seeds=20).summary())
    print("strategist baseline:", evaluate(raw, StrategistAgent(), seeds=20).summary())
    from balatro_gym.env.planner import PlannerAgent

    print("planner baseline:   ", evaluate(raw, PlannerAgent(), seeds=20).summary())


if __name__ == "__main__":
    main()
