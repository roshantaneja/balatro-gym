"""Train MaskablePPO on a Balatro env.

Requires the train extra:  uv pip install -e ".[train]"

Usage:
  python examples/train_maskable_ppo.py --env balatro_gym/BalatroBlind-v0 --steps 200000
  python examples/train_maskable_ppo.py --env balatro_gym/BalatroScriptedShop-v0
"""

from __future__ import annotations

import argparse

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
    args = parser.parse_args()

    env = gym.make(args.env)

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        seed=args.seed,
        verbose=1,
        tensorboard_log="./tb",
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
    from balatro_gym.env.baselines import HeuristicAgent, RandomLegalAgent
    from balatro_gym.env.eval import evaluate

    raw = env.unwrapped
    print("random baseline:   ", evaluate(raw, RandomLegalAgent(), seeds=20).summary())
    print("heuristic baseline:", evaluate(raw, HeuristicAgent(), seeds=20).summary())


if __name__ == "__main__":
    main()
