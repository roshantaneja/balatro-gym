"""Evaluation harness: fixed seed set, win rate / ante-reached metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvalResult:
    episodes: int = 0
    wins: int = 0
    antes: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    money_final: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.episodes, 1)

    @property
    def mean_ante(self) -> float:
        return float(np.mean(self.antes)) if self.antes else 0.0

    def summary(self) -> str:
        return (
            f"episodes={self.episodes} win_rate={self.win_rate:.3f} "
            f"mean_ante={self.mean_ante:.2f} max_ante={max(self.antes, default=0)} "
            f"mean_steps={np.mean(self.steps):.0f} mean_money={np.mean(self.money_final):.1f}"
        )


def evaluate(env, agent, seeds: list[int] | int = 100, max_steps: int = 3000) -> EvalResult:
    """agent: object with .act(env, obs, mask) -> int (see baselines), or a
    callable(obs, mask) -> int (trained-policy adapter)."""
    if isinstance(seeds, int):
        seeds = list(range(seeds))
    act = agent.act if hasattr(agent, "act") else (lambda e, o, m: agent(o, m))

    result = EvalResult()
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        for step in range(max_steps):
            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            if len(legal) == 0:
                break
            obs, r, terminated, truncated, info = env.step(int(act(env, obs, mask)))
            if terminated or truncated:
                break
        stats = info.get("episode_stats", {})
        result.episodes += 1
        result.wins += 1 if stats.get("win") else 0
        result.antes.append(stats.get("ante_reached", env.run.ante))
        result.steps.append(step + 1)
        result.money_final.append(env.run.money)
    return result
