"""BalatroBlindEnv: single-blind curriculum env (trainable at M1).

Identical action/observation spaces to the full env so policy weights
transfer; only the initial-state factory and termination differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from balatro_gym.engine import BlindCleared, BlindKind, HandType, Phase, new_blind_scenario

from ..rewards import RewardConfig
from .full import BalatroEnv


@dataclass(frozen=True)
class ScenarioConfig:
    ante: int = 1
    blind_kind: BlindKind = BlindKind.SMALL
    deck: str = "d_red"
    joker_ids: tuple[str, ...] = ()
    hand_levels: dict = field(default_factory=dict)
    hands: int | None = None
    discards: int | None = None
    money: int = 4


def default_scenario_sampler(np_random) -> ScenarioConfig:
    """Randomized curriculum: early antes, occasional jokers/levels."""
    from balatro_gym.engine import JOKERS

    ante = int(np_random.integers(1, 4))
    jokers: tuple[str, ...] = ()
    pool = sorted(JOKERS)
    if pool and np_random.random() < 0.7:
        k = int(np_random.integers(1, 4))
        jokers = tuple(np_random.choice(pool, size=min(k, len(pool)), replace=False))
    levels: dict[HandType, int] = {}
    if np_random.random() < 0.5:
        ht = HandType(int(np_random.integers(1, 10)))
        levels[ht] = int(np_random.integers(2, 5))
    kind = BlindKind(int(np_random.integers(0, 2)))  # small/big (bosses via full env)
    return ScenarioConfig(ante=ante, blind_kind=kind, joker_ids=jokers, hand_levels=levels)


class BalatroBlindEnv(BalatroEnv):
    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        scenario_sampler: Callable | None = None,
        reward_config: RewardConfig | None = None,
        reward_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(
            reward_config=reward_config or RewardConfig(mode=reward_mode or "shaped"),
            enable_skip_blind=False,
            **kwargs,
        )
        self.scenario = scenario
        self.scenario_sampler = scenario_sampler

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        import gymnasium as gym

        gym.Env.reset(self, seed=seed)
        cfg = self.scenario
        if cfg is None:
            sampler = self.scenario_sampler or default_scenario_sampler
            cfg = sampler(self.np_random)
        seed_str = f"blind-{self.np_random.integers(0, 2**63)}"
        self.game = new_blind_scenario(
            seed_str,
            ante=cfg.ante,
            blind_kind=cfg.blind_kind,
            deck_id=cfg.deck,
            joker_ids=list(cfg.joker_ids),
            hand_levels=cfg.hand_levels,
            hands=cfg.hands,
            discards=cfg.discards,
            money=cfg.money,
        )
        self._pending = None
        self._episode_steps = 0
        self._mask_cache = None
        self._refresh_hand_order()
        return self._obs(), self._info()

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        # Episode ends when the blind resolves either way.
        events = info.get("events", [])
        cleared = any(isinstance(e, BlindCleared) for e in events)
        if cleared:
            terminated = True
            info["episode_stats"] = self._episode_stats(won=True)
        elif self.run.phase == Phase.GAME_OVER:
            terminated = True
        return obs, reward, terminated, truncated, info
