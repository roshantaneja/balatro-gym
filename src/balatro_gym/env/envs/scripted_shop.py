"""BalatroScriptedShopEnv: full runs where non-play phases are auto-piloted.

The agent only sees PLAYING (and TARGET) decisions; configured controllers
handle BLIND_SELECT / SHOP / PACK_OPEN. Rewards accumulate across auto-steps.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from balatro_gym.engine import Phase

from .. import action_codec as codec
from .full import BalatroEnv

Controller = Callable[["BalatroScriptedShopEnv", np.ndarray], int]


def always_play_blind(env: "BalatroScriptedShopEnv", mask: np.ndarray) -> int:
    return codec.SELECT_BLIND


def greedy_shop(env: "BalatroScriptedShopEnv", mask: np.ndarray) -> int:
    """Buy affordable jokers cheapest-first, then buffoon packs, then leave."""
    run = env.run
    shop = run.shop
    if shop is not None:
        buyable = [
            (item.cost, i)
            for i, item in enumerate(shop.card_slots[:codec.N_BUY_CARD_SLOTS])
            if item.kind == "joker" and mask[codec.BUY_CARD_BASE + i]
        ]
        if buyable:
            _, slot = min(buyable)
            return codec.BUY_CARD_BASE + slot
        for i, offer in enumerate(shop.pack_slots[:codec.N_BUY_PACK_SLOTS]):
            if "buffoon" in offer.pack_id and mask[codec.BUY_PACK_BASE + i]:
                return codec.BUY_PACK_BASE + i
    return codec.LEAVE_SHOP_IDX


def greedy_pack(env: "BalatroScriptedShopEnv", mask: np.ndarray) -> int:
    for i in range(codec.N_PACK_OPTIONS):
        if mask[codec.PICK_PACK_BASE + i]:
            return codec.PICK_PACK_BASE + i
    return codec.SKIP_PACK_IDX


def strategist_shop(env: "BalatroScriptedShopEnv", mask: np.ndarray) -> int:
    """Community-strategy shop (docs/strategy.md): tiered scaling-joker buys
    under an interest reserve, decay selling, targeted rerolls."""
    return _strategist()._shop_phase(env, mask)


def strategist_pack(env: "BalatroScriptedShopEnv", mask: np.ndarray) -> int:
    return _strategist()._pack_phase(env, mask)


_STRATEGIST = None


def _strategist():
    global _STRATEGIST
    if _STRATEGIST is None:
        from ..baselines import StrategistAgent

        _STRATEGIST = StrategistAgent()
    return _STRATEGIST


DEFAULT_CONTROLLERS: dict[str, Controller] = {
    "blind_select": always_play_blind,
    "shop": strategist_shop,
    "pack_open": strategist_pack,
}

# The original naive controllers remain available:
GREEDY_CONTROLLERS: dict[str, Controller] = {
    "blind_select": always_play_blind,
    "shop": greedy_shop,
    "pack_open": greedy_pack,
}


class BalatroScriptedShopEnv(BalatroEnv):
    def __init__(self, controllers: dict[str, Controller] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.controllers = dict(DEFAULT_CONTROLLERS)
        if controllers:
            self.controllers.update(controllers)

    def _auto_advance(self) -> tuple[float, list, bool]:
        """Run controllers until an agent-owned phase or termination."""
        total_reward = 0.0
        all_events = []
        guard = 0
        while self.run.phase not in (Phase.WON, Phase.GAME_OVER):
            phase = self._env_phase()
            controller = self.controllers.get(phase)
            if controller is None:
                return total_reward, all_events, False
            mask = self.action_masks()
            action = controller(self, mask)
            if not mask[action]:  # controller bug fallback: first legal action
                legal = np.flatnonzero(mask)
                if len(legal) == 0:
                    break
                action = int(legal[0])
            obs, r, term, trunc, info = BalatroEnv.step(self, action)
            total_reward += r
            all_events.extend(info.get("events", []))
            guard += 1
            if guard > 500:
                raise RuntimeError("scripted controller loop did not converge")
        return total_reward, all_events, True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        r, events, _ = self._auto_advance()
        # Reward earned during auto-advance on reset is unattributable; it is
        # negligible in practice (blind-select only) and dropped by design.
        obs = self._obs()
        info = self._info()
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        if not terminated:
            extra, events, done = self._auto_advance()
            reward += extra
            info.setdefault("events", []).extend(events)
            terminated = self.run.phase in (Phase.WON, Phase.GAME_OVER)
            obs = self._obs()
            mask_info = self._info()
            info.update(mask_info)
        return obs, reward, terminated, truncated, info
