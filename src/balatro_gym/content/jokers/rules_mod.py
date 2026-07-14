"""Passive rule-modification jokers. All are copyable=False (wiki compat)."""

from __future__ import annotations

from dataclasses import replace

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Rarity


class _FourFingers(JokerEffect):
    """All Flushes and Straights can be made with 4 cards."""

    def modify_eval_rules(self, rules, j):
        return replace(rules, flush_size=4, straight_size=4)


register_joker(
    JokerDef("j_four_fingers", "Four Fingers", Rarity.UNCOMMON, 7, _FourFingers(), copyable=False)
)


class _Splash(JokerEffect):
    """Every played card counts in scoring."""

    def modify_eval_rules(self, rules, j):
        return replace(rules, splash=True)


register_joker(JokerDef("j_splash", "Splash", Rarity.COMMON, 3, _Splash(), copyable=False))


class _Pareidolia(JokerEffect):
    """All cards are considered face cards."""

    def modify_eval_rules(self, rules, j):
        return replace(rules, all_faces=True)


register_joker(
    JokerDef("j_pareidolia", "Pareidolia", Rarity.UNCOMMON, 5, _Pareidolia(), copyable=False)
)
