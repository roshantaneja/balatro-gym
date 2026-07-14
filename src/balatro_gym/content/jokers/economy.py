"""Economy jokers from the v1 set."""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Rarity


class _GoldenJoker(JokerEffect):
    """Earn $4 at end of round."""

    def round_end_money(self, run, j):
        return 4


# Wiki compat-copyable = 0: Blueprint does not copy Golden Joker.
register_joker(
    JokerDef("j_golden", "Golden Joker", Rarity.COMMON, 6, _GoldenJoker(), copyable=False)
)


class _Egg(JokerEffect):
    """Gains $3 of sell value at end of round."""

    def update_on_round_end(self, run, j):
        j.sell_value += 3


# Wiki compat-copyable = 0.
register_joker(JokerDef("j_egg", "Egg", Rarity.COMMON, 4, _Egg(), copyable=False))


class _BusinessCard(JokerEffect):
    """Scored face cards have a 1 in 2 chance to give $2 (Pareidolia-aware)."""

    def on_card_scored(self, ctx, j, card):
        if ctx.is_face(card) and ctx.chance(1, 2):
            ctx.add_money(2, "joker", "j_business")


register_joker(JokerDef("j_business", "Business Card", Rarity.COMMON, 4, _BusinessCard()))
