"""On-held (stage 4) jokers from the v1 set."""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Rank, Rarity


class _Baron(JokerEffect):
    """Each King held in hand gives x1.5 Mult."""

    def on_card_held(self, ctx, j, card):
        if not card.is_stone and card.rank == Rank.KING:
            ctx.mul_mult(1.5, "joker", "j_baron")


register_joker(JokerDef("j_baron", "Baron", Rarity.RARE, 8, _Baron()))


class _ShootTheMoon(JokerEffect):
    """Each Queen held in hand gives +13 Mult."""

    def on_card_held(self, ctx, j, card):
        if not card.is_stone and card.rank == Rank.QUEEN:
            ctx.add_mult(13, "joker", "j_shoot_the_moon")


register_joker(
    JokerDef("j_shoot_the_moon", "Shoot the Moon", Rarity.COMMON, 5, _ShootTheMoon())
)


class _RaisedFist(JokerEffect):
    """Adds double the rank chip value of the lowest-ranked card held in hand
    to Mult. Fires when that lowest card's held trigger runs."""

    def on_card_held(self, ctx, j, card):
        held = [c for c in ctx.held if not c.is_stone]
        if not held:
            return
        lowest = min(held, key=lambda c: int(c.rank))
        # Fire on the lowest card only (last occurrence if tied, matching game).
        lowest_ties = [c for c in held if int(c.rank) == int(lowest.rank)]
        if card.uid == lowest_ties[-1].uid:
            ctx.add_mult(2 * card.rank.chip_value, "joker", "j_raised_fist")


register_joker(JokerDef("j_raised_fist", "Raised Fist", Rarity.COMMON, 5, _RaisedFist()))
