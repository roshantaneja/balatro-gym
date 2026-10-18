"""Legendary jokers (only obtainable from The Soul). Chicot and Perkeo live
with the board and economy jokers respectively.
"""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Rank, Rarity


class _Canio(JokerEffect):
    """Gains x1 Mult per face card destroyed."""

    def on_independent(self, ctx, j):
        ctx.mul_mult(j.state["x"], "joker", "j_canio")

    def update_on_card_destroyed(self, run, j, card):
        from balatro_gym.engine.hand_eval import is_face
        from balatro_gym.engine.scoring import fold_eval_rules

        if is_face(card, fold_eval_rules(run)):
            j.state["x"] += 1


register_joker(
    JokerDef("j_canio", "Canio", Rarity.LEGENDARY, 20, _Canio(), init_state={"x": 1.0})
)


class _Triboulet(JokerEffect):
    """Scored Kings and Queens each give x2 Mult."""

    def on_card_scored(self, ctx, j, card):
        if not card.is_stone and card.rank in (Rank.KING, Rank.QUEEN):
            ctx.mul_mult(2.0, "joker", "j_triboulet")


register_joker(JokerDef("j_triboulet", "Triboulet", Rarity.LEGENDARY, 20, _Triboulet()))


class _Yorick(JokerEffect):
    """Gains x1 Mult every 23 cards discarded."""

    def on_independent(self, ctx, j):
        ctx.mul_mult(j.state["x"], "joker", "j_yorick")

    def update_on_discard(self, run, j, cards):
        j.state["discards"] += len(cards)
        while j.state["discards"] >= 23:
            j.state["discards"] -= 23
            j.state["x"] += 1


register_joker(
    JokerDef("j_yorick", "Yorick", Rarity.LEGENDARY, 20, _Yorick(),
             init_state={"x": 1.0, "discards": 0})
)
