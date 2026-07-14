"""On-scored (stage 3.3) jokers from the v1 set."""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Rank, Rarity, Suit

from .helpers import suit_mult_joker

# The four suit jokers: +3 Mult per scored card of the suit.
suit_mult_joker("j_greedy", "Greedy Joker", Rarity.COMMON, 5, Suit.DIAMONDS, 3)
suit_mult_joker("j_lusty", "Lusty Joker", Rarity.COMMON, 5, Suit.HEARTS, 3)
suit_mult_joker("j_wrathful", "Wrathful Joker", Rarity.COMMON, 5, Suit.SPADES, 3)
suit_mult_joker("j_gluttonous", "Gluttonous Joker", Rarity.COMMON, 5, Suit.CLUBS, 3)


class _OddTodd(JokerEffect):
    """+31 Chips per scored card with an odd rank (A, 3, 5, 7, 9)."""

    ODD = {Rank.ACE, Rank.THREE, Rank.FIVE, Rank.SEVEN, Rank.NINE}

    def on_card_scored(self, ctx, j, card):
        if not card.is_stone and card.rank in self.ODD:
            ctx.add_chips(31, "joker", "j_odd_todd")


register_joker(JokerDef("j_odd_todd", "Odd Todd", Rarity.COMMON, 4, _OddTodd()))


class _Scholar(JokerEffect):
    """Scored Aces give +20 Chips and +4 Mult."""

    def on_card_scored(self, ctx, j, card):
        if not card.is_stone and card.rank == Rank.ACE:
            ctx.add_chips(20, "joker", "j_scholar")
            ctx.add_mult(4, "joker", "j_scholar")


register_joker(JokerDef("j_scholar", "Scholar", Rarity.COMMON, 4, _Scholar()))


class _Photograph(JokerEffect):
    """First scored face card gives x2 Mult."""

    def on_card_scored(self, ctx, j, card):
        faces = [c for c in ctx.hand_result.scoring_cards if ctx.is_face(c)]
        if faces and card.uid == faces[0].uid:
            ctx.mul_mult(2.0, "joker", "j_photograph")


register_joker(JokerDef("j_photograph", "Photograph", Rarity.COMMON, 5, _Photograph()))


class _Bloodstone(JokerEffect):
    """1 in 2 chance for scored Hearts to give x1.5 Mult."""

    def on_card_scored(self, ctx, j, card):
        if ctx.card_has_suit(card, Suit.HEARTS) and ctx.chance(1, 2):
            ctx.mul_mult(1.5, "joker", "j_bloodstone")


register_joker(JokerDef("j_bloodstone", "Bloodstone", Rarity.UNCOMMON, 7, _Bloodstone()))
