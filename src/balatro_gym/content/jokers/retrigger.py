"""Retrigger jokers from the v1 set."""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Rank, Rarity


class _Dusk(JokerEffect):
    """Retrigger all scored cards on the FINAL hand of the round.

    round.play_hand decrements hands_left AFTER score_play runs, so during
    scoring of the final hand hands_left == 1."""

    def retrigger_scored(self, ctx, j, card):
        rnd = ctx.run.round
        if rnd is not None and rnd.hands_left == 1:
            return 1
        return 0


register_joker(JokerDef("j_dusk", "Dusk", Rarity.UNCOMMON, 5, _Dusk()))


class _Hack(JokerEffect):
    """Retrigger each scored 2, 3, 4, or 5."""

    LOW = {Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE}

    def retrigger_scored(self, ctx, j, card):
        if not card.is_stone and card.rank in self.LOW:
            return 1
        return 0


register_joker(JokerDef("j_hack", "Hack", Rarity.UNCOMMON, 6, _Hack()))


class _Mime(JokerEffect):
    """Retrigger all held-in-hand card abilities."""

    def retrigger_held(self, ctx, j, card):
        return 1


register_joker(JokerDef("j_mime", "Mime", Rarity.UNCOMMON, 5, _Mime()))
