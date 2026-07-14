"""Independent (stage 5) jokers from the v1 set."""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import HandType, Rarity

from .helpers import contains_hand_joker, flat_independent_joker

# Joker: +4 Mult
flat_independent_joker("j_joker", "Joker", Rarity.COMMON, 2, mult=4)

# Jolly Joker: +8 Mult if played hand contains a Pair
contains_hand_joker("j_jolly", "Jolly Joker", Rarity.COMMON, 3, HandType.PAIR, mult=8)

# Sly Joker: +50 Chips if played hand contains a Pair
contains_hand_joker("j_sly", "Sly Joker", Rarity.COMMON, 3, HandType.PAIR, chips=50)

# The Duo: x2 Mult if played hand contains a Pair
contains_hand_joker("j_duo", "The Duo", Rarity.RARE, 8, HandType.PAIR, xmult=2)


class _HalfJoker(JokerEffect):
    """+20 Mult if played hand contains 3 or fewer cards."""

    def on_independent(self, ctx, j):
        if len(ctx.played) <= 3:
            ctx.add_mult(20, "joker", "j_half")


register_joker(JokerDef("j_half", "Half Joker", Rarity.COMMON, 5, _HalfJoker()))


class _Banner(JokerEffect):
    """+30 Chips per remaining discard."""

    def on_independent(self, ctx, j):
        discards = ctx.run.round.discards_left if ctx.run.round else 0
        if discards:
            ctx.add_chips(30 * discards, "joker", "j_banner")


register_joker(JokerDef("j_banner", "Banner", Rarity.COMMON, 5, _Banner()))


class _Supernova(JokerEffect):
    """+Mult equal to the number of times the played hand type has been played
    this run (includes the current play — counts are bumped post-scoring, so
    we add 1 here)."""

    def on_independent(self, ctx, j):
        n = ctx.run.hand_play_counts.get(ctx.hand_result.hand_type, 0) + 1
        ctx.add_mult(n, "joker", "j_supernova")


register_joker(JokerDef("j_supernova", "Supernova", Rarity.COMMON, 5, _Supernova()))


class _AbstractJoker(JokerEffect):
    """+3 Mult per Joker card owned (including itself)."""

    def on_independent(self, ctx, j):
        n = len(ctx.run.jokers)
        if n:
            ctx.add_mult(3 * n, "joker", "j_abstract")


register_joker(JokerDef("j_abstract", "Abstract Joker", Rarity.COMMON, 4, _AbstractJoker()))
