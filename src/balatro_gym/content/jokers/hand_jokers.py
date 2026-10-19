"""Hand-type jokers: the +mult / +chips / xmult 'contains' families and the
scalers that grow when a hand shape is played.

All 'contains' jokers check `hand in ctx.hand_result.contained` — a Full
House contains a Pair, Three of a Kind and Two Pair, a Straight Flush
contains a Straight and a Flush, etc.
"""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import HandType, Rarity

from .helpers import contains_hand_joker

# +Mult family
contains_hand_joker("j_zany", "Zany Joker", Rarity.COMMON, 4, HandType.THREE_KIND, mult=12)
contains_hand_joker("j_mad", "Mad Joker", Rarity.COMMON, 4, HandType.TWO_PAIR, mult=10)
contains_hand_joker("j_crazy", "Crazy Joker", Rarity.COMMON, 4, HandType.STRAIGHT, mult=12)
contains_hand_joker("j_droll", "Droll Joker", Rarity.COMMON, 4, HandType.FLUSH, mult=10)

# +Chips family
contains_hand_joker("j_wily", "Wily Joker", Rarity.COMMON, 4, HandType.THREE_KIND, chips=100)
contains_hand_joker("j_clever", "Clever Joker", Rarity.COMMON, 4, HandType.TWO_PAIR, chips=80)
contains_hand_joker("j_devious", "Devious Joker", Rarity.COMMON, 4, HandType.STRAIGHT, chips=100)
contains_hand_joker("j_crafty", "Crafty Joker", Rarity.COMMON, 4, HandType.FLUSH, chips=80)

# xMult family (The Duo lives in independent.py)
contains_hand_joker("j_trio", "The Trio", Rarity.RARE, 8, HandType.THREE_KIND, xmult=3)
contains_hand_joker("j_family", "The Family", Rarity.RARE, 8, HandType.FOUR_KIND, xmult=4)
contains_hand_joker("j_order", "The Order", Rarity.RARE, 8, HandType.STRAIGHT, xmult=3)
contains_hand_joker("j_tribe", "The Tribe", Rarity.RARE, 8, HandType.FLUSH, xmult=2)


# ---------------------------------------------------------------------------
# Shape scalers (gain when the hand is played, read at stage 5 — so the hand
# that triggers the gain already benefits, as in the game)
# ---------------------------------------------------------------------------

class _SpareTrousers(JokerEffect):
    """Gains +2 Mult if the played hand contains a Two Pair."""

    def on_independent(self, ctx, j):
        ctx.add_mult(j.state["mult"], "joker", "j_trousers")

    def update_on_hand_played(self, run, j, played, hr):
        if HandType.TWO_PAIR in hr.contained:
            j.state["mult"] += 2


register_joker(
    JokerDef("j_trousers", "Spare Trousers", Rarity.UNCOMMON, 6, _SpareTrousers(),
             init_state={"mult": 0}, perishable_compat=False)
)


class _Runner(JokerEffect):
    """Gains +15 Chips if the played hand contains a Straight."""

    def on_independent(self, ctx, j):
        ctx.add_chips(j.state["chips"], "joker", "j_runner")

    def update_on_hand_played(self, run, j, played, hr):
        if HandType.STRAIGHT in hr.contained:
            j.state["chips"] += 15


register_joker(
    JokerDef("j_runner", "Runner", Rarity.COMMON, 5, _Runner(),
             init_state={"chips": 0}, perishable_compat=False)
)


class _SquareJoker(JokerEffect):
    """Gains +4 Chips if the played hand has exactly 4 cards."""

    def on_independent(self, ctx, j):
        ctx.add_chips(j.state["chips"], "joker", "j_square")

    def update_on_hand_played(self, run, j, played, hr):
        if len(played) == 4:
            j.state["chips"] += 4


register_joker(
    JokerDef("j_square", "Square Joker", Rarity.COMMON, 4, _SquareJoker(),
             init_state={"chips": 0}, perishable_compat=False)
)


class _CardSharp(JokerEffect):
    """x3 Mult if the played hand type has already been played this round."""

    def on_independent(self, ctx, j):
        rnd = ctx.run.round
        if rnd is not None and int(ctx.hand_result.hand_type) in rnd.hand_types_played:
            ctx.mul_mult(3.0, "joker", "j_card_sharp")


register_joker(JokerDef("j_card_sharp", "Card Sharp", Rarity.UNCOMMON, 6, _CardSharp()))


class _Obelisk(JokerEffect):
    """Gains x0.2 Mult per consecutive hand played without playing your most
    played hand; resets when a most-played hand is played. The check uses the
    counts BEFORE this hand (they are bumped after scoring)."""

    def on_independent(self, ctx, j):
        ctx.mul_mult(j.state["x"], "joker", "j_obelisk")

    def update_on_hand_played(self, run, j, played, hr):
        counts = run.hand_play_counts
        top = max(counts.values()) if counts else 0
        most = {ht for ht, n in counts.items() if n == top and top > 0}
        if hr.hand_type in most or not most:
            j.state["x"] = 1.0
        else:
            j.state["x"] = round(j.state["x"] + 0.2, 6)


register_joker(
    JokerDef("j_obelisk", "Obelisk", Rarity.RARE, 8, _Obelisk(),
             init_state={"x": 1.0}, perishable_compat=False)
)
