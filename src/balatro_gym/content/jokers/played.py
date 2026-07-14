"""On-played (stage 2) jokers: scaling state and hand-level manipulation.

Scaling convention (see engine/scoring.py stage 2): `update_on_hand_played`
runs at stage 2 (originals only), the accumulated value is read at stage 5 —
so the very first hand already reads the post-update value. That matches the
real game for Green Joker / Ride the Bus (they gain "when hand is played",
before their scoring trigger).
"""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.hand_eval import is_face
from balatro_gym.engine.levels import HAND_LEVEL_INC
from balatro_gym.engine.scoring import fold_eval_rules
from balatro_gym.engine.types import Rarity


class _GreenJoker(JokerEffect):
    """+1 Mult per hand played, -1 Mult per discard (never below 0).

    Wiki: loses 1 per discard ACTION (not per card discarded)."""

    def on_independent(self, ctx, j):
        ctx.add_mult(j.state["mult"], "joker", "j_green_joker")

    def update_on_hand_played(self, run, j, played, hr):
        j.state["mult"] += 1

    def update_on_discard(self, run, j, cards):
        j.state["mult"] = max(0, j.state["mult"] - 1)


register_joker(
    JokerDef(
        "j_green_joker", "Green Joker", Rarity.COMMON, 4, _GreenJoker(),
        init_state={"mult": 0},
    )
)


class _RideTheBus(JokerEffect):
    """+1 Mult per consecutive hand played without a SCORING face card;
    resets to 0 when a face card scores. Pareidolia-aware (folded rules)."""

    def on_independent(self, ctx, j):
        ctx.add_mult(j.state["mult"], "joker", "j_ride_the_bus")

    def update_on_hand_played(self, run, j, played, hr):
        rules = fold_eval_rules(run)
        if any(is_face(c, rules) for c in hr.scoring_cards):
            j.state["mult"] = 0
        else:
            j.state["mult"] += 1


register_joker(
    JokerDef(
        "j_ride_the_bus", "Ride the Bus", Rarity.COMMON, 6, _RideTheBus(),
        init_state={"mult": 0},
    )
)


class _DNA(JokerEffect):
    """If the FIRST hand of the round has exactly 1 card, add a permanent copy
    of it to the deck and draw it to hand.

    round.hands_played is bumped by round.play_hand AFTER score_play, so
    hands_played == 0 identifies the first hand during scoring. Runs as an
    update hook (originals only), so a Blueprint copy adds no extra card."""

    def update_on_hand_played(self, run, j, played, hr):
        rnd = run.round
        if rnd is None or rnd.hands_played != 0 or len(played) != 1:
            return
        copy = played[0].clone()
        copy.uid = run.new_uid()
        run.full_deck.append(copy)
        rnd.hand.append(copy)


register_joker(JokerDef("j_dna", "DNA", Rarity.RARE, 8, _DNA()))


class _SpaceJoker(JokerEffect):
    """1 in 4 chance to upgrade the level of the played poker hand.

    Wiki: "The upgrade is triggered before scoring of the hand, thus the hand
    will get higher score itself." Base chips/mult were already read at stage 1
    from the OLD level, so we bump the level AND add the per-level increment to
    the live context — net effect: the current hand scores at the new level.
    Scoring hook, so a Blueprint copy rolls (and can upgrade) again."""

    def on_hand_played(self, ctx, j):
        if not ctx.chance(1, 4):
            return
        ht = ctx.hand_result.hand_type
        ctx.run.hand_levels[ht] += 1
        inc_c, inc_m = HAND_LEVEL_INC[ht]
        ctx.add_chips(inc_c, "joker", "j_space")
        ctx.add_mult(inc_m, "joker", "j_space")


register_joker(JokerDef("j_space", "Space Joker", Rarity.UNCOMMON, 5, _SpaceJoker()))
