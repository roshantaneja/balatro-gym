"""On-scored (stage 3.3) jokers and card-level retriggers.

Rank/suit predicates: `card.is_stone` cards have no rank or suit and never
match; suit checks go through ctx.card_has_suit (Wild / Smeared aware);
face checks through ctx.is_face (Pareidolia aware).
"""

from __future__ import annotations

from balatro_gym.engine import cardops
from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Enhancement, Rank, Rarity, Suit


def _rank_in(card, ranks) -> bool:
    return not card.is_stone and card.rank in ranks


class _Fibonacci(JokerEffect):
    """Each scored Ace, 2, 3, 5 or 8 gives +8 Mult."""

    RANKS = {Rank.ACE, Rank.TWO, Rank.THREE, Rank.FIVE, Rank.EIGHT}

    def on_card_scored(self, ctx, j, card):
        if _rank_in(card, self.RANKS):
            ctx.add_mult(8, "joker", "j_fibonacci")


register_joker(JokerDef("j_fibonacci", "Fibonacci", Rarity.UNCOMMON, 8, _Fibonacci()))


class _EvenSteven(JokerEffect):
    """Each scored even-ranked card (10, 8, 6, 4, 2) gives +4 Mult."""

    RANKS = {Rank.TWO, Rank.FOUR, Rank.SIX, Rank.EIGHT, Rank.TEN}

    def on_card_scored(self, ctx, j, card):
        if _rank_in(card, self.RANKS):
            ctx.add_mult(4, "joker", "j_even_steven")


register_joker(JokerDef("j_even_steven", "Even Steven", Rarity.COMMON, 4, _EvenSteven()))


class _ScaryFace(JokerEffect):
    """Scored face cards give +30 Chips."""

    def on_card_scored(self, ctx, j, card):
        if ctx.is_face(card):
            ctx.add_chips(30, "joker", "j_scary_face")


register_joker(JokerDef("j_scary_face", "Scary Face", Rarity.COMMON, 4, _ScaryFace()))


class _SmileyFace(JokerEffect):
    """Scored face cards give +5 Mult."""

    def on_card_scored(self, ctx, j, card):
        if ctx.is_face(card):
            ctx.add_mult(5, "joker", "j_smiley")


register_joker(JokerDef("j_smiley", "Smiley Face", Rarity.COMMON, 4, _SmileyFace()))


class _WalkieTalkie(JokerEffect):
    """Each scored 10 or 4 gives +10 Chips and +4 Mult."""

    def on_card_scored(self, ctx, j, card):
        if _rank_in(card, {Rank.TEN, Rank.FOUR}):
            ctx.add_chips(10, "joker", "j_walkie_talkie")
            ctx.add_mult(4, "joker", "j_walkie_talkie")


register_joker(JokerDef("j_walkie_talkie", "Walkie Talkie", Rarity.COMMON, 4, _WalkieTalkie()))


class _WeeJoker(JokerEffect):
    """Gains +8 Chips when each scored 2 is scored; the chips are paid at
    stage 5. Copies read the state but do not grow it."""

    def on_card_scored(self, ctx, j, card):
        if _rank_in(card, {Rank.TWO}) and not ctx.is_copy:
            j.state["chips"] += 8

    def on_independent(self, ctx, j):
        ctx.add_chips(j.state["chips"], "joker", "j_wee")


register_joker(
    JokerDef("j_wee", "Wee Joker", Rarity.RARE, 8, _WeeJoker(),
             init_state={"chips": 0}, perishable_compat=False)
)


class _Arrowhead(JokerEffect):
    """Scored Spades give +50 Chips."""

    def on_card_scored(self, ctx, j, card):
        if ctx.card_has_suit(card, Suit.SPADES):
            ctx.add_chips(50, "joker", "j_arrowhead")


register_joker(JokerDef("j_arrowhead", "Arrowhead", Rarity.UNCOMMON, 7, _Arrowhead()))


class _OnyxAgate(JokerEffect):
    """Scored Clubs give +7 Mult."""

    def on_card_scored(self, ctx, j, card):
        if ctx.card_has_suit(card, Suit.CLUBS):
            ctx.add_mult(7, "joker", "j_onyx_agate")


register_joker(JokerDef("j_onyx_agate", "Onyx Agate", Rarity.UNCOMMON, 7, _OnyxAgate()))


class _RoughGem(JokerEffect):
    """Scored Diamonds earn $1."""

    def on_card_scored(self, ctx, j, card):
        if ctx.card_has_suit(card, Suit.DIAMONDS):
            ctx.add_money(1, "joker", "j_rough_gem")


register_joker(JokerDef("j_rough_gem", "Rough Gem", Rarity.UNCOMMON, 7, _RoughGem()))


class _GoldenTicket(JokerEffect):
    """Scored Gold cards earn $4."""

    def on_card_scored(self, ctx, j, card):
        if card.enhancement == Enhancement.GOLD:
            ctx.add_money(4, "joker", "j_ticket")


register_joker(JokerDef("j_ticket", "Golden Ticket", Rarity.COMMON, 5, _GoldenTicket()))


class _Hiker(JokerEffect):
    """Every scored card permanently gains +5 Chips."""

    def on_card_scored(self, ctx, j, card):
        card.extra_chips += 5
        ctx.note("joker", "j_hiker")


register_joker(JokerDef("j_hiker", "Hiker", Rarity.UNCOMMON, 5, _Hiker()))


class _MidasMask(JokerEffect):
    """Scored face cards become Gold cards."""

    def on_card_scored(self, ctx, j, card):
        if ctx.is_face(card) and card.enhancement != Enhancement.GOLD:
            card.enhancement = Enhancement.GOLD
            ctx.note("joker", "j_midas_mask")


register_joker(JokerDef("j_midas_mask", "Midas Mask", Rarity.UNCOMMON, 7, _MidasMask(), copyable=False))


class _Vampire(JokerEffect):
    """Gains x0.1 Mult per scored Enhanced card, removing the enhancement.
    The xMult is applied at stage 5 (copies read it; only the original grows)."""

    def on_card_scored(self, ctx, j, card):
        if card.enhancement != Enhancement.NONE:
            card.enhancement = Enhancement.NONE
            if not ctx.is_copy:
                j.state["x"] = round(j.state["x"] + 0.1, 6)
            ctx.note("joker", "j_vampire")

    def on_independent(self, ctx, j):
        ctx.mul_mult(j.state["x"], "joker", "j_vampire")


register_joker(
    JokerDef("j_vampire", "Vampire", Rarity.UNCOMMON, 7, _Vampire(),
             init_state={"x": 1.0}, perishable_compat=False)
)


class _LuckyCat(JokerEffect):
    """Gains x0.25 Mult every time a Lucky card triggers."""

    def on_independent(self, ctx, j):
        ctx.mul_mult(j.state["x"], "joker", "j_lucky_cat")

    def update_on_lucky(self, run, j):
        j.state["x"] = round(j.state["x"] + 0.25, 6)


register_joker(
    JokerDef("j_lucky_cat", "Lucky Cat", Rarity.UNCOMMON, 6, _LuckyCat(),
             init_state={"x": 1.0}, perishable_compat=False)
)


class _TheIdol(JokerEffect):
    """Each scored card of [rank] of [suit] gives x2 Mult; the card changes
    every round (sampled from the deck)."""

    def _reroll(self, run, j):
        card = cardops.random_deck_card(run)
        if card is not None:
            j.state["rank"] = int(card.rank)
            j.state["suit"] = int(card.suit)

    def on_acquired(self, run, j):
        self._reroll(run, j)

    def on_card_scored(self, ctx, j, card):
        if (
            not card.is_stone
            and int(card.rank) == j.state["rank"]
            and ctx.card_has_suit(card, Suit(j.state["suit"]))
        ):
            ctx.mul_mult(2.0, "joker", "j_idol")

    def update_on_round_end(self, run, j):
        self._reroll(run, j)


register_joker(
    JokerDef("j_idol", "The Idol", Rarity.UNCOMMON, 6, _TheIdol(),
             init_state={"rank": 14, "suit": 0})
)


# ---------------------------------------------------------------------------
# Retriggers
# ---------------------------------------------------------------------------

class _HangingChad(JokerEffect):
    """Retrigger the first scored card 2 additional times."""

    def retrigger_scored(self, ctx, j, card):
        scoring = ctx.hand_result.scoring_cards
        if scoring and card.uid == scoring[0].uid:
            return 2
        return 0


register_joker(JokerDef("j_hanging_chad", "Hanging Chad", Rarity.COMMON, 4, _HangingChad()))


class _SockAndBuskin(JokerEffect):
    """Retrigger all scored face cards."""

    def retrigger_scored(self, ctx, j, card):
        return 1 if ctx.is_face(card) else 0


register_joker(JokerDef("j_sock_and_buskin", "Sock and Buskin", Rarity.UNCOMMON, 6, _SockAndBuskin()))


class _Seltzer(JokerEffect):
    """Retrigger every scored card for the next 10 hands, then it is drunk."""

    def retrigger_scored(self, ctx, j, card):
        return 1 if j.state["hands"] > 0 else 0

    def update_after_scoring(self, run, j, result):
        j.state["hands"] -= 1
        if j.state["hands"] <= 0:
            run.pending_destroy.append(j.uid)


register_joker(
    JokerDef("j_seltzer", "Seltzer", Rarity.UNCOMMON, 6, _Seltzer(),
             init_state={"hands": 10}, eternal_compat=False)
)
