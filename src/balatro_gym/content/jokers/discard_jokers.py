"""Discard-driven jokers (Mail-In Rebate, Castle).

Both re-roll their target at round end by sampling a random card from the
run's full deck (wiki: the rank/suit "is chosen proportionally to the deck
composition"), via the seeded "misc" stream.
"""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.hand_eval import suit_matches
from balatro_gym.engine.scoring import fold_eval_rules
from balatro_gym.engine.types import Rarity, Suit


def _deck_sample(run):
    """A random non-stone card from the full deck (stones have no rank/suit)."""
    pool = [c for c in run.full_deck if not c.is_stone]
    if not pool:
        return None
    return run.rng.stream("misc").choice(pool)


class _MailInRebate(JokerEffect):
    """Earn $5 for each discarded card of [rank]; rank changes every round.

    Money lands via update_on_discard (originals only), mutating run.money
    directly — there is no scoring context during a discard."""

    def update_on_discard(self, run, j, cards):
        n = sum(1 for c in cards if not c.is_stone and int(c.rank) == j.state["rank"])
        if n:
            run.money += 5 * n

    def update_on_round_end(self, run, j):
        card = _deck_sample(run)
        if card is not None:
            j.state["rank"] = int(card.rank)


register_joker(
    JokerDef(
        "j_mail_in", "Mail-In Rebate", Rarity.COMMON, 4, _MailInRebate(),
        init_state={"rank": 13},  # static default (King); re-rolled each round end
    )
)


class _Castle(JokerEffect):
    """Gains +3 Chips per discarded card of [suit] (permanently); suit changes
    every round. Wild cards match any suit (suit_matches with folded rules)."""

    def on_independent(self, ctx, j):
        ctx.add_chips(j.state["chips"], "joker", "j_castle")

    def update_on_discard(self, run, j, cards):
        rules = fold_eval_rules(run)
        suit = Suit(j.state["suit"])
        n = sum(1 for c in cards if suit_matches(c, suit, rules))
        if n:
            j.state["chips"] += 3 * n

    def update_on_round_end(self, run, j):
        card = _deck_sample(run)
        if card is not None:
            j.state["suit"] = int(card.suit)


register_joker(
    JokerDef(
        "j_castle", "Castle", Rarity.UNCOMMON, 6, _Castle(),
        init_state={"chips": 0, "suit": 0}, perishable_compat=False,
    )
)
