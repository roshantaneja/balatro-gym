"""Factory helpers for parametric joker families — data-driven without a DSL."""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import HandType, Rarity, Suit


def flat_independent_joker(
    jid: str, name: str, rarity: Rarity, cost: int, *, chips: float = 0, mult: float = 0
) -> JokerDef:
    """Unconditional +chips/+mult once per hand (stage 5)."""

    class _E(JokerEffect):
        def on_independent(self, ctx, j):
            if chips:
                ctx.add_chips(chips, "joker", jid)
            if mult:
                ctx.add_mult(mult, "joker", jid)

    return register_joker(JokerDef(jid, name, rarity, cost, _E()))


def contains_hand_joker(
    jid: str,
    name: str,
    rarity: Rarity,
    cost: int,
    hand: HandType,
    *,
    chips: float = 0,
    mult: float = 0,
    xmult: float = 1,
) -> JokerDef:
    """+chips/+mult/xmult if the played hand CONTAINS the given hand type
    (Jolly/Sly/The Duo family — 'contains', not 'is')."""

    class _E(JokerEffect):
        def on_independent(self, ctx, j):
            if hand in ctx.hand_result.contained:
                if chips:
                    ctx.add_chips(chips, "joker", jid)
                if mult:
                    ctx.add_mult(mult, "joker", jid)
                if xmult != 1:
                    ctx.mul_mult(xmult, "joker", jid)

    return register_joker(JokerDef(jid, name, rarity, cost, _E()))


def suit_mult_joker(
    jid: str, name: str, rarity: Rarity, cost: int, suit: Suit, mult: float
) -> JokerDef:
    """+mult per scored card of the given suit (Greedy/Lusty/Wrathful/Gluttonous).
    Wild/Smeared-aware via ctx.card_has_suit."""

    class _E(JokerEffect):
        def on_card_scored(self, ctx, j, card):
            if ctx.card_has_suit(card, suit):
                ctx.add_mult(mult, "joker", jid)

    return register_joker(JokerDef(jid, name, rarity, cost, _E()))
