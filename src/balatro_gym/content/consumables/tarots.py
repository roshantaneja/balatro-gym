"""Tarot cards, v1 subset (Tarot Cards.wiki).

Card-targeting tarots mutate the Card objects in place; round containers
alias the same objects as run.full_deck, so changes persist for the run
without any syncing (see engine/round.py module docstring).

Wiki notes honoured here:
- Strength: "Rank order: A -> 2 -> 3 ... -> K -> A" — a King becomes an Ace
  and an Ace wraps around to a 2.
- Death: "convert the LEFT card into the RIGHT card" — targets arrive in
  selection order, so targets[0] is the left card, targets[1] the right.
- The Hermit "does nothing if you are below $0".
- The Wheel of Fortune: "12.5% chance of giving Foil, 8.75% chance for
  Holographic, and a 3.75% chance of Polychrome" (= 1 in 4 total); "will not
  apply to any Joker that already has an edition (including Negative)"; and
  "cannot be used if all Jokers already have an edition".
"""

from __future__ import annotations

from balatro_gym.engine.cards import Card
from balatro_gym.engine.effects import (
    ConsumableDef,
    ConsumableEffect,
    register_consumable,
)
from balatro_gym.engine.state import RunState
from balatro_gym.engine.types import ConsumableKind, Edition, Enhancement, Rank, Suit

TAROT_BASE_COST = 3


def _tarot(cid: str, name: str, effect: ConsumableEffect) -> None:
    register_consumable(
        ConsumableDef(
            id=cid,
            name=name,
            kind=ConsumableKind.TAROT,
            base_cost=TAROT_BASE_COST,
            effect=effect,
        )
    )


# ---------------------------------------------------------------------------
# Enhancement tarots (Magician / Empress / Hierophant / Tower)
# ---------------------------------------------------------------------------

class EnhanceEffect(ConsumableEffect):
    """Set an enhancement on the targeted cards."""

    def __init__(self, enhancement: Enhancement, arity: tuple[int, int]):
        self.enhancement = enhancement
        self.target_arity = arity

    def use(self, run: RunState, targets: list[Card]) -> None:
        for c in targets:
            c.enhancement = self.enhancement


_tarot("c_magician", "The Magician", EnhanceEffect(Enhancement.LUCKY, (1, 2)))
_tarot("c_empress", "The Empress", EnhanceEffect(Enhancement.MULT, (1, 2)))
_tarot("c_hierophant", "The Hierophant", EnhanceEffect(Enhancement.BONUS, (1, 2)))
_tarot("c_tower", "The Tower", EnhanceEffect(Enhancement.STONE, (1, 1)))


# ---------------------------------------------------------------------------
# Suit converters (Star / Sun / Moon / World)
# ---------------------------------------------------------------------------

class SuitConvertEffect(ConsumableEffect):
    """Convert up to 3 selected cards to a suit; rank/enhancement unchanged."""

    target_arity = (1, 3)

    def __init__(self, suit: Suit):
        self.suit = suit

    def use(self, run: RunState, targets: list[Card]) -> None:
        for c in targets:
            c.suit = self.suit


_tarot("c_star", "The Star", SuitConvertEffect(Suit.DIAMONDS))
_tarot("c_sun", "The Sun", SuitConvertEffect(Suit.HEARTS))
_tarot("c_moon", "The Moon", SuitConvertEffect(Suit.CLUBS))
_tarot("c_world", "The World", SuitConvertEffect(Suit.SPADES))


# ---------------------------------------------------------------------------
# Strength — raise rank of up to 2 cards by 1 (K -> A, A wraps to 2)
# ---------------------------------------------------------------------------

class StrengthEffect(ConsumableEffect):
    target_arity = (1, 2)

    def use(self, run: RunState, targets: list[Card]) -> None:
        for c in targets:
            c.rank = Rank.TWO if c.rank == Rank.ACE else Rank(int(c.rank) + 1)


_tarot("c_strength", "Strength", StrengthEffect())


# ---------------------------------------------------------------------------
# Death — convert the left card into a copy of the right card
# ---------------------------------------------------------------------------

class DeathEffect(ConsumableEffect):
    target_arity = (2, 2)

    def use(self, run: RunState, targets: list[Card]) -> None:
        left, right = targets
        left.rank = right.rank
        left.suit = right.suit
        left.enhancement = right.enhancement
        left.edition = right.edition
        left.seal = right.seal
        left.extra_chips = right.extra_chips
        # left.uid intentionally kept: the card object keeps its identity.


_tarot("c_death", "Death", DeathEffect())


# ---------------------------------------------------------------------------
# Money tarots (Hermit / Temperance)
# ---------------------------------------------------------------------------

class HermitEffect(ConsumableEffect):
    """Double money (max +$20). Does nothing below $0."""

    def use(self, run: RunState, targets: list[Card]) -> None:
        run.money += min(max(run.money, 0), 20)


_tarot("c_hermit", "The Hermit", HermitEffect())


class TemperanceEffect(ConsumableEffect):
    """Gives the total sell value of all current Jokers (max $50)."""

    def use(self, run: RunState, targets: list[Card]) -> None:
        run.money += min(sum(j.sell_value for j in run.jokers), 50)


_tarot("c_temperance", "Temperance", TemperanceEffect())


# ---------------------------------------------------------------------------
# The Wheel of Fortune
# ---------------------------------------------------------------------------

class WheelOfFortuneEffect(ConsumableEffect):
    """1 in 4 chance to add Foil/Holo/Polychrome to a random edition-less Joker."""

    # Sub-chances from Tarot Cards.wiki usage notes (sum to 25%).
    _EDITIONS = [(Edition.FOIL, 12.5), (Edition.HOLO, 8.75), (Edition.POLY, 3.75)]

    def can_use(self, run: RunState) -> bool:
        return any(j.edition == Edition.NONE for j in run.jokers)

    def use(self, run: RunState, targets: list[Card]) -> None:
        rng = run.rng.stream("consumable")
        # "1 in 4" listed chance, scaled by Oops! All 6s style multipliers.
        if rng.random() >= min(1.0, 0.25 * run.chance_multiplier):
            return  # nope!
        x = rng.random() * 25.0
        edition = Edition.POLY
        for ed, w in self._EDITIONS:
            x -= w
            if x <= 0:
                edition = ed
                break
        pool = [j for j in run.jokers if j.edition == Edition.NONE]
        rng.choice(pool).edition = edition


_tarot("c_wheel", "The Wheel of Fortune", WheelOfFortuneEffect())
