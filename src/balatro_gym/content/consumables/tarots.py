"""All 22 Tarot cards.

Card-targeting tarots mutate the Card objects in place; round containers
(and the hand dealt by an open Arcana pack) alias the same objects as
run.full_deck, so changes persist for the run without any syncing.

Rules honoured here:
- Strength: rank goes up one step; a King becomes an Ace and an Ace wraps to 2.
- Death: convert the LEFT selected card into the RIGHT one — targets arrive
  in selection order, so targets[0] is the left card.
- The Hermit does nothing below $0; Temperance caps at $50.
- The Wheel of Fortune: 1 in 4 to add Foil (12.5%) / Holographic (8.75%) /
  Polychrome (3.75%) to a random edition-less joker; unusable when every
  joker already has an edition.
- The Fool creates a copy of the last Tarot or Planet used this run (never
  The Fool itself); The High Priestess / The Emperor create up to 2 random
  Planet / Tarot cards (limited by free slots); Judgement creates a random
  joker (needs a free slot); The Hanged Man destroys up to 2 selected cards.
"""

from __future__ import annotations

from balatro_gym.engine import cardops
from balatro_gym.engine.cards import Card
from balatro_gym.engine.effects import (
    CONSUMABLES,
    ConsumableDef,
    ConsumableEffect,
    register_consumable,
)
from balatro_gym.engine.state import RunState
from balatro_gym.engine.types import ConsumableKind, Edition, Enhancement, Rank, Suit

TAROT_BASE_COST = 3


def _tarot(cid: str, name: str, effect: ConsumableEffect, *, track_last: bool = True) -> None:
    register_consumable(
        ConsumableDef(
            id=cid,
            name=name,
            kind=ConsumableKind.TAROT,
            base_cost=TAROT_BASE_COST,
            effect=effect,
            track_last=track_last,
        )
    )


# ---------------------------------------------------------------------------
# Enhancement tarots
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
_tarot("c_lovers", "The Lovers", EnhanceEffect(Enhancement.WILD, (1, 1)))
_tarot("c_chariot", "The Chariot", EnhanceEffect(Enhancement.STEEL, (1, 1)))
_tarot("c_justice", "Justice", EnhanceEffect(Enhancement.GLASS, (1, 1)))
_tarot("c_devil", "The Devil", EnhanceEffect(Enhancement.GOLD, (1, 1)))
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
# Strength / Death / Hanged Man
# ---------------------------------------------------------------------------

class StrengthEffect(ConsumableEffect):
    target_arity = (1, 2)

    def use(self, run: RunState, targets: list[Card]) -> None:
        for c in targets:
            c.rank = Rank.TWO if c.rank == Rank.ACE else Rank(int(c.rank) + 1)


_tarot("c_strength", "Strength", StrengthEffect())


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


_tarot("c_death", "Death", DeathEffect())


class HangedManEffect(ConsumableEffect):
    target_arity = (1, 2)

    def use(self, run: RunState, targets: list[Card]) -> None:
        cardops.destroy_cards(run, list(targets))


_tarot("c_hanged_man", "The Hanged Man", HangedManEffect())


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

    _EDITIONS = [(Edition.FOIL, 12.5), (Edition.HOLO, 8.75), (Edition.POLY, 3.75)]

    def can_use(self, run: RunState) -> bool:
        return any(j.edition == Edition.NONE for j in run.jokers)

    def use(self, run: RunState, targets: list[Card]) -> None:
        rng = run.rng.stream("consumable")
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


# ---------------------------------------------------------------------------
# Creators: Fool / High Priestess / Emperor / Judgement
# ---------------------------------------------------------------------------

class FoolEffect(ConsumableEffect):
    """Create the last Tarot or Planet card used this run (not The Fool)."""

    def can_use(self, run: RunState) -> bool:
        last = run.last_consumable
        if last is None or last not in CONSUMABLES or last == "c_fool":
            return False
        if CONSUMABLES[last].kind not in (ConsumableKind.TAROT, ConsumableKind.PLANET):
            return False
        return cardops.has_consumable_room(run)

    def use(self, run: RunState, targets: list[Card]) -> None:
        assert run.last_consumable is not None
        cardops.create_consumable(run, run.last_consumable)


_tarot("c_fool", "The Fool", FoolEffect(), track_last=False)


class CreateConsumablesEffect(ConsumableEffect):
    """Create up to N random consumables of a kind (as many as slots allow)."""

    def __init__(self, kind: ConsumableKind, n: int):
        self.kind = kind
        self.n = n

    def use(self, run: RunState, targets: list[Card]) -> None:
        for _ in range(self.n):
            if cardops.create_random_consumable(run, self.kind) is None:
                break


_tarot("c_high_priestess", "The High Priestess", CreateConsumablesEffect(ConsumableKind.PLANET, 2))
_tarot("c_emperor", "The Emperor", CreateConsumablesEffect(ConsumableKind.TAROT, 2))


class JudgementEffect(ConsumableEffect):
    """Create a random Joker (needs a free slot)."""

    def can_use(self, run: RunState) -> bool:
        return cardops.has_joker_room(run) and bool(cardops.joker_pool(run, None))

    def use(self, run: RunState, targets: list[Card]) -> None:
        jid = cardops.random_joker_id(run, None, stream="consumable")
        if jid is not None:
            cardops.create_joker(run, jid)


_tarot("c_judgement", "Judgement", JudgementEffect())
