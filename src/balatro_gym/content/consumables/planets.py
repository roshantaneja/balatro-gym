"""All 12 Planet cards (Planet Cards.wiki).

Each planet levels its associated poker hand by +1; the per-level chip/mult
increments live in engine.levels.HAND_LEVEL_INC, so the effect here is just
``run.hand_levels[ht] += 1``.

The three secret planets (Planet X, Ceres, Eris) "can only be found in the
shop after playing their associated poker hand type" (Planet Cards.wiki) —
we enforce that at use time via can_use(run) checking run.hands_discovered.
"""

from __future__ import annotations

from balatro_gym.engine.effects import (
    PLANET_FOR_HAND,
    ConsumableDef,
    ConsumableEffect,
    register_consumable,
)
from balatro_gym.engine.state import RunState
from balatro_gym.engine.types import ConsumableKind, HandType, SECRET_HANDS

PLANET_BASE_COST = 3


class PlanetEffect(ConsumableEffect):
    """Level the planet's hand type by +1. Untargeted."""

    target_arity = (0, 0)

    def __init__(self, hand_type: HandType):
        self.hand_type = hand_type

    def can_use(self, run: RunState) -> bool:
        # Secret planets are unusable until their hand has been discovered.
        if self.hand_type in SECRET_HANDS:
            return self.hand_type in run.hands_discovered
        return True

    def use(self, run: RunState, targets: list) -> None:
        run.hand_levels[self.hand_type] += 1


def _planet(cid: str, name: str, hand_type: HandType) -> None:
    register_consumable(
        ConsumableDef(
            id=cid,
            name=name,
            kind=ConsumableKind.PLANET,
            base_cost=PLANET_BASE_COST,
            effect=PlanetEffect(hand_type),
        )
    )
    PLANET_FOR_HAND[hand_type] = cid


_planet("c_pluto", "Pluto", HandType.HIGH_CARD)
_planet("c_mercury", "Mercury", HandType.PAIR)
_planet("c_uranus", "Uranus", HandType.TWO_PAIR)
_planet("c_venus", "Venus", HandType.THREE_KIND)
_planet("c_saturn", "Saturn", HandType.STRAIGHT)
_planet("c_jupiter", "Jupiter", HandType.FLUSH)
_planet("c_earth", "Earth", HandType.FULL_HOUSE)
_planet("c_mars", "Mars", HandType.FOUR_KIND)
_planet("c_neptune", "Neptune", HandType.STRAIGHT_FLUSH)
# Secret planets — only usable (and only meaningful) once discovered.
_planet("c_planet_x", "Planet X", HandType.FIVE_KIND)
_planet("c_ceres", "Ceres", HandType.FLUSH_HOUSE)
_planet("c_eris", "Eris", HandType.FLUSH_FIVE)
