"""Blueprint and Brainstorm — pure copy pointers; resolution lives in the engine."""

from __future__ import annotations

from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Rarity

# The effect object is never dispatched for copy jokers (resolve_joker follows
# copy_target before reaching an effect), but a base instance keeps the type uniform.
register_joker(
    JokerDef(
        "j_blueprint",
        "Blueprint",
        Rarity.RARE,
        10,
        JokerEffect(),
        copyable=True,
        copy_target="right",
    )
)

register_joker(
    JokerDef(
        "j_brainstorm",
        "Brainstorm",
        Rarity.RARE,
        10,
        JokerEffect(),
        copyable=True,
        copy_target="leftmost",
    )
)
