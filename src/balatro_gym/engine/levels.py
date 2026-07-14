"""Hand base values, per-level increments, and blind requirement tables.

Sources: Poker Hands.wiki, Planet Cards.wiki, Blinds and Antes.wiki.
"""

from __future__ import annotations

from .types import BlindKind, HandType

# HandType -> (base_chips, base_mult) at level 1
HAND_BASE: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (5, 1),
    HandType.PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.FIVE_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}

# HandType -> (+chips, +mult) per level above 1 (from that hand's Planet card)
HAND_LEVEL_INC: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (10, 1),
    HandType.PAIR: (15, 1),
    HandType.TWO_PAIR: (20, 1),
    HandType.THREE_KIND: (20, 2),
    HandType.STRAIGHT: (30, 3),
    HandType.FLUSH: (15, 2),
    HandType.FULL_HOUSE: (25, 2),
    HandType.FOUR_KIND: (30, 3),
    HandType.STRAIGHT_FLUSH: (40, 4),
    HandType.FIVE_KIND: (35, 3),
    HandType.FLUSH_HOUSE: (40, 4),
    HandType.FLUSH_FIVE: (50, 3),
}


def hand_value(hand_type: HandType, level: int) -> tuple[float, float]:
    """(chips, mult) for a hand type at the given level (level >= 1)."""
    base_c, base_m = HAND_BASE[hand_type]
    inc_c, inc_m = HAND_LEVEL_INC[hand_type]
    lvl = max(1, level)
    return float(base_c + (lvl - 1) * inc_c), float(base_m + (lvl - 1) * inc_m)


# Ante -> base chip requirement (White stake). Index 0 = ante 0 (Hieroglyph).
ANTE_BASE_CHIPS: dict[int, float] = {
    0: 100,
    1: 300,
    2: 800,
    3: 2_000,
    4: 5_000,
    5: 11_000,
    6: 20_000,
    7: 35_000,
    8: 50_000,
    # Endless (kept for completeness; v1 stops at 8)
    9: 110_000,
    10: 560_000,
    11: 7_200_000,
    12: 300_000_000,
}

BLIND_SCORE_MULT: dict[BlindKind, float] = {
    BlindKind.SMALL: 1.0,
    BlindKind.BIG: 1.5,
    BlindKind.BOSS: 2.0,  # boss effects may override (Wall 4x, Needle 1x, Violet Vessel 6x)
}

BLIND_REWARD: dict[BlindKind, int] = {
    BlindKind.SMALL: 3,
    BlindKind.BIG: 4,
    BlindKind.BOSS: 5,  # finisher bosses pay $8 (handled by boss def)
}

WIN_ANTE = 8

# Economy defaults
INTEREST_PER = 5        # $1 interest per $5 held...
INTEREST_CAP = 5        # ...capped at $5 per round
UNUSED_HAND_PAYOUT = 1  # $1 per unused hand at round end
BASE_REROLL_COST = 5
