"""Hand base values, per-level increments, blind requirement tables, stakes.

Numbers verified against the game's prototype tables (see docs/fidelity.md).
"""

from __future__ import annotations

import math

from .types import BlindKind, HandType, Stake

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


# ---------------------------------------------------------------------------
# Stakes
# ---------------------------------------------------------------------------

STAKE_ORDER: list[Stake] = [
    Stake.WHITE, Stake.RED, Stake.GREEN, Stake.BLACK,
    Stake.BLUE, Stake.PURPLE, Stake.ORANGE, Stake.GOLD,
]


def stake_level(stake: Stake) -> int:
    """1-based difficulty index (White=1 ... Gold=8)."""
    return STAKE_ORDER.index(stake) + 1


# Ante -> base chip requirement for antes 1..8, per scaling tier.
# Tier 1: White/Red/Black/Blue; tier 2: Green; tier 3: Purple/Orange/Gold.
_ANTE_TABLES: dict[int, list[int]] = {
    1: [300, 800, 2_000, 5_000, 11_000, 20_000, 35_000, 50_000],
    2: [300, 900, 2_600, 8_000, 20_000, 36_000, 60_000, 100_000],
    3: [300, 1_000, 3_200, 9_000, 25_000, 60_000, 110_000, 200_000],
}


def scaling_tier(stake: Stake) -> int:
    lvl = stake_level(stake)
    if lvl >= 6:
        return 3
    if lvl >= 3:
        return 2
    return 1


def ante_base_chips(ante: int, stake: Stake = Stake.WHITE) -> float:
    """Base chip requirement for an ante (Small Blind = 1x of this).

    Ante 0 (Hieroglyph/Petroglyph) is 100. Antes beyond 8 use the game's
    endless formula so Endless-style evaluation keeps working."""
    table = _ANTE_TABLES[scaling_tier(stake)]
    if ante < 1:
        return 100.0
    if ante <= 8:
        return float(table[ante - 1])
    k = 0.75
    a, b, c, d = table[7], 1.6, ante - 8, 1 + 0.2 * (ante - 8)
    amount = math.floor(a * (b + (k * c) ** d) ** c)
    amount = amount - amount % (10 ** math.floor(math.log10(amount) - 1))
    return float(amount)


# Backwards-compatible White-stake table (index 0 = ante 0).
ANTE_BASE_CHIPS: dict[int, float] = {a: ante_base_chips(a) for a in range(13)}

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
INTEREST_PER = 5             # $1 interest per $5 held...
INTEREST_CAP = 5             # ...capped at $5 per round at base ($25 held)
INTEREST_CAP_DOLLARS = 25    # Seed Money -> 50, Money Tree -> 100
UNUSED_HAND_PAYOUT = 1       # $1 per unused hand at round end
BASE_REROLL_COST = 5
