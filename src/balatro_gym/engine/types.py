"""Core enums shared across the engine. No game content lives here."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def chip_value(self) -> int:
        # Ace scores 11; face cards 10; number cards their value.
        if self == Rank.ACE:
            return 11
        if self >= Rank.JACK:
            return 10
        return int(self)

    @property
    def is_face(self) -> bool:
        return self in (Rank.JACK, Rank.QUEEN, Rank.KING)


class Suit(IntEnum):
    SPADES = 0
    HEARTS = 1
    CLUBS = 2
    DIAMONDS = 3


class Enhancement(StrEnum):
    NONE = "none"
    BONUS = "bonus"      # +30 chips when scored
    MULT = "mult"        # +4 mult when scored
    WILD = "wild"        # counts as every suit
    GLASS = "glass"      # x2 mult when scored, 1 in 4 chance to break
    STEEL = "steel"      # x1.5 mult while held in hand
    STONE = "stone"      # +50 chips, no rank/suit, always scores
    GOLD = "gold"        # $3 if held in hand at end of round
    LUCKY = "lucky"      # 1 in 5: +20 mult; 1 in 15: +$20


class Edition(StrEnum):
    NONE = "none"
    FOIL = "foil"            # +50 chips
    HOLO = "holo"            # +10 mult
    POLY = "poly"            # x1.5 mult
    NEGATIVE = "negative"    # joker: +1 joker slot


class Seal(StrEnum):
    NONE = "none"
    RED = "red"        # retrigger this card once
    BLUE = "blue"      # creates Planet of played hand if held at end of round
    GOLD = "gold"      # +$3 when this card is played and scores
    PURPLE = "purple"  # creates a Tarot when discarded


class HandType(IntEnum):
    """Value doubles as detection precedence: higher wins."""

    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_KIND = 8
    STRAIGHT_FLUSH = 9
    FIVE_KIND = 10
    FLUSH_HOUSE = 11
    FLUSH_FIVE = 12


# The three "secret" hands are hidden until first discovered in a run.
SECRET_HANDS = frozenset({HandType.FIVE_KIND, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE})


class Rarity(StrEnum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    LEGENDARY = "legendary"


class ConsumableKind(StrEnum):
    TAROT = "tarot"
    PLANET = "planet"
    SPECTRAL = "spectral"


class Phase(StrEnum):
    BLIND_SELECT = "blind_select"
    PLAYING = "playing"
    SHOP = "shop"
    PACK_OPEN = "pack_open"
    WON = "won"
    GAME_OVER = "game_over"


class BlindKind(IntEnum):
    SMALL = 0
    BIG = 1
    BOSS = 2


class Stake(StrEnum):
    """The 8 difficulty stakes, in unlock order. Each stake includes every
    modifier of the stakes below it (see engine.levels.STAKE_ORDER)."""

    WHITE = "white"    # base game
    RED = "red"        # Small Blind gives no reward money
    GREEN = "green"    # required score scales faster per ante
    BLACK = "black"    # shop can have Eternal jokers (can't be sold or destroyed)
    BLUE = "blue"      # -1 discard per round
    PURPLE = "purple"  # required score scales faster still
    ORANGE = "orange"  # shop can have Perishable jokers (debuffed after 5 rounds)
    GOLD = "gold"      # shop can have Rental jokers ($3 per round)


# Joker stickers (JokerInst.stickers entries).
STICKER_ETERNAL = "eternal"        # cannot be sold or destroyed
STICKER_PERISHABLE = "perishable"  # debuffed after 5 rounds
STICKER_RENTAL = "rental"          # costs $1, charges $3 at the end of every round
PERISHABLE_ROUNDS = 5
RENTAL_FEE = 3
