"""Playing cards and deck construction."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .types import Edition, Enhancement, Rank, Seal, Suit


@dataclass(slots=True)
class Card:
    uid: int
    rank: Rank
    suit: Suit
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: Seal = Seal.NONE
    extra_chips: int = 0  # permanent per-card bonus (Hiker etc.)

    # NOTE: "debuffed" is derived per-context (boss effects, The Pillar) via
    # ScoreContext.is_debuffed / Game helpers — never stored on the card.

    @property
    def is_stone(self) -> bool:
        return self.enhancement == Enhancement.STONE

    @property
    def base_chips(self) -> int:
        if self.is_stone:
            return 50 + self.extra_chips
        return self.rank.chip_value + self.extra_chips

    def clone(self) -> Card:
        return replace(self)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "rank": int(self.rank),
            "suit": int(self.suit),
            "enhancement": str(self.enhancement),
            "edition": str(self.edition),
            "seal": str(self.seal),
            "extra_chips": self.extra_chips,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        return cls(
            uid=d["uid"],
            rank=Rank(d["rank"]),
            suit=Suit(d["suit"]),
            enhancement=Enhancement(d["enhancement"]),
            edition=Edition(d["edition"]),
            seal=Seal(d["seal"]),
            extra_chips=d["extra_chips"],
        )

    def __repr__(self) -> str:  # compact, e.g. "K♠", "A♥(glass,red)"
        suit_sym = {Suit.SPADES: "♠", Suit.HEARTS: "♥", Suit.CLUBS: "♣", Suit.DIAMONDS: "♦"}
        rank_sym = {11: "J", 12: "Q", 13: "K", 14: "A"}
        r = rank_sym.get(int(self.rank), str(int(self.rank)))
        mods = [
            str(m)
            for m in (self.enhancement, self.edition, self.seal)
            if not str(m).startswith("none")
        ]
        return f"{r}{suit_sym[self.suit]}" + (f"({','.join(mods)})" if mods else "")


def make_standard_deck(start_uid: int = 0) -> list[Card]:
    """52 unmodified cards; uids are start_uid..start_uid+51."""
    uid = start_uid
    deck: list[Card] = []
    for suit in Suit:
        for rank in Rank:
            deck.append(Card(uid=uid, rank=rank, suit=suit))
            uid += 1
    return deck
