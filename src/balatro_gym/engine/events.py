"""Events emitted by Game.apply(). Rewards are computed from these, not state diffs."""

from __future__ import annotations

from dataclasses import dataclass

from .scoring import ScoreResult
from .types import BlindKind


@dataclass(frozen=True, slots=True)
class Event:
    pass


@dataclass(frozen=True, slots=True)
class HandScored(Event):
    result: ScoreResult
    chips_scored: float   # cumulative this round, after this hand
    chips_needed: float


@dataclass(frozen=True, slots=True)
class Discarded(Event):
    count: int


@dataclass(frozen=True, slots=True)
class BlindCleared(Event):
    ante: int
    blind_kind: BlindKind


@dataclass(frozen=True, slots=True)
class BlindSkipped(Event):
    ante: int
    blind_kind: BlindKind
    tag_id: str | None


@dataclass(frozen=True, slots=True)
class RoundPayout(Event):
    blind_reward: int
    unused_hands: int
    interest: int
    other: int  # jokers, tags, gold cards
    total: int


@dataclass(frozen=True, slots=True)
class MoneyChanged(Event):
    delta: int
    source: str


@dataclass(frozen=True, slots=True)
class RunWon(Event):
    ante: int


@dataclass(frozen=True, slots=True)
class RunLost(Event):
    ante: int
    blind_kind: BlindKind
    chips_scored: float
    chips_needed: float


@dataclass(frozen=True, slots=True)
class ItemBought(Event):
    kind: str  # "joker" | "tarot" | "planet" | "spectral" | "voucher" | "pack" | "card"
    item_id: str
    cost: int


@dataclass(frozen=True, slots=True)
class JokerSold(Event):
    joker_id: str
    value: int


@dataclass(frozen=True, slots=True)
class ConsumableUsed(Event):
    consumable_id: str
