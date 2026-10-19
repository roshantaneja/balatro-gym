"""Engine-native actions. The env layer maps these to/from Discrete indices."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Action:
    pass


@dataclass(frozen=True, slots=True)
class SelectBlind(Action):
    pass


@dataclass(frozen=True, slots=True)
class SkipBlind(Action):
    pass


@dataclass(frozen=True, slots=True)
class RerollBoss(Action):
    """Director's Cut / Retcon: re-roll the ante's boss blind for $10."""


@dataclass(frozen=True, slots=True)
class PlayHand(Action):
    indices: tuple[int, ...]  # positions in round.hand, 1-5 of them


@dataclass(frozen=True, slots=True)
class Discard(Action):
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SellJoker(Action):
    slot: int


@dataclass(frozen=True, slots=True)
class SellConsumable(Action):
    slot: int


@dataclass(frozen=True, slots=True)
class SwapJokers(Action):
    """Swap the jokers in slots `slot` and `slot + 1` (board reordering)."""

    slot: int


@dataclass(frozen=True, slots=True)
class UseConsumable(Action):
    slot: int
    targets: tuple[int, ...] = field(default=())  # positions in the visible hand


@dataclass(frozen=True, slots=True)
class BuyShopCard(Action):
    slot: int


@dataclass(frozen=True, slots=True)
class BuyPack(Action):
    slot: int


@dataclass(frozen=True, slots=True)
class BuyVoucher(Action):
    slot: int = 0


@dataclass(frozen=True, slots=True)
class Reroll(Action):
    pass


@dataclass(frozen=True, slots=True)
class LeaveShop(Action):
    pass


@dataclass(frozen=True, slots=True)
class PickPackItem(Action):
    index: int
    targets: tuple[int, ...] = field(default=())  # pack-hand targets for arcana/spectral picks


@dataclass(frozen=True, slots=True)
class SkipPack(Action):
    pass
