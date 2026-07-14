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
class PlayHand(Action):
    indices: tuple[int, ...]  # positions in round.hand, 1-5 of them


@dataclass(frozen=True, slots=True)
class Discard(Action):
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SellJoker(Action):
    slot: int


@dataclass(frozen=True, slots=True)
class UseConsumable(Action):
    slot: int
    targets: tuple[int, ...] = field(default=())  # positions in round.hand


@dataclass(frozen=True, slots=True)
class BuyShopCard(Action):
    slot: int


@dataclass(frozen=True, slots=True)
class BuyPack(Action):
    slot: int


@dataclass(frozen=True, slots=True)
class BuyVoucher(Action):
    pass


@dataclass(frozen=True, slots=True)
class Reroll(Action):
    pass


@dataclass(frozen=True, slots=True)
class LeaveShop(Action):
    pass


@dataclass(frozen=True, slots=True)
class PickPackItem(Action):
    index: int
    targets: tuple[int, ...] = field(default=())  # hand targets for arcana-style picks


@dataclass(frozen=True, slots=True)
class SkipPack(Action):
    pass
