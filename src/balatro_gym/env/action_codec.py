"""Discrete(556) action space: the single source of truth for index <-> action.

Layout (fixed global indices):
    0         SELECT_BLIND
    1         SKIP_BLIND
    2-219     PLAY_HAND(subset)   -- all C(8,k), k=1..5, ascending bitmask order (218)
    220-437   DISCARD(subset)     -- same subset table (218)
    438-529   SELECT_TARGETS(subset) -- all C(8,k), k=1..3 (92); TARGET micro-phase
    530-532   USE_CONSUMABLE(slot 0-2)
    533-538   SELL_JOKER(slot 0-5)
    539-541   BUY_SHOP_CARD(slot 0-2)
    542-543   BUY_PACK(slot 0-1)
    544       BUY_VOUCHER
    545       REROLL
    546       LEAVE_SHOP
    547-551   PICK_PACK_ITEM(0-4)
    552       SKIP_PACK
    553-555   reserved

Hand slots are canonically sorted by the env (rank-desc, suit tiebreak) so
subset indices refer to stable, content-based positions.
"""

from __future__ import annotations

import numpy as np

from balatro_gym.engine import (
    Action,
    BuyPack,
    BuyShopCard,
    BuyVoucher,
    Discard,
    LeaveShop,
    PickPackItem,
    PlayHand,
    Reroll,
    SelectBlind,
    SellJoker,
    SkipBlind,
    SkipPack,
    UseConsumable,
)

HAND_SLOTS = 8
MAX_PLAY = 5
MAX_TARGETS = 3


def _subsets(max_k: int) -> list[tuple[int, ...]]:
    out = []
    for mask in range(1, 1 << HAND_SLOTS):
        if 1 <= mask.bit_count() <= max_k:
            out.append(tuple(i for i in range(HAND_SLOTS) if mask >> i & 1))
    return out


SUBSETS_5: list[tuple[int, ...]] = _subsets(MAX_PLAY)   # 218
SUBSETS_3: list[tuple[int, ...]] = _subsets(MAX_TARGETS)  # 92
_SUBSET5_INDEX: dict[tuple[int, ...], int] = {s: i for i, s in enumerate(SUBSETS_5)}
_SUBSET3_INDEX: dict[tuple[int, ...], int] = {s: i for i, s in enumerate(SUBSETS_3)}

N_PLAY = len(SUBSETS_5)
N_TARGET = len(SUBSETS_3)

SELECT_BLIND = 0
SKIP_BLIND = 1
PLAY_BASE = 2
DISCARD_BASE = PLAY_BASE + N_PLAY            # 220
TARGET_BASE = DISCARD_BASE + N_PLAY          # 438
USE_CONSUMABLE_BASE = TARGET_BASE + N_TARGET  # 530
N_CONSUMABLE_SLOTS = 3
SELL_JOKER_BASE = USE_CONSUMABLE_BASE + N_CONSUMABLE_SLOTS  # 533
N_SELL_SLOTS = 6
BUY_CARD_BASE = SELL_JOKER_BASE + N_SELL_SLOTS  # 539
N_BUY_CARD_SLOTS = 3
BUY_PACK_BASE = BUY_CARD_BASE + N_BUY_CARD_SLOTS  # 542
N_BUY_PACK_SLOTS = 2
BUY_VOUCHER_IDX = BUY_PACK_BASE + N_BUY_PACK_SLOTS  # 544
REROLL_IDX = BUY_VOUCHER_IDX + 1  # 545
LEAVE_SHOP_IDX = REROLL_IDX + 1  # 546
PICK_PACK_BASE = LEAVE_SHOP_IDX + 1  # 547
N_PACK_OPTIONS = 5
SKIP_PACK_IDX = PICK_PACK_BASE + N_PACK_OPTIONS  # 552
N_ACTIONS = 556

assert SKIP_PACK_IDX == 552


def encode(action: Action) -> int:
    """EngineAction -> global index. Raises KeyError for unencodable actions."""
    match action:
        case SelectBlind():
            return SELECT_BLIND
        case SkipBlind():
            return SKIP_BLIND
        case PlayHand(indices=idxs):
            return PLAY_BASE + _SUBSET5_INDEX[tuple(sorted(idxs))]
        case Discard(indices=idxs):
            return DISCARD_BASE + _SUBSET5_INDEX[tuple(sorted(idxs))]
        case UseConsumable(slot=s, targets=()) if s < N_CONSUMABLE_SLOTS:
            return USE_CONSUMABLE_BASE + s
        case SellJoker(slot=s) if s < N_SELL_SLOTS:
            return SELL_JOKER_BASE + s
        case BuyShopCard(slot=s) if s < N_BUY_CARD_SLOTS:
            return BUY_CARD_BASE + s
        case BuyPack(slot=s) if s < N_BUY_PACK_SLOTS:
            return BUY_PACK_BASE + s
        case BuyVoucher():
            return BUY_VOUCHER_IDX
        case Reroll():
            return REROLL_IDX
        case LeaveShop():
            return LEAVE_SHOP_IDX
        case PickPackItem(index=i) if i < N_PACK_OPTIONS:
            return PICK_PACK_BASE + i
        case SkipPack():
            return SKIP_PACK_IDX
    raise KeyError(f"unencodable action: {action!r}")


def decode_kind(index: int) -> tuple[str, tuple[int, ...] | int | None]:
    """Global index -> (kind, payload). Payload: subset tuple, slot int, or None."""
    if index == SELECT_BLIND:
        return "select_blind", None
    if index == SKIP_BLIND:
        return "skip_blind", None
    if PLAY_BASE <= index < DISCARD_BASE:
        return "play", SUBSETS_5[index - PLAY_BASE]
    if DISCARD_BASE <= index < TARGET_BASE:
        return "discard", SUBSETS_5[index - DISCARD_BASE]
    if TARGET_BASE <= index < USE_CONSUMABLE_BASE:
        return "select_targets", SUBSETS_3[index - TARGET_BASE]
    if USE_CONSUMABLE_BASE <= index < SELL_JOKER_BASE:
        return "use_consumable", index - USE_CONSUMABLE_BASE
    if SELL_JOKER_BASE <= index < BUY_CARD_BASE:
        return "sell_joker", index - SELL_JOKER_BASE
    if BUY_CARD_BASE <= index < BUY_PACK_BASE:
        return "buy_card", index - BUY_CARD_BASE
    if BUY_PACK_BASE <= index < BUY_VOUCHER_IDX:
        return "buy_pack", index - BUY_PACK_BASE
    if index == BUY_VOUCHER_IDX:
        return "buy_voucher", None
    if index == REROLL_IDX:
        return "reroll", None
    if index == LEAVE_SHOP_IDX:
        return "leave_shop", None
    if PICK_PACK_BASE <= index < SKIP_PACK_IDX:
        return "pick_pack", index - PICK_PACK_BASE
    if index == SKIP_PACK_IDX:
        return "skip_pack", None
    return "reserved", None


def to_string(index: int) -> str:
    kind, payload = decode_kind(index)
    if kind in ("play", "discard", "select_targets"):
        return f"{kind}({','.join(map(str, payload))})"
    if payload is not None:
        return f"{kind}[{payload}]"
    return kind


def build_mask(
    engine_actions: list[Action],
    *,
    pending_targets: tuple[int, int, int] | None = None,
    hand_n: int = 0,
) -> np.ndarray:
    """Boolean mask over the 556 actions.

    pending_targets: (consumable_slot, lo, hi) when the env is in the TARGET
    micro-phase — then ONLY SELECT_TARGETS subsets of legal arity are legal.
    Otherwise the mask is the encoding of the engine's legal actions, with
    targeted UseConsumable actions collapsed to their bare USE_CONSUMABLE index.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if pending_targets is not None:
        _, lo, hi = pending_targets
        for i, subset in enumerate(SUBSETS_3):
            k = len(subset)
            if lo <= k <= hi and all(p < hand_n for p in subset):
                mask[TARGET_BASE + i] = True
        return mask

    for a in engine_actions:
        if isinstance(a, UseConsumable) and a.targets:
            # Collapse targeted variants: agent first commits to the consumable,
            # then picks targets in the TARGET micro-phase.
            if a.slot < N_CONSUMABLE_SLOTS:
                mask[USE_CONSUMABLE_BASE + a.slot] = True
            continue
        try:
            mask[encode(a)] = True
        except (KeyError, IndexError):
            continue  # action outside codec capacity (e.g. joker slot > 5)
    return mask
