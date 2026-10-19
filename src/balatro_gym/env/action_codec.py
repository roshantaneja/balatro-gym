"""Discrete action space: the single source of truth for index <-> action.

Layout (fixed global indices, computed from the block sizes below):
    SELECT_BLIND, SKIP_BLIND, REROLL_BOSS
    PLAY_HAND(subset)      all C(HAND_SLOTS, k<=5), ascending bitmask order
    DISCARD(subset)        same subset table
    SELECT_TARGETS(subset) all C(HAND_SLOTS, k<=3); TARGET micro-phase for
                           consumables AND for arcana/spectral pack picks
    USE_CONSUMABLE(slot)   N_CONSUMABLE_SLOTS
    SELL_CONSUMABLE(slot)  N_CONSUMABLE_SLOTS
    SELL_JOKER(slot)       N_SELL_SLOTS
    SWAP_JOKERS(slot)      N_SELL_SLOTS - 1  (swap slot with slot+1)
    BUY_SHOP_CARD(slot)    N_BUY_CARD_SLOTS
    BUY_PACK(slot)         N_BUY_PACK_SLOTS
    BUY_VOUCHER(slot)      N_VOUCHER_SLOTS
    REROLL, LEAVE_SHOP
    PICK_PACK_ITEM(0-4), SKIP_PACK

Hand slots are canonically sorted by the env (rank-desc, suit tiebreak) so
subset indices refer to stable, content-based positions. HAND_SLOTS = 12
covers every vanilla hand size short of stacking several hand-size boosts
(cards beyond the 12 highest-ranked are not selectable through the env).
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
    RerollBoss,
    SelectBlind,
    SellConsumable,
    SellJoker,
    SkipBlind,
    SkipPack,
    SwapJokers,
    UseConsumable,
)

HAND_SLOTS = 12
MAX_PLAY = 5
MAX_TARGETS = 3


def _subsets(max_k: int) -> list[tuple[int, ...]]:
    out = []
    for mask in range(1, 1 << HAND_SLOTS):
        if 1 <= mask.bit_count() <= max_k:
            out.append(tuple(i for i in range(HAND_SLOTS) if mask >> i & 1))
    return out


SUBSETS_5: list[tuple[int, ...]] = _subsets(MAX_PLAY)
SUBSETS_3: list[tuple[int, ...]] = _subsets(MAX_TARGETS)
_SUBSET5_INDEX: dict[tuple[int, ...], int] = {s: i for i, s in enumerate(SUBSETS_5)}
_SUBSET3_INDEX: dict[tuple[int, ...], int] = {s: i for i, s in enumerate(SUBSETS_3)}

N_PLAY = len(SUBSETS_5)
N_TARGET = len(SUBSETS_3)

SELECT_BLIND = 0
SKIP_BLIND = 1
REROLL_BOSS_IDX = 2
PLAY_BASE = 3
DISCARD_BASE = PLAY_BASE + N_PLAY
TARGET_BASE = DISCARD_BASE + N_PLAY
N_CONSUMABLE_SLOTS = 6
USE_CONSUMABLE_BASE = TARGET_BASE + N_TARGET
SELL_CONSUMABLE_BASE = USE_CONSUMABLE_BASE + N_CONSUMABLE_SLOTS
N_SELL_SLOTS = 10
SELL_JOKER_BASE = SELL_CONSUMABLE_BASE + N_CONSUMABLE_SLOTS
SWAP_JOKER_BASE = SELL_JOKER_BASE + N_SELL_SLOTS
N_SWAP_SLOTS = N_SELL_SLOTS - 1
N_BUY_CARD_SLOTS = 4
BUY_CARD_BASE = SWAP_JOKER_BASE + N_SWAP_SLOTS
N_BUY_PACK_SLOTS = 2
BUY_PACK_BASE = BUY_CARD_BASE + N_BUY_CARD_SLOTS
N_VOUCHER_SLOTS = 3
BUY_VOUCHER_BASE = BUY_PACK_BASE + N_BUY_PACK_SLOTS
BUY_VOUCHER_IDX = BUY_VOUCHER_BASE  # first voucher slot (compat)
REROLL_IDX = BUY_VOUCHER_BASE + N_VOUCHER_SLOTS
LEAVE_SHOP_IDX = REROLL_IDX + 1
PICK_PACK_BASE = LEAVE_SHOP_IDX + 1
N_PACK_OPTIONS = 5
SKIP_PACK_IDX = PICK_PACK_BASE + N_PACK_OPTIONS
N_ACTIONS = SKIP_PACK_IDX + 1


def encode(action: Action) -> int:
    """EngineAction -> global index. Raises KeyError for unencodable actions."""
    match action:
        case SelectBlind():
            return SELECT_BLIND
        case SkipBlind():
            return SKIP_BLIND
        case RerollBoss():
            return REROLL_BOSS_IDX
        case PlayHand(indices=idxs):
            return PLAY_BASE + _SUBSET5_INDEX[tuple(sorted(idxs))]
        case Discard(indices=idxs):
            return DISCARD_BASE + _SUBSET5_INDEX[tuple(sorted(idxs))]
        case UseConsumable(slot=s, targets=()) if s < N_CONSUMABLE_SLOTS:
            return USE_CONSUMABLE_BASE + s
        case SellConsumable(slot=s) if s < N_CONSUMABLE_SLOTS:
            return SELL_CONSUMABLE_BASE + s
        case SellJoker(slot=s) if s < N_SELL_SLOTS:
            return SELL_JOKER_BASE + s
        case SwapJokers(slot=s) if s < N_SWAP_SLOTS:
            return SWAP_JOKER_BASE + s
        case BuyShopCard(slot=s) if s < N_BUY_CARD_SLOTS:
            return BUY_CARD_BASE + s
        case BuyPack(slot=s) if s < N_BUY_PACK_SLOTS:
            return BUY_PACK_BASE + s
        case BuyVoucher(slot=s) if s < N_VOUCHER_SLOTS:
            return BUY_VOUCHER_BASE + s
        case Reroll():
            return REROLL_IDX
        case LeaveShop():
            return LEAVE_SHOP_IDX
        case PickPackItem(index=i, targets=()) if i < N_PACK_OPTIONS:
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
    if index == REROLL_BOSS_IDX:
        return "reroll_boss", None
    if PLAY_BASE <= index < DISCARD_BASE:
        return "play", SUBSETS_5[index - PLAY_BASE]
    if DISCARD_BASE <= index < TARGET_BASE:
        return "discard", SUBSETS_5[index - DISCARD_BASE]
    if TARGET_BASE <= index < USE_CONSUMABLE_BASE:
        return "select_targets", SUBSETS_3[index - TARGET_BASE]
    if USE_CONSUMABLE_BASE <= index < SELL_CONSUMABLE_BASE:
        return "use_consumable", index - USE_CONSUMABLE_BASE
    if SELL_CONSUMABLE_BASE <= index < SELL_JOKER_BASE:
        return "sell_consumable", index - SELL_CONSUMABLE_BASE
    if SELL_JOKER_BASE <= index < SWAP_JOKER_BASE:
        return "sell_joker", index - SELL_JOKER_BASE
    if SWAP_JOKER_BASE <= index < BUY_CARD_BASE:
        return "swap_jokers", index - SWAP_JOKER_BASE
    if BUY_CARD_BASE <= index < BUY_PACK_BASE:
        return "buy_card", index - BUY_CARD_BASE
    if BUY_PACK_BASE <= index < BUY_VOUCHER_BASE:
        return "buy_pack", index - BUY_PACK_BASE
    if BUY_VOUCHER_BASE <= index < REROLL_IDX:
        return "buy_voucher", index - BUY_VOUCHER_BASE
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
    pending_targets: tuple[int, int] | None = None,
    hand_n: int = 0,
) -> np.ndarray:
    """Boolean mask over the N_ACTIONS actions.

    pending_targets: (lo, hi) when the env is in the TARGET micro-phase —
    then ONLY SELECT_TARGETS subsets of legal arity are legal. Otherwise the
    mask is the encoding of the engine's legal actions, with targeted
    UseConsumable / PickPackItem actions collapsed to their bare index (the
    agent commits to the card first, then picks targets).
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if pending_targets is not None:
        lo, hi = pending_targets
        for i, subset in enumerate(SUBSETS_3):
            k = len(subset)
            if lo <= k <= hi and all(p < hand_n for p in subset):
                mask[TARGET_BASE + i] = True
        return mask

    for a in engine_actions:
        if isinstance(a, UseConsumable) and a.targets:
            if a.slot < N_CONSUMABLE_SLOTS:
                mask[USE_CONSUMABLE_BASE + a.slot] = True
            continue
        if isinstance(a, PickPackItem) and a.targets:
            if a.index < N_PACK_OPTIONS:
                mask[PICK_PACK_BASE + a.index] = True
            continue
        try:
            mask[encode(a)] = True
        except (KeyError, IndexError):
            continue  # action outside codec capacity (e.g. hand slot > 11)
    return mask
