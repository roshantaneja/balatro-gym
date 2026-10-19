"""Question / answer schema for the typed decision model.

A *question* names one decision and fixes its answer type; an *answer* is a
typed value with the model's calibrated confidence. Two questions cover the
whole game:

- ``next_action`` (enum): which of the currently legal actions to take. The
  answer value is the action's index in the flat action space and its label
  is a readable description (``"play 7♠ 7♥"``, ``"buy joker j_duo for $8"``).
- ``will_win`` (bool): whether this run will clear ante 8 from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from balatro_gym.engine import cardops
from balatro_gym.env import action_codec as codec

_SUITS = {0: "♠", 1: "♥", 2: "♣", 3: "♦"}
_RANKS = {11: "J", 12: "Q", 13: "K", 14: "A"}


@dataclass(frozen=True)
class Question:
    name: str
    description: str
    answer_type: str  # "enum" | "bool"


@dataclass(frozen=True)
class Answer:
    question: str
    value: Any
    confidence: float
    label: str = ""


NEXT_ACTION = Question("next_action", "Which legal action should be taken now?", "enum")
WILL_WIN = Question("will_win", "Will this run clear the ante-8 boss blind?", "bool")


def questions_for(env) -> list[Question]:
    """The questions that apply to the env's current phase."""
    if env.run.phase.value in ("won", "game_over"):
        return []
    return [NEXT_ACTION, WILL_WIN]


def _card_label(card) -> str:
    if card.is_stone:
        return "stone"
    r = int(card.rank)
    rank = _RANKS.get(r, str(r))
    suit = _SUITS.get(int(card.suit), "?")
    return f"{rank}{suit}"


def _slot_cards(env, slots: tuple[int, ...]) -> str:
    hand = cardops.current_hand(env.run)
    order = env._hand_order
    names = []
    for s in slots:
        if s < len(order) and order[s] < len(hand):
            names.append(_card_label(hand[order[s]]))
    return " ".join(names)


def describe_action(env, index: int) -> str:
    """Readable label for a flat action index in the env's current state."""
    run = env.run
    kind, payload = codec.decode_kind(index)
    shop = run.shop
    try:
        if kind == "play":
            return f"play {_slot_cards(env, payload)}"
        if kind == "discard":
            return f"discard {_slot_cards(env, payload)}"
        if kind == "select_targets":
            return f"target {_slot_cards(env, payload)}"
        if kind == "use_consumable":
            return f"use {run.consumables[payload].id}"
        if kind == "sell_consumable":
            return f"sell {run.consumables[payload].id}"
        if kind == "sell_joker":
            return f"sell joker {run.jokers[payload].id}"
        if kind == "swap_jokers":
            return f"swap jokers {payload} and {payload + 1}"
        if kind == "buy_card" and shop is not None:
            it = shop.card_slots[payload]
            return f"buy {it.kind} {it.item_id} for ${it.cost}"
        if kind == "buy_pack" and shop is not None:
            offer = shop.pack_slots[payload]
            return f"buy {offer.pack_id} for ${offer.cost}"
        if kind == "buy_voucher" and shop is not None:
            return f"buy voucher {shop.vouchers[payload]}"
        if kind == "reroll" and shop is not None:
            return f"reroll the shop for ${shop.reroll_cost}"
        if kind == "pick_pack" and run.open_pack is not None:
            pack = run.open_pack
            what = pack.option_ids[payload] if payload < len(pack.option_ids) else "?"
            return f"pick {pack.option_kinds[payload]} {what}"
        if kind == "skip_blind":
            tag = run.blind_tags[run.blind_index] if run.blind_index < len(run.blind_tags) else None
            return f"skip the blind for {tag.id if tag else 'a tag'}"
    except (IndexError, AttributeError):
        pass
    return kind.replace("_", " ")
