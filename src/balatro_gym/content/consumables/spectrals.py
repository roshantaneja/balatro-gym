"""All 18 Spectral cards ($4 each).

Hand-affecting spectrals operate on the visible hand: the round hand during
a blind, or the hand dealt by an open Spectral pack (engine.cardops.current_hand).

Rules honoured here:
- Familiar / Grim / Incantation destroy 1 random card in hand and add 3
  enhanced face cards / 2 enhanced Aces / 4 enhanced numbered cards to the
  hand (random enhancement other than Stone, random suit).
- Talisman / Deja Vu / Trance / Medium add Gold / Red / Blue / Purple seals.
- Aura adds a random edition (Foil / Holographic / Polychrome) to 1 card.
- Wraith creates a Rare joker and sets money to $0.
- Sigil / Ouija convert every card in hand to one random suit / rank; Ouija
  also costs 1 hand size permanently.
- Ectoplasm adds Negative to a random joker and costs hand size: 1 for the
  first use, 2 for the second, and so on.
- Immolate destroys 5 random cards in hand for $20.
- Ankh copies a random joker and destroys the rest; Hex makes a random joker
  Polychrome and destroys the rest (Eternal jokers survive both).
- Cryptid adds 2 copies of the selected card.
- The Soul creates a Legendary joker; Black Hole levels every hand.
- The Soul and Black Hole are never in shops; they replace a pack option
  with a 0.3% chance (Soul: Arcana/Spectral packs, Black Hole: Celestial/
  Spectral packs).
"""

from __future__ import annotations

from balatro_gym.engine import cardops
from balatro_gym.engine.cards import Card
from balatro_gym.engine.effects import ConsumableDef, ConsumableEffect, register_consumable
from balatro_gym.engine.state import RunState
from balatro_gym.engine.types import (
    STICKER_ETERNAL,
    ConsumableKind,
    Edition,
    Enhancement,
    HandType,
    Rank,
    Rarity,
    Seal,
    Suit,
)

SPECTRAL_BASE_COST = 4


def _spectral(cid: str, name: str, effect: ConsumableEffect, **kw) -> None:
    register_consumable(
        ConsumableDef(
            id=cid, name=name, kind=ConsumableKind.SPECTRAL,
            base_cost=SPECTRAL_BASE_COST, effect=effect, **kw,
        )
    )


_ENHANCEMENTS = [e for e in Enhancement if e not in (Enhancement.NONE, Enhancement.STONE)]
_FACE_RANKS = [Rank.JACK, Rank.QUEEN, Rank.KING]
_NUMBER_RANKS = [r for r in Rank if int(r) <= 10]


def _hand_nonempty(run: RunState) -> bool:
    return len(cardops.current_hand(run)) > 0


# ---------------------------------------------------------------------------
# Destroy-and-create (Familiar / Grim / Incantation)
# ---------------------------------------------------------------------------

class ReplaceEffect(ConsumableEffect):
    def __init__(self, ranks: list[Rank], n: int):
        self.ranks = ranks
        self.n = n

    def can_use(self, run: RunState) -> bool:
        return _hand_nonempty(run)

    def use(self, run: RunState, targets: list[Card]) -> None:
        rng = run.rng.stream("consumable")
        hand = cardops.current_hand(run)
        if hand:
            cardops.destroy_cards(run, [rng.choice(hand)])
        for _ in range(self.n):
            card = Card(
                uid=run.new_uid(),
                rank=rng.choice(self.ranks),
                suit=rng.choice(list(Suit)),
                enhancement=rng.choice(_ENHANCEMENTS),
            )
            cardops.add_card(run, card, to_hand=True)


_spectral("c_familiar", "Familiar", ReplaceEffect(_FACE_RANKS, 3))
_spectral("c_grim", "Grim", ReplaceEffect([Rank.ACE], 2))
_spectral("c_incantation", "Incantation", ReplaceEffect(_NUMBER_RANKS, 4))


# ---------------------------------------------------------------------------
# Seals and editions on one selected card
# ---------------------------------------------------------------------------

class SealEffect(ConsumableEffect):
    target_arity = (1, 1)

    def __init__(self, seal: Seal):
        self.seal = seal

    def use(self, run: RunState, targets: list[Card]) -> None:
        targets[0].seal = self.seal


_spectral("c_talisman", "Talisman", SealEffect(Seal.GOLD))
_spectral("c_deja_vu", "Deja Vu", SealEffect(Seal.RED))
_spectral("c_trance", "Trance", SealEffect(Seal.BLUE))
_spectral("c_medium", "Medium", SealEffect(Seal.PURPLE))


class AuraEffect(ConsumableEffect):
    target_arity = (1, 1)

    def use(self, run: RunState, targets: list[Card]) -> None:
        targets[0].edition = run.rng.stream("consumable").choice(
            [Edition.FOIL, Edition.HOLO, Edition.POLY]
        )


_spectral("c_aura", "Aura", AuraEffect())


class CryptidEffect(ConsumableEffect):
    target_arity = (1, 1)

    def use(self, run: RunState, targets: list[Card]) -> None:
        for _ in range(2):
            cardops.add_card(run, cardops.copy_card(run, targets[0]), to_hand=True)


_spectral("c_cryptid", "Cryptid", CryptidEffect())


# ---------------------------------------------------------------------------
# Whole-hand converters
# ---------------------------------------------------------------------------

class SigilEffect(ConsumableEffect):
    def can_use(self, run: RunState) -> bool:
        return _hand_nonempty(run)

    def use(self, run: RunState, targets: list[Card]) -> None:
        suit = run.rng.stream("consumable").choice(list(Suit))
        for c in cardops.current_hand(run):
            c.suit = suit


_spectral("c_sigil", "Sigil", SigilEffect())


class OuijaEffect(ConsumableEffect):
    def can_use(self, run: RunState) -> bool:
        return _hand_nonempty(run)

    def use(self, run: RunState, targets: list[Card]) -> None:
        rank = run.rng.stream("consumable").choice(list(Rank))
        for c in cardops.current_hand(run):
            c.rank = rank
        run.hand_size -= 1


_spectral("c_ouija", "Ouija", OuijaEffect())


class ImmolateEffect(ConsumableEffect):
    def can_use(self, run: RunState) -> bool:
        return _hand_nonempty(run)

    def use(self, run: RunState, targets: list[Card]) -> None:
        hand = list(cardops.current_hand(run))
        rng = run.rng.stream("consumable")
        victims = rng.sample(hand, min(5, len(hand)))
        cardops.destroy_cards(run, victims)
        run.money += 20


_spectral("c_immolate", "Immolate", ImmolateEffect())


# ---------------------------------------------------------------------------
# Joker spectrals
# ---------------------------------------------------------------------------

class WraithEffect(ConsumableEffect):
    def can_use(self, run: RunState) -> bool:
        return cardops.has_joker_room(run) and bool(cardops.joker_pool(run, Rarity.RARE))

    def use(self, run: RunState, targets: list[Card]) -> None:
        jid = cardops.random_joker_id(run, Rarity.RARE, stream="consumable")
        if jid is not None:
            cardops.create_joker(run, jid)
        run.money = 0


_spectral("c_wraith", "Wraith", WraithEffect())


class EctoplasmEffect(ConsumableEffect):
    def can_use(self, run: RunState) -> bool:
        return bool(run.jokers)

    def use(self, run: RunState, targets: list[Card]) -> None:
        j = run.rng.stream("consumable").choice(run.jokers)
        if j.edition != Edition.NEGATIVE:
            j.edition = Edition.NEGATIVE
            run.joker_slots += 1
        uses = run.stats.get("ectoplasm_used", 0)
        run.hand_size -= 1 + uses
        run.stats["ectoplasm_used"] = uses + 1


_spectral("c_ectoplasm", "Ectoplasm", EctoplasmEffect())


def _destroy_others(run: RunState, keep_uid: int) -> None:
    for j in list(run.jokers):
        if j.uid != keep_uid and STICKER_ETERNAL not in j.stickers:
            cardops.destroy_joker(run, j)


class AnkhEffect(ConsumableEffect):
    def can_use(self, run: RunState) -> bool:
        return bool(run.jokers)

    def use(self, run: RunState, targets: list[Card]) -> None:
        src = run.rng.stream("consumable").choice(run.jokers)
        _destroy_others(run, src.uid)
        copy = cardops.new_joker(run, src.id, edition=src.edition if src.edition != Edition.NEGATIVE else Edition.NONE)
        copy.state = dict(src.state)
        copy.sell_value = src.sell_value
        cardops.add_joker(run, copy)


_spectral("c_ankh", "Ankh", AnkhEffect())


class HexEffect(ConsumableEffect):
    def can_use(self, run: RunState) -> bool:
        return bool(run.jokers)

    def use(self, run: RunState, targets: list[Card]) -> None:
        keep = run.rng.stream("consumable").choice(run.jokers)
        _destroy_others(run, keep.uid)
        keep.edition = Edition.POLY


_spectral("c_hex", "Hex", HexEffect())


class SoulEffect(ConsumableEffect):
    def can_use(self, run: RunState) -> bool:
        return cardops.has_joker_room(run) and bool(
            cardops.joker_pool(run, Rarity.LEGENDARY, legendary=True)
        )

    def use(self, run: RunState, targets: list[Card]) -> None:
        jid = cardops.random_joker_id(run, Rarity.LEGENDARY, stream="consumable", legendary=True)
        if jid is not None:
            cardops.create_joker(run, jid)


_spectral("c_soul", "The Soul", SoulEffect(), in_pool=False, pool_weight=0.003,
          hidden_in=("arcana", "spectral"))


class BlackHoleEffect(ConsumableEffect):
    def use(self, run: RunState, targets: list[Card]) -> None:
        for ht in HandType:
            run.hand_levels[ht] += 1


_spectral("c_black_hole", "Black Hole", BlackHoleEffect(), in_pool=False, pool_weight=0.003,
          hidden_in=("celestial", "spectral"))
