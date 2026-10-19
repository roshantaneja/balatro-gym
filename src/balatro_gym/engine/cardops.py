"""Shared mutations and queries used by content and orchestrators.

Everything that creates/destroys cards, jokers or consumables goes through
here so the joker hooks that watch those events (Hologram, Glass Joker,
Canio, Campfire, ...) fire from exactly one place.
"""

from __future__ import annotations

import random

from .cards import Card
from .effects import (
    CONSUMABLES,
    JOKERS,
    active_jokers,
    joker_def,
    new_joker_state,
)
from .state import ConsumableInst, JokerInst, RunState
from .types import (
    PERISHABLE_ROUNDS,
    STICKER_ETERNAL,
    STICKER_PERISHABLE,
    ConsumableKind,
    Edition,
    Enhancement,
    Rarity,
)

EDITION_SURCHARGE = {Edition.FOIL: 2, Edition.HOLO: 3, Edition.POLY: 5, Edition.NEGATIVE: 5}


# ---------------------------------------------------------------------------
# Derived numbers
# ---------------------------------------------------------------------------

def effective_hand_size(run: RunState) -> int:
    n = run.hand_size + sum(
        joker_def(j.id).effect.hand_size_mod(run, j) for j in active_jokers(run)
    )
    if run.round is not None:
        n += run.round.hand_size_bonus
    return max(0, n)


def effective_hands(run: RunState) -> int:
    n = run.hands_per_round + sum(
        joker_def(j.id).effect.hands_mod(run, j) for j in active_jokers(run)
    )
    return max(1, n)


def effective_discards(run: RunState) -> int:
    n = run.discards_per_round + sum(
        joker_def(j.id).effect.discards_mod(run, j) for j in active_jokers(run)
    )
    return max(0, n)


def debt_allowance(run: RunState) -> int:
    return sum(joker_def(j.id).effect.debt_allowance(run, j) for j in active_jokers(run))


def can_afford(run: RunState, cost: int) -> bool:
    return run.money + debt_allowance(run) >= cost


def refresh_chance_multiplier(run: RunState) -> None:
    m = 1
    for j in active_jokers(run):
        m *= joker_def(j.id).effect.chance_multiplier(run, j)
    run.chance_multiplier = m


def allows_duplicates(run: RunState) -> bool:
    return any(joker_def(j.id).effect.allows_duplicates(run, j) for j in active_jokers(run))


def item_is_free(run: RunState, kind: str, item_id: str) -> bool:
    return any(
        joker_def(j.id).effect.item_is_free(run, j, kind, item_id) for j in active_jokers(run)
    )


def chance(run: RunState, num: int, den: int, stream: str = "chance") -> bool:
    """Seeded probability roll; the chance multiplier (Oops! All 6s) scales num."""
    eff = min(1.0, (num * run.chance_multiplier) / den)
    return run.rng.stream(stream).random() < eff


# ---------------------------------------------------------------------------
# Hand access (round hand, or the hand dealt by an open Arcana/Spectral pack)
# ---------------------------------------------------------------------------

def current_hand(run: RunState) -> list[Card]:
    if run.open_pack is not None and run.open_pack.hand:
        return run.open_pack.hand
    if run.round is not None:
        return run.round.hand
    return []


# ---------------------------------------------------------------------------
# Playing cards
# ---------------------------------------------------------------------------

def add_card(run: RunState, card: Card, *, to_hand: bool = True) -> Card:
    """Add a brand-new card to the deck (and to the current hand when one is
    visible, matching the game's 'added to hand' behaviour for spectrals)."""
    run.full_deck.append(card)
    if to_hand:
        if run.open_pack is not None and run.open_pack.hand:
            run.open_pack.hand.append(card)
        elif run.round is not None:
            run.round.hand.append(card)
    for j in active_jokers(run):
        joker_def(j.id).effect.update_on_card_added(run, j, card)
    return card


def copy_card(run: RunState, src: Card) -> Card:
    c = src.clone()
    c.uid = run.new_uid()
    return c


def destroy_cards(run: RunState, cards: list[Card]) -> None:
    """Remove cards from the deck and every round/pack container."""
    if not cards:
        return
    gone = {c.uid for c in cards}
    run.full_deck = [c for c in run.full_deck if c.uid not in gone]
    rnd = run.round
    if rnd is not None:
        rnd.hand = [c for c in rnd.hand if c.uid not in gone]
        rnd.deck_remaining = [c for c in rnd.deck_remaining if c.uid not in gone]
        rnd.discarded = [c for c in rnd.discarded if c.uid not in gone]
        rnd.face_down_uids = [u for u in rnd.face_down_uids if u not in gone]
    if run.open_pack is not None:
        run.open_pack.hand = [c for c in run.open_pack.hand if c.uid not in gone]
    run.stats["cards_destroyed"] = run.stats.get("cards_destroyed", 0) + len(cards)
    for card in cards:
        if card.enhancement == Enhancement.GLASS:
            run.stats["glass_broken"] = run.stats.get("glass_broken", 0) + 1
        for j in active_jokers(run):
            joker_def(j.id).effect.update_on_card_destroyed(run, j, card)


# ---------------------------------------------------------------------------
# Jokers
# ---------------------------------------------------------------------------

def joker_pool(run: RunState, rarity: Rarity | None = None, *, legendary: bool = False) -> list[str]:
    """Shop/pack/creation pool: unowned (unless Showman), rarity-filtered,
    pool-flag-filtered (Gros Michel extinct -> Cavendish)."""
    owned = set() if allows_duplicates(run) else {j.id for j in run.jokers}
    out = []
    for jid, d in JOKERS.items():
        if rarity is not None and d.rarity != rarity:
            continue
        if rarity is None and d.rarity == Rarity.LEGENDARY and not legendary:
            continue
        if jid in owned:
            continue
        if d.pool_requires and not run.stats.get(d.pool_requires):
            continue
        if d.pool_excludes and run.stats.get(d.pool_excludes):
            continue
        out.append(jid)
    return sorted(out)


def random_joker_id(
    run: RunState, rarity: Rarity | None = None, stream: str = "misc", *, legendary: bool = False
) -> str | None:
    pool = joker_pool(run, rarity, legendary=legendary)
    if not pool and rarity is not None:
        pool = joker_pool(run, None)
    if not pool:
        return None
    return run.rng.stream(stream).choice(pool)


def has_joker_room(run: RunState, edition: Edition = Edition.NONE) -> bool:
    if edition == Edition.NEGATIVE:
        return True
    return len(run.jokers) < run.joker_slots


def new_joker(
    run: RunState,
    jid: str,
    *,
    edition: Edition = Edition.NONE,
    stickers: list[str] | tuple[str, ...] = (),
    cost: int | None = None,
) -> JokerInst:
    base = JOKERS[jid].base_cost if cost is None else cost
    j = JokerInst(
        id=jid,
        uid=run.new_uid(),
        edition=edition,
        stickers=frozenset(stickers),
        sell_value=max(1, base // 2),
        state=new_joker_state(jid),
    )
    if STICKER_PERISHABLE in j.stickers:
        j.perish_left = PERISHABLE_ROUNDS
    return j


def add_joker(run: RunState, j: JokerInst, *, index: int | None = None) -> JokerInst:
    """Place a joker instance on the board and fire acquisition hooks."""
    if index is None:
        run.jokers.append(j)
    else:
        run.jokers.insert(index, j)
    if j.edition == Edition.NEGATIVE:
        run.joker_slots += 1
    joker_def(j.id).effect.on_acquired(run, j)
    refresh_chance_multiplier(run)
    return j


def create_joker(
    run: RunState, jid: str, *, edition: Edition = Edition.NONE, stream: str = "misc"
) -> JokerInst | None:
    """Spawn a joker if there is room (Riff-Raff, Judgement, The Soul, ...)."""
    if not has_joker_room(run, edition):
        return None
    return add_joker(run, new_joker(run, jid, edition=edition))


def remove_joker(run: RunState, j: JokerInst, *, sold: bool = False) -> None:
    """Remove a joker from the board (rebinding run.jokers, see content/jokers
    module docstrings) and fire the removal hooks."""
    if j.uid not in {x.uid for x in run.jokers}:
        return
    run.jokers = [x for x in run.jokers if x.uid != j.uid]
    if j.edition == Edition.NEGATIVE:
        run.joker_slots -= 1
    eff = joker_def(j.id).effect
    if sold:
        eff.on_sold(run, j)
    eff.on_removed(run, j)
    refresh_chance_multiplier(run)


def destroy_joker(run: RunState, j: JokerInst) -> bool:
    """Destroy (not sell) a joker; Eternal jokers refuse."""
    if STICKER_ETERNAL in j.stickers:
        return False
    remove_joker(run, j, sold=False)
    run.stats["jokers_destroyed"] = run.stats.get("jokers_destroyed", 0) + 1
    return True


def flush_pending_destroys(run: RunState) -> None:
    """Remove jokers that asked to self-destruct during scoring."""
    if not run.pending_destroy:
        return
    uids = set(run.pending_destroy)
    run.pending_destroy = []
    for j in list(run.jokers):
        if j.uid in uids:
            destroy_joker(run, j)


# ---------------------------------------------------------------------------
# Consumables
# ---------------------------------------------------------------------------

def consumables_used_slots(run: RunState) -> int:
    return sum(1 for c in run.consumables if c.edition != Edition.NEGATIVE)


def has_consumable_room(run: RunState, edition: Edition = Edition.NONE) -> bool:
    if edition == Edition.NEGATIVE:
        return True
    return consumables_used_slots(run) < run.consumable_slots


def consumable_pool(run: RunState, kind: ConsumableKind) -> list[str]:
    owned = set() if allows_duplicates(run) else {c.id for c in run.consumables}
    out = []
    for cid, d in CONSUMABLES.items():
        if d.kind != kind or not d.in_pool or cid in owned:
            continue
        if kind == ConsumableKind.PLANET and not planet_discovered(run, cid):
            continue
        out.append(cid)
    return sorted(out)


def planet_discovered(run: RunState, cid: str) -> bool:
    """Secret-hand planets only appear once their hand has been played."""
    from .effects import PLANET_FOR_HAND

    for hand, planet_id in PLANET_FOR_HAND.items():
        if planet_id == cid:
            return hand in run.hands_discovered
    return True


def random_consumable_id(run: RunState, kind: ConsumableKind, stream: str = "consumable") -> str | None:
    pool = consumable_pool(run, kind)
    if not pool:
        return None
    return run.rng.stream(stream).choice(pool)


def new_consumable(run: RunState, cid: str, edition: Edition = Edition.NONE) -> ConsumableInst:
    d = CONSUMABLES[cid]
    return ConsumableInst(
        id=cid, kind=d.kind, uid=run.new_uid(), edition=edition,
        sell_value=max(1, d.base_cost // 2),
    )


def create_consumable(
    run: RunState, cid: str, *, edition: Edition = Edition.NONE
) -> ConsumableInst | None:
    """Give the player a consumable if there is room (8 Ball, seals, tags, ...)."""
    if not has_consumable_room(run, edition):
        return None
    inst = new_consumable(run, cid, edition)
    run.consumables.append(inst)
    return inst


def create_random_consumable(
    run: RunState, kind: ConsumableKind, *, stream: str = "consumable", edition: Edition = Edition.NONE
) -> ConsumableInst | None:
    if not has_consumable_room(run, edition):
        return None
    cid = random_consumable_id(run, kind, stream)
    if cid is None:
        return None
    return create_consumable(run, cid, edition=edition)


# ---------------------------------------------------------------------------
# Vouchers
# ---------------------------------------------------------------------------

def grant_voucher(run: RunState, vid: str) -> None:
    """Give the run a voucher outright (deck starts, Magic/Nebula/Zodiac)."""
    from .effects import VOUCHERS

    if vid in run.vouchers:
        return
    run.vouchers = run.vouchers | {vid}
    d = VOUCHERS[vid]
    if d.apply:
        d.apply(run)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Misc random helpers
# ---------------------------------------------------------------------------

def random_deck_card(run: RunState, stream: str = "misc") -> Card | None:
    """A random non-stone card from the full deck (rank/suit re-rolls)."""
    pool = [c for c in run.full_deck if not c.is_stone]
    if not pool:
        return None
    return run.rng.stream(stream).choice(pool)


def rng(run: RunState, stream: str = "misc") -> random.Random:
    return run.rng.stream(stream)
