"""Shop generation, reroll, buy/sell, booster packs.

Numbers from The Shop.wiki / Booster Packs.wiki:
- 2 card slots (weights Joker 20 / Tarot 4 / Planet 4), 2 pack slots, 1 voucher.
- Reroll $5, +$1 per reroll, resets each shop; rerolls only the card slots.
- Joker rarity roll: 70% common / 25% uncommon / 5% rare.
- Sell value = floor(cost / 2), min $1.
- First shop of a run always offers a Buffoon pack.
"""

from __future__ import annotations

from .cards import Card
from .effects import CONSUMABLES, JOKERS, PACKS, VOUCHERS, joker_def, new_joker_state
from .events import Event, ItemBought, JokerSold
from .levels import BASE_REROLL_COST
from .state import ConsumableInst, JokerInst, OpenPack, PackOffer, RunState, ShopItem, ShopState
from .types import ConsumableKind, Edition, Enhancement, Phase, Rank, Rarity, Seal, Suit

CARD_KIND_WEIGHTS = [("joker", 20.0), ("tarot", 4.0), ("planet", 4.0)]
RARITY_WEIGHTS = [(Rarity.COMMON, 70.0), (Rarity.UNCOMMON, 25.0), (Rarity.RARE, 5.0)]
EDITION_SURCHARGE = {Edition.FOIL: 2, Edition.HOLO: 3, Edition.POLY: 5, Edition.NEGATIVE: 5}
N_PACK_SLOTS = 2


def n_card_slots(run: RunState) -> int:
    return run.stats.get("shop_card_slots", 2)  # Overstock voucher raises this


def _category_has_content(category: str) -> bool:
    if category == "standard":
        return True
    if category == "buffoon":
        return bool(JOKERS)
    want = {
        "celestial": ConsumableKind.PLANET,
        "arcana": ConsumableKind.TAROT,
        "spectral": ConsumableKind.SPECTRAL,
    }[category]
    return any(c.kind == want for c in CONSUMABLES.values())


def sell_value(cost: int) -> int:
    return max(1, cost // 2)


def _weighted_choice(rng, pairs):
    total = sum(w for _, w in pairs)
    x = rng.random() * total
    for item, w in pairs:
        x -= w
        if x <= 0:
            return item
    return pairs[-1][0]


def _roll_joker_edition(run: RunState) -> Edition:
    # Base shop edition rates (Hone voucher doubles them — seam for M8).
    r = run.rng.stream("edition").random()
    if r < 0.02:
        return Edition.FOIL
    if r < 0.034:
        return Edition.HOLO
    if r < 0.040:
        return Edition.POLY
    return Edition.NONE


def _roll_card_slot(run: RunState) -> ShopItem | None:
    rng = run.rng.stream("shop")
    kinds = [
        (k, w)
        for k, w in CARD_KIND_WEIGHTS
        if (k == "joker" and JOKERS)
        or (k == "tarot" and any(c.kind == ConsumableKind.TAROT for c in CONSUMABLES.values()))
        or (k == "planet" and any(c.kind == ConsumableKind.PLANET for c in CONSUMABLES.values()))
    ]
    if not kinds:
        return None
    kind = _weighted_choice(rng, kinds)

    if kind == "joker":
        owned = {j.id for j in run.jokers}
        rarity = _weighted_choice(rng, RARITY_WEIGHTS)
        pool = sorted(
            jid
            for jid, d in JOKERS.items()
            if d.rarity == rarity and jid not in owned
        )
        if not pool:  # fall back to any rarity
            pool = sorted(jid for jid, d in JOKERS.items() if d.rarity != Rarity.LEGENDARY and jid not in owned)
        if not pool:
            return None
        jid = rng.choice(pool)
        edition = _roll_joker_edition(run)
        cost = JOKERS[jid].base_cost + EDITION_SURCHARGE.get(edition, 0)
        return ShopItem("joker", jid, cost, edition)

    want = ConsumableKind.TAROT if kind == "tarot" else ConsumableKind.PLANET
    pool = sorted(
        cid
        for cid, d in CONSUMABLES.items()
        if d.kind == want and _planet_discovered(run, cid)
    )
    if not pool:
        return None
    cid = rng.choice(pool)
    return ShopItem(kind, cid, CONSUMABLES[cid].base_cost)


def _planet_discovered(run: RunState, cid: str) -> bool:
    """Secret-hand planets only appear once their hand has been played."""
    from .effects import PLANET_FOR_HAND

    for hand, planet_id in PLANET_FOR_HAND.items():
        if planet_id == cid:
            return hand in run.hands_discovered
    return True  # non-planets and unmapped ids are unrestricted


def _roll_pack_offer(run: RunState, force_buffoon: bool = False) -> PackOffer | None:
    offerable = [
        (pid, d) for pid, d in sorted(PACKS.items()) if _category_has_content(d.category)
    ]
    if not offerable:
        return None
    rng = run.rng.stream("shop")
    if force_buffoon:
        buffoons = sorted(pid for pid, d in offerable if d.category == "buffoon" and "normal" in pid)
        if buffoons:
            pid = buffoons[0]
            return PackOffer(pid, PACKS[pid].cost)
    pid = _weighted_choice(rng, [(pid, d.weight) for pid, d in offerable])
    return PackOffer(pid, PACKS[pid].cost)


def _roll_voucher(run: RunState) -> str | None:
    pool = sorted(
        vid
        for vid, d in VOUCHERS.items()
        if vid not in run.vouchers and (d.requires is None or d.requires in run.vouchers)
    )
    if not pool:
        return None
    return run.rng.stream("shop").choice(pool)


def open_shop(run: RunState) -> None:
    first_shop = run.stats.get("shops_opened", 0) == 0
    run.stats["shops_opened"] = run.stats.get("shops_opened", 0) + 1

    card_slots = [s for s in (_roll_card_slot(run) for _ in range(n_card_slots(run))) if s]
    pack_slots = []
    for i in range(N_PACK_SLOTS):
        offer = _roll_pack_offer(run, force_buffoon=(first_shop and i == 0))
        if offer:
            pack_slots.append(offer)

    run.shop = ShopState(
        card_slots=card_slots,
        pack_slots=pack_slots,
        voucher=_roll_voucher(run),
        reroll_cost=BASE_REROLL_COST,
    )
    run.phase = Phase.SHOP


def reroll(run: RunState) -> list[Event]:
    shop = run.shop
    assert shop is not None
    run.money -= shop.reroll_cost
    shop.reroll_cost += 1
    shop.card_slots = [s for s in (_roll_card_slot(run) for _ in range(n_card_slots(run))) if s]
    return []


def buy_shop_card(run: RunState, slot: int) -> list[Event]:
    shop = run.shop
    assert shop is not None
    item = shop.card_slots.pop(slot)
    run.money -= item.cost
    if item.kind == "joker":
        j = JokerInst(
            id=item.item_id,
            uid=run.new_uid(),
            edition=item.edition,
            sell_value=sell_value(item.cost),
            state=new_joker_state(item.item_id),
        )
        run.jokers.append(j)
        joker_def(j.id).effect.on_acquired(run, j)
        if item.edition == Edition.NEGATIVE:
            run.joker_slots += 1
    else:
        kind = ConsumableKind.TAROT if item.kind == "tarot" else ConsumableKind.PLANET
        run.consumables.append(ConsumableInst(id=item.item_id, kind=kind, uid=run.new_uid()))
    return [ItemBought(item.kind, item.item_id, item.cost)]


def sell_joker(run: RunState, slot: int) -> list[Event]:
    j = run.jokers.pop(slot)
    value = j.sell_value
    run.money += value
    joker_def(j.id).effect.on_sold(run, j)
    if j.edition == Edition.NEGATIVE:
        run.joker_slots -= 1
    return [JokerSold(j.id, value)]


def buy_voucher(run: RunState) -> list[Event]:
    shop = run.shop
    assert shop is not None and shop.voucher is not None
    vid = shop.voucher
    defn = VOUCHERS[vid]
    run.money -= defn.cost
    run.vouchers = run.vouchers | {vid}
    shop.voucher = None
    if defn.apply:
        defn.apply(run)  # type: ignore[operator]
    return [ItemBought("voucher", vid, defn.cost)]


# ---------------------------------------------------------------------------
# Booster packs
# ---------------------------------------------------------------------------

_PACK_CARD_SUITS = list(Suit)
_PACK_CARD_RANKS = list(Rank)


def _roll_standard_card(run: RunState) -> Card:
    rng = run.rng.stream("packs")
    c = Card(
        uid=run.new_uid(),
        rank=rng.choice(_PACK_CARD_RANKS),
        suit=rng.choice(_PACK_CARD_SUITS),
    )
    # Standard-pack modifier rates (approximate wiki rates; seam for tuning).
    if rng.random() < 0.40:
        c.enhancement = rng.choice(
            [e for e in Enhancement if e not in (Enhancement.NONE, Enhancement.STONE)]
        )
    if rng.random() < 0.20:
        c.seal = rng.choice([Seal.RED, Seal.BLUE, Seal.GOLD, Seal.PURPLE])
    if rng.random() < 0.08:
        c.edition = rng.choice([Edition.FOIL, Edition.HOLO, Edition.POLY])
    return c


def _roll_pack_options(run: RunState, pack_id: str) -> OpenPack:
    defn = PACKS[pack_id]
    rng = run.rng.stream("packs")
    pack = OpenPack(pack_id=pack_id, picks_left=defn.picks)

    if defn.category == "standard":
        for _ in range(defn.options):
            pack.option_kinds.append("card")
            pack.option_ids.append("")
            pack.option_cards.append(_roll_standard_card(run))
        return pack

    if defn.category == "buffoon":
        owned = {j.id for j in run.jokers}
        pool = sorted(
            jid for jid, d in JOKERS.items() if d.rarity != Rarity.LEGENDARY and jid not in owned
        )
    elif defn.category == "celestial":
        pool = sorted(
            cid
            for cid, d in CONSUMABLES.items()
            if d.kind == ConsumableKind.PLANET and _planet_discovered(run, cid)
        )
    elif defn.category == "arcana":
        pool = sorted(cid for cid, d in CONSUMABLES.items() if d.kind == ConsumableKind.TAROT)
    else:  # spectral
        pool = sorted(cid for cid, d in CONSUMABLES.items() if d.kind == ConsumableKind.SPECTRAL)

    kind = {"buffoon": "joker", "celestial": "planet", "arcana": "tarot", "spectral": "spectral"}[
        defn.category
    ]
    n = min(defn.options, len(pool))
    picks = rng.sample(pool, n) if n else []
    for pid in picks:
        pack.option_kinds.append(kind)
        pack.option_ids.append(pid)
        pack.option_cards.append(None)
    return pack


def buy_pack(run: RunState, slot: int) -> list[Event]:
    shop = run.shop
    assert shop is not None
    offer = shop.pack_slots.pop(slot)
    run.money -= offer.cost
    shop.open_pack = _roll_pack_options(run, offer.pack_id)
    run.phase = Phase.PACK_OPEN
    return [ItemBought("pack", offer.pack_id, offer.cost)]


def pick_pack_item(run: RunState, index: int) -> list[Event]:
    shop = run.shop
    assert shop is not None and shop.open_pack is not None
    pack = shop.open_pack
    kind = pack.option_kinds[index]
    events: list[Event] = []

    if kind == "card":
        card = pack.option_cards[index]
        assert card is not None
        run.full_deck.append(card)
        events.append(ItemBought("card", repr(card), 0))
    elif kind == "joker":
        jid = pack.option_ids[index]
        j = JokerInst(
            id=jid,
            uid=run.new_uid(),
            sell_value=sell_value(JOKERS[jid].base_cost),
            state=new_joker_state(jid),
        )
        run.jokers.append(j)
        joker_def(jid).effect.on_acquired(run, j)
        events.append(ItemBought("joker", jid, 0))
    else:  # tarot / planet / spectral
        cid = pack.option_ids[index]
        ckind = CONSUMABLES[cid].kind
        run.consumables.append(ConsumableInst(id=cid, kind=ckind, uid=run.new_uid()))
        events.append(ItemBought(str(ckind), cid, 0))

    # Remove the chosen option.
    del pack.option_kinds[index]
    del pack.option_ids[index]
    del pack.option_cards[index]
    pack.picks_left -= 1
    if pack.picks_left <= 0 or not pack.option_kinds:
        shop.open_pack = None
        run.phase = Phase.SHOP
    return events


def skip_pack(run: RunState) -> list[Event]:
    shop = run.shop
    assert shop is not None
    shop.open_pack = None
    run.phase = Phase.SHOP
    return []


def leave_shop(run: RunState) -> list[Event]:
    run.shop = None
    run.phase = Phase.BLIND_SELECT
    return []
