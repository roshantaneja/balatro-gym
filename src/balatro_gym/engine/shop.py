"""Shop generation, reroll, buy/sell, booster packs.

Numbers verified against the game's tables:
- Card slots: 2 (+1 Overstock, +1 Overstock Plus). Kind weights: Joker 20,
  Tarot 4 (x2.4 Tarot Merchant / x8 Tarot Tycoon), Planet 4 (same vouchers),
  playing cards 0 (4 with Magic Trick), Spectral 0 (2 with the Ghost Deck).
- Joker edition odds in the shop: 4% total at base rate — Negative 0.3%,
  Polychrome 0.3%, Holographic 1.4%, Foil 2.0%; Hone x2, Glow Up x4 (Negative
  is not scaled by the edition rate).
- Rarity roll: 70% common / 25% uncommon / 5% rare (Legendary only via The Soul).
- Reroll: $5 base (-2 Reroll Surplus, -2 Reroll Glut), +$1 per reroll, resets
  each shop; rerolls only the card slots. Chaos the Clown: 1 free reroll/shop.
- Sell value = floor(cost / 2), min $1 (jokers and consumables).
- First shop of a run always offers a Buffoon pack.
- Stakes: Black 30% Eternal, Orange 30% Perishable (mutually exclusive roll),
  Gold 30% Rental (independent roll; Rental jokers cost $1).
- Arcana/Spectral/Celestial pack cards are USED immediately when picked (the
  pack deals a hand from your deck so card-targeting consumables can be aimed).
"""

from __future__ import annotations

from .cardops import (
    EDITION_SURCHARGE,
    active_jokers,
    add_card,
    add_joker,
    allows_duplicates,
    consumable_pool,
    has_joker_room,
    item_is_free,
    joker_pool,
    new_consumable,
    new_joker,
    remove_joker,
)
from .cards import Card
from .effects import (
    CONSUMABLES,
    PACKS,
    PLANET_FOR_HAND,
    TAGS,
    VOUCHERS,
    consumable_def,
    joker_def,
)
from .events import ConsumableUsed, Event, ItemBought, JokerSold
from .state import OpenPack, PackOffer, RunState, ShopItem, ShopState
from .types import (
    STICKER_ETERNAL,
    STICKER_PERISHABLE,
    STICKER_RENTAL,
    ConsumableKind,
    Edition,
    Enhancement,
    HandType,
    Phase,
    Rank,
    Rarity,
    Seal,
    Suit,
)

RARITY_WEIGHTS = [(Rarity.COMMON, 70.0), (Rarity.UNCOMMON, 25.0), (Rarity.RARE, 5.0)]
N_PACK_SLOTS = 2
RENTAL_COST = 1


def n_card_slots(run: RunState) -> int:
    return run.shop_card_slots


def sell_value(cost: int) -> int:
    return max(1, cost // 2)


def discounted(run: RunState, base: int) -> int:
    """Clearance Sale / Liquidation: percentage off, rounded, min $1."""
    pct = run.mods.get("shop_discount", 0)
    if not pct:
        return base
    return max(1, int(base * (100 - pct) / 100 + 0.5))


def _weighted_choice(rng, pairs):
    total = sum(w for _, w in pairs)
    x = rng.random() * total
    for item, w in pairs:
        x -= w
        if x <= 0:
            return item
    return pairs[-1][0]


# ---------------------------------------------------------------------------
# Rolls
# ---------------------------------------------------------------------------

def roll_joker_edition(run: RunState, *, allow_negative: bool = True, scale: float = 1.0) -> Edition:
    rate = run.mods.get("edition_rate", 1) * scale
    r = run.rng.stream("edition").random()
    if allow_negative and r > 1 - 0.003 * scale:
        return Edition.NEGATIVE
    if r > 1 - 0.006 * rate:
        return Edition.POLY
    if r > 1 - 0.02 * rate:
        return Edition.HOLO
    if r > 1 - 0.04 * rate:
        return Edition.FOIL
    return Edition.NONE


def roll_stickers(run: RunState) -> list[str]:
    rng = run.rng.stream("edition")
    stickers: list[str] = []
    poll = rng.random()
    if run.mods.get("eternals") and poll > 0.7:
        stickers.append(STICKER_ETERNAL)
    elif run.mods.get("perishables") and 0.4 < poll <= 0.7:
        stickers.append(STICKER_PERISHABLE)
    if run.mods.get("rentals") and rng.random() > 0.7:
        stickers.append(STICKER_RENTAL)
    return stickers


def _finalize_stickers(jid: str, stickers: list[str]) -> list[str]:
    d = joker_def(jid)
    out = []
    for s in stickers:
        if s == STICKER_ETERNAL and not d.eternal_compat:
            continue
        if s == STICKER_PERISHABLE and not d.perishable_compat:
            continue
        out.append(s)
    return out


def roll_shop_joker(run: RunState, rarity: Rarity | None = None, *, free: bool = False) -> ShopItem | None:
    rng = run.rng.stream("shop")
    if rarity is None:
        rarity = _weighted_choice(rng, RARITY_WEIGHTS)
    pool = joker_pool(run, rarity)
    if not pool:
        pool = joker_pool(run, None)
    if not pool:
        return None
    jid = rng.choice(pool)
    edition = roll_joker_edition(run)
    stickers = _finalize_stickers(jid, roll_stickers(run))
    cost = joker_cost(run, jid, edition, stickers)
    return ShopItem("joker", jid, 0 if free else cost, edition, stickers)


def joker_cost(run: RunState, jid: str, edition: Edition, stickers: list[str]) -> int:
    if STICKER_RENTAL in stickers:
        return RENTAL_COST
    return discounted(run, joker_def(jid).base_cost + EDITION_SURCHARGE.get(edition, 0))


def _kind_weights(run: RunState) -> list[tuple[str, float]]:
    pairs = [
        ("joker", 20.0),
        ("tarot", 4.0 * run.mods.get("tarot_rate", 1.0)),
        ("planet", 4.0 * run.mods.get("planet_rate", 1.0)),
        ("card", run.mods.get("card_rate", 0.0)),
        ("spectral", run.mods.get("spectral_rate", 0.0)),
    ]
    kinds = []
    for k, w in pairs:
        if w <= 0:
            continue
        if k == "tarot" and not consumable_pool(run, ConsumableKind.TAROT):
            continue
        if k == "planet" and not consumable_pool(run, ConsumableKind.PLANET):
            continue
        if k == "spectral" and not consumable_pool(run, ConsumableKind.SPECTRAL):
            continue
        if k == "joker" and not joker_pool(run, None):
            continue
        kinds.append((k, w))
    return kinds


def _roll_card_slot(run: RunState) -> ShopItem | None:
    rng = run.rng.stream("shop")
    kinds = _kind_weights(run)
    if not kinds:
        return None
    kind = _weighted_choice(rng, kinds)
    if kind == "joker":
        return roll_shop_joker(run)
    if kind == "card":
        card = roll_playing_card(run, modifiers=bool(run.mods.get("illusion")), stream="shop")
        return ShopItem("card", "", discounted(run, 1), card.edition, [], card)
    want = {"tarot": ConsumableKind.TAROT, "planet": ConsumableKind.PLANET,
            "spectral": ConsumableKind.SPECTRAL}[kind]
    pool = consumable_pool(run, want)
    if not pool:
        return None
    cid = rng.choice(pool)
    cost = 0 if item_is_free(run, kind, cid) else discounted(run, CONSUMABLES[cid].base_cost)
    return ShopItem(kind, cid, cost)


def _pack_offerable(run: RunState) -> list[tuple[str, float]]:
    out = []
    for pid, d in sorted(PACKS.items()):
        if d.category == "buffoon" and not joker_pool(run, None):
            continue
        if d.category in ("arcana", "celestial", "spectral"):
            want = {"arcana": ConsumableKind.TAROT, "celestial": ConsumableKind.PLANET,
                    "spectral": ConsumableKind.SPECTRAL}[d.category]
            if not any(c.kind == want and c.in_pool for c in CONSUMABLES.values()):
                continue
        out.append((pid, d.weight))
    return out


def _roll_pack_offer(run: RunState, force_buffoon: bool = False) -> PackOffer | None:
    offerable = _pack_offerable(run)
    if not offerable:
        return None
    rng = run.rng.stream("shop")
    if force_buffoon:
        buffoons = sorted(pid for pid, _ in offerable if PACKS[pid].category == "buffoon" and "normal" in pid)
        if buffoons:
            pid = buffoons[0]
            return PackOffer(pid, pack_cost(run, pid))
    pid = _weighted_choice(rng, offerable)
    return PackOffer(pid, pack_cost(run, pid))


def pack_cost(run: RunState, pid: str) -> int:
    if item_is_free(run, "pack", pid):
        return 0
    return discounted(run, PACKS[pid].cost)


# ---------------------------------------------------------------------------
# Shop lifecycle
# ---------------------------------------------------------------------------

def open_shop(run: RunState) -> None:
    first_shop = run.stats.get("shops_opened", 0) == 0
    run.stats["shops_opened"] = run.stats.get("shops_opened", 0) + 1

    card_slots = [s for s in (_roll_card_slot(run) for _ in range(n_card_slots(run))) if s]
    pack_slots = []
    for i in range(N_PACK_SLOTS):
        offer = _roll_pack_offer(run, force_buffoon=(first_shop and i == 0))
        if offer:
            pack_slots.append(offer)

    shop = ShopState(
        card_slots=card_slots,
        pack_slots=pack_slots,
        vouchers=[run.ante_voucher] if run.ante_voucher else [],
        reroll_cost=max(0, run.reroll_base),
        free_rerolls=sum(joker_def(j.id).effect.free_rerolls(run, j) for j in active_jokers(run)),
    )
    run.shop = shop
    run.phase = Phase.SHOP
    # Shop tags (Uncommon/Rare/edition/Voucher/Coupon/D6), in acquisition order.
    for tag in list(run.tags):
        d = TAGS.get(tag.id)
        if d and d.effect.on_shop_open(run, tag, shop):
            run.tags = [t for t in run.tags if t is not tag]


def reroll(run: RunState) -> list[Event]:
    shop = run.shop
    assert shop is not None
    if shop.free_rerolls > 0:
        shop.free_rerolls -= 1
    else:
        run.money -= shop.reroll_cost
        shop.reroll_cost += 1
    run.stats["rerolls"] = run.stats.get("rerolls", 0) + 1
    shop.card_slots = [s for s in (_roll_card_slot(run) for _ in range(n_card_slots(run))) if s]
    for j in active_jokers(run):
        joker_def(j.id).effect.update_on_reroll(run, j)
    return []


def reroll_cost_now(run: RunState) -> int:
    shop = run.shop
    assert shop is not None
    return 0 if shop.free_rerolls > 0 else shop.reroll_cost


def buy_shop_card(run: RunState, slot: int) -> list[Event]:
    shop = run.shop
    assert shop is not None
    item = shop.card_slots.pop(slot)
    run.money -= item.cost
    if item.kind == "joker":
        j = new_joker(run, item.item_id, edition=item.edition, stickers=item.stickers,
                      cost=item.cost if item.cost > 0 else None)
        j.sell_value = sell_value(item.cost) if item.cost > 0 else sell_value(joker_def(j.id).base_cost)
        add_joker(run, j)
    elif item.kind == "card":
        assert item.card is not None
        add_card(run, item.card, to_hand=False)
    else:
        inst = new_consumable(run, item.item_id)
        inst.sell_value = sell_value(item.cost) if item.cost > 0 else inst.sell_value
        run.consumables.append(inst)
    run.stats["shop_spent"] = run.stats.get("shop_spent", 0) + item.cost
    return [ItemBought(item.kind, item.item_id or repr(item.card), item.cost)]


def sell_joker(run: RunState, slot: int) -> list[Event]:
    j = run.jokers[slot]
    value = j.sell_value
    run.money += value
    remove_joker(run, j, sold=True)
    run.stats["cards_sold"] = run.stats.get("cards_sold", 0) + 1
    for other in active_jokers(run):
        joker_def(other.id).effect.update_on_other_sold(run, other, j.uid)
    from .scoring import get_boss

    boss = get_boss(run)
    if boss is not None:
        boss.on_joker_sold(run)
    return [JokerSold(j.id, value)]


def sell_consumable(run: RunState, slot: int) -> list[Event]:
    c = run.consumables.pop(slot)
    run.money += c.sell_value
    run.stats["cards_sold"] = run.stats.get("cards_sold", 0) + 1
    for other in active_jokers(run):
        joker_def(other.id).effect.update_on_other_sold(run, other, c.uid)
    return [JokerSold(c.id, c.sell_value)]


def buy_voucher(run: RunState, slot: int = 0) -> list[Event]:
    shop = run.shop
    assert shop is not None and shop.vouchers
    vid = shop.vouchers.pop(slot)
    defn = VOUCHERS[vid]
    run.money -= defn.cost
    run.vouchers = run.vouchers | {vid}
    if run.ante_voucher == vid:
        run.ante_voucher = None
    if defn.apply:
        defn.apply(run)  # type: ignore[operator]
    run.stats["vouchers_bought"] = run.stats.get("vouchers_bought", 0) + 1
    return [ItemBought("voucher", vid, defn.cost)]


def swap_jokers(run: RunState, slot: int) -> list[Event]:
    js = run.jokers
    js[slot], js[slot + 1] = js[slot + 1], js[slot]
    return []


def leave_shop(run: RunState) -> list[Event]:
    for j in active_jokers(run):
        joker_def(j.id).effect.update_on_shop_leave(run, j)
    run.shop = None
    run.phase = Phase.BLIND_SELECT
    return []


# ---------------------------------------------------------------------------
# Playing-card generation (standard packs, Magic Trick, Certificate, ...)
# ---------------------------------------------------------------------------

_PACK_CARD_SUITS = list(Suit)
_PACK_CARD_RANKS = list(Rank)


def roll_playing_card(run: RunState, *, modifiers: bool = True, stream: str = "packs") -> Card:
    """A random playing card. With modifiers: 40% enhancement, 20% seal, 8% edition
    (standard-pack rates)."""
    rng = run.rng.stream(stream)
    c = Card(
        uid=run.new_uid(),
        rank=rng.choice(_PACK_CARD_RANKS),
        suit=rng.choice(_PACK_CARD_SUITS),
    )
    if modifiers:
        if rng.random() < 0.40:
            c.enhancement = rng.choice(
                [e for e in Enhancement if e != Enhancement.NONE]
            )
        if rng.random() < 0.20:
            c.seal = rng.choice([Seal.RED, Seal.BLUE, Seal.GOLD, Seal.PURPLE])
        if rng.random() < 0.08:
            c.edition = rng.choice([Edition.FOIL, Edition.HOLO, Edition.POLY])
    return c


# ---------------------------------------------------------------------------
# Booster packs
# ---------------------------------------------------------------------------

def _roll_pack_options(run: RunState, pack_id: str) -> OpenPack:
    defn = PACKS[pack_id]
    rng = run.rng.stream("packs")
    pack = OpenPack(pack_id=pack_id, picks_left=defn.picks)

    if defn.category == "standard":
        for _ in range(defn.options):
            pack.option_kinds.append("card")
            pack.option_ids.append("")
            pack.option_cards.append(roll_playing_card(run))
            pack.option_editions.append(str(Edition.NONE))
        return pack

    if defn.category == "buffoon":
        chosen: list[str] = []
        for _ in range(defn.options):
            rarity = _weighted_choice(rng, RARITY_WEIGHTS)
            pool = [j for j in joker_pool(run, rarity) if j not in chosen or allows_duplicates(run)]
            if not pool:
                pool = [j for j in joker_pool(run, None) if j not in chosen or allows_duplicates(run)]
            if not pool:
                break
            jid = rng.choice(pool)
            chosen.append(jid)
            pack.option_kinds.append("joker")
            pack.option_ids.append(jid)
            pack.option_cards.append(None)
            pack.option_editions.append(str(roll_joker_edition(run)))
        return pack

    kind = {"celestial": "planet", "arcana": "tarot", "spectral": "spectral"}[defn.category]
    want = {"celestial": ConsumableKind.PLANET, "arcana": ConsumableKind.TAROT,
            "spectral": ConsumableKind.SPECTRAL}[defn.category]
    chosen = []
    if defn.category == "celestial" and run.mods.get("telescope"):
        # Telescope: the pack always contains the Planet of the most played hand.
        best = max(HandType, key=lambda ht: (run.hand_play_counts.get(ht, 0), -int(ht)))
        pid = PLANET_FOR_HAND.get(best)
        if pid is not None and run.hand_play_counts.get(best, 0) > 0:
            chosen.append(pid)
            pack.option_kinds.append(kind)
            pack.option_ids.append(pid)
            pack.option_cards.append(None)
            pack.option_editions.append(str(Edition.NONE))
    for _ in range(defn.options - len(chosen)):
        k = kind
        w = want
        if defn.category == "arcana" and run.mods.get("omen_globe") and rng.random() < 0.2:
            k, w = "spectral", ConsumableKind.SPECTRAL
        pool = [c for c in consumable_pool(run, w) if c not in chosen]
        # Hidden cards with pack-only odds (The Soul / Black Hole).
        extra = sorted(
            c for c, d in CONSUMABLES.items()
            if not d.in_pool and defn.category in d.hidden_in and c not in chosen
        )
        for cid in extra:
            if rng.random() < CONSUMABLES[cid].pool_weight:
                pool = [cid]
                break
        if not pool:
            break
        cid = rng.choice(sorted(pool))
        chosen.append(cid)
        pack.option_kinds.append(k)
        pack.option_ids.append(cid)
        pack.option_cards.append(None)
        pack.option_editions.append(str(Edition.NONE))

    if defn.category in ("arcana", "spectral"):
        pack.hand = deal_pack_hand(run)
    return pack


def deal_pack_hand(run: RunState) -> list[Card]:
    """Deal a hand-size sample of the deck for targeting pack consumables.
    Returns the SAME Card objects as run.full_deck."""
    from .cardops import effective_hand_size

    n = min(effective_hand_size(run), len(run.full_deck))
    if n <= 0:
        return []
    return run.rng.stream("packs").sample(run.full_deck, n)


def open_pack(run: RunState, pack_id: str, *, return_phase: Phase) -> None:
    if run.open_pack is not None:
        # A pack is already being opened (Double Tag + pack tag): queue it.
        if return_phase == Phase.PACK_OPEN:
            return_phase = Phase(run.open_pack.return_phase)
        run.pending_packs.append([pack_id, str(return_phase.value)])
        return
    pack = _roll_pack_options(run, pack_id)
    pack.return_phase = str(return_phase.value)
    run.open_pack = pack
    run.phase = Phase.PACK_OPEN
    run.stats["packs_opened"] = run.stats.get("packs_opened", 0) + 1
    for j in active_jokers(run):
        joker_def(j.id).effect.update_on_pack_opened(run, j)


def buy_pack(run: RunState, slot: int) -> list[Event]:
    shop = run.shop
    assert shop is not None
    offer = shop.pack_slots.pop(slot)
    run.money -= offer.cost
    run.stats["shop_spent"] = run.stats.get("shop_spent", 0) + offer.cost
    open_pack(run, offer.pack_id, return_phase=Phase.SHOP)
    return [ItemBought("pack", offer.pack_id, offer.cost)]


def pack_pick_legal(run: RunState, index: int) -> bool:
    pack = run.open_pack
    assert pack is not None
    kind = pack.option_kinds[index]
    if kind == "joker":
        return has_joker_room(run, Edition(pack.option_editions[index]))
    if kind == "card":
        return True
    eff = consumable_def(pack.option_ids[index]).effect
    if not eff.can_use(run):
        return False
    lo, _ = eff.target_arity
    return lo == 0 or len(pack.hand) >= lo


def pick_pack_item(run: RunState, index: int, targets: tuple[int, ...] = ()) -> list[Event]:
    pack = run.open_pack
    assert pack is not None
    kind = pack.option_kinds[index]
    events: list[Event] = []

    if kind == "card":
        card = pack.option_cards[index]
        assert card is not None
        add_card(run, card, to_hand=False)
        events.append(ItemBought("card", repr(card), 0))
    elif kind == "joker":
        jid = pack.option_ids[index]
        j = new_joker(run, jid, edition=Edition(pack.option_editions[index]))
        add_joker(run, j)
        events.append(ItemBought("joker", jid, 0))
    else:  # tarot / planet / spectral: used immediately
        cid = pack.option_ids[index]
        eff = consumable_def(cid).effect
        cards = [pack.hand[i] for i in targets]
        eff.use(run, cards)
        _note_consumable_used(run, cid)
        events.append(ConsumableUsed(cid))

    # Remove the chosen option.
    del pack.option_kinds[index]
    del pack.option_ids[index]
    del pack.option_cards[index]
    del pack.option_editions[index]
    pack.picks_left -= 1
    if pack.picks_left <= 0 or not pack.option_kinds:
        _close_pack(run)
    return events


def _note_consumable_used(run: RunState, cid: str) -> None:
    kind = CONSUMABLES[cid].kind
    run.stats["consumables_used"] = run.stats.get("consumables_used", 0) + 1
    key = f"{kind.value}s_used"
    run.stats[key] = run.stats.get(key, 0) + 1
    if kind == ConsumableKind.PLANET:
        used = set(run.stats.get("planets_used_ids", []))
        used.add(cid)
        run.stats["planets_used_ids"] = sorted(used)
    if CONSUMABLES[cid].track_last:
        run.last_consumable = cid
    inst = new_consumable(run, cid)
    for j in active_jokers(run):
        joker_def(j.id).effect.update_on_consumable_used(run, j, inst)


def skip_pack(run: RunState) -> list[Event]:
    for j in active_jokers(run):
        joker_def(j.id).effect.update_on_pack_skipped(run, j)
    _close_pack(run)
    return []


def _close_pack(run: RunState) -> None:
    pack = run.open_pack
    assert pack is not None
    run.open_pack = None
    run.phase = Phase(pack.return_phase)
    if run.phase == Phase.SHOP and run.shop is None:
        run.phase = Phase.BLIND_SELECT
    if run.pending_packs:
        pid, phase = run.pending_packs.pop(0)
        open_pack(run, pid, return_phase=Phase(phase))
