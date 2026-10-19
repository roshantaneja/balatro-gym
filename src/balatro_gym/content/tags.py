"""All 24 skip-blind tags.

Tag semantics (my own words; numbers verified against the game):
- Shop tags wait for the next shop: Uncommon/Rare add a free joker of that
  rarity; Foil/Holographic/Polychrome/Negative give the next base-edition
  joker in the shop that edition and make it free; Voucher adds an extra
  voucher; Coupon makes the initial cards and packs free; D6 starts rerolls
  at $0.
- Immediate tags fire when acquired: Handy ($1 per hand played this run),
  Garbage ($1 per unused discard this run), Economy (double money, max +$40),
  Top-up (up to 2 Common jokers), Orbital (+3 levels to a hand chosen when
  the tag appears), Speed ($5 per blind skipped this run), Boss (rerolls the
  boss), and the five pack tags (open a free Mega pack of that type).
- Investment pays $25 after the next boss is beaten; Juggle gives +3 hand
  size for the next round; Double copies the next tag acquired.

Per-ante availability (min_ante) matches the game's tag table.
"""

from __future__ import annotations

from balatro_gym.engine import cardops
from balatro_gym.engine.effects import TagDef, TagEffect, register_tag
from balatro_gym.engine.state import RunState, ShopState, TagInst
from balatro_gym.engine.types import BlindKind, Edition, Phase, Rarity

# ---------------------------------------------------------------------------
# Shop tags
# ---------------------------------------------------------------------------

class _FreeJoker(TagEffect):
    def __init__(self, rarity: Rarity):
        self.rarity = rarity

    def on_shop_open(self, run: RunState, tag: TagInst, shop: ShopState) -> bool:
        from balatro_gym.engine.shop import roll_shop_joker

        item = roll_shop_joker(run, self.rarity, free=True)
        if item is None:
            return True
        shop.card_slots.insert(0, item)
        return True


class _EditionTag(TagEffect):
    def __init__(self, edition: Edition):
        self.edition = edition

    def on_shop_open(self, run: RunState, tag: TagInst, shop: ShopState) -> bool:
        for item in shop.card_slots:
            if item.kind == "joker" and item.edition == Edition.NONE:
                item.edition = self.edition
                item.cost = 0
                return True
        return False  # persists until a base-edition joker shows up


class _VoucherTag(TagEffect):
    def on_shop_open(self, run: RunState, tag: TagInst, shop: ShopState) -> bool:
        from balatro_gym.engine.effects import VOUCHERS

        pool = sorted(
            vid for vid, d in VOUCHERS.items()
            if vid not in run.vouchers and vid not in shop.vouchers
            and (d.requires is None or d.requires in run.vouchers)
        )
        if pool:
            shop.vouchers.append(run.rng.stream("shop").choice(pool))
        return True


class _Coupon(TagEffect):
    def on_shop_open(self, run: RunState, tag: TagInst, shop: ShopState) -> bool:
        for item in shop.card_slots:
            item.cost = 0
        for offer in shop.pack_slots:
            offer.cost = 0
        return True


class _D6(TagEffect):
    def on_shop_open(self, run: RunState, tag: TagInst, shop: ShopState) -> bool:
        shop.reroll_cost = 0
        return True


# ---------------------------------------------------------------------------
# Immediate tags
# ---------------------------------------------------------------------------

class _Handy(TagEffect):
    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        run.money += run.stats.get("hands_played_run", 0)
        return True


class _Garbage(TagEffect):
    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        run.money += run.stats.get("discards_unused_run", 0)
        return True


class _Economy(TagEffect):
    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        run.money += min(max(run.money, 0), 40)
        return True


class _TopUp(TagEffect):
    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        for _ in range(2):
            jid = cardops.random_joker_id(run, Rarity.COMMON)
            if jid is None or not cardops.has_joker_room(run):
                break
            cardops.create_joker(run, jid)
        return True


class _Orbital(TagEffect):
    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        from balatro_gym.engine.types import HandType

        hand = tag.state.get("hand")
        if hand is None:
            hand = int(run.rng.stream("misc").choice(sorted(run.hands_discovered)))
        run.hand_levels[HandType(hand)] += 3
        return True


class _Speed(TagEffect):
    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        run.money += 5 * run.stats.get("blinds_skipped", 0)
        return True


class _BossTag(TagEffect):
    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        from balatro_gym.engine.round import reroll_boss

        reroll_boss(run)
        return True


class _PackTag(TagEffect):
    def __init__(self, pack_id: str):
        self.pack_id = pack_id

    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        from balatro_gym.engine.effects import PACKS
        from balatro_gym.engine.shop import open_pack

        if self.pack_id in PACKS:
            open_pack(run, self.pack_id, return_phase=Phase(run.phase))
        return True


# ---------------------------------------------------------------------------
# Deferred tags
# ---------------------------------------------------------------------------

class _Investment(TagEffect):
    def round_end_money(self, run: RunState, tag: TagInst) -> tuple[int, bool]:
        if run.round is not None and run.round.blind_kind == BlindKind.BOSS:
            return 25, True
        return 0, False


class _Juggle(TagEffect):
    def on_round_start(self, run: RunState, tag: TagInst) -> bool:
        assert run.round is not None
        run.round.hand_size_bonus += 3
        return True


class _Double(TagEffect):
    def on_tag_acquired(self, run: RunState, tag: TagInst, new: TagInst) -> bool:
        if new.id == "t_double":
            return False
        from balatro_gym.engine.round import acquire_tag

        # Consume ourselves first so the copy can't re-trigger this Double.
        run.tags = [t for t in run.tags if t is not tag]
        acquire_tag(run, new.clone())
        return False


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

register_tag(TagDef("t_uncommon", "Uncommon Tag", _FreeJoker(Rarity.UNCOMMON)))
register_tag(TagDef("t_rare", "Rare Tag", _FreeJoker(Rarity.RARE)))
register_tag(TagDef("t_negative", "Negative Tag", _EditionTag(Edition.NEGATIVE), min_ante=2))
register_tag(TagDef("t_foil", "Foil Tag", _EditionTag(Edition.FOIL)))
register_tag(TagDef("t_holo", "Holographic Tag", _EditionTag(Edition.HOLO)))
register_tag(TagDef("t_polychrome", "Polychrome Tag", _EditionTag(Edition.POLY)))
register_tag(TagDef("t_investment", "Investment Tag", _Investment()))
register_tag(TagDef("t_voucher", "Voucher Tag", _VoucherTag()))
register_tag(TagDef("t_boss", "Boss Tag", _BossTag()))
register_tag(TagDef("t_standard", "Standard Tag", _PackTag("p_standard_mega"), min_ante=2))
register_tag(TagDef("t_charm", "Charm Tag", _PackTag("p_arcana_mega")))
register_tag(TagDef("t_meteor", "Meteor Tag", _PackTag("p_celestial_mega"), min_ante=2))
register_tag(TagDef("t_buffoon", "Buffoon Tag", _PackTag("p_buffoon_mega"), min_ante=2))
register_tag(TagDef("t_handy", "Handy Tag", _Handy(), min_ante=2))
register_tag(TagDef("t_garbage", "Garbage Tag", _Garbage(), min_ante=2))
register_tag(TagDef("t_ethereal", "Ethereal Tag", _PackTag("p_spectral_mega"), min_ante=2))
register_tag(TagDef("t_coupon", "Coupon Tag", _Coupon()))
register_tag(TagDef("t_double", "Double Tag", _Double()))
register_tag(TagDef("t_juggle", "Juggle Tag", _Juggle()))
register_tag(TagDef("t_d6", "D6 Tag", _D6()))
register_tag(TagDef("t_top_up", "Top-up Tag", _TopUp(), min_ante=2))
register_tag(TagDef("t_speed", "Speed Tag", _Speed()))
register_tag(TagDef("t_orbital", "Orbital Tag", _Orbital(), min_ante=2, init_state={"hand": None}))
register_tag(TagDef("t_economy", "Economy Tag", _Economy()))
