"""Run/round/shop state. Everything is cheaply cloneable and JSON-serializable.

clone()/to_dict()/from_dict() are the contract that makes golden tests and
(later) MCTS possible — keep every field a JSON scalar, list, or dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import Card, make_standard_deck
from .levels import BASE_REROLL_COST, INTEREST_CAP_DOLLARS
from .rng import RunRNG
from .types import BlindKind, ConsumableKind, Edition, HandType, Phase, Stake


@dataclass(slots=True)
class JokerInst:
    id: str  # content key, e.g. "j_green_joker"
    uid: int
    edition: Edition = Edition.NONE
    stickers: frozenset[str] = frozenset()  # "eternal" | "perishable" | "rental"
    sell_value: int = 1
    state: dict = field(default_factory=dict)  # per-joker mutable state (JSON scalars)
    perish_left: int = -1   # rounds until a Perishable joker is debuffed (-1 = n/a)
    debuffed: bool = False  # Perishable expired (permanent) — boss debuffs are per-round

    def clone(self) -> JokerInst:
        return JokerInst(
            id=self.id,
            uid=self.uid,
            edition=self.edition,
            stickers=self.stickers,
            sell_value=self.sell_value,
            state=dict(self.state),
            perish_left=self.perish_left,
            debuffed=self.debuffed,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "uid": self.uid,
            "edition": str(self.edition),
            "stickers": sorted(self.stickers),
            "sell_value": self.sell_value,
            "state": dict(self.state),
            "perish_left": self.perish_left,
            "debuffed": self.debuffed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> JokerInst:
        return cls(
            id=d["id"],
            uid=d["uid"],
            edition=Edition(d["edition"]),
            stickers=frozenset(d["stickers"]),
            sell_value=d["sell_value"],
            state=dict(d["state"]),
            perish_left=d.get("perish_left", -1),
            debuffed=d.get("debuffed", False),
        )


@dataclass(slots=True)
class ConsumableInst:
    id: str  # content key, e.g. "c_mercury"
    kind: ConsumableKind
    uid: int
    edition: Edition = Edition.NONE  # NEGATIVE: does not take a slot
    sell_value: int = 1

    def clone(self) -> ConsumableInst:
        return ConsumableInst(
            id=self.id, kind=self.kind, uid=self.uid, edition=self.edition,
            sell_value=self.sell_value,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": str(self.kind), "uid": self.uid,
            "edition": str(self.edition), "sell_value": self.sell_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConsumableInst:
        return cls(
            id=d["id"], kind=ConsumableKind(d["kind"]), uid=d["uid"],
            edition=Edition(d["edition"]), sell_value=d.get("sell_value", 1),
        )


@dataclass(slots=True)
class TagInst:
    """A held skip tag. `state` carries per-instance data (Orbital's hand)."""

    id: str
    state: dict = field(default_factory=dict)

    def clone(self) -> TagInst:
        return TagInst(id=self.id, state=dict(self.state))

    def to_dict(self) -> dict:
        return {"id": self.id, "state": dict(self.state)}

    @classmethod
    def from_dict(cls, d: dict) -> TagInst:
        if isinstance(d, str):  # legacy: bare id
            return cls(id=d)
        return cls(id=d["id"], state=dict(d.get("state", {})))


@dataclass(slots=True)
class ShopItem:
    kind: str  # "joker" | "tarot" | "planet" | "spectral" | "card"
    item_id: str  # content key; "" for playing cards
    cost: int
    edition: Edition = Edition.NONE
    stickers: list[str] = field(default_factory=list)  # joker stickers (stakes)
    card: Card | None = None  # inline playing card (Magic Trick / Illusion)

    def clone(self) -> ShopItem:
        return ShopItem(
            self.kind, self.item_id, self.cost, self.edition, list(self.stickers),
            self.card.clone() if self.card else None,
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "item_id": self.item_id, "cost": self.cost,
            "edition": str(self.edition), "stickers": list(self.stickers),
            "card": self.card.to_dict() if self.card else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ShopItem:
        return cls(
            d["kind"], d["item_id"], d["cost"], Edition(d["edition"]),
            list(d.get("stickers", [])),
            Card.from_dict(d["card"]) if d.get("card") else None,
        )


@dataclass(slots=True)
class PackOffer:
    pack_id: str  # content key, e.g. "p_buffoon_normal"
    cost: int

    def to_dict(self) -> dict:
        return {"pack_id": self.pack_id, "cost": self.cost}

    @classmethod
    def from_dict(cls, d: dict) -> PackOffer:
        return cls(d["pack_id"], d["cost"])


@dataclass(slots=True)
class OpenPack:
    pack_id: str
    picks_left: int
    # Options are heterogeneous: jokers/consumables as content ids, cards inline.
    option_kinds: list[str] = field(default_factory=list)  # "joker"|"tarot"|"planet"|"spectral"|"card"
    option_ids: list[str] = field(default_factory=list)    # content key; for cards, ""
    option_cards: list[Card | None] = field(default_factory=list)
    option_editions: list[str] = field(default_factory=list)  # joker editions (Edition values)
    # Arcana/Spectral packs deal a hand from the deck so targeted consumables
    # can be used straight from the pack (as in the real game). The cards are
    # the SAME objects as run.full_deck, so mutations persist.
    hand: list[Card] = field(default_factory=list)
    return_phase: str = str(Phase.SHOP)  # phase to resume once the pack closes

    def to_dict(self) -> dict:
        return {
            "pack_id": self.pack_id,
            "picks_left": self.picks_left,
            "option_kinds": list(self.option_kinds),
            "option_ids": list(self.option_ids),
            "option_cards": [c.to_dict() if c else None for c in self.option_cards],
            "option_editions": list(self.option_editions),
            "hand": [c.to_dict() for c in self.hand],
            "return_phase": self.return_phase,
        }

    @classmethod
    def from_dict(cls, d: dict, card_map: dict[int, Card] | None = None) -> OpenPack:
        def m(cd: dict) -> Card:
            if card_map is not None and cd["uid"] in card_map:
                return card_map[cd["uid"]]
            return Card.from_dict(cd)

        n = len(d["option_kinds"])
        return cls(
            pack_id=d["pack_id"],
            picks_left=d["picks_left"],
            option_kinds=list(d["option_kinds"]),
            option_ids=list(d["option_ids"]),
            option_cards=[Card.from_dict(c) if c else None for c in d["option_cards"]],
            option_editions=list(d.get("option_editions", ["none"] * n)),
            hand=[m(c) for c in d.get("hand", [])],
            return_phase=d.get("return_phase", str(Phase.SHOP)),
        )


@dataclass(slots=True)
class ShopState:
    card_slots: list[ShopItem] = field(default_factory=list)
    pack_slots: list[PackOffer] = field(default_factory=list)
    vouchers: list[str] = field(default_factory=list)  # ante voucher + Voucher-tag extras
    reroll_cost: int = BASE_REROLL_COST
    free_rerolls: int = 0  # Chaos the Clown

    @property
    def voucher(self) -> str | None:
        """First voucher on offer (compat shim for single-voucher callers)."""
        return self.vouchers[0] if self.vouchers else None

    def clone(self, card_map: dict[int, Card] | None = None) -> ShopState:
        return ShopState(
            card_slots=[s.clone() for s in self.card_slots],
            pack_slots=[PackOffer(p.pack_id, p.cost) for p in self.pack_slots],
            vouchers=list(self.vouchers),
            reroll_cost=self.reroll_cost,
            free_rerolls=self.free_rerolls,
        )

    def to_dict(self) -> dict:
        return {
            "card_slots": [s.to_dict() for s in self.card_slots],
            "pack_slots": [p.to_dict() for p in self.pack_slots],
            "vouchers": list(self.vouchers),
            "reroll_cost": self.reroll_cost,
            "free_rerolls": self.free_rerolls,
        }

    @classmethod
    def from_dict(cls, d: dict, card_map: dict[int, Card] | None = None) -> ShopState:
        vouchers = list(d.get("vouchers", []))
        if not vouchers and d.get("voucher"):  # legacy single-voucher field
            vouchers = [d["voucher"]]
        return cls(
            card_slots=[ShopItem.from_dict(s) for s in d["card_slots"]],
            pack_slots=[PackOffer.from_dict(p) for p in d["pack_slots"]],
            vouchers=vouchers,
            reroll_cost=d["reroll_cost"],
            free_rerolls=d.get("free_rerolls", 0),
        )


@dataclass(slots=True)
class RoundState:
    blind_kind: BlindKind
    boss_id: str | None
    chips_needed: float
    chips_scored: float = 0.0
    hands_left: int = 4
    discards_left: int = 3
    hand: list[Card] = field(default_factory=list)
    deck_remaining: list[Card] = field(default_factory=list)
    discarded: list[Card] = field(default_factory=list)
    played_uids: list[int] = field(default_factory=list)  # cards played this round
    hands_played: int = 0
    discards_used: int = 0
    hand_types_played: list[int] = field(default_factory=list)  # HandType ints, this round (Eye/Mouth)
    face_down_uids: list[int] = field(default_factory=list)  # hidden in obs (House/Wheel/Mark)
    hand_size_bonus: int = 0  # Juggle tag / Manacle (round-scoped)
    boss_state: dict = field(default_factory=dict)
    # boss_state keys used by the engine:
    #   "disabled": True        -> boss effect switched off (Chicot/Luchador)
    #   "disabled_joker_uid"    -> Crimson Heart's current victim
    #   "forced_uid"            -> Cerulean Bell's forced card
    #   "jokers_hidden": True   -> Amber Acorn flipped the jokers face down
    #   "verdant": True         -> Verdant Leaf debuff still active
    #   "triggered": True       -> boss ability fired this hand (Matador)

    def clone(self, card_map: dict[int, Card] | None = None) -> RoundState:
        """card_map: uid -> already-cloned Card (from RunState.full_deck) so that
        round containers keep referencing the SAME objects as full_deck after a
        clone. Cards not in the map (e.g. removed from full_deck by glass
        breakage while still in the play area) are cloned directly."""

        def m(c: Card) -> Card:
            if card_map is not None:
                mapped = card_map.get(c.uid)
                if mapped is not None:
                    return mapped
            return c.clone()

        return RoundState(
            blind_kind=self.blind_kind,
            boss_id=self.boss_id,
            chips_needed=self.chips_needed,
            chips_scored=self.chips_scored,
            hands_left=self.hands_left,
            discards_left=self.discards_left,
            hand=[m(c) for c in self.hand],
            deck_remaining=[m(c) for c in self.deck_remaining],
            discarded=[m(c) for c in self.discarded],
            played_uids=list(self.played_uids),
            hands_played=self.hands_played,
            discards_used=self.discards_used,
            hand_types_played=list(self.hand_types_played),
            face_down_uids=list(self.face_down_uids),
            hand_size_bonus=self.hand_size_bonus,
            boss_state=dict(self.boss_state),
        )

    def to_dict(self) -> dict:
        return {
            "blind_kind": int(self.blind_kind),
            "boss_id": self.boss_id,
            "chips_needed": self.chips_needed,
            "chips_scored": self.chips_scored,
            "hands_left": self.hands_left,
            "discards_left": self.discards_left,
            "hand": [c.to_dict() for c in self.hand],
            "deck_remaining": [c.to_dict() for c in self.deck_remaining],
            "discarded": [c.to_dict() for c in self.discarded],
            "played_uids": list(self.played_uids),
            "hands_played": self.hands_played,
            "discards_used": self.discards_used,
            "hand_types_played": list(self.hand_types_played),
            "face_down_uids": list(self.face_down_uids),
            "hand_size_bonus": self.hand_size_bonus,
            "boss_state": dict(self.boss_state),
        }

    @classmethod
    def from_dict(cls, d: dict, card_map: dict[int, Card] | None = None) -> RoundState:
        def m(cd: dict) -> Card:
            if card_map is not None:
                mapped = card_map.get(cd["uid"])
                if mapped is not None:
                    return mapped
            return Card.from_dict(cd)

        return cls(
            blind_kind=BlindKind(d["blind_kind"]),
            boss_id=d["boss_id"],
            chips_needed=d["chips_needed"],
            chips_scored=d["chips_scored"],
            hands_left=d["hands_left"],
            discards_left=d["discards_left"],
            hand=[m(c) for c in d["hand"]],
            deck_remaining=[m(c) for c in d["deck_remaining"]],
            discarded=[m(c) for c in d["discarded"]],
            played_uids=list(d["played_uids"]),
            hands_played=d["hands_played"],
            discards_used=d.get("discards_used", 0),
            hand_types_played=list(d["hand_types_played"]),
            face_down_uids=list(d.get("face_down_uids", [])),
            hand_size_bonus=d.get("hand_size_bonus", d.get("boss_state", {}).get("hand_size_bonus", 0)),
            boss_state=dict(d["boss_state"]),
        )


@dataclass(slots=True)
class RunState:
    seed: str
    rng: RunRNG
    stake: Stake = Stake.WHITE
    deck_id: str = "d_red"
    ante: int = 1
    money: int = 4
    full_deck: list[Card] = field(default_factory=list)
    jokers: list[JokerInst] = field(default_factory=list)
    consumables: list[ConsumableInst] = field(default_factory=list)
    joker_slots: int = 5
    consumable_slots: int = 2
    hand_size: int = 8          # permanent base (vouchers/decks/spectrals); jokers add on top
    hands_per_round: int = 4    # permanent base; jokers add on top
    discards_per_round: int = 3
    hand_levels: dict[HandType, int] = field(
        default_factory=lambda: {ht: 1 for ht in HandType}
    )
    hand_play_counts: dict[HandType, int] = field(
        default_factory=lambda: {ht: 0 for ht in HandType}
    )
    hands_discovered: set[HandType] = field(
        default_factory=lambda: {ht for ht in HandType if ht < HandType.FIVE_KIND}
    )
    vouchers: frozenset[str] = frozenset()
    tags: list[TagInst] = field(default_factory=list)
    phase: Phase = Phase.BLIND_SELECT
    blind_index: int = 0  # 0=small, 1=big, 2=boss within current ante
    skips_used: list[int] = field(default_factory=list)  # BlindKind ints skipped (for tags)
    bosses_seen: list[str] = field(default_factory=list)
    round: RoundState | None = None
    shop: ShopState | None = None
    open_pack: OpenPack | None = None  # booster being opened (shop or tag-granted)
    pending_packs: list[list] = field(default_factory=list)  # [pack_id, return_phase] queued
    next_uid: int = 0
    chance_multiplier: int = 1  # product of joker chance multipliers (Oops! All 6s)
    stats: dict = field(default_factory=dict)  # tarots_used, glass_broken, hands_played_run, ...
    # --- run-wide modifiers written by decks / vouchers / stakes (engine reads
    # them by key; content never needs engine edits to add one) ---
    mods: dict = field(default_factory=dict)
    interest_cap: int = INTEREST_CAP_DOLLARS  # dollars that earn interest (25 / 50 / 100)
    interest_amount: int = 1                   # $ per $5 held (To the Moon adds)
    shop_card_slots: int = 2
    reroll_base: int = BASE_REROLL_COST        # Reroll Surplus / Glut lower this
    # --- per-ante pre-rolls (visible to the agent before choosing to skip) ---
    ante_boss: str | None = None               # boss chosen for this ante
    ante_voucher: str | None = None            # voucher on offer this ante (None once bought)
    blind_tags: list[TagInst | None] = field(default_factory=lambda: [None, None])
    boss_rerolls_used: int = 0                 # Director's Cut (1) / Retcon (inf)
    ante_played_uids: list[int] = field(default_factory=list)  # The Pillar
    last_consumable: str | None = None         # The Fool copies this
    pending_destroy: list[int] = field(default_factory=list)   # joker uids to remove post-scoring

    # ---------------------------------------------------------------- helpers
    def new_uid(self) -> int:
        u = self.next_uid
        self.next_uid += 1
        return u

    def clone(self) -> RunState:
        cloned_deck = [c.clone() for c in self.full_deck]
        card_map = {c.uid: c for c in cloned_deck}
        return RunState(
            seed=self.seed,
            rng=self.rng.clone(),
            stake=self.stake,
            deck_id=self.deck_id,
            ante=self.ante,
            money=self.money,
            full_deck=cloned_deck,
            jokers=[j.clone() for j in self.jokers],
            consumables=[c.clone() for c in self.consumables],
            joker_slots=self.joker_slots,
            consumable_slots=self.consumable_slots,
            hand_size=self.hand_size,
            hands_per_round=self.hands_per_round,
            discards_per_round=self.discards_per_round,
            hand_levels=dict(self.hand_levels),
            hand_play_counts=dict(self.hand_play_counts),
            hands_discovered=set(self.hands_discovered),
            vouchers=self.vouchers,
            tags=[t.clone() for t in self.tags],
            phase=self.phase,
            blind_index=self.blind_index,
            skips_used=list(self.skips_used),
            bosses_seen=list(self.bosses_seen),
            round=self.round.clone(card_map) if self.round else None,
            shop=self.shop.clone(card_map) if self.shop else None,
            open_pack=OpenPack.from_dict(self.open_pack.to_dict(), card_map) if self.open_pack else None,
            pending_packs=[list(p) for p in self.pending_packs],
            next_uid=self.next_uid,
            chance_multiplier=self.chance_multiplier,
            stats=_deep_copy_json(self.stats),
            mods=dict(self.mods),
            interest_cap=self.interest_cap,
            interest_amount=self.interest_amount,
            shop_card_slots=self.shop_card_slots,
            reroll_base=self.reroll_base,
            ante_boss=self.ante_boss,
            ante_voucher=self.ante_voucher,
            blind_tags=[t.clone() if t else None for t in self.blind_tags],
            boss_rerolls_used=self.boss_rerolls_used,
            ante_played_uids=list(self.ante_played_uids),
            last_consumable=self.last_consumable,
            pending_destroy=list(self.pending_destroy),
        )

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "rng": self.rng.to_dict(),
            "stake": str(self.stake),
            "deck_id": self.deck_id,
            "ante": self.ante,
            "money": self.money,
            "full_deck": [c.to_dict() for c in self.full_deck],
            "jokers": [j.to_dict() for j in self.jokers],
            "consumables": [c.to_dict() for c in self.consumables],
            "joker_slots": self.joker_slots,
            "consumable_slots": self.consumable_slots,
            "hand_size": self.hand_size,
            "hands_per_round": self.hands_per_round,
            "discards_per_round": self.discards_per_round,
            "hand_levels": {int(k): v for k, v in self.hand_levels.items()},
            "hand_play_counts": {int(k): v for k, v in self.hand_play_counts.items()},
            "hands_discovered": sorted(int(h) for h in self.hands_discovered),
            "vouchers": sorted(self.vouchers),
            "tags": [t.to_dict() for t in self.tags],
            "phase": str(self.phase),
            "blind_index": self.blind_index,
            "skips_used": list(self.skips_used),
            "bosses_seen": list(self.bosses_seen),
            "round": self.round.to_dict() if self.round else None,
            "shop": self.shop.to_dict() if self.shop else None,
            "open_pack": self.open_pack.to_dict() if self.open_pack else None,
            "pending_packs": [list(p) for p in self.pending_packs],
            "next_uid": self.next_uid,
            "chance_multiplier": self.chance_multiplier,
            "stats": _deep_copy_json(self.stats),
            "mods": dict(self.mods),
            "interest_cap": self.interest_cap,
            "interest_amount": self.interest_amount,
            "shop_card_slots": self.shop_card_slots,
            "reroll_base": self.reroll_base,
            "ante_boss": self.ante_boss,
            "ante_voucher": self.ante_voucher,
            "blind_tags": [t.to_dict() if t else None for t in self.blind_tags],
            "boss_rerolls_used": self.boss_rerolls_used,
            "ante_played_uids": list(self.ante_played_uids),
            "last_consumable": self.last_consumable,
            "pending_destroy": list(self.pending_destroy),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunState:
        full_deck = [Card.from_dict(c) for c in d["full_deck"]]
        card_map = {c.uid: c for c in full_deck}
        return cls(
            seed=d["seed"],
            rng=RunRNG.from_dict(d["rng"]),
            stake=Stake(d["stake"]),
            deck_id=d["deck_id"],
            ante=d["ante"],
            money=d["money"],
            full_deck=full_deck,
            jokers=[JokerInst.from_dict(j) for j in d["jokers"]],
            consumables=[ConsumableInst.from_dict(c) for c in d["consumables"]],
            joker_slots=d["joker_slots"],
            consumable_slots=d["consumable_slots"],
            hand_size=d["hand_size"],
            hands_per_round=d["hands_per_round"],
            discards_per_round=d["discards_per_round"],
            hand_levels={HandType(int(k)): v for k, v in d["hand_levels"].items()},
            hand_play_counts={HandType(int(k)): v for k, v in d["hand_play_counts"].items()},
            hands_discovered={HandType(h) for h in d["hands_discovered"]},
            vouchers=frozenset(d["vouchers"]),
            tags=[TagInst.from_dict(t) for t in d["tags"]],
            phase=Phase(d["phase"]),
            blind_index=d["blind_index"],
            skips_used=list(d["skips_used"]),
            bosses_seen=list(d["bosses_seen"]),
            round=RoundState.from_dict(d["round"], card_map) if d["round"] else None,
            shop=ShopState.from_dict(d["shop"], card_map) if d["shop"] else None,
            open_pack=OpenPack.from_dict(d["open_pack"], card_map) if d.get("open_pack") else None,
            pending_packs=[list(p) for p in d.get("pending_packs", [])],
            next_uid=d["next_uid"],
            chance_multiplier=d["chance_multiplier"],
            stats=_deep_copy_json(d["stats"]),
            mods=dict(d.get("mods", {})),
            interest_cap=d.get("interest_cap", INTEREST_CAP_DOLLARS),
            interest_amount=d.get("interest_amount", 1),
            shop_card_slots=d.get("shop_card_slots", 2),
            reroll_base=d.get("reroll_base", BASE_REROLL_COST),
            ante_boss=d.get("ante_boss"),
            ante_voucher=d.get("ante_voucher"),
            blind_tags=[TagInst.from_dict(t) if t else None for t in d.get("blind_tags", [None, None])],
            boss_rerolls_used=d.get("boss_rerolls_used", 0),
            ante_played_uids=list(d.get("ante_played_uids", [])),
            last_consumable=d.get("last_consumable"),
            pending_destroy=list(d.get("pending_destroy", [])),
        )


def _deep_copy_json(x):
    if isinstance(x, dict):
        return {k: _deep_copy_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_deep_copy_json(v) for v in x]
    return x


def new_run_state(seed: str, deck_id: str = "d_red", stake: Stake = Stake.WHITE) -> RunState:
    """Bare run state with a fresh standard deck. Deck/stake modifiers are
    applied by content/decks and engine.run.Game."""
    run = RunState(seed=seed, rng=RunRNG(seed), deck_id=deck_id, stake=stake)
    run.full_deck = make_standard_deck(start_uid=0)
    run.next_uid = len(run.full_deck)
    run.stats["starting_deck_size"] = len(run.full_deck)
    return run
