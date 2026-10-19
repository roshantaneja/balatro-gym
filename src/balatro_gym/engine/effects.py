"""Effect hook contracts and content registries.

The engine layer knows nothing about specific jokers: content registers
JokerDef/ConsumableDef/BossDef/... objects into the registries below, and the
scoring/round/shop orchestrators dispatch through the hook interfaces.

Design invariants:
- Effects are STATELESS singletons; all per-instance mutability lives in
  JokerInst.state (a JSON-scalar dict).
- Scoring hooks (`on_*`) run for Blueprint/Brainstorm copies too, reading the
  copy target's state. State-mutation hooks (`update_*`) run ONLY for the
  original joker — that split is what makes copy semantics correct by
  construction (Blueprint + Green Joker pays the mult twice but scales once).
  Scoring hooks that must mutate state anyway (Vampire, Wee Joker) check
  `ctx.is_copy` and skip the mutation for copies.
- Copy resolution is generic: JokerDef.copy_target is "right" / "leftmost" /
  None. Chains are followed with a cycle guard; uncopyable targets no-op.
- Debuffed jokers (Perishable expiry, Crimson Heart) are skipped entirely —
  no hook of theirs fires, and copies that resolve to them resolve to nothing.
- Polled modifiers (hand size, free rerolls, debt, chance multiplier, ...) are
  queried by the engine when needed, so a joker's passive effect is always
  consistent with the board without any acquire/sell bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import ConsumableKind, Rarity

if TYPE_CHECKING:
    from .cards import Card
    from .hand_eval import EvalRules, HandResult
    from .scoring import ScoreContext
    from .state import ConsumableInst, JokerInst, RunState, ShopState, TagInst


# ---------------------------------------------------------------------------
# Jokers
# ---------------------------------------------------------------------------

class JokerEffect:
    """Base with do-nothing defaults. Override only what the joker needs.

    Hooks are listed in activation-sequence order. `ctx` is the live
    ScoreContext; `j` is the JokerInst whose STATE the hook should read (the
    copy target under Blueprint), while edition bonuses are handled by the
    orchestrator using the physical instance.
    """

    # --- scoring-time hooks (run for copies too) ---
    def on_hand_played(self, ctx: ScoreContext, j: JokerInst) -> None: ...

    def on_card_scored(self, ctx: ScoreContext, j: JokerInst, card: Card) -> None: ...

    def retrigger_scored(self, ctx: ScoreContext, j: JokerInst, card: Card) -> int:
        return 0

    def on_card_held(self, ctx: ScoreContext, j: JokerInst, card: Card) -> None: ...

    def retrigger_held(self, ctx: ScoreContext, j: JokerInst, card: Card) -> int:
        return 0

    def on_independent(self, ctx: ScoreContext, j: JokerInst) -> None: ...

    def on_other_joker(self, ctx: ScoreContext, j: JokerInst, other: JokerInst) -> None:
        """Baseball Card: fires once per *other* joker during stage 5."""

    def on_discard(
        self, ctx_run: RunState, j: JokerInst, cards: list[Card], last: bool
    ) -> None: ...

    # --- state-update hooks (originals only, never copies) ---
    def update_on_hand_played(
        self, run: RunState, j: JokerInst, played: list[Card], hr: HandResult
    ) -> None: ...

    def update_after_scoring(self, run: RunState, j: JokerInst, result) -> None:
        """After the whole hand resolved (Loyalty/Seltzer countdowns that must
        not affect the current hand, Sixth Sense/Superposition creations)."""

    def update_on_discard(self, run: RunState, j: JokerInst, cards: list[Card]) -> None: ...

    def update_on_round_end(self, run: RunState, j: JokerInst) -> None: ...

    def update_on_lucky(self, run: RunState, j: JokerInst) -> None:
        """A Lucky card trigger landed (Lucky Cat)."""

    def update_on_card_destroyed(self, run: RunState, j: JokerInst, card: Card) -> None:
        """A playing card left the deck for good (Glass Joker, Canio)."""

    def update_on_card_added(self, run: RunState, j: JokerInst, card: Card) -> None:
        """A playing card was added to the deck (Hologram)."""

    def update_on_consumable_used(
        self, run: RunState, j: JokerInst, inst: ConsumableInst
    ) -> None:
        """Constellation (planets), Fortune Teller reads stats instead."""

    def update_on_pack_opened(self, run: RunState, j: JokerInst) -> None:
        """Hallucination."""

    def update_on_pack_skipped(self, run: RunState, j: JokerInst) -> None:
        """Red Card."""

    def update_on_reroll(self, run: RunState, j: JokerInst) -> None:
        """Flash Card."""

    def update_on_blind_skipped(self, run: RunState, j: JokerInst) -> None:
        """Throwback reads stats; hook kept for symmetry."""

    def update_on_other_sold(self, run: RunState, j: JokerInst, sold_uid: int) -> None:
        """Any card (joker or consumable) was sold (Campfire)."""

    def update_on_boss_defeated(self, run: RunState, j: JokerInst) -> None:
        """Rocket grows, Campfire resets."""

    def update_on_shop_leave(self, run: RunState, j: JokerInst) -> None:
        """Perkeo."""

    # --- lifecycle / economy ---
    def round_end_money(self, run: RunState, j: JokerInst) -> int:
        return 0

    def on_blind_selected(self, run: RunState, j: JokerInst) -> None:
        """Runs when a blind starts, BEFORE the boss applies its start effect
        (so Chicot can disable it) and before the first draw."""

    def on_acquired(self, run: RunState, j: JokerInst) -> None: ...

    def on_sold(self, run: RunState, j: JokerInst) -> None:
        """This joker was sold (Luchador, Diet Cola, Invisible Joker)."""

    def on_removed(self, run: RunState, j: JokerInst) -> None:
        """This joker left the board for any reason (sold or destroyed)."""

    def prevent_loss(self, run: RunState, j: JokerInst) -> bool:
        """Mr. Bones: return True to survive a failed blind (joker is destroyed)."""
        return False

    # --- polled passive modifiers ---
    def modify_eval_rules(self, rules: EvalRules, j: JokerInst) -> EvalRules:
        return rules

    def hand_size_mod(self, run: RunState, j: JokerInst) -> int:
        return 0

    def hands_mod(self, run: RunState, j: JokerInst) -> int:
        return 0

    def discards_mod(self, run: RunState, j: JokerInst) -> int:
        return 0

    def free_rerolls(self, run: RunState, j: JokerInst) -> int:
        return 0

    def debt_allowance(self, run: RunState, j: JokerInst) -> int:
        return 0

    def interest_bonus(self, run: RunState, j: JokerInst) -> int:
        return 0

    def chance_multiplier(self, run: RunState, j: JokerInst) -> int:
        return 1

    def allows_duplicates(self, run: RunState, j: JokerInst) -> bool:
        return False

    def item_is_free(self, run: RunState, j: JokerInst, kind: str, item_id: str) -> bool:
        """Astronomer: planets and celestial packs are free."""
        return False


@dataclass(frozen=True)
class JokerDef:
    id: str
    name: str
    rarity: Rarity
    base_cost: int
    effect: JokerEffect
    copyable: bool = True            # False for passive/N-A jokers (Blueprint no-ops)
    copy_target: str | None = None   # "right" (Blueprint) | "leftmost" (Brainstorm)
    init_state: dict = field(default_factory=dict)
    eternal_compat: bool = True      # False: never spawns Eternal (self-destructing jokers)
    perishable_compat: bool = True   # False: never spawns Perishable (scaling jokers)
    pool_requires: str | None = None  # stats flag that must be set for the shop pool (Cavendish)
    pool_excludes: str | None = None  # stats flag that removes it from the pool (Gros Michel)


JOKERS: dict[str, JokerDef] = {}


def register_joker(defn: JokerDef) -> JokerDef:
    if defn.id in JOKERS:
        raise ValueError(f"duplicate joker id: {defn.id}")
    JOKERS[defn.id] = defn
    return defn


def joker_def(joker_id: str) -> JokerDef:
    return JOKERS[joker_id]


def new_joker_state(joker_id: str) -> dict:
    return dict(JOKERS[joker_id].init_state)


def joker_active(run: RunState, j: JokerInst) -> bool:
    """False when the joker is debuffed (Perishable expired / Crimson Heart)."""
    if j.debuffed:
        return False
    rnd = run.round
    if rnd is not None and rnd.boss_state.get("disabled_joker_uid") == j.uid:
        # Crimson Heart's victim is only active once the boss itself is disabled.
        return bool(rnd.boss_state.get("disabled", False))
    return True


def resolve_joker(
    jokers: list[JokerInst], idx: int, run: RunState | None = None
) -> tuple[JokerEffect, JokerInst] | None:
    """Resolve slot `idx` through copy chains.

    Returns (effect, state_owner). state_owner is the final target instance —
    hooks read ITS state. Returns None when the slot resolves to nothing
    (empty copy target, cycle, uncopyable final target, or a debuffed joker
    anywhere along the chain).
    """
    visited = {idx}
    cur = idx
    copied = False
    while True:
        inst = jokers[cur]
        if run is not None and not joker_active(run, inst):
            return None
        defn = JOKERS[inst.id]
        if defn.copy_target is None:
            if copied and not defn.copyable:
                return None
            return defn.effect, inst
        # follow the copy pointer
        if defn.copy_target == "right":
            nxt = cur + 1
        elif defn.copy_target == "leftmost":
            nxt = 0
        else:  # pragma: no cover - registration error
            raise ValueError(f"bad copy_target {defn.copy_target!r} on {defn.id}")
        if nxt >= len(jokers) or nxt in visited:
            return None
        visited.add(nxt)
        cur = nxt
        copied = True


def is_copy_slot(jokers: list[JokerInst], idx: int) -> bool:
    return JOKERS[jokers[idx].id].copy_target is not None


def active_jokers(run: RunState) -> list[JokerInst]:
    """Physical jokers whose own (non-copy) hooks should fire."""
    return [j for j in run.jokers if joker_active(run, j)]


# ---------------------------------------------------------------------------
# Consumables
# ---------------------------------------------------------------------------

class ConsumableEffect:
    # arity of hand-card targets: (min, max); (0, 0) = untargeted
    target_arity: tuple[int, int] = (0, 0)

    def can_use(self, run: RunState) -> bool:
        return True

    def use(self, run: RunState, targets: list[Card]) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ConsumableDef:
    id: str
    name: str
    kind: ConsumableKind
    base_cost: int
    effect: ConsumableEffect
    in_pool: bool = True          # False: never rolled from shops/packs normally
    pool_weight: float = 1.0      # hidden cards: per-slot chance to replace a pack option
    hidden_in: tuple[str, ...] = ()  # pack categories where a hidden card may appear
    track_last: bool = True       # False: using it does not update run.last_consumable (The Fool)


CONSUMABLES: dict[str, ConsumableDef] = {}

# HandType -> planet consumable id; filled by content/consumables/planets.py.
# Consumed by round.py (blue seal), shop.py (Telescope) and obs encoding.
PLANET_FOR_HAND: dict = {}


def register_consumable(defn: ConsumableDef) -> ConsumableDef:
    if defn.id in CONSUMABLES:
        raise ValueError(f"duplicate consumable id: {defn.id}")
    CONSUMABLES[defn.id] = defn
    return defn


def consumable_def(cid: str) -> ConsumableDef:
    return CONSUMABLES[cid]


# ---------------------------------------------------------------------------
# Boss blinds
# ---------------------------------------------------------------------------

class BossEffect:
    min_ante: int = 1
    is_finisher: bool = False
    triggers_every_hand: bool = False  # Matador: ability fires on every played hand

    def score_multiplier(self) -> float:
        """Multiplier of the ante base chip requirement (default boss 2x)."""
        return 2.0

    def reward(self) -> int:
        return 8 if self.is_finisher else 5

    def on_blind_start(self, run: RunState) -> None:
        """Manacle -1 hand size, Needle hands=1, Water discards=0, ..."""

    def card_debuffed(self, run: RunState, card: Card) -> bool:
        return False

    def validate_play(self, run: RunState, cards: list[Card]) -> str | None:
        """Return a reason string if this play is illegal (Psychic/Eye/Mouth/Bell)."""
        return None

    def validate_discard(self, run: RunState, cards: list[Card]) -> str | None:
        """Cerulean Bell: the forced card must be part of every selection."""
        return None

    def modify_base_score(
        self, ctx: ScoreContext, chips: float, mult: float
    ) -> tuple[float, float]:
        """The Flint: halve base chips and mult."""
        return chips, mult

    def on_hand_played(self, ctx: ScoreContext) -> None:
        """The Arm de-level, The Ox money=0, The Tooth -$1/card. Runs pre-base."""

    def after_hand(self, run: RunState) -> None:
        """The Hook: discard 2 random cards after each played hand."""

    def after_discard(self, run: RunState) -> None: ...

    def after_refill(self, run: RunState) -> None:
        """Runs after every draw (Cerulean Bell re-picks its forced card)."""

    def draw_face_down(self, run: RunState, card: Card) -> bool:
        return False

    def draw_count(self, run: RunState) -> int | None:
        """The Serpent: fixed number of cards to draw after a play/discard."""
        return None

    def on_joker_sold(self, run: RunState) -> None:
        """Verdant Leaf lifts its debuff."""

    def on_defeat(self, run: RunState) -> None: ...


@dataclass(frozen=True)
class BossDef:
    id: str
    name: str
    effect: BossEffect


BOSSES: dict[str, BossDef] = {}


def register_boss(defn: BossDef) -> BossDef:
    if defn.id in BOSSES:
        raise ValueError(f"duplicate boss id: {defn.id}")
    BOSSES[defn.id] = defn
    return defn


def boss_def(bid: str) -> BossDef:
    return BOSSES[bid]


# ---------------------------------------------------------------------------
# Decks / Vouchers / Tags / Packs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeckDef:
    id: str
    name: str
    apply: object  # Callable[[RunState], None] — applied once at run start


DECKS: dict[str, DeckDef] = {}


def register_deck(defn: DeckDef) -> DeckDef:
    if defn.id in DECKS:
        raise ValueError(f"duplicate deck id: {defn.id}")
    DECKS[defn.id] = defn
    return defn


@dataclass(frozen=True)
class VoucherDef:
    id: str
    name: str
    cost: int
    apply: object  # Callable[[RunState], None] — applied once on purchase
    requires: str | None = None


VOUCHERS: dict[str, VoucherDef] = {}


def register_voucher(defn: VoucherDef) -> VoucherDef:
    if defn.id in VOUCHERS:
        raise ValueError(f"duplicate voucher id: {defn.id}")
    VOUCHERS[defn.id] = defn
    return defn


class TagEffect:
    """Skip-tag behaviour. Each hook returns True when the tag is consumed."""

    def on_acquire(self, run: RunState, tag: TagInst) -> bool:
        """Immediate tags (Handy, Economy, Top-up, pack tags, Boss)."""
        return False

    def on_shop_open(self, run: RunState, tag: TagInst, shop: ShopState) -> bool:
        """Shop tags (Uncommon/Rare/edition/Voucher/Coupon/D6)."""
        return False

    def on_round_start(self, run: RunState, tag: TagInst) -> bool:
        """Juggle."""
        return False

    def round_end_money(self, run: RunState, tag: TagInst) -> tuple[int, bool]:
        """(dollars, consumed) at round end — Investment after a boss."""
        return 0, False

    def on_tag_acquired(self, run: RunState, tag: TagInst, new: TagInst) -> bool:
        """Double Tag: another tag was just acquired."""
        return False


@dataclass(frozen=True)
class TagDef:
    id: str
    name: str
    effect: TagEffect
    min_ante: int = 1
    init_state: dict = field(default_factory=dict)


TAGS: dict[str, TagDef] = {}


def register_tag(defn: TagDef) -> TagDef:
    if defn.id in TAGS:
        raise ValueError(f"duplicate tag id: {defn.id}")
    TAGS[defn.id] = defn
    return defn


@dataclass(frozen=True)
class PackDef:
    id: str
    name: str
    category: str  # "standard" | "arcana" | "celestial" | "buffoon" | "spectral"
    cost: int
    options: int   # cards shown
    picks: int     # cards chosen
    weight: float  # shop appearance weight


PACKS: dict[str, PackDef] = {}


def register_pack(defn: PackDef) -> PackDef:
    if defn.id in PACKS:
        raise ValueError(f"duplicate pack id: {defn.id}")
    PACKS[defn.id] = defn
    return defn


@dataclass(frozen=True)
class StakeDef:
    """Stake modifiers are applied cumulatively (every lower stake too)."""

    stake: object  # Stake
    apply: object  # Callable[[RunState], None]


STAKES: dict = {}


def register_stake(defn: StakeDef) -> StakeDef:
    STAKES[defn.stake] = defn
    return defn
