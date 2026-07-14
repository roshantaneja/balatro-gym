"""Effect hook contracts and content registries.

The engine layer knows nothing about specific jokers: content registers
JokerDef/ConsumableDef/BossDef/... objects into the registries below, and the
scoring/round orchestrators dispatch through the hook interfaces.

Design invariants:
- Effects are STATELESS singletons; all per-instance mutability lives in
  JokerInst.state (a JSON-scalar dict).
- Scoring hooks (`on_*`) run for Blueprint/Brainstorm copies too, reading the
  copy target's state. State-mutation hooks (`update_*`) run ONLY for the
  original joker — that split is what makes copy semantics correct by
  construction (Blueprint + Green Joker pays the mult twice but scales once).
- Copy resolution is generic: JokerDef.copy_target is "right" / "leftmost" /
  None. Chains are followed with a cycle guard; uncopyable targets no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import ConsumableKind, Rarity

if TYPE_CHECKING:
    from .cards import Card
    from .hand_eval import EvalRules, HandResult
    from .scoring import ScoreContext
    from .state import JokerInst, RunState


# ---------------------------------------------------------------------------
# Jokers
# ---------------------------------------------------------------------------

class JokerEffect:
    """Base with do-nothing defaults. Override only what the joker needs.

    Hooks are listed in activation-sequence order (Guide: Activation Sequence).
    `ctx` is the live ScoreContext; `j` is the JokerInst whose STATE the hook
    should read (the copy target under Blueprint), while edition bonuses are
    handled by the orchestrator using the physical instance.
    """

    # --- scoring-time hooks (run for copies too) ---
    def on_hand_played(self, ctx: "ScoreContext", j: "JokerInst") -> None: ...

    def on_card_scored(self, ctx: "ScoreContext", j: "JokerInst", card: "Card") -> None: ...

    def retrigger_scored(self, ctx: "ScoreContext", j: "JokerInst", card: "Card") -> int:
        return 0

    def on_card_held(self, ctx: "ScoreContext", j: "JokerInst", card: "Card") -> None: ...

    def retrigger_held(self, ctx: "ScoreContext", j: "JokerInst", card: "Card") -> int:
        return 0

    def on_independent(self, ctx: "ScoreContext", j: "JokerInst") -> None: ...

    def on_other_joker(self, ctx: "ScoreContext", j: "JokerInst", other: "JokerInst") -> None:
        """Baseball Card: fires once per *other* joker during stage 5."""

    def on_discard(
        self, ctx_run: "RunState", j: "JokerInst", cards: list["Card"], last: bool
    ) -> None: ...

    # --- state-update hooks (originals only, never copies) ---
    def update_on_hand_played(
        self, run: "RunState", j: "JokerInst", played: list["Card"], hr: "HandResult"
    ) -> None: ...

    def update_on_discard(self, run: "RunState", j: "JokerInst", cards: list["Card"]) -> None: ...

    def update_on_round_end(self, run: "RunState", j: "JokerInst") -> None: ...

    # --- lifecycle / economy ---
    def round_end_money(self, run: "RunState", j: "JokerInst") -> int:
        return 0

    def on_blind_selected(self, run: "RunState", j: "JokerInst") -> None: ...

    def on_acquired(self, run: "RunState", j: "JokerInst") -> None: ...

    def on_sold(self, run: "RunState", j: "JokerInst") -> None: ...

    def on_consumable_used(self, run: "RunState", j: "JokerInst", kind: ConsumableKind) -> None: ...

    # --- polled rule-modification queries (passive) ---
    def modify_eval_rules(self, rules: "EvalRules", j: "JokerInst") -> "EvalRules":
        return rules


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


def resolve_joker(
    jokers: list["JokerInst"], idx: int
) -> tuple[JokerEffect, "JokerInst"] | None:
    """Resolve slot `idx` through copy chains.

    Returns (effect, state_owner). state_owner is the final target instance —
    hooks read ITS state. Returns None when the slot resolves to nothing
    (empty copy target, cycle, or uncopyable final target).
    """
    visited = {idx}
    cur = idx
    copied = False
    while True:
        defn = JOKERS[jokers[cur].id]
        if defn.copy_target is None:
            if copied and not defn.copyable:
                return None
            return defn.effect, jokers[cur]
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


def is_copy_slot(jokers: list["JokerInst"], idx: int) -> bool:
    return JOKERS[jokers[idx].id].copy_target is not None


# ---------------------------------------------------------------------------
# Consumables
# ---------------------------------------------------------------------------

class ConsumableEffect:
    # arity of hand-card targets: (min, max); (0, 0) = untargeted
    target_arity: tuple[int, int] = (0, 0)

    def can_use(self, run: "RunState") -> bool:
        return True

    def use(self, run: "RunState", targets: list["Card"]) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ConsumableDef:
    id: str
    name: str
    kind: ConsumableKind
    base_cost: int
    effect: ConsumableEffect


CONSUMABLES: dict[str, ConsumableDef] = {}

# HandType -> planet consumable id; filled by content/consumables/planets.py.
# Consumed by round.py (blue seal) and obs encoding.
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

    def score_multiplier(self) -> float:
        """Multiplier of the ante base chip requirement (default boss 2x)."""
        return 2.0

    def reward(self) -> int:
        return 8 if self.is_finisher else 5

    def on_blind_start(self, run: "RunState") -> None:
        """Manacle -1 hand size, Needle hands=1, Water discards=0, ..."""

    def card_debuffed(self, run: "RunState", card: "Card") -> bool:
        return False

    def validate_play(self, run: "RunState", cards: list["Card"]) -> str | None:
        """Return a reason string if this play is illegal (Psychic/Eye/Mouth)."""
        return None

    def modify_base_score(
        self, ctx: "ScoreContext", chips: float, mult: float
    ) -> tuple[float, float]:
        """The Flint: halve base chips and mult."""
        return chips, mult

    def on_hand_played(self, ctx: "ScoreContext") -> None:
        """The Arm de-level, The Ox money=0, The Tooth -$1/card. Runs pre-base."""

    def after_hand(self, run: "RunState") -> None:
        """The Hook: discard 2 random cards after each played hand."""

    def draw_face_down(self, run: "RunState", card: "Card") -> bool:
        return False

    def on_defeat(self, run: "RunState") -> None: ...


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
# Decks / Vouchers / Tags / Packs — simple def registries; behavior lives in
# small hook callables so M1 doesn't over-build.
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


@dataclass(frozen=True)
class TagDef:
    id: str
    name: str
    # hooks consulted by round/shop code; JSON-simple to keep M1 lean
    on_acquire: object | None = None      # Callable[[RunState], None]
    round_end_money: object | None = None  # Callable[[RunState], int] (Handy/Garbage/Investment)
    on_round_start: object | None = None   # Callable[[RunState], None] (Juggle)


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
