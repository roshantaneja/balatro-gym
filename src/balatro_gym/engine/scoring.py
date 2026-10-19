"""The scoring pipeline: a direct transcription of the game's activation sequence.

Stages:
  1. Boss effects (Flint halves base; Arm de-levels — pre-base)
  2. On-Played jokers (and their state updates: Green Joker, Ride the Bus, ...)
  3. Scored cards left->right; per card, retrigger-looped:
       base chips -> enhancement -> seal (gold) -> edition -> On-Scored jokers
     Retrigger order: red seal first, then retrigger jokers left->right.
  4. Held-in-hand cards left->right, retrigger-looped:
       Steel -> On-Held jokers
  5. Per joker slot left->right:
       Foil/Holo -> on_independent -> on_other_joker pass (Baseball) -> Poly
     Independent jokers are never retriggered.
  6. Observatory: x1.5 Mult per Planet in the consumable area that matches the
     played hand type (run.mods["observatory"]).
  7. Plasma balancing: chips and mult are averaged (run.mods["plasma"]).

Every chips/mult/money mutation goes through ScoreContext and appends a
ScoreEvent — the audit log is the fidelity contract, always on.

Debuffed jokers (Perishable expiry, Crimson Heart) are filtered out of the
slot list, so none of their hooks run. Boss effects are ignored when the boss
has been disabled (Chicot, Luchador).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import Card
from .effects import (
    PLANET_FOR_HAND,
    BossEffect,
    JokerEffect,
    active_jokers,
    boss_def,
    is_copy_slot,
    joker_def,
    resolve_joker,
)
from .hand_eval import DEFAULT_RULES, EvalRules, HandResult, evaluate_hand, is_face, suit_matches
from .levels import hand_value
from .state import JokerInst, RunState
from .types import Edition, Enhancement, Seal, Suit


@dataclass(slots=True)
class ScoreEvent:
    stage: str        # "base" | "on_played" | "card_scored[2]" | "card_scored[2]/retrigger:red_seal" | ...
    source_kind: str  # "hand" | "boss" | "card" | "enhancement" | "seal" | "edition" | "joker" | "voucher" | "deck"
    source_id: str
    op: str           # "+c" | "+m" | "xm" | "+$" | "retrigger" | "info"
    amount: float
    chips_after: float
    mult_after: float


class ScoreContext:
    """Mutable accumulator threaded through one hand's scoring."""

    __slots__ = (
        "_stage",
        "boss",
        "boss_triggered",
        "chips",
        "eval_rules",
        "events",
        "hand_result",
        "held",
        "is_copy",
        "lucky_triggers",
        "money_delta",
        "mult",
        "played",
        "run",
    )

    def __init__(
        self,
        run: RunState,
        played: list[Card],
        held: list[Card],
        hand_result: HandResult,
        eval_rules: EvalRules,
        boss: BossEffect | None,
    ):
        self.run = run
        self.played = played
        self.held = held
        self.hand_result = hand_result
        self.eval_rules = eval_rules
        self.boss = boss
        self.chips = 0.0
        self.mult = 0.0
        self.money_delta = 0
        self.events: list[ScoreEvent] = []
        self._stage = "init"
        self.is_copy = False          # True while dispatching a Blueprint/Brainstorm slot
        self.boss_triggered = False   # Matador
        self.lucky_triggers = 0       # Lucky Cat

    # ------------------------------------------------------------- mutators
    def _log(self, source_kind: str, source_id: str, op: str, amount: float) -> None:
        self.events.append(
            ScoreEvent(self._stage, source_kind, source_id, op, amount, self.chips, self.mult)
        )

    def add_chips(self, n: float, source_kind: str, source_id: str) -> None:
        if n == 0:
            return
        self.chips += n
        self._log(source_kind, source_id, "+c", n)

    def add_mult(self, n: float, source_kind: str, source_id: str) -> None:
        if n == 0:
            return
        self.mult += n
        self._log(source_kind, source_id, "+m", n)

    def mul_mult(self, x: float, source_kind: str, source_id: str) -> None:
        if x == 1:
            return
        self.mult *= x
        self._log(source_kind, source_id, "xm", x)

    def add_money(self, n: int, source_kind: str, source_id: str) -> None:
        if n == 0:
            return
        self.money_delta += n
        self._log(source_kind, source_id, "+$", n)

    def note(self, source_kind: str, source_id: str) -> None:
        self._log(source_kind, source_id, "info", 0.0)

    # -------------------------------------------------------------- queries
    def chance(self, num: int, den: int) -> bool:
        """Seeded probability roll; Oops! All 6s multiplies num."""
        eff = min(1.0, (num * self.run.chance_multiplier) / den)
        return self.run.rng.stream("chance").random() < eff

    def is_debuffed(self, card: Card) -> bool:
        if self.boss is not None and self.boss.card_debuffed(self.run, card):
            return True
        return False

    def card_has_suit(self, card: Card, suit: Suit) -> bool:
        if self.is_debuffed(card) and card.enhancement == Enhancement.WILD:
            # Debuffed Wild reverts to its printed suit.
            return card.suit == suit
        return suit_matches(card, suit, self.eval_rules)

    def is_face(self, card: Card) -> bool:
        return is_face(card, self.eval_rules)

    def is_final_hand(self) -> bool:
        """True while scoring the last hand of the round (hands_left is
        decremented after score_play returns)."""
        rnd = self.run.round
        return rnd is not None and rnd.hands_left == 1

    def is_first_hand(self) -> bool:
        rnd = self.run.round
        return rnd is not None and rnd.hands_played == 0


@dataclass(slots=True)
class ScoreResult:
    hand_result: HandResult
    chips: float
    mult: float
    score: float          # floor(chips * mult)
    money_delta: int
    events: list[ScoreEvent]
    broken_card_uids: list[int] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            f"{'stage':<38} {'source':<28} {'op':>4} {'amount':>10} {'chips':>12} {'mult':>10}"
        ]
        for e in self.events:
            src = f"{e.source_kind}:{e.source_id}"
            lines.append(
                f"{e.stage:<38} {src:<28} {e.op:>4} {e.amount:>10.2f} {e.chips_after:>12.2f} {e.mult_after:>10.2f}"
            )
        lines.append(
            f"TOTAL: {self.chips:.2f} chips x {self.mult:.2f} mult = {self.score:.0f}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "hand_type": int(self.hand_result.hand_type),
            "chips": self.chips,
            "mult": self.mult,
            "score": self.score,
            "money_delta": self.money_delta,
            "events": [
                {
                    "stage": e.stage,
                    "source_kind": e.source_kind,
                    "source_id": e.source_id,
                    "op": e.op,
                    "amount": e.amount,
                    "chips_after": e.chips_after,
                    "mult_after": e.mult_after,
                }
                for e in self.events
            ],
        }


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def fold_eval_rules(run: RunState) -> EvalRules:
    rules = DEFAULT_RULES
    for j in active_jokers(run):
        rules = joker_def(j.id).effect.modify_eval_rules(rules, j)
    return rules


def _resolved_slots(run: RunState) -> list[tuple[int, JokerEffect, JokerInst]]:
    """(slot_index, effect, state_owner) for every slot that resolves."""
    out = []
    for i in range(len(run.jokers)):
        r = resolve_joker(run.jokers, i, run)
        if r is not None:
            out.append((i, r[0], r[1]))
    return out


def get_boss(run: RunState) -> BossEffect | None:
    rnd = run.round
    if rnd is not None and rnd.boss_id is not None and not rnd.boss_state.get("disabled"):
        return boss_def(rnd.boss_id).effect
    return None


def _dispatch(ctx: ScoreContext, run: RunState, i: int, fn, *args) -> None:
    ctx.is_copy = is_copy_slot(run.jokers, i)
    fn(*args)
    ctx.is_copy = False


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def score_play(
    run: RunState,
    played: list[Card],
    held: list[Card] | None = None,
) -> ScoreResult:
    """Score a played selection. Mutates run (money, hand counts, joker state,
    glass breakage in full_deck) but does NOT touch round bookkeeping
    (hands_left, drawing) — that's round.py's job.
    """
    if held is None:
        if run.round is not None:
            played_uids = {c.uid for c in played}
            held = [c for c in run.round.hand if c.uid not in played_uids]
        else:
            held = []

    boss = get_boss(run)
    rules = fold_eval_rules(run)
    hr = evaluate_hand(played, rules)
    ctx = ScoreContext(run, played, held, hr, rules, boss)
    slots = _resolved_slots(run)

    # --- stage 1: boss, then base -------------------------------------------------
    ctx._stage = "boss"
    if boss is not None:
        boss.on_hand_played(ctx)  # The Arm de-levels BEFORE base is read
        if any(ctx.is_debuffed(c) for c in played):
            ctx.boss_triggered = True

    ctx._stage = "base"
    base_c, base_m = hand_value(hr.hand_type, run.hand_levels[hr.hand_type])
    if boss is not None:
        nc, nm = boss.modify_base_score(ctx, base_c, base_m)  # The Flint
        if (nc, nm) != (base_c, base_m):
            ctx.boss_triggered = True
        base_c, base_m = nc, nm
    ctx.chips = base_c
    ctx.mult = base_m
    ctx._log("hand", hr.hand_type.name.lower(), "info", 0.0)

    # --- stage 2: on-played jokers + their state updates --------------------------
    ctx._stage = "on_played"
    for i, eff, owner in slots:
        _dispatch(ctx, run, i, eff.on_hand_played, ctx, owner)
        if not is_copy_slot(run.jokers, i):
            eff.update_on_hand_played(run, run.jokers[i], played, hr)

    # --- stage 3: scored cards L->R ------------------------------------------------
    for ci, card in enumerate(hr.scoring_cards):
        if ctx.is_debuffed(card):
            ctx._stage = f"card_scored[{ci}]"
            ctx._log("card", repr(card), "info", 0.0)  # debuffed: contributes nothing
            continue
        # Build the retrigger list once, up front (static per card).
        passes: list[str] = ["initial"]
        if card.seal == Seal.RED:
            passes.append("retrigger:red_seal")
        for i, eff, owner in slots:
            n = eff.retrigger_scored(ctx, owner, card)
            passes.extend([f"retrigger:{run.jokers[i].id}"] * n)
        for pass_name in passes:
            suffix = "" if pass_name == "initial" else f"/{pass_name}"
            ctx._stage = f"card_scored[{ci}]{suffix}"
            _score_one_card(ctx, slots, run, card)

    # --- stage 4: held-in-hand L->R -------------------------------------------------
    for hi, card in enumerate(held):
        if ctx.is_debuffed(card):
            continue
        passes = ["initial"]
        if card.seal == Seal.RED:
            passes.append("retrigger:red_seal")
        for i, eff, owner in slots:
            n = eff.retrigger_held(ctx, owner, card)
            passes.extend([f"retrigger:{run.jokers[i].id}"] * n)
        for pass_name in passes:
            suffix = "" if pass_name == "initial" else f"/{pass_name}"
            ctx._stage = f"card_held[{hi}]{suffix}"
            if card.enhancement == Enhancement.STEEL:
                ctx.mul_mult(1.5, "enhancement", "steel")
            for i, eff, owner in slots:
                _dispatch(ctx, run, i, eff.on_card_held, ctx, owner, card)

    # --- stage 5: joker editions + independent + on-other-joker + poly -------------
    for i, eff, owner in slots:
        inst = run.jokers[i]  # editions belong to the PHYSICAL joker
        ctx._stage = f"joker[{i}]:{inst.id}"
        if inst.edition == Edition.FOIL:
            ctx.add_chips(50, "edition", "foil")
        elif inst.edition == Edition.HOLO:
            ctx.add_mult(10, "edition", "holo")
        _dispatch(ctx, run, i, eff.on_independent, ctx, owner)
        # Baseball-style pass: every effect gets a look at this joker.
        for k, eff2, owner2 in slots:
            if k != i:
                _dispatch(ctx, run, k, eff2.on_other_joker, ctx, owner2, inst)
        if inst.edition == Edition.POLY:
            ctx.mul_mult(1.5, "edition", "poly")

    # --- stage 6: Observatory -------------------------------------------------------
    obs_x = run.mods.get("observatory", 0)
    if obs_x:
        ctx._stage = "observatory"
        planet_id = PLANET_FOR_HAND.get(hr.hand_type)
        for c in run.consumables:
            if planet_id is not None and c.id == planet_id:
                ctx.mul_mult(obs_x, "voucher", "observatory")

    # --- stage 7: Plasma balancing --------------------------------------------------
    if run.mods.get("plasma"):
        ctx._stage = "plasma"
        avg = (ctx.chips + ctx.mult) / 2.0
        ctx.chips = avg
        ctx.mult = avg
        ctx._log("deck", "plasma", "info", avg)

    score = float(int(ctx.chips * ctx.mult))  # floor, matching community calculator

    # --- post-scoring bookkeeping ---------------------------------------------------
    run.money += ctx.money_delta
    run.hand_play_counts[hr.hand_type] = run.hand_play_counts.get(hr.hand_type, 0) + 1
    run.stats["hands_played_run"] = run.stats.get("hands_played_run", 0) + 1
    if hr.hand_type not in run.hands_discovered:
        run.hands_discovered.add(hr.hand_type)
    if ctx.lucky_triggers:
        run.stats["lucky_triggers"] = run.stats.get("lucky_triggers", 0) + ctx.lucky_triggers
        for j in active_jokers(run):
            for _ in range(ctx.lucky_triggers):
                joker_def(j.id).effect.update_on_lucky(run, j)
    if ctx.boss_triggered and run.round is not None:
        run.round.boss_state["triggered"] = True

    broken: list[Card] = []
    for card in hr.scoring_cards:
        if (
            card.enhancement == Enhancement.GLASS
            and not ctx.is_debuffed(card)
            and ctx.chance(1, 4)
        ):
            broken.append(card)
    result = ScoreResult(
        hand_result=hr,
        chips=ctx.chips,
        mult=ctx.mult,
        score=score,
        money_delta=ctx.money_delta,
        events=ctx.events,
        broken_card_uids=[c.uid for c in broken],
    )
    if broken:
        from .cardops import destroy_cards

        destroy_cards(run, broken)

    # Post-hand state updates (originals only), then deferred self-destructs.
    for j in active_jokers(run):
        joker_def(j.id).effect.update_after_scoring(run, j, result)
    from .cardops import flush_pending_destroys

    flush_pending_destroys(run)
    return result


def _score_one_card(
    ctx: ScoreContext,
    slots: list[tuple[int, JokerEffect, JokerInst]],
    run: RunState,
    card: Card,
) -> None:
    """One pass of the per-card pipeline (stage 3.1-3.3). Retriggers rerun all of it."""
    ctx.add_chips(card.base_chips, "card", repr(card))

    enh = card.enhancement
    if enh == Enhancement.BONUS:
        ctx.add_chips(30, "enhancement", "bonus")
    elif enh == Enhancement.MULT:
        ctx.add_mult(4, "enhancement", "mult")
    elif enh == Enhancement.GLASS:
        ctx.mul_mult(2.0, "enhancement", "glass")
    elif enh == Enhancement.LUCKY:
        if ctx.chance(1, 5):
            ctx.add_mult(20, "enhancement", "lucky")
            ctx.lucky_triggers += 1
        if ctx.chance(1, 15):
            ctx.add_money(20, "enhancement", "lucky")
            ctx.lucky_triggers += 1

    if card.seal == Seal.GOLD:
        ctx.add_money(3, "seal", "gold")

    if card.edition == Edition.FOIL:
        ctx.add_chips(50, "edition", "foil")
    elif card.edition == Edition.HOLO:
        ctx.add_mult(10, "edition", "holo")
    elif card.edition == Edition.POLY:
        ctx.mul_mult(1.5, "edition", "poly")

    for i, eff, owner in slots:
        _dispatch(ctx, run, i, eff.on_card_scored, ctx, owner, card)
