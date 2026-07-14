"""Misc v1 jokers: Ancient, decaying/self-destructing food, Fortune Teller,
Baseball Card, 8 Ball.

Self-destruct pattern: round.py's round-end hook loop iterates `run.jokers`
directly, so effects never mutate that list in place. Instead they REBIND
`run.jokers` to a filtered copy — the in-progress iterator keeps walking the
old list (no joker gets skipped), and the run sees the new list afterwards.
Mid-scoring removal is never safe (scoring.py stages 3/5 index into
run.jokers by slot), so hand-driven decay (Ice Cream) only flags-by-value
mid-round and physically melts at round end.
"""

from __future__ import annotations

from balatro_gym.engine.effects import CONSUMABLES, JOKERS, JokerDef, JokerEffect, register_joker
from balatro_gym.engine.state import ConsumableInst
from balatro_gym.engine.types import ConsumableKind, Rank, Rarity, Suit


def _self_destruct(run, j):
    """Remove j from run.jokers by REBINDING (see module docstring)."""
    run.jokers = [x for x in run.jokers if x.uid != j.uid]


class _AncientJoker(JokerEffect):
    """Each played card of [suit] gives x1.5 Mult when scored; the suit
    changes at end of round (never re-picking the current suit — wiki)."""

    def on_card_scored(self, ctx, j, card):
        if ctx.card_has_suit(card, Suit(j.state["suit"])):
            ctx.mul_mult(1.5, "joker", "j_ancient")

    def update_on_round_end(self, run, j):
        others = [int(s) for s in Suit if int(s) != j.state["suit"]]
        j.state["suit"] = run.rng.stream("misc").choice(others)


register_joker(
    JokerDef("j_ancient", "Ancient Joker", Rarity.RARE, 8, _AncientJoker(), init_state={"suit": 0})
)


class _IceCream(JokerEffect):
    """+100 Chips, -5 Chips for every hand played; melts at 0.

    Wiki: "the first hand will provide the full 100 chips, not 95" — i.e. the
    decrement conceptually lands AFTER the read. Our update hook runs at
    stage 2 (before the stage-5 read), so state["chips"] already reflects the
    post-hand value during scoring and the read compensates with +5. Between
    hands, state["chips"] equals the displayed in-game value. Physically
    removed at round end once it reaches 0 (mid-scoring removal is unsafe);
    until then it contributes nothing."""

    def on_independent(self, ctx, j):
        val = j.state["chips"] + 5
        if val > 0:
            ctx.add_chips(val, "joker", "j_ice_cream")

    def update_on_hand_played(self, run, j, played, hr):
        j.state["chips"] -= 5

    def update_on_round_end(self, run, j):
        if j.state["chips"] <= 0:
            _self_destruct(run, j)


register_joker(
    JokerDef("j_ice_cream", "Ice Cream", Rarity.COMMON, 5, _IceCream(), init_state={"chips": 100})
)


class _Popcorn(JokerEffect):
    """+20 Mult, -4 Mult per round played; eaten after the 5th round."""

    def on_independent(self, ctx, j):
        if j.state["mult"] > 0:
            ctx.add_mult(j.state["mult"], "joker", "j_popcorn")

    def update_on_round_end(self, run, j):
        j.state["mult"] -= 4
        if j.state["mult"] <= 0:
            _self_destruct(run, j)


register_joker(
    JokerDef("j_popcorn", "Popcorn", Rarity.COMMON, 5, _Popcorn(), init_state={"mult": 20})
)


class _GrosMichel(JokerEffect):
    """+15 Mult; 1 in 6 chance to go extinct at end of round."""

    def on_independent(self, ctx, j):
        ctx.add_mult(15, "joker", "j_gros_michel")

    def update_on_round_end(self, run, j):
        eff = min(1.0, run.chance_multiplier / 6)
        if run.rng.stream("chance").random() < eff:
            _self_destruct(run, j)


register_joker(JokerDef("j_gros_michel", "Gros Michel", Rarity.COMMON, 5, _GrosMichel()))


class _FortuneTeller(JokerEffect):
    """+1 Mult per Tarot card used this run (retroactive by construction)."""

    def on_independent(self, ctx, j):
        n = ctx.run.stats.get("tarots_used", 0)
        if n:
            ctx.add_mult(n, "joker", "j_fortune_teller")


register_joker(
    JokerDef("j_fortune_teller", "Fortune Teller", Rarity.COMMON, 6, _FortuneTeller())
)


class _BaseballCard(JokerEffect):
    """Each Uncommon joker gives x1.5 Mult (exactly Uncommon — not Rare+).

    Dispatched via the stage-5 on_other_joker pass: fires once per OTHER
    physical joker slot, right after that joker's own independent effect."""

    def on_other_joker(self, ctx, j, other):
        if JOKERS[other.id].rarity == Rarity.UNCOMMON:
            ctx.mul_mult(1.5, "joker", "j_baseball")


register_joker(JokerDef("j_baseball", "Baseball Card", Rarity.RARE, 8, _BaseballCard()))


class _EightBall(JokerEffect):
    """Each scored 8 has a 1 in 4 chance to create a Tarot card (if room).

    Tarot picked uniformly from registered TAROT consumables via the seeded
    "consumable" stream; no-op if none are registered or slots are full."""

    def on_card_scored(self, ctx, j, card):
        if card.is_stone or card.rank != Rank.EIGHT:
            return
        run = ctx.run
        if len(run.consumables) >= run.consumable_slots:
            return
        if not ctx.chance(1, 4):
            return
        tarot_ids = sorted(
            cid for cid, d in CONSUMABLES.items() if d.kind == ConsumableKind.TAROT
        )
        if not tarot_ids:
            return
        cid = run.rng.stream("consumable").choice(tarot_ids)
        run.consumables.append(
            ConsumableInst(id=cid, kind=ConsumableKind.TAROT, uid=run.new_uid())
        )


register_joker(JokerDef("j_8ball", "8 Ball", Rarity.COMMON, 5, _EightBall()))
