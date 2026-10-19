"""All 28 boss blinds (23 regular + 5 ante-8 finishers).

Judgment calls (documented where the rule text is terse):
- The Flint: base chips and mult are halved, rounded UP.
- The Arm: the de-level is a permanent mutation of run.hand_levels (min 1),
  applied before base chips/mult are read.
- The Ox: the "most played hand" is fixed when the blind starts; ties go to
  the lowest hand type (the game seeds its tracker with High Card and only a
  strictly greater count displaces it).
- Suit debuffs are matched against the PRINTED suit (a debuffed Wild card
  reverts to its base suit); Stone cards are never suit-debuffed.
- The House: only the initial deal is face down.
- The Plant / The Mark: "face card" honours Pareidolia (folded eval rules).
- The Pillar: every card played earlier this ante is debuffed, including
  cards played earlier in this very round.
- Cerulean Bell: a random card in hand is "forced"; every play and discard
  must include it. A new card is forced after every draw.
- Crimson Heart: a random joker is disabled when the blind starts and again
  after every hand played (it may be the same one).
- Amber Acorn: jokers are shuffled and hidden (no reordering, ids masked).
- Verdant Leaf: all cards are debuffed until any joker is sold.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from balatro_gym.engine.effects import BossDef, BossEffect, register_boss
from balatro_gym.engine.hand_eval import evaluate_hand, is_face
from balatro_gym.engine.types import HandType, Suit

if TYPE_CHECKING:
    from balatro_gym.engine.cards import Card
    from balatro_gym.engine.scoring import ScoreContext
    from balatro_gym.engine.state import RunState


def _rules(run: RunState):
    from balatro_gym.engine.scoring import fold_eval_rules

    return fold_eval_rules(run)


def _trigger(run: RunState) -> None:
    if run.round is not None:
        run.round.boss_state["triggered"] = True


# ---------------------------------------------------------------------------
# Ante 1
# ---------------------------------------------------------------------------

class _Hook(BossEffect):
    triggers_every_hand = True
    """Discards 2 random cards held in hand after every played hand."""

    def after_hand(self, run: RunState) -> None:
        rnd = run.round
        assert rnd is not None
        rng = run.rng.stream("boss")
        for _ in range(min(2, len(rnd.hand))):
            card = rnd.hand.pop(rng.randrange(len(rnd.hand)))
            rnd.discarded.append(card)
            if card.uid in rnd.face_down_uids:
                rnd.face_down_uids.remove(card.uid)
        _trigger(run)


class _SuitDebuff(BossEffect):
    """All cards of one PRINTED suit are debuffed (Club/Goad/Window/Head)."""

    def __init__(self, suit: Suit):
        self.suit = suit

    def card_debuffed(self, run: RunState, card: Card) -> bool:
        return not card.is_stone and card.suit == self.suit


class _Psychic(BossEffect):
    """Must play 5 cards (not all cards need to score)."""

    def validate_play(self, run: RunState, cards: list[Card]) -> str | None:
        if len(cards) != 5:
            return "The Psychic: must play exactly 5 cards"
        return None


class _Manacle(BossEffect):
    """-1 Hand Size."""

    def on_blind_start(self, run: RunState) -> None:
        assert run.round is not None
        run.round.hand_size_bonus -= 1


class _Pillar(BossEffect):
    """Cards played previously this ante are debuffed."""

    def card_debuffed(self, run: RunState, card: Card) -> bool:
        return card.uid in run.ante_played_uids


# ---------------------------------------------------------------------------
# Ante 2
# ---------------------------------------------------------------------------

class _House(BossEffect):
    """First hand is drawn face down."""

    min_ante = 2

    def draw_face_down(self, run: RunState, card: Card) -> bool:
        rnd = run.round
        assert rnd is not None
        return rnd.hands_played == 0 and not rnd.played_uids and not rnd.discarded


class _Wheel(BossEffect):
    """1 in 7 cards get drawn face down."""

    min_ante = 2

    def draw_face_down(self, run: RunState, card: Card) -> bool:
        return run.rng.stream("boss").random() < 1 / 7


class _Arm(BossEffect):
    """Decrease level of played poker hand by 1 (min level 1, pre-scoring)."""

    min_ante = 2

    def on_hand_played(self, ctx: ScoreContext) -> None:
        ht = ctx.hand_result.hand_type
        lvl = ctx.run.hand_levels[ht]
        if lvl > 1:
            ctx.run.hand_levels[ht] = lvl - 1
            ctx.boss_triggered = True


class _Flint(BossEffect):
    """Base Chips and Mult for played poker hands are halved (rounded up)."""

    min_ante = 2

    def modify_base_score(
        self, ctx: ScoreContext, chips: float, mult: float
    ) -> tuple[float, float]:
        return float(math.ceil(chips / 2)), float(math.ceil(mult / 2))


class _Wall(BossEffect):
    """Extra large blind: 4x base."""

    min_ante = 2

    def score_multiplier(self) -> float:
        return 4.0


class _Needle(BossEffect):
    """Play only 1 hand; requirement drops to 1x base."""

    min_ante = 2

    def score_multiplier(self) -> float:
        return 1.0

    def on_blind_start(self, run: RunState) -> None:
        assert run.round is not None
        run.round.hands_left = 1


class _Fish(BossEffect):
    triggers_every_hand = True
    """Cards drawn face down after each hand played."""

    min_ante = 2

    def draw_face_down(self, run: RunState, card: Card) -> bool:
        rnd = run.round
        assert rnd is not None
        return rnd.hands_played > 0 and rnd.boss_state.get("last_action") == "play"

    def after_hand(self, run: RunState) -> None:
        assert run.round is not None
        run.round.boss_state["last_action"] = "play"
        _trigger(run)

    def after_discard(self, run: RunState) -> None:
        assert run.round is not None
        run.round.boss_state["last_action"] = "discard"


class _Water(BossEffect):
    """Start with 0 discards."""

    min_ante = 2

    def on_blind_start(self, run: RunState) -> None:
        assert run.round is not None
        run.round.discards_left = 0


class _Mouth(BossEffect):
    """Play only 1 hand type this round."""

    min_ante = 2

    def validate_play(self, run: RunState, cards: list[Card]) -> str | None:
        rnd = run.round
        assert rnd is not None
        if not rnd.hand_types_played:
            return None
        ht = evaluate_hand(cards, _rules(run)).hand_type
        if int(ht) != rnd.hand_types_played[0]:
            return "The Mouth: only one hand type may be played this round"
        return None


class _Mark(BossEffect):
    """All face cards are drawn face down."""

    min_ante = 2

    def draw_face_down(self, run: RunState, card: Card) -> bool:
        return is_face(card, _rules(run))


# ---------------------------------------------------------------------------
# Ante 3+
# ---------------------------------------------------------------------------

class _Eye(BossEffect):
    """No repeat hand types this round."""

    min_ante = 3

    def validate_play(self, run: RunState, cards: list[Card]) -> str | None:
        rnd = run.round
        assert rnd is not None
        ht = evaluate_hand(cards, _rules(run)).hand_type
        if int(ht) in rnd.hand_types_played:
            return "The Eye: hand type already played this round"
        return None


class _Tooth(BossEffect):
    triggers_every_hand = True
    """Lose $1 per card played."""

    min_ante = 3

    def on_hand_played(self, ctx: ScoreContext) -> None:
        ctx.add_money(-len(ctx.played), "boss", "b_tooth")
        ctx.boss_triggered = True


class _Plant(BossEffect):
    """All face cards are debuffed."""

    min_ante = 4

    def card_debuffed(self, run: RunState, card: Card) -> bool:
        return is_face(card, _rules(run))


class _Serpent(BossEffect):
    triggers_every_hand = True
    """After playing or discarding, always draw 3 cards."""

    min_ante = 5

    def draw_count(self, run: RunState) -> int | None:
        return 3

    def after_hand(self, run: RunState) -> None:
        _trigger(run)


def _most_played_hand(run: RunState) -> HandType:
    best = HandType.HIGH_CARD
    best_n = run.hand_play_counts.get(best, 0)
    for ht in HandType:
        n = run.hand_play_counts.get(ht, 0)
        if n > best_n:
            best, best_n = ht, n
    return best


class _Ox(BossEffect):
    """Playing your most played hand this run sets money to $0."""

    min_ante = 6

    def on_blind_start(self, run: RunState) -> None:
        assert run.round is not None
        run.round.boss_state["ox_target"] = int(_most_played_hand(run))

    def on_hand_played(self, ctx: ScoreContext) -> None:
        rnd = ctx.run.round
        target = rnd.boss_state.get("ox_target") if rnd is not None else None
        if target is not None and int(ctx.hand_result.hand_type) == target:
            ctx.run.money = 0
            ctx.boss_triggered = True


# ---------------------------------------------------------------------------
# Finishers (ante 8)
# ---------------------------------------------------------------------------

class _AmberAcorn(BossEffect):
    """Flips and shuffles all Joker cards."""

    min_ante = 8
    is_finisher = True

    def on_blind_start(self, run: RunState) -> None:
        assert run.round is not None
        run.rng.stream("boss").shuffle(run.jokers)
        run.round.boss_state["jokers_hidden"] = True


class _VerdantLeaf(BossEffect):
    """All cards debuffed until 1 Joker sold."""

    min_ante = 8
    is_finisher = True

    def on_blind_start(self, run: RunState) -> None:
        assert run.round is not None
        run.round.boss_state["verdant"] = True

    def card_debuffed(self, run: RunState, card: Card) -> bool:
        rnd = run.round
        return rnd is not None and bool(rnd.boss_state.get("verdant"))

    def on_joker_sold(self, run: RunState) -> None:
        if run.round is not None:
            run.round.boss_state["verdant"] = False


class _VioletVessel(BossEffect):
    """Very large blind: 6x base."""

    min_ante = 8
    is_finisher = True

    def score_multiplier(self) -> float:
        return 6.0


class _CrimsonHeart(BossEffect):
    triggers_every_hand = True
    """One random Joker disabled every hand."""

    min_ante = 8
    is_finisher = True

    def _pick(self, run: RunState) -> None:
        rnd = run.round
        assert rnd is not None
        if run.jokers:
            rnd.boss_state["disabled_joker_uid"] = run.rng.stream("boss").choice(run.jokers).uid
        else:
            rnd.boss_state.pop("disabled_joker_uid", None)

    def on_blind_start(self, run: RunState) -> None:
        self._pick(run)

    def after_hand(self, run: RunState) -> None:
        self._pick(run)
        _trigger(run)


class _CeruleanBell(BossEffect):
    """Forces 1 card to always be selected."""

    min_ante = 8
    is_finisher = True

    def after_refill(self, run: RunState) -> None:
        rnd = run.round
        assert rnd is not None
        forced = rnd.boss_state.get("forced_uid")
        if forced is not None and any(c.uid == forced for c in rnd.hand):
            return
        if rnd.hand:
            rnd.boss_state["forced_uid"] = run.rng.stream("boss").choice(rnd.hand).uid
        else:
            rnd.boss_state.pop("forced_uid", None)

    def _check(self, run: RunState, cards: list[Card]) -> str | None:
        rnd = run.round
        assert rnd is not None
        forced = rnd.boss_state.get("forced_uid")
        if forced is not None and not any(c.uid == forced for c in cards):
            return "Cerulean Bell: the forced card must be selected"
        return None

    def validate_play(self, run: RunState, cards: list[Card]) -> str | None:
        return self._check(run, cards)

    def validate_discard(self, run: RunState, cards: list[Card]) -> str | None:
        return self._check(run, cards)


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

register_boss(BossDef("b_hook", "The Hook", _Hook()))
register_boss(BossDef("b_club", "The Club", _SuitDebuff(Suit.CLUBS)))
register_boss(BossDef("b_goad", "The Goad", _SuitDebuff(Suit.SPADES)))
register_boss(BossDef("b_window", "The Window", _SuitDebuff(Suit.DIAMONDS)))
register_boss(BossDef("b_head", "The Head", _SuitDebuff(Suit.HEARTS)))
register_boss(BossDef("b_psychic", "The Psychic", _Psychic()))
register_boss(BossDef("b_manacle", "The Manacle", _Manacle()))
register_boss(BossDef("b_pillar", "The Pillar", _Pillar()))
register_boss(BossDef("b_house", "The House", _House()))
register_boss(BossDef("b_wheel", "The Wheel", _Wheel()))
register_boss(BossDef("b_arm", "The Arm", _Arm()))
register_boss(BossDef("b_flint", "The Flint", _Flint()))
register_boss(BossDef("b_wall", "The Wall", _Wall()))
register_boss(BossDef("b_needle", "The Needle", _Needle()))
register_boss(BossDef("b_fish", "The Fish", _Fish()))
register_boss(BossDef("b_water", "The Water", _Water()))
register_boss(BossDef("b_mouth", "The Mouth", _Mouth()))
register_boss(BossDef("b_mark", "The Mark", _Mark()))
register_boss(BossDef("b_eye", "The Eye", _Eye()))
register_boss(BossDef("b_tooth", "The Tooth", _Tooth()))
register_boss(BossDef("b_plant", "The Plant", _Plant()))
register_boss(BossDef("b_serpent", "The Serpent", _Serpent()))
register_boss(BossDef("b_ox", "The Ox", _Ox()))
register_boss(BossDef("b_amber_acorn", "Amber Acorn", _AmberAcorn()))
register_boss(BossDef("b_verdant_leaf", "Verdant Leaf", _VerdantLeaf()))
register_boss(BossDef("b_violet_vessel", "Violet Vessel", _VioletVessel()))
register_boss(BossDef("b_crimson_heart", "Crimson Heart", _CrimsonHeart()))
register_boss(BossDef("b_cerulean_bell", "Cerulean Bell", _CeruleanBell()))
