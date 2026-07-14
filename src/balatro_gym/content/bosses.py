"""Boss blinds (v1 set).

Source: balatro_wiki/wikitext/Main/"Blinds and Antes.wiki" (the individual
boss pages are all redirects into its List of Blinds table).

Judgment calls (documented where the wiki table is terse):
- The Flint: the table says base chips and mult are "halved"; the in-game
  behavior rounds the halves UP (e.g. High Card 5 chips / 1 mult -> 3 / 1),
  so we use ceil. For integer bases ceil(x/2) == round-half-up(x*0.5).
- The Arm: "Decrease level of played poker hand by 1 (hand levels can go as
  low as Level 1, and are reduced before scoring)". The reduction is a plain
  mutation of run.hand_levels, i.e. permanent for the run (the wiki does not
  scope it to the round, matching the game).
- The Ox: "Playing your most played hand this run sets money to $0". The
  target hand is fixed when the blind starts (the game displays a single
  hand type on the blind). Ties are broken toward the LOWEST HandType: the
  game seeds its most-played tracker with High Card and only a strictly
  greater count displaces it, so ties favor the earlier/lower baseline.
- Suit debuffs: matched against the PRINTED suit. Per the wiki, "Debuffed
  Wild Cards will return to their base suit" — so a Wild card is debuffed
  iff its printed suit matches, and Stone cards (no suit) are never
  suit-debuffed.
- The House: "First hand is drawn face down" — cards are dealt face down
  only during the initial deal (no hand played, nothing played or discarded
  yet); replacements after the first play or discard are face up.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from balatro_gym.engine.effects import BossDef, BossEffect, register_boss
from balatro_gym.engine.types import HandType, Suit

if TYPE_CHECKING:
    from balatro_gym.engine.cards import Card
    from balatro_gym.engine.scoring import ScoreContext
    from balatro_gym.engine.state import RunState


# ---------------------------------------------------------------------------
# Ante 1
# ---------------------------------------------------------------------------

class _Hook(BossEffect):
    """Discards 2 random cards held in hand after every played hand."""

    def after_hand(self, run: "RunState") -> None:
        rnd = run.round
        assert rnd is not None
        rng = run.rng.stream("boss")
        for _ in range(min(2, len(rnd.hand))):
            card = rnd.hand.pop(rng.randrange(len(rnd.hand)))
            rnd.discarded.append(card)
            if card.uid in rnd.face_down_uids:
                rnd.face_down_uids.remove(card.uid)


class _SuitDebuff(BossEffect):
    """All cards of one PRINTED suit are debuffed (Club/Goad/Window/Head)."""

    def __init__(self, suit: Suit):
        self.suit = suit

    def card_debuffed(self, run: "RunState", card: "Card") -> bool:
        # Stone cards have no suit; Wild cards revert to their printed suit
        # when debuffed, so the printed suit decides for them too.
        return not card.is_stone and card.suit == self.suit


class _Psychic(BossEffect):
    """Must play 5 cards (not all cards need to score)."""

    def validate_play(self, run: "RunState", cards: list["Card"]) -> str | None:
        if len(cards) != 5:
            return "The Psychic: must play exactly 5 cards"
        return None


class _Manacle(BossEffect):
    """-1 Hand Size."""

    def on_blind_start(self, run: "RunState") -> None:
        assert run.round is not None
        run.round.boss_state["hand_size_bonus"] = -1


# ---------------------------------------------------------------------------
# Ante 2
# ---------------------------------------------------------------------------

class _House(BossEffect):
    """First hand is drawn face down."""

    min_ante = 2

    def draw_face_down(self, run: "RunState", card: "Card") -> bool:
        rnd = run.round
        assert rnd is not None
        # True only for the initial deal: refill_hand runs before any play or
        # discard exactly once (from start_blind).
        return rnd.hands_played == 0 and not rnd.played_uids and not rnd.discarded


class _Arm(BossEffect):
    """Decrease level of played poker hand by 1 (min level 1, pre-scoring).

    The de-level is permanent for the run and happens BEFORE base chips/mult
    are read (score_play calls on_hand_played ahead of the base stage).
    """

    min_ante = 2

    def on_hand_played(self, ctx: "ScoreContext") -> None:
        ht = ctx.hand_result.hand_type
        lvl = ctx.run.hand_levels[ht]
        if lvl > 1:
            ctx.run.hand_levels[ht] = lvl - 1


class _Flint(BossEffect):
    """Base Chips and Mult for played poker hands are halved (rounded up)."""

    min_ante = 2

    def modify_base_score(
        self, ctx: "ScoreContext", chips: float, mult: float
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

    def on_blind_start(self, run: "RunState") -> None:
        assert run.round is not None
        run.round.hands_left = 1


# ---------------------------------------------------------------------------
# Ante 6
# ---------------------------------------------------------------------------

def _most_played_hand(run: "RunState") -> HandType:
    """Most played hand this run; ties favor the lowest HandType (see module
    docstring). With no hands played at all, defaults to High Card."""
    best = HandType.HIGH_CARD
    best_n = run.hand_play_counts.get(best, 0)
    for ht in HandType:  # ascending enum order -> strictly-greater displaces
        n = run.hand_play_counts.get(ht, 0)
        if n > best_n:
            best, best_n = ht, n
    return best


class _Ox(BossEffect):
    """Playing your most played hand this run sets money to $0."""

    min_ante = 6

    def on_blind_start(self, run: "RunState") -> None:
        assert run.round is not None
        run.round.boss_state["ox_target"] = int(_most_played_hand(run))

    def on_hand_played(self, ctx: "ScoreContext") -> None:
        rnd = ctx.run.round
        target = rnd.boss_state.get("ox_target") if rnd is not None else None
        if target is not None and int(ctx.hand_result.hand_type) == target:
            ctx.run.money = 0


# ---------------------------------------------------------------------------
# Finisher (ante 8)
# ---------------------------------------------------------------------------

class _VioletVessel(BossEffect):
    """Very large blind: 6x base. Finisher reward is $8 (base class)."""

    min_ante = 8
    is_finisher = True

    def score_multiplier(self) -> float:
        return 6.0


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
register_boss(BossDef("b_house", "The House", _House()))
register_boss(BossDef("b_arm", "The Arm", _Arm()))
register_boss(BossDef("b_flint", "The Flint", _Flint()))
register_boss(BossDef("b_wall", "The Wall", _Wall()))
register_boss(BossDef("b_needle", "The Needle", _Needle()))
register_boss(BossDef("b_ox", "The Ox", _Ox()))
register_boss(BossDef("b_violet_vessel", "Violet Vessel", _VioletVessel()))
