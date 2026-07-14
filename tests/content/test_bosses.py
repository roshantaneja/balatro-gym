"""Boss blind behavior (content/bosses.py).

NOTE on golden coverage: the golden YAML runner (tests/golden/test_golden.py)
builds a bare run via make_run() and calls score_play with run.round=None, so
get_boss() always resolves to None — it cannot express boss blinds. The exact
hand-derived Flint/Arm scores are therefore covered here as unit tests
instead of as 5x_*.yaml golden cases.
"""

from __future__ import annotations

import pytest

from balatro_gym.engine import (
    BlindKind,
    Discard,
    Game,
    HandType,
    IllegalActionError,
    Phase,
    PlayHand,
    new_blind_scenario,
    score_play,
)
from balatro_gym.engine.effects import BOSSES, boss_def
from balatro_gym.engine.round import blind_requirement, pick_boss, refill_hand
from balatro_gym.engine.state import RoundState

from tests.helpers import card, cards, make_run

ANTE1_BOSSES = {
    "b_hook", "b_club", "b_goad", "b_window", "b_head", "b_psychic", "b_manacle",
}


def boss_game(seed: str, boss_id: str, *, ante: int = 1, prep=None, **kw) -> Game:
    """A Game in PLAYING on a specific boss blind.

    pick_boss is random, so we start a boss scenario, then rebuild the round
    with the requested boss and re-run its on_blind_start before the deal.
    `prep(run)` runs before on_blind_start (e.g. to seed hand_play_counts).
    """
    g = new_blind_scenario(seed, ante=ante, blind_kind=BlindKind.BOSS, **kw)
    run = g.run
    boss = boss_def(boss_id).effect
    deck = list(run.full_deck)
    run.rng.stream("shuffle").shuffle(deck)
    run.round = RoundState(
        blind_kind=BlindKind.BOSS,
        boss_id=boss_id,
        chips_needed=blind_requirement(run, BlindKind.BOSS, boss),
        hands_left=run.hands_per_round,
        discards_left=run.discards_per_round,
        deck_remaining=deck,
    )
    run.phase = Phase.PLAYING
    if prep is not None:
        prep(run)
    boss.on_blind_start(run)
    refill_hand(run)
    return g


# ---------------------------------------------------------------------------
# Registration / requirement multipliers
# ---------------------------------------------------------------------------

def test_all_v1_bosses_registered():
    expected = ANTE1_BOSSES | {
        "b_house", "b_arm", "b_flint", "b_wall", "b_needle", "b_ox",
        "b_violet_vessel",
    }
    assert expected <= set(BOSSES)


def test_wall_requirement_4x_base():
    g = boss_game("wall", "b_wall", ante=2)
    assert g.run.round.chips_needed == 3200  # 800 * 4


def test_needle_requirement_1x_and_one_hand():
    g = boss_game("needle", "b_needle", ante=2)
    assert g.run.round.chips_needed == 800  # 800 * 1
    assert g.run.round.hands_left == 1


def test_violet_vessel_requirement_and_reward():
    g = boss_game("vv", "b_violet_vessel", ante=8)
    assert g.run.round.chips_needed == 300_000  # 50k * 6
    eff = boss_def("b_violet_vessel").effect
    assert eff.is_finisher
    assert eff.reward() == 8


def test_regular_boss_reward_is_5():
    assert boss_def("b_wall").effect.reward() == 5


# ---------------------------------------------------------------------------
# Suit debuffs (The Club / Goad / Window / Head)
# ---------------------------------------------------------------------------

def test_suit_debuff_targets_by_boss():
    run = make_run()
    for bid, suit_spec in [("b_club", "9C"), ("b_goad", "9S"),
                           ("b_window", "9D"), ("b_head", "9H")]:
        assert boss_def(bid).effect.card_debuffed(run, card(suit_spec))


def test_debuffed_pair_scores_hand_base_only():
    # Both clubs debuffed: no card chips, but the pair is still DETECTED.
    g = boss_game("club", "b_club")
    res = score_play(g.run, cards("9C", "9C"))
    assert res.hand_result.hand_type == HandType.PAIR
    assert res.chips == 10  # pair L1 base only
    assert res.mult == 2
    assert res.score == 20


def test_debuffed_enhancement_does_not_fire():
    g = boss_game("club", "b_club")
    res = score_play(g.run, cards("9C(mult)", "9C"))
    assert res.chips == 10 and res.mult == 2  # no +4 mult from enhancement


def test_debuffed_card_fires_no_joker_hooks():
    # Gluttonous Joker: +3 mult per scored Club. Debuffed clubs stay silent.
    g = boss_game("club", "b_club", joker_ids=["j_gluttonous"])
    res = score_play(g.run, cards("9C", "9C"))
    assert res.mult == 2  # not 2 + 3 + 3
    # Control: same joker, hearts-debuff boss -> clubs fire normally.
    g2 = boss_game("head", "b_head", joker_ids=["j_gluttonous"])
    res2 = score_play(g2.run, cards("9C", "9C"))
    assert res2.chips == 28 and res2.mult == 8


def test_non_target_suit_unaffected():
    g = boss_game("club", "b_club")
    res = score_play(g.run, cards("9H", "9H"))
    assert res.chips == 28 and res.mult == 2


def test_wild_card_debuff_uses_printed_suit():
    run = make_run()
    eff = boss_def("b_club").effect
    # Wild whose PRINTED suit matches -> debuffed (reverts to printed suit).
    assert eff.card_debuffed(run, card("9C(wild)"))
    # Wild of another printed suit -> not debuffed, even though a live Wild
    # would count as a Club for suit matching.
    assert not eff.card_debuffed(run, card("9H(wild)"))


def test_debuffed_wild_scores_nothing():
    g = boss_game("club", "b_club")
    res = score_play(g.run, cards("9C(wild)", "9C"))
    assert res.hand_result.hand_type == HandType.PAIR
    assert res.chips == 10 and res.mult == 2


def test_stone_card_never_suit_debuffed():
    run = make_run()
    for bid in ("b_club", "b_goad", "b_window", "b_head"):
        assert not boss_def(bid).effect.card_debuffed(run, card("9C(stone)"))
        assert not boss_def(bid).effect.card_debuffed(run, card("9H(stone)"))


# ---------------------------------------------------------------------------
# The Psychic — must play exactly 5 cards
# ---------------------------------------------------------------------------

def test_psychic_legal_actions_only_5_card_plays():
    g = boss_game("psy", "b_psychic")
    acts = g.legal_actions()
    plays = [a for a in acts if isinstance(a, PlayHand)]
    assert plays, "5-card plays must exist"
    assert all(len(a.indices) == 5 for a in plays)
    # Discards are unrestricted.
    assert any(isinstance(a, Discard) and len(a.indices) == 1 for a in acts)


def test_psychic_rejects_4_card_play():
    g = boss_game("psy", "b_psychic")
    with pytest.raises(IllegalActionError):
        g.apply(PlayHand((0, 1, 2, 3)))
    g.apply(PlayHand((0, 1, 2, 3, 4)))  # 5 cards is fine


# ---------------------------------------------------------------------------
# The Manacle — -1 hand size
# ---------------------------------------------------------------------------

def test_manacle_hand_refills_to_7():
    g = boss_game("man", "b_manacle")
    assert g.run.round.boss_state["hand_size_bonus"] == -1
    assert len(g.run.round.hand) == 7
    g.apply(Discard((0, 1)))
    assert len(g.run.round.hand) == 7


# ---------------------------------------------------------------------------
# The House — first hand drawn face down
# ---------------------------------------------------------------------------

def test_house_initial_deal_face_down_then_face_up():
    g = boss_game("house", "b_house", ante=2)
    rnd = g.run.round
    initial_uids = {c.uid for c in rnd.hand}
    assert set(rnd.face_down_uids) == initial_uids  # full hand face down
    g.apply(PlayHand((0, 1)))
    rnd = g.run.round
    assert rnd is not None
    hand_uids = {c.uid for c in rnd.hand}
    new_uids = hand_uids - initial_uids
    assert len(new_uids) == 2  # refilled after the play
    # Played cards flipped, newly drawn cards are face up.
    assert set(rnd.face_down_uids) == initial_uids & hand_uids
    assert not new_uids & set(rnd.face_down_uids)


# ---------------------------------------------------------------------------
# The Arm — de-level the played hand before scoring, permanently
# ---------------------------------------------------------------------------

def test_arm_delevels_before_base():
    g = boss_game("arm", "b_arm", ante=2, hand_levels={HandType.PAIR: 2})
    res = score_play(g.run, cards("9S", "9H"))
    # Scored at L1 (10c x2m + 18 card chips), not L2 (25c x3m).
    assert res.chips == 28 and res.mult == 2 and res.score == 56
    assert g.run.hand_levels[HandType.PAIR] == 1  # permanent for the run


def test_arm_level_floor_is_1():
    g = boss_game("arm", "b_arm", ante=2)
    assert g.run.hand_levels[HandType.PAIR] == 1
    res = score_play(g.run, cards("9S", "9H"))
    assert res.chips == 28 and res.mult == 2
    assert g.run.hand_levels[HandType.PAIR] == 1


# ---------------------------------------------------------------------------
# The Flint — base chips and mult halved (rounded up)
# ---------------------------------------------------------------------------

def test_flint_halves_base_rounded_up():
    g = boss_game("flint", "b_flint", ante=2)
    # High Card base 5c/1m -> ceil = 3c/1m; plus the 7's card chips.
    res = score_play(g.run, cards("7S"))
    assert res.chips == 10 and res.mult == 1 and res.score == 10


def test_flint_pair_exact():
    g = boss_game("flint", "b_flint", ante=2)
    # Pair base 10c/2m -> 5c/1m; +9+9 card chips = 23 x 1 = 23.
    res = score_play(g.run, cards("9S", "9H"))
    assert res.chips == 23 and res.mult == 1 and res.score == 23


def test_flint_only_halves_base_not_later_stages():
    g = boss_game("flint", "b_flint", ante=2)
    # +4 mult enhancement applies AFTER the halved base: (5+18)c, (1+4)m.
    res = score_play(g.run, cards("9S(mult)", "9H"))
    assert res.chips == 23 and res.mult == 5 and res.score == 115


# ---------------------------------------------------------------------------
# The Hook — discard 2 random held cards after each played hand
# ---------------------------------------------------------------------------

def test_hook_discards_two_after_play():
    g = boss_game("hook", "b_hook")
    rnd = g.run.round
    assert len(rnd.hand) == 8 and len(rnd.deck_remaining) == 44
    g.apply(PlayHand((0,)))
    rnd = g.run.round
    assert rnd is not None  # 1 low card can't clear a 600-chip boss
    # 1 played + 2 hooked away, refilled back to 8 -> deck depletes by 3.
    assert len(rnd.hand) == 8
    assert len(rnd.discarded) == 2
    assert len(rnd.deck_remaining) == 44 - 3


def test_hook_depletes_deck_faster():
    g = boss_game("hook2", "b_hook")
    for _ in range(3):
        if g.run.round is None or g.run.phase != Phase.PLAYING:
            break
        g.apply(PlayHand((0,)))
    rnd = g.run.round
    assert rnd is not None
    assert len(rnd.discarded) == 6
    assert len(rnd.deck_remaining) == 44 - 9


# ---------------------------------------------------------------------------
# The Ox — playing the most played hand sets money to $0
# ---------------------------------------------------------------------------

def _seed_counts(counts):
    def prep(run):
        for ht, n in counts.items():
            run.hand_play_counts[ht] = n
    return prep


def test_ox_zeroes_money_on_most_played_hand():
    g = boss_game(
        "ox", "b_ox", money=25,
        prep=_seed_counts({HandType.PAIR: 5, HandType.HIGH_CARD: 2}),
    )
    assert g.run.round.boss_state["ox_target"] == int(HandType.PAIR)
    assert g.run.money == 25
    score_play(g.run, cards("9S", "9H"))  # a pair
    assert g.run.money == 0


def test_ox_other_hands_keep_money():
    g = boss_game(
        "ox2", "b_ox", money=25,
        prep=_seed_counts({HandType.PAIR: 5, HandType.HIGH_CARD: 2}),
    )
    score_play(g.run, cards("9S"))  # high card, not the target
    assert g.run.money == 25


def test_ox_tie_favors_lowest_hand_type():
    g = boss_game(
        "ox3", "b_ox",
        prep=_seed_counts({HandType.HIGH_CARD: 3, HandType.PAIR: 3}),
    )
    assert g.run.round.boss_state["ox_target"] == int(HandType.HIGH_CARD)


# ---------------------------------------------------------------------------
# Boss selection
# ---------------------------------------------------------------------------

def test_ante8_picks_only_finisher():
    g = Game(seed="fin")
    g.run.ante = 8
    assert pick_boss(g.run) == "b_violet_vessel"


def test_ante1_pool_is_min_ante_1_non_finishers():
    g = Game(seed="pool")
    g.run.ante = 1
    picks = {pick_boss(g.run) for _ in range(len(ANTE1_BOSSES))}
    assert picks == ANTE1_BOSSES  # no repeats until the pool is exhausted


def test_no_repeat_until_pool_exhausted():
    g = Game(seed="rep")
    g.run.ante = 1
    seen = [pick_boss(g.run) for _ in range(len(ANTE1_BOSSES) + 1)]
    assert len(set(seen[:-1])) == len(ANTE1_BOSSES)
    assert seen[-1] in ANTE1_BOSSES  # pool recycles afterwards
