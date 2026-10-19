"""Per-joker tests for the v1 content set (played/retrigger/economy/rules/
discard/misc modules). Expected numbers are hand-derived; derivations in
comments."""

from __future__ import annotations

import pytest

from balatro_gym.engine import (
    CONSUMABLES,
    BlindKind,
    ConsumableInst,
    ConsumableKind,
    HandType,
    Phase,
    Suit,
    joker_def,
    resolve_joker,
    score_play,
)
from balatro_gym.engine.round import discard as engine_discard
from balatro_gym.engine.state import RoundState
from tests.helpers import card, cards, joker, make_run


def with_round(run, *, hands_left=4, discards_left=3, hand=None, hands_played=0):
    run.round = RoundState(
        blind_kind=BlindKind.SMALL,
        boss_id=None,
        chips_needed=300.0,
        hands_left=hands_left,
        discards_left=discards_left,
        hand=hand if hand is not None else [],
        hands_played=hands_played,
    )
    run.phase = Phase.PLAYING
    return run


def run_round_end(run) -> int:
    """Mirror of round.py's _win_blind joker loop (money + round-end updates)."""
    other = 0
    for j in run.jokers:
        eff = joker_def(j.id).effect
        other += eff.round_end_money(run, j)
        eff.update_on_round_end(run, j)
    return other


# ---------------------------------------------------------------------------
# Green Joker
# ---------------------------------------------------------------------------

class TestGreenJoker:
    def test_first_hand_reads_one(self):
        # Gain lands at stage 2, read at stage 5: pair 28 x (2+1) = 84.
        run = make_run(jokers=[joker("j_green_joker")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 84
        assert run.jokers[0].state["mult"] == 1

    def test_discard_decrements_once_per_action(self):
        run = make_run(jokers=[joker("j_green_joker", state={"mult": 3})])
        with_round(run, hand=cards("2S", "3H", "4D"))
        engine_discard(run, (0, 1))  # one action, two cards -> -1
        assert run.jokers[0].state["mult"] == 2

    def test_never_below_zero(self):
        run = make_run(jokers=[joker("j_green_joker")])
        with_round(run, hand=cards("2S", "3H"))
        engine_discard(run, (0,))
        assert run.jokers[0].state["mult"] == 0

    def test_blueprint_pays_twice_scales_once(self):
        # State 2 -> updated ONCE to 3; read twice: 28 x (2+3+3) = 224.
        run = make_run(
            jokers=[joker("j_blueprint"), joker("j_green_joker", state={"mult": 2})]
        )
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 224
        assert run.jokers[1].state["mult"] == 3


# ---------------------------------------------------------------------------
# Ride the Bus
# ---------------------------------------------------------------------------

class TestRideTheBus:
    def test_gains_without_scoring_face(self):
        run = make_run(jokers=[joker("j_ride_the_bus")])
        res = score_play(run, cards("9S", "9H"))  # 28 x (2+1) = 84
        assert res.score == 84
        res = score_play(run, cards("9D", "9C"))  # 28 x (2+2) = 112
        assert res.score == 112
        assert run.jokers[0].state["mult"] == 2

    def test_scoring_face_resets(self):
        run = make_run(jokers=[joker("j_ride_the_bus", state={"mult": 5})])
        # Pair of kings: reset happens at stage 2, read 0 at stage 5.
        # chips 10+10+10 = 30, mult 2 -> 60.
        res = score_play(run, cards("KS", "KH"))
        assert res.score == 60
        assert run.jokers[0].state["mult"] == 0

    def test_non_scoring_face_does_not_reset(self):
        # A+K high card: only the Ace scores, the King does not.
        run = make_run(jokers=[joker("j_ride_the_bus", state={"mult": 5})])
        score_play(run, cards("AS", "KH"))
        assert run.jokers[0].state["mult"] == 6

    def test_pareidolia_makes_everything_reset(self):
        run = make_run(
            jokers=[joker("j_pareidolia"), joker("j_ride_the_bus", state={"mult": 3})]
        )
        score_play(run, cards("9S", "9H"))
        assert run.jokers[1].state["mult"] == 0


# ---------------------------------------------------------------------------
# DNA
# ---------------------------------------------------------------------------

class TestDNA:
    def test_copies_single_card_on_first_hand(self):
        run = make_run(jokers=[joker("j_dna")])
        with_round(run)
        deck_before = len(run.full_deck)
        original = card("9S(bonus)")
        score_play(run, [original])
        assert len(run.full_deck) == deck_before + 1
        copy = run.full_deck[-1]
        assert (copy.rank, copy.suit, copy.enhancement) == (
            original.rank, original.suit, original.enhancement
        )
        assert copy.uid != original.uid
        assert run.round.hand[-1] is copy  # drawn to hand

    def test_no_copy_with_two_cards(self):
        run = make_run(jokers=[joker("j_dna")])
        with_round(run)
        deck_before = len(run.full_deck)
        score_play(run, cards("9S", "9H"))
        assert len(run.full_deck) == deck_before

    def test_no_copy_on_second_hand(self):
        run = make_run(jokers=[joker("j_dna")])
        with_round(run, hands_played=1)
        deck_before = len(run.full_deck)
        score_play(run, cards("9S"))
        assert len(run.full_deck) == deck_before

    def test_blueprint_does_not_double_copy(self):
        run = make_run(jokers=[joker("j_blueprint"), joker("j_dna")])
        with_round(run)
        deck_before = len(run.full_deck)
        score_play(run, cards("9S"))
        assert len(run.full_deck) == deck_before + 1


# ---------------------------------------------------------------------------
# Space Joker
# ---------------------------------------------------------------------------

class TestSpaceJoker:
    def test_upgrade_applies_to_current_hand(self):
        # When it fires: Pair goes L1 -> L2 and the current hand scores at L2:
        # (10+15 + 18) x (2+1) = 43 x 3 = 129. Otherwise plain 56.
        fired = missed = False
        for i in range(100):
            run = make_run(jokers=[joker("j_space")], seed=f"space-{i}")
            res = score_play(run, cards("9S", "9H"))
            if run.hand_levels[HandType.PAIR] == 2:
                assert res.score == 129
                fired = True
            else:
                assert run.hand_levels[HandType.PAIR] == 1
                assert res.score == 56
                missed = True
            if fired and missed:
                break
        assert fired and missed


# ---------------------------------------------------------------------------
# Dusk / Hack / Mime (retriggers)
# ---------------------------------------------------------------------------

class TestDusk:
    def test_retriggers_on_final_hand(self):
        # hands_left == 1 during scoring of the round's final hand.
        # Pair: 10 + (9+9)*2 = 46 chips x 2 = 92.
        run = make_run(jokers=[joker("j_dusk")])
        with_round(run, hands_left=1)
        assert score_play(run, cards("9S", "9H")).score == 92

    def test_inactive_before_final_hand(self):
        run = make_run(jokers=[joker("j_dusk")])
        with_round(run, hands_left=2)
        assert score_play(run, cards("9S", "9H")).score == 56

    def test_no_round_is_safe(self):
        run = make_run(jokers=[joker("j_dusk")])
        assert score_play(run, cards("9S", "9H")).score == 56


class TestHack:
    def test_retriggers_low_ranks(self):
        # Pair of 5s: 10 + (5+5)*2 = 30 chips x 2 = 60.
        run = make_run(jokers=[joker("j_hack")])
        assert score_play(run, cards("5S", "5H")).score == 60

    def test_ignores_high_ranks(self):
        run = make_run(jokers=[joker("j_hack")])
        assert score_play(run, cards("9S", "9H")).score == 56

    def test_blueprint_adds_another_retrigger(self):
        # 3 passes per 5: 10 + (5+5)*3 = 40 chips x 2 = 80.
        run = make_run(jokers=[joker("j_blueprint"), joker("j_hack")])
        assert score_play(run, cards("5S", "5H")).score == 80


class TestMime:
    def test_retriggers_held_steel(self):
        # Steel fires twice: 28 x (2 x 1.5 x 1.5) = 126.
        run = make_run(jokers=[joker("j_mime")])
        res = score_play(run, cards("9S", "9H"), held=cards("KD(steel)"))
        assert res.score == 126

    def test_blueprint_mime_triple_steel(self):
        # 3 passes: 28 x (2 x 1.5^3) = 189.
        run = make_run(jokers=[joker("j_blueprint"), joker("j_mime")])
        res = score_play(run, cards("9S", "9H"), held=cards("KD(steel)"))
        assert res.score == 189


# ---------------------------------------------------------------------------
# Economy: Golden Joker, Egg, Business Card
# ---------------------------------------------------------------------------

class TestGoldenJoker:
    def test_four_dollars_at_round_end(self):
        run = make_run(jokers=[joker("j_golden")])
        assert run_round_end(run) == 4

    def test_not_copyable(self):
        run = make_run(jokers=[joker("j_blueprint"), joker("j_golden")])
        assert resolve_joker(run.jokers, 0) is None


class TestEgg:
    def test_gains_sell_value(self):
        run = make_run(jokers=[joker("j_egg")])
        assert run.jokers[0].sell_value == 1
        run_round_end(run)
        run_round_end(run)
        assert run.jokers[0].sell_value == 7

    def test_not_copyable(self):
        run = make_run(jokers=[joker("j_blueprint"), joker("j_egg")])
        assert resolve_joker(run.jokers, 0) is None


class TestBusinessCard:
    def test_face_cards_sometimes_pay(self):
        # Two scored kings -> two 1-in-2 rolls: $0/$2/$4.
        seen = set()
        for i in range(60):
            run = make_run(jokers=[joker("j_business")], seed=f"biz-{i}")
            res = score_play(run, cards("KS", "KH"))
            assert res.money_delta in (0, 2, 4)
            seen.add(res.money_delta)
        assert 0 in seen and 4 in seen

    def test_non_face_never_pays(self):
        for i in range(20):
            run = make_run(jokers=[joker("j_business")], seed=f"biz-n{i}")
            assert score_play(run, cards("9S", "9H")).money_delta == 0

    def test_pareidolia_enables_any_card(self):
        paid = False
        for i in range(60):
            run = make_run(
                jokers=[joker("j_pareidolia"), joker("j_business")], seed=f"biz-p{i}"
            )
            if score_play(run, cards("9S", "9H")).money_delta > 0:
                paid = True
                break
        assert paid


# ---------------------------------------------------------------------------
# Rule modifiers: Four Fingers, Splash, Pareidolia
# ---------------------------------------------------------------------------

class TestFourFingers:
    def test_four_card_flush(self):
        # Flush L1 35 x 4; K+J+9+5 = 34 chips -> 69 x 4 = 276.
        run = make_run(jokers=[joker("j_four_fingers")])
        res = score_play(run, cards("KS", "JS", "9S", "5S"))
        assert res.hand_result.hand_type == HandType.FLUSH
        assert res.score == 276

    def test_four_card_straight(self):
        # Straight L1 30 x 4; 2+3+4+5 = 14 chips -> 44 x 4 = 176.
        run = make_run(jokers=[joker("j_four_fingers")])
        res = score_play(run, cards("2S", "3H", "4D", "5S"))
        assert res.hand_result.hand_type == HandType.STRAIGHT
        assert res.score == 176

    def test_without_it_no_four_card_flush(self):
        run = make_run()
        res = score_play(run, cards("KS", "JS", "9S", "5S"))
        assert res.hand_result.hand_type == HandType.HIGH_CARD

    def test_not_copyable(self):
        run = make_run(jokers=[joker("j_blueprint"), joker("j_four_fingers")])
        assert resolve_joker(run.jokers, 0) is None


class TestSplash:
    def test_all_played_cards_score(self):
        # High card K with splash: 5 + 10+9+4 = 28 chips x 1 = 28.
        run = make_run(jokers=[joker("j_splash")])
        res = score_play(run, cards("KS", "9H", "4D"))
        assert res.score == 28
        assert len(res.hand_result.scoring_cards) == 3


class TestPareidolia:
    def test_photograph_triggers_on_non_face(self):
        # First "face" is the 9S: 28 x (2 x2) = 112.
        run = make_run(jokers=[joker("j_pareidolia"), joker("j_photograph")])
        assert score_play(run, cards("9S", "9H")).score == 112


# ---------------------------------------------------------------------------
# Discard jokers: Mail-In Rebate, Castle
# ---------------------------------------------------------------------------

class TestMailInRebate:
    def test_pays_per_matching_discarded_card(self):
        run = make_run(jokers=[joker("j_mail_in", state={"rank": 9})])
        with_round(run, hand=cards("9S", "9C", "4D"))
        before = run.money
        engine_discard(run, (0, 1, 2))
        assert run.money == before + 10  # two 9s x $5

    def test_no_pay_on_other_ranks(self):
        run = make_run(jokers=[joker("j_mail_in", state={"rank": 9})])
        with_round(run, hand=cards("KS", "4D"))
        before = run.money
        engine_discard(run, (0, 1))
        assert run.money == before

    def test_rank_rerolls_at_round_end(self):
        run = make_run(jokers=[joker("j_mail_in")])
        run_round_end(run)
        assert 2 <= run.jokers[0].state["rank"] <= 14

    def test_blueprint_does_not_double_pay(self):
        run = make_run(
            jokers=[joker("j_blueprint"), joker("j_mail_in", state={"rank": 9})]
        )
        with_round(run, hand=cards("9S", "4D"))
        before = run.money
        engine_discard(run, (0,))
        assert run.money == before + 5


class TestCastle:
    def test_accumulates_on_matching_discards(self):
        run = make_run(
            jokers=[joker("j_castle", state={"suit": int(Suit.SPADES)})]
        )
        with_round(run, hand=cards("9S", "KS", "4H"))
        engine_discard(run, (0, 1, 2))
        assert run.jokers[0].state["chips"] == 6  # two spades x 3

    def test_wild_card_counts(self):
        run = make_run(
            jokers=[joker("j_castle", state={"suit": int(Suit.SPADES)})]
        )
        with_round(run, hand=cards("9H(wild)",))
        engine_discard(run, (0,))
        assert run.jokers[0].state["chips"] == 3

    def test_reads_accumulated_chips(self):
        # 28+30 = 58 chips x 2 = 116.
        run = make_run(jokers=[joker("j_castle", state={"chips": 30, "suit": 0})])
        assert score_play(run, cards("9S", "9H")).score == 116

    def test_suit_rerolls_at_round_end(self):
        run = make_run(jokers=[joker("j_castle")])
        run_round_end(run)
        assert run.jokers[0].state["suit"] in (0, 1, 2, 3)

    def test_blueprint_scales_once_pays_twice(self):
        # Chips read twice (28 + 12 + 12 = 52 -> x2 = 104), scaled once.
        run = make_run(
            jokers=[
                joker("j_blueprint"),
                joker("j_castle", state={"chips": 12, "suit": int(Suit.SPADES)}),
            ]
        )
        with_round(run, hand=cards("2S", "9S", "9H"))
        engine_discard(run, (0,))
        assert run.jokers[1].state["chips"] == 15  # +3 exactly once
        run.round.hand = cards("9S", "9H")
        res = score_play(run, run.round.hand)
        assert res.chips == 28 + 15 + 15


# ---------------------------------------------------------------------------
# Ancient Joker
# ---------------------------------------------------------------------------

class TestAncientJoker:
    def test_xmult_on_matching_suit(self):
        # One heart scores: 28 x (2 x 1.5) = 84.
        run = make_run(jokers=[joker("j_ancient", state={"suit": int(Suit.HEARTS)})])
        assert score_play(run, cards("9H", "9S")).score == 84

    def test_stacks_per_matching_card(self):
        # Two hearts: 28 x (2 x 1.5 x 1.5) = 126.
        run = make_run(jokers=[joker("j_ancient", state={"suit": int(Suit.HEARTS)})])
        assert score_play(run, cards("9H", "9H")).score == 126

    def test_no_match_no_effect(self):
        run = make_run(jokers=[joker("j_ancient", state={"suit": int(Suit.HEARTS)})])
        assert score_play(run, cards("9S", "9C")).score == 56

    def test_wild_matches(self):
        run = make_run(jokers=[joker("j_ancient", state={"suit": int(Suit.HEARTS)})])
        assert score_play(run, cards("9S(wild)", "9C")).score == 84

    def test_suit_changes_and_never_repeats(self):
        run = make_run(jokers=[joker("j_ancient", state={"suit": 2})])
        for _ in range(10):
            prev = run.jokers[0].state["suit"]
            run_round_end(run)
            cur = run.jokers[0].state["suit"]
            assert cur in (0, 1, 2, 3) and cur != prev


# ---------------------------------------------------------------------------
# Ice Cream / Popcorn / Gros Michel (decay + self-destruct)
# ---------------------------------------------------------------------------

class TestIceCream:
    def test_first_hand_full_100(self):
        # Wiki: "the first hand will provide the full 100 chips, not 95".
        # (28+100) x 2 = 256; state then shows 95.
        run = make_run(jokers=[joker("j_ice_cream")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 256
        assert run.jokers[0].state["chips"] == 95

    def test_last_lick_then_melts(self):
        run = make_run(jokers=[joker("j_ice_cream", state={"chips": 5})])
        res = score_play(run, cards("9S", "9H"))  # gives 5: (28+5) x 2 = 66
        assert res.score == 66
        assert run.jokers == []  # melts right after the hand that empties it

    def test_survives_round_end_with_chips_left(self):
        run = make_run(jokers=[joker("j_ice_cream", state={"chips": 95})])
        run_round_end(run)
        assert len(run.jokers) == 1

    def test_melted_contributes_nothing(self):
        run = make_run(jokers=[joker("j_ice_cream", state={"chips": 0})])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 56
        assert len(run.jokers) == 0

    def test_mid_hand_decay_does_not_break_scoring(self):
        # Ice Cream reaching 0 during stage 2 must not disturb the joker to
        # its right: (28+5) x (2+4) = 198.
        run = make_run(jokers=[joker("j_ice_cream", state={"chips": 5}), joker("j_joker")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 198
        assert [j.id for j in run.jokers] == ["j_joker"]  # removal deferred past scoring

    def test_blueprint_pays_twice_decays_once(self):
        # (28 + 100 + 100) x 2 = 456; single decrement.
        run = make_run(jokers=[joker("j_blueprint"), joker("j_ice_cream")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 456
        assert run.jokers[1].state["chips"] == 95


class TestPopcorn:
    def test_fresh_popcorn(self):
        # 28 x (2+20) = 616.
        run = make_run(jokers=[joker("j_popcorn")])
        assert score_play(run, cards("9S", "9H")).score == 616

    def test_decays_four_per_round_and_expires(self):
        run = make_run(jokers=[joker("j_popcorn")])
        for expected in (16, 12, 8, 4):
            run_round_end(run)
            assert run.jokers[0].state["mult"] == expected
        run_round_end(run)  # 4 -> 0: eaten
        assert run.jokers == []

    def test_expiry_does_not_skip_next_joker(self):
        # Mirrors round.py's `for j in run.jokers` loop: Popcorn's removal
        # rebinds run.jokers, so Egg (to its right) still gets its update.
        run = make_run(jokers=[joker("j_popcorn", state={"mult": 4}), joker("j_egg")])
        run_round_end(run)
        assert [j.id for j in run.jokers] == ["j_egg"]
        assert run.jokers[0].sell_value == 4


class TestGrosMichel:
    def test_flat_mult(self):
        # 28 x (2+15) = 476.
        run = make_run(jokers=[joker("j_gros_michel")])
        assert score_play(run, cards("9S", "9H")).score == 476

    def test_sometimes_goes_extinct_at_round_end(self):
        survived = extinct = False
        for i in range(100):
            run = make_run(jokers=[joker("j_gros_michel")], seed=f"gros-{i}")
            run_round_end(run)
            if run.jokers:
                survived = True
            else:
                extinct = True
            if survived and extinct:
                break
        assert survived and extinct


# ---------------------------------------------------------------------------
# Fortune Teller / Baseball Card / 8 Ball
# ---------------------------------------------------------------------------

class TestFortuneTeller:
    def test_mult_per_tarot_used(self):
        run = make_run(jokers=[joker("j_fortune_teller")])
        run.stats["tarots_used"] = 3
        assert score_play(run, cards("9S", "9H")).score == 140  # 28 x (2+3)

    def test_zero_without_tarots(self):
        run = make_run(jokers=[joker("j_fortune_teller")])
        assert score_play(run, cards("9S", "9H")).score == 56


class TestBaseballCard:
    def test_xmult_per_uncommon(self):
        # Bloodstone is Uncommon (spades/clubs avoid its own trigger):
        # 28 x (2 x 1.5) = 84.
        run = make_run(jokers=[joker("j_baseball"), joker("j_bloodstone")])
        assert score_play(run, cards("9S", "9C")).score == 84

    def test_two_uncommons(self):
        # Bloodstone + Mime: 28 x (2 x 1.5 x 1.5) = 126.
        run = make_run(
            jokers=[joker("j_baseball"), joker("j_bloodstone"), joker("j_mime")]
        )
        assert score_play(run, cards("9S", "9C")).score == 126

    def test_common_and_rare_do_not_count(self):
        # j_joker is Common: only its own +4. 28 x 6 = 168.
        run = make_run(jokers=[joker("j_baseball"), joker("j_joker")])
        assert score_play(run, cards("9S", "9C")).score == 168

    def test_blueprint_copies_baseball(self):
        # Copy + original each give x1.5 for Bloodstone: 28 x (2 x 2.25) = 126.
        run = make_run(
            jokers=[joker("j_blueprint"), joker("j_baseball"), joker("j_bloodstone")]
        )
        assert score_play(run, cards("9S", "9C")).score == 126


class TestEightBall:
    def _tarots_registered(self) -> bool:
        return any(d.kind == ConsumableKind.TAROT for d in CONSUMABLES.values())

    def test_scored_eights_sometimes_create_tarot(self):
        if not self._tarots_registered():
            pytest.skip("no tarot consumables registered")
        created = False
        for i in range(60):
            run = make_run(jokers=[joker("j_8ball")], seed=f"8b-{i}")
            score_play(run, cards("8S", "8H"))
            if run.consumables:
                created = True
                assert all(c.kind == ConsumableKind.TAROT for c in run.consumables)
                assert len(run.consumables) <= run.consumable_slots
                break
        assert created

    def test_non_eights_never_create(self):
        for i in range(20):
            run = make_run(jokers=[joker("j_8ball")], seed=f"8b-n{i}")
            score_play(run, cards("9S", "9H"))
            assert run.consumables == []

    def test_no_creation_when_slots_full(self):
        if not self._tarots_registered():
            pytest.skip("no tarot consumables registered")
        for i in range(30):
            run = make_run(jokers=[joker("j_8ball")], seed=f"8b-f{i}")
            filler_id = sorted(
                cid for cid, d in CONSUMABLES.items() if d.kind == ConsumableKind.TAROT
            )[0]
            run.consumables = [
                ConsumableInst(id=filler_id, kind=ConsumableKind.TAROT, uid=run.new_uid())
                for _ in range(run.consumable_slots)
            ]
            score_play(run, cards("8S", "8H"))
            assert len(run.consumables) == run.consumable_slots
