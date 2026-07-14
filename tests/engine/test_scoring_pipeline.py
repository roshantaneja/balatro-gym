"""Scoring pipeline: base values, modifiers, ordering, retriggers, copies.

Expected numbers are hand-derived from wiki rules; derivations in comments.
"""

from balatro_gym.engine import HandType, score_play

from tests.helpers import cards, joker, make_run


def test_pair_of_nines():
    # Pair L1 = 10 chips x 2 mult; cards 9+9 -> (10+18) x 2 = 56
    run = make_run()
    res = score_play(run, cards("9S", "9H"))
    assert res.hand_result.hand_type == HandType.PAIR
    assert res.score == 56


def test_high_card_only_top_card_scores():
    # High Card L1 = 5 x 1; only the King scores: (5+10) x 1 = 15
    run = make_run()
    res = score_play(run, cards("KS", "9H", "4D"))
    assert res.score == 15


def test_hand_level_scaling():
    # Pair L3 = 10+2*15 = 40 chips, 2+2*1 = 4 mult -> (40+18) x 4 = 232
    run = make_run(hand_levels={HandType.PAIR: 3})
    res = score_play(run, cards("9S", "9H"))
    assert res.score == 232


def test_flush_with_face_cards():
    # Flush L1 = 35 x 4; spades K+J+9+5+2 = 10+10+9+5+2 = 36 -> (35+36) x 4 = 284
    run = make_run()
    res = score_play(run, cards("KS", "JS", "9S", "5S", "2S"))
    assert res.score == 284


class TestEnhancements:
    def test_bonus(self):
        # (10 + 9 + 9 + 30) x 2 = 116
        run = make_run()
        res = score_play(run, cards("9S(bonus)", "9H"))
        assert res.score == 116

    def test_mult_enhancement(self):
        # chips 28, mult 2+4 = 6 -> 168
        run = make_run()
        res = score_play(run, cards("9S(mult)", "9H"))
        assert res.score == 168

    def test_glass(self):
        # chips 28, mult 2x2 = 4 -> 112
        run = make_run()
        res = score_play(run, cards("9S(glass)", "9H"))
        assert res.score == 112

    def test_steel_held(self):
        # pair 28 x 2, steel held x1.5 -> 28 x 3 = 84
        run = make_run()
        res = score_play(run, cards("9S", "9H"), held=cards("KD(steel)"))
        assert res.score == 84

    def test_stone_always_scores(self):
        # Pair detected on 9s; stone adds 50: (10+18+50) x 2 = 156
        run = make_run()
        res = score_play(run, cards("9S", "9H", "4D(stone)"))
        assert res.score == 156


class TestEditionsAndSeals:
    def test_foil_card(self):
        # (10 + 9+50 + 9) x 2 = 156
        run = make_run()
        res = score_play(run, cards("9S(foil)", "9H"))
        assert res.score == 156

    def test_holo_card(self):
        # chips 28 x (2+10) = 336
        run = make_run()
        res = score_play(run, cards("9S(holo)", "9H"))
        assert res.score == 336

    def test_poly_card(self):
        # 28 x (2x1.5) = 84
        run = make_run()
        res = score_play(run, cards("9S(poly)", "9H"))
        assert res.score == 84

    def test_gold_seal_money(self):
        run = make_run()
        start = run.money
        res = score_play(run, cards("9S(gold_seal)", "9H"))
        assert res.money_delta == 3
        assert run.money == start + 3

    def test_red_seal_retriggers_card(self):
        # 9S counted twice: (10 + 9 + 9 + 9) x 2 = 74
        run = make_run()
        res = score_play(run, cards("9S(red_seal)", "9H"))
        assert res.score == 74

    def test_red_seal_retriggers_enhancement_too(self):
        # bonus fires twice: (10 + (9+30)*2 + 9) x 2 = 194
        run = make_run()
        res = score_play(run, cards("9S(bonus,red_seal)", "9H"))
        assert res.score == 194


class TestJokerOrdering:
    """The wiki's canonical 640-vs-480 example: additive-then-multiplicative
    depends on joker order, left to right."""

    def test_additive_before_xmult(self):
        # Full House 40 x 4 base, no card chips? Use stone-free known cards:
        # 9,9,9,K,K = 9*3 + 10*2 = 47 chips -> chips 87.
        # [Joker(+4), Duo(x2 if pair)]: mult = (4+4)*2 = 16 -> 87*16 = 1392
        run = make_run(jokers=[joker("j_joker"), joker("j_duo")])
        res = score_play(run, cards("9S", "9H", "9D", "KC", "KS"))
        assert res.score == 1392

    def test_xmult_before_additive(self):
        # [Duo, Joker]: mult = 4*2+4 = 12 -> 87*12 = 1044
        run = make_run(jokers=[joker("j_duo"), joker("j_joker")])
        res = score_play(run, cards("9S", "9H", "9D", "KC", "KS"))
        assert res.score == 1044


class TestScoredJokers:
    def test_greedy_counts_diamonds(self):
        # Pair of 9s both diamonds? Use 9D 9D: (10+9+9) x (2 + 3 + 3) = 224
        run = make_run(jokers=[joker("j_greedy")])
        res = score_play(run, cards("9D", "9D"))
        assert res.score == 224

    def test_greedy_ignores_other_suits(self):
        run = make_run(jokers=[joker("j_greedy")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 56

    def test_wild_card_counts_for_suit_jokers(self):
        # Wild 9 counts as diamond: (28) x (2+3) = 140
        run = make_run(jokers=[joker("j_greedy")])
        res = score_play(run, cards("9S(wild)", "9H"))
        assert res.score == 140

    def test_scholar_aces(self):
        # Pair of aces: chips 10+11+20+11+20 = 72, mult 2+4+4 = 10 -> 720
        run = make_run(jokers=[joker("j_scholar")])
        res = score_play(run, cards("AS", "AH"))
        assert res.score == 720

    def test_photograph_first_face_only(self):
        # Pair of kings: chips 10+10+10 = 30, mult 2 x2 (first king only) = 4 -> 120
        run = make_run(jokers=[joker("j_photograph")])
        res = score_play(run, cards("KS", "KH"))
        assert res.score == 120


class TestHeldJokers:
    def test_baron_kings_in_hand(self):
        # Pair of 9s = 28 x 2; two held kings x1.5 each -> 2 * 1.5 * 1.5 = 4.5 -> 126
        run = make_run(jokers=[joker("j_baron")])
        res = score_play(run, cards("9S", "9H"), held=cards("KD", "KC", "4H"))
        assert res.score == 126

    def test_raised_fist_lowest_card(self):
        # held lowest = 4H -> +8 mult; pair 28 x (2+8) = 280
        run = make_run(jokers=[joker("j_raised_fist")])
        res = score_play(run, cards("9S", "9H"), held=cards("KD", "4H"))
        assert res.score == 280


class TestCopyJokers:
    def test_blueprint_copies_right(self):
        # [Blueprint, Joker]: Blueprint acts as another +4 Mult.
        # pair 28 x (2+4+4) = 280
        run = make_run(jokers=[joker("j_blueprint"), joker("j_joker")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 280

    def test_blueprint_with_nothing_to_copy(self):
        run = make_run(jokers=[joker("j_joker"), joker("j_blueprint")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 28 * 6  # only the real Joker fires

    def test_brainstorm_copies_leftmost(self):
        # [Joker, x, Brainstorm] -> Brainstorm = +4 Mult again
        run = make_run(jokers=[joker("j_joker"), joker("j_greedy"), joker("j_brainstorm")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 28 * (2 + 4 + 4)

    def test_blueprint_chain(self):
        # [Blueprint, Blueprint, Joker] -> both blueprints resolve to Joker: +4 x3
        run = make_run(
            jokers=[joker("j_blueprint"), joker("j_blueprint"), joker("j_joker")]
        )
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 28 * (2 + 12)

    def test_brainstorm_cycle_guard(self):
        # Brainstorm at slot 0 points at itself -> resolves to nothing, no crash.
        run = make_run(jokers=[joker("j_brainstorm"), joker("j_joker")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 28 * 6


class TestJokerEditions:
    def test_foil_joker(self):
        # Joker(+4) with foil: +50 chips before it -> (28+50) x 6 = 468
        run = make_run(jokers=[joker("j_joker", edition="foil")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 468

    def test_poly_joker_applies_after(self):
        # Joker(+4, poly): mult (2+4)*1.5 = 9 -> 28 x 9 = 252
        run = make_run(jokers=[joker("j_joker", edition="poly")])
        res = score_play(run, cards("9S", "9H"))
        assert res.score == 252


def test_audit_log_is_complete():
    run = make_run(jokers=[joker("j_joker")])
    res = score_play(run, cards("9S", "9H"))
    # base info + two card chips + joker mult
    ops = [e.op for e in res.events]
    assert ops.count("+c") == 2
    assert ops.count("+m") == 1
    assert "TOTAL" in res.explain()
