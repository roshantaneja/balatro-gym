"""Hand evaluator: all 13 types, contains-sets, precedence, rule modifiers."""

from balatro_gym.engine import EvalRules, HandType, evaluate_hand

from tests.helpers import cards


def ht(specs, rules=EvalRules()):
    return evaluate_hand(cards(*specs), rules)


class TestHandTypes:
    def test_high_card(self):
        r = ht(["KS", "9H", "4D"])
        assert r.hand_type == HandType.HIGH_CARD
        # Only the single highest card scores.
        assert [repr(c) for c in r.scoring_cards] == ["K♠"]

    def test_pair(self):
        r = ht(["9S", "9H", "KD"])
        assert r.hand_type == HandType.PAIR
        assert len(r.scoring_cards) == 2

    def test_two_pair(self):
        r = ht(["9S", "9H", "KD", "KC"])
        assert r.hand_type == HandType.TWO_PAIR
        assert len(r.scoring_cards) == 4

    def test_three_of_a_kind(self):
        r = ht(["9S", "9H", "9D", "KC"])
        assert r.hand_type == HandType.THREE_KIND
        assert len(r.scoring_cards) == 3

    def test_straight(self):
        r = ht(["5S", "6H", "7D", "8C", "9S"])
        assert r.hand_type == HandType.STRAIGHT
        assert len(r.scoring_cards) == 5

    def test_ace_low_straight(self):
        r = ht(["AS", "2H", "3D", "4C", "5S"])
        assert r.hand_type == HandType.STRAIGHT

    def test_ace_high_straight(self):
        r = ht(["10S", "JH", "QD", "KC", "AS"])
        assert r.hand_type == HandType.STRAIGHT

    def test_no_wraparound_straight(self):
        r = ht(["QS", "KH", "AD", "2C", "3S"])
        assert r.hand_type == HandType.HIGH_CARD

    def test_flush(self):
        r = ht(["2S", "5S", "9S", "JS", "KS"])
        assert r.hand_type == HandType.FLUSH
        assert len(r.scoring_cards) == 5

    def test_full_house(self):
        r = ht(["9S", "9H", "9D", "KC", "KS"])
        assert r.hand_type == HandType.FULL_HOUSE
        assert len(r.scoring_cards) == 5

    def test_four_of_a_kind(self):
        r = ht(["9S", "9H", "9D", "9C", "KS"])
        assert r.hand_type == HandType.FOUR_KIND
        assert len(r.scoring_cards) == 4

    def test_straight_flush(self):
        r = ht(["5S", "6S", "7S", "8S", "9S"])
        assert r.hand_type == HandType.STRAIGHT_FLUSH
        assert not r.is_royal

    def test_royal_flush(self):
        r = ht(["10S", "JS", "QS", "KS", "AS"])
        assert r.hand_type == HandType.STRAIGHT_FLUSH
        assert r.is_royal

    def test_five_of_a_kind(self):
        r = ht(["9S", "9H", "9D", "9C", "9S"])
        assert r.hand_type == HandType.FIVE_KIND

    def test_flush_house(self):
        r = ht(["9S", "9S", "9S", "KS", "KS"])
        assert r.hand_type == HandType.FLUSH_HOUSE

    def test_flush_five(self):
        r = ht(["9S", "9S", "9S", "9S", "9S"])
        assert r.hand_type == HandType.FLUSH_FIVE


class TestContainsSemantics:
    def test_four_kind_is_not_two_pair(self):
        r = ht(["9S", "9H", "9D", "9C"])
        assert HandType.TWO_PAIR not in r.contained  # distinct ranks required

    def test_five_kind_is_not_full_house(self):
        r = ht(["9S", "9H", "9D", "9C", "9S"])
        assert HandType.FULL_HOUSE not in r.contained

    def test_full_house_contains_pair_and_three(self):
        r = ht(["9S", "9H", "9D", "KC", "KS"])
        assert HandType.PAIR in r.contained
        assert HandType.THREE_KIND in r.contained
        assert HandType.TWO_PAIR in r.contained

    def test_precedence_max_wins(self):
        r = ht(["9S", "9S", "9S", "KS", "KS"])
        assert HandType.FULL_HOUSE in r.contained
        assert HandType.FLUSH in r.contained
        assert r.hand_type == HandType.FLUSH_HOUSE


class TestRuleModifiers:
    def test_four_fingers_flush(self):
        rules = EvalRules(flush_size=4, straight_size=4)
        r = ht(["2S", "5S", "9S", "JS", "KH"], rules)
        assert r.hand_type == HandType.FLUSH
        assert len(r.scoring_cards) == 4  # only the 4 spades score

    def test_four_fingers_straight(self):
        rules = EvalRules(flush_size=4, straight_size=4)
        r = ht(["5S", "6H", "7D", "8C", "KS"], rules)
        assert r.hand_type == HandType.STRAIGHT

    def test_shortcut_gap_straight(self):
        rules = EvalRules(straight_max_gap=2)
        r = ht(["2S", "4H", "6D", "8C", "10S"], rules)
        assert r.hand_type == HandType.STRAIGHT

    def test_splash_scores_everything(self):
        rules = EvalRules(splash=True)
        r = ht(["9S", "9H", "KD", "4C", "2S"], rules)
        assert r.hand_type == HandType.PAIR
        assert len(r.scoring_cards) == 5

    def test_wild_makes_flush(self):
        r = ht(["2S", "5S", "9S", "JS", "KH(wild)"])
        assert r.hand_type == HandType.FLUSH
        assert len(r.scoring_cards) == 5

    def test_smeared_suits(self):
        rules = EvalRules(smeared_suits=True)
        r = ht(["2S", "5C", "9S", "JC", "KS"], rules)  # spades + clubs = black class
        assert r.hand_type == HandType.FLUSH


class TestStoneCards:
    def test_stone_excluded_from_detection_but_scores(self):
        r = ht(["9S", "9H", "KD(stone)"])
        assert r.hand_type == HandType.PAIR
        assert len(r.scoring_cards) == 3  # pair + stone

    def test_all_stones_is_high_card(self):
        r = ht(["KD(stone)", "2S(stone)"])
        assert r.hand_type == HandType.HIGH_CARD
        assert len(r.scoring_cards) == 2
