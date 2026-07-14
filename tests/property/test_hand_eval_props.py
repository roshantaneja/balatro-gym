"""Property tests: evaluator invariants and scoring determinism."""

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

from balatro_gym.engine import (
    Card,
    Edition,
    Enhancement,
    EvalRules,
    HandType,
    Rank,
    Seal,
    Suit,
    evaluate_hand,
    score_play,
)

from tests.helpers import joker, make_run

_uids = itertools.count(50_000)


@st.composite
def random_card(draw):
    return Card(
        uid=next(_uids),
        rank=draw(st.sampled_from(list(Rank))),
        suit=draw(st.sampled_from(list(Suit))),
        enhancement=draw(st.sampled_from(list(Enhancement))),
        edition=draw(st.sampled_from([Edition.NONE, Edition.FOIL, Edition.HOLO, Edition.POLY])),
        seal=draw(st.sampled_from(list(Seal))),
    )


played_strategy = st.lists(random_card(), min_size=1, max_size=5)
rules_strategy = st.builds(
    EvalRules,
    flush_size=st.sampled_from([4, 5]),
    straight_size=st.sampled_from([4, 5]),
    straight_max_gap=st.sampled_from([1, 2]),
    smeared_suits=st.booleans(),
    splash=st.booleans(),
)


@given(played=played_strategy, rules=rules_strategy)
@settings(max_examples=300)
def test_hand_type_is_max_of_contained(played, rules):
    r = evaluate_hand(played, rules)
    if r.contained:
        assert r.hand_type == max(r.contained)
    else:
        assert r.hand_type == HandType.HIGH_CARD  # all-stone plays


@given(played=played_strategy, rules=rules_strategy)
@settings(max_examples=300)
def test_scoring_cards_are_played_cards_and_stones_always_score(played, rules):
    r = evaluate_hand(played, rules)
    played_uids = {c.uid for c in played}
    assert all(c.uid in played_uids for c in r.scoring_cards)
    scoring_uids = {c.uid for c in r.scoring_cards}
    for c in played:
        if c.is_stone:
            assert c.uid in scoring_uids
    # scoring cards preserve play order
    order = {c.uid: i for i, c in enumerate(played)}
    idxs = [order[c.uid] for c in r.scoring_cards]
    assert idxs == sorted(idxs)


@given(played=played_strategy, rules=rules_strategy)
@settings(max_examples=200)
def test_conjunction_hands(played, rules):
    r = evaluate_hand(played, rules)
    c = r.contained
    assert (HandType.STRAIGHT_FLUSH in c) == (HandType.STRAIGHT in c and HandType.FLUSH in c)
    assert (HandType.FLUSH_HOUSE in c) == (HandType.FULL_HOUSE in c and HandType.FLUSH in c)
    assert (HandType.FLUSH_FIVE in c) == (HandType.FIVE_KIND in c and HandType.FLUSH in c)


@given(played=played_strategy)
@settings(max_examples=100)
def test_evaluation_is_deterministic(played):
    r1 = evaluate_hand(played)
    r2 = evaluate_hand(played)
    assert r1.hand_type == r2.hand_type
    assert [c.uid for c in r1.scoring_cards] == [c.uid for c in r2.scoring_cards]


@given(played=played_strategy, seed=st.text(min_size=1, max_size=8))
@settings(max_examples=100)
def test_scoring_determinism_same_seed(played, seed):
    """Same seed + same play => identical audit log (incl. chance rolls)."""

    def run_once():
        run = make_run(jokers=[joker("j_bloodstone"), joker("j_joker")], seed=seed)
        res = score_play(run, [c.clone() for c in played])
        return res.to_dict()

    assert run_once() == run_once()


@given(played=played_strategy)
@settings(max_examples=50)
def test_clone_independence(played):
    run = make_run(jokers=[joker("j_joker")])
    snapshot = run.clone()
    before = snapshot.to_dict()
    score_play(run, [c.clone() for c in played])  # mutates run
    assert snapshot.to_dict() == before
