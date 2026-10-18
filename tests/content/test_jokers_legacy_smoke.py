"""Smoke coverage for the pre-existing joker set, one exact score per joker.

These jokers have deeper tests in tests/engine/test_scoring_pipeline.py; this
table keeps a 1:1 id->test mapping under tests/content/ (enforced by
test_registry_meta.py). Derivations in the notes column.
"""

from __future__ import annotations

import pytest

from balatro_gym.engine import score_play
from tests.helpers import cards, joker, make_run

# (joker ids, played, held, expected score, derivation)
SMOKE = [
    (["j_joker"], ["9S", "9H"], [], 168, "28 x (2+4)"),
    (["j_jolly"], ["9S", "9H"], [], 280, "28 x (2+8)"),
    (["j_sly"], ["9S", "9H"], [], 156, "(28+50) x 2"),
    (["j_duo"], ["9S", "9H"], [], 112, "28 x (2x2)"),
    (["j_half"], ["9S", "9H"], [], 616, "<=3 cards: 28 x (2+20)"),
    (["j_banner"], ["9S", "9H"], [], 56, "no round -> 0 discards left -> no chips"),
    (["j_supernova"], ["9S", "9H"], [], 84, "first Pair this run: 28 x (2+1)"),
    (["j_abstract"], ["9S", "9H"], [], 140, "1 joker: 28 x (2+3)"),
    (["j_greedy"], ["9D", "9D"], [], 224, "28 x (2+3+3)"),
    (["j_lusty"], ["9H", "9H"], [], 224, "28 x (2+3+3)"),
    (["j_wrathful"], ["9S", "9S"], [], 224, "28 x (2+3+3)"),
    (["j_gluttonous"], ["9C", "9C"], [], 224, "28 x (2+3+3)"),
    (["j_odd_todd"], ["9S", "9H"], [], 180, "(28+31+31) x 2"),
    (["j_scholar"], ["AS", "AH"], [], 720, "(10+11+20+11+20) x (2+4+4)"),
    (["j_photograph"], ["KS", "KH"], [], 120, "(10+10+10) x (2x2), first face only"),
    (["j_baron"], ["9S", "9H"], ["KD"], 84, "28 x (2 x 1.5)"),
    (["j_shoot_the_moon"], ["9S", "9H"], ["QD"], 420, "28 x (2+13)"),
    (["j_raised_fist"], ["9S", "9H"], ["4H"], 280, "28 x (2 + 2*4)"),
    (["j_blueprint", "j_joker"], ["9S", "9H"], [], 280, "28 x (2+4+4)"),
    (["j_joker", "j_brainstorm"], ["9S", "9H"], [], 280, "28 x (2+4+4)"),
]


@pytest.mark.parametrize("jids,played,held,expected,note", SMOKE, ids=lambda v: str(v))
def test_legacy_joker_smoke(jids, played, held, expected, note):
    run = make_run(jokers=[joker(j) for j in jids])
    res = score_play(run, cards(*played), held=cards(*held))
    assert res.score == expected, note


def test_bloodstone_chance_outcomes():
    # j_bloodstone: each scored heart has a 1 in 2 chance of x1.5.
    # Two hearts -> 0/1/2 procs: 56, 84, or 126.
    seen = set()
    for i in range(60):
        run = make_run(jokers=[joker("j_bloodstone")], seed=f"blood-{i}")
        res = score_play(run, cards("9H", "9H"))
        assert res.score in (56, 84, 126)
        seen.add(res.score)
    assert len(seen) > 1
