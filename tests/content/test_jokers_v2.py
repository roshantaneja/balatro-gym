"""Exact-score smoke coverage for the jokers added in v2 (one row per joker
whose effect shows up in a single scored hand). Derivations in the notes.

Baseline reference: Pair 9S 9H = (10 + 9 + 9) chips x 2 mult = 56.
Behavioural jokers (hand size, economy, lifecycle) are covered in
test_jokers_v2_behaviour.py.
"""

from __future__ import annotations

import pytest

from balatro_gym.engine import score_play
from tests.helpers import cards, joker, make_run

# (joker specs, played, held, expected score, derivation)
SMOKE = [
    # --- contains-hand families ---
    (["j_zany"], ["9S", "9H", "9D"], [], 855, "3oak (30+27) x (3+12)"),
    (["j_mad"], ["9S", "9H", "5D", "5C"], [], 576, "two pair (20+28) x (2+10)"),
    (["j_crazy"], ["5S", "6H", "7D", "8C", "9S"], [], 1040, "straight (30+35) x (4+12)"),
    (["j_droll"], ["2S", "4S", "6S", "8S", "10S"], [], 910, "flush (35+30) x (4+10)"),
    (["j_wily"], ["9S", "9H", "9D"], [], 471, "(57+100) x 3"),
    (["j_clever"], ["9S", "9H", "5D", "5C"], [], 256, "(48+80) x 2"),
    (["j_devious"], ["5S", "6H", "7D", "8C", "9S"], [], 660, "(65+100) x 4"),
    (["j_crafty"], ["2S", "4S", "6S", "8S", "10S"], [], 580, "(65+80) x 4"),
    (["j_trio"], ["9S", "9H", "9D"], [], 513, "57 x (3x3)"),
    (["j_family"], ["9S", "9H", "9D", "9C"], [], 2688, "4oak (60+36) x (7x4)"),
    (["j_order"], ["5S", "6H", "7D", "8C", "9S"], [], 780, "65 x (4x3)"),
    (["j_tribe"], ["2S", "4S", "6S", "8S", "10S"], [], 520, "65 x (4x2)"),
    # --- shape scalers (gain lands before the stage-5 read) ---
    (["j_trousers"], ["9S", "9H", "5D", "5C"], [], 192, "two pair: +2 -> 48 x 4"),
    (["j_runner"], ["5S", "6H", "7D", "8C", "9S"], [], 320, "straight: +15 chips -> 80 x 4"),
    (["j_square"], ["9S", "9H", "5D", "5C"], [], 104, "4 cards: +4 chips -> 52 x 2"),
    (["j_card_sharp"], ["9S", "9H"], [], 56, "no round -> nothing played before"),
    (["j_obelisk"], ["9S", "9H"], [], 56, "x1 before any most-played hand exists"),
    # --- on-scored ---
    (["j_fibonacci"], ["AS", "AH"], [], 576, "(10+22) x (2+8+8)"),
    (["j_even_steven"], ["8S", "8H"], [], 260, "26 x (2+4+4)"),
    (["j_scary_face"], ["KS", "KH"], [], 180, "(30+60) x 2"),
    (["j_smiley"], ["KS", "KH"], [], 360, "30 x (2+5+5)"),
    (["j_walkie_talkie"], ["4S", "4H"], [], 380, "(18+20) x (2+8)"),
    (["j_wee"], ["2S", "2H"], [], 60, "(14+16) x 2 after two +8 gains"),
    (["j_arrowhead"], ["9S", "9S"], [], 256, "(28+100) x 2"),
    (["j_onyx_agate"], ["9C", "9C"], [], 448, "28 x (2+14)"),
    (["j_rough_gem"], ["9D", "9D"], [], 56, "money only"),
    (["j_ticket"], ["9S(gold)", "9H(gold)"], [], 56, "money only"),
    (["j_hiker"], ["9S", "9H"], [], 56, "permanent chips land after this hand"),
    (["j_midas_mask"], ["KS", "KH"], [], 60, "faces turn gold; 30 x 2"),
    (["j_vampire"], ["9S(bonus)", "9H"], [], 127, "(28+30) x (2 x 1.1) = 127.6 floored"),
    (["j_lucky_cat"], ["9S", "9H"], [], 56, "x1 until a lucky trigger"),
    ([{"id": "j_idol", "state": {"rank": 9, "suit": 0}}], ["9S", "9H"], [], 112, "9S matches -> x2"),
    (["j_triboulet"], ["KS", "KH"], [], 240, "30 x (2x2x2)"),
    # --- retriggers ---
    (["j_hanging_chad"], ["9S", "9H"], [], 92, "first card x3: (10+27+9) x 2"),
    (["j_sock_and_buskin"], ["KS", "KH"], [], 100, "faces twice: (10+20+20) x 2"),
    (["j_seltzer"], ["9S", "9H"], [], 92, "all cards twice: (10+18+18) x 2"),
    # --- independent ---
    (["j_stencil"], ["9S", "9H"], [], 280, "5 slots, 1 joker counting itself empty: x5"),
    (["j_mystic_summit"], ["9S", "9H"], [], 56, "no round -> no discards info"),
    ([{"id": "j_loyalty_card", "state": {"remaining": 0}}], ["9S", "9H"], [], 224, "6th hand: x4"),
    (["j_steel_joker"], ["9S", "9H"], [], 56, "no steel cards"),
    (["j_stone"], ["9S", "9H"], [], 56, "no stone cards"),
    (["j_blue_joker"], ["9S", "9H"], [], 264, "52-card deck: +104 chips"),
    (["j_erosion"], ["9S", "9H"], [], 56, "deck at starting size"),
    (["j_bull"], ["9S", "9H"], [], 72, "$4 -> +8 chips"),
    (["j_bootstraps"], ["9S", "9H"], [], 56, "$4 -> 0 lots of $5"),
    (["j_swashbuckler", "j_joker"], ["9S", "9H"], [], 196, "28 x (2 + 1 sell + 4)"),
    (["j_drivers_license"], ["9S", "9H"], [], 56, "fewer than 16 enhanced"),
    (["j_acrobat"], ["9S", "9H"], [], 56, "no round -> not final hand"),
    (["j_blackboard"], ["9S", "9H"], [], 168, "nothing held -> x3"),
    (["j_blackboard"], ["9S", "9H"], ["9D"], 56, "a diamond held breaks it"),
    (["j_flower_pot"], ["9S", "9H", "9D", "9C"], [], 2016, "all suits scoring: 96 x 7 x 3"),
    (["j_seeing_double"], ["9C", "9H"], [], 112, "club + heart -> x2"),
    (["j_cavendish"], ["9S", "9H"], [], 168, "x3"),
    (["j_stuntman"], ["9S", "9H"], [], 556, "(28+250) x 2"),
    ([{"id": "j_todo_list", "state": {"hand": 2}}], ["9S", "9H"], [], 56, "pays $4, score unchanged"),
    (["j_matador"], ["9S", "9H"], [], 56, "no boss"),
    (["j_constellation"], ["9S", "9H"], [], 56, "x1"),
    (["j_hologram"], ["9S", "9H"], [], 56, "x1"),
    (["j_campfire"], ["9S", "9H"], [], 56, "x1"),
    (["j_glass"], ["9S", "9H"], [], 56, "x1"),
    (["j_hit_the_road"], ["9S", "9H"], [], 56, "x1"),
    (["j_madness"], ["9S", "9H"], [], 56, "x1"),
    (["j_ramen"], ["9S", "9H"], [], 112, "x2 fresh"),
    (["j_yorick"], ["9S", "9H"], [], 56, "x1"),
    (["j_canio"], ["9S", "9H"], [], 56, "x1"),
    (["j_ceremonial"], ["9S", "9H"], [], 56, "+0 mult"),
    (["j_flash"], ["9S", "9H"], [], 56, "+0 mult"),
    (["j_red_card"], ["9S", "9H"], [], 56, "+0 mult"),
    (["j_throwback"], ["9S", "9H"], [], 56, "no skips"),
    (["j_vagabond"], ["9S", "9H"], [], 56, "creates a tarot, score unchanged"),
    (["j_seance"], ["5S", "6S", "7S", "8S", "9S"], [], 1080, "straight flush (100+35) x 8"),
    (["j_superposition"], ["AS", "2H", "3D", "4C", "5S"], [], 220, "(30+25) x 4"),
    # --- no scoring effect at all ---
    (["j_credit_card"], ["9S", "9H"], [], 56, "passive"),
    (["j_delayed_grat"], ["9S", "9H"], [], 56, "passive"),
    (["j_faceless"], ["9S", "9H"], [], 56, "passive"),
    (["j_cloud_9"], ["9S", "9H"], [], 56, "passive"),
    (["j_rocket"], ["9S", "9H"], [], 56, "passive"),
    (["j_gift"], ["9S", "9H"], [], 56, "passive"),
    (["j_reserved_parking"], ["9S", "9H"], [], 56, "money only"),
    (["j_to_the_moon"], ["9S", "9H"], [], 56, "passive"),
    (["j_satellite"], ["9S", "9H"], [], 56, "passive"),
    (["j_trading"], ["9S", "9H"], [], 56, "passive"),
    (["j_diet_cola"], ["9S", "9H"], [], 56, "passive"),
    (["j_luchador"], ["9S", "9H"], [], 56, "passive"),
    (["j_chaos"], ["9S", "9H"], [], 56, "passive"),
    (["j_astronomer"], ["9S", "9H"], [], 56, "passive"),
    (["j_perkeo"], ["9S", "9H"], [], 56, "passive"),
    (["j_burglar"], ["9S", "9H"], [], 56, "passive"),
    (["j_juggler"], ["9S", "9H"], [], 56, "passive"),
    (["j_drunkard"], ["9S", "9H"], [], 56, "passive"),
    (["j_troubadour"], ["9S", "9H"], [], 56, "passive"),
    (["j_merry_andy"], ["9S", "9H"], [], 56, "passive"),
    (["j_turtle_bean"], ["9S", "9H"], [], 56, "passive"),
    (["j_marble"], ["9S", "9H"], [], 56, "passive"),
    (["j_certificate"], ["9S", "9H"], [], 56, "passive"),
    (["j_riff_raff"], ["9S", "9H"], [], 56, "passive"),
    (["j_cartomancer"], ["9S", "9H"], [], 56, "passive"),
    (["j_showman"], ["9S", "9H"], [], 56, "passive"),
    (["j_oops"], ["9S", "9H"], [], 56, "passive"),
    (["j_shortcut"], ["9S", "9H"], [], 56, "passive on a pair"),
    (["j_smeared"], ["9S", "9H"], [], 56, "passive on a pair"),
    (["j_mr_bones"], ["9S", "9H"], [], 56, "passive"),
    (["j_invisible"], ["9S", "9H"], [], 56, "passive"),
    (["j_chicot"], ["9S", "9H"], [], 56, "passive"),
    (["j_sixth_sense"], ["9S", "9H"], [], 56, "passive outside a round"),
    (["j_burnt"], ["9S", "9H"], [], 56, "passive"),
    (["j_hallucination"], ["9S", "9H"], [], 56, "passive"),
]


def _joker(spec):
    if isinstance(spec, str):
        return joker(spec)
    return joker(spec["id"], state=spec.get("state"))


@pytest.mark.parametrize("jspecs,played,held,expected,note", SMOKE, ids=lambda v: str(v)[:40])
def test_v2_joker_smoke(jspecs, played, held, expected, note):
    run = make_run(jokers=[_joker(j) for j in jspecs])
    res = score_play(run, cards(*played), held=cards(*held))
    assert res.score == expected, note + "\n" + res.explain()


def test_misprint_range():
    seen = set()
    for i in range(40):
        run = make_run(jokers=[joker("j_misprint")], seed=f"mis-{i}")
        res = score_play(run, cards("9S", "9H"))
        mult = res.mult - 2
        assert 0 <= mult <= 23
        seen.add(mult)
    assert len(seen) > 5


def test_money_jokers_pay():
    run = make_run(jokers=[joker("j_rough_gem")])
    assert score_play(run, cards("9D", "9D")).money_delta == 2
    run = make_run(jokers=[joker("j_ticket")])
    assert score_play(run, cards("9S(gold)", "9H(gold)")).money_delta == 8
    run = make_run(jokers=[joker("j_todo_list", state={"hand": 2})])
    assert score_play(run, cards("9S", "9H")).money_delta == 4
    run = make_run(jokers=[joker("j_bootstraps")])
    run.money = 10
    assert score_play(run, cards("9S", "9H")).score == 168  # 28 x (2+4)


def test_hiker_midas_vampire_mutate_cards():
    run = make_run(jokers=[joker("j_hiker")])
    hand = cards("9S", "9H")
    score_play(run, hand)
    assert all(c.extra_chips == 5 for c in hand)
    run = make_run(jokers=[joker("j_midas_mask")])
    hand = cards("KS", "KH")
    score_play(run, hand)
    assert all(str(c.enhancement) == "gold" for c in hand)
    run = make_run(jokers=[joker("j_vampire")])
    hand = cards("9S(bonus)", "9H(mult)")
    score_play(run, hand)
    assert all(str(c.enhancement) == "none" for c in hand)
    assert abs(run.jokers[0].state["x"] - 1.2) < 1e-9


def test_copies_never_grow_vampire():
    # Blueprint (left) fires first and eats the enhancement without growing;
    # Vampire then sees a plain card. With Brainstorm the real Vampire goes
    # first (leftmost), grows once, and the copy finds nothing left.
    run = make_run(jokers=[joker("j_blueprint"), joker("j_vampire")])
    score_play(run, cards("9S(bonus)", "9H"))
    assert abs(run.jokers[1].state["x"] - 1.0) < 1e-9
    run = make_run(jokers=[joker("j_vampire"), joker("j_brainstorm")])
    score_play(run, cards("9S(bonus)", "9H"))
    assert abs(run.jokers[0].state["x"] - 1.1) < 1e-9


def test_wee_and_seltzer_state():
    run = make_run(jokers=[joker("j_wee")])
    score_play(run, cards("2S", "2H"))
    assert run.jokers[0].state["chips"] == 16
    run = make_run(jokers=[joker("j_seltzer", state={"hands": 1})])
    score_play(run, cards("9S", "9H"))
    assert run.jokers == []  # drunk after its last hand


def test_loyalty_cycle():
    run = make_run(jokers=[joker("j_loyalty_card")])
    scores = [score_play(run, cards("9S", "9H")).score for _ in range(7)]
    assert scores == [56, 56, 56, 56, 56, 224, 56]


def test_creators_create():
    run = make_run(jokers=[joker("j_vagabond")])
    score_play(run, cards("9S", "9H"))
    assert len(run.consumables) == 1 and run.consumables[0].kind.value == "tarot"
    run = make_run(jokers=[joker("j_seance")])
    score_play(run, cards("5S", "6S", "7S", "8S", "9S"))
    assert run.consumables[0].kind.value == "spectral"
    run = make_run(jokers=[joker("j_superposition")])
    score_play(run, cards("AS", "2H", "3D", "4C", "5S"))
    assert run.consumables[0].kind.value == "tarot"
