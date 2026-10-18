"""The 14 bosses added in v2 (10 regular + 4 finishers), ante pre-rolls and
boss disabling."""

from __future__ import annotations

from balatro_gym.engine import (
    BOSSES,
    BlindKind,
    Discard,
    Game,
    HandType,
    Phase,
    PlayHand,
    SellJoker,
    SwapJokers,
    new_blind_scenario,
)
from balatro_gym.engine.round import begin_ante
from tests.helpers import cards


def boss_game(seed, boss_id, *, ante=1, jokers=(), **kw):
    return new_blind_scenario(seed, ante=ante, blind_kind=BlindKind.BOSS, boss_id=boss_id,
                              joker_ids=list(jokers), **kw)


def test_all_28_registered_with_antes():
    assert len(BOSSES) == 28
    finishers = {b for b, d in BOSSES.items() if d.effect.is_finisher}
    assert finishers == {"b_amber_acorn", "b_verdant_leaf", "b_violet_vessel", "b_crimson_heart", "b_cerulean_bell"}
    assert BOSSES["b_eye"].effect.min_ante == 3 and BOSSES["b_plant"].effect.min_ante == 4
    assert BOSSES["b_serpent"].effect.min_ante == 5 and BOSSES["b_ox"].effect.min_ante == 6
    assert all(BOSSES[b].effect.reward() == 8 for b in finishers)


def test_ante_prerolls_boss_visible_before_play():
    g = Game(seed="pre")
    assert g.run.ante_boss in BOSSES and g.run.ante_boss in g.run.bosses_seen
    assert BOSSES[g.run.ante_boss].effect.min_ante <= 1
    g.run.ante = 8
    begin_ante(g.run)
    assert BOSSES[g.run.ante_boss].effect.is_finisher


def test_wheel_and_mark_face_down():
    g = boss_game("wheel", "b_wheel", ante=2)
    total = 0
    for i in range(10):
        g = boss_game(f"wheel{i}", "b_wheel", ante=2)
        total += len(g.run.round.face_down_uids)
    assert 0 < total < 40  # ~1 in 7 of 80 cards
    g = boss_game("mark", "b_mark", ante=2)
    rnd = g.run.round
    faces = {c.uid for c in rnd.hand if c.rank.is_face}
    assert set(rnd.face_down_uids) == faces


def test_fish_draws_face_down_after_play_only():
    g = boss_game("fish", "b_fish", ante=2)
    rnd = g.run.round
    assert rnd.face_down_uids == []
    g.apply(Discard((0, 1)))
    assert rnd.face_down_uids == []
    g.apply(PlayHand((0,)))
    rnd = g.run.round
    assert len(rnd.face_down_uids) == 1


def test_water_and_serpent():
    g = boss_game("water", "b_water", ante=2)
    assert g.run.round.discards_left == 0
    g = boss_game("serpent", "b_serpent", ante=5)
    g.run.round.hand = g.run.round.hand[:8]
    g.apply(Discard((0,)))
    assert len(g.run.round.hand) == 10  # 7 left + 3 drawn
    g.apply(PlayHand((0, 1, 2, 3, 4)))
    assert len(g.run.round.hand) == 8  # 5 left + 3 drawn


def test_eye_and_mouth_restrict_hand_types():
    g = boss_game("eye", "b_eye", ante=3, hand_levels={HandType.PAIR: 1})
    rnd = g.run.round
    rnd.hand = cards("9S", "9H", "8S", "8H", "2D", "3C", "4S", "5S")
    g.apply(PlayHand((0, 1)))
    rnd = g.run.round
    legal = g.legal_actions()
    pair_again = PlayHand((0, 1))  # 8S 8H now at the front
    assert pair_again not in legal
    assert PlayHand((0,)) in legal  # high card is a new type

    g = boss_game("mouth", "b_mouth", ante=2)
    rnd = g.run.round
    rnd.hand = cards("9S", "9H", "8S", "8H", "2D", "3C", "4S", "5S")
    g.apply(PlayHand((0, 1)))
    legal = g.legal_actions()
    assert PlayHand((0, 1)) in legal  # another pair is fine
    assert PlayHand((0,)) not in legal  # high card is a different type


def test_plant_and_pillar_debuffs():
    g = boss_game("plant", "b_plant", ante=4)
    rnd = g.run.round
    rnd.hand[:2] = cards("KS", "KH")
    res = g.apply(PlayHand((0, 1)))[0].result
    assert res.chips == 10  # faces contribute nothing
    g = boss_game("pillar", "b_pillar")
    rnd = g.run.round
    first = rnd.hand[0]
    g.run.ante_played_uids = [first.uid]
    rnd.hand[1] = cards("9S")[0]
    rnd.hand[0] = first
    first.rank = cards("9H")[0].rank
    res = g.apply(PlayHand((0, 1)))[0].result
    assert res.chips == 10 + 9  # only the fresh card scores


def test_tooth_costs_money():
    g = boss_game("tooth", "b_tooth", ante=3)
    money = g.run.money
    res = g.apply(PlayHand((0, 1, 2)))[0].result
    assert res.money_delta == -3 and g.run.money == money - 3


def test_amber_acorn_hides_and_locks_jokers():
    g = boss_game("acorn", "b_amber_acorn", ante=8, jokers=["j_joker", "j_jolly", "j_sly"])
    assert g.run.round.boss_state["jokers_hidden"]
    assert not any(isinstance(a, SwapJokers) for a in g.legal_actions())


def test_verdant_leaf_until_sale():
    g = boss_game("leaf", "b_verdant_leaf", ante=8, jokers=["j_joker"])
    rnd = g.run.round
    rnd.hand[:2] = cards("9S", "9H")
    res = g.apply(PlayHand((0, 1)))[0].result
    assert res.chips == 10  # everything debuffed
    g.apply(SellJoker(0))
    rnd = g.run.round
    rnd.hand[:2] = cards("9S", "9H")
    res = g.apply(PlayHand((0, 1)))[0].result
    assert res.chips == 28


def test_crimson_heart_disables_a_joker():
    g = boss_game("heart", "b_crimson_heart", ante=8, jokers=["j_joker"])
    rnd = g.run.round
    assert rnd.boss_state["disabled_joker_uid"] == g.run.jokers[0].uid
    rnd.hand[:2] = cards("9S", "9H")
    res = g.apply(PlayHand((0, 1)))[0].result
    assert res.mult == 2  # +4 never fired


def test_cerulean_bell_forces_a_card():
    g = boss_game("bell", "b_cerulean_bell", ante=8)
    rnd = g.run.round
    forced = rnd.boss_state["forced_uid"]
    idx = next(i for i, c in enumerate(rnd.hand) if c.uid == forced)
    legal = g.legal_actions()
    plays = [a for a in legal if isinstance(a, PlayHand)]
    assert plays and all(idx in a.indices for a in plays)
    assert all(idx in a.indices for a in legal if isinstance(a, Discard))
    g.apply(plays[0])
    rnd = g.run.round
    assert rnd.boss_state.get("forced_uid") in {c.uid for c in rnd.hand}


def test_violet_vessel_requirement():
    g = boss_game("vessel", "b_violet_vessel", ante=8)
    assert g.run.round.chips_needed == 300_000


def test_chicot_and_luchador_disable_boss_effects():
    g = boss_game("chi", "b_plant", ante=4, jokers=["j_chicot"])
    rnd = g.run.round
    rnd.hand[:2] = cards("KS", "KH")
    res = g.apply(PlayHand((0, 1)))[0].result
    assert res.chips == 30  # faces score normally


def test_psychic_with_small_hand_is_a_loss_not_a_hang():
    g = boss_game("psy", "b_psychic", hands=2, discards=0)
    g.run.round.hand = g.run.round.hand[:5]
    g.run.round.deck_remaining = []
    g.apply(PlayHand((0, 1, 2, 3, 4)))
    assert g.run.phase == Phase.GAME_OVER
