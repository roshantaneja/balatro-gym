"""Behavioural tests for v2 jokers whose effects live outside a single scored
hand: hand-size / hands / discards modifiers, blind-start creators, economy,
lifecycle, loss prevention, pool gating."""

from __future__ import annotations

from balatro_gym.engine import (
    BlindKind,
    BuyPack,
    Discard,
    Edition,
    Enhancement,
    Game,
    HandType,
    LeaveShop,
    Phase,
    PlayHand,
    Rarity,
    Reroll,
    SellJoker,
    SkipBlind,
    SkipPack,
    cardops,
    new_blind_scenario,
    score_play,
)
from balatro_gym.engine.hand_eval import evaluate_hand
from balatro_gym.engine.round import _win_blind
from balatro_gym.engine.scoring import fold_eval_rules
from tests.helpers import cards, joker, make_run


def scenario(*jids, seed="beh", **kw):
    return new_blind_scenario(seed, joker_ids=list(jids), **kw)


# ---------------------------------------------------------------------------
# Polled modifiers
# ---------------------------------------------------------------------------

def test_hand_size_modifiers():
    assert len(scenario("j_juggler").run.round.hand) == 9
    assert len(scenario("j_troubadour").run.round.hand) == 10
    assert len(scenario("j_merry_andy").run.round.hand) == 7
    assert len(scenario("j_stuntman").run.round.hand) == 6
    assert len(scenario("j_turtle_bean").run.round.hand) == 13


def test_hands_and_discards_modifiers():
    g = scenario("j_troubadour")
    assert g.run.round.hands_left == 3  # 4 - 1
    g = scenario("j_drunkard")
    assert g.run.round.discards_left == 5  # red deck 4 + 1
    g = scenario("j_merry_andy")
    assert g.run.round.discards_left == 7


def test_burglar_blind_start():
    g = scenario("j_burglar")
    assert g.run.round.hands_left == 7 and g.run.round.discards_left == 0


def test_turtle_bean_decays_and_is_eaten():
    run = make_run(jokers=[joker("j_turtle_bean")])
    from balatro_gym.engine.effects import joker_def

    for h in (4, 3, 2, 1):
        joker_def("j_turtle_bean").effect.update_on_round_end(run, run.jokers[0])
        assert run.jokers[0].state["h"] == h
    joker_def("j_turtle_bean").effect.update_on_round_end(run, run.jokers[0])
    assert run.jokers == []


def test_credit_card_debt_and_chaos_free_reroll():
    g = scenario("j_credit_card")
    assert cardops.can_afford(g.run, 20)
    assert not cardops.can_afford(g.run, 30)
    run = make_run(jokers=[joker("j_chaos")])
    from balatro_gym.engine.shop import open_shop

    run.money = 0
    open_shop(run)
    assert run.shop.free_rerolls == 1
    g = Game(seed="chaos")
    g.run.jokers = [joker("j_chaos")]
    from balatro_gym.engine.shop import open_shop as _os

    _os(g.run)
    g.run.money = 0
    assert any(isinstance(a, Reroll) for a in g.legal_actions())
    g.apply(Reroll())
    assert g.run.money == 0 and g.run.shop.free_rerolls == 0


def test_oops_doubles_chances():
    run = make_run(jokers=[joker("j_oops")])
    cardops.refresh_chance_multiplier(run)
    assert run.chance_multiplier == 2
    run.jokers.append(joker("j_oops"))
    cardops.refresh_chance_multiplier(run)
    assert run.chance_multiplier == 4


def test_shortcut_and_smeared_rules():
    run = make_run(jokers=[joker("j_shortcut")])
    assert evaluate_hand(cards("2S", "4H", "6D", "8C", "10S"), fold_eval_rules(run)).hand_type == HandType.STRAIGHT
    run = make_run(jokers=[joker("j_smeared")])
    assert evaluate_hand(cards("2H", "4D", "6H", "8D", "10H"), fold_eval_rules(run)).hand_type == HandType.FLUSH


def test_showman_allows_duplicates_in_pool():
    run = make_run(jokers=[joker("j_joker")])
    assert "j_joker" not in cardops.joker_pool(run, Rarity.COMMON)
    run.jokers.append(joker("j_showman"))
    assert "j_joker" in cardops.joker_pool(run, Rarity.COMMON)


def test_legendaries_never_in_normal_pool_and_cavendish_gated():
    run = make_run()
    pool = cardops.joker_pool(run, None)
    assert "j_canio" not in pool and "j_perkeo" not in pool and "j_cavendish" not in pool
    assert "j_gros_michel" in pool
    run.stats["gros_michel_extinct"] = True
    pool = cardops.joker_pool(run, None)
    assert "j_cavendish" in pool and "j_gros_michel" not in pool
    assert "j_canio" in cardops.joker_pool(run, Rarity.LEGENDARY, legendary=True)


# ---------------------------------------------------------------------------
# Round-conditioned scoring
# ---------------------------------------------------------------------------

def test_card_sharp_acrobat_mystic_blue_with_round():
    g = scenario("j_card_sharp", hand_levels={HandType.PAIR: 1})
    rnd = g.run.round
    rnd.hand = cards("9S", "9H", "9D", "9C", "2S", "3S", "4S", "5S")
    g.apply(PlayHand((0, 1)))                 # first pair: 56
    rnd = g.run.round
    rnd.hand[:2] = cards("8S", "8H")
    res = g.apply(PlayHand((0, 1)))[0].result  # pair again: x3
    assert res.score == 26 * 2 * 3

    g = scenario("j_acrobat", hands=1)
    g.run.round.hand[:2] = cards("9S", "9H")
    assert g.apply(PlayHand((0, 1)))[0].result.score == 168

    g = scenario("j_mystic_summit", discards=0)
    g.run.round.hand[:2] = cards("9S", "9H")
    assert g.apply(PlayHand((0, 1)))[0].result.score == 28 * 17

    g = scenario("j_blue_joker")
    g.run.round.hand[:2] = cards("9S", "9H")
    n = len(g.run.round.deck_remaining)
    assert g.apply(PlayHand((0, 1)))[0].result.score == (28 + 2 * n) * 2


def test_obelisk_scales_off_most_played():
    run = make_run(jokers=[joker("j_obelisk")])
    run.hand_play_counts[HandType.PAIR] = 3
    assert score_play(run, cards("9S", "9H", "9D")).score == 205  # floor(57 x 3 x 1.2): not most played
    assert score_play(run, cards("9S", "9H")).score == 56  # pair resets to x1


def test_matador_pays_when_boss_triggers():
    g = scenario("j_matador", blind_kind=BlindKind.BOSS, boss_id="b_tooth", ante=3)
    g.run.round.hand[:2] = cards("9S", "9H")
    res = g.apply(PlayHand((0, 1)))[0].result
    assert res.money_delta == 8 - 2


def test_lucky_cat_grows_on_lucky_trigger():
    hits = 0
    for i in range(60):
        run = make_run(jokers=[joker("j_lucky_cat")], seed=f"cat-{i}")
        score_play(run, cards("9S(lucky)", "9H(lucky)"))
        if run.jokers[0].state["x"] > 1:
            hits += 1
            assert run.jokers[0].state["x"] in (1.25, 1.5, 1.75, 2.0)
    assert hits > 5


def test_idol_and_todo_reroll_each_round():
    run = make_run(jokers=[joker("j_idol", state={"rank": 14, "suit": 0}), joker("j_todo_list")])
    from balatro_gym.engine.effects import joker_def

    seen_idol, seen_todo = set(), set()
    for _ in range(12):
        for j in run.jokers:
            joker_def(j.id).effect.update_on_round_end(run, j)
        seen_idol.add((run.jokers[0].state["rank"], run.jokers[0].state["suit"]))
        seen_todo.add(run.jokers[1].state["hand"])
    assert len(seen_idol) > 1 and len(seen_todo) > 1


# ---------------------------------------------------------------------------
# Event-driven scalers
# ---------------------------------------------------------------------------

def test_yorick_hit_the_road_ramen_on_discard():
    g = scenario("j_yorick", "j_hit_the_road", "j_ramen")
    rnd = g.run.round
    rnd.hand[:5] = cards("JS", "JH", "2D", "3C", "4S")
    g.apply(Discard((0, 1, 2, 3, 4)))
    y, h, r = g.run.jokers
    assert y.state["discards"] == 5
    assert h.state["x"] == 2.0  # two jacks
    assert abs(r.state["x"] - 1.95) < 1e-9
    _win_blind(g.run)
    assert g.run.jokers[1].state["x"] == 1.0  # resets at round end


def test_ramen_eaten_at_x1():
    g = scenario("j_ramen")
    g.run.jokers[0].state["x"] = 1.01
    g.apply(Discard((0,)))
    assert g.run.jokers == []


def test_yorick_gains_every_23():
    run = make_run(jokers=[joker("j_yorick")])
    from balatro_gym.engine.effects import joker_def

    joker_def("j_yorick").effect.update_on_discard(run, run.jokers[0], cards(*["2S"] * 25))
    assert run.jokers[0].state["x"] == 2.0 and run.jokers[0].state["discards"] == 2


def test_glass_joker_canio_on_card_destruction():
    run = make_run(jokers=[joker("j_glass"), joker("j_canio")])
    cardops.destroy_cards(run, cards("KS(glass)", "9H"))
    assert run.jokers[0].state["x"] == 1.75
    assert run.jokers[1].state["x"] == 2.0


def test_hologram_on_card_added():
    run = make_run(jokers=[joker("j_hologram")])
    cardops.add_card(run, cards("9S")[0], to_hand=False)
    assert run.jokers[0].state["x"] == 1.25


def test_constellation_on_planet_used():
    g = Game(seed="const")
    g.run.jokers = [joker("j_constellation")]
    cardops.create_consumable(g.run, "c_mercury")
    from balatro_gym.engine import UseConsumable

    g.apply(UseConsumable(0))
    assert abs(g.run.jokers[0].state["x"] - 1.1) < 1e-9
    assert g.run.hand_levels[HandType.PAIR] == 2


def test_campfire_sells_and_resets():
    g = Game(seed="camp")
    g.run.jokers = [joker("j_campfire"), joker("j_joker")]
    g.apply(SellJoker(1))
    assert g.run.jokers[0].state["x"] == 1.25
    from balatro_gym.engine.effects import joker_def

    joker_def("j_campfire").effect.update_on_boss_defeated(g.run, g.run.jokers[0])
    assert g.run.jokers[0].state["x"] == 1.0


def test_flash_card_red_card_throwback():
    g = Game(seed="flash")
    g.run.jokers = [joker("j_flash"), joker("j_red_card"), joker("j_throwback")]
    g.run.money = 50
    g.apply(SkipBlind())
    assert g.run.stats["blinds_skipped"] == 1
    run = make_run(jokers=[joker("j_throwback")])
    run.stats["blinds_skipped"] = 2
    assert score_play(run, cards("9S", "9H")).score == 84  # 28 x (2 x 1.5)
    from balatro_gym.engine.shop import open_shop

    open_shop(g.run)
    g.apply(Reroll())
    assert g.run.jokers[0].state["mult"] == 2
    g.run.money = 50
    g.apply(BuyPack(0))
    g.apply(SkipPack())
    assert g.run.jokers[1].state["mult"] == 3


def test_ceremonial_dagger_and_madness():
    g = scenario("j_ceremonial", "j_joker")
    assert [j.id for j in g.run.jokers] == ["j_ceremonial"]
    assert g.run.jokers[0].state["mult"] == 2  # 2 x sell value 1
    g = scenario("j_madness", "j_joker")
    assert [j.id for j in g.run.jokers] == ["j_madness"] and g.run.jokers[0].state["x"] == 1.5
    g = scenario("j_madness", "j_joker", blind_kind=BlindKind.BOSS)
    assert len(g.run.jokers) == 2  # boss blinds don't trigger it


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------

def test_round_end_money_jokers():
    g = scenario("j_rocket", "j_cloud_9", "j_satellite", "j_delayed_grat", "j_golden")
    run = g.run
    run.stats["planets_used_ids"] = ["c_mercury", "c_venus"]
    before = run.money
    from balatro_gym.engine.effects import joker_def

    other = sum(joker_def(j.id).effect.round_end_money(run, j) for j in run.jokers)
    # rocket 1 + four 9s + 2 planets + 2 x 4 discards (red deck) + golden 4
    assert other == 1 + 4 + 2 + 8 + 4
    joker_def("j_rocket").effect.update_on_boss_defeated(run, run.jokers[0])
    assert run.jokers[0].state["dollars"] == 3
    assert run.money == before


def test_gift_card_and_to_the_moon():
    g = scenario("j_gift", "j_to_the_moon")
    run = g.run
    cardops.create_consumable(run, "c_mercury")
    from balatro_gym.engine.effects import joker_def

    before = [j.sell_value for j in run.jokers]
    joker_def("j_gift").effect.update_on_round_end(run, run.jokers[0])
    assert [j.sell_value for j in run.jokers] == [b + 1 for b in before]
    assert run.consumables[0].sell_value == 2
    run.money = 25
    run.round.chips_scored = run.round.chips_needed
    events = _win_blind(run)
    payout = [e for e in events if type(e).__name__ == "RoundPayout"][0]
    assert payout.interest == 10  # $2 per $5


def test_faceless_and_trading_card():
    g = scenario("j_faceless", "j_trading")
    rnd = g.run.round
    rnd.hand[:3] = cards("KS", "QH", "JD")
    money = g.run.money
    g.apply(Discard((0, 1, 2)))
    assert g.run.money == money + 5
    g = scenario("j_trading")
    victim = g.run.round.hand[0]
    money = g.run.money
    g.apply(Discard((0,)))
    assert g.run.money == money + 3
    assert all(c.uid != victim.uid for c in g.run.full_deck)


def test_diet_cola_and_luchador_on_sell():
    g = Game(seed="cola")
    g.run.jokers = [joker("j_diet_cola")]
    g.apply(SellJoker(0))
    assert [t.id for t in g.run.tags] == ["t_double"]
    g = scenario("j_luchador", blind_kind=BlindKind.BOSS, boss_id="b_needle", ante=2)
    assert g.run.round.hands_left == 1
    g.apply(SellJoker(0))
    assert g.run.round.boss_state["disabled"] is True


def test_astronomer_free_planets_and_perkeo():
    run = make_run(jokers=[joker("j_astronomer")])
    from balatro_gym.engine.shop import open_shop, pack_cost

    assert pack_cost(run, "p_celestial_normal") == 0
    open_shop(run)
    for it in run.shop.card_slots:
        if it.kind == "planet":
            assert it.cost == 0
    g = Game(seed="perkeo")
    g.run.jokers = [joker("j_perkeo")]
    cardops.create_consumable(g.run, "c_mercury")
    open_shop(g.run)
    g.apply(LeaveShop())
    assert len(g.run.consumables) == 2 and g.run.consumables[1].edition == Edition.NEGATIVE


def test_reserved_parking_pays_sometimes():
    paid = 0
    for i in range(40):
        run = make_run(jokers=[joker("j_reserved_parking")], seed=f"park-{i}")
        paid += score_play(run, cards("9S", "9H"), held=cards("KD", "QD")).money_delta
    assert 0 < paid < 80


# ---------------------------------------------------------------------------
# Blind-start creators and lifecycle
# ---------------------------------------------------------------------------

def test_marble_certificate_riff_raff_cartomancer():
    g = scenario("j_marble", "j_certificate")
    assert sum(1 for c in g.run.full_deck if c.is_stone) == 1
    assert sum(1 for c in g.run.full_deck if str(c.seal) != "none") == 1
    assert len(g.run.full_deck) == 54
    g = scenario("j_riff_raff")
    assert len(g.run.jokers) == 3
    assert all(g.run.jokers[i].id != "j_riff_raff" for i in (1, 2))
    g = scenario("j_cartomancer")
    assert len(g.run.consumables) == 1


def test_hallucination_on_pack_open():
    hits = 0
    for i in range(30):
        g = Game(seed=f"hall-{i}")
        g.run.jokers = [joker("j_hallucination")]
        g.run.money = 50
        from balatro_gym.engine.shop import open_shop

        open_shop(g.run)
        g.apply(BuyPack(0))
        hits += len(g.run.consumables)
    assert 0 < hits < 30


def test_mr_bones_saves_once():
    g = scenario("j_mr_bones", hands=1)
    rnd = g.run.round
    rnd.chips_needed = 1000
    rnd.hand[:2] = cards("9S", "9H")  # scores 56 >= 25% of ... no; bump requirement
    rnd.chips_needed = 200  # 56 >= 50
    g.apply(PlayHand((0, 1)))
    assert g.run.phase == Phase.SHOP and g.run.jokers == []


def test_mr_bones_does_not_save_below_quarter():
    g = scenario("j_mr_bones", hands=1)
    g.run.round.chips_needed = 1000
    g.run.round.hand[:2] = cards("9S", "9H")
    g.apply(PlayHand((0, 1)))
    assert g.run.phase == Phase.GAME_OVER


def test_invisible_joker_copies_after_two_rounds():
    g = Game(seed="inv")
    g.run.jokers = [joker("j_invisible"), joker("j_joker")]
    g.apply(SellJoker(0))
    assert [j.id for j in g.run.jokers] == ["j_joker"]
    g.run.jokers = [joker("j_invisible", state={"rounds": 2}), joker("j_joker")]
    g.apply(SellJoker(0))
    assert [j.id for j in g.run.jokers] == ["j_joker", "j_joker"]


def test_chicot_disables_boss():
    g = scenario("j_chicot", blind_kind=BlindKind.BOSS, boss_id="b_needle", ante=2)
    assert g.run.round.boss_state["disabled"] is True
    assert g.run.round.hands_left == 4  # Needle's hands=1 never applied


def test_sixth_sense_and_burnt_joker():
    g = scenario("j_sixth_sense")
    g.run.round.hand[0] = cards("6S")[0]
    six = g.run.round.hand[0]
    g.apply(PlayHand((0,)))
    assert len(g.run.consumables) == 1 and g.run.consumables[0].kind.value == "spectral"
    assert all(c.uid != six.uid for c in g.run.full_deck)
    g = scenario("j_burnt")
    g.run.round.hand[:2] = cards("9S", "9H")
    g.apply(Discard((0, 1)))
    assert g.run.hand_levels[HandType.PAIR] == 2
    g.run.round.hand[:2] = cards("8S", "8H")
    g.apply(Discard((0, 1)))
    assert g.run.hand_levels[HandType.PAIR] == 2  # only the first discard


def test_steel_stone_erosion_drivers_license_counts():
    run = make_run(jokers=[joker("j_steel_joker"), joker("j_stone"), joker("j_erosion"), joker("j_drivers_license")])
    for c in run.full_deck[:16]:
        c.enhancement = Enhancement.STEEL
    run.full_deck[16].enhancement = Enhancement.STONE
    run.full_deck = run.full_deck[:50]
    res = score_play(run, cards("9S", "9H"))
    # chips 28 + 25 (stone) = 53; mult in slot order: 2 x 4.2 (steel) + 8 (erosion) x 3 (licence)
    assert res.chips == 53 and abs(res.mult - (2 * 4.2 + 8) * 3) < 1e-6


def test_flower_pot_with_wild_and_stencil_slots():
    run = make_run(jokers=[joker("j_flower_pot")])
    res = score_play(run, cards("9S", "9H", "9D", "9C(wild)"))
    assert res.mult == 21
    run = make_run(jokers=[joker("j_stencil"), joker("j_joker")])
    # 5 slots, 2 jokers, stencil counts itself: 5 - 2 + 1 = 4 empty -> x4, then +4: 28 x (2x4+4)
    assert score_play(run, cards("9S", "9H")).score == 336
