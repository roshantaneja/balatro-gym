"""All 24 tags, 32 vouchers, 15 decks and 8 stakes."""

from __future__ import annotations

import pytest

from balatro_gym.engine import (
    DECKS,
    STAKES,
    TAGS,
    VOUCHERS,
    BlindKind,
    Edition,
    Game,
    HandType,
    Phase,
    PlayHand,
    SelectBlind,
    SkipBlind,
    SkipPack,
    Stake,
    TagInst,
    cardops,
    new_blind_scenario,
    score_play,
)
from balatro_gym.engine.levels import ante_base_chips
from balatro_gym.engine.round import acquire_tag
from balatro_gym.engine.shop import open_shop
from tests.helpers import cards, make_run

# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def test_tag_registry():
    assert len(TAGS) == 24
    assert TAGS["t_negative"].min_ante == 2 and TAGS["t_uncommon"].min_ante == 1


def test_skip_grants_prerolled_tag_and_ante1_pool_respects_min_ante():
    g = Game(seed="skip")
    tag = g.run.blind_tags[0]
    assert tag is not None and TAGS[tag.id].min_ante <= 1
    g.apply(SkipBlind())
    assert g.run.blind_index == 1
    # immediate tags consume themselves; others are held
    assert all(TAGS[t.id].min_ante <= 1 for t in g.run.tags)


def _shop_game(seed="tagshop", money=50):
    g = Game(seed=seed)
    g.run.money = money
    return g


def test_free_joker_tags():
    for tid, rarity in (("t_uncommon", "uncommon"), ("t_rare", "rare")):
        g = _shop_game(tid)
        g.run.tags = [TagInst(tid)]
        open_shop(g.run)
        from balatro_gym.engine import JOKERS

        item = g.run.shop.card_slots[0]
        assert item.kind == "joker" and item.cost == 0 and JOKERS[item.item_id].rarity.value == rarity
        assert g.run.tags == []


def test_edition_tags_persist_until_a_joker_shows():
    g = _shop_game("edtag")
    g.run.tags = [TagInst("t_negative")]
    open_shop(g.run)
    jokers = [it for it in g.run.shop.card_slots if it.kind == "joker"]
    if jokers:
        assert jokers[0].edition == Edition.NEGATIVE and jokers[0].cost == 0 and g.run.tags == []
    else:
        assert [t.id for t in g.run.tags] == ["t_negative"]


def test_voucher_coupon_d6_tags():
    g = _shop_game("vtag")
    g.run.tags = [TagInst("t_voucher"), TagInst("t_coupon"), TagInst("t_d6")]
    open_shop(g.run)
    shop = g.run.shop
    assert len(shop.vouchers) == 2 and len(set(shop.vouchers)) == 2
    assert all(it.cost == 0 for it in shop.card_slots) and all(p.cost == 0 for p in shop.pack_slots)
    assert shop.reroll_cost == 0 and g.run.tags == []


def test_immediate_money_tags():
    g = Game(seed="imm")
    g.run.stats["hands_played_run"] = 7
    g.run.stats["discards_unused_run"] = 3
    g.run.stats["blinds_skipped"] = 2
    g.run.money = 30
    for tid, gain in (("t_handy", 7), ("t_garbage", 3), ("t_speed", 10), ("t_economy", 40)):
        money = g.run.money
        acquire_tag(g.run, TagInst(tid))
        assert g.run.money == money + gain, tid
    assert g.run.tags == []


def test_top_up_orbital_boss_tags():
    g = Game(seed="topup")
    acquire_tag(g.run, TagInst("t_top_up"))
    assert len(g.run.jokers) == 2
    g = Game(seed="orbital")
    acquire_tag(g.run, TagInst("t_orbital", {"hand": int(HandType.PAIR)}))
    assert g.run.hand_levels[HandType.PAIR] == 4
    g = Game(seed="bosstag")
    old = g.run.ante_boss
    acquire_tag(g.run, TagInst("t_boss"))
    assert g.run.ante_boss != old


@pytest.mark.parametrize("tid,pack", [
    ("t_standard", "p_standard_mega"), ("t_charm", "p_arcana_mega"), ("t_meteor", "p_celestial_mega"),
    ("t_buffoon", "p_buffoon_mega"), ("t_ethereal", "p_spectral_mega"),
])
def test_pack_tags_open_a_free_mega_pack(tid, pack):
    g = Game(seed=tid)
    acquire_tag(g.run, TagInst(tid))
    assert g.run.phase == Phase.PACK_OPEN and g.run.open_pack.pack_id == pack
    assert g.run.open_pack.picks_left == 2
    g.apply(SkipPack())
    assert g.run.phase == Phase.BLIND_SELECT and g.run.tags == []


def test_investment_juggle_double():
    g = new_blind_scenario("inv", blind_kind=BlindKind.BOSS, hand_levels={HandType.PAIR: 40})
    g.run.tags = [TagInst("t_investment")]
    g.run.round.hand[:2] = cards("9S", "9H")
    events = g.apply(PlayHand((0, 1)))
    payout = [e for e in events if type(e).__name__ == "RoundPayout"][0]
    assert payout.other >= 25 and g.run.tags == []
    g = Game(seed="juggle")
    g.run.tags = [TagInst("t_juggle")]
    g.apply(SelectBlind())
    assert len(g.run.round.hand) == 11 and g.run.tags == []
    g = Game(seed="double")
    g.run.tags = [TagInst("t_double")]
    acquire_tag(g.run, TagInst("t_investment"))
    assert [t.id for t in g.run.tags] == ["t_investment", "t_investment"]
    g.run.tags = [TagInst("t_double")]
    g.run.money = 10
    acquire_tag(g.run, TagInst("t_economy"))
    assert g.run.money == 40 and g.run.tags == []  # doubled twice: 10 -> 20 -> 40


# ---------------------------------------------------------------------------
# Vouchers
# ---------------------------------------------------------------------------

def test_voucher_registry_and_requirements():
    assert len(VOUCHERS) == 32
    assert sum(1 for v in VOUCHERS.values() if v.requires) == 16
    run = make_run()
    from balatro_gym.engine.round import roll_voucher

    for _ in range(20):
        v = roll_voucher(run)
        assert VOUCHERS[v].requires is None
    run.vouchers = frozenset(VOUCHERS) - {"v_overstock_plus"}
    assert roll_voucher(run) == "v_overstock_plus"


def test_voucher_effects():
    run = make_run()
    checks = {
        "v_overstock": lambda r: r.shop_card_slots == 3,
        "v_overstock_plus": lambda r: r.shop_card_slots == 4,
        "v_clearance_sale": lambda r: r.mods["shop_discount"] == 25,
        "v_liquidation": lambda r: r.mods["shop_discount"] == 50,
        "v_hone": lambda r: r.mods["edition_rate"] == 2,
        "v_glow_up": lambda r: r.mods["edition_rate"] == 4,
        "v_reroll_surplus": lambda r: r.reroll_base == 3,
        "v_reroll_glut": lambda r: r.reroll_base == 1,
        "v_crystal_ball": lambda r: r.consumable_slots == 3,
        "v_omen_globe": lambda r: r.mods["omen_globe"],
        "v_telescope": lambda r: r.mods["telescope"],
        "v_observatory": lambda r: r.mods["observatory"] == 1.5,
        "v_grabber": lambda r: r.hands_per_round == 5,
        "v_nacho_tong": lambda r: r.hands_per_round == 6,
        "v_wasteful": lambda r: r.discards_per_round == 4,
        "v_recyclomancy": lambda r: r.discards_per_round == 5,
        "v_tarot_merchant": lambda r: r.mods["tarot_rate"] == 2.4,
        "v_tarot_tycoon": lambda r: r.mods["tarot_rate"] == 8.0,
        "v_planet_merchant": lambda r: r.mods["planet_rate"] == 2.4,
        "v_planet_tycoon": lambda r: r.mods["planet_rate"] == 8.0,
        "v_seed_money": lambda r: r.interest_cap == 50,
        "v_money_tree": lambda r: r.interest_cap == 100,
        "v_blank": lambda r: True,
        "v_antimatter": lambda r: r.joker_slots == 6,
        "v_magic_trick": lambda r: r.mods["card_rate"] == 4.0,
        "v_illusion": lambda r: r.mods["illusion"],
        "v_hieroglyph": lambda r: r.ante == 0 and r.hands_per_round == 5,
        "v_petroglyph": lambda r: r.ante == -1 and r.discards_per_round == 4,
        "v_directors_cut": lambda r: r.mods["boss_rerolls"] == 1,
        "v_retcon": lambda r: r.mods["boss_rerolls"] == -1,
        "v_paint_brush": lambda r: r.hand_size == 9,
        "v_palette": lambda r: r.hand_size == 10,
    }
    for vid, check in checks.items():
        cardops.grant_voucher(run, vid)
        assert check(run), vid
    assert run.vouchers == frozenset(VOUCHERS)


def test_observatory_stage_and_telescope_pack():
    run = make_run()
    cardops.grant_voucher(run, "v_telescope")
    cardops.grant_voucher(run, "v_observatory")
    cardops.create_consumable(run, "c_mercury")
    assert score_play(run, cards("9S", "9H")).score == 84  # 28 x (2 x 1.5)
    run.hand_play_counts[HandType.FLUSH] = 5
    from balatro_gym.engine.shop import _roll_pack_options

    pack = _roll_pack_options(run, "p_celestial_normal")
    assert "c_jupiter" in pack.option_ids


def test_shop_discount_and_magic_trick():
    g = Game(seed="disc")
    cardops.grant_voucher(g.run, "v_liquidation")
    cardops.grant_voucher(g.run, "v_magic_trick")
    g.run.mods["card_rate"] = 1000.0  # force playing cards
    open_shop(g.run)
    assert g.run.shop.card_slots and all(it.kind == "card" and it.card is not None for it in g.run.shop.card_slots)
    assert all(p.cost in (2, 3, 4) for p in g.run.shop.pack_slots)


def test_boss_reroll_voucher():
    g = Game(seed="dc")
    g.run.money = 30
    from balatro_gym.engine import RerollBoss

    assert not any(isinstance(a, RerollBoss) for a in g.legal_actions())
    cardops.grant_voucher(g.run, "v_directors_cut")
    assert any(isinstance(a, RerollBoss) for a in g.legal_actions())
    old = g.run.ante_boss
    g.apply(RerollBoss())
    assert g.run.money == 20 and g.run.ante_boss != old
    assert not any(isinstance(a, RerollBoss) for a in g.legal_actions())  # once per ante
    cardops.grant_voucher(g.run, "v_retcon")
    assert any(isinstance(a, RerollBoss) for a in g.legal_actions())


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------

def test_all_decks_start_correctly():
    assert len(DECKS) == 15
    r = Game(seed="d", deck_id="d_red").run
    assert r.discards_per_round == 4
    r = Game(seed="d", deck_id="d_blue").run
    assert r.hands_per_round == 5
    r = Game(seed="d", deck_id="d_yellow").run
    assert r.money == 14
    r = Game(seed="d", deck_id="d_green").run
    assert r.mods["no_interest"] and r.mods["extra_hand_bonus"] == 2
    r = Game(seed="d", deck_id="d_black").run
    assert r.joker_slots == 6 and r.hands_per_round == 3
    r = Game(seed="d", deck_id="d_magic").run
    assert "v_crystal_ball" in r.vouchers and [c.id for c in r.consumables] == ["c_fool", "c_fool"]
    r = Game(seed="d", deck_id="d_nebula").run
    assert "v_telescope" in r.vouchers and r.consumable_slots == 1
    r = Game(seed="d", deck_id="d_ghost").run
    assert r.mods["spectral_rate"] == 2 and [c.id for c in r.consumables] == ["c_hex"]
    r = Game(seed="d", deck_id="d_abandoned").run
    assert len(r.full_deck) == 40 and not any(c.rank.is_face for c in r.full_deck)
    assert r.stats["starting_deck_size"] == 40
    r = Game(seed="d", deck_id="d_checkered").run
    assert {int(c.suit) for c in r.full_deck} == {0, 1} and len(r.full_deck) == 52
    r = Game(seed="d", deck_id="d_zodiac").run
    assert {"v_tarot_merchant", "v_planet_merchant", "v_overstock"} <= r.vouchers
    r = Game(seed="d", deck_id="d_painted").run
    assert r.hand_size == 10 and r.joker_slots == 4
    r = Game(seed="d", deck_id="d_anaglyph").run
    assert r.mods["anaglyph_tag"] == "t_double"
    r = Game(seed="d", deck_id="d_plasma").run
    assert r.mods["plasma"] and r.mods["ante_scaling"] == 2
    r = Game(seed="d", deck_id="d_erratic").run
    from collections import Counter

    assert max(Counter(c.rank for c in r.full_deck).values()) != 4 or max(Counter(c.suit for c in r.full_deck).values()) != 13


def test_plasma_balances_and_doubles_requirement():
    g = new_blind_scenario("plasma", deck_id="d_plasma")
    assert g.run.round.chips_needed == 600
    run = make_run()
    run.mods["plasma"] = True
    res = score_play(run, cards("9S", "9H"))
    assert res.chips == 15 and res.mult == 15 and res.score == 225


def test_green_deck_payout_and_anaglyph():
    g = new_blind_scenario("green", deck_id="d_green", hand_levels={HandType.PAIR: 40})
    g.run.money = 25
    g.run.round.hand[:2] = cards("9S", "9H")
    events = g.apply(PlayHand((0, 1)))
    payout = [e for e in events if type(e).__name__ == "RoundPayout"][0]
    assert payout.interest == 0 and payout.other == 2 * 3 + 1 * 3  # 3 hands, 3 discards left


# ---------------------------------------------------------------------------
# Stakes
# ---------------------------------------------------------------------------

def test_stake_scaling_tables():
    assert ante_base_chips(8, Stake.WHITE) == 50_000
    assert ante_base_chips(8, Stake.GREEN) == 100_000
    assert ante_base_chips(8, Stake.PURPLE) == 200_000
    assert ante_base_chips(8, Stake.BLUE) == 100_000  # Blue includes Green's scaling
    assert ante_base_chips(9, Stake.WHITE) == 110_000
    assert ante_base_chips(0) == 100


def test_stake_modifiers_cumulative():
    assert len(STAKES) == 8
    r = Game(seed="s", stake="red").run
    assert r.mods.get("no_small_reward")
    r = Game(seed="s", stake="blue").run
    assert r.discards_per_round == 3 and r.mods.get("eternals") and r.mods.get("no_small_reward")
    r = Game(seed="s", stake="gold").run
    assert r.mods.get("rentals") and r.mods.get("perishables") and r.mods.get("eternals")


def test_red_stake_small_blind_pays_nothing():
    g = new_blind_scenario("red", stake="red", hand_levels={HandType.PAIR: 40})
    g.run.round.hand[:2] = cards("9S", "9H")
    events = g.apply(PlayHand((0, 1)))
    payout = [e for e in events if type(e).__name__ == "RoundPayout"][0]
    assert payout.blind_reward == 0


def test_stickers_from_gold_stake_shops():
    stickers = set()
    for i in range(30):
        g = Game(seed=f"gold-{i}", stake="gold")
        open_shop(g.run)
        for it in g.run.shop.card_slots:
            if it.kind == "joker":
                stickers |= set(it.stickers)
                if "rental" in it.stickers:
                    assert it.cost == 1
    assert {"eternal", "perishable", "rental"} <= stickers


def test_perishable_rental_eternal_behaviour():
    from balatro_gym.engine.round import _win_blind

    g = new_blind_scenario("stick", hand_levels={HandType.PAIR: 40})
    run = g.run
    j = cardops.new_joker(run, "j_joker", stickers=["perishable", "rental"])
    cardops.add_joker(run, j)
    assert j.perish_left == 5
    e = cardops.new_joker(run, "j_jolly", stickers=["eternal"])
    cardops.add_joker(run, e)
    from balatro_gym.engine import SellJoker

    assert not any(isinstance(a, SellJoker) and a.slot == 1 for a in g.legal_actions())
    assert not cardops.destroy_joker(run, e)
    for _ in range(5):
        run.round.chips_scored = run.round.chips_needed
        run.round.hand = cards("9S", "9H", "2D", "3D", "4D", "5D", "6D", "7D")
        events = _win_blind(run)
        payout = [ev for ev in events if type(ev).__name__ == "RoundPayout"][0]
        assert payout.other <= -3 + 0  # rental fee charged
        run.ante = 1
        run.round = g.run.round or run.round
        from balatro_gym.engine.round import start_blind

        start_blind(run, BlindKind.SMALL)
    assert j.perish_left == 0 and j.debuffed
    with_debuffed = score_play(run, cards("9S", "9H"), held=[]).score
    run.jokers = [x for x in run.jokers if x.uid != j.uid]
    assert score_play(run, cards("9S", "9H"), held=[]).score == with_debuffed  # it added nothing
