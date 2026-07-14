"""v1 consumables: 12 Planets + 13 Tarots (Planet Cards.wiki / Tarot Cards.wiki)."""

from __future__ import annotations

import pytest

from balatro_gym.engine import (
    CONSUMABLES,
    ConsumableInst,
    ConsumableKind,
    Edition,
    Enhancement,
    Game,
    HandType,
    IllegalActionError,
    PlayHand,
    Rank,
    Seal,
    SelectBlind,
    Suit,
    UseConsumable,
    hand_value,
    new_blind_scenario,
    score_play,
)
from balatro_gym.engine.effects import PLANET_FOR_HAND
from balatro_gym.engine.types import SECRET_HANDS

from tests.helpers import cards, joker, make_run

PLANET_IDS = {
    HandType.HIGH_CARD: "c_pluto",
    HandType.PAIR: "c_mercury",
    HandType.TWO_PAIR: "c_uranus",
    HandType.THREE_KIND: "c_venus",
    HandType.STRAIGHT: "c_saturn",
    HandType.FLUSH: "c_jupiter",
    HandType.FULL_HOUSE: "c_earth",
    HandType.FOUR_KIND: "c_mars",
    HandType.STRAIGHT_FLUSH: "c_neptune",
    HandType.FIVE_KIND: "c_planet_x",
    HandType.FLUSH_HOUSE: "c_ceres",
    HandType.FLUSH_FIVE: "c_eris",
}

TAROT_IDS = [
    "c_magician", "c_empress", "c_hierophant", "c_strength", "c_death",
    "c_hermit", "c_temperance", "c_tower", "c_star", "c_sun", "c_moon",
    "c_world", "c_wheel",
]


def use(run, cid: str, targets=()):
    CONSUMABLES[cid].effect.use(run, list(targets))


def give(run, cid: str) -> ConsumableInst:
    inst = ConsumableInst(id=cid, kind=CONSUMABLES[cid].kind, uid=run.new_uid())
    run.consumables.append(inst)
    return inst


# ---------------------------------------------------------------------------
# Planets
# ---------------------------------------------------------------------------

def test_all_planets_registered():
    for ht, cid in PLANET_IDS.items():
        d = CONSUMABLES[cid]
        assert d.kind == ConsumableKind.PLANET
        assert d.base_cost == 3
        assert d.effect.target_arity == (0, 0)


def test_planet_for_hand_mapping_complete():
    assert PLANET_FOR_HAND == {ht: cid for ht, cid in PLANET_IDS.items()}


@pytest.mark.parametrize("ht,cid", sorted(PLANET_IDS.items()))
def test_planet_levels_its_hand(ht, cid):
    run = make_run()
    run.hands_discovered |= SECRET_HANDS  # allow secret planets
    use(run, cid)
    assert run.hand_levels[ht] == 2
    assert all(run.hand_levels[o] == 1 for o in HandType if o != ht)


def test_mercury_changes_pair_scoring():
    run = make_run()
    before = score_play(run, cards("9S", "9H"), held=[]).score
    assert before == (10 + 9 + 9) * 2  # level-1 Pair: 10 chips x2 mult

    use(run, "c_mercury")
    after = score_play(run, cards("9S", "9H"), held=[]).score
    c, m = hand_value(HandType.PAIR, 2)
    assert after == (c + 9 + 9) * m == (25 + 18) * 3  # Mercury: +15 chips +1 mult


def test_secret_planet_can_use_gating():
    run = make_run()
    for ht in SECRET_HANDS:
        cid = PLANET_IDS[ht]
        assert not CONSUMABLES[cid].effect.can_use(run)
        run.hands_discovered.add(ht)
        assert CONSUMABLES[cid].effect.can_use(run)
    # Non-secret planets are always usable.
    assert CONSUMABLES["c_pluto"].effect.can_use(make_run())


def test_secret_planet_illegal_through_game():
    g = new_blind_scenario("secret-planet")
    give(g.run, "c_planet_x")
    assert not any(isinstance(a, UseConsumable) for a in g.legal_actions())
    with pytest.raises(IllegalActionError):
        g.apply(UseConsumable(0))
    # Discovering Five of a Kind makes it legal.
    g.run.hands_discovered.add(HandType.FIVE_KIND)
    assert UseConsumable(0) in g.legal_actions()
    g.apply(UseConsumable(0))
    assert g.run.hand_levels[HandType.FIVE_KIND] == 2
    assert g.run.consumables == []


def test_blue_seal_creates_planet_of_final_hand():
    # High Card leveled sky-high: any single-card play clears the small blind.
    g = new_blind_scenario("blue-seal", hand_levels={HandType.HIGH_CARD: 30})
    rnd = g.run.round
    rnd.hand[3].seal = Seal.BLUE  # held (not played) at end of round
    events = g.apply(PlayHand((0,)))  # single card => HIGH_CARD
    assert g.run.round is None  # blind cleared
    assert [c.id for c in g.run.consumables] == ["c_pluto"]
    assert g.run.consumables[0].kind == ConsumableKind.PLANET


# ---------------------------------------------------------------------------
# Tarots
# ---------------------------------------------------------------------------

def test_all_tarots_registered():
    for cid in TAROT_IDS:
        d = CONSUMABLES[cid]
        assert d.kind == ConsumableKind.TAROT
        assert d.base_cost == 3


@pytest.mark.parametrize(
    "cid,enh",
    [
        ("c_magician", Enhancement.LUCKY),
        ("c_empress", Enhancement.MULT),
        ("c_hierophant", Enhancement.BONUS),
    ],
)
def test_enhancer_tarots(cid, enh):
    run = make_run()
    assert CONSUMABLES[cid].effect.target_arity == (1, 2)
    c1, c2 = run.full_deck[0], run.full_deck[13]
    use(run, cid, [c1, c2])
    assert c1.enhancement == enh and c2.enhancement == enh
    # Mutation persists in full_deck by identity.
    assert run.full_deck[0] is c1 and run.full_deck[0].enhancement == enh


def test_tower_makes_stone():
    run = make_run()
    assert CONSUMABLES["c_tower"].effect.target_arity == (1, 1)
    c = run.full_deck[5]
    use(run, "c_tower", [c])
    assert c.enhancement == Enhancement.STONE
    assert c.base_chips == 50  # stone cards score a flat 50 chips


@pytest.mark.parametrize(
    "cid,suit",
    [
        ("c_star", Suit.DIAMONDS),
        ("c_sun", Suit.HEARTS),
        ("c_moon", Suit.CLUBS),
        ("c_world", Suit.SPADES),
    ],
)
def test_suit_converters(cid, suit):
    run = make_run()
    assert CONSUMABLES[cid].effect.target_arity == (1, 3)
    picked = [run.full_deck[i] for i in (0, 17, 30)]
    ranks = [c.rank for c in picked]
    enh_before = [c.enhancement for c in picked]
    use(run, cid, picked)
    assert all(c.suit == suit for c in picked)
    # "the rank and enhancements on the chosen cards remains unchanged"
    assert [c.rank for c in picked] == ranks
    assert [c.enhancement for c in picked] == enh_before


def test_strength_raises_rank_with_wraparound():
    run = make_run()
    nine, king, ace = cards("9S", "KH", "AD")
    use(run, "c_strength", [nine, king])
    assert nine.rank == Rank.TEN
    assert king.rank == Rank.ACE  # K -> A
    use(run, "c_strength", [ace])
    assert ace.rank == Rank.TWO  # A wraps to 2 (Tarot Cards.wiki rank order)


def test_death_copies_right_onto_left():
    run = make_run()
    left, right = cards("2S", "KH(glass,red_seal,foil)")
    right.extra_chips = 7
    left_uid = left.uid
    use(run, "c_death", [left, right])
    assert (left.rank, left.suit) == (Rank.KING, Suit.HEARTS)
    assert left.enhancement == Enhancement.GLASS
    assert left.edition == Edition.FOIL
    assert left.seal == Seal.RED
    assert left.extra_chips == 7
    assert left.uid == left_uid  # identity kept
    # Right card untouched.
    assert right.rank == Rank.KING and right.seal == Seal.RED


def test_death_left_is_first_target_in_selection_order():
    # Through the Game facade, targets are hand positions in the given order:
    # UseConsumable(slot, (3, 1)) means left=hand[3], right=hand[1].
    g = new_blind_scenario("death-order")
    give(g.run, "c_death")
    rnd = g.run.round
    left, right = rnd.hand[3], rnd.hand[1]
    want = (right.rank, right.suit)
    g.apply(UseConsumable(0, (3, 1)))
    assert (left.rank, left.suit) == want


def test_hermit_doubles_money_capped():
    run = make_run()
    run.money = 13
    use(run, "c_hermit")
    assert run.money == 26
    run.money = 50
    use(run, "c_hermit")
    assert run.money == 70  # max +$20
    run.money = 0
    use(run, "c_hermit")
    assert run.money == 0


def test_temperance_pays_joker_sell_value_capped():
    run = make_run()
    use(run, "c_temperance")
    assert run.money == 4  # no jokers -> +$0

    run = make_run(jokers=[joker("j_joker"), joker("j_half")])
    run.jokers[0].sell_value = 3
    run.jokers[1].sell_value = 5
    run.money = 0
    use(run, "c_temperance")
    assert run.money == 8

    run.jokers[0].sell_value = 60  # over the cap
    use(run, "c_temperance")
    assert run.money == 8 + 50  # max $50


def test_wheel_of_fortune_hit_and_miss():
    # Seeds found by trial: "wheel-1" hits (Holo), "wheel-0" misses.
    run = make_run(jokers=[joker("j_joker")], seed="wheel-1")
    use(run, "c_wheel")
    assert run.jokers[0].edition == Edition.HOLO

    run = make_run(jokers=[joker("j_joker")], seed="wheel-0")
    use(run, "c_wheel")
    assert run.jokers[0].edition == Edition.NONE


def test_wheel_of_fortune_skips_editioned_jokers():
    # Only edition-less jokers are eligible targets.
    run = make_run(
        jokers=[joker("j_joker", edition="poly"), joker("j_half")], seed="wheel-1"
    )
    use(run, "c_wheel")
    assert run.jokers[0].edition == Edition.POLY  # untouched
    assert run.jokers[1].edition == Edition.HOLO


def test_wheel_of_fortune_can_use_gating():
    eff = CONSUMABLES["c_wheel"].effect
    assert not eff.can_use(make_run())  # no jokers
    assert not eff.can_use(make_run(jokers=[joker("j_joker", edition="foil")]))
    assert eff.can_use(make_run(jokers=[joker("j_joker", edition="foil"), joker("j_half")]))


# ---------------------------------------------------------------------------
# Through the Game facade
# ---------------------------------------------------------------------------

def test_targeted_tarot_through_game():
    g = new_blind_scenario("facade-sun")
    give(g.run, "c_sun")
    rnd = g.run.round

    acts = [a for a in g.legal_actions() if isinstance(a, UseConsumable)]
    # Targeted arities enumerated over hand positions: C(8,1)+C(8,2)+C(8,3).
    assert len(acts) == 8 + 28 + 56
    assert UseConsumable(0, (0, 1, 2)) in acts

    targets = [rnd.hand[i] for i in (0, 1, 2)]
    events = g.apply(UseConsumable(0, (0, 1, 2)))
    assert all(c.suit == Suit.HEARTS for c in targets)
    assert g.run.consumables == []  # consumed
    assert g.run.stats["tarots_used"] == 1
    # Same objects live in full_deck -> the change persists for the run.
    by_uid = {c.uid: c for c in g.run.full_deck}
    for c in targets:
        assert by_uid[c.uid] is c


def test_targeted_tarot_not_enumerated_outside_playing():
    g = Game(seed="facade-blindselect", enable_skip_blind=False)
    give(g.run, "c_sun")     # targeted: needs a hand on the table
    give(g.run, "c_hermit")  # untargeted: usable anywhere
    acts = [a for a in g.legal_actions() if isinstance(a, UseConsumable)]
    assert acts == [UseConsumable(1)]


def test_untargeted_planet_through_game():
    g = new_blind_scenario("facade-mercury")
    give(g.run, "c_mercury")
    assert UseConsumable(0) in g.legal_actions()
    g.apply(UseConsumable(0))
    assert g.run.hand_levels[HandType.PAIR] == 2
    assert g.run.consumables == []
    assert g.run.stats["planets_used"] == 1
