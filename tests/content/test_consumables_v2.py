"""The 9 new tarots, all 18 spectrals, pack immediate-use, negative consumables."""

from __future__ import annotations

from balatro_gym.engine import (
    CONSUMABLES,
    BuyPack,
    Edition,
    Enhancement,
    Game,
    Phase,
    PickPackItem,
    Rarity,
    Seal,
    SellConsumable,
    UseConsumable,
    cardops,
    new_blind_scenario,
)
from balatro_gym.engine.shop import open_shop
from tests.helpers import joker, make_run


def playing(seed="cons", jokers=()):
    return new_blind_scenario(seed, joker_ids=list(jokers))


def give(run, cid):
    return cardops.create_consumable(run, cid)


def test_all_registered():
    tarots = {c for c, d in CONSUMABLES.items() if d.kind.value == "tarot"}
    spectrals = {c for c, d in CONSUMABLES.items() if d.kind.value == "spectral"}
    assert len(tarots) == 22 and len(spectrals) == 18
    assert all(CONSUMABLES[c].base_cost == 3 for c in tarots)
    assert all(CONSUMABLES[c].base_cost == 4 for c in spectrals)


def test_fool_copies_last_used():
    g = Game(seed="fool")
    give(g.run, "c_fool")
    assert not any(isinstance(a, UseConsumable) for a in g.legal_actions())
    give(g.run, "c_mercury")
    g.apply(UseConsumable(1))
    assert g.run.last_consumable == "c_mercury"
    g.apply(UseConsumable(0))  # the Fool
    assert [c.id for c in g.run.consumables] == ["c_mercury"]
    assert g.run.last_consumable == "c_mercury"  # the Fool never records itself


def test_high_priestess_emperor_judgement():
    g = Game(seed="hp")
    give(g.run, "c_high_priestess")
    g.apply(UseConsumable(0))
    assert len(g.run.consumables) == 2 and all(c.kind.value == "planet" for c in g.run.consumables)
    g = Game(seed="emp")
    give(g.run, "c_emperor")
    g.apply(UseConsumable(0))
    assert len(g.run.consumables) == 2 and all(c.kind.value == "tarot" for c in g.run.consumables)
    g = Game(seed="jud")
    give(g.run, "c_judgement")
    g.apply(UseConsumable(0))
    assert len(g.run.jokers) == 1


def test_single_target_enhancers_and_hanged_man():
    g = playing()
    rnd = g.run.round
    for cid, enh in (("c_lovers", Enhancement.WILD), ("c_chariot", Enhancement.STEEL),
                     ("c_justice", Enhancement.GLASS), ("c_devil", Enhancement.GOLD)):
        give(g.run, cid)
        target = rnd.hand[0]
        g.apply(UseConsumable(0, (0,)))
        assert target.enhancement == enh
    give(g.run, "c_hanged_man")
    victims = [rnd.hand[0], rnd.hand[1]]
    n = len(g.run.full_deck)
    g.apply(UseConsumable(0, (0, 1)))
    assert len(g.run.full_deck) == n - 2
    assert all(c.uid not in {v.uid for v in victims} for c in rnd.hand)


def test_familiar_grim_incantation():
    for cid, n_new, pred in (
        ("c_familiar", 3, lambda c: c.rank.is_face),
        ("c_grim", 2, lambda c: int(c.rank) == 14),
        ("c_incantation", 4, lambda c: int(c.rank) <= 10),
    ):
        g = playing(seed=cid)
        rnd = g.run.round
        before = {c.uid for c in rnd.hand}
        give(g.run, cid)
        g.apply(UseConsumable(0))
        assert len(rnd.hand) == 8 - 1 + n_new
        new = [c for c in rnd.hand if c.uid not in before]
        assert len(new) == n_new
        assert all(pred(c) and c.enhancement not in (Enhancement.NONE, Enhancement.STONE) for c in new)
        assert len(g.run.full_deck) == 52 - 1 + n_new


def test_seals_aura_cryptid():
    g = playing(seed="seals")
    rnd = g.run.round
    for cid, seal in (("c_talisman", Seal.GOLD), ("c_deja_vu", Seal.RED),
                      ("c_trance", Seal.BLUE), ("c_medium", Seal.PURPLE)):
        give(g.run, cid)
        g.apply(UseConsumable(0, (0,)))
        assert rnd.hand[0].seal == seal
    give(g.run, "c_aura")
    g.apply(UseConsumable(0, (1,)))
    assert rnd.hand[1].edition in (Edition.FOIL, Edition.HOLO, Edition.POLY)
    give(g.run, "c_cryptid")
    src = rnd.hand[2]
    g.apply(UseConsumable(0, (2,)))
    copies = [c for c in rnd.hand if c.rank == src.rank and c.suit == src.suit and c.uid != src.uid]
    assert len(copies) >= 2 and len(g.run.full_deck) == 54


def test_sigil_ouija_ectoplasm():
    g = playing(seed="sig", jokers=["j_joker"])
    rnd = g.run.round
    give(g.run, "c_sigil")
    g.apply(UseConsumable(0))
    assert len({c.suit for c in rnd.hand}) == 1
    give(g.run, "c_ouija")
    g.apply(UseConsumable(0))
    assert len({c.rank for c in rnd.hand}) == 1 and g.run.hand_size == 7
    give(g.run, "c_ectoplasm")
    g.apply(UseConsumable(0))
    assert g.run.jokers[0].edition == Edition.NEGATIVE and g.run.joker_slots == 6
    assert g.run.hand_size == 6
    give(g.run, "c_ectoplasm")
    g.apply(UseConsumable(0))
    assert g.run.hand_size == 4  # second use costs 2


def test_immolate_wraith():
    g = playing(seed="imm")
    give(g.run, "c_immolate")
    money = g.run.money
    g.apply(UseConsumable(0))
    assert len(g.run.round.hand) == 3 and len(g.run.full_deck) == 47 and g.run.money == money + 20
    g = Game(seed="wraith")
    g.run.money = 30
    give(g.run, "c_wraith")
    g.apply(UseConsumable(0))
    from balatro_gym.engine import JOKERS

    assert g.run.money == 0 and JOKERS[g.run.jokers[0].id].rarity == Rarity.RARE


def test_ankh_hex_soul_black_hole():
    g = Game(seed="ankh")
    g.run.jokers = [joker("j_joker"), joker("j_jolly"), joker("j_sly")]
    give(g.run, "c_ankh")
    g.apply(UseConsumable(0))
    assert len(g.run.jokers) == 2 and g.run.jokers[0].id == g.run.jokers[1].id
    g = Game(seed="hex")
    keep = joker("j_joker")
    keep.stickers = frozenset({"eternal"})
    g.run.jokers = [keep, joker("j_jolly")]
    give(g.run, "c_hex")
    g.apply(UseConsumable(0))
    assert len(g.run.jokers) >= 1 and any(j.edition == Edition.POLY for j in g.run.jokers)
    assert any(j.uid == keep.uid for j in g.run.jokers)  # eternal survives
    g = Game(seed="soul")
    give(g.run, "c_soul")
    g.apply(UseConsumable(0))
    assert JOKERS_rarity(g.run.jokers[0].id) == Rarity.LEGENDARY
    g = Game(seed="bh")
    give(g.run, "c_black_hole")
    g.apply(UseConsumable(0))
    assert all(v == 2 for v in g.run.hand_levels.values())


def JOKERS_rarity(jid):
    from balatro_gym.engine import JOKERS

    return JOKERS[jid].rarity


def test_hidden_cards_not_in_shop_pool():
    run = make_run()
    from balatro_gym.engine.types import ConsumableKind

    assert "c_soul" not in cardops.consumable_pool(run, ConsumableKind.SPECTRAL)
    assert "c_black_hole" not in cardops.consumable_pool(run, ConsumableKind.SPECTRAL)


def test_negative_consumable_takes_no_slot_and_sells():
    g = Game(seed="neg")
    give(g.run, "c_mercury")
    give(g.run, "c_venus")
    assert not cardops.has_consumable_room(g.run)
    assert cardops.has_consumable_room(g.run, Edition.NEGATIVE)
    cardops.create_consumable(g.run, "c_mars", edition=Edition.NEGATIVE)
    assert len(g.run.consumables) == 3
    money = g.run.money
    g.apply(SellConsumable(0))
    assert g.run.money == money + 1 and len(g.run.consumables) == 2


def test_arcana_pack_deals_hand_and_uses_immediately():
    g = Game(seed="arcana")
    g.run.money = 50
    open_shop(g.run)
    from balatro_gym.engine.state import PackOffer

    g.run.shop.pack_slots = [PackOffer("p_arcana_jumbo", 6)]
    g.apply(BuyPack(0))
    pack = g.run.open_pack
    assert pack is not None and len(pack.hand) == 8 and pack.return_phase == "shop"
    assert all(any(c.uid == h.uid for c in g.run.full_deck) for h in pack.hand)
    # find a card-targeting option and use it on the dealt hand
    legal = [a for a in g.legal_actions() if isinstance(a, PickPackItem) and a.targets]
    if legal:
        act = legal[0]
        cid = pack.option_ids[act.index]
        card = pack.hand[act.targets[0]]
        g.apply(act)
        assert g.run.consumables == []  # used, not stored
        if cid in ("c_magician", "c_empress", "c_hierophant"):
            assert card.enhancement != Enhancement.NONE
    assert g.run.phase in (Phase.PACK_OPEN, Phase.SHOP)
