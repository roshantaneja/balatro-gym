"""Shop: generation, buying, rerolls, packs, vouchers, tags, RNG stream isolation."""

import tests.helpers  # noqa: F401  (loads content)
from balatro_gym.engine import (
    BlindKind,
    BuyPack,
    BuyShopCard,
    BuyVoucher,
    Game,
    HandType,
    LeaveShop,
    Phase,
    PickPackItem,
    PlayHand,
    Reroll,
    SelectBlind,
    SellJoker,
    SkipBlind,
    new_blind_scenario,
)

LEVELED = {HandType.PAIR: 30, HandType.HIGH_CARD: 30}


def clear_first_blind(seed="shop-test", money=20) -> Game:
    """A game that has just cleared the ante-1 small blind and sits in the shop."""
    g = new_blind_scenario(seed, hand_levels=LEVELED, money=money)
    g.apply(PlayHand((0, 1)))
    assert g.run.phase == Phase.SHOP
    return g


def test_shop_layout():
    g = clear_first_blind()
    shop = g.run.shop
    assert shop is not None
    assert len(shop.card_slots) == 2
    assert len(shop.pack_slots) == 2
    assert shop.voucher is not None
    assert shop.reroll_cost == 5


def test_first_shop_has_buffoon_pack():
    g = clear_first_blind()
    cats = [p.pack_id for p in g.run.shop.pack_slots]
    assert any("buffoon" in pid for pid in cats)


def test_buy_joker():
    g = clear_first_blind(money=50)
    shop = g.run.shop
    joker_slots = [i for i, s in enumerate(shop.card_slots) if s.kind == "joker"]
    if not joker_slots:  # reroll until a joker shows (bounded)
        for _ in range(10):
            g.apply(Reroll())
            joker_slots = [i for i, s in enumerate(g.run.shop.card_slots) if s.kind == "joker"]
            if joker_slots:
                break
    assert joker_slots, "no joker appeared in 10 rerolls"
    slot = joker_slots[0]
    item = g.run.shop.card_slots[slot]
    money_before = g.run.money
    g.apply(BuyShopCard(slot))
    assert g.run.money == money_before - item.cost
    assert g.run.jokers[-1].id == item.item_id
    assert g.run.jokers[-1].sell_value == max(1, item.cost // 2)


def test_reroll_cost_escalates_and_resets():
    g = clear_first_blind(money=100)
    assert g.run.shop.reroll_cost == 5
    g.apply(Reroll())
    assert g.run.shop.reroll_cost == 6
    g.apply(Reroll())
    assert g.run.shop.reroll_cost == 7
    # Leave, clear next blind, shop resets to $5.
    g.apply(LeaveShop())
    g.apply(SelectBlind())  # big blind
    g.apply(PlayHand((0, 1)))
    assert g.run.phase == Phase.SHOP
    assert g.run.shop.reroll_cost == 5


def test_reroll_does_not_touch_packs_or_voucher():
    g = clear_first_blind(money=100)
    packs_before = [p.pack_id for p in g.run.shop.pack_slots]
    voucher_before = g.run.shop.voucher
    g.apply(Reroll())
    assert [p.pack_id for p in g.run.shop.pack_slots] == packs_before
    assert g.run.shop.voucher == voucher_before


def test_buy_pack_and_pick():
    g = clear_first_blind(money=50)
    pack_offer = g.run.shop.pack_slots[0]
    g.apply(BuyPack(0))
    assert g.run.phase == Phase.PACK_OPEN
    pack = g.run.open_pack
    assert pack is not None
    assert pack.pack_id == pack_offer.pack_id
    n_options = len(pack.option_kinds)
    picks = pack.picks_left
    assert n_options > 0 and picks >= 1

    legal = g.legal_actions()
    pick_acts = [a for a in legal if isinstance(a, PickPackItem)]
    assert pick_acts
    jokers_before = len(g.run.jokers)
    deck_before = len(g.run.full_deck)
    cons_before = len(g.run.consumables)
    kind = pack.option_kinds[pick_acts[0].index]
    events = g.apply(pick_acts[0])
    gained = (
        len(g.run.jokers) - jokers_before
        + len(g.run.full_deck) - deck_before
        + len(g.run.consumables) - cons_before
    )
    if kind in ("joker", "card"):
        assert gained == 1
    else:
        # Pack consumables are used immediately, as in the game.
        from balatro_gym.engine import ConsumableUsed

        assert any(isinstance(e, ConsumableUsed) for e in events)


def test_buy_voucher_applies():
    g = clear_first_blind(money=50)
    vid = g.run.shop.voucher
    hands_before = g.run.hands_per_round
    discards_before = g.run.discards_per_round
    cons_before = g.run.consumable_slots
    g.apply(BuyVoucher())
    assert vid in g.run.vouchers
    assert g.run.shop.voucher is None
    if vid == "v_grabber":
        assert g.run.hands_per_round == hands_before + 1
    elif vid == "v_wasteful":
        assert g.run.discards_per_round == discards_before + 1
    elif vid == "v_crystal_ball":
        assert g.run.consumable_slots == cons_before + 1
    elif vid == "v_overstock":
        assert g.run.stats["shop_card_slots"] == 3


def test_sell_joker():
    g = clear_first_blind(money=50)
    from tests.helpers import joker

    g.run.jokers.append(joker("j_joker"))
    g.run.jokers[-1].sell_value = 2
    money_before = g.run.money
    g.apply(SellJoker(len(g.run.jokers) - 1))
    assert g.run.money == money_before + 2
    assert not any(j.id == "j_joker" for j in g.run.jokers)


def test_skip_blind_grants_tag():
    g = Game(seed="tag-test")
    events = g.apply(SkipBlind())
    from balatro_gym.engine import BlindSkipped

    ev = next(e for e in events if isinstance(e, BlindSkipped))
    assert ev.tag_id is not None
    # Handy/Garbage self-consume instantly; Investment/Juggle persist.
    if ev.tag_id in ("t_investment", "t_juggle"):
        assert ev.tag_id in g.run.tags
    else:
        assert ev.tag_id not in g.run.tags


def test_shop_stream_isolated_from_chance():
    """Consuming chance rolls mid-round must not perturb the next shop."""

    def shop_contents(burn_chance: int) -> list[str]:
        g = new_blind_scenario("stream-iso", hand_levels=LEVELED, money=20)
        for _ in range(burn_chance):
            g.run.rng.stream("chance").random()
        g.apply(PlayHand((0, 1)))
        shop = g.run.shop
        return [s.item_id for s in shop.card_slots] + [p.pack_id for p in shop.pack_slots]

    assert shop_contents(0) == shop_contents(17)


def test_boss_blind_reward_paid():
    g = new_blind_scenario("boss-pay", blind_kind=BlindKind.BOSS, hand_levels=LEVELED)
    events = g.apply(PlayHand((0, 1)))
    from balatro_gym.engine import RoundPayout

    payout = next(e for e in events if isinstance(e, RoundPayout))
    assert payout.blind_reward == 5
    assert g.run.ante == 2  # boss cleared -> next ante
    assert g.run.blind_index == 0
