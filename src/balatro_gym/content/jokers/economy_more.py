"""Economy jokers: money at round end, on discard, on held cards, on sale,
and passive economy modifiers (debt, interest, free rerolls, free planets).
"""

from __future__ import annotations

from balatro_gym.engine import cardops
from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import Edition, Rank, Rarity


class _CreditCard(JokerEffect):
    """Go up to -$20 in debt."""

    def debt_allowance(self, run, j):
        return 20


register_joker(JokerDef("j_credit_card", "Credit Card", Rarity.COMMON, 1, _CreditCard(), copyable=False))


class _DelayedGratification(JokerEffect):
    """Earn $2 per discard if no discards were used by the end of the round."""

    def round_end_money(self, run, j):
        rnd = run.round
        if rnd is not None and rnd.discards_used == 0:
            return 2 * rnd.discards_left
        return 0


register_joker(
    JokerDef("j_delayed_grat", "Delayed Gratification", Rarity.COMMON, 4, _DelayedGratification(),
             copyable=False)
)


class _FacelessJoker(JokerEffect):
    """Earn $5 if 3 or more face cards are discarded at the same time."""

    def update_on_discard(self, run, j, cards):
        from balatro_gym.engine.hand_eval import is_face
        from balatro_gym.engine.scoring import fold_eval_rules

        rules = fold_eval_rules(run)
        if sum(1 for c in cards if is_face(c, rules)) >= 3:
            run.money += 5


register_joker(JokerDef("j_faceless", "Faceless Joker", Rarity.COMMON, 4, _FacelessJoker()))


class _Cloud9(JokerEffect):
    """Earn $1 per 9 in the full deck at end of round."""

    def round_end_money(self, run, j):
        return sum(1 for c in run.full_deck if not c.is_stone and c.rank == Rank.NINE)


register_joker(JokerDef("j_cloud_9", "Cloud 9", Rarity.UNCOMMON, 7, _Cloud9(), copyable=False))


class _Rocket(JokerEffect):
    """Earn $1 at end of round; payout rises by $2 every Boss Blind defeated."""

    def round_end_money(self, run, j):
        return j.state["dollars"]

    def update_on_boss_defeated(self, run, j):
        j.state["dollars"] += 2


register_joker(
    JokerDef("j_rocket", "Rocket", Rarity.UNCOMMON, 6, _Rocket(), init_state={"dollars": 1},
             copyable=False, perishable_compat=False)
)


class _GiftCard(JokerEffect):
    """Add $1 of sell value to every Joker and Consumable at end of round."""

    def update_on_round_end(self, run, j):
        for x in run.jokers:
            x.sell_value += 1
        for c in run.consumables:
            c.sell_value += 1


register_joker(JokerDef("j_gift", "Gift Card", Rarity.UNCOMMON, 6, _GiftCard(), copyable=False))


class _ReservedParking(JokerEffect):
    """Each face card held in hand has a 1 in 2 chance to give $1."""

    def on_card_held(self, ctx, j, card):
        if ctx.is_face(card) and ctx.chance(1, 2):
            ctx.add_money(1, "joker", "j_reserved_parking")


register_joker(JokerDef("j_reserved_parking", "Reserved Parking", Rarity.COMMON, 6, _ReservedParking()))


class _ToTheMoon(JokerEffect):
    """Earn an extra $1 of interest per $5 held at end of round."""

    def interest_bonus(self, run, j):
        return 1


register_joker(JokerDef("j_to_the_moon", "To the Moon", Rarity.UNCOMMON, 5, _ToTheMoon(), copyable=False))


class _Satellite(JokerEffect):
    """Earn $1 at end of round per unique Planet card used this run."""

    def round_end_money(self, run, j):
        return len(run.stats.get("planets_used_ids", []))


register_joker(JokerDef("j_satellite", "Satellite", Rarity.UNCOMMON, 6, _Satellite(), copyable=False))


class _TradingCard(JokerEffect):
    """If the first discard of the round is a single card, destroy it and earn $3."""

    def update_on_discard(self, run, j, cards):
        rnd = run.round
        if rnd is None or rnd.discards_used != 0 or len(cards) != 1:
            return
        cardops.destroy_cards(run, [cards[0]])
        run.money += 3


register_joker(JokerDef("j_trading", "Trading Card", Rarity.UNCOMMON, 6, _TradingCard(), copyable=False))


class _DietCola(JokerEffect):
    """Sell this card to create a free Double Tag."""

    def on_sold(self, run, j):
        from balatro_gym.engine.effects import TAGS
        from balatro_gym.engine.round import acquire_tag
        from balatro_gym.engine.state import TagInst

        if "t_double" in TAGS:
            acquire_tag(run, TagInst(id="t_double"))


register_joker(JokerDef("j_diet_cola", "Diet Cola", Rarity.UNCOMMON, 6, _DietCola(), eternal_compat=False))


class _Luchador(JokerEffect):
    """Sell this card to disable the current Boss Blind."""

    def on_sold(self, run, j):
        rnd = run.round
        if rnd is not None and rnd.boss_id is not None:
            rnd.boss_state["disabled"] = True


register_joker(JokerDef("j_luchador", "Luchador", Rarity.UNCOMMON, 5, _Luchador(), eternal_compat=False))


class _ChaosTheClown(JokerEffect):
    """1 free Reroll per shop."""

    def free_rerolls(self, run, j):
        return 1


register_joker(JokerDef("j_chaos", "Chaos the Clown", Rarity.COMMON, 4, _ChaosTheClown(), copyable=False))


class _Astronomer(JokerEffect):
    """All Planet cards and Celestial Packs in the shop are free."""

    def item_is_free(self, run, j, kind, item_id):
        return kind == "planet" or (kind == "pack" and "celestial" in item_id)


register_joker(JokerDef("j_astronomer", "Astronomer", Rarity.UNCOMMON, 8, _Astronomer(), copyable=False))


class _Perkeo(JokerEffect):
    """Creates a Negative copy of a random consumable when leaving the shop."""

    def update_on_shop_leave(self, run, j):
        if not run.consumables:
            return
        src = run.rng.stream("consumable").choice(run.consumables)
        cardops.create_consumable(run, src.id, edition=Edition.NEGATIVE)


register_joker(JokerDef("j_perkeo", "Perkeo", Rarity.LEGENDARY, 20, _Perkeo()))
