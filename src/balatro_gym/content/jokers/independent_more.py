"""Independent (stage 5) jokers: board/deck/money-conditioned effects and
the xMult scalers whose growth comes from events outside scoring.
"""

from __future__ import annotations

from itertools import permutations

from balatro_gym.engine import cardops
from balatro_gym.engine.effects import JOKERS, JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import ConsumableKind, Enhancement, HandType, Rank, Rarity, Suit


class _JokerStencil(JokerEffect):
    """x1 Mult per empty joker slot (Stencil counts itself as empty)."""

    def on_independent(self, ctx, j):
        run = ctx.run
        empty = run.joker_slots - len(run.jokers) + sum(1 for x in run.jokers if x.id == "j_stencil")
        if empty > 1:
            ctx.mul_mult(float(empty), "joker", "j_stencil")


register_joker(JokerDef("j_stencil", "Joker Stencil", Rarity.UNCOMMON, 8, _JokerStencil()))


class _MysticSummit(JokerEffect):
    """+15 Mult when 0 discards remain."""

    def on_independent(self, ctx, j):
        rnd = ctx.run.round
        if rnd is not None and rnd.discards_left == 0:
            ctx.add_mult(15, "joker", "j_mystic_summit")


register_joker(JokerDef("j_mystic_summit", "Mystic Summit", Rarity.COMMON, 5, _MysticSummit()))


class _Misprint(JokerEffect):
    """+0 to +23 Mult, random each hand."""

    def on_independent(self, ctx, j):
        ctx.add_mult(ctx.run.rng.stream("chance").randint(0, 23), "joker", "j_misprint")


register_joker(JokerDef("j_misprint", "Misprint", Rarity.COMMON, 4, _Misprint()))


class _LoyaltyCard(JokerEffect):
    """x4 Mult every 6th hand played. state["remaining"] counts down the
    hands still to play before the bonus (shown as "N remaining")."""

    def on_independent(self, ctx, j):
        if j.state["remaining"] == 0:
            ctx.mul_mult(4.0, "joker", "j_loyalty_card")

    def update_after_scoring(self, run, j, result):
        if j.state["remaining"] == 0:
            j.state["remaining"] = 5
        else:
            j.state["remaining"] -= 1


register_joker(
    JokerDef("j_loyalty_card", "Loyalty Card", Rarity.UNCOMMON, 5, _LoyaltyCard(),
             init_state={"remaining": 5})
)


class _SteelJoker(JokerEffect):
    """x0.2 Mult per Steel card in the full deck (x1 base)."""

    def on_independent(self, ctx, j):
        n = sum(1 for c in ctx.run.full_deck if c.enhancement == Enhancement.STEEL)
        if n:
            ctx.mul_mult(1 + 0.2 * n, "joker", "j_steel_joker")


register_joker(JokerDef("j_steel_joker", "Steel Joker", Rarity.UNCOMMON, 7, _SteelJoker()))


class _StoneJoker(JokerEffect):
    """+25 Chips per Stone card in the full deck."""

    def on_independent(self, ctx, j):
        n = sum(1 for c in ctx.run.full_deck if c.is_stone)
        if n:
            ctx.add_chips(25 * n, "joker", "j_stone")


register_joker(JokerDef("j_stone", "Stone Joker", Rarity.UNCOMMON, 6, _StoneJoker()))


class _BlueJoker(JokerEffect):
    """+2 Chips per card remaining in the draw pile."""

    def on_independent(self, ctx, j):
        rnd = ctx.run.round
        n = len(rnd.deck_remaining) if rnd is not None else len(ctx.run.full_deck)
        if n:
            ctx.add_chips(2 * n, "joker", "j_blue_joker")


register_joker(JokerDef("j_blue_joker", "Blue Joker", Rarity.COMMON, 5, _BlueJoker()))


class _Erosion(JokerEffect):
    """+4 Mult per card fewer than the deck's starting size."""

    def on_independent(self, ctx, j):
        start = ctx.run.stats.get("starting_deck_size", 52)
        n = max(0, start - len(ctx.run.full_deck))
        if n:
            ctx.add_mult(4 * n, "joker", "j_erosion")


register_joker(JokerDef("j_erosion", "Erosion", Rarity.UNCOMMON, 6, _Erosion()))


class _Bull(JokerEffect):
    """+2 Chips per $1 you have."""

    def on_independent(self, ctx, j):
        m = max(0, ctx.run.money)
        if m:
            ctx.add_chips(2 * m, "joker", "j_bull")


register_joker(JokerDef("j_bull", "Bull", Rarity.UNCOMMON, 6, _Bull()))


class _Bootstraps(JokerEffect):
    """+2 Mult per $5 you have."""

    def on_independent(self, ctx, j):
        n = max(0, ctx.run.money) // 5
        if n:
            ctx.add_mult(2 * n, "joker", "j_bootstraps")


register_joker(JokerDef("j_bootstraps", "Bootstraps", Rarity.UNCOMMON, 7, _Bootstraps()))


class _Swashbuckler(JokerEffect):
    """+Mult equal to the total sell value of all OTHER jokers."""

    def on_independent(self, ctx, j):
        total = sum(x.sell_value for x in ctx.run.jokers if x.uid != j.uid)
        if total:
            ctx.add_mult(total, "joker", "j_swashbuckler")


register_joker(JokerDef("j_swashbuckler", "Swashbuckler", Rarity.COMMON, 4, _Swashbuckler()))


class _DriversLicense(JokerEffect):
    """x3 Mult with 16 or more enhanced cards in the full deck."""

    def on_independent(self, ctx, j):
        n = sum(1 for c in ctx.run.full_deck if c.enhancement != Enhancement.NONE)
        if n >= 16:
            ctx.mul_mult(3.0, "joker", "j_drivers_license")


register_joker(JokerDef("j_drivers_license", "Driver's License", Rarity.RARE, 7, _DriversLicense()))


class _Acrobat(JokerEffect):
    """x3 Mult on the final hand of the round."""

    def on_independent(self, ctx, j):
        if ctx.is_final_hand():
            ctx.mul_mult(3.0, "joker", "j_acrobat")


register_joker(JokerDef("j_acrobat", "Acrobat", Rarity.UNCOMMON, 6, _Acrobat()))


class _Blackboard(JokerEffect):
    """x3 Mult if every card held in hand is a Spade or a Club (none held counts)."""

    def on_independent(self, ctx, j):
        ok = all(
            ctx.card_has_suit(c, Suit.SPADES) or ctx.card_has_suit(c, Suit.CLUBS)
            for c in ctx.held
        )
        if ok:
            ctx.mul_mult(3.0, "joker", "j_blackboard")


register_joker(JokerDef("j_blackboard", "Blackboard", Rarity.UNCOMMON, 6, _Blackboard()))


def _covers_all_suits(ctx, cards) -> bool:
    """Can the cards be assigned one suit each so all four suits appear?"""
    cards = [c for c in cards if not c.is_stone]
    if len(cards) < 4:
        return False
    options = [{s for s in Suit if ctx.card_has_suit(c, s)} for c in cards]
    for combo in permutations(range(len(cards)), 4):
        if all(list(Suit)[k] in options[combo[k]] for k in range(4)):
            return True
    return False


class _FlowerPot(JokerEffect):
    """x3 Mult if the scoring cards contain all four suits."""

    def on_independent(self, ctx, j):
        if _covers_all_suits(ctx, ctx.hand_result.scoring_cards):
            ctx.mul_mult(3.0, "joker", "j_flower_pot")


register_joker(JokerDef("j_flower_pot", "Flower Pot", Rarity.UNCOMMON, 6, _FlowerPot()))


class _SeeingDouble(JokerEffect):
    """x2 Mult if the scoring cards contain a Club and a card of another suit."""

    def on_independent(self, ctx, j):
        cards = [c for c in ctx.hand_result.scoring_cards if not c.is_stone]
        for club in cards:
            if not ctx.card_has_suit(club, Suit.CLUBS):
                continue
            for other in cards:
                if other.uid == club.uid:
                    continue
                if any(ctx.card_has_suit(other, s) for s in Suit if s != Suit.CLUBS):
                    ctx.mul_mult(2.0, "joker", "j_seeing_double")
                    return


register_joker(JokerDef("j_seeing_double", "Seeing Double", Rarity.UNCOMMON, 6, _SeeingDouble()))


class _Cavendish(JokerEffect):
    """x3 Mult; 1 in 1000 chance to be destroyed at end of round.
    Only appears once Gros Michel has gone extinct."""

    def on_independent(self, ctx, j):
        ctx.mul_mult(3.0, "joker", "j_cavendish")

    def update_on_round_end(self, run, j):
        if cardops.chance(run, 1, 1000):
            cardops.destroy_joker(run, j)


register_joker(
    JokerDef("j_cavendish", "Cavendish", Rarity.COMMON, 4, _Cavendish(),
             eternal_compat=False, pool_requires="gros_michel_extinct")
)


class _Stuntman(JokerEffect):
    """+250 Chips, -2 hand size."""

    def on_independent(self, ctx, j):
        ctx.add_chips(250, "joker", "j_stuntman")

    def hand_size_mod(self, run, j):
        return -2


register_joker(JokerDef("j_stuntman", "Stuntman", Rarity.RARE, 7, _Stuntman()))


class _ToDoList(JokerEffect):
    """$4 if the played hand is [type]; the type changes each round."""

    def _reroll(self, run, j):
        pool = sorted(int(h) for h in run.hands_discovered if h != HandType(j.state["hand"]))
        if pool:
            j.state["hand"] = run.rng.stream("misc").choice(pool)

    def on_independent(self, ctx, j):
        if int(ctx.hand_result.hand_type) == j.state["hand"]:
            ctx.add_money(4, "joker", "j_todo_list")

    def update_on_round_end(self, run, j):
        self._reroll(run, j)


register_joker(
    JokerDef("j_todo_list", "To Do List", Rarity.COMMON, 4, _ToDoList(),
             init_state={"hand": int(HandType.HIGH_CARD)})
)


class _Matador(JokerEffect):
    """$8 if the played hand triggers the Boss Blind's ability."""

    def on_independent(self, ctx, j):
        boss = ctx.boss
        if boss is not None and (ctx.boss_triggered or boss.triggers_every_hand):
            ctx.add_money(8, "joker", "j_matador")


register_joker(JokerDef("j_matador", "Matador", Rarity.UNCOMMON, 7, _Matador()))


# ---------------------------------------------------------------------------
# xMult scalers driven by run events
# ---------------------------------------------------------------------------

class _XScaler(JokerEffect):
    """Base: multiply by state["x"] at stage 5."""

    jid = ""

    def on_independent(self, ctx, j):
        ctx.mul_mult(j.state["x"], "joker", self.jid)


class _Constellation(_XScaler):
    """Gains x0.1 Mult per Planet card used."""

    jid = "j_constellation"

    def update_on_consumable_used(self, run, j, inst):
        if inst.kind == ConsumableKind.PLANET:
            j.state["x"] = round(j.state["x"] + 0.1, 6)


register_joker(
    JokerDef("j_constellation", "Constellation", Rarity.UNCOMMON, 6, _Constellation(),
             init_state={"x": 1.0}, perishable_compat=False)
)


class _Hologram(_XScaler):
    """Gains x0.25 Mult per playing card added to the deck."""

    jid = "j_hologram"

    def update_on_card_added(self, run, j, card):
        j.state["x"] = round(j.state["x"] + 0.25, 6)


register_joker(
    JokerDef("j_hologram", "Hologram", Rarity.UNCOMMON, 7, _Hologram(),
             init_state={"x": 1.0}, perishable_compat=False)
)


class _Throwback(JokerEffect):
    """x0.25 Mult per blind skipped this run."""

    def on_independent(self, ctx, j):
        n = ctx.run.stats.get("blinds_skipped", 0)
        if n:
            ctx.mul_mult(1 + 0.25 * n, "joker", "j_throwback")


register_joker(JokerDef("j_throwback", "Throwback", Rarity.UNCOMMON, 6, _Throwback()))


class _Campfire(_XScaler):
    """Gains x0.25 Mult per card sold; resets when a Boss Blind is defeated."""

    jid = "j_campfire"

    def update_on_other_sold(self, run, j, sold_uid):
        j.state["x"] = round(j.state["x"] + 0.25, 6)

    def update_on_boss_defeated(self, run, j):
        j.state["x"] = 1.0


register_joker(
    JokerDef("j_campfire", "Campfire", Rarity.RARE, 9, _Campfire(), init_state={"x": 1.0})
)


class _GlassJoker(_XScaler):
    """Gains x0.75 Mult per Glass card destroyed."""

    jid = "j_glass"

    def update_on_card_destroyed(self, run, j, card):
        if card.enhancement == Enhancement.GLASS:
            j.state["x"] = round(j.state["x"] + 0.75, 6)


register_joker(
    JokerDef("j_glass", "Glass Joker", Rarity.UNCOMMON, 6, _GlassJoker(),
             init_state={"x": 1.0}, perishable_compat=False)
)


class _HitTheRoad(_XScaler):
    """Gains x0.5 Mult per Jack discarded this round; resets at round end."""

    jid = "j_hit_the_road"

    def update_on_discard(self, run, j, cards):
        n = sum(1 for c in cards if not c.is_stone and c.rank == Rank.JACK)
        if n:
            j.state["x"] = round(j.state["x"] + 0.5 * n, 6)

    def update_on_round_end(self, run, j):
        j.state["x"] = 1.0


register_joker(
    JokerDef("j_hit_the_road", "Hit the Road", Rarity.RARE, 8, _HitTheRoad(), init_state={"x": 1.0})
)


class _Madness(_XScaler):
    """When a Small or Big Blind is selected, gain x0.5 Mult and destroy a
    random other joker."""

    jid = "j_madness"

    def on_blind_selected(self, run, j):
        rnd = run.round
        if rnd is None or rnd.boss_id is not None:
            return
        j.state["x"] = round(j.state["x"] + 0.5, 6)
        others = [x for x in run.jokers if x.uid != j.uid and "eternal" not in x.stickers]
        if others:
            cardops.destroy_joker(run, run.rng.stream("misc").choice(others))


register_joker(
    JokerDef("j_madness", "Madness", Rarity.UNCOMMON, 7, _Madness(),
             init_state={"x": 1.0}, perishable_compat=False)
)


class _CeremonialDagger(JokerEffect):
    """When a blind is selected, destroy the joker to the right and gain
    double its sell value as Mult."""

    def on_independent(self, ctx, j):
        ctx.add_mult(j.state["mult"], "joker", "j_ceremonial")

    def on_blind_selected(self, run, j):
        idx = next((i for i, x in enumerate(run.jokers) if x.uid == j.uid), None)
        if idx is None or idx + 1 >= len(run.jokers):
            return
        victim = run.jokers[idx + 1]
        if cardops.destroy_joker(run, victim):
            j.state["mult"] += 2 * victim.sell_value


register_joker(
    JokerDef("j_ceremonial", "Ceremonial Dagger", Rarity.UNCOMMON, 6, _CeremonialDagger(),
             init_state={"mult": 0}, perishable_compat=False)
)


class _FlashCard(JokerEffect):
    """Gains +2 Mult per shop reroll."""

    def on_independent(self, ctx, j):
        ctx.add_mult(j.state["mult"], "joker", "j_flash")

    def update_on_reroll(self, run, j):
        j.state["mult"] += 2


register_joker(
    JokerDef("j_flash", "Flash Card", Rarity.UNCOMMON, 5, _FlashCard(),
             init_state={"mult": 0}, perishable_compat=False)
)


class _RedCard(JokerEffect):
    """Gains +3 Mult per Booster Pack skipped."""

    def on_independent(self, ctx, j):
        ctx.add_mult(j.state["mult"], "joker", "j_red_card")

    def update_on_pack_skipped(self, run, j):
        j.state["mult"] += 3


register_joker(
    JokerDef("j_red_card", "Red Card", Rarity.COMMON, 5, _RedCard(),
             init_state={"mult": 0}, perishable_compat=False)
)


class _Ramen(_XScaler):
    """x2 Mult, loses x0.01 per card discarded; eaten when it reaches x1."""

    jid = "j_ramen"

    def update_on_discard(self, run, j, cards):
        j.state["x"] = round(j.state["x"] - 0.01 * len(cards), 6)
        if j.state["x"] <= 1.0:
            run.pending_destroy.append(j.uid)


register_joker(
    JokerDef("j_ramen", "Ramen", Rarity.UNCOMMON, 6, _Ramen(), init_state={"x": 2.0},
             eternal_compat=False)
)


# ---------------------------------------------------------------------------
# On-played consumable creators
# ---------------------------------------------------------------------------

class _Vagabond(JokerEffect):
    """Create a Tarot card if the hand is played with $4 or less."""

    def on_hand_played(self, ctx, j):
        if ctx.run.money <= 4:
            if cardops.create_random_consumable(ctx.run, ConsumableKind.TAROT):
                ctx.note("joker", "j_vagabond")


register_joker(JokerDef("j_vagabond", "Vagabond", Rarity.RARE, 8, _Vagabond()))


class _Seance(JokerEffect):
    """If the played hand is a Straight Flush, create a random Spectral card."""

    def on_hand_played(self, ctx, j):
        if ctx.hand_result.hand_type == HandType.STRAIGHT_FLUSH:
            if cardops.create_random_consumable(ctx.run, ConsumableKind.SPECTRAL):
                ctx.note("joker", "j_seance")


register_joker(JokerDef("j_seance", "Seance", Rarity.UNCOMMON, 6, _Seance()))


class _Superposition(JokerEffect):
    """Create a Tarot card if the hand contains an Ace and a Straight."""

    def on_hand_played(self, ctx, j):
        if HandType.STRAIGHT in ctx.hand_result.contained and any(
            not c.is_stone and c.rank == Rank.ACE for c in ctx.hand_result.scoring_cards
        ):
            if cardops.create_random_consumable(ctx.run, ConsumableKind.TAROT):
                ctx.note("joker", "j_superposition")


register_joker(JokerDef("j_superposition", "Superposition", Rarity.COMMON, 4, _Superposition()))


class _SixthSense(JokerEffect):
    """If the first hand of the round is a single 6, destroy it and create a
    Spectral card (if there is room)."""

    def update_after_scoring(self, run, j, result):
        rnd = run.round
        if rnd is None or rnd.hands_played != 0:
            return
        played = result.hand_result.scoring_cards
        if len(played) != 1 or played[0].is_stone or played[0].rank != Rank.SIX:
            return
        if not cardops.has_consumable_room(run):
            return
        cardops.destroy_cards(run, [played[0]])
        cardops.create_random_consumable(run, ConsumableKind.SPECTRAL)


register_joker(JokerDef("j_sixth_sense", "Sixth Sense", Rarity.UNCOMMON, 6, _SixthSense(), copyable=False))


class _BurntJoker(JokerEffect):
    """Upgrade the level of the first discarded poker hand each round."""

    def update_on_discard(self, run, j, cards):
        rnd = run.round
        if rnd is None or rnd.discards_used != 0:
            return
        from balatro_gym.engine.hand_eval import evaluate_hand
        from balatro_gym.engine.scoring import fold_eval_rules

        ht = evaluate_hand(cards, fold_eval_rules(run)).hand_type
        run.hand_levels[ht] += 1


register_joker(JokerDef("j_burnt", "Burnt Joker", Rarity.RARE, 8, _BurntJoker()))


class _Hallucination(JokerEffect):
    """1 in 2 chance to create a Tarot card whenever a Booster Pack is opened."""

    def update_on_pack_opened(self, run, j):
        if cardops.chance(run, 1, 2):
            cardops.create_random_consumable(run, ConsumableKind.TAROT)


register_joker(JokerDef("j_hallucination", "Hallucination", Rarity.COMMON, 4, _Hallucination()))


def _uncommon_count(run) -> int:
    return sum(1 for j in run.jokers if JOKERS[j.id].rarity == Rarity.UNCOMMON)
