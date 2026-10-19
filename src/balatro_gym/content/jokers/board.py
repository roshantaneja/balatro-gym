"""Board / lifecycle jokers: hand-size and hand/discard modifiers, blind-start
creators, probability and pool modifiers, loss prevention, sell effects.

Hand size, hands and discards are POLLED (hand_size_mod / hands_mod /
discards_mod) so the numbers always match the current board — no
acquire/sell bookkeeping to get out of sync.
"""

from __future__ import annotations

from dataclasses import replace

from balatro_gym.engine import cardops
from balatro_gym.engine.cards import Card
from balatro_gym.engine.effects import JokerDef, JokerEffect, register_joker
from balatro_gym.engine.types import ConsumableKind, Edition, Enhancement, Rank, Rarity, Seal, Suit


class _Burglar(JokerEffect):
    """When a blind is selected: +3 hands, lose all discards."""

    def on_blind_selected(self, run, j):
        rnd = run.round
        if rnd is not None:
            rnd.hands_left += 3
            rnd.discards_left = 0


register_joker(JokerDef("j_burglar", "Burglar", Rarity.UNCOMMON, 6, _Burglar()))


class _Juggler(JokerEffect):
    """+1 hand size."""

    def hand_size_mod(self, run, j):
        return 1


register_joker(JokerDef("j_juggler", "Juggler", Rarity.COMMON, 4, _Juggler(), copyable=False))


class _Drunkard(JokerEffect):
    """+1 discard per round."""

    def discards_mod(self, run, j):
        return 1


register_joker(JokerDef("j_drunkard", "Drunkard", Rarity.COMMON, 4, _Drunkard(), copyable=False))


class _Troubadour(JokerEffect):
    """+2 hand size, -1 hand per round."""

    def hand_size_mod(self, run, j):
        return 2

    def hands_mod(self, run, j):
        return -1


register_joker(JokerDef("j_troubadour", "Troubadour", Rarity.UNCOMMON, 6, _Troubadour(), copyable=False))


class _MerryAndy(JokerEffect):
    """+3 discards per round, -1 hand size."""

    def discards_mod(self, run, j):
        return 3

    def hand_size_mod(self, run, j):
        return -1


register_joker(JokerDef("j_merry_andy", "Merry Andy", Rarity.UNCOMMON, 7, _MerryAndy(), copyable=False))


class _TurtleBean(JokerEffect):
    """+5 hand size, reduced by 1 every round; eaten at 0."""

    def hand_size_mod(self, run, j):
        return j.state["h"]

    def update_on_round_end(self, run, j):
        j.state["h"] -= 1
        if j.state["h"] <= 0:
            cardops.destroy_joker(run, j)


register_joker(
    JokerDef("j_turtle_bean", "Turtle Bean", Rarity.UNCOMMON, 6, _TurtleBean(),
             init_state={"h": 5}, copyable=False, eternal_compat=False)
)


def _add_to_draw_pile(run, card: Card) -> None:
    """Add a new card to the deck and into the current round's draw pile."""
    cardops.add_card(run, card, to_hand=False)
    rnd = run.round
    if rnd is not None:
        pos = run.rng.stream("misc").randint(0, len(rnd.deck_remaining))
        rnd.deck_remaining.insert(pos, card)


class _MarbleJoker(JokerEffect):
    """Adds a Stone card to the deck when a blind is selected."""

    def on_blind_selected(self, run, j):
        card = Card(uid=run.new_uid(), rank=Rank.ACE, suit=Suit.SPADES, enhancement=Enhancement.STONE)
        _add_to_draw_pile(run, card)


register_joker(JokerDef("j_marble", "Marble Joker", Rarity.UNCOMMON, 6, _MarbleJoker()))


class _Certificate(JokerEffect):
    """When a blind is selected, add a random playing card with a random seal."""

    def on_blind_selected(self, run, j):
        from balatro_gym.engine.shop import roll_playing_card

        card = roll_playing_card(run, modifiers=False, stream="misc")
        card.seal = run.rng.stream("misc").choice([Seal.RED, Seal.BLUE, Seal.GOLD, Seal.PURPLE])
        _add_to_draw_pile(run, card)


register_joker(JokerDef("j_certificate", "Certificate", Rarity.UNCOMMON, 6, _Certificate()))


class _RiffRaff(JokerEffect):
    """When a blind is selected, create 2 Common jokers (if there is room)."""

    def on_blind_selected(self, run, j):
        for _ in range(2):
            if not cardops.has_joker_room(run):
                break
            jid = cardops.random_joker_id(run, Rarity.COMMON)
            if jid is None:
                break
            cardops.create_joker(run, jid)


register_joker(JokerDef("j_riff_raff", "Riff-Raff", Rarity.COMMON, 6, _RiffRaff()))


class _Cartomancer(JokerEffect):
    """Create a Tarot card when a blind is selected (if there is room)."""

    def on_blind_selected(self, run, j):
        cardops.create_random_consumable(run, ConsumableKind.TAROT)


register_joker(JokerDef("j_cartomancer", "Cartomancer", Rarity.UNCOMMON, 6, _Cartomancer()))


class _Showman(JokerEffect):
    """Joker, Tarot, Planet and Spectral cards may appear multiple times."""

    def allows_duplicates(self, run, j):
        return True


register_joker(JokerDef("j_showman", "Showman", Rarity.UNCOMMON, 5, _Showman(), copyable=False))


class _Oops(JokerEffect):
    """Doubles all listed probabilities."""

    def chance_multiplier(self, run, j):
        return 2


register_joker(JokerDef("j_oops", "Oops! All 6s", Rarity.UNCOMMON, 4, _Oops(), copyable=False))


class _Shortcut(JokerEffect):
    """Straights can be made with gaps of 1 rank."""

    def modify_eval_rules(self, rules, j):
        return replace(rules, straight_max_gap=2)


register_joker(JokerDef("j_shortcut", "Shortcut", Rarity.UNCOMMON, 7, _Shortcut(), copyable=False))


class _Smeared(JokerEffect):
    """Hearts and Diamonds count as one suit; Spades and Clubs count as one."""

    def modify_eval_rules(self, rules, j):
        return replace(rules, smeared_suits=True)


register_joker(JokerDef("j_smeared", "Smeared Joker", Rarity.UNCOMMON, 7, _Smeared(), copyable=False))


class _MrBones(JokerEffect):
    """Prevents death if chips scored are at least 25% of required; self-destructs."""

    def prevent_loss(self, run, j):
        rnd = run.round
        return rnd is not None and rnd.chips_scored >= 0.25 * rnd.chips_needed


register_joker(
    JokerDef("j_mr_bones", "Mr. Bones", Rarity.UNCOMMON, 5, _MrBones(), copyable=False,
             eternal_compat=False)
)


class _InvisibleJoker(JokerEffect):
    """After 2 full rounds, sell this card to duplicate a random other joker."""

    def update_on_round_end(self, run, j):
        j.state["rounds"] += 1

    def on_sold(self, run, j):
        if j.state["rounds"] < 2:
            return
        others = [x for x in run.jokers if x.uid != j.uid]
        if not others:
            return
        src = run.rng.stream("misc").choice(others)
        edition = src.edition if src.edition != Edition.NEGATIVE else Edition.NONE
        if not cardops.has_joker_room(run, edition):
            return
        copy = cardops.new_joker(run, src.id, edition=edition)
        copy.state = dict(src.state)
        copy.sell_value = src.sell_value
        cardops.add_joker(run, copy)


register_joker(
    JokerDef("j_invisible", "Invisible Joker", Rarity.RARE, 8, _InvisibleJoker(),
             init_state={"rounds": 0}, copyable=False, eternal_compat=False)
)


class _Chicot(JokerEffect):
    """Disables the effect of every Boss Blind."""

    def on_blind_selected(self, run, j):
        rnd = run.round
        if rnd is not None and rnd.boss_id is not None:
            rnd.boss_state["disabled"] = True


register_joker(JokerDef("j_chicot", "Chicot", Rarity.LEGENDARY, 20, _Chicot(), copyable=False))
