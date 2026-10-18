"""All 15 starting decks.

Deck effects are applied once at run start (engine.run.Game). Some write
run.mods keys the engine reads: no_interest / extra_hand_bonus /
extra_discard_bonus (Green), spectral_rate (Ghost), anaglyph_tag (Anaglyph),
plasma + ante_scaling (Plasma).
"""

from __future__ import annotations

from balatro_gym.engine import cardops
from balatro_gym.engine.effects import DeckDef, register_deck
from balatro_gym.engine.state import RunState
from balatro_gym.engine.types import Rank, Suit


def _red(run: RunState) -> None:
    run.discards_per_round += 1


def _blue(run: RunState) -> None:
    run.hands_per_round += 1


def _yellow(run: RunState) -> None:
    run.money += 10


def _green(run: RunState) -> None:
    run.mods["extra_hand_bonus"] = 2
    run.mods["extra_discard_bonus"] = 1
    run.mods["no_interest"] = True


def _black(run: RunState) -> None:
    run.joker_slots += 1
    run.hands_per_round -= 1


def _magic(run: RunState) -> None:
    cardops.grant_voucher(run, "v_crystal_ball")
    for _ in range(2):
        cardops.create_consumable(run, "c_fool")


def _nebula(run: RunState) -> None:
    cardops.grant_voucher(run, "v_telescope")
    run.consumable_slots -= 1


def _ghost(run: RunState) -> None:
    run.mods["spectral_rate"] = 2.0
    cardops.create_consumable(run, "c_hex")


def _abandoned(run: RunState) -> None:
    run.full_deck = [c for c in run.full_deck if not c.rank.is_face]
    run.stats["starting_deck_size"] = len(run.full_deck)


def _checkered(run: RunState) -> None:
    for c in run.full_deck:
        if c.suit == Suit.CLUBS:
            c.suit = Suit.SPADES
        elif c.suit == Suit.DIAMONDS:
            c.suit = Suit.HEARTS


def _zodiac(run: RunState) -> None:
    for vid in ("v_tarot_merchant", "v_planet_merchant", "v_overstock"):
        cardops.grant_voucher(run, vid)


def _painted(run: RunState) -> None:
    run.hand_size += 2
    run.joker_slots -= 1


def _anaglyph(run: RunState) -> None:
    run.mods["anaglyph_tag"] = "t_double"


def _plasma(run: RunState) -> None:
    run.mods["plasma"] = True
    run.mods["ante_scaling"] = 2


def _erratic(run: RunState) -> None:
    rng = run.rng.stream("shuffle")
    ranks, suits = list(Rank), list(Suit)
    for c in run.full_deck:
        c.rank = rng.choice(ranks)
        c.suit = rng.choice(suits)


register_deck(DeckDef("d_red", "Red Deck", _red))
register_deck(DeckDef("d_blue", "Blue Deck", _blue))
register_deck(DeckDef("d_yellow", "Yellow Deck", _yellow))
register_deck(DeckDef("d_green", "Green Deck", _green))
register_deck(DeckDef("d_black", "Black Deck", _black))
register_deck(DeckDef("d_magic", "Magic Deck", _magic))
register_deck(DeckDef("d_nebula", "Nebula Deck", _nebula))
register_deck(DeckDef("d_ghost", "Ghost Deck", _ghost))
register_deck(DeckDef("d_abandoned", "Abandoned Deck", _abandoned))
register_deck(DeckDef("d_checkered", "Checkered Deck", _checkered))
register_deck(DeckDef("d_zodiac", "Zodiac Deck", _zodiac))
register_deck(DeckDef("d_painted", "Painted Deck", _painted))
register_deck(DeckDef("d_anaglyph", "Anaglyph Deck", _anaglyph))
register_deck(DeckDef("d_plasma", "Plasma Deck", _plasma))
register_deck(DeckDef("d_erratic", "Erratic Deck", _erratic))
