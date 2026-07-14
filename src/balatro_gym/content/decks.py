"""Starting decks (v1: Red, Blue, Yellow — numeric modifiers only)."""

from __future__ import annotations

from balatro_gym.engine.effects import DeckDef, register_deck
from balatro_gym.engine.state import RunState


def _red(run: RunState) -> None:
    run.discards_per_round += 1


def _blue(run: RunState) -> None:
    run.hands_per_round += 1


def _yellow(run: RunState) -> None:
    run.money += 10


register_deck(DeckDef("d_red", "Red Deck", _red))
register_deck(DeckDef("d_blue", "Blue Deck", _blue))
register_deck(DeckDef("d_yellow", "Yellow Deck", _yellow))
