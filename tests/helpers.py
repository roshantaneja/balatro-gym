"""Shared test builders: compact card specs and run construction.

Card spec grammar: "<rank><suit>[(mod,mod,...)]"
  rank: 2-10, J, Q, K, A        suit: S, H, C, D
  mods: enhancements  bonus|mult|wild|glass|steel|stone|gold|lucky
        editions      foil|holo|poly|negative
        seals         red_seal|blue_seal|gold_seal|purple_seal
e.g. "9S", "KH(glass)", "AS(foil,red_seal)"
"""

from __future__ import annotations

import itertools

from balatro_gym.content import load_all
from balatro_gym.engine import (
    Card,
    Edition,
    Enhancement,
    JokerInst,
    Rank,
    RunState,
    Seal,
    Suit,
    new_joker_state,
    new_run_state,
)

load_all()

_RANKS = {
    "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE, "6": Rank.SIX,
    "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE, "10": Rank.TEN, "T": Rank.TEN,
    "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE,
}
_SUITS = {"S": Suit.SPADES, "H": Suit.HEARTS, "C": Suit.CLUBS, "D": Suit.DIAMONDS}
_ENH = {e.value: e for e in Enhancement if e != Enhancement.NONE}
_ED = {e.value: e for e in Edition if e != Edition.NONE}
_SEALS = {f"{s.value}_seal": s for s in Seal if s != Seal.NONE}

_uid_counter = itertools.count(1000)


def card(spec: str) -> Card:
    mods: list[str] = []
    if "(" in spec:
        spec, rest = spec.split("(", 1)
        mods = [m.strip() for m in rest.rstrip(")").split(",")]
    suit = _SUITS[spec[-1].upper()]
    rank = _RANKS[spec[:-1].upper()]
    c = Card(uid=next(_uid_counter), rank=rank, suit=suit)
    for m in mods:
        if m in _ENH:
            if c.enhancement != Enhancement.NONE:
                raise ValueError(f"card {spec!r}: at most one enhancement")
            c.enhancement = _ENH[m]
        elif m in _ED:
            if c.edition != Edition.NONE:
                raise ValueError(f"card {spec!r}: at most one edition")
            c.edition = _ED[m]
        elif m in _SEALS:
            if c.seal != Seal.NONE:
                raise ValueError(f"card {spec!r}: at most one seal")
            c.seal = _SEALS[m]
        else:
            raise ValueError(f"unknown card mod: {m!r}")
    return c


def cards(*specs: str) -> list[Card]:
    return [card(s) for s in specs]


def joker(jid: str, *, edition: str = "none", state: dict | None = None) -> JokerInst:
    j = JokerInst(id=jid, uid=next(_uid_counter), edition=Edition(edition))
    j.state = new_joker_state(jid)
    if state:
        j.state.update(state)
    return j


def make_run(
    *,
    jokers: list[JokerInst] | None = None,
    hand_levels: dict | None = None,
    seed: str = "test-seed",
) -> RunState:
    run = new_run_state(seed=seed)
    if jokers:
        run.jokers = jokers
    if hand_levels:
        for ht, lvl in hand_levels.items():
            run.hand_levels[ht] = lvl
    return run
