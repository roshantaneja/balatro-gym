"""Named RNG streams for reproducibility.

Separate streams mean that (e.g.) how many chance rolls jokers consumed
mid-round never perturbs the next shop's contents: same seed + same shop
decisions => same shop, forever. We make no attempt to match real-game seeds.
"""

from __future__ import annotations

import random

STREAMS = ("shuffle", "shop", "packs", "chance", "boss", "consumable", "edition", "misc")


class RunRNG:
    __slots__ = ("_streams",)

    def __init__(self, seed: str):
        self._streams = {name: random.Random(f"{seed}::{name}") for name in STREAMS}

    def stream(self, name: str) -> random.Random:
        return self._streams[name]

    # --- clone / serialization -------------------------------------------------
    def getstate(self) -> dict:
        return {name: r.getstate() for name, r in self._streams.items()}

    def setstate(self, state: dict) -> None:
        for name, s in state.items():
            self._streams[name].setstate(s)

    def clone(self) -> RunRNG:
        new = object.__new__(RunRNG)
        new._streams = {}
        for name, r in self._streams.items():
            nr = random.Random()
            nr.setstate(r.getstate())
            new._streams[name] = nr
        return new

    def to_dict(self) -> dict:
        # Random.getstate() is (version, tuple[int,...], gauss_next) — JSON-safe lists.
        out = {}
        for name, r in self._streams.items():
            version, internal, gauss = r.getstate()
            out[name] = [version, list(internal), gauss]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> RunRNG:
        new = object.__new__(cls)
        new._streams = {}
        for name in STREAMS:
            r = random.Random()
            version, internal, gauss = d[name]
            r.setstate((version, tuple(internal), gauss))
            new._streams[name] = r
        return new
