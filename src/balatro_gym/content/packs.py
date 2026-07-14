"""Booster pack definitions — all 15 (5 categories x 3 sizes).

Costs, option/pick counts, and spawn weights from Booster Packs.wiki.
Categories with no registered content (e.g. spectral before spectrals ship)
are filtered out of shop offers by the engine.
"""

from __future__ import annotations

from balatro_gym.engine.effects import PackDef, register_pack

# (size, cost, weight_scale) — weights: Standard/Arcana/Celestial 4/2/0.5,
# Buffoon 1.2/0.6/0.15, Spectral 0.6/0.3/0.07.
_SIZES = [("normal", 4), ("jumbo", 6), ("mega", 8)]

_SPECS = {
    # category: (options, picks) per size, weights per size
    "standard": ([(3, 1), (5, 1), (5, 2)], [4.0, 2.0, 0.5]),
    "arcana": ([(3, 1), (5, 1), (5, 2)], [4.0, 2.0, 0.5]),
    "celestial": ([(3, 1), (5, 1), (5, 2)], [4.0, 2.0, 0.5]),
    "buffoon": ([(2, 1), (4, 1), (4, 2)], [1.2, 0.6, 0.15]),
    "spectral": ([(2, 1), (4, 1), (4, 2)], [0.6, 0.3, 0.07]),
}

for category, (shapes, weights) in _SPECS.items():
    for (size, cost), (options, picks), weight in zip(_SIZES, shapes, weights):
        register_pack(
            PackDef(
                id=f"p_{category}_{size}",
                name=f"{size.title()} {category.title()} Pack",
                category=category,
                cost=cost,
                options=options,
                picks=picks,
                weight=weight,
            )
        )
