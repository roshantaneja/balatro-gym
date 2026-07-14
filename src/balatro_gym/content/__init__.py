"""Game content: jokers, consumables, bosses, decks, vouchers, tags.

load_all() auto-imports every module in this package (recursively), which
registers content into the engine.effects registries. New content modules
need no wiring — just drop the file in.
"""

import importlib
import pkgutil

_loaded = False


def load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    import balatro_gym.content as pkg

    for mod in pkgutil.walk_packages(pkg.__path__, prefix="balatro_gym.content."):
        importlib.import_module(mod.name)
