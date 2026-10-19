"""The 8 stakes. Modifiers are cumulative: engine.run.Game applies every
stake up to the chosen one, so each entry only adds its own twist.

- White: base game.
- Red: the Small Blind pays no reward money.
- Green: blind requirements scale faster (engine.levels, tier 2 table).
- Black: shop jokers may be Eternal (30%).
- Blue: one fewer discard per round.
- Purple: requirements scale faster still (tier 3 table).
- Orange: shop jokers may be Perishable (30%; debuffed after 5 rounds).
- Gold: shop jokers may be Rental (30%; cost $1, charge $3 every round).
"""

from __future__ import annotations

from balatro_gym.engine.effects import StakeDef, register_stake
from balatro_gym.engine.state import RunState
from balatro_gym.engine.types import Stake


def _white(run: RunState) -> None:
    pass


def _red(run: RunState) -> None:
    run.mods["no_small_reward"] = True


def _green(run: RunState) -> None:
    pass  # scaling is derived from run.stake in engine.levels


def _black(run: RunState) -> None:
    run.mods["eternals"] = True


def _blue(run: RunState) -> None:
    run.discards_per_round -= 1


def _purple(run: RunState) -> None:
    pass  # scaling is derived from run.stake in engine.levels


def _orange(run: RunState) -> None:
    run.mods["perishables"] = True


def _gold(run: RunState) -> None:
    run.mods["rentals"] = True


register_stake(StakeDef(Stake.WHITE, _white))
register_stake(StakeDef(Stake.RED, _red))
register_stake(StakeDef(Stake.GREEN, _green))
register_stake(StakeDef(Stake.BLACK, _black))
register_stake(StakeDef(Stake.BLUE, _blue))
register_stake(StakeDef(Stake.PURPLE, _purple))
register_stake(StakeDef(Stake.ORANGE, _orange))
register_stake(StakeDef(Stake.GOLD, _gold))
