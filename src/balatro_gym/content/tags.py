"""Skip-blind tags (v1: 4 money/utility tags)."""

from __future__ import annotations

from balatro_gym.engine.effects import TagDef, register_tag
from balatro_gym.engine.state import RunState
from balatro_gym.engine.types import BlindKind


# Handy: gives $1 per hand played so far this run, immediately on acquire.
def _handy_acquire(run: RunState) -> None:
    run.money += run.stats.get("hands_played_run", 0)
    run.tags.remove("t_handy")


# Garbage: gives $1 per unused discard so far this run, immediately on acquire.
def _garbage_acquire(run: RunState) -> None:
    run.money += run.stats.get("discards_unused_run", 0)
    run.tags.remove("t_garbage")


# Investment: +$25 after the next Boss Blind is defeated.
def _investment_payout(run: RunState) -> int:
    if run.round is not None and run.round.blind_kind == BlindKind.BOSS:
        run.tags.remove("t_investment")
        return 25
    return 0


# Juggle: +3 hand size for the next round only.
def _juggle_start(run: RunState) -> None:
    assert run.round is not None
    run.round.boss_state["hand_size_bonus"] = run.round.boss_state.get("hand_size_bonus", 0) + 3
    run.tags.remove("t_juggle")


register_tag(TagDef("t_handy", "Handy Tag", on_acquire=_handy_acquire))
register_tag(TagDef("t_garbage", "Garbage Tag", on_acquire=_garbage_acquire))
register_tag(TagDef("t_investment", "Investment Tag", round_end_money=_investment_payout))
register_tag(TagDef("t_juggle", "Juggle Tag", on_round_start=_juggle_start))
