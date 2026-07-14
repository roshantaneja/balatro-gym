"""Vouchers (v1: 4 numeric-cap vouchers)."""

from __future__ import annotations

from balatro_gym.engine.effects import VoucherDef, register_voucher
from balatro_gym.engine.state import RunState


def _overstock(run: RunState) -> None:
    run.stats["shop_card_slots"] = 3


def _grabber(run: RunState) -> None:
    run.hands_per_round += 1


def _wasteful(run: RunState) -> None:
    run.discards_per_round += 1


def _crystal_ball(run: RunState) -> None:
    run.consumable_slots += 1


register_voucher(VoucherDef("v_overstock", "Overstock", 10, _overstock))
register_voucher(VoucherDef("v_grabber", "Grabber", 10, _grabber))
register_voucher(VoucherDef("v_wasteful", "Wasteful", 10, _wasteful))
register_voucher(VoucherDef("v_crystal_ball", "Crystal Ball", 10, _crystal_ball))
