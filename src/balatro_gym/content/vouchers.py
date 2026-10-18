"""All 32 vouchers (16 base / upgrade pairs). Each costs $10.

Vouchers either change a run field directly (slots, hands, discards, hand
size, interest cap, reroll price, ante) or write a key into run.mods that the
engine reads where it matters:
  shop_discount (%), edition_rate, tarot_rate / planet_rate (shop weight
  multipliers), card_rate (playing cards in the shop), illusion, telescope,
  observatory (x mult per matching planet), omen_globe, boss_rerolls
  (1 = once per ante, -1 = unlimited).
"""

from __future__ import annotations

from balatro_gym.engine.effects import VoucherDef, register_voucher
from balatro_gym.engine.state import RunState


def _v(vid: str, name: str, fn, requires: str | None = None) -> None:
    register_voucher(VoucherDef(vid, name, 10, fn, requires))


# Shop size
_v("v_overstock", "Overstock", lambda r: setattr(r, "shop_card_slots", r.shop_card_slots + 1))
_v("v_overstock_plus", "Overstock Plus",
   lambda r: setattr(r, "shop_card_slots", r.shop_card_slots + 1), "v_overstock")

# Prices
_v("v_clearance_sale", "Clearance Sale", lambda r: r.mods.__setitem__("shop_discount", 25))
_v("v_liquidation", "Liquidation", lambda r: r.mods.__setitem__("shop_discount", 50), "v_clearance_sale")

# Editions
_v("v_hone", "Hone", lambda r: r.mods.__setitem__("edition_rate", 2))
_v("v_glow_up", "Glow Up", lambda r: r.mods.__setitem__("edition_rate", 4), "v_hone")

# Rerolls
_v("v_reroll_surplus", "Reroll Surplus", lambda r: setattr(r, "reroll_base", r.reroll_base - 2))
_v("v_reroll_glut", "Reroll Glut",
   lambda r: setattr(r, "reroll_base", r.reroll_base - 2), "v_reroll_surplus")

# Consumables
_v("v_crystal_ball", "Crystal Ball", lambda r: setattr(r, "consumable_slots", r.consumable_slots + 1))
_v("v_omen_globe", "Omen Globe", lambda r: r.mods.__setitem__("omen_globe", True), "v_crystal_ball")

# Planets
_v("v_telescope", "Telescope", lambda r: r.mods.__setitem__("telescope", True))
_v("v_observatory", "Observatory", lambda r: r.mods.__setitem__("observatory", 1.5), "v_telescope")

# Hands / discards
_v("v_grabber", "Grabber", lambda r: setattr(r, "hands_per_round", r.hands_per_round + 1))
_v("v_nacho_tong", "Nacho Tong",
   lambda r: setattr(r, "hands_per_round", r.hands_per_round + 1), "v_grabber")
_v("v_wasteful", "Wasteful", lambda r: setattr(r, "discards_per_round", r.discards_per_round + 1))
_v("v_recyclomancy", "Recyclomancy",
   lambda r: setattr(r, "discards_per_round", r.discards_per_round + 1), "v_wasteful")

# Shop consumable frequency (base weight 4 -> 9.6 -> 32)
_v("v_tarot_merchant", "Tarot Merchant", lambda r: r.mods.__setitem__("tarot_rate", 2.4))
_v("v_tarot_tycoon", "Tarot Tycoon", lambda r: r.mods.__setitem__("tarot_rate", 8.0), "v_tarot_merchant")
_v("v_planet_merchant", "Planet Merchant", lambda r: r.mods.__setitem__("planet_rate", 2.4))
_v("v_planet_tycoon", "Planet Tycoon", lambda r: r.mods.__setitem__("planet_rate", 8.0), "v_planet_merchant")

# Interest
_v("v_seed_money", "Seed Money", lambda r: setattr(r, "interest_cap", 50))
_v("v_money_tree", "Money Tree", lambda r: setattr(r, "interest_cap", 100), "v_seed_money")

# Joker slots
_v("v_blank", "Blank", lambda r: None)
_v("v_antimatter", "Antimatter", lambda r: setattr(r, "joker_slots", r.joker_slots + 1), "v_blank")

# Playing cards in the shop
_v("v_magic_trick", "Magic Trick", lambda r: r.mods.__setitem__("card_rate", 4.0))
_v("v_illusion", "Illusion", lambda r: r.mods.__setitem__("illusion", True), "v_magic_trick")


# Ante
def _hieroglyph(r: RunState) -> None:
    r.ante -= 1
    r.hands_per_round -= 1


def _petroglyph(r: RunState) -> None:
    r.ante -= 1
    r.discards_per_round -= 1


_v("v_hieroglyph", "Hieroglyph", _hieroglyph)
_v("v_petroglyph", "Petroglyph", _petroglyph, "v_hieroglyph")

# Boss rerolls
_v("v_directors_cut", "Director's Cut", lambda r: r.mods.__setitem__("boss_rerolls", 1))
_v("v_retcon", "Retcon", lambda r: r.mods.__setitem__("boss_rerolls", -1), "v_directors_cut")

# Hand size
_v("v_paint_brush", "Paint Brush", lambda r: setattr(r, "hand_size", r.hand_size + 1))
_v("v_palette", "Palette", lambda r: setattr(r, "hand_size", r.hand_size + 1), "v_paint_brush")
