"""Dict observation encoder — the numpy boundary.

IDs go in dedicated int64 vectors for embedding lookup; everything else is
dense float32. Buffers are preallocated per env instance and rewritten in
place each step; encode() returns copies (vector envs stack them).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from balatro_gym.engine import (
    BOSSES,
    CONSUMABLES,
    JOKERS,
    PACKS,
    VOUCHERS,
    Card,
    Edition,
    Enhancement,
    HandType,
    Phase,
    Rank,
    RunState,
    Seal,
    Suit,
    boss_def,
)
from balatro_gym.engine.levels import hand_value

from . import norms

HAND_SLOTS = 8
CARD_FEATS = 40
JOKER_SLOTS = 6
JOKER_FEATS = 8
CONSUMABLE_SLOTS = 3
SHOP_CARD_SLOTS = 3
PACK_OPTIONS = 5
SHOP_FEATS = 3 * 5 + 2 * 10 + 2 + 1  # cards + packs + voucher + reroll = 38
DECK_FEATS = 72
CONTEXT_FEATS = 32
BOSS_FEATS = 12

_ENH_LIST = list(Enhancement)          # 9 incl. NONE
_EDITION_CARD = [Edition.NONE, Edition.FOIL, Edition.HOLO, Edition.POLY]
_EDITION_JOKER = [Edition.FOIL, Edition.HOLO, Edition.POLY, Edition.NEGATIVE]
_SEAL_LIST = list(Seal)                # 5 incl. NONE
_PHASES = [
    Phase.BLIND_SELECT, Phase.PLAYING, Phase.SHOP, Phase.PACK_OPEN,
]  # + env-level TARGET + terminal -> 6 one-hot slots
_PACK_CATEGORIES = ["standard", "arcana", "celestial", "buffoon", "spectral"]
_PACK_SIZES = ["normal", "jumbo", "mega"]
_OPTION_KINDS = ["card", "joker", "tarot", "planet", "spectral"]


def canonical_order(hand: list[Card]) -> list[int]:
    """Engine-hand indices in canonical display order: rank desc, suit asc
    (S>H>C>D), then uid. Content-based, not value-ranked."""
    return sorted(
        range(len(hand)),
        key=lambda i: (-int(hand[i].rank), int(hand[i].suit), hand[i].uid),
    )


class ObsEncoder:
    def __init__(self):
        # Stable catalogs (sorted registry keys; 0 = empty).
        self.joker_index = {jid: i + 1 for i, jid in enumerate(sorted(JOKERS))}
        self.consumable_index = {cid: i + 1 for i, cid in enumerate(sorted(CONSUMABLES))}
        self.boss_index = {bid: i + 1 for i, bid in enumerate(sorted(BOSSES))}
        self.voucher_index = {vid: i + 1 for i, vid in enumerate(sorted(VOUCHERS))}
        # Combined item catalog for shop/pack slots: jokers then consumables.
        self.item_index = {}
        for jid, idx in self.joker_index.items():
            self.item_index[("joker", jid)] = idx
        off = len(self.joker_index)
        for cid, idx in self.consumable_index.items():
            self.item_index[("consumable", cid)] = off + idx
        self.n_items = 1 + len(self.joker_index) + len(self.consumable_index)

        self._buf = {
            "hand": np.zeros((HAND_SLOTS, CARD_FEATS), dtype=np.float32),
            "joker_ids": np.zeros(JOKER_SLOTS, dtype=np.int64),
            "joker_state": np.zeros((JOKER_SLOTS, JOKER_FEATS), dtype=np.float32),
            "consumable_ids": np.zeros(CONSUMABLE_SLOTS, dtype=np.int64),
            "shop": np.zeros(SHOP_FEATS, dtype=np.float32),
            "shop_item_ids": np.zeros(SHOP_CARD_SLOTS + 1, dtype=np.int64),  # +1 = voucher
            "pack": np.zeros((PACK_OPTIONS, len(_OPTION_KINDS) + 1), dtype=np.float32),
            "pack_item_ids": np.zeros(PACK_OPTIONS, dtype=np.int64),
            "pack_cards": np.zeros((PACK_OPTIONS, CARD_FEATS), dtype=np.float32),
            "hand_levels": np.zeros((len(HandType), 3), dtype=np.float32),
            "deck": np.zeros(DECK_FEATS, dtype=np.float32),
            "context": np.zeros(CONTEXT_FEATS, dtype=np.float32),
            "boss_id": np.zeros(1, dtype=np.int64),
            "boss": np.zeros(BOSS_FEATS, dtype=np.float32),
        }

    # ------------------------------------------------------------------ space
    def space(self) -> spaces.Dict:
        f32 = np.float32
        return spaces.Dict(
            {
                "hand": spaces.Box(0, 1, (HAND_SLOTS, CARD_FEATS), f32),
                "joker_ids": spaces.MultiDiscrete(
                    [1 + len(self.joker_index)] * JOKER_SLOTS, dtype=np.int64
                ),
                "joker_state": spaces.Box(-1, 10, (JOKER_SLOTS, JOKER_FEATS), f32),
                "consumable_ids": spaces.MultiDiscrete(
                    [1 + len(self.consumable_index)] * CONSUMABLE_SLOTS, dtype=np.int64
                ),
                "shop": spaces.Box(0, 10, (SHOP_FEATS,), f32),
                "shop_item_ids": spaces.MultiDiscrete(
                    [self.n_items] * SHOP_CARD_SLOTS + [1 + len(self.voucher_index)],
                    dtype=np.int64,
                ),
                "pack": spaces.Box(0, 1, (PACK_OPTIONS, len(_OPTION_KINDS) + 1), f32),
                "pack_item_ids": spaces.MultiDiscrete([self.n_items] * PACK_OPTIONS, dtype=np.int64),
                "pack_cards": spaces.Box(0, 1, (PACK_OPTIONS, CARD_FEATS), f32),
                "hand_levels": spaces.Box(0, 10, (len(HandType), 3), f32),
                "deck": spaces.Box(0, 1, (DECK_FEATS,), f32),
                "context": spaces.Box(-1, 10, (CONTEXT_FEATS,), f32),
                "boss_id": spaces.MultiDiscrete([1 + len(self.boss_index)], dtype=np.int64),
                "boss": spaces.Box(0, 6, (BOSS_FEATS,), f32),
            }
        )

    # ---------------------------------------------------------------- encoding
    def _card_row(self, row: np.ndarray, card: Card, debuffed: bool, face_down: bool) -> None:
        row.fill(0.0)
        if face_down:
            # Face-down cards expose nothing but presence + the flag.
            row[36] = 1.0  # face_down
            row[37] = 1.0  # present
            return
        row[int(card.rank) - 2] = 1.0                      # 0-12 rank
        row[13 + int(card.suit)] = 1.0                     # 13-16 suit
        row[17] = card.base_chips / 61.0                   # chips (stone+extra can exceed 11)
        row[18 + _ENH_LIST.index(card.enhancement)] = 1.0  # 18-26 enhancement
        ed = card.edition if card.edition in _EDITION_CARD else Edition.NONE
        row[27 + _EDITION_CARD.index(ed)] = 1.0            # 27-30 edition
        row[31 + _SEAL_LIST.index(card.seal)] = 1.0        # 31-35 seal
        row[36] = 0.0                                      # face_down
        row[37] = 1.0                                      # present
        row[38] = 1.0 if debuffed else 0.0
        # 39 pad

    def encode(
        self,
        run: RunState,
        *,
        env_phase: str,
        hand_order: list[int],
        pending: tuple[int, int, int] | None,
        enable_skip: bool,
    ) -> dict:
        b = self._buf
        rnd = run.round
        boss = (
            boss_def(rnd.boss_id).effect if (rnd is not None and rnd.boss_id) else None
        )
        face_down = set(rnd.face_down_uids) if rnd is not None else set()

        # --- hand (canonical order) ---
        b["hand"].fill(0.0)
        if rnd is not None:
            for slot, hi in enumerate(hand_order[:HAND_SLOTS]):
                card = rnd.hand[hi]
                debuffed = boss.card_debuffed(run, card) if boss else False
                self._card_row(b["hand"][slot], card, debuffed, card.uid in face_down)

        # --- jokers ---
        b["joker_ids"].fill(0)
        b["joker_state"].fill(0.0)
        for i, j in enumerate(run.jokers[:JOKER_SLOTS]):
            b["joker_ids"][i] = self.joker_index.get(j.id, 0)
            row = b["joker_state"][i]
            if j.edition in _EDITION_JOKER:
                row[_EDITION_JOKER.index(j.edition)] = 1.0
            row[4] = min(j.sell_value, 20) / 20.0
            counter = 0.0
            for v in j.state.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    counter = max(counter, float(v))
            row[5] = norms.norm_counter(counter)
            row[6] = 1.0 if "eternal" in j.stickers else 0.0
            row[7] = 1.0  # present

        # --- consumables ---
        b["consumable_ids"].fill(0)
        for i, c in enumerate(run.consumables[:CONSUMABLE_SLOTS]):
            b["consumable_ids"][i] = self.consumable_index.get(c.id, 0)

        # --- shop ---
        b["shop"].fill(0.0)
        b["shop_item_ids"].fill(0)
        shop = run.shop
        if shop is not None:
            for i, item in enumerate(shop.card_slots[:SHOP_CARD_SLOTS]):
                base = i * 5
                kind_idx = {"joker": 0, "tarot": 1, "planet": 2}.get(item.kind, 2)
                b["shop"][base + kind_idx] = 1.0
                b["shop"][base + 3] = item.cost / 10.0
                b["shop"][base + 4] = 1.0  # present
                key = ("joker", item.item_id) if item.kind == "joker" else ("consumable", item.item_id)
                b["shop_item_ids"][i] = self.item_index.get(key, 0)
            for i, offer in enumerate(shop.pack_slots[:2]):
                base = 15 + i * 10
                defn = PACKS[offer.pack_id]
                b["shop"][base + _PACK_CATEGORIES.index(defn.category)] = 1.0
                size = offer.pack_id.rsplit("_", 1)[-1]
                if size in _PACK_SIZES:
                    b["shop"][base + 5 + _PACK_SIZES.index(size)] = 1.0
                b["shop"][base + 8] = offer.cost / 10.0
                b["shop"][base + 9] = 1.0
            b["shop"][35] = 1.0 if shop.voucher else 0.0
            b["shop"][36] = 1.0 if shop.voucher else 0.0  # voucher cost / 10 == 1.0
            b["shop"][37] = shop.reroll_cost / 10.0
            b["shop_item_ids"][SHOP_CARD_SLOTS] = (
                self.voucher_index.get(shop.voucher, 0) if shop.voucher else 0
            )

        # --- open pack ---
        b["pack"].fill(0.0)
        b["pack_item_ids"].fill(0)
        b["pack_cards"].fill(0.0)
        pack = shop.open_pack if shop is not None else None
        if pack is not None:
            for i in range(min(len(pack.option_kinds), PACK_OPTIONS)):
                kind = pack.option_kinds[i]
                b["pack"][i][_OPTION_KINDS.index(kind)] = 1.0
                b["pack"][i][len(_OPTION_KINDS)] = 1.0  # present
                if kind == "card":
                    card = pack.option_cards[i]
                    if card is not None:
                        self._card_row(b["pack_cards"][i], card, False, False)
                else:
                    key = ("joker", pack.option_ids[i]) if kind == "joker" else (
                        "consumable", pack.option_ids[i]
                    )
                    b["pack_item_ids"][i] = self.item_index.get(key, 0)

        # --- hand levels ---
        for ht in HandType:
            i = int(ht) - 1
            lvl = run.hand_levels[ht]
            chips, mult = hand_value(ht, lvl)
            b["hand_levels"][i][0] = min(lvl, 16) / 16.0
            b["hand_levels"][i][1] = norms.norm_score(chips * mult)
            b["hand_levels"][i][2] = min(run.hand_play_counts.get(ht, 0), 20) / 20.0

        # --- deck composition (remaining draw pile + full deck) ---
        b["deck"].fill(0.0)
        remaining = rnd.deck_remaining if rnd is not None else run.full_deck
        self._deck_block(b["deck"][:36], remaining)
        self._deck_block(b["deck"][36:], run.full_deck)

        # --- context ---
        ctx = b["context"]
        ctx.fill(0.0)
        phase_idx = {
            "blind_select": 0, "playing": 1, "shop": 2, "pack_open": 3,
            "target": 4, "terminal": 5,
        }[env_phase]
        ctx[phase_idx] = 1.0
        ctx[6] = run.ante / 8.0
        blind_idx = min(run.blind_index, 2)
        ctx[7 + blind_idx] = 1.0  # 7-9 small/big/boss
        if rnd is not None:
            ctx[10] = norms.norm_score(rnd.chips_needed)
            ctx[11] = norms.norm_score(rnd.chips_scored)
            ctx[12] = norms.norm_progress(rnd.chips_scored, rnd.chips_needed)
            ctx[13] = rnd.hands_left / 10.0
            ctx[14] = rnd.discards_left / 10.0
        ctx[15] = norms.norm_money(run.money)
        ctx[16] = norms.norm_money_linear(run.money)
        ctx[17] = max(run.joker_slots - len(run.jokers), 0) / 6.0
        ctx[18] = max(run.consumable_slots - len(run.consumables), 0) / 3.0
        ctx[19] = run.hand_size / 12.0
        ctx[20] = (pack.picks_left / 5.0) if pack is not None else 0.0
        if pending is not None:
            ctx[21] = 1.0
            ctx[22] = pending[1] / 3.0
            ctx[23] = pending[2] / 3.0
        ctx[24] = 1.0 if enable_skip else 0.0
        ctx[25] = len(remaining) / 52.0
        ctx[26] = len(rnd.hand) / 10.0 if rnd is not None else 0.0
        ctx[27] = min(len(run.tags), 5) / 5.0
        ctx[28] = min(run.stats.get("hands_played_run", 0), 100) / 100.0
        # 29-31 pad

        # --- boss ---
        b["boss_id"][0] = self.boss_index.get(rnd.boss_id, 0) if (rnd and rnd.boss_id) else 0
        bf = b["boss"]
        bf.fill(0.0)
        if boss is not None and rnd is not None:
            # Probe with a plain rank-10 card of each suit.
            debuffed_suits = [
                s for s in Suit
                if boss.card_debuffed(run, Card(uid=-1, rank=Rank.TEN, suit=s))
            ]
            if debuffed_suits:
                for s in debuffed_suits:
                    bf[1 + int(s)] = 1.0  # 1-4 per-suit debuff flags
            else:
                bf[0] = 1.0  # no suit debuff
            bf[5] = boss.score_multiplier() / 6.0
            bf[6] = (
                len(rnd.face_down_uids) / max(len(rnd.hand), 1) if rnd.hand else 0.0
            )
            # forced play count (Psychic): probe validate_play sizes 1..5
            forced = 0
            probe = rnd.hand[:5]
            if probe and boss.validate_play(run, probe[:1]) is not None:
                for k in range(2, min(5, len(rnd.hand)) + 1):
                    if boss.validate_play(run, rnd.hand[:k]) is None:
                        forced = k
                        break
            bf[7] = forced / 5.0
            bf[8] = 1.0 if rnd.hands_left == 1 and run.hands_per_round > 1 else 0.0
            bf[9] = 1.0 if boss.is_finisher else 0.0
            # 10-11 pad

        return {k: v.copy() for k, v in b.items()}

    @staticmethod
    def _deck_block(block: np.ndarray, cards: list[Card]) -> None:
        # 13 rank + 4 suit + 9 enhancement + 4 edition + 5 seal + 1 total = 36
        for c in cards:
            if not c.is_stone:
                block[int(c.rank) - 2] += 1.0
                block[13 + int(c.suit)] += 1.0
            block[17 + _ENH_LIST.index(c.enhancement)] += 1.0
            ed = c.edition if c.edition in _EDITION_CARD else Edition.NONE
            block[26 + _EDITION_CARD.index(ed)] += 1.0
            block[30 + _SEAL_LIST.index(c.seal)] += 1.0
        n = max(len(cards), 1)
        block[:35] /= n
        block[35] = len(cards) / 52.0
