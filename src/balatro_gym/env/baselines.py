"""Baseline agents: the benchmark floor for trained policies.

RandomLegalAgent  — uniform over legal actions.
HeuristicAgent    — greedy max-immediate-score; naive shop.
StrategistAgent   — community-strategy agent (see docs/strategy.md):
                    pair-spam + scaling jokers + interest hoarding.
"""

from __future__ import annotations

import numpy as np

from balatro_gym.engine import JOKERS, HandType, Phase, score_play

from . import action_codec as codec


class RandomLegalAgent:
    """Uniform over legal actions."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))


class HeuristicAgent:
    """Greedy: plays the max-immediate-score subset; discards low cards when
    the best play is weak and discards remain; buys cheap jokers in shops."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, mask: np.ndarray) -> int:
        run = env.run
        if run.phase == Phase.PLAYING and env._pending is None:
            return self._play_phase(env, mask)
        # Shop/pack/blind phases: prefer buy-joker > buy-pack > pick > select blind > leave.
        for base, n in (
            (codec.BUY_CARD_BASE, codec.N_BUY_CARD_SLOTS),
            (codec.BUY_PACK_BASE, codec.N_BUY_PACK_SLOTS),
            (codec.PICK_PACK_BASE, codec.N_PACK_OPTIONS),
        ):
            for i in range(n):
                if mask[base + i]:
                    return base + i
        for idx in (codec.SELECT_BLIND, codec.SKIP_PACK_IDX, codec.LEAVE_SHOP_IDX):
            if mask[idx]:
                return idx
        return int(self.rng.choice(np.flatnonzero(mask)))

    def _play_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        rnd = run.round
        assert rnd is not None

        best_idx, best_score = None, -1.0
        for i, subset in enumerate(codec.SUBSETS_5):
            gidx = codec.PLAY_BASE + i
            if not mask[gidx]:
                continue
            engine_idxs = tuple(sorted(env._hand_order[p] for p in subset))
            clone = run.clone()
            assert clone.round is not None
            played = [clone.round.hand[j] for j in engine_idxs]
            clone.round.hand = [
                c for j, c in enumerate(clone.round.hand) if j not in set(engine_idxs)
            ]
            res = score_play(clone, played, held=clone.round.hand)
            if res.score > best_score:
                best_score, best_idx = res.score, gidx

        if best_idx is None:
            return int(self.rng.choice(np.flatnonzero(mask)))

        need = rnd.chips_needed - rnd.chips_scored
        weak = best_score < need / max(rnd.hands_left, 1)
        if weak and rnd.discards_left > 0:
            # Discard the lowest-rank cards (up to 4).
            order = env._hand_order
            low = sorted(range(len(order)), key=lambda p: int(rnd.hand[order[p]].rank))[:4]
            subset = tuple(sorted(low[: min(4, len(low))]))
            try:
                didx = codec.DISCARD_BASE + codec._SUBSET5_INDEX[subset]
                if mask[didx]:
                    return didx
            except KeyError:
                pass
        return best_idx


# ---------------------------------------------------------------------------
# StrategistAgent — codified community strategy (docs/strategy.md)
# ---------------------------------------------------------------------------

# Joker shop-priority tiers (higher = better). See docs/strategy.md.
JOKER_TIERS: dict[str, int] = {
    # S: scalers, pair-spam xmult, copies — always buy
    "j_supernova": 5, "j_green_joker": 5, "j_ride_the_bus": 5, "j_castle": 5,
    "j_duo": 5, "j_blueprint": 5, "j_brainstorm": 5,
    # A: strong support — buy if half the interest reserve survives
    "j_space": 4, "j_photograph": 4, "j_half": 4, "j_sly": 4, "j_banner": 4,
    "j_hack": 4, "j_golden": 4,
    # B: decent early buys
    "j_joker": 3, "j_jolly": 3, "j_abstract": 3, "j_scholar": 3,
    "j_odd_todd": 3, "j_fortune_teller": 3, "j_bloodstone": 3, "j_mime": 3,
    "j_dusk": 3, "j_baron": 3,
    # C: tempo only (antes 1-3)
    "j_ice_cream": 2, "j_popcorn": 2, "j_gros_michel": 2, "j_business": 2,
    "j_greedy": 2, "j_lusty": 2, "j_wrathful": 2, "j_gluttonous": 2,
    "j_shoot_the_moon": 2, "j_raised_fist": 2, "j_mail_in": 2, "j_8ball": 2,
    "j_egg": 2,
    # D: situational / avoid
    "j_baseball": 1, "j_dna": 1, "j_splash": 1, "j_four_fingers": 1,
    "j_pareidolia": 0,  # breaks Ride the Bus (all cards become scoring faces)
}

INTEREST_CAP_MONEY = 25  # $ needed for max interest

# Planet id -> hand it levels (for spending planets on the spam hand).
PLANET_HAND = {
    "c_pluto": HandType.HIGH_CARD, "c_mercury": HandType.PAIR,
    "c_uranus": HandType.TWO_PAIR, "c_venus": HandType.THREE_KIND,
    "c_saturn": HandType.STRAIGHT, "c_jupiter": HandType.FLUSH,
    "c_earth": HandType.FULL_HOUSE, "c_mars": HandType.FOUR_KIND,
    "c_neptune": HandType.STRAIGHT_FLUSH, "c_planet_x": HandType.FIVE_KIND,
    "c_ceres": HandType.FLUSH_HOUSE, "c_eris": HandType.FLUSH_FIVE,
}


class StrategistAgent(HeuristicAgent):
    """Pair-spam scaling build with interest discipline.

    Play phase: greedy immediate score, EXCEPT
      - prefers the run's spam hand (most-played) when within 25% of best,
      - avoids scoring face cards when Ride the Bus is owned (if close),
      - discards only truly dead hands, never when Green Joker is owned
        unless desperate.
    Shop phase: tiered joker buying under an interest reserve, decay selling,
    targeted rerolls, planet/economy consumable usage.
    """

    # ------------------------------------------------------------- helpers
    def _reserve(self, run) -> int:
        """Money we refuse to dip below (interest hoarding)."""
        if run.ante <= 2:
            return 0
        if run.ante == 3:
            return 10
        if run.ante == 4:
            return 15
        return INTEREST_CAP_MONEY

    def _tier(self, jid: str, run) -> int:
        t = JOKER_TIERS.get(jid, 1)
        if t == 2 and run.ante > 3:
            t = 1  # tempo jokers fall off
        if jid in ("j_blueprint", "j_brainstorm"):
            # A copy joker is only as good as what it can copy.
            copyable = any(JOKER_TIERS.get(j.id, 1) >= 4 for j in run.jokers)
            t = 5 if copyable else 1
        if jid == "j_baseball":
            uncommons = sum(
                1 for j in run.jokers
                if JOKERS[j.id].rarity.value == "uncommon"
            )
            t = 3 if uncommons >= 2 else 1
        return t

    def _spam_hand(self, run) -> HandType:
        counts = run.hand_play_counts
        best = max(counts, key=lambda ht: (counts.get(ht, 0), -int(ht)))
        return best if counts.get(best, 0) > 0 else HandType.PAIR

    def _owned(self, run, jid: str) -> bool:
        return any(j.id == jid for j in run.jokers)

    # ------------------------------------------------------------ main act
    def act(self, env, obs, mask: np.ndarray) -> int:
        run = env.run
        if run.phase == Phase.PLAYING and env._pending is None:
            use = self._consumable_action(env, mask)
            if use is not None:
                return use
            return self._play_phase(env, mask)
        if str(run.phase.value) == "shop":
            return self._shop_phase(env, mask)
        if str(run.phase.value) == "pack_open":
            return self._pack_phase(env, mask)
        # blind select: never skip (v1 tags aren't worth a shop visit)
        if mask[codec.SELECT_BLIND]:
            return codec.SELECT_BLIND
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))

    # --------------------------------------------------------- consumables
    def _consumable_action(self, env, mask: np.ndarray) -> int | None:
        run = env.run
        spam = self._spam_hand(run)
        for i, inst in enumerate(run.consumables[:codec.N_CONSUMABLE_SLOTS]):
            idx = codec.USE_CONSUMABLE_BASE + i
            if not mask[idx]:
                continue
            if inst.id in PLANET_HAND:
                # Level the spam hand immediately; off-hand planets only if
                # both slots are clogged.
                if PLANET_HAND[inst.id] == spam or len(run.consumables) >= run.consumable_slots:
                    return idx
            elif inst.id == "c_hermit" and run.money >= 15:
                return idx
            elif inst.id == "c_temperance" and sum(j.sell_value for j in run.jokers) >= 8:
                return idx
        return None

    # ---------------------------------------------------------- play phase
    def _play_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        rnd = run.round
        assert rnd is not None
        spam = self._spam_hand(run)
        has_rtb = self._owned(run, "j_ride_the_bus")
        has_green = self._owned(run, "j_green_joker")
        # The no-discard / spam-hand doctrine only applies once a joker engine
        # exists; with an empty board, play pure greedy and dig hard.
        scaler_owned = any(
            self._owned(run, jid)
            for jid in ("j_supernova", "j_green_joker", "j_ride_the_bus", "j_castle")
        )
        has_engine = len(run.jokers) >= 2

        candidates = []  # (global_idx, score, hand_type, scores_face)
        for i, subset in enumerate(codec.SUBSETS_5):
            gidx = codec.PLAY_BASE + i
            if not mask[gidx]:
                continue
            engine_idxs = tuple(sorted(env._hand_order[p] for p in subset))
            clone = run.clone()
            assert clone.round is not None
            played = [clone.round.hand[j] for j in engine_idxs]
            clone.round.hand = [
                c for j, c in enumerate(clone.round.hand) if j not in set(engine_idxs)
            ]
            res = score_play(clone, played, held=clone.round.hand)
            scores_face = any(
                c.rank.is_face for c in res.hand_result.scoring_cards if not c.is_stone
            )
            candidates.append((gidx, res.score, res.hand_result.hand_type, scores_face))

        if not candidates:
            legal = np.flatnonzero(mask)
            return int(self.rng.choice(legal))

        best_score = max(c[1] for c in candidates)
        need = rnd.chips_needed - rnd.chips_scored

        def pick_value(c):
            gidx, score, ht, face = c
            v = score
            if score < need:  # only shape when not immediately lethal
                if scaler_owned and ht == spam:
                    v *= 1.25  # spam-hand preference (Supernova/level scaling)
                if has_rtb and face:
                    v *= 0.6   # don't reset Ride the Bus
            return v

        gidx, score, ht, face = max(candidates, key=pick_value)

        # Discard: dig hard with no engine; with an engine, only when weak.
        if not has_engine:
            weak_factor = 1.0
        elif has_green:
            weak_factor = 0.3
        else:
            weak_factor = 0.5
        weak = best_score < (need / max(rnd.hands_left, 1)) * weak_factor
        if weak and rnd.discards_left > 0 and rnd.hands_left > 1:
            didx = self._discard_action(env, mask)
            if didx is not None:
                return didx
        return gidx

    def _discard_action(self, env, mask: np.ndarray) -> int | None:
        """Discard singleton ranks (lowest first) — dig without breaking pairs."""
        run = env.run
        rnd = run.round
        assert rnd is not None
        from collections import Counter

        order = env._hand_order
        counts = Counter(rnd.hand[i].rank for i in order)
        singles = [
            p for p in range(len(order)) if counts[rnd.hand[order[p]].rank] == 1
        ]
        singles.sort(key=lambda p: int(rnd.hand[order[p]].rank))
        pick = singles[:4]
        if not pick:  # everything is paired: discard the lowest pair's cards
            low = sorted(range(len(order)), key=lambda p: int(rnd.hand[order[p]].rank))
            pick = low[:2]
        subset = tuple(sorted(pick))
        idx = codec._SUBSET5_INDEX.get(subset)
        if idx is not None and mask[codec.DISCARD_BASE + idx]:
            return codec.DISCARD_BASE + idx
        return None

    # ---------------------------------------------------------- shop phase
    def _shop_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        shop = run.shop
        reserve = self._reserve(run)

        use = self._consumable_action(env, mask)
        if use is not None:
            return use

        # Sell decayed tempo jokers when the board is full.
        if shop is not None and len(run.jokers) >= run.joker_slots:
            shop_has_upgrade = any(
                it.kind == "joker" and self._tier(it.item_id, run) >= 4
                for it in shop.card_slots
            )
            if shop_has_upgrade:
                for i, j in enumerate(run.jokers[:codec.N_SELL_SLOTS]):
                    decayed = (
                        (j.id == "j_ice_cream" and j.state.get("chips", 100) <= 40)
                        or (j.id == "j_popcorn" and j.state.get("mult", 20) <= 8)
                        or self._tier(j.id, run) <= 1
                    )
                    idx = codec.SELL_JOKER_BASE + i
                    if decayed and mask[idx]:
                        return idx

        if shop is not None:
            # Buy jokers by tier under the interest reserve.
            buys = []
            for i, it in enumerate(shop.card_slots[:codec.N_BUY_CARD_SLOTS]):
                idx = codec.BUY_CARD_BASE + i
                if not mask[idx]:
                    continue
                if it.kind == "joker":
                    tier = self._tier(it.item_id, run)
                    left = run.money - it.cost
                    affordable = (
                        tier >= 5
                        or (tier == 4 and left >= reserve // 2)
                        or (tier == 3 and left >= reserve)
                        or (tier == 2 and run.ante <= 3 and left >= reserve)
                    )
                    if affordable:
                        buys.append((tier, -it.cost, idx))
                elif it.kind == "planet":
                    spam = self._spam_hand(run)
                    if PLANET_HAND.get(it.item_id) == spam and run.money - it.cost >= reserve:
                        buys.append((4, -it.cost, idx))
                elif it.kind == "tarot" and run.money - it.cost >= reserve + 5:
                    buys.append((1.5, -it.cost, idx))  # below any early joker
            if buys:
                if not run.jokers:
                    # Empty board: tempo doctrine — cheapest joker first, any
                    # tier (scalers are worth ~0 with nothing beside them);
                    # higher tier breaks cost ties.
                    buys.sort(key=lambda b: (b[1], b[0]), reverse=True)
                else:
                    buys.sort(reverse=True)
                return buys[0][2]

            # Voucher: permanent value, but never at the cost of early tempo —
            # only once the interest engine is established.
            if mask[codec.BUY_VOUCHER_IDX] and run.money - 10 >= max(reserve, 15):
                return codec.BUY_VOUCHER_IDX

            # Packs: buffoon early; celestial only when rich.
            for i, offer in enumerate(shop.pack_slots[:codec.N_BUY_PACK_SLOTS]):
                idx = codec.BUY_PACK_BASE + i
                if not mask[idx]:
                    continue
                left = run.money - offer.cost
                if "buffoon" in offer.pack_id and (run.ante <= 3 or left >= reserve):
                    return idx
                if "arcana" in offer.pack_id and left >= reserve + 5:
                    return idx
                if "celestial" in offer.pack_id and left >= reserve + 10:
                    return idx

            # Reroll: hunting for scalers with spare money.
            board_full = len(run.jokers) >= run.joker_slots
            if (
                mask[codec.REROLL_IDX]
                and not board_full
                and run.money - shop.reroll_cost >= max(reserve, 20)
                and shop.reroll_cost <= 7
            ):
                return codec.REROLL_IDX

        if mask[codec.LEAVE_SHOP_IDX]:
            return codec.LEAVE_SHOP_IDX
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))

    # ---------------------------------------------------------- pack phase
    def _pack_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        shop = run.shop
        pack = shop.open_pack if shop else None
        if pack is not None:
            spam = self._spam_hand(run)
            best = (-1, -1)  # (value, idx)
            for i, kind in enumerate(pack.option_kinds[:codec.N_PACK_OPTIONS]):
                idx = codec.PICK_PACK_BASE + i
                if not mask[idx]:
                    continue
                if kind == "joker":
                    value = 10 + self._tier(pack.option_ids[i], run)
                elif kind == "planet":
                    value = 12 if PLANET_HAND.get(pack.option_ids[i]) == spam else 4
                elif kind == "tarot":
                    value = 6 if pack.option_ids[i] in ("c_hermit", "c_temperance") else 5
                else:  # playing card: enhanced ones only
                    card = pack.option_cards[i]
                    value = 3 if (card and str(card.enhancement) != "none") else 1
                if value > best[0]:
                    best = (value, idx)
            if best[1] >= 0 and best[0] >= 3:
                return best[1]
        if mask[codec.SKIP_PACK_IDX]:
            return codec.SKIP_PACK_IDX
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))
