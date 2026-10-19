"""Baseline agents: the benchmark floor for trained policies.

RandomLegalAgent  — uniform over legal actions.
HeuristicAgent    — greedy max-immediate-score; naive shop.
StrategistAgent   — community-strategy agent (see docs/strategy.md):
                    pair-spam + scaling jokers + interest hoarding, aware of
                    the full content catalog (150 jokers, 52 consumables,
                    28 bosses, vouchers, tags, stakes).

The strategist's play-phase evaluation scores a PRUNED candidate set (best
representative of every hand shape present in the hand plus every 1-2 card
play) with the real scoring pipeline on a cloned state, so it stays fast with
12-card hands and is exact for whatever it considers.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from balatro_gym.engine import (
    JOKERS,
    VOUCHERS,
    Edition,
    Enhancement,
    HandType,
    Phase,
    Rarity,
    Suit,
    cardops,
    evaluate_hand,
    score_play,
)
from balatro_gym.engine.levels import hand_value
from balatro_gym.engine.scoring import fold_eval_rules

from . import action_codec as codec


class RandomLegalAgent:
    """Uniform over legal actions."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))


# ---------------------------------------------------------------------------
# Candidate generation shared by the heuristic agents
# ---------------------------------------------------------------------------

def _proxy_value(hr, played, run) -> float:
    chips, mult = hand_value(hr.hand_type, run.hand_levels[hr.hand_type])
    return (chips + sum(c.base_chips for c in hr.scoring_cards)) * mult


def play_candidates(env, mask: np.ndarray, *, max_candidates: int = 48) -> list[tuple[int, tuple[int, ...]]]:
    """Legal play actions worth evaluating: (global_idx, engine_indices).

    Keeps every 1-2 card play, then the best `max_candidates` of the rest by a
    cheap proxy (hand base value plus card chips), de-duplicated by
    (hand type, scoring cards, size)."""
    run = env.run
    rnd = run.round
    rules = fold_eval_rules(run)
    seen: dict = {}
    small: list = []
    for i, subset in enumerate(codec.SUBSETS_5):
        gidx = codec.PLAY_BASE + i
        if not mask[gidx]:
            continue
        engine_idxs = tuple(sorted(env._hand_order[p] for p in subset))
        played = [rnd.hand[j] for j in engine_idxs]
        if len(played) <= 2:
            small.append((gidx, engine_idxs))
            continue
        hr = evaluate_hand(played, rules)
        key = (hr.hand_type, frozenset(c.uid for c in hr.scoring_cards), len(played))
        val = _proxy_value(hr, played, run)
        if key not in seen or seen[key][0] < val:
            seen[key] = (val, gidx, engine_idxs)
    ranked = sorted(seen.values(), key=lambda t: -t[0])[:max_candidates]
    return small + [(g, e) for _, g, e in ranked]


def score_candidate(env, engine_idxs: tuple[int, ...]):
    run = env.run
    clone = run.clone()
    assert clone.round is not None
    played = [clone.round.hand[j] for j in engine_idxs]
    clone.round.hand = [c for j, c in enumerate(clone.round.hand) if j not in set(engine_idxs)]
    return score_play(clone, played, held=clone.round.hand)


class HeuristicAgent:
    """Greedy: plays the max-immediate-score candidate; discards low cards when
    the best play is weak and discards remain; buys cheap jokers in shops."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, mask: np.ndarray) -> int:
        run = env.run
        if run.phase == Phase.PLAYING and env._pending is None:
            return self._play_phase(env, mask)
        if env._pending is not None:
            return int(np.flatnonzero(mask)[0])
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
        for gidx, engine_idxs in play_candidates(env, mask):
            res = score_candidate(env, engine_idxs)
            if res.score > best_score:
                best_score, best_idx = res.score, gidx

        if best_idx is None:
            return int(self.rng.choice(np.flatnonzero(mask)))

        need = rnd.chips_needed - rnd.chips_scored
        weak = best_score < need / max(rnd.hands_left, 1)
        if weak and rnd.discards_left > 0:
            order = env._hand_order
            low = sorted(range(len(order)), key=lambda p: int(rnd.hand[order[p]].rank))[:4]
            subset = tuple(sorted(low[: min(4, len(low))]))
            didx = codec.DISCARD_BASE + codec._SUBSET5_INDEX.get(subset, -10**9)
            if 0 <= didx < codec.N_ACTIONS and mask[didx]:
                return didx
        return best_idx


# ---------------------------------------------------------------------------
# StrategistAgent — codified community strategy (docs/strategy.md)
# ---------------------------------------------------------------------------

# Joker catalog: id -> (shop tier 0-5, ordering class).
# Classes: "add" (+chips/+mult, go LEFT), "x" (xMult, go RIGHT), "copy"
# (Blueprint/Brainstorm), "econ" (money), "passive" (rules/slots/no scoring),
# "retrig" (retriggers, left of xMult). Tiers follow docs/strategy.md:
# 5 = always buy, 4 = buy if half the reserve survives, 3 = decent, 2 = tempo
# (antes 1-3 only), 1 = situational, 0 = avoid.
JOKER_INFO: dict[str, tuple[int, str]] = {
    # --- flat / contains-hand adders ---
    "j_joker": (3, "add"), "j_jolly": (3, "add"), "j_zany": (2, "add"), "j_mad": (2, "add"),
    "j_crazy": (2, "add"), "j_droll": (2, "add"), "j_sly": (4, "add"), "j_wily": (2, "add"),
    "j_clever": (2, "add"), "j_devious": (2, "add"), "j_crafty": (2, "add"), "j_half": (4, "add"),
    "j_abstract": (3, "add"), "j_banner": (4, "add"), "j_mystic_summit": (3, "add"),
    "j_misprint": (3, "add"), "j_scholar": (3, "add"), "j_odd_todd": (3, "add"),
    "j_even_steven": (3, "add"), "j_fibonacci": (4, "add"), "j_scary_face": (3, "add"),
    "j_smiley": (3, "add"), "j_walkie_talkie": (3, "add"), "j_arrowhead": (3, "add"),
    "j_onyx_agate": (3, "add"), "j_greedy": (2, "add"), "j_lusty": (2, "add"),
    "j_wrathful": (2, "add"), "j_gluttonous": (2, "add"), "j_raised_fist": (2, "add"),
    "j_shoot_the_moon": (2, "add"), "j_blue_joker": (3, "add"), "j_stone": (2, "add"),
    "j_erosion": (2, "add"), "j_bull": (4, "add"), "j_bootstraps": (3, "add"),
    "j_swashbuckler": (2, "add"), "j_stuntman": (3, "add"), "j_ice_cream": (2, "add"),
    "j_popcorn": (2, "add"), "j_gros_michel": (2, "add"), "j_cavendish": (4, "x"),
    # --- scalers (additive) ---
    "j_supernova": (5, "add"), "j_green_joker": (5, "add"), "j_ride_the_bus": (5, "add"),
    "j_castle": (5, "add"), "j_runner": (3, "add"), "j_square": (3, "add"), "j_trousers": (3, "add"),
    "j_wee": (4, "add"), "j_flash": (2, "add"), "j_red_card": (2, "add"),
    "j_ceremonial": (2, "add"), "j_fortune_teller": (3, "add"), "j_hiker": (3, "add"),
    # --- xMult ---
    "j_duo": (5, "x"), "j_trio": (3, "x"), "j_family": (2, "x"), "j_order": (2, "x"),
    "j_tribe": (3, "x"), "j_photograph": (4, "x"), "j_baron": (3, "x"), "j_bloodstone": (3, "x"),
    "j_ancient": (3, "x"), "j_baseball": (1, "x"), "j_stencil": (2, "x"), "j_loyalty_card": (3, "x"),
    "j_steel_joker": (2, "x"), "j_acrobat": (3, "x"), "j_card_sharp": (4, "x"),
    "j_blackboard": (2, "x"), "j_flower_pot": (2, "x"), "j_seeing_double": (2, "x"),
    "j_drivers_license": (2, "x"), "j_idol": (2, "x"), "j_triboulet": (5, "x"),
    "j_obelisk": (4, "x"), "j_constellation": (5, "x"), "j_vampire": (4, "x"), "j_hologram": (5, "x"),
    "j_lucky_cat": (3, "x"), "j_throwback": (3, "x"), "j_campfire": (5, "x"), "j_glass": (3, "x"),
    "j_hit_the_road": (2, "x"), "j_madness": (3, "x"), "j_ramen": (3, "x"), "j_yorick": (4, "x"),
    "j_canio": (4, "x"),
    # --- copies / retriggers ---
    "j_blueprint": (5, "copy"), "j_brainstorm": (5, "copy"),
    "j_hack": (4, "retrig"), "j_dusk": (3, "retrig"), "j_mime": (3, "retrig"),
    "j_hanging_chad": (4, "retrig"), "j_sock_and_buskin": (3, "retrig"), "j_seltzer": (3, "retrig"),
    # --- economy ---
    "j_golden": (4, "econ"), "j_egg": (2, "econ"), "j_business": (2, "econ"), "j_mail_in": (2, "econ"),
    "j_credit_card": (1, "econ"), "j_delayed_grat": (2, "econ"), "j_faceless": (1, "econ"),
    "j_cloud_9": (2, "econ"), "j_rocket": (4, "econ"), "j_gift": (3, "econ"),
    "j_reserved_parking": (2, "econ"), "j_to_the_moon": (3, "econ"), "j_satellite": (3, "econ"),
    "j_trading": (1, "econ"), "j_ticket": (1, "econ"), "j_rough_gem": (2, "econ"),
    "j_matador": (1, "econ"), "j_todo_list": (2, "econ"), "j_diet_cola": (2, "econ"),
    "j_luchador": (2, "econ"), "j_chaos": (2, "econ"), "j_astronomer": (3, "econ"),
    "j_perkeo": (5, "econ"), "j_8ball": (2, "econ"), "j_vagabond": (3, "econ"),
    "j_seance": (1, "econ"), "j_superposition": (2, "econ"), "j_sixth_sense": (1, "econ"),
    "j_cartomancer": (3, "econ"), "j_hallucination": (3, "econ"), "j_riff_raff": (2, "econ"),
    "j_space": (4, "econ"), "j_burnt": (3, "econ"), "j_dna": (1, "passive"),
    # --- passives ---
    "j_four_fingers": (1, "passive"), "j_shortcut": (1, "passive"), "j_smeared": (1, "passive"),
    "j_splash": (1, "passive"), "j_pareidolia": (0, "passive"), "j_showman": (1, "passive"),
    "j_oops": (2, "passive"), "j_juggler": (2, "passive"), "j_drunkard": (2, "passive"),
    "j_troubadour": (2, "passive"), "j_merry_andy": (1, "passive"), "j_turtle_bean": (1, "passive"),
    "j_marble": (1, "passive"), "j_certificate": (2, "passive"), "j_burglar": (3, "passive"),
    "j_mr_bones": (2, "passive"), "j_invisible": (3, "passive"), "j_chicot": (5, "passive"),
}

INTEREST_CAP_MONEY = 25  # $ needed for max interest (base)

# Planet id -> hand it levels (for spending planets on the spam hand).
PLANET_HAND = {
    "c_pluto": HandType.HIGH_CARD, "c_mercury": HandType.PAIR,
    "c_uranus": HandType.TWO_PAIR, "c_venus": HandType.THREE_KIND,
    "c_saturn": HandType.STRAIGHT, "c_jupiter": HandType.FLUSH,
    "c_earth": HandType.FULL_HOUSE, "c_mars": HandType.FOUR_KIND,
    "c_neptune": HandType.STRAIGHT_FLUSH, "c_planet_x": HandType.FIVE_KIND,
    "c_ceres": HandType.FLUSH_HOUSE, "c_eris": HandType.FLUSH_FIVE,
}

# Tags worth skipping a Small Blind for (value 0-10), see docs/strategy.md.
TAG_VALUE: dict[str, float] = {
    "t_negative": 10, "t_polychrome": 8, "t_rare": 8, "t_holo": 6, "t_foil": 5,
    "t_voucher": 6, "t_charm": 6, "t_buffoon": 6, "t_meteor": 4, "t_ethereal": 6,
    "t_standard": 2, "t_top_up": 5, "t_double": 5, "t_investment": 6, "t_economy": 5,
    "t_coupon": 5, "t_d6": 3, "t_uncommon": 6, "t_boss": 2, "t_handy": 3,
    "t_garbage": 2, "t_juggle": 1, "t_speed": 2, "t_orbital": 4,
}

# Voucher priority (higher first). Grabber/Wasteful tempo, Overstock & Telescope
# build value, Seed Money / Money Tree economy.
VOUCHER_VALUE: dict[str, float] = {
    "v_overstock": 6, "v_overstock_plus": 5, "v_clearance_sale": 5, "v_liquidation": 6,
    "v_hone": 4, "v_glow_up": 4, "v_reroll_surplus": 5, "v_reroll_glut": 5,
    "v_crystal_ball": 4, "v_omen_globe": 2, "v_telescope": 5, "v_observatory": 6,
    "v_grabber": 6, "v_nacho_tong": 5, "v_wasteful": 4, "v_recyclomancy": 3,
    "v_tarot_merchant": 4, "v_tarot_tycoon": 3, "v_planet_merchant": 4, "v_planet_tycoon": 3,
    "v_seed_money": 6, "v_money_tree": 6, "v_blank": 1, "v_antimatter": 8,
    "v_magic_trick": 2, "v_illusion": 3, "v_hieroglyph": 2, "v_petroglyph": 1,
    "v_directors_cut": 4, "v_retcon": 4, "v_paint_brush": 7, "v_palette": 6,
}

# Bosses that hurt a joker/pair-spam build (reroll candidates with Director's Cut).
HARD_BOSSES = {"b_needle": 4, "b_wall": 3, "b_ox": 3, "b_psychic": 1, "b_eye": 3,
               "b_mouth": 1, "b_plant": 1, "b_pillar": 2, "b_flint": 3, "b_arm": 3,
               "b_verdant_leaf": 2, "b_crimson_heart": 3, "b_amber_acorn": 2, "b_violet_vessel": 4}


def _classify(jid: str) -> str:
    return JOKER_INFO.get(jid, (1, "add"))[1]


class StrategistAgent(HeuristicAgent):
    """Pair-spam scaling build with interest discipline.

    Play phase: greedy immediate score over a pruned candidate set, EXCEPT
      - prefers the run's spam hand (most-played) when within 25% of best,
      - avoids scoring face cards when Ride the Bus is owned (if close),
      - discards only truly dead hands, never when Green Joker is owned
        unless desperate; discard choice keeps pairs / flush / straight draws.
    Blind select: skips Small Blinds for premium tags when the board is safe,
      rerolls run-killing bosses with Director's Cut, orders jokers
      (+mult before xMult, copies next to their target).
    Shop: tiered joker buying under an interest reserve, decay selling,
      vouchers by priority, packs by type, targeted rerolls, consumable use.
    """

    # ------------------------------------------------------------- helpers
    def _reserve(self, run) -> int:
        """Money we refuse to dip below (interest hoarding)."""
        if run.mods.get("no_interest"):
            return 0
        if run.ante <= 3:
            return 0
        if run.ante <= 5:
            return 5
        return 10

    def _tier(self, jid: str, run) -> int:
        t = JOKER_INFO.get(jid, (1, "add"))[0]
        if t == 2 and run.ante > 3 and jid in ("j_ice_cream", "j_popcorn", "j_gros_michel",
                                                 "j_greedy", "j_lusty", "j_wrathful", "j_gluttonous"):
            t = 1  # tempo jokers fall off
        if jid in ("j_blueprint", "j_brainstorm"):
            copyable = any(
                JOKER_INFO.get(j.id, (1, "add"))[0] >= 4 and JOKERS[j.id].copyable
                for j in run.jokers
            )
            t = 5 if copyable else 1
        if jid == "j_baseball":
            uncommons = sum(1 for j in run.jokers if JOKERS[j.id].rarity == Rarity.UNCOMMON)
            t = 4 if uncommons >= 2 else 1
        if jid == "j_pareidolia" and self._owned(run, "j_ride_the_bus"):
            t = 0
        if jid in ("j_steel_joker", "j_stone", "j_drivers_license", "j_glass"):
            n = sum(1 for c in run.full_deck if c.enhancement != Enhancement.NONE)
            t = 3 if n >= 8 else t
        if jid == "j_stencil":
            t = 4 if run.joker_slots - len(run.jokers) >= 3 else 1
        if jid == "j_cavendish" and run.stats.get("gros_michel_extinct"):
            t = 4
        kind = _classify(jid)
        scoring = sum(1 for j in run.jokers if _classify(j.id) in ("add", "x", "retrig", "copy"))
        if kind == "econ" and (scoring >= 2 or run.ante >= 4):
            t = min(t, 1)
        if kind == "passive" and jid not in ("j_chicot", "j_burglar", "j_oops"):
            t = min(t, 1)
        return t

    def _scoring_value(self, jid: str, run) -> float:
        """Shop desirability: tier plus a bonus for the classes that win runs."""
        t = self._tier(jid, run)
        kind = _classify(jid)
        bonus = {"x": 1.0, "add": 0.5, "retrig": 0.5, "copy": 0.5}.get(kind, 0.0)
        return t + bonus

    def _spam_hand(self, run) -> HandType:
        counts = run.hand_play_counts
        best = max(counts, key=lambda ht: (counts.get(ht, 0), -int(ht)))
        return best if counts.get(best, 0) > 0 else HandType.PAIR

    def _planet_value(self, run, cid: str) -> float:
        """How much a planet is worth to this run: levels on the hands we
        actually play; nothing for hands we never play."""
        ht = PLANET_HAND.get(cid)
        if ht is None:
            return 0.0
        counts = run.hand_play_counts
        total = max(1, sum(counts.values()))
        share = counts.get(ht, 0) / total
        if ht == self._spam_hand(run):
            return 12.0
        if share >= 0.25:
            return 6.0
        return 2.0 + 4.0 * share

    def _has_xmult(self, run) -> bool:
        return any(_classify(j.id) == "x" for j in run.jokers)

    def _owned(self, run, jid: str) -> bool:
        return any(j.id == jid for j in run.jokers)

    def _board_strength(self, run) -> float:
        """Rough score-per-hand estimate from jokers + spam hand level."""
        spam = self._spam_hand(run)
        chips, mult = hand_value(spam, run.hand_levels[spam])
        chips += 30
        for j in run.jokers:
            kind = _classify(j.id)
            tier = JOKER_INFO.get(j.id, (1, "add"))[0]
            if kind == "add":
                mult += 4 + 2 * tier
                for v in j.state.values():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        mult += min(float(v), 60)
            elif kind == "x":
                x = 1.5
                for v in j.state.values():
                    if isinstance(v, float) and v > 1:
                        x = max(x, v)
                mult *= x
            elif kind == "retrig":
                chips *= 1.5
        return chips * mult

    # ------------------------------------------------------------ main act
    def act(self, env, obs, mask: np.ndarray) -> int:
        run = env.run
        if env._pending is not None:
            return self._target_phase(env, mask)
        if run.phase == Phase.PLAYING:
            use = self._consumable_action(env, mask)
            if use is not None:
                return use
            return self._play_phase(env, mask)
        if run.phase == Phase.SHOP:
            return self._shop_phase(env, mask)
        if run.phase == Phase.PACK_OPEN:
            return self._pack_phase(env, mask)
        if run.phase == Phase.BLIND_SELECT:
            return self._blind_phase(env, mask)
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))

    # ----------------------------------------------------------- ordering
    def _order_key(self, run, idx: int) -> float:
        j = run.jokers[idx]
        kind = _classify(j.id)
        base = {"passive": 0, "econ": 1, "add": 2, "retrig": 3, "copy": 4, "x": 5}[kind]
        if j.id == "j_brainstorm":
            return -1.0  # Brainstorm copies the leftmost: put the best joker first instead
        if j.id == "j_blueprint":
            return 4.5  # directly left of the first xMult joker
        return float(base)

    def _swap_action(self, env, mask: np.ndarray) -> int | None:
        """One adjacent swap toward the target ordering, if any is out of place."""
        run = env.run
        n = min(len(run.jokers), codec.N_SELL_SLOTS)
        keys = [self._order_key(run, i) for i in range(n)]
        start = 0
        # Brainstorm copies the leftmost joker: move the best (highest-tier)
        # one there, then keep the rest in tier order without touching slot 0
        # (sorting slot 0 as well would swap the best joker back out forever).
        if any(j.id == "j_brainstorm" for j in run.jokers[:n]) and n > 1:
            best = max(range(n), key=lambda i: (JOKER_INFO.get(run.jokers[i].id, (1, "add"))[0]
                                                 if run.jokers[i].id != "j_brainstorm" else -1))
            if best > 0 and mask[codec.SWAP_JOKER_BASE + best - 1]:
                return codec.SWAP_JOKER_BASE + best - 1
            start = 1
        for i in range(start, n - 1):
            if keys[i] > keys[i + 1] and mask[codec.SWAP_JOKER_BASE + i]:
                return codec.SWAP_JOKER_BASE + i
        return None

    # --------------------------------------------------------- blind phase
    def _blind_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        swap = self._swap_action(env, mask)
        if swap is not None:
            return swap
        use = self._consumable_action(env, mask)
        if use is not None:
            return use
        # Boss reroll (Director's Cut / Retcon) for run-killing bosses.
        if mask[codec.REROLL_BOSS_IDX] and run.ante_boss in HARD_BOSSES:
            if HARD_BOSSES[run.ante_boss] >= 3 and run.money - 10 >= self._reserve(run) + 10:
                return codec.REROLL_BOSS_IDX
        # Skip for premium tags when the board comfortably beats the next blind.
        if mask[codec.SKIP_BLIND] and run.blind_index < 2:
            tag = run.blind_tags[run.blind_index] if run.blind_index < len(run.blind_tags) else None
            value = TAG_VALUE.get(tag.id, 0) if tag else 0
            if tag and tag.id == "t_double":
                nxt = run.blind_tags[1] if run.blind_index == 0 else None
                value = TAG_VALUE.get(nxt.id, 0) if nxt else 2
            threshold = 6 if run.blind_index == 0 else 8
            if value >= threshold and run.ante >= 2:
                from balatro_gym.engine.levels import ante_base_chips

                need = ante_base_chips(run.ante, run.stake) * run.mods.get("ante_scaling", 1) * 2.0
                if self._board_strength(run) * max(cardops.effective_hands(run), 1) >= 2.5 * need:
                    return codec.SKIP_BLIND
        if mask[codec.SELECT_BLIND]:
            return codec.SELECT_BLIND
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))

    # --------------------------------------------------------- consumables
    def _consumable_value(self, env, inst) -> float:
        """How much we want to use this consumable right now (<= 0: hold)."""
        run = env.run
        cid = inst.id
        self._spam_hand(run)
        slots_full = cardops.consumables_used_slots(run) >= run.consumable_slots
        clog = 1.0 if slots_full else 0.0
        if cid in PLANET_HAND:
            pv = self._planet_value(run, cid)
            return 5.0 if pv >= 6 else clog
        table = {
            "c_hermit": 4.0 if run.money >= 15 else (clog if run.money >= 5 else 0.0),
            "c_temperance": 4.0 if sum(j.sell_value for j in run.jokers) >= 8 else clog,
            "c_high_priestess": 3.0, "c_emperor": 3.0, "c_fool": 3.0, "c_judgement": 3.0,
            "c_wheel": 3.0, "c_black_hole": 9.0, "c_soul": 9.0, "c_wraith": 4.0 if run.money <= 5 else clog,
            "c_immolate": 2.0 if run.phase == Phase.PLAYING and run.round and run.round.hands_left >= 2 else 0.0,
            "c_ectoplasm": 3.0 if cardops.effective_hand_size(run) >= 8 and run.jokers else 0.0,
            "c_aura": 3.0, "c_talisman": 2.0, "c_deja_vu": 3.0, "c_trance": 2.0, "c_medium": 2.0,
            "c_cryptid": 2.0, "c_familiar": 1.5, "c_grim": 1.5, "c_incantation": 1.0,
            "c_empress": 3.0, "c_hierophant": 3.0, "c_magician": 2.0, "c_lovers": 2.0,
            "c_chariot": 2.0, "c_justice": 1.5, "c_devil": 1.5, "c_tower": 1.0,
            "c_strength": 1.0, "c_death": 2.0, "c_hanged_man": 1.5 if run.ante <= 4 else clog,
            "c_star": 1.0, "c_sun": 1.0, "c_moon": 1.0, "c_world": 1.0,
            "c_sigil": clog, "c_ouija": clog, "c_ankh": clog, "c_hex": clog,
        }
        v = table.get(cid, clog)
        if cid in ("c_ankh", "c_hex") and len(run.jokers) == 1 and JOKER_INFO.get(run.jokers[0].id, (1, "add"))[0] >= 4:
            v = 4.0
        return v

    def _consumable_action(self, env, mask: np.ndarray) -> int | None:
        run = env.run
        best = (0.0, None)
        for i, inst in enumerate(run.consumables[:codec.N_CONSUMABLE_SLOTS]):
            idx = codec.USE_CONSUMABLE_BASE + i
            if not mask[idx]:
                continue
            v = self._consumable_value(env, inst)
            if v > best[0]:
                best = (v, idx)
        if best[1] is not None:
            return best[1]
        # Sell junk when every slot is clogged (frees room for packs / 8-Ball).
        if cardops.consumables_used_slots(run) >= run.consumable_slots:
            for i, inst in enumerate(run.consumables[:codec.N_CONSUMABLE_SLOTS]):
                idx = codec.SELL_CONSUMABLE_BASE + i
                if mask[idx] and self._consumable_value(env, inst) <= 1.0:
                    return idx
        return None

    # ------------------------------------------------------------ targets
    def _target_phase(self, env, mask: np.ndarray) -> int:
        """Choose targets for the pending consumable / pack pick."""
        run = env.run
        pkind, index, lo, hi = env._pending
        cid = run.consumables[index].id if pkind == "consumable" else run.open_pack.option_ids[index]
        hand = cardops.current_hand(run)
        order = env._hand_order
        n = min(len(order), codec.HAND_SLOTS)
        self._spam_hand(run)
        counts = Counter(hand[order[p]].rank for p in range(n))

        def card(p):
            return hand[order[p]]

        def want_rank(p):
            c = card(p)
            return (counts[c.rank] >= 2, int(c.rank))

        slots = list(range(n))
        if cid in ("c_hanged_man",):
            pick = sorted(slots, key=lambda p: (counts[card(p).rank] >= 2, int(card(p).rank)))[:hi]
        elif cid == "c_death":
            # left becomes right: left = worst card, right = best card
            ranked = sorted(slots, key=want_rank)
            pick = [ranked[0], ranked[-1]] if n >= 2 else slots[:2]
        elif cid in ("c_star", "c_sun", "c_moon", "c_world"):
            target_suit = {"c_star": Suit.DIAMONDS, "c_sun": Suit.HEARTS, "c_moon": Suit.CLUBS,
                           "c_world": Suit.SPADES}[cid]
            pick = sorted((p for p in slots if card(p).suit != target_suit), key=lambda p: -int(card(p).rank))[:hi]
            if not pick:
                pick = slots[:lo]
        elif cid == "c_strength":
            pick = sorted(slots, key=lambda p: (-counts[card(p).rank], -int(card(p).rank)))[:hi]
        elif cid in ("c_chariot", "c_lovers"):
            pick = sorted(slots, key=lambda p: -int(card(p).rank))[:hi]
        else:
            # Enhancements / seals / editions: cards we play the most (paired or spam rank).
            pick = sorted(
                (p for p in slots if card(p).enhancement == Enhancement.NONE or cid in ("c_aura", "c_talisman", "c_deja_vu", "c_trance", "c_medium", "c_cryptid")),
                key=lambda p: (-(counts[card(p).rank] >= 2), -int(card(p).rank)),
            )[:hi] or slots[:hi]
        pick = tuple(sorted(pick[:hi]))
        while len(pick) < lo and n > len(pick):
            pick = tuple(sorted(set(pick) | {next(p for p in slots if p not in pick)}))
        idx = codec.TARGET_BASE + codec._SUBSET3_INDEX.get(pick, -10**9)
        if 0 <= idx < codec.N_ACTIONS and mask[idx]:
            return idx
        legal = np.flatnonzero(mask)
        return int(legal[0]) if len(legal) else 0

    # ---------------------------------------------------------- play phase
    def _play_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        rnd = run.round
        assert rnd is not None
        spam = self._spam_hand(run)
        has_rtb = self._owned(run, "j_ride_the_bus")
        has_green = self._owned(run, "j_green_joker")
        scaler_owned = any(
            self._owned(run, jid)
            for jid in ("j_supernova", "j_green_joker", "j_ride_the_bus", "j_castle", "j_obelisk")
        )
        len(run.jokers) >= 2

        candidates = []  # (global_idx, score, hand_type, scores_face, n_played)
        for gidx, engine_idxs in play_candidates(env, mask):
            res = score_candidate(env, engine_idxs)
            scores_face = any(
                c.rank.is_face for c in res.hand_result.scoring_cards if not c.is_stone
            )
            candidates.append((gidx, res.score, res.hand_result.hand_type, scores_face, len(engine_idxs)))

        if not candidates:
            legal = np.flatnonzero(mask)
            return int(self.rng.choice(legal))

        best_score = max(c[1] for c in candidates)
        need = rnd.chips_needed - rnd.chips_scored

        def pick_value(c):
            gidx, score, ht, face, n = c
            v = score
            if score < need:  # only shape when not immediately lethal
                if scaler_owned and ht == spam:
                    v *= 1.25  # spam-hand preference (Supernova/level scaling)
                if has_rtb and face:
                    v *= 0.6   # don't reset Ride the Bus
                if self._owned(run, "j_obelisk") and ht == spam:
                    v *= 0.8
            return v

        gidx, score, ht, face, n = max(candidates, key=pick_value)

        # Discard when the best play is off the pace needed to clear the blind
        # (never when it wins outright). Green Joker makes discards costly, so
        # only dig when clearly behind.
        pace = need / max(rnd.hands_left, 1)
        weak_factor = 0.5 if has_green else 1.0
        weak = best_score < need and best_score < pace * weak_factor
        if weak and rnd.discards_left > 0 and rnd.hands_left >= 1:
            didx = self._discard_action(env, mask)
            if didx is not None:
                return didx
        return gidx

    def _discard_action(self, env, mask: np.ndarray) -> int | None:
        """Discard cards that don't help the strongest draw (pairs, 4-flush,
        open straight), lowest first; joker-aware for Castle/Mail-In/Yorick."""
        run = env.run
        rnd = run.round
        assert rnd is not None
        order = env._hand_order
        n = min(len(order), codec.HAND_SLOTS)
        cards = [rnd.hand[order[p]] for p in range(n)]
        counts = Counter(c.rank for c in cards)
        suits = Counter(c.suit for c in cards if not c.is_stone)
        keep: set[int] = set()
        if counts and max(counts.values()) >= 2:
            keep |= {p for p in range(n) if counts[cards[p].rank] >= 2}
        if suits and max(suits.values()) >= 4:
            s = max(suits, key=suits.get)
            keep |= {p for p in range(n) if cards[p].suit == s}
        castle_suit = next((Suit(j.state["suit"]) for j in run.jokers if j.id == "j_castle"), None)
        mail_rank = next((j.state["rank"] for j in run.jokers if j.id == "j_mail_in"), None)
        pool = [p for p in range(n) if p not in keep]

        def discard_pref(p):
            c = cards[p]
            score = int(c.rank)
            if castle_suit is not None and c.suit == castle_suit:
                score -= 20
            if mail_rank is not None and int(c.rank) == mail_rank:
                score -= 20
            if self._owned(run, "j_hit_the_road") and int(c.rank) == 11:
                score -= 20
            return score

        pool.sort(key=discard_pref)
        pick = pool[:5]
        if not pick:  # everything is paired: discard the lowest pair's cards
            low = sorted(range(n), key=lambda p: int(cards[p].rank))
            pick = low[:2]
        if self._owned(run, "j_faceless"):
            faces = [p for p in range(n) if cards[p].rank.is_face and p not in keep]
            if len(faces) >= 3:
                pick = faces[:5]
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

        swap = self._swap_action(env, mask)
        if swap is not None:
            return swap
        use = self._consumable_action(env, mask)
        if use is not None:
            return use

        def can_afford_now(cost: int) -> bool:
            return run.money + cardops.debt_allowance(run) >= cost

        if shop is not None:
            # Sell decayed / expired / rental-drain jokers when an upgrade is on offer
            # or the joker is dead weight.
            board_full = len(run.jokers) >= run.joker_slots
            worst = min((self._scoring_value(j.id, run) for j in run.jokers), default=99)
            shop_has_upgrade = any(
                it.kind == "joker" and can_afford_now(it.cost)
                and self._scoring_value(it.item_id, run) >= max(4, worst + 1.5)
                for it in shop.card_slots
            )
            for i, j in enumerate(run.jokers[:codec.N_SELL_SLOTS]):
                idx = codec.SELL_JOKER_BASE + i
                if not mask[idx]:
                    continue
                decayed = (
                    (j.id == "j_ice_cream" and j.state.get("chips", 100) <= 40)
                    or (j.id == "j_popcorn" and j.state.get("mult", 20) <= 8)
                    or (j.id == "j_turtle_bean" and j.state.get("h", 5) <= 1)
                    or j.debuffed
                    or (board_full and shop_has_upgrade and self._scoring_value(j.id, run) == worst)
                    or ("rental" in j.stickers and self._tier(j.id, run) <= 2)
                    or (j.id == "j_invisible" and j.state.get("rounds", 0) >= 2 and len(run.jokers) >= 2)
                    or (j.id == "j_diet_cola" and run.ante >= 2)
                )
                if decayed:
                    return idx

            buys = []
            for i, it in enumerate(shop.card_slots[:codec.N_BUY_CARD_SLOTS]):
                idx = codec.BUY_CARD_BASE + i
                if not mask[idx]:
                    continue
                left = run.money - it.cost
                if it.kind == "joker":
                    tier = self._tier(it.item_id, run)
                    if "perishable" in it.stickers and _classify(it.item_id) == "add":
                        tier = min(tier, 2)
                    if "eternal" in it.stickers and tier <= 2:
                        continue
                    value = self._scoring_value(it.item_id, run)
                    if it.edition != Edition.NONE:
                        value += 0.5
                    kind = _classify(it.item_id)
                    if kind == "x" and tier >= 3 and not self._has_xmult(run):
                        tier, value = 5, value + 1.0  # first xMult is the build's backbone
                    affordable = (
                        tier >= 5
                        or it.cost == 0
                        or (tier == 4 and left >= reserve // 2)
                        or (tier == 3 and left >= reserve)
                        or (tier == 2 and run.ante <= 3 and left >= reserve)
                        or (tier >= 2 and len(run.jokers) < 2 and kind in ("add", "x", "retrig"))
                    )
                    if affordable and len(run.jokers) < run.joker_slots:
                        buys.append((value, -it.cost, idx))
                elif it.kind == "planet":
                    pv = self._planet_value(run, it.item_id)
                    if pv >= 6 and left >= min(reserve, 5):
                        buys.append((4.5 if pv >= 12 else 3.5, -it.cost, idx))
                    elif it.cost == 0:
                        buys.append((2, 0, idx))
                elif it.kind in ("tarot", "spectral"):
                    want = self._consumable_value(env, cardops.new_consumable(run, it.item_id))
                    if want >= 3 and left >= reserve:
                        buys.append((2.5, -it.cost, idx))
                    elif it.cost == 0:
                        buys.append((2, 0, idx))
                elif it.kind == "card" and it.card is not None:
                    good = it.card.enhancement != Enhancement.NONE or it.card.seal.value != "none"
                    if good and left >= reserve + 5:
                        buys.append((1.5, -it.cost, idx))
            if buys:
                if not run.jokers:
                    buys.sort(key=lambda b: (b[1], b[0]), reverse=True)
                else:
                    buys.sort(reverse=True)
                return buys[0][2]

            # Vouchers: permanent value once the interest engine is established.
            vouchers = []
            for i, vid in enumerate(shop.vouchers[:codec.N_VOUCHER_SLOTS]):
                idx = codec.BUY_VOUCHER_BASE + i
                if mask[idx]:
                    vouchers.append((VOUCHER_VALUE.get(vid, 2), idx))
            if vouchers:
                value, idx = max(vouchers)
                cost = VOUCHERS[shop.vouchers[idx - codec.BUY_VOUCHER_BASE]].cost
                if value >= 5 and run.money - cost >= reserve + 10:
                    return idx

            # Packs: buffoon early; arcana when rich; celestial only for a spam hand.
            for i, offer in enumerate(shop.pack_slots[:codec.N_BUY_PACK_SLOTS]):
                idx = codec.BUY_PACK_BASE + i
                if not mask[idx]:
                    continue
                left = run.money - offer.cost
                pid = offer.pack_id
                if offer.cost == 0:
                    return idx
                if "buffoon" in pid and left >= reserve and len(run.jokers) < run.joker_slots:
                    return idx
                if "celestial" in pid and left >= reserve and run.stats.get("hands_played_run", 0) >= 4:
                    return idx
                if "spectral" in pid and left >= reserve + 5:
                    return idx
                if "arcana" in pid and left >= reserve + 5:
                    return idx
                if "standard" in pid and left >= reserve + 15 and run.ante >= 3:
                    return idx

            # Reroll: hunting for scalers / xMult with spare money. A full
            # board still rerolls when its weakest joker is replaceable.
            board_full = len(run.jokers) >= run.joker_slots
            cost = 0 if shop.free_rerolls > 0 else shop.reroll_cost
            weak_board = worst < 4.5 or not self._has_xmult(run)
            if mask[codec.REROLL_IDX] and (
                cost == 0
                or (not board_full and run.money - cost >= max(reserve, 12) and cost <= 7)
                or (board_full and weak_board and run.money - cost >= reserve + 20 and cost <= 8)
            ):
                return codec.REROLL_IDX

        if mask[codec.LEAVE_SHOP_IDX]:
            return codec.LEAVE_SHOP_IDX
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))

    # ---------------------------------------------------------- pack phase
    def _pack_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        pack = run.open_pack
        if pack is not None:
            self._spam_hand(run)
            best = (-1.0, -1)  # (value, idx)
            for i, kind in enumerate(pack.option_kinds[:codec.N_PACK_OPTIONS]):
                idx = codec.PICK_PACK_BASE + i
                if not mask[idx]:
                    continue
                if kind == "joker":
                    value = 10 + self._tier(pack.option_ids[i], run)
                    if i < len(pack.option_editions) and pack.option_editions[i] != "none":
                        value += 1
                elif kind == "planet":
                    value = self._planet_value(run, pack.option_ids[i])
                elif kind in ("tarot", "spectral"):
                    value = 3 + self._consumable_value(env, cardops.new_consumable(run, pack.option_ids[i]))
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


# Backwards-compatible alias: tier lookup table.
JOKER_TIERS: dict[str, int] = {k: v[0] for k, v in JOKER_INFO.items()}
