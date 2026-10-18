"""PlannerAgent: a simulation-based agent built on the strategist.

Every decision that shapes the board (what to buy, sell, level, reorder,
whether to skip or reroll a boss) is scored by *measuring* the board's
hand-scoring power on cloned engine state: deal K random hands from the
run's deck, find the best play for each with the real scoring pipeline and
average the result. Scaling jokers are evaluated on a projected state (what
they will look like a few rounds from now), so the agent still invests in
them early. Play-phase decisions keep the strategist's fast greedy policy
with pace-based discarding, since those have to be made ~100 times per run.
"""

from __future__ import annotations

import random

import numpy as np

from balatro_gym.engine import (
    VOUCHERS,
    Edition,
    HandType,
    Phase,
    cardops,
    evaluate_hand,
    score_play,
)
from balatro_gym.engine.levels import ante_base_chips, hand_value
from balatro_gym.engine.scoring import fold_eval_rules
from balatro_gym.engine.state import RoundState

from . import action_codec as codec
from .baselines import JOKER_INFO, PLANET_HAND, TAG_VALUE, VOUCHER_VALUE, StrategistAgent, _classify

# Projected growth per round for scaling jokers (state key -> per-round delta),
# used when valuing a purchase. Rough community-consensus rates for a
# spam-hand build (4 hands / round, ~3 discards).
GROWTH: dict[str, dict[str, float]] = {
    "j_green_joker": {"mult": 3.0},
    "j_ride_the_bus": {"mult": 2.5},
    "j_castle": {"chips": 9.0},
    "j_wee": {"chips": 12.0},
    "j_runner": {"chips": 4.0},
    "j_square": {"chips": 3.0},
    "j_trousers": {"mult": 1.5},
    "j_constellation": {"x": 0.12},
    "j_hologram": {"x": 0.15},
    "j_vampire": {"x": 0.08},
    "j_lucky_cat": {"x": 0.05},
    "j_glass": {"x": 0.05},
    "j_campfire": {"x": 0.4},
    "j_obelisk": {"x": 0.3},
    "j_yorick": {"x": 0.15},
    "j_madness": {"x": 0.6},
    "j_ceremonial": {"mult": 1.0},
    "j_flash": {"mult": 1.0},
    "j_red_card": {"mult": 0.5},
    "j_hit_the_road": {"x": 0.2},
    "j_rocket": {"dollars": 0.7},
    "j_loyalty_card": {},
    "j_egg": {},
}
PROJECT_ROUNDS = 6

# Expected dollars per round from economy jokers (rough averages for a
# 4-hand round); converted into a power-equivalent bonus when valuing a buy.
ECON_DOLLARS: dict[str, float] = {
    "j_golden": 4.0, "j_rocket": 3.0, "j_to_the_moon": 2.5, "j_egg": 3.0, "j_gift": 2.0,
    "j_cloud_9": 1.5, "j_delayed_grat": 3.0, "j_reserved_parking": 1.0, "j_business": 1.5,
    "j_satellite": 1.0, "j_mail_in": 2.0, "j_faceless": 0.5, "j_rough_gem": 1.5,
    "j_ticket": 0.5, "j_matador": 1.0, "j_todo_list": 1.0, "j_credit_card": 1.0,
    "j_trading": 0.8, "j_cartomancer": 1.0, "j_vagabond": 0.3, "j_8ball": 0.5,
    "j_hallucination": 0.5, "j_riff_raff": 1.5, "j_chaos": 1.5, "j_astronomer": 1.5,
}
DOLLAR_TO_POWER = 0.012  # +$1 of expected income ~ +1.2% power (a $5 joker ~ +6%)


def rounds_left(run) -> int:
    return max(1, (8 - run.ante) * 3 + (2 - min(run.blind_index, 2)))


class PlannerAgent(StrategistAgent):
    COMMIT_CANDIDATES = (HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_KIND,
                         HandType.FLUSH, HandType.HIGH_CARD, HandType.FULL_HOUSE)

    def __init__(self, seed: int = 0, samples: int = 5, top_k: int = 10,
                 commit: HandType | None = HandType.PAIR, adaptive: bool = False):
        super().__init__(seed)
        self.samples = samples
        self.top_k = top_k
        # The build's spam hand. Every planet, joker valuation and play
        # decision is aligned to it; the strongest hand is still played
        # whenever the spam hand would fall off pace. With `adaptive`, the
        # commitment is re-measured at every shop: the candidate hand with
        # the highest simulated power wins (10% hysteresis).
        self.commit = commit
        self.adaptive = adaptive
        self._power_cache: dict = {}
        self.log: list | None = None  # set to [] to record shop decisions

    def _spam_hand(self, run) -> HandType:
        if self.commit is not None:
            return self.commit
        return super()._spam_hand(run)

    def _reserve(self, run) -> int:
        """Interest-first economy: reach the interest cap by ante 4 and keep
        it; rerolls and marginal buys come out of the surplus only."""
        if run.mods.get("no_interest"):
            return 0
        if run.ante <= 2:
            return 0
        if run.ante <= 4:
            return 10
        return 15

    # ------------------------------------------------------------- simulation
    def _sim_rng(self, run, salt: str) -> random.Random:
        return random.Random(f"{run.seed}:{run.ante}:{run.stats.get('hands_played_run', 0)}:{salt}")

    def best_play_score(self, run, hand: list) -> float:
        """Best single-hand score for `hand` on a throwaway copy of run."""
        base = run.clone()
        # A representative mid-round context: one spam hand already played
        # (Card Sharp fires), two hands left (Acrobat does not), one discard
        # left (Mystic Summit does not, Banner pays a little).
        spam = int(self._spam_hand(run))
        base.round = RoundState(
            blind_kind=run.round.blind_kind if run.round else 0,
            boss_id=None,
            chips_needed=1.0,
            hands_left=2,
            discards_left=1,
            hands_played=1,
            hand_types_played=[spam],
            hand=list(hand),
            deck_remaining=[c for c in base.full_deck if c.uid not in {h.uid for h in hand}],
        )
        base.round.hand = [c for c in base.full_deck if c.uid in {h.uid for h in hand}]
        rules = fold_eval_rules(base)
        cards = base.round.hand
        seen: dict = {}
        for idxs in structural_candidates(cards, rules):
            played = [cards[i] for i in idxs]
            hr = evaluate_hand(played, rules)
            key = (hr.hand_type, frozenset(c.uid for c in hr.scoring_cards), len(played))
            chips, mult = hand_value(hr.hand_type, base.hand_levels[hr.hand_type])
            val = (chips + sum(c.base_chips for c in hr.scoring_cards)) * mult
            if key not in seen or seen[key][0] < val:
                seen[key] = (val, idxs)
        ranked = sorted(seen.values(), key=lambda t: -t[0])
        if self.commit is not None:
            spam_plays = [t for t in ranked if evaluate_hand([cards[i] for i in t[1]], rules).hand_type == self.commit]
            if spam_plays:
                ranked = spam_plays
        ranked = ranked[: self.top_k]
        best = 0.0
        for _, idxs in ranked:
            trial = base.clone()
            played = [trial.round.hand[i] for i in idxs]
            trial.round.hand = [c for i, c in enumerate(trial.round.hand) if i not in set(idxs)]
            res = score_play(trial, played, held=trial.round.hand)
            best = max(best, res.score)
        return best

    def power(self, run, *, salt: str = "", samples: int | None = None) -> float:
        """Expected best-play score of one hand dealt from this run's deck."""
        k = samples or self.samples
        rng = self._sim_rng(run, salt)
        n = cardops.effective_hand_size(run)
        deck = list(run.full_deck)
        total = 0.0
        for _ in range(k):
            rng.shuffle(deck)
            total += self.best_play_score(run, deck[:n])
        return total / k

    def projected(self, run):
        """Clone with scaling jokers advanced PROJECT_ROUNDS rounds."""
        c = run.clone()
        spam = self._spam_hand(run)
        c.hand_play_counts[spam] = c.hand_play_counts.get(spam, 0) + 3 * PROJECT_ROUNDS
        for j in c.jokers:
            for key, rate in GROWTH.get(j.id, {}).items():
                if key in j.state:
                    j.state[key] = j.state[key] + rate * PROJECT_ROUNDS
        return c

    def board_power(self, run, salt: str = "") -> float:
        return self.power(self.projected(run), salt=salt)

    def choose_commit(self, run) -> HandType:
        """Re-measure which spam hand this board scores best with."""
        current = self.commit
        proj = self.projected(run)
        scores = {}
        for ht in self.COMMIT_CANDIDATES:
            self.commit = ht
            scores[ht] = self.power(proj, salt="commit", samples=4)
        best = max(scores, key=scores.get)
        if current in scores and scores[best] < 1.10 * scores[current]:
            best = current
        self.commit = best
        return best

    # ------------------------------------------------------------- evaluation
    def value_of_joker(self, run, jid: str, edition: Edition = Edition.NONE, *, replace_uid: int | None = None) -> float:
        """Power ratio after adding (optionally replacing) a joker."""
        trial = run.clone()
        if replace_uid is not None:
            victim = next(j for j in trial.jokers if j.uid == replace_uid)
            cardops.remove_joker(trial, victim)
        if not cardops.has_joker_room(trial, edition):
            return 0.0
        j = cardops.new_joker(trial, jid, edition=edition)
        cardops.add_joker(trial, j)
        value = self.power(self.projected(trial), salt="base")  # same hands as the base
        dollars = ECON_DOLLARS.get(jid, 0.0) * min(rounds_left(run), 18)
        if dollars:
            base = self.power(self.projected(run), salt="base")
            value += base * min(0.5, dollars * DOLLAR_TO_POWER)
        return value

    def value_of_planet(self, run, cid: str) -> float:
        ht = PLANET_HAND.get(cid)
        if ht is None:
            return 0.0
        trial = run.clone()
        trial.hand_levels[ht] += 1
        return self.power(self.projected(trial), salt="base")

    def requirement(self, run, kind_mult: float = 2.0) -> float:
        return ante_base_chips(run.ante, run.stake) * run.mods.get("ante_scaling", 1) * kind_mult

    # ------------------------------------------------------------- ordering
    def _best_order_swap(self, env, mask: np.ndarray) -> int | None:
        run = env.run
        n = min(len(run.jokers), codec.N_SELL_SLOTS)
        if n < 2:
            return None
        base = self.power(run, salt="order", samples=3)
        best = (base * 1.02, None)
        for i in range(n - 1):
            if not mask[codec.SWAP_JOKER_BASE + i]:
                continue
            trial = run.clone()
            trial.jokers[i], trial.jokers[i + 1] = trial.jokers[i + 1], trial.jokers[i]
            p = self.power(trial, salt="order", samples=3)
            if p > best[0]:
                best = (p, codec.SWAP_JOKER_BASE + i)
        return best[1]

    # ------------------------------------------------------------ blind phase
    def _blind_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        key = (run.ante, run.blind_index, len(run.jokers), tuple(j.uid for j in run.jokers))
        if self._power_cache.get("order_key") != key:
            swap = self._best_order_swap(env, mask)
            if swap is not None:
                return swap
            self._power_cache["order_key"] = key
        use = self._consumable_action(env, mask)
        if use is not None:
            return use
        if mask[codec.REROLL_BOSS_IDX] and run.ante_boss in ("b_needle", "b_wall", "b_violet_vessel", "b_ox", "b_flint", "b_arm", "b_eye", "b_crimson_heart"):
            if run.money - 10 >= self._reserve(run) + 5:
                return codec.REROLL_BOSS_IDX
        if mask[codec.SKIP_BLIND] and run.blind_index < 2 and run.ante >= 2:
            tag = run.blind_tags[run.blind_index] if run.blind_index < len(run.blind_tags) else None
            value = TAG_VALUE.get(tag.id, 0) if tag else 0
            if tag and tag.id == "t_double":
                nxt = run.blind_tags[1] if run.blind_index == 0 else None
                value = TAG_VALUE.get(nxt.id, 0) if nxt else 2
            threshold = 6 if run.blind_index == 0 else 8
            if value >= threshold:
                hands = cardops.effective_hands(run)
                if self.power(run, salt="skip") * hands >= 2.0 * self.requirement(run):
                    return codec.SKIP_BLIND
        if mask[codec.SELECT_BLIND]:
            return codec.SELECT_BLIND
        return int(self.rng.choice(np.flatnonzero(mask)))

    # ------------------------------------------------------------- consumables
    def _consumable_value(self, env, inst) -> float:
        run = env.run
        if inst.id in PLANET_HAND:
            ht = PLANET_HAND[inst.id]
            spam = self._spam_hand(run)
            counts = run.hand_play_counts
            total = max(1, sum(counts.values()))
            if ht == spam or (self.commit is None and counts.get(ht, 0) / total >= 0.3):
                if run.mods.get("observatory") and ht == spam:
                    # Observatory: a held spam planet is x1.5 every hand. Keep
                    # them in the slots; only level up when the slots are full.
                    held = sum(1 for c in run.consumables if c.id == inst.id)
                    return 6.0 if cardops.consumables_used_slots(run) >= run.consumable_slots and held > 1 else 0.0
                return 6.0
            return 0.0  # off-build planets are sold, never used
        if inst.id == "c_fool" and run.last_consumable in PLANET_HAND:
            return 6.0 if PLANET_HAND[run.last_consumable] == self._spam_hand(run) else 0.5
        return super()._consumable_value(env, inst)

    # ------------------------------------------------------------- shop phase
    def _shop_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        shop = run.shop
        reserve = self._reserve(run)
        if shop is None:
            return super()._shop_phase(env, mask)

        key = ("shop", run.ante, run.blind_index, len(run.jokers), tuple(j.uid for j in run.jokers))
        if self._power_cache.get("order_key") != key:
            if self.adaptive and self._power_cache.get("commit_key") != key[:3]:
                self.choose_commit(run)
                self._power_cache["commit_key"] = key[:3]
            swap = self._best_order_swap(env, mask)
            if swap is not None:
                return swap
            self._power_cache["order_key"] = key
        use = self._consumable_action(env, mask)
        if use is not None:
            return use

        base = self.power(self.projected(run), salt="base")
        now = self.power(run, salt="now")  # unprojected: what the board does today
        need = self.requirement(run)  # next boss requirement as the yardstick
        hands = cardops.effective_hands(run)
        deficit = need / max(now * hands, 1.0)  # >1 means we cannot beat the boss yet

        def money_ok(cost: int, importance: float) -> bool:
            left = run.money + cardops.debt_allowance(run) - cost
            if importance >= 1.6 or deficit > 1.0:
                return left >= -cardops.debt_allowance(run)  # build-defining, or dead anyway
            if importance >= 1.25:
                return left >= reserve - 12
            return left >= reserve

        best = (1.0, None, 0)  # (ratio, action, cost)

        # --- jokers on offer: add, or replace the weakest when the board is full ---
        for i, it in enumerate(shop.card_slots[:codec.N_BUY_CARD_SLOTS]):
            idx = codec.BUY_CARD_BASE + i
            if it.kind == "joker":
                if not (run.money + cardops.debt_allowance(run) >= it.cost):
                    continue
                if "eternal" in it.stickers and JOKER_INFO.get(it.item_id, (1, "add"))[0] <= 2:
                    continue
                if cardops.has_joker_room(run, it.edition):
                    if not mask[idx]:
                        continue
                    ratio = self.value_of_joker(run, it.item_id, it.edition) / max(base, 1.0)
                    if "perishable" in it.stickers:
                        ratio = 1 + (ratio - 1) * 0.6
                    if "rental" in it.stickers:
                        ratio = 1 + (ratio - 1) * 0.8
                    if it.cost == 0:
                        ratio += 0.05
                    if ratio > best[0] and money_ok(it.cost, ratio):
                        best = (ratio, idx, it.cost)
                else:
                    # replacement: sell the joker whose removal costs least, then buy
                    for j in run.jokers:
                        if "eternal" in j.stickers:
                            continue
                        ratio = self.value_of_joker(run, it.item_id, it.edition, replace_uid=j.uid) / max(base, 1.0)
                        # Replacing forfeits half the old joker's price: demand a real step up.
                        if ratio >= 1.12 and ratio > best[0] + 0.02 and money_ok(it.cost - j.sell_value, ratio):
                            sidx = codec.SELL_JOKER_BASE + run.jokers.index(j)
                            if sidx < codec.SELL_JOKER_BASE + codec.N_SELL_SLOTS and mask[sidx]:
                                best = (ratio, sidx, it.cost)
            elif it.kind == "planet" and mask[idx]:
                ratio = self.value_of_planet(run, it.item_id) / max(base, 1.0)
                if ratio > best[0] and money_ok(it.cost, ratio):
                    best = (ratio, idx, it.cost)
            elif it.kind in ("tarot", "spectral") and mask[idx]:
                want = super()._consumable_value(env, cardops.new_consumable(run, it.item_id))
                if want >= 3 and (it.cost == 0 or run.money - it.cost >= reserve + 5):
                    ratio = 1.0 + 0.02 * want
                    if ratio > best[0]:
                        best = (ratio, idx, it.cost)

        # --- celestial packs: a guaranteed spam planet with Telescope, else a chance ---
        spam_planet = next((cid for cid, ht in PLANET_HAND.items() if ht == self._spam_hand(run)), None)
        if spam_planet is not None:
            from balatro_gym.engine import PACKS

            planet_ratio = self.value_of_planet(run, spam_planet) / max(base, 1.0)
            for i, offer in enumerate(shop.pack_slots[:codec.N_BUY_PACK_SLOTS]):
                idx = codec.BUY_PACK_BASE + i
                if not mask[idx] or "celestial" not in offer.pack_id:
                    continue
                options = PACKS[offer.pack_id].options
                p_hit = 1.0 if run.mods.get("telescope") else 1 - (11 / 12) ** options
                ratio = 1 + p_hit * (planet_ratio - 1)
                if ratio > best[0] and money_ok(offer.cost, ratio):
                    best = (ratio, idx, offer.cost)

        # Only buy things that actually move the needle (>= +3% power), or
        # anything free.
        if self.log is not None:
            self.log.append((run.ante, run.blind_index, run.money, round(base), round(need), round(deficit, 2),
                             [(it.kind, it.item_id, it.cost) for it in shop.card_slots], best))
        if best[1] is not None and (best[0] >= 1.03 or best[2] == 0):
            return best[1]

        # --- vouchers ---
        vouchers = []
        for i, vid in enumerate(shop.vouchers[:codec.N_VOUCHER_SLOTS]):
            idx = codec.BUY_VOUCHER_BASE + i
            if mask[idx]:
                value = VOUCHER_VALUE.get(vid, 2)
                if self.commit is not None and vid in ("v_telescope", "v_observatory"):
                    value = 9  # the committed build's planet engine
                vouchers.append((value, idx, VOUCHERS[vid].cost))
        if vouchers:
            value, idx, cost = max(vouchers)
            if value >= 5 and run.money - cost >= reserve + 8:
                return idx

        # --- packs ---
        for i, offer in enumerate(shop.pack_slots[:codec.N_BUY_PACK_SLOTS]):
            idx = codec.BUY_PACK_BASE + i
            if not mask[idx]:
                continue
            pid = offer.pack_id
            left = run.money - offer.cost
            if offer.cost == 0:
                return idx
            if "buffoon" in pid and left >= reserve - 4 and (len(run.jokers) < run.joker_slots or deficit > 0.5):
                return idx
            if "celestial" in pid and left >= reserve and run.stats.get("hands_played_run", 0) >= 4:
                return idx
            if "spectral" in pid and left >= reserve + 4:
                return idx
            if "arcana" in pid and left >= reserve + 6:
                return idx

        # --- reroll while behind the curve and able to pay ---
        cost = 0 if shop.free_rerolls > 0 else shop.reroll_cost
        no_x = not any(_classify(j.id) == "x" for j in run.jokers)
        if mask[codec.REROLL_IDX] and (
            cost == 0
            or (deficit > 1.0 and run.money - cost >= 0)  # dead anyway: spend everything
            or (deficit > 0.6 and run.money - cost >= reserve - 10 and cost <= 9)
            or ((deficit > 0.3 or no_x) and run.money - cost >= reserve and cost <= 9)
            or (run.money - cost >= reserve + 15 and cost <= 10)
        ):
            return codec.REROLL_IDX

        if mask[codec.LEAVE_SHOP_IDX]:
            return codec.LEAVE_SHOP_IDX
        return int(self.rng.choice(np.flatnonzero(mask)))

    # ------------------------------------------------------------- pack phase
    def _pack_phase(self, env, mask: np.ndarray) -> int:
        run = env.run
        pack = run.open_pack
        if pack is None:
            return super()._pack_phase(env, mask)
        base = self.power(self.projected(run), salt="base")
        best = (-1.0, -1)
        for i, kind in enumerate(pack.option_kinds[:codec.N_PACK_OPTIONS]):
            idx = codec.PICK_PACK_BASE + i
            if not mask[idx]:
                continue
            if kind == "joker":
                ed = Edition(pack.option_editions[i]) if i < len(pack.option_editions) else Edition.NONE
                ratio = self.value_of_joker(run, pack.option_ids[i], ed) / max(base, 1.0)
                value = 3 + 60 * (ratio - 1.0) + (1 if ed != Edition.NONE else 0)
            elif kind == "planet":
                ratio = self.value_of_planet(run, pack.option_ids[i]) / max(base, 1.0)
                value = 3 + 60 * (ratio - 1.0)
            elif kind in ("tarot", "spectral"):
                value = 3 + super()._consumable_value(env, cardops.new_consumable(run, pack.option_ids[i]))
            else:
                card = pack.option_cards[i]
                value = 3 if (card and str(card.enhancement) != "none") else 1
            if value > best[0]:
                best = (value, idx)
        if best[1] >= 0 and best[0] >= 3.5:
            return best[1]
        if mask[codec.SKIP_PACK_IDX]:
            return codec.SKIP_PACK_IDX
        return int(self.rng.choice(np.flatnonzero(mask)))


# ---------------------------------------------------------------------------
# Play-phase rollouts (engine-level, no env needed)
# ---------------------------------------------------------------------------


def structural_candidates(cards, rules) -> list[tuple[int, ...]]:
    """Index tuples of the plays a player would consider: every rank group
    (pairs, trips, quads and their sub-groups), the best two pair and full
    house, the top flush per suit, every straight, and the top singles.
    ~10-25 candidates instead of all C(n, <=5) subsets."""
    from collections import defaultdict

    from balatro_gym.engine import Suit
    from balatro_gym.engine.hand_eval import suit_matches

    n = len(cards)
    out: set[tuple[int, ...]] = set()
    by_rank: dict = defaultdict(list)
    for i, c in enumerate(cards):
        if not c.is_stone:
            by_rank[int(c.rank)].append(i)
    order = sorted(range(n), key=lambda i: (-int(cards[i].rank), i))
    # singles: top two ranks
    for i in order[:2]:
        out.add((i,))
    groups = sorted(((r, idx) for r, idx in by_rank.items() if len(idx) >= 2), key=lambda t: -t[0])
    for r, idx in groups:
        out.add(tuple(idx[:5]))
        if len(idx) >= 3:
            out.add(tuple(idx[:2]))
        if len(idx) >= 4:
            out.add(tuple(idx[:3]))
    if len(groups) >= 2:
        a, b = groups[0][1], groups[1][1]
        out.add(tuple(sorted(a[:2] + b[:2])))
        if len(a) >= 3:
            out.add(tuple(sorted(a[:3] + b[:2])))
        if len(b) >= 3:
            out.add(tuple(sorted(b[:3] + a[:2])))
    # flushes: best `flush_size` of each suit (highest ranks)
    for suit in Suit:
        members = [i for i in order if suit_matches(cards[i], suit, rules)]
        if len(members) >= rules.flush_size:
            out.add(tuple(sorted(members[:5])))
            if len(members) > 5:
                out.add(tuple(sorted(members[:rules.flush_size])))
    # straights: walk rank values with the allowed gap, one card per rank
    values: dict = {}
    for i, c in enumerate(cards):
        if c.is_stone:
            continue
        v = int(c.rank)
        values.setdefault(v, i)
        if v == 14:
            values.setdefault(1, i)
    vs = sorted(values)
    size, gap = rules.straight_size, rules.straight_max_gap
    for start in range(len(vs)):
        run_ = [vs[start]]
        j = start
        while len(run_) < size and j + 1 < len(vs):
            j += 1
            if vs[j] - run_[-1] <= gap:
                run_.append(vs[j])
            else:
                break
        if len(run_) == size and not (1 in run_ and 14 in run_):
            out.add(tuple(sorted(values[v] for v in run_)))
    return [t for t in out if 1 <= len(t) <= 5]

def _engine_candidates(run, top_k: int = 10) -> list[tuple[float, tuple[int, ...]]]:
    """(proxy value, hand indices) for the best distinct plays in run.round.hand."""
    rules = fold_eval_rules(run)
    cards = run.round.hand
    seen: dict = {}
    for idxs in structural_candidates(cards, rules):
        played = [cards[i] for i in idxs]
        hr = evaluate_hand(played, rules)
        key = (hr.hand_type, frozenset(c.uid for c in hr.scoring_cards), len(played))
        chips, mult = hand_value(hr.hand_type, run.hand_levels[hr.hand_type])
        val = (chips + sum(c.base_chips for c in hr.scoring_cards)) * mult
        if key not in seen or seen[key][0] < val:
            seen[key] = (val, idxs)
    return sorted(seen.values(), key=lambda t: -t[0])[:top_k]


def _legal_play(run, idxs) -> bool:
    from balatro_gym.engine.scoring import get_boss

    boss = get_boss(run)
    if boss is None:
        return True
    return boss.validate_play(run, [run.round.hand[i] for i in idxs]) is None


def _legal_discard(run, idxs) -> bool:
    from balatro_gym.engine.scoring import get_boss

    boss = get_boss(run)
    if boss is None:
        return True
    return boss.validate_discard(run, [run.round.hand[i] for i in idxs]) is None


def _best_play(run, top_k: int = 10):
    """(score, indices) of the best legal play, scored for real on clones."""
    best = (-1.0, None)
    for _, idxs in _engine_candidates(run, top_k):
        if not _legal_play(run, idxs):
            continue
        trial = run.clone()
        played = [trial.round.hand[i] for i in idxs]
        trial.round.hand = [c for i, c in enumerate(trial.round.hand) if i not in set(idxs)]
        res = score_play(trial, played, held=trial.round.hand)
        if res.score > best[0]:
            best = (res.score, idxs)
    return best


def _simple_discard(run) -> tuple[int, ...] | None:
    """Lowest singletons (keeping pairs and 4-flushes), up to 5 cards."""
    from collections import Counter

    cards = run.round.hand
    n = len(cards)
    counts = Counter(c.rank for c in cards)
    suits = Counter(c.suit for c in cards if not c.is_stone)
    keep = {i for i in range(n) if counts[cards[i].rank] >= 2}
    if suits and max(suits.values()) >= 4:
        s = max(suits, key=suits.get)
        keep |= {i for i in range(n) if cards[i].suit == s}
    pool = sorted((i for i in range(n) if i not in keep), key=lambda i: int(cards[i].rank))[:5]
    if not pool:
        pool = sorted(range(n), key=lambda i: int(cards[i].rank))[:2]
    idxs = tuple(sorted(pool))
    return idxs if idxs and _legal_discard(run, idxs) else None


def rollout_clear(run, first_action, rng: random.Random, top_k: int = 8) -> float:
    """Play out the rest of the blind greedily on a clone after `first_action`
    (("play", idxs) or ("discard", idxs)). Returns 1.0 when the blind clears,
    otherwise the fraction of the requirement that was scored (partial credit
    so options can still be ranked when nothing clears)."""
    from balatro_gym.engine import round as round_mod

    sim = run.clone()
    rng.shuffle(sim.round.deck_remaining)
    kind, idxs = first_action
    if kind == "play":
        round_mod.play_hand(sim, tuple(sorted(idxs)))
    else:
        round_mod.discard(sim, tuple(sorted(idxs)))
    guard = 0
    while sim.phase == Phase.PLAYING and sim.round is not None and guard < 12:
        guard += 1
        rnd = sim.round
        need = rnd.chips_needed - rnd.chips_scored
        score, play = _best_play(sim, top_k)
        if play is None:
            return 0.0
        pace = need / max(rnd.hands_left, 1)
        if score < need and score < pace and rnd.discards_left > 0:
            d = _simple_discard(sim)
            if d is not None:
                round_mod.discard(sim, d)
                continue
        round_mod.play_hand(sim, tuple(sorted(play)))
    if sim.phase != Phase.GAME_OVER and (sim.round is None or sim.round.chips_scored >= sim.round.chips_needed):
        return 1.0
    if sim.round is None:
        return 0.0
    return min(0.95, sim.round.chips_scored / max(sim.round.chips_needed, 1.0))


def _committed_choice(self, env, mask: np.ndarray) -> int | None:
    """Play the committed hand when it clears the blind or keeps pace."""
    from .baselines import play_candidates, score_candidate

    run = env.run
    rnd = run.round
    need = rnd.chips_needed - rnd.chips_scored
    pace = need / max(rnd.hands_left, 1)
    best_spam = (-1.0, None)
    best_any = (-1.0, None)
    for gidx, engine_idxs in play_candidates(env, mask):
        res = score_candidate(env, engine_idxs)
        if res.score > best_any[0]:
            best_any = (res.score, gidx)
        if res.hand_result.hand_type == self.commit and res.score > best_spam[0]:
            best_spam = (res.score, gidx)
    if best_spam[1] is None:
        return None
    if best_spam[0] >= need or best_spam[0] >= pace or best_any[0] < pace * 0.5:
        return best_spam[1]
    return None


def _planner_play_phase(self, env, mask: np.ndarray) -> int:
    """Committed-hand policy (fallback: strategist greedy), plus rollout
    arbitration when the best play cannot clear the blind and discards remain."""
    run = env.run
    rnd = run.round
    base_choice = None
    if self.commit is not None:
        base_choice = _committed_choice(self, env, mask)
    if base_choice is None:
        base_choice = StrategistAgent._play_phase(self, env, mask)
    kind, payload = codec.decode_kind(base_choice)
    need = rnd.chips_needed - rnd.chips_scored
    if rnd.discards_left == 0 or rnd.hands_left == 0:
        return base_choice
    # Engine-level view of the strategist's choice.
    engine_idxs = tuple(sorted(env._hand_order[p] for p in payload))
    best_score, best_play = _best_play(run, self.top_k)
    if best_play is None or best_score >= need:
        return base_choice
    options: list[tuple[str, tuple[int, ...], int]] = [("play", best_play, None)]
    if kind == "discard":
        options.append(("discard", engine_idxs, base_choice))
    simple = _simple_discard(run)
    if simple is not None and simple != engine_idxs:
        options.append(("discard", simple, None))
    rng = self._sim_rng(run, f"roll:{rnd.hands_left}:{rnd.discards_left}:{len(rnd.hand)}")
    scores = []
    seeds = [rng.random() for _ in range(self.rollouts)]  # common random numbers
    for okind, idxs, _ in options:
        scores.append(sum(rollout_clear(run, (okind, idxs), random.Random(sd)) for sd in seeds))
    pick = max(range(len(options)), key=lambda i: (scores[i], i == 0))
    okind, idxs, action = options[pick]
    if action is not None:
        return action
    remap = {eng: slot for slot, eng in enumerate(env._hand_order)}
    subset = tuple(sorted(remap[i] for i in idxs))
    base = codec.PLAY_BASE if okind == "play" else codec.DISCARD_BASE
    sidx = codec._SUBSET5_INDEX.get(subset)
    if sidx is not None and mask[base + sidx]:
        return base + sidx
    return base_choice


PlannerAgent._play_phase = _planner_play_phase
PlannerAgent.rollouts = 8
