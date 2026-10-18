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
from typing import ClassVar

import numpy as np

from balatro_gym.engine import (
    BOSSES,
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

BOSSES_TO_REROLL = {"b_needle", "b_wall", "b_violet_vessel", "b_ox", "b_flint", "b_arm", "b_eye",
                    "b_mouth", "b_crimson_heart", "b_psychic", "b_plant", "b_pillar", "b_verdant_leaf"}

# Projected growth per round for scaling jokers (state key -> per-round delta),
# used when valuing a purchase. Rough community-consensus rates for a
# spam-hand build (4 hands / round, ~3 discards).
GROWTH: dict[str, dict[str, float]] = {
    "j_green_joker": {"mult": 2.0},  # +1 per hand, -1 per discard
    "j_ride_the_bus": {"mult": 2.0},  # resets on a scoring face card
    "j_castle": {"chips": 9.0},
    "j_wee": {"chips": 12.0},
    "j_runner": {"chips": 4.0},
    "j_square": {"chips": 12.0},  # pair spam is padded to 4 cards (see _with_fillers)
    "j_trousers": {"mult": 0.5},  # two pair is rarely the spam hand
    "j_constellation": {"x": 0.04},  # ~0.4 planets used per round
    "j_hologram": {"x": 0.05},
    "j_vampire": {"x": 0.04},  # needs enhanced cards to feed on
    "j_lucky_cat": {"x": 0.05},
    "j_glass": {"x": 0.05},
    "j_campfire": {"x": 0.1},  # resets after every boss
    "j_obelisk": {"x": 0.05},  # resets whenever the spam hand is played
    "j_yorick": {"x": 0.15},
    "j_flash": {"mult": 1.0},
    "j_red_card": {"mult": 0.5},
    "j_rocket": {"dollars": 0.7},
    "j_loyalty_card": {},
    "j_egg": {},
    # Not listed on purpose: Madness and Ceremonial Dagger destroy jokers,
    # Hit the Road resets every round.
}
PROJECT_ROUNDS = 6

# Expected dollars per round from economy jokers (rough averages for a
# 4-hand round); converted into a power-equivalent bonus when valuing a buy.
ECON_DOLLARS: dict[str, float] = {
    "j_golden": 4.0, "j_rocket": 1.5, "j_to_the_moon": 2.5, "j_egg": 1.5, "j_gift": 2.0,
    "j_cloud_9": 1.5, "j_delayed_grat": 3.0, "j_reserved_parking": 1.0, "j_business": 1.5,
    "j_satellite": 1.0, "j_mail_in": 2.0, "j_faceless": 0.5, "j_rough_gem": 1.5,
    "j_ticket": 0.5, "j_matador": 1.0, "j_todo_list": 1.0, "j_credit_card": 1.0,
    "j_trading": 0.8, "j_cartomancer": 1.0, "j_vagabond": 0.3, "j_8ball": 0.5,
    "j_hallucination": 0.5, "j_riff_raff": 1.5, "j_chaos": 5.5, "j_astronomer": 1.5,
}
DOLLAR_TO_POWER = 0.012  # +$1 of expected income ~ +1.2% power (a $5 joker ~ +6%).
# Doubling this (and the cap below) measured 22% vs 26% at 120 seeds: not better.


def rounds_left(run) -> int:
    return max(1, (8 - run.ante) * 3 + (2 - min(run.blind_index, 2)))


class PlannerAgent(StrategistAgent):
    COMMIT_CANDIDATES = (HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_KIND,
                         HandType.FLUSH, HandType.HIGH_CARD, HandType.FULL_HOUSE)

    def __init__(self, seed: int = 0, samples: int = 5, top_k: int = 10,
                 commit: HandType | None = HandType.PAIR, adaptive: bool = False,
                 reserve_cap: int = 25, hunt_floor: int = 25, min_gain_per_dollar: float = 0.0,
                 early_reserve: bool = False):
        super().__init__(seed)
        self.samples = samples
        self.top_k = top_k
        self.reserve_cap = reserve_cap  # money kept for interest in antes 2-6
        # Money kept while rerolling for an xMult joker. At the interest cap
        # (the default) the agent never rerolls below its reserve: 26% vs 20%
        # win rate at 120 seeds against a $8 floor, because the interest it
        # keeps buys more rerolls over the run than the early ones it skips.
        self.hunt_floor = hunt_floor
        # Opportunity cost of a dollar: a purchase must raise measured power by
        # at least this fraction per dollar spent (a reroll is ~15% to show an
        # xMult joker, so money kept for rerolls has real value).
        self.min_gain_per_dollar = min_gain_per_dollar
        self.early_reserve = early_reserve  # hold the full interest reserve from ante 2
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
        if run.ante <= 1:
            return 0
        if run.ante >= 8:
            return 0   # nothing after the last boss: spend it all
        if run.ante == 7:
            return min(self.reserve_cap, 12)
        if run.ante == 2 and not self.early_reserve:
            return min(self.reserve_cap, 10)  # a scoring board first, interest second
        if run.ante == 3 and not self.early_reserve:
            return min(self.reserve_cap, 15)
        return min(run.interest_cap, self.reserve_cap)

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
            if j.id == "j_burnt":   # levels the first discarded hand each round
                c.hand_levels[spam] += PROJECT_ROUNDS
            if j.id == "j_space":   # 1 in 4 per played hand
                c.hand_levels[spam] += PROJECT_ROUNDS  # ~4 hands x 1/4 per round
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
        if current in scores and scores[best] < 1.05 * scores[current]:
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
        scoring = sum(1 for j in run.jokers if _classify(j.id) in ("add", "x", "retrig", "copy"))
        if dollars and (scoring >= 2 or len(run.jokers) <= 1):
            # Income matters once the board scores, and on an empty board
            # where the first purchases are what the income buys.
            base = self.power(self.projected(run), salt="base")
            value += base * min(0.5, dollars * DOLLAR_TO_POWER)
        return value

    def value_of_planet(self, run, cid: str) -> float:
        ht = PLANET_HAND.get(cid)
        if ht is None:
            return 0.0
        trial = run.clone()
        trial.hand_levels[ht] += 1
        value = self.power(self.projected(trial), salt="base")
        if self.adaptive and ht != self.commit and ht in self.COMMIT_CANDIDATES:
            # Worth what it would be if the build moved onto this hand
            # (discounted: switching forfeits the levels already bought).
            saved = self.commit
            self.commit = ht
            try:
                value = max(value, 0.9 * self.power(self.projected(trial), salt="base"))
            finally:
                self.commit = saved
        return value

    def requirement(self, run, kind_mult: float = 2.0) -> float:
        return ante_base_chips(run.ante, run.stake) * run.mods.get("ante_scaling", 1) * kind_mult

    # (hands available, power multiplier) a boss effectively leaves us with
    BOSS_HANDICAP: ClassVar[dict[str, tuple[int, float]]] = {
        "b_needle": (1, 1.0), "b_eye": (99, 0.6), "b_mouth": (99, 0.7), "b_psychic": (99, 0.8),
        "b_plant": (99, 0.75), "b_pillar": (99, 0.7), "b_head": (99, 0.8), "b_club": (99, 0.8),
        "b_goad": (99, 0.8), "b_window": (99, 0.8), "b_flint": (99, 0.6), "b_arm": (99, 0.85),
        "b_hook": (99, 0.8), "b_fish": (99, 0.85), "b_house": (99, 0.9), "b_mark": (99, 0.85),
        "b_water": (99, 0.85), "b_serpent": (99, 0.9), "b_tooth": (99, 0.95), "b_ox": (99, 0.9),
        "b_wheel": (99, 0.9), "b_manacle": (99, 0.9), "b_verdant_leaf": (99, 0.7),
        "b_crimson_heart": (99, 0.7), "b_amber_acorn": (99, 0.9), "b_cerulean_bell": (99, 0.8),
    }

    def boss_handicap(self, boss_id) -> tuple[int, float]:
        return self.BOSS_HANDICAP.get(boss_id, (99, 1.0))

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
        if mask[codec.REROLL_BOSS_IDX] and run.ante_boss in BOSSES_TO_REROLL:
            boss_hands, boss_pen = self.boss_handicap(run.ante_boss)
            mult = BOSSES[run.ante_boss].effect.score_multiplier()
            hands = min(cardops.effective_hands(run), boss_hands)
            if self.power(run, salt="skip") * boss_pen * hands < 1.3 * self.requirement(run, mult):
                return codec.REROLL_BOSS_IDX
        if mask[codec.SKIP_BLIND] and run.blind_index < 2 and run.ante >= 2:
            tag = run.blind_tags[run.blind_index] if run.blind_index < len(run.blind_tags) else None
            value = TAG_VALUE.get(tag.id, 0) if tag else 0
            if tag and tag.id == "t_double":
                nxt = run.blind_tags[1] if run.blind_index == 0 else None
                value = TAG_VALUE.get(nxt.id, 0) if nxt else 2
            if tag and tag.id == "t_orbital" and tag.state.get("hand") == int(self._spam_hand(run)):
                value = 10  # +3 levels on the build's hand
            if tag and tag.id == "t_meteor" and run.mods.get("telescope"):
                value = 8  # a Mega Celestial pack with Telescope: two spam planets
            # Cash tags pay what the run has accumulated; a skip forfeits the
            # blind reward, the interest cash-out and the unused-hand dollars.
            cash = 0
            if tag and tag.id == "t_handy":
                cash = run.stats.get("hands_played_run", 0)
            elif tag and tag.id == "t_garbage":
                cash = run.stats.get("discards_unused_run", 0)
            elif tag and tag.id == "t_economy":
                cash = min(max(run.money, 0), 40)
            elif tag and tag.id == "t_investment":
                cash = 25
            elif tag and tag.id == "t_speed":
                cash = 5 * (run.stats.get("blinds_skipped", 0) + 1)
            if cash:
                forfeit = (3 if run.blind_index == 0 else 4) + min(max(run.money, 0) // 5, 5) + 2
                if cash - forfeit >= 30:
                    value = max(value, 10)
                elif cash - forfeit >= 15:
                    value = max(value, 8)
            threshold = 6 if run.blind_index == 0 else 8
            if value >= threshold:
                hands = cardops.effective_hands(run)
                power = self.power(run, salt="skip")
                if value >= 8:
                    # Premium tags (a free rare joker, an edition, +3 levels):
                    # skip whenever the blind faced after the skip is clearable.
                    if run.blind_index == 0:
                        ok = power * hands >= 1.3 * self.requirement(run, 1.5)
                    else:
                        b_hands, b_pen = self.boss_handicap(run.ante_boss)
                        mult = BOSSES[run.ante_boss].effect.score_multiplier() if run.ante_boss in BOSSES else 2.0
                        ok = power * b_pen * min(hands, b_hands) >= 1.3 * self.requirement(run, mult)
                    if ok:
                        return codec.SKIP_BLIND
                elif power * hands >= 2.0 * self.requirement(run):
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
            movable = (
                self.adaptive
                and ht in self.COMMIT_CANDIDATES
                and run.hand_levels[ht] + 1 >= run.hand_levels[spam]
            )
            if any(j.id == "j_constellation" for j in run.jokers) and ht != spam:
                return 4.0  # +x0.1 per planet used: every planet feeds Constellation
            if ht == spam or movable or (self.commit is None and counts.get(ht, 0) / total >= 0.3):
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

    def _consumable_action(self, env, mask: np.ndarray) -> int | None:
        run = env.run
        use = super()._consumable_action(env, mask)
        if use is not None:
            return use
        if any(j.id == "j_campfire" for j in run.jokers):
            # Campfire gains x0.25 per card sold: junk is worth more sold now.
            for i, inst in enumerate(run.consumables[: codec.N_CONSUMABLE_SLOTS]):
                idx = codec.SELL_CONSUMABLE_BASE + i
                if mask[idx] and self._consumable_value(env, inst) <= 1.0:
                    return idx
        return None

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
        # The blind we actually face next decides whether we are "dead anyway".
        from balatro_gym.engine import BOSSES

        if run.blind_index >= 2:
            next_mult = BOSSES[run.ante_boss].effect.score_multiplier() if run.ante_boss in BOSSES else 2.0
            boss_hands, boss_pen = self.boss_handicap(run.ante_boss)
            hands_next = min(hands, boss_hands)
            power_next = now * boss_pen
        else:
            next_mult = 1.0 if run.blind_index == 0 else 1.5
            hands_next, power_next = hands, now
        dead_now = self.requirement(run, next_mult) > 0.9 * power_next * hands_next
        # The boss after that is what rerolls and purchases should prepare for.
        boss_hands, boss_pen = self.boss_handicap(run.ante_boss)
        deficit = need / max(now * boss_pen * min(hands, boss_hands), 1.0)

        behind = deficit > 1.0  # the ante's boss beats the board as it stands
        scoring_jokers = sum(1 for j in run.jokers if _classify(j.id) in ("add", "x", "retrig", "copy"))

        def money_ok(cost: int, importance: float) -> bool:
            left = run.money - cost  # Credit Card debt is for emergencies only
            if importance >= 1.6 or dead_now:
                return left >= -cardops.debt_allowance(run)  # build-defining, or dead anyway
            if importance >= 1.3:
                return left >= (0 if behind else reserve - 10)
            if importance >= 1.12:
                # A permanent +12% is worth a few rounds of reduced interest;
                # when the boss is out of reach it is worth most of the bankroll.
                return left >= (reserve - 15 if behind else reserve - 5)
            return left >= reserve

        best = (1.0, None, 0)  # (ratio, action, cost)
        picks: list[tuple[float, float, int, int]] = []  # (gain per dollar, ratio, action, cost)

        def consider(ratio: float, idx: int, cost: int) -> None:
            if cost == 0 or ratio >= 1.03 + self.min_gain_per_dollar * cost:
                picks.append(((ratio - 1.0) / (cost + 2.0), ratio, idx, cost))

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
                    if money_ok(it.cost, ratio):
                        consider(ratio, idx, it.cost)
                else:
                    # replacement: sell the joker whose removal costs least, then buy
                    for j in run.jokers:
                        if "eternal" in j.stickers:
                            continue
                        ratio = self.value_of_joker(run, it.item_id, it.edition, replace_uid=j.uid) / max(base, 1.0)
                        # Replacing forfeits half the old joker's price: demand a real step up.
                        if ratio >= 1.12 and money_ok(it.cost - j.sell_value, ratio):
                            sidx = codec.SELL_JOKER_BASE + run.jokers.index(j)
                            if sidx < codec.SELL_JOKER_BASE + codec.N_SELL_SLOTS and mask[sidx]:
                                consider(ratio, sidx, max(1, it.cost - j.sell_value))
            elif it.kind == "planet" and mask[idx]:
                ratio = self.value_of_planet(run, it.item_id) / max(base, 1.0)
                if money_ok(it.cost, ratio):
                    consider(ratio, idx, it.cost)
            elif it.kind in ("tarot", "spectral") and mask[idx]:
                gain_dollars = 0.0
                if it.item_id == "c_hermit":
                    gain_dollars = min(20, max(0, run.money - it.cost))
                elif it.item_id == "c_temperance":
                    gain_dollars = min(50, sum(j.sell_value for j in run.jokers))
                if gain_dollars >= 8:
                    # Money tarots are valued like income: dollars -> power.
                    ratio = 1.0 + gain_dollars * DOLLAR_TO_POWER
                    if money_ok(it.cost, ratio):
                        consider(ratio, idx, it.cost)
                    continue
                want = super()._consumable_value(env, cardops.new_consumable(run, it.item_id))
                if want >= 3 and (it.cost == 0 or run.money - it.cost >= reserve + 5):
                    consider(1.0 + 0.02 * want, idx, it.cost)

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
                # A planet gamble before the first scoring joker spends the
                # money the next shop's joker needs; only take it from surplus.
                if p_hit < 1.0 and scoring_jokers == 0 and run.money - offer.cost < 8:
                    continue
                ratio = 1 + p_hit * (planet_ratio - 1)
                if money_ok(offer.cost, ratio):
                    consider(ratio, idx, offer.cost)

        # --- standard packs feed Hologram (+x0.25 per card added to the deck) ---
        if any(j.id == "j_hologram" for j in run.jokers):
            from balatro_gym.engine import PACKS

            x_now = sum(j.state.get("x", 1.0) for j in run.jokers if j.id == "j_hologram")
            for i, offer in enumerate(shop.pack_slots[:codec.N_BUY_PACK_SLOTS]):
                idx = codec.BUY_PACK_BASE + i
                if not mask[idx] or "standard" not in offer.pack_id:
                    continue
                n_picks = PACKS[offer.pack_id].picks
                ratio = (x_now + 0.25 * n_picks) / max(x_now, 1.0)
                if money_ok(offer.cost, ratio):
                    consider(ratio, idx, offer.cost)

        # --- buffoon packs: value by sampling the joker pool (best of N options) ---
        if cardops.has_joker_room(run):
            from balatro_gym.engine import PACKS, Rarity

            for i, offer in enumerate(shop.pack_slots[:codec.N_BUY_PACK_SLOTS]):
                idx = codec.BUY_PACK_BASE + i
                if not mask[idx] or "buffoon" not in offer.pack_id:
                    continue
                pool = cardops.joker_pool(run, Rarity.COMMON) + cardops.joker_pool(run, Rarity.UNCOMMON)
                if not pool:
                    continue
                rng = self._sim_rng(run, "buffoon")
                n_opts = PACKS[offer.pack_id].options
                bests = []
                for _ in range(3):  # expected best-of-N over a few sampled packs
                    draws = [rng.choice(pool) for _ in range(n_opts)]
                    bests.append(max(self.value_of_joker(run, jid) / max(base, 1.0) for jid in draws))
                ratio = sum(bests) / len(bests)
                if money_ok(offer.cost, ratio):
                    consider(ratio, idx, offer.cost)

        if picks:
            picks.sort(reverse=True)
            best = (picks[0][1], picks[0][2], picks[0][3])

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
                if vid in ("v_directors_cut", "v_retcon"):
                    value = 7  # rerolling killer bosses saves runs
                if vid in ("v_reroll_surplus", "v_reroll_glut"):
                    value = 7  # $2 off every reroll: a dozen rerolls a run pays it back twice
                vouchers.append((value, idx, VOUCHERS[vid].cost))
        if vouchers:
            value, idx, cost = max(vouchers)
            if value >= 9 and run.money + cardops.debt_allowance(run) >= cost:
                return idx  # the planet engine: buy on sight, whatever the reserve
            if value >= 7 and run.money - cost >= reserve - 5:
                return idx  # Director's Cut / Retcon: a boss reroll saves runs outright
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
            if "buffoon" in pid and (left >= reserve or dead_now) and (len(run.jokers) < run.joker_slots or deficit > 0.5):
                return idx
            if "celestial" in pid and left >= reserve and run.stats.get("hands_played_run", 0) >= 4 and (
                scoring_jokers > 0 or run.mods.get("telescope") or left >= 8
            ):
                return idx
            if "spectral" in pid and left >= reserve + 4:
                return idx
            if "arcana" in pid and left >= reserve + 6:
                return idx

        # --- reroll while behind the curve and able to pay ---
        cost = 0 if shop.free_rerolls > 0 else shop.reroll_cost
        no_x = not any(_classify(j.id) == "x" for j in run.jokers)
        keep = 4  # always keep a cheap joker's price after a reroll
        # Without a multiplicative joker the board cannot keep up with the
        # requirement curve past ante 4; `hunt_floor` says how far below the
        # interest reserve the hunt may dip (not at all by default).
        hunting = (no_x and run.ante >= 3) or behind
        if mask[codec.REROLL_IDX] and (
            cost == 0
            or (dead_now and run.money - cost >= keep)  # dead anyway: spend everything
            or (hunting and run.money - cost >= max(self.hunt_floor, reserve - 15) and cost <= 9)
            or (deficit > 0.8 and run.money - cost >= max(keep, reserve - 5) and cost <= 9)
            or ((deficit > 0.3 or no_x) and run.money - cost >= max(keep, reserve) and cost <= 9)
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
                if any(j.id == "j_hologram" for j in run.jokers):
                    value += 3  # any card added to the deck is +x0.25
            if value > best[0]:
                best = (value, idx)
        threshold = 3.0 if len(run.jokers) < 2 and pack.option_kinds and pack.option_kinds[0] == "joker" else 3.5
        if best[1] >= 0 and best[0] >= threshold:
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


def _best_plays(run, top_k: int = 10) -> list[tuple[float, tuple[int, ...]]]:
    """Legal plays scored for real on clones, best first."""
    out = []
    for _, idxs in _engine_candidates(run, top_k):
        if not _legal_play(run, idxs):
            continue
        trial = run.clone()
        played = [trial.round.hand[i] for i in idxs]
        trial.round.hand = [c for i, c in enumerate(trial.round.hand) if i not in set(idxs)]
        res = score_play(trial, played, held=trial.round.hand)
        out.append((res.score, idxs))
    out.sort(key=lambda t: -t[0])
    return out


def _best_play(run, top_k: int = 10):
    """(score, indices) of the best legal play, scored for real on clones."""
    plays = _best_plays(run, top_k)
    return plays[0] if plays else (-1.0, None)


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


def _draw_discards(run) -> list[tuple[int, ...]]:
    """Discard shapes that chase a draw: keep the biggest suit group (flush
    draw) or the longest rank run (straight draw), dump the rest (<= 5)."""
    from collections import Counter

    cards = run.round.hand
    n = len(cards)
    out: list[tuple[int, ...]] = []
    suits = Counter(c.suit for c in cards if not c.is_stone)
    if suits:
        s_best, cnt = max(suits.items(), key=lambda kv: kv[1])
        if 3 <= cnt <= 4:
            dump = [i for i in range(n) if cards[i].is_stone or cards[i].suit != s_best]
            dump.sort(key=lambda i: int(cards[i].rank))
            idxs = tuple(sorted(dump[:5]))
            if idxs and _legal_discard(run, idxs):
                out.append(idxs)
    values = sorted({int(c.rank) for c in cards if not c.is_stone})
    best_run: list[int] = []
    for i in range(len(values)):
        run_ = [values[i]]
        for v in values[i + 1:]:
            if v - run_[-1] <= 1:
                run_.append(v)
            else:
                break
        if len(run_) > len(best_run):
            best_run = run_
    if 3 <= len(best_run) <= 4:
        keep_vals = set(best_run)
        seen: set = set()
        dump = []
        for i in sorted(range(n), key=lambda i: -int(cards[i].rank)):
            v = int(cards[i].rank)
            if v in keep_vals and v not in seen and not cards[i].is_stone:
                seen.add(v)
            else:
                dump.append(i)
        dump.sort(key=lambda i: int(cards[i].rank))
        idxs = tuple(sorted(dump[:5]))
        if idxs and _legal_discard(run, idxs):
            out.append(idxs)
    return out


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
        round_mod.play_hand(sim, _with_fillers(sim, tuple(sorted(play))))
    if sim.phase != Phase.GAME_OVER and (sim.round is None or sim.round.chips_scored >= sim.round.chips_needed):
        return 1.0
    if sim.round is None:
        return 0.0
    return min(0.95, sim.round.chips_scored / max(sim.round.chips_needed, 1.0))


def _with_fillers(run, idxs: tuple[int, ...]) -> tuple[int, ...]:
    """Pad a small play (high card / pair / trips) with junk cards that will
    not score, so they are redrawn: a free discard. Keeps held-card assets
    (steel, gold, seals, Baron / Shoot the Moon faces, paired ranks and a
    flush draw), respects Half Joker (<= 3 cards) and Square Joker (4), and
    falls back to the original play unless the padded one scores at least
    as much."""
    from collections import Counter

    rnd = run.round
    if rnd is None or len(idxs) >= 5:
        return idxs
    if rnd.boss_id in ("b_tooth", "b_serpent"):
        return idxs  # The Tooth charges per card played; The Serpent redraws only three
    cards = rnd.hand
    rules = fold_eval_rules(run)
    played = [cards[i] for i in idxs]
    hr = evaluate_hand(played, rules)
    if hr.hand_type not in (HandType.HIGH_CARD, HandType.PAIR, HandType.THREE_KIND):
        return idxs
    owned = {j.id for j in run.jokers}
    limit = 5
    if "j_square" in owned:
        limit = 4
    if "j_half" in owned:
        limit = 3
    if len(idxs) >= limit:
        return idxs
    held = [i for i in range(len(cards)) if i not in idxs]
    rank_counts = Counter(int(cards[i].rank) for i in held if not cards[i].is_stone)
    played_ranks = {int(c.rank) for c in played if not c.is_stone}
    suits = Counter(cards[i].suit for i in held if not cards[i].is_stone)
    draw_suit = next((su for su, n in suits.items() if n >= 4), None)
    keep_faces = bool(owned & {"j_baron", "j_shoot_the_moon"})

    def junk(i: int) -> bool:
        c = cards[i]
        if c.is_stone:
            return True  # stones score when played, whatever the hand
        if str(c.enhancement) in ("steel", "gold") or str(c.seal) != "none":
            return False
        r = int(c.rank)
        if r in played_ranks or rank_counts[r] >= 2:
            return False
        if keep_faces and r in (12, 13):
            return False
        return draw_suit is None or c.suit != draw_suit

    fillers = sorted((i for i in held if junk(i)), key=lambda i: (not cards[i].is_stone, int(cards[i].rank)))
    fillers = fillers[: limit - len(idxs)]
    if not fillers:
        return idxs
    padded = tuple(sorted(idxs + tuple(fillers)))
    if not _legal_play(run, padded):
        return idxs
    base_score = score_play(run.clone(), played, held=[cards[i] for i in held]).score
    trial = run.clone()
    new_played = [trial.round.hand[i] for i in padded]
    new_held = [c for i, c in enumerate(trial.round.hand) if i not in set(padded)]
    if score_play(trial, new_played, held=new_held).score + 1e-9 < base_score:
        return idxs
    return padded


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
    if best_spam[0] >= need:
        return best_spam[1]
    scaler = any(j.id in GROWTH or j.id == "j_supernova" for j in run.jokers)
    if scaler and rnd.hands_left >= 3 and best_spam[0] >= 1.5 * pace:
        return best_spam[1]
    return None


def _planner_play_phase(self, env, mask: np.ndarray) -> int:
    """Committed-hand policy (fallback: strategist greedy), plus rollout
    arbitration when the best play cannot clear the blind and discards remain."""
    run = env.run
    rnd = run.round
    base_choice = None
    if self.commit is not None and rnd.boss_id != "b_eye":  # The Mouth wants one hand type: the spam hand
        base_choice = _committed_choice(self, env, mask)
    if base_choice is None:
        base_choice = StrategistAgent._play_phase(self, env, mask)
    kind, payload = codec.decode_kind(base_choice)
    if kind not in ("play", "discard"):
        return base_choice  # the strategist used a consumable or sold something
    need = rnd.chips_needed - rnd.chips_scored
    if rnd.hands_left == 0:
        return _pad_action(env, mask, base_choice)
    # Engine-level view of the strategist's choice.
    engine_idxs = tuple(sorted(env._hand_order[p] for p in payload))
    plays = _best_plays(run, self.top_k)
    if not plays:
        return _pad_action(env, mask, base_choice)
    best_score, _best_play = plays[0]
    if best_score >= need:
        return _pad_action(env, mask, base_choice)
    at_risk = best_score * rnd.hands_left < self.risk_margin * need
    if not at_risk or (rnd.discards_left == 0 and len(plays) < 2):
        return _pad_action(env, mask, base_choice)
    # Arbitrate the top plays (and the discard shapes when discards remain)
    # by rollouts of the rest of the blind.
    options: list[tuple[str, tuple[int, ...], int]] = [("play", idxs, None) for _, idxs in plays[:4]]
    if kind == "play" and engine_idxs not in {o[1] for o in options}:
        options.append(("play", engine_idxs, base_choice))
    if rnd.discards_left == 0:
        pass
    elif kind == "discard":
        options.append(("discard", engine_idxs, base_choice))
    if rnd.discards_left > 0 and rnd.discards_used == 0 and any(j.id == "j_burnt" for j in run.jokers):
        from collections import Counter

        counts = Counter(c.rank for c in rnd.hand if not c.is_stone)
        pairs = sorted((r for r, n in counts.items() if n >= 2), key=int)
        if pairs:
            low = [i for i, c in enumerate(rnd.hand) if not c.is_stone and c.rank == pairs[0]][:2]
            d = tuple(sorted(low))
            if _legal_discard(run, d):
                remap = {eng: slot for slot, eng in enumerate(env._hand_order)}
                subset = tuple(sorted(remap[i] for i in d))
                sidx = codec._SUBSET5_INDEX.get(subset)
                if sidx is not None and mask[codec.DISCARD_BASE + sidx]:
                    return codec.DISCARD_BASE + sidx
    if rnd.discards_left > 0:
        simple = _simple_discard(run)
        if simple is not None and simple != engine_idxs:
            options.append(("discard", simple, None))
        seen_d = {o[1] for o in options if o[0] == "discard"}
        for d in _draw_discards(run):
            if d not in seen_d:
                options.append(("discard", d, None))
                seen_d.add(d)
    rng = self._sim_rng(run, f"roll:{rnd.hands_left}:{rnd.discards_left}:{len(rnd.hand)}")
    scores = []
    n_roll = self.rollouts * (2 if rnd.boss_id is not None else 1)  # the run is on the line
    seeds = [rng.random() for _ in range(n_roll)]  # common random numbers
    for okind, idxs, _ in options:
        scores.append(sum(rollout_clear(run, (okind, idxs), random.Random(sd)) for sd in seeds))
    pick = max(range(len(options)), key=lambda i: (scores[i], i == 0))
    okind, idxs, action = options[pick]
    if action is not None:
        return _pad_action(env, mask, action)
    remap = {eng: slot for slot, eng in enumerate(env._hand_order)}
    subset = tuple(sorted(remap[i] for i in idxs))
    base = codec.PLAY_BASE if okind == "play" else codec.DISCARD_BASE
    sidx = codec._SUBSET5_INDEX.get(subset)
    if sidx is not None and mask[base + sidx]:
        return _pad_action(env, mask, base + sidx)
    return _pad_action(env, mask, base_choice)


def _pad_action(env, mask: np.ndarray, action: int) -> int:
    """Encode `action` with junk fillers added when it is a play (see
    `_with_fillers`); any other action is returned unchanged."""
    kind, payload = codec.decode_kind(action)
    if kind != "play":
        return action
    run = env.run
    idxs = tuple(sorted(env._hand_order[p] for p in payload))
    padded = _with_fillers(run, idxs)
    if padded == idxs:
        return action
    remap = {eng: slot for slot, eng in enumerate(env._hand_order)}
    try:
        subset = tuple(sorted(remap[i] for i in padded))
    except KeyError:
        return action
    sidx = codec._SUBSET5_INDEX.get(subset)
    if sidx is not None and mask[codec.PLAY_BASE + sidx]:
        return codec.PLAY_BASE + sidx
    return action


PlannerAgent._play_phase = _planner_play_phase
PlannerAgent.rollouts = 40  # 10 was measurably worse: 13% of lost blinds cleared with 40
PlannerAgent.risk_margin = 2.0  # arbitrate by rollouts when best play x hands left < margin x need
