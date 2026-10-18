"""LookaheadPlannerAgent: shop decisions chosen by rolling out the rest of the ante.

The PlannerAgent values each shop item in isolation. This agent keeps that
policy for everything except the contested shop decisions (buy a joker or a
pack, reroll, leave) and settles those by simulation: every candidate is
applied on a cloned run whose RNG streams are re-seeded (so the rollout
cannot peek at what the real reroll or the real next shop will contain),
the rest of the ante is played by a cheap inner planner, and the candidate
with the best outcome wins. The outcome score is: the ante cleared (or the
run won), plus the projected power margin against the next ante's boss and
a little for money carried over; a lost rollout scores by the blinds it
cleared first.
"""

from __future__ import annotations

import math

import numpy as np

from balatro_gym.engine import HandType, Phase, cardops
from balatro_gym.engine.levels import ante_base_chips
from balatro_gym.engine.rng import RunRNG

from . import action_codec as codec
from .baselines import StrategistAgent, _classify
from .planner import PlannerAgent

# P(win | state at the start of an ante), logistic regression on 1,357 blind-select
# states from 77 planner runs: features (1, log power margin vs the ante's boss,
# pair level, xMult jokers, ante/8, (ante/8)^2, margin x ante/8).
WIN_MODEL = (-2.0959, -0.1997, 0.1564, 0.1206, -1.7614, 0.7058, 2.475)


class LookaheadPlannerAgent(PlannerAgent):
    def __init__(self, *args, samples_ahead: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples_ahead = samples_ahead
        self._inner_env = None
        self._inner = PlannerAgent(seed=1, samples=2, top_k=4, commit=self.commit)
        self._inner.rollouts = 4
        # The inner planner plays rounds with the strategist's greedy policy:
        # the endgame arbitration is the bulk of a rollout's cost.
        self._inner._play_phase = lambda env, mask: StrategistAgent._play_phase(self._inner, env, mask)
        self.lookahead_log: list | None = None

    # ------------------------------------------------------------------ shop
    def _shop_phase(self, env, mask: np.ndarray) -> int:
        action = super()._shop_phase(env, mask)
        run = env.run
        shop = run.shop
        if shop is None or run.ante >= 8:
            return action
        kind, payload = codec.decode_kind(action)
        contested = kind in ("reroll", "leave_shop", "buy_pack") or (
            kind == "buy_card" and shop.card_slots[payload].kind == "joker" and shop.card_slots[payload].cost >= 4
        )
        if not contested:
            return action
        cands = [action]
        if mask[codec.REROLL_IDX] and codec.REROLL_IDX not in cands and run.money - shop.reroll_cost >= 0:
            cands.append(codec.REROLL_IDX)
        if mask[codec.LEAVE_SHOP_IDX] and codec.LEAVE_SHOP_IDX not in cands:
            cands.append(codec.LEAVE_SHOP_IDX)
        # the priciest affordable joker on offer, if the greedy pick was not a joker
        if kind != "buy_card":
            best = None
            for i, it in enumerate(shop.card_slots[: codec.N_BUY_CARD_SLOTS]):
                idx = codec.BUY_CARD_BASE + i
                if it.kind == "joker" and mask[idx] and (best is None or it.cost > best[0]):
                    best = (it.cost, idx)
            if best is not None:
                cands.append(best[1])
        if len(cands) == 1:
            return action
        # The shop before the boss decides the ante: sample more futures there.
        n = self.samples_ahead + (2 if run.blind_index == 2 else 0)
        scores = {}
        for a in cands:
            scores[a] = sum(self._rollout(env, a, k) for k in range(n)) / n
        chosen = max(cands, key=lambda a: (scores[a], a == action))
        if self.lookahead_log is not None:
            self.lookahead_log.append((run.ante, run.blind_index, run.money, {codec.decode_kind(a)[0]: round(scores[a], 3) for a in cands}, codec.decode_kind(chosen)[0]))
        return chosen

    # --------------------------------------------------------------- rollout
    def _rollout(self, env, action: int, k: int) -> float:
        from .envs import BalatroEnv

        run = env.run
        if self._inner_env is None:
            stake = getattr(env.stake, "value", env.stake)
            self._inner_env = BalatroEnv(deck=env.deck_id, stake=stake)
            self._inner_env.reset(seed=0)
        ienv = self._inner_env
        clone = run.clone()
        clone.rng = RunRNG(f"{run.seed}:lookahead:{run.ante}:{run.blind_index}:{run.stats.get('hands_played_run', 0)}:{k}")
        ienv.game.run = clone
        ienv._mask_cache = None
        ienv._pending = None
        ienv._refresh_hand_order()
        obs, info = ienv._obs(), ienv._info()
        if not info["action_mask"][action]:
            return -1.0
        start_ante = clone.ante
        obs, _r, term, _trunc, info = ienv.step(action)
        steps = 0
        self._inner._power_cache.clear()
        while not term and ienv.run.ante == start_ante and steps < 400:
            a = self._inner.act(ienv, obs, info["action_mask"])
            obs, _r, term, _trunc, info = ienv.step(a)
            steps += 1
        r2 = ienv.run
        if r2.phase == Phase.WON:
            return 1.0
        if r2.phase == Phase.GAME_OVER:
            return 0.02 * r2.blind_index  # blinds cleared this ante before dying
        proj = self._inner.power(self._inner.projected(r2), salt="lookahead")
        req = ante_base_chips(r2.ante, r2.stake) * r2.mods.get("ante_scaling", 1) * 2.0
        margin = math.log(max(proj, 1.0) * max(cardops.effective_hands(r2), 1) / req)
        pair = min(r2.hand_levels[HandType.PAIR], 15)
        nx = min(sum(1 for j in r2.jokers if _classify(j.id) == "x"), 3)
        a = r2.ante / 8.0
        feats = (1.0, margin, pair, nx, a, a * a, margin * a)
        logit = sum(w * f for w, f in zip(WIN_MODEL, feats))
        p_win = 1.0 / (1.0 + math.exp(-logit))
        # Money carried over is a tie-breaker only: the model has no money term.
        return p_win + 0.0005 * min(max(r2.money, 0), 60) * max(0, 8 - r2.ante) / 6.0
