"""Baseline agents: the benchmark floor for trained policies."""

from __future__ import annotations

import numpy as np

from balatro_gym.engine import Phase, score_play

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
