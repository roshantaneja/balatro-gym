"""In-round options for the typed model: a multiple-choice answer.

Instead of naming hand slots, the in-round question offers the model a short
list of named candidate actions (the strongest plays, each with and without
junk padding, and the sensible discard shapes), each described by features a
player can see: the immediate score against what the blind still needs, the
hand type and its level, how many cards score, hands and discards left. The
model scores the options and the answer is the best one.
"""

from __future__ import annotations

import numpy as np

from balatro_gym.engine import HandType, evaluate_hand
from balatro_gym.engine.scoring import fold_eval_rules
from balatro_gym.env import action_codec as codec
from balatro_gym.env.baselines import score_candidate

MAX_OPTIONS = 24
N_HAND_TYPES = len(HandType)
OPTION_FEATS = 12 + N_HAND_TYPES


def _encode(env, engine_idxs: tuple[int, ...], kind: str) -> int | None:
    remap = {eng: slot for slot, eng in enumerate(env._hand_order)}
    try:
        subset = tuple(sorted(remap[i] for i in engine_idxs))
    except KeyError:
        return None
    sidx = codec._SUBSET5_INDEX.get(subset)
    if sidx is None:
        return None
    return (codec.PLAY_BASE if kind == "play" else codec.DISCARD_BASE) + sidx


def _pad_to_five(run, idxs: tuple[int, ...]) -> tuple[int, ...] | None:
    """Fill a play up to five cards with the lowest cards not already in it."""
    cards = run.round.hand
    extra = sorted((i for i in range(len(cards)) if i not in idxs), key=lambda i: int(cards[i].rank))
    out = tuple(sorted(idxs + tuple(extra[: 5 - len(idxs)])))
    return out if len(out) == 5 else None


def round_options(env, mask: np.ndarray) -> tuple[list[int], np.ndarray]:
    """(action indices, feature matrix [n, OPTION_FEATS]) for the current round."""
    from balatro_gym.env.planner import (
        _best_plays,
        _draw_discards,
        _engine_candidates,
        _legal_play,
        _simple_discard,
        _with_fillers,
    )

    run = env.run
    rnd = run.round
    if rnd is None:
        return [], np.zeros((0, OPTION_FEATS), dtype=np.float32)
    need = max(rnd.chips_needed - rnd.chips_scored, 1.0)
    pace = need / max(rnd.hands_left, 1)
    rules = fold_eval_rules(run)
    actions: list[int] = []
    feats: list[np.ndarray] = []

    def add(action: int | None, engine_idxs: tuple[int, ...], kind: str) -> None:
        if action is None or action in actions or not mask[action] or len(actions) >= MAX_OPTIONS:
            return
        f = np.zeros(OPTION_FEATS, dtype=np.float32)
        if kind == "play":
            res = score_candidate(env, engine_idxs)
            hr = res.hand_result
            score = float(res.score)
            f[0] = min(score / need, 3.0)
            f[1] = min(score / pace, 3.0)
            f[2] = 1.0 if score >= need else 0.0
            f[3] = len(hr.scoring_cards) / 5.0
            f[4] = run.hand_levels[hr.hand_type] / 10.0
            f[12 + int(hr.hand_type) - 1] = 1.0
        else:
            played = [rnd.hand[i] for i in engine_idxs]
            hr = evaluate_hand(played, rules) if played else None
            f[5] = 1.0  # is a discard
            if hr is not None:
                f[12 + int(hr.hand_type) - 1] = 1.0
        f[6] = len(engine_idxs) / 5.0
        f[7] = rnd.hands_left / 4.0
        f[8] = rnd.discards_left / 4.0
        f[9] = run.ante / 8.0
        f[10] = 1.0 if rnd.boss_id else 0.0
        f[11] = min(np.log10(need) / 6.0, 1.0)
        actions.append(action)
        feats.append(f)

    plays = _best_plays(run, 8)
    if not plays:  # a boss forbids every structural play (The Psychic: five cards only)
        plays = [(0.0, _pad_to_five(run, idxs)) for _, idxs in _engine_candidates(run, 8)]
        plays = [(s, idxs) for s, idxs in plays if idxs is not None and _legal_play(run, idxs)]
    for _score, idxs in plays:
        add(_encode(env, idxs, "play"), idxs, "play")
        padded = _with_fillers(run, tuple(sorted(idxs)))
        if padded != tuple(sorted(idxs)):
            add(_encode(env, padded, "play"), padded, "play")
    if rnd.discards_left > 0:
        d = _simple_discard(run)
        if d:
            add(_encode(env, d, "discard"), d, "discard")
        for d in _draw_discards(run):
            add(_encode(env, d, "discard"), d, "discard")
    if not feats:
        return [], np.zeros((0, OPTION_FEATS), dtype=np.float32)
    return actions, np.stack(feats)


def padded_options(env, mask: np.ndarray, label: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Fixed-size arrays for datasets: actions [MAX_OPTIONS] (-1 = none),
    features [MAX_OPTIONS, OPTION_FEATS], and the label's option index (-1
    when the label is not among the options)."""
    actions, feats = round_options(env, mask)
    acts = np.full(MAX_OPTIONS, -1, dtype=np.int64)
    fm = np.zeros((MAX_OPTIONS, OPTION_FEATS), dtype=np.float32)
    n = len(actions)
    acts[:n] = actions
    fm[:n] = feats
    idx = actions.index(label) if label is not None and label in actions else -1
    return acts, fm, idx
