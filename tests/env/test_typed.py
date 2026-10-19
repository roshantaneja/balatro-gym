"""Typed decision model: question schema, action labels, model plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from balatro_gym.engine import Phase
from balatro_gym.env import action_codec as codec
from balatro_gym.env.envs import BalatroEnv
from balatro_gym.typed import NEXT_ACTION, WILL_WIN, describe_action, questions_for


def test_questions_and_labels_cover_every_legal_action():
    env = BalatroEnv()
    _obs, _info = env.reset(seed=11)
    assert questions_for(env) == [NEXT_ACTION, WILL_WIN]
    env.step(codec.SELECT_BLIND)
    assert env.run.phase == Phase.PLAYING
    mask = env.action_masks()
    labels = {describe_action(env, i) for i in np.flatnonzero(mask)}
    assert any(label.startswith("play ") for label in labels)
    assert any(label.startswith("discard ") for label in labels)
    assert all(label for label in labels)


def test_typed_net_forward_and_calibration():
    torch = pytest.importorskip("torch")
    from balatro_gym.env.obs.encoder import ObsEncoder
    from balatro_gym.typed.model import (
        TypedDecisionNet,
        expected_calibration_error,
        fit_temperature,
        obs_to_tensors,
    )

    env = BalatroEnv()
    obs, info = env.reset(seed=2)
    net = TypedDecisionNet(ObsEncoder().space(), codec.N_ACTIONS, hidden=32, emb=4)
    logits, win = net(obs_to_tensors(obs))
    assert logits.shape == (1, codec.N_ACTIONS) and win.shape == (1,)
    mask = torch.as_tensor(info["action_mask"])[None]
    probs = net.masked_log_probs(logits, mask).exp()
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert float(probs[0][~mask[0]].sum()) < 1e-6  # illegal actions get no mass
    targets = torch.as_tensor([int(np.flatnonzero(info["action_mask"])[0])])
    t = fit_temperature(logits.detach(), mask, targets)
    assert 0.2 <= t <= 8.0
    assert expected_calibration_error(np.array([0.9, 0.6]), np.array([1.0, 0.0])) > 0


def test_round_options_are_legal_and_scoreable():
    torch = pytest.importorskip("torch")
    from balatro_gym.env.obs.encoder import ObsEncoder
    from balatro_gym.typed.candidates import (
        MAX_OPTIONS,
        OPTION_FEATS,
        padded_options,
        round_options,
    )
    from balatro_gym.typed.model import TypedDecisionNet, obs_to_tensors

    env = BalatroEnv()
    obs, _info = env.reset(seed=7)
    obs, _r, _t, _tr, info = env.step(codec.SELECT_BLIND)
    mask = info["action_mask"]
    actions, feats = round_options(env, mask)
    assert actions and len(actions) == len(set(actions)) <= MAX_OPTIONS
    assert all(mask[a] for a in actions)
    assert feats.shape == (len(actions), OPTION_FEATS)
    assert any(codec.decode_kind(a)[0] == "discard" for a in actions)
    acts, fm, label_idx = padded_options(env, mask, actions[2])
    assert acts.shape == (MAX_OPTIONS,) and fm.shape == (MAX_OPTIONS, OPTION_FEATS) and label_idx == 2
    net = TypedDecisionNet(ObsEncoder().space(), codec.N_ACTIONS, hidden=32, emb=4)
    with torch.no_grad():
        chosen, conf = net.decode_options(obs_to_tensors(obs), actions, feats)
    assert chosen in actions and 0.0 < conf <= 1.0
