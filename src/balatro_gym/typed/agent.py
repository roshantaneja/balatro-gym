"""TypedModelAgent: answers the typed questions for a live env and plays it."""

from __future__ import annotations

import numpy as np

from .questions import NEXT_ACTION, WILL_WIN, Answer, describe_action


class TypedModelAgent:
    """Loads a checkpoint from `scripts/typed/train.py`.

    `ask(env, obs, mask)` returns one `Answer` per question (the next action
    with its calibrated confidence, and the win probability); `act` is the
    baseline-style interface used by `balatro_gym.env.eval.evaluate`.
    """

    def __init__(self, path: str, sample: bool = False, seed: int = 0, play_with: str | None = None):
        """`play_with`: None (the model plays every phase), "strategist" or
        "planner" to hand the in-round play/discard decisions to that agent
        while the model answers blind-select, shop and pack questions."""
        import torch

        from balatro_gym.content import load_all
        from balatro_gym.env.obs.encoder import ObsEncoder

        from .model import TypedDecisionNet

        load_all()  # the encoder's ID vocabularies are the content registries
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.net = TypedDecisionNet(ObsEncoder().space(), cfg["n_actions"], cfg["hidden"], cfg["emb"])
        self.net.load_state_dict(ckpt["state_dict"], strict=False)
        self.options = bool(ckpt.get("metrics", {}).get("option_n", 0))
        self.net.eval()
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        self.metrics = ckpt.get("metrics", {})
        self._player = None
        if play_with == "strategist":
            from balatro_gym.env.baselines import StrategistAgent

            self._player = StrategistAgent(seed=seed)
        elif play_with == "planner":
            from balatro_gym.env.planner import PlannerAgent

            self._player = PlannerAgent(seed=seed)

    def _probs(self, obs, mask: np.ndarray):
        import torch

        from .model import obs_to_tensors

        with torch.no_grad():
            logits, win_logit = self.net(obs_to_tensors(obs))
            lp = self.net.masked_log_probs(logits, torch.as_tensor(mask)[None])
            return lp.exp()[0].numpy(), float(self.net.win_probability(win_logit)[0])

    def _structured(self, env, obs, mask: np.ndarray) -> tuple[int | None, float]:
        """The in-round answer: the best of the named candidate options when
        the checkpoint learned to score them, else the composed play."""
        if env._env_phase() != "playing":
            return None, 0.0
        import torch

        from .candidates import round_options
        from .model import obs_to_tensors

        with torch.no_grad():
            if self.options:
                actions, feats = round_options(env, mask)
                if actions:
                    return self.net.decode_options(obs_to_tensors(obs), actions, feats)
            return self.net.decode_play(obs_to_tensors(obs), mask)

    def ask(self, env, obs, mask: np.ndarray) -> dict[str, Answer]:
        probs, p_win = self._probs(obs, mask)
        idx, conf = self._structured(env, obs, mask)
        if idx is None:
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
        return {
            NEXT_ACTION.name: Answer(NEXT_ACTION.name, idx, conf, describe_action(env, idx)),
            WILL_WIN.name: Answer(WILL_WIN.name, p_win >= 0.5, p_win if p_win >= 0.5 else 1 - p_win,
                                  f"P(win) = {p_win:.2f}"),
        }

    def act(self, env, obs, mask: np.ndarray) -> int:
        if self._player is not None and env._env_phase() == "playing":
            return self._player._play_phase(env, mask)
        idx, _conf = self._structured(env, obs, mask)
        if idx is not None and not self.sample:
            return idx
        probs, _ = self._probs(obs, mask)
        if self.sample:
            probs = probs * mask
            probs = probs / probs.sum() if probs.sum() > 0 else mask / mask.sum()
            return int(self.rng.choice(len(probs), p=probs))
        idx = int(np.argmax(probs))
        if not mask[idx]:
            idx = int(np.flatnonzero(mask)[0])
        return idx
