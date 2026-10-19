"""The typed decision network (torch; install the `train` extra).

Input: the env's Dict observation. ID fields (jokers, consumables, shop and
pack items, vouchers on offer, boss, skip tags, deck) go through embeddings;
dense fields are flattened and concatenated. Two heads: a distribution over
the flat action space (masked to the legal actions, with a calibration
temperature fitted after training) and a win probability for the run.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from .candidates import OPTION_FEATS

HAND_SLOTS = 12
OPTION_HIDDEN = 64

ID_KEYS = (
    "joker_ids", "consumable_ids", "shop_item_ids", "shop_voucher_ids", "pack_item_ids",
    "boss_id", "ante_boss_id", "blind_tag_ids", "deck_id",
)


def obs_to_tensors(obs: dict, batch: bool = False) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in obs.items():
        arr = np.asarray(v)
        if not batch:
            arr = arr[None]
        if k in ID_KEYS:
            out[k] = torch.as_tensor(arr, dtype=torch.long)
        else:
            out[k] = torch.as_tensor(arr, dtype=torch.float32)
    return out


class TypedDecisionNet(nn.Module):
    def __init__(self, obs_space: spaces.Dict, n_actions: int, hidden: int = 512, emb: int = 16):
        super().__init__()
        self.keys = sorted(obs_space.spaces)
        self.embeddings = nn.ModuleDict()
        self.id_slots: dict[str, int] = {}
        dense = 0
        for k in self.keys:
            sp = obs_space.spaces[k]
            if k in ID_KEYS:
                vocab = int(np.max(sp.nvec)) + 1
                self.embeddings[k] = nn.Embedding(vocab, emb)
                self.id_slots[k] = int(np.prod(sp.shape))
            else:
                dense += int(np.prod(sp.shape))
        total = dense + emb * sum(self.id_slots.values())
        self.body = nn.Sequential(
            nn.Linear(total, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.action_head = nn.Linear(hidden, n_actions)
        self.win_head = nn.Linear(hidden, 1)
        # Structured play answer: play-or-discard, which hand slots, how many.
        # A 12-way multi-select is far easier to learn than a 1,585-way softmax.
        self.kind_head = nn.Linear(hidden, 2)
        self.slot_head = nn.Linear(hidden, HAND_SLOTS)
        self.size_head = nn.Linear(hidden, 5)
        # Multiple-choice in-round answer: score each named candidate action
        # (see `candidates.py`) from the state features and the option's own
        # features; the answer is a softmax over the offered options.
        self.option_proj = nn.Linear(hidden, OPTION_HIDDEN)
        self.option_mlp = nn.Sequential(
            nn.Linear(OPTION_HIDDEN + OPTION_FEATS, OPTION_HIDDEN), nn.ReLU(), nn.Linear(OPTION_HIDDEN, 1)
        )
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.option_log_temperature = nn.Parameter(torch.zeros(1), requires_grad=False)
        # Platt scaling for the win head: sigmoid(a * logit + b), fitted on held-out runs.
        self.win_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.win_shift = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.n_actions = n_actions
        self.hidden = hidden
        self.emb = emb

    def features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        b = next(iter(obs.values())).shape[0]
        for k in self.keys:
            x = obs[k]
            if k in ID_KEYS:
                vocab = self.embeddings[k].num_embeddings
                parts.append(self.embeddings[k](x.long().clamp(0, vocab - 1)).reshape(b, -1))
            else:
                parts.append(x.float().reshape(b, -1))
        return torch.cat(parts, dim=1)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(self.features(obs))
        return self.action_head(h), self.win_head(h).squeeze(-1)

    def play_heads(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.body(self.features(obs))
        return self.kind_head(h), self.slot_head(h), self.size_head(h)

    def option_scores(self, h: torch.Tensor, feats: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        """Logits over the offered options: h [B, hidden], feats [B, M, F],
        acts [B, M] (-1 marks an empty slot, which gets no mass)."""
        ctx = self.option_proj(h)[:, None, :].expand(-1, feats.shape[1], -1)
        logits = self.option_mlp(torch.cat([ctx, feats], dim=-1)).squeeze(-1)
        return logits.masked_fill(acts < 0, -1e9)

    def option_log_probs(self, logits: torch.Tensor, calibrated: bool = True) -> torch.Tensor:
        if calibrated:
            logits = logits / torch.exp(self.option_log_temperature)
        return torch.log_softmax(logits, dim=-1)

    def decode_options(self, obs: dict[str, torch.Tensor], actions: list[int], feats: np.ndarray) -> tuple[int, float]:
        """Best of the offered options and its (softmax) confidence."""
        h = self.body(self.features(obs))
        f = torch.as_tensor(feats, dtype=torch.float32)[None]
        a = torch.as_tensor(actions, dtype=torch.long)[None]
        probs = self.option_log_probs(self.option_scores(h, f, a))[0].exp()
        best = int(probs.argmax())
        return actions[best], float(probs[best])

    def decode_play(self, obs: dict[str, torch.Tensor], mask: np.ndarray) -> tuple[int | None, float]:
        """Compose a legal play/discard action from the structured heads.
        Returns (action index, confidence) or (None, 0) when no legal
        composition exists (boss restrictions, no discards left)."""
        from balatro_gym.env import action_codec as codec

        kind_logit, slot_logit, size_logit = self.play_heads(obs)
        kind_p = torch.softmax(kind_logit[0], dim=-1)
        slot_p = torch.sigmoid(slot_logit[0])
        size_p = torch.softmax(size_logit[0], dim=-1)
        order = torch.argsort(slot_p, descending=True).tolist()
        bases = [codec.PLAY_BASE, codec.DISCARD_BASE]
        for kind in torch.argsort(kind_p, descending=True).tolist():
            for size in torch.argsort(size_p, descending=True).tolist():
                subset = tuple(sorted(order[: size + 1]))
                idx = codec._SUBSET5_INDEX.get(subset)
                if idx is None:
                    continue
                action = bases[kind] + idx
                if mask[action]:
                    conf = float(kind_p[kind] * size_p[size] * slot_p[list(subset)].mean())
                    return action, conf
        return None, 0.0

    def win_probability(self, win_logit: torch.Tensor, calibrated: bool = True) -> torch.Tensor:
        if calibrated:
            win_logit = self.win_scale * win_logit + self.win_shift
        return torch.sigmoid(win_logit)

    def masked_log_probs(self, logits: torch.Tensor, mask: torch.Tensor, calibrated: bool = True) -> torch.Tensor:
        if calibrated:
            logits = logits / torch.exp(self.log_temperature)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        return torch.log_softmax(logits, dim=-1)


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Mean |confidence - accuracy| over equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor) -> float:
    """Temperature that minimises the masked NLL on held-out decisions."""
    best_t, best_nll = 1.0, math.inf
    for t in np.exp(np.linspace(math.log(0.1), math.log(8.0), 60)):
        lp = torch.log_softmax((logits / float(t)).masked_fill(~mask.bool(), -1e9), dim=-1)
        nll = -lp.gather(1, targets[:, None]).mean().item()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def fit_platt(logits: torch.Tensor, targets: torch.Tensor, steps: int = 500) -> tuple[float, float]:
    """Scale and shift for a binary logit that minimise the held-out BCE."""
    a = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=steps)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(a * logits + b, targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.item()), float(b.item())
