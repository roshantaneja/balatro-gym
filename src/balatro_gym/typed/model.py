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
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=False)
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
    for t in np.exp(np.linspace(math.log(0.25), math.log(8.0), 60)):
        lp = torch.log_softmax((logits / float(t)).masked_fill(~mask.bool(), -1e9), dim=-1)
        nll = -lp.gather(1, targets[:, None]).mean().item()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t
