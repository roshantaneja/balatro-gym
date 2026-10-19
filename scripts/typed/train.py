"""Train the typed decision model on planner-labelled decisions.

    python scripts/typed/train.py --data "data/typed/*.npz" --out models/typed_v1.pt

Behaviour cloning of the planner's action with a legal-action mask, plus a
win-probability head, then temperature scaling on held-out runs so the
confidences are calibrated. Reports accuracy by phase, the expected
calibration error of the action confidence and of the win probability.
"""

from __future__ import annotations

import argparse
import glob
import json
import time

import numpy as np
import torch
from torch import nn

from balatro_gym.content import load_all
from balatro_gym.env import action_codec as codec
from balatro_gym.env.obs.encoder import ObsEncoder
from balatro_gym.typed.model import (
    ID_KEYS,
    TypedDecisionNet,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
)

PHASE_NAMES = {0: "blind_select", 1: "playing", 2: "shop", 3: "pack_open", 4: "target"}


def load(pattern: str) -> dict[str, np.ndarray]:
    parts = [dict(np.load(f)) for f in sorted(glob.glob(pattern))]
    if not parts:
        raise SystemExit(f"no data matches {pattern}")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def structured_labels(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kind: 0 play / 1 discard / -1 other, slot multi-hot, size-1) per decision."""
    kind = np.full(len(actions), -1, dtype=np.int64)
    slots = np.zeros((len(actions), 12), dtype=np.float32)
    size = np.zeros(len(actions), dtype=np.int64)
    for i, a in enumerate(actions):
        k, payload = codec.decode_kind(int(a))
        if k in ("play", "discard"):
            kind[i] = 0 if k == "play" else 1
            slots[i, list(payload)] = 1.0
            size[i] = len(payload) - 1
    return kind, slots, size


def batches(data: dict, idx: np.ndarray, size: int, obs_keys: list[str], shuffle: bool, rng):
    order = rng.permutation(idx) if shuffle else idx
    for s in range(0, len(order), size):
        sel = order[s : s + size]
        obs = {}
        for k in obs_keys:
            arr = data[k][sel]
            obs[k] = torch.as_tensor(arr, dtype=torch.long if k in ID_KEYS else torch.float32)
        yield (
            obs,
            torch.as_tensor(data["mask"][sel]),
            torch.as_tensor(data["action"][sel], dtype=torch.long),
            torch.as_tensor(data["won"][sel], dtype=torch.float32),
            torch.as_tensor(data["kind"][sel], dtype=torch.long),
            torch.as_tensor(data["slots"][sel], dtype=torch.float32),
            torch.as_tensor(data["size"][sel], dtype=torch.long),
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/typed/*.npz")
    p.add_argument("--out", default="models/typed_v1.pt")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--emb", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = load(args.data)
    data["kind"], data["slots"], data["size"] = structured_labels(data["action"])
    load_all()  # the encoder's ID vocabularies are the content registries
    space = ObsEncoder().space()
    obs_keys = sorted(space.spaces)
    seeds = np.unique(data["seed"])
    rng.shuffle(seeds)
    if len(seeds) >= 5:  # hold out whole runs so calibration is measured on unseen games
        n_val = max(1, int(len(seeds) * args.val_frac))
        val_seeds = set(seeds[:n_val].tolist())
        is_val = np.array([s in val_seeds for s in data["seed"]])
    else:  # tiny datasets (smoke tests): split records
        is_val = rng.random(len(data["seed"])) < args.val_frac
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"{len(data['action'])} decisions from {len(seeds)} runs; "
          f"train {len(train_idx)} / val {len(val_idx)}; win rate {data['won'][::1].mean():.3f}")

    net = TypedDecisionNet(space, codec.N_ACTIONS, args.hidden, args.emb)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        net.train()
        t, tot, n = time.time(), 0.0, 0
        for obs, mask, action, won, kind, slots, size in batches(data, train_idx, args.batch, obs_keys, True, rng):
            h = net.body(net.features(obs))
            logits, win_logit = net.action_head(h), net.win_head(h).squeeze(-1)
            lp = net.masked_log_probs(logits, mask, calibrated=False)
            loss = -lp.gather(1, action[:, None]).mean() + 0.5 * bce(win_logit, won)
            played = kind >= 0
            if played.any():
                hp = h[played]
                loss = loss + ce(net.kind_head(hp), kind[played])
                loss = loss + bce(net.slot_head(hp), slots[played]) * 4.0
                loss = loss + ce(net.size_head(hp), size[played])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(action)
            n += len(action)
        acc = evaluate(net, data, val_idx, args.batch, obs_keys)["accuracy"]
        print(f"epoch {epoch + 1}: loss {tot / n:.3f}  val accuracy {acc:.3f}  ({time.time() - t:.0f}s)", flush=True)

    # Calibration on held-out runs.
    net.eval()
    logits_all, mask_all, act_all, win_all, won_all = [], [], [], [], []
    with torch.no_grad():
        for obs, mask, action, won, *_ in batches(data, val_idx, args.batch, obs_keys, False, rng):
            logits, win_logit = net(obs)
            logits_all.append(logits)
            mask_all.append(mask)
            act_all.append(action)
            win_all.append(win_logit)
            won_all.append(won)
    temp = fit_temperature(torch.cat(logits_all), torch.cat(mask_all), torch.cat(act_all))
    net.log_temperature.data = torch.tensor([np.log(temp)], dtype=torch.float32)
    a, b = fit_platt(torch.cat(win_all), torch.cat(won_all))
    net.win_scale.data = torch.tensor([a], dtype=torch.float32)
    net.win_shift.data = torch.tensor([b], dtype=torch.float32)
    metrics = evaluate(net, data, val_idx, args.batch, obs_keys)
    metrics["temperature"] = temp
    metrics["win_platt"] = [a, b]
    print(json.dumps(metrics, indent=1))

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "config": {"n_actions": codec.N_ACTIONS, "hidden": args.hidden, "emb": args.emb},
            "metrics": metrics,
        },
        args.out,
    )
    print(f"saved {args.out}")


def evaluate(net: TypedDecisionNet, data: dict, idx: np.ndarray, batch: int, obs_keys: list[str]) -> dict:
    net.eval()
    conf, correct, phases, p_win, won, play_hits, play_n = [], [], [], [], [], 0, 0
    with torch.no_grad():
        for obs, mask, action, w, kind, slots, size in batches(data, idx, batch, obs_keys, False, np.random.default_rng(0)):
            logits, win_logit = net(obs)
            probs = net.masked_log_probs(logits, mask).exp()
            top = probs.argmax(dim=1)
            conf.append(probs.max(dim=1).values.numpy())
            correct.append((top == action).numpy())
            p_win.append(net.win_probability(win_logit).numpy())
            won.append(w.numpy())
            for i in np.flatnonzero((kind >= 0).numpy()):  # structured decode per decision
                single = {k: v[i : i + 1] for k, v in obs.items()}
                a, _c = net.decode_play(single, mask[i].numpy())
                play_hits += int(a == int(action[i]))
                play_n += 1
    conf, correct = np.concatenate(conf), np.concatenate(correct).astype(float)
    p_win, won = np.concatenate(p_win), np.concatenate(won)
    phases = data["phase"][idx]
    out = {
        "accuracy": float(correct.mean()),
        "action_ece": expected_calibration_error(conf, correct),
        "mean_confidence": float(conf.mean()),
        "win_ece": expected_calibration_error(np.where(p_win >= 0.5, p_win, 1 - p_win),
                                              ((p_win >= 0.5) == won.astype(bool)).astype(float)),
        "win_brier": float(np.mean((p_win - won) ** 2)),
        "win_base_rate": float(won.mean()),
        "structured_play_accuracy": float(play_hits / max(play_n, 1)),
    }
    for code, name in PHASE_NAMES.items():
        sel = phases == code
        if sel.any():
            out[f"accuracy_{name}"] = float(correct[sel].mean())
    return out


if __name__ == "__main__":
    main()
