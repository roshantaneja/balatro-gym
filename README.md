# balatro-gym

A pure-Python reimplementation of [Balatro](https://www.playbalatro.com/) exposed as a
[Gymnasium](https://gymnasium.farama.org/) environment for reinforcement learning,
with exact legal-action masking (works out of the box with SB3-contrib MaskablePPO
and CleanRL-style `info["action_mask"]` training loops).

Rules are implemented from the [Balatro wiki](https://balatrogame.fandom.com/) (a
full scrape lives in `balatro_wiki/`). Fan project for RL research — unaffiliated
with LocalThunk or Playstack.

## What's implemented (v1)

- **Full run loop**: antes 1-8, small/big/boss blinds, skip-for-tag, shop, booster
  packs, vouchers, win/loss. White stake; Red/Blue/Yellow decks.
- **Faithful scoring**: the wiki's 7-stage activation sequence — boss effects,
  on-played jokers, per-card scoring (enhancement → seal → edition → on-scored
  jokers) with red-seal/joker retriggers, held-card effects, joker editions,
  independent jokers, Blueprint/Brainstorm copy chains. Every score comes with a
  step-by-step audit log (`ScoreResult.explain()`).
- **Content**: 40+ jokers covering every effect mechanic (scaling, decay,
  self-destruct, retriggers, copies, rule modifiers like Four Fingers/Splash/
  Pareidolia), all 12 planets, 13 tarots, all enhancements/editions/seals,
  14 boss blinds including the Violet Vessel finisher.
- **Three environments** (one engine, identical spaces — weights transfer):
  - `balatro_gym/BalatroBlind-v0` — single-blind curriculum env, randomized
    scenarios, shaped reward. Fastest to learn.
  - `balatro_gym/BalatroScriptedShop-v0` — full runs, heuristic auto-shop; the
    agent only plays hands.
  - `balatro_gym/Balatro-v0` — the whole game.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"     # engine + gymnasium
pytest                                     # 259 tests

# play it yourself
python scripts/play_repl.py --env full --seed 7

# random/heuristic baselines
python - <<'PY'
from balatro_gym.env.envs import BalatroEnv
from balatro_gym.env.baselines import RandomLegalAgent, HeuristicAgent
from balatro_gym.env.eval import evaluate
env = BalatroEnv()
print("random:   ", evaluate(env, RandomLegalAgent(), seeds=20).summary())
print("heuristic:", evaluate(env, HeuristicAgent(), seeds=20).summary())
PY

# train
uv pip install -e ".[train]"
python examples/train_maskable_ppo.py --env balatro_gym/BalatroBlind-v0 --steps 200000
```

## Using the env

```python
import gymnasium as gym
import numpy as np
import balatro_gym.env  # registers envs

env = gym.make("balatro_gym/Balatro-v0")
obs, info = env.reset(seed=0)
mask = info["action_mask"]                # bool[556]; also env.unwrapped.action_masks()
action = np.random.choice(np.flatnonzero(mask))
obs, reward, terminated, truncated, info = env.step(action)
```

- **Action space**: flat `Discrete(556)` — blind select/skip, all C(8,k≤5) play and
  discard subsets, consumable use + target selection, shop buys/sells/rerolls,
  pack picks. Illegal actions are masked; boss restrictions (e.g. The Psychic's
  must-play-5) fall out of the engine's legal-action enumeration.
- **Observation**: `Dict` of dense float32 features (hand cards, hand levels, deck
  composition, blind/boss context, shop) plus int64 ID vectors (jokers,
  consumables, boss) meant for embedding lookup.
- **Reward**: `+1` per blind cleared, `+5` for winning (default); optional shaped
  mode adds capped, ante-weighted score-progress deltas. Pluggable via
  `RewardConfig`.

## Architecture

```
src/balatro_gym/
  engine/    # generic machinery: state, scoring pipeline, hand eval, shop, RNG,
             # phase machine + legal_actions()  (imports nothing from content)
  content/   # every joker/consumable/boss/deck/voucher/tag as a registered
             # effect class — auto-discovered, drop-in files
  env/       # Gymnasium layer: action codec, obs encoder, rewards, 3 envs
```

Fidelity is enforced by golden YAML scoring tests (hand-derived expected values
with derivations in comments), per-joker unit tests, and hypothesis property
tests (evaluator invariants, determinism, clone independence). Engine state is
cheaply cloneable and JSON-serializable for future MCTS work.

## Performance

Random-legal-action rollouts on one core: ~5,500 engine steps/sec, ~1,000 full
env steps/sec (obs encoding included). `python benchmarks/bench_steps.py`.

## Roadmap

Remaining content (the other ~110 jokers, spectrals, more bosses/vouchers/tags/
decks, stakes/stickers), face-down obs handling for The Wheel/Mark, Observatory/
Plasma scoring stages, and an AlphaZero-style search on top of the cloneable
engine state.
