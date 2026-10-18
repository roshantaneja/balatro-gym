# balatro-gym

A pure-Python reimplementation of [Balatro](https://www.playbalatro.com/) exposed as a
[Gymnasium](https://gymnasium.farama.org/) environment for reinforcement learning,
with exact legal-action masking (works out of the box with SB3-contrib MaskablePPO
and CleanRL-style `info["action_mask"]` training loops).

The full 1.0 content set is implemented: 150 jokers, 22 tarots, 12 planets,
18 spectrals, 28 boss blinds, 32 vouchers, 15 decks, 8 stakes, 24 skip tags,
15 booster packs, every enhancement/edition/seal/sticker. Fan project for RL
research, unaffiliated with LocalThunk or Playstack. See `docs/fidelity.md` for
what was verified and where judgment calls were made.

## What's implemented

- **Full run loop**: antes 1-8 across all 8 stakes and 15 decks; small/big/boss
  blinds; skip-for-tag with the tag and the ante's boss pre-rolled and visible on
  the blind-select screen; shops (card slots, packs, vouchers, rerolls, stickers,
  discounts, free rerolls), booster packs with immediate consumable use on a dealt
  hand, joker reordering, selling jokers and consumables, boss rerolls (Director's
  Cut / Retcon), win/loss with Mr. Bones.
- **Faithful scoring**: the game's activation sequence — boss effects, on-played
  jokers, per-card scoring (enhancement → seal → edition → on-scored jokers) with
  red-seal/joker retriggers, held-card effects, joker editions, independent
  jokers, Blueprint/Brainstorm copy chains, Observatory and Plasma stages. Every
  score comes with a step-by-step audit log (`ScoreResult.explain()`).
- **Content**: every joker (scaling, decay, self-destruct, retriggers, copies, rule
  modifiers, probability, economy, board effects, Legendaries), all consumables
  (including The Soul / Black Hole pack odds), all bosses incl. the five
  finishers, all vouchers/tags/decks/stakes, Eternal/Perishable/Rental stickers.
- **Three environments** (one engine, identical spaces — weights transfer):
  - `balatro_gym/BalatroBlind-v0` — single-blind curriculum env with randomized
    scenarios (antes, jokers, levels, decks, boss blinds), shaped reward.
  - `balatro_gym/BalatroScriptedShop-v0` — full runs, strategist auto-shop and
    auto-blind-select; the agent only plays hands.
  - `balatro_gym/Balatro-v0` — the whole game (`deck=`, `stake=` kwargs).
- **Baselines**: `RandomLegalAgent`, greedy `HeuristicAgent`, `StrategistAgent`
  (a community-strategy agent aware of the full catalog: joker tiers and
  ordering classes, skip-tag values, voucher priorities, consumable usage with
  target selection, pace-based discarding, boss rerolls) and `PlannerAgent`
  (simulation-based: every shop, ordering, planet and skip decision is scored
  by measuring the board's hand-scoring power on cloned engine state, with
  rollout arbitration for endgame discards). Numbers below.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"     # engine + gymnasium
pytest                                     # 500+ tests

# play it yourself
python scripts/play_repl.py --env full --seed 7

# WATCH an agent play in the browser (strategist, heuristic, random, or a model)
python scripts/watch.py                                  # http://localhost:8712
python scripts/watch.py --agent strategist --deck d_plasma --stake gold
python scripts/watch.py --agent model --model ppo_balatro.zip

# baselines
python - <<'PY'
from balatro_gym.env.envs import BalatroEnv
from balatro_gym.env.baselines import RandomLegalAgent, HeuristicAgent, StrategistAgent
from balatro_gym.env.eval import evaluate
env = BalatroEnv()
for name, agent in [("random", RandomLegalAgent()), ("heuristic", HeuristicAgent()), ("strategist", StrategistAgent())]:
    print(name, evaluate(env, agent, seeds=20).summary())
PY

# train
uv pip install -e ".[train]"
python examples/train_maskable_ppo.py --env balatro_gym/BalatroBlind-v0 --steps 200000
python examples/train_maskable_ppo.py --env balatro_gym/BalatroScriptedShop-v0 --steps 2000000
```

## Using the env

```python
import gymnasium as gym
import numpy as np
import balatro_gym.env  # registers envs

env = gym.make("balatro_gym/Balatro-v0", deck="d_red", stake="white")
obs, info = env.reset(seed=0)
mask = info["action_mask"]                # bool[N_ACTIONS]; also env.unwrapped.action_masks()
action = np.random.choice(np.flatnonzero(mask))
obs, reward, terminated, truncated, info = env.step(action)
```

- **Action space**: flat `Discrete(3519)` — blind select / skip / boss reroll, all
  C(12,k≤5) play and discard subsets, consumable use + target selection, selling
  consumables and jokers, adjacent joker swaps, shop buys (4 card slots, 2 packs,
  3 vouchers), reroll, leave, pack picks (with target selection for card-targeting
  tarots/spectrals), skip pack. Illegal actions are masked; boss restrictions (The
  Psychic, The Eye, The Mouth, Cerulean Bell) fall out of the engine's legal-action
  enumeration. Layout in `balatro_gym/env/action_codec.py`.
- **Observation**: `Dict` of dense float32 features (12 hand cards, hand levels,
  deck composition, blind/boss context, shop, pack, stake) plus int64 ID vectors
  (jokers, consumables, shop items, vouchers on offer, boss, the ante's boss, the
  two skip tags, deck) meant for embedding lookup, and multi-hot owned vouchers /
  held tags.
- **Reward**: `+1` per blind cleared, `+5` for winning (default); optional shaped
  mode adds capped, ante-weighted score-progress deltas. Pluggable via
  `RewardConfig`.

## Baseline results (White stake, Red Deck, 100 seeds)

| agent | win rate | mean ante | sec / episode |
|---|---|---|---|
| random | 0% | 1.0 | 0.01 |
| greedy heuristic | 0% | 3.5 | 1.2 |
| strategist | 5% | 4.5 | 1.3 |
| planner (pair build) | 4% | 4.7 | ~10 |

None of the baselines beats the game reliably yet. The planner's diagnostics
point at where the gap is: started with $100 instead of $4 it wins 22% of runs
(mean ante 6.4), and its winning boards share deep spam-hand levels (9-18
planets on the pair) plus one xMult joker. Money and planet access, not play
quality, are the binding constraints for a scripted agent; see
`docs/strategy.md` for the measured levers and the variants that were tried.
`examples/train_maskable_ppo.py` prints every baseline next to a trained policy.

## Architecture

```
src/balatro_gym/
  engine/    # generic machinery: state, scoring pipeline, hand eval, shop, RNG,
             # phase machine + legal_actions(), cardops (create/destroy/pools)
             # (imports nothing from content)
  content/   # every joker/consumable/boss/deck/voucher/tag/stake as a registered
             # effect class — auto-discovered, drop-in files
  env/       # Gymnasium layer: action codec, obs encoder, rewards, 3 envs, baselines
```

Content hooks into the engine through `JokerEffect` / `ConsumableEffect` /
`BossEffect` / `TagEffect` and the `run.mods` dictionary that vouchers, decks and
stakes write into (`edition_rate`, `shop_discount`, `observatory`, `plasma`, ...),
so adding content never needs engine edits.

Fidelity is enforced by golden YAML scoring tests (hand-derived expected values
with derivations in comments), per-joker unit tests (one exact score per joker
plus behavioural tests), consumable/boss/tag/voucher/deck/stake tests, hypothesis
property tests, and random full-game stress sweeps across every deck and stake.
Engine state is cheaply cloneable and JSON-serializable for search (MCTS) work.

## Performance

Random-legal-action rollouts on one core: ~6,000 engine steps/sec, ~1,000 full
env steps/sec (obs encoding included). `python benchmarks/bench_steps.py`.

## Roadmap

Stronger search-based baselines (MCTS on the cloneable engine state), a
hand-size-adaptive action codec, Endless mode and challenge decks.
