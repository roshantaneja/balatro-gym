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

## Baseline results (White stake, Red Deck, 100 seeds; planner 240)

| agent | win rate | mean ante | sec / episode |
|---|---|---|---|
| random | 0% | 1.0 | 0.01 |
| greedy heuristic | 0% | 3.5 | 1.2 |
| strategist | 5% | 4.5 | 1.3 |
| planner (pair build) | 20% (47/240) | 6.2 | ~60 |
| lookahead planner (seeds 0-119) | 20% (24/120) | 5.9 | ~480 |

None of the baselines beats the game reliably yet. The planner almost never
loses before ante 4 and reaches ante 8 in a third of runs; what kills it is
antes 5-7 on boards that never found a multiplicative joker (half of its
losses have none) or deep spam-hand levels (losers average pair level 4.6,
winners 10.5). Money is the binding constraint: started with $100 instead
of $4 the same agent wins 57% of runs. See `docs/strategy.md` for the measured levers and every
variant that was tried; `python scripts/eval_agents.py --agent planner
--seeds 120 --workers 4` reproduces the table (per-seed boards, levels,
money and the losing blind are printed for diagnosis) and
`scripts/replay_losses.py` replays every lost blind against a
perfect-information oracle to separate play mistakes from board problems.
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

## Typed decision model (branch `feature/typesafe-model`)

`balatro_gym.typed` turns the game into a "piece of state, named questions,
typed answers with confidence" interface in the style of TypeSafe AI's Jev:
`next_action` (an enum over the legal actions, with readable labels) and
`will_win` (a boolean), each answered with a calibrated probability.

```bash
python scripts/typed/collect.py --seeds 50 --start 0 --out data/typed/part0.npz   # planner-labelled decisions
python scripts/typed/train.py --data "data/typed/*.npz" --out models/typed_v1.pt   # behaviour cloning + win head + temperature scaling
python scripts/typed/eval.py --model models/typed_v1.pt --seeds 60 --workers 4     # win rate and forecast calibration
```

```python
from balatro_gym.typed.agent import TypedModelAgent
agent = TypedModelAgent("models/typed_v1.pt")
answers = agent.ask(env, obs, info["action_mask"])
answers["next_action"].label, answers["next_action"].confidence   # 'buy joker j_duo for $8', 0.71
answers["will_win"].value, answers["will_win"].confidence         # False, 0.88
```

The in-round answer is multiple choice (`typed/candidates.py`): the model is
offered the strongest plays (with and without junk padding, which the planner
uses as a free discard) and the sensible discard shapes, each described by
what a player can see (score against what the blind still needs, hand type
and level, cards scoring, hands and discards left), and scores them. Naming
one of 1,585 play subsets directly was the first version's weak point.

Results (200 planner runs plus one DAgger round of 100 runs, where the
model drives half the steps and the planner labels every state; held-out
runs for validation; play on 60 unseen seeds):

| held-out exact match with the planner | blind select 89% · shop 57% · pack 80% · in-round option 75% (flat 3,519-way answer: 19%) |
|---|---|
| action confidence calibration (ECE) | 0.03 (flat answer) · 0.12 (option answer, underconfident) |
| win forecast | Brier 0.19 at a 0.30 base rate: calibrated, weakly informative |
| model plays every phase | 1.7% wins (1/60), mean ante 3.5 (first version: 0%, 1.0) |
| model answers shop/blind/pack, planner plays rounds | 1.7% wins, mean ante 4.3 (planner alone: 20%, 6.2) |

The option answer fixed the in-round play (mean ante 1.0 to 3.5 playing
alone); the shop answers are now the bottleneck: with the planner playing
the rounds, the model's shop, blind and pack choices still lose half the
planner's wins. Those choices depend on money, joker synergies and the
coming boss in ways 200 runs of cloning do not capture; the planner itself
settles them with rollouts.
