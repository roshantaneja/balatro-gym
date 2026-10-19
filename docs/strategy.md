# Balatro strategy notes (for heuristics and reward design)

Distilled from community consensus (gold-stake guides and thread replies),
adapted to the full content catalog. These rules drive `StrategistAgent` in
`balatro_gym/env/baselines.py`; the tables there (`JOKER_INFO`, `TAG_VALUE`,
`VOUCHER_VALUE`, `HARD_BOSSES`) are the knobs to tune.

## Core doctrine: score with jokers, not hands

1. **The flush fallacy.** Flushes are the best ante-1 hand but a trap as a
   strategy: hard to find (≤1-2 per round), few jokers scale them, and many
   bosses (suit debuffs, The Water, The Arm, The Eye) counter them.
2. **Scaling is king.** The most run-winning jokers gain value every hand or
   round: Supernova, Green Joker, Ride the Bus, Castle, Wee Joker (chips/mult),
   and the xMult scalers — Constellation, Vampire, Hologram, Campfire, Obelisk,
   Lucky Cat, Glass Joker, Yorick, Canio. A winning board ≈ **1 big-chip joker
   + 1-2 +mult jokers + 1-2 xMult jokers (+ a retrigger or copy)**.
3. **One xMult early.** Without a multiplicative joker a board stalls around
   ante 5-6 (requirements 11k-35k). The strategist treats the first xMult
   joker of tier ≥3 as an always-buy and rerolls for one when rich.
4. **High Card / Pair spam.** Joker builds don't care what hand you play, so
   play the cheapest hand every turn: it triggers per-hand scalers 4x/round,
   levels one hand type deeply, and shrugs off most boss effects. Corollaries:
   Half Joker (≤3 cards) and The Duo (contains Pair) become near-unconditional;
   Ride the Bus wants *no scoring face cards*; Green Joker wants *no discards*;
   Obelisk wants you to AVOID your most played hand (anti-synergy with spam).
5. **Tempo vs scaling.** Tempo jokers (flat +mult/+chips, decaying jokers like
   Ice Cream/Popcorn/Gros Michel/Seltzer/Ramen) exist to survive antes 1-3
   while scalers grow. Buy them early, sell them late when slots are needed.
6. **Joker order matters**: +chips/+mult before xMult (left to right);
   retriggers left of xMult; Blueprint directly left of its target; the best
   joker leftmost for Brainstorm. The strategist bubble-sorts the board with
   adjacent swaps in the shop and at blind select.

## Economy doctrine

7. **Interest is a build engine, but scoring comes first.** +$1 per $5 held,
   capped at $5/round ($25; Seed Money/Money Tree raise it). The strategist
   keeps no reserve through ante 3, $5 at antes 4-5, $10 after — a joker or a
   planet for the spam hand is worth more than a dollar of interest early.
8. **Economy jokers are slots.** Golden Joker, Rocket, To the Moon, Egg, Gift
   Card and friends are fine on an empty board; once two scoring jokers are
   down they drop to tier 1 and get sold for upgrades.
9. **Rerolls > speculative packs** when no committed build. Celestial packs
   only pay with a committed hand type (or Telescope); Buffoon packs are
   almost always worth it early; Arcana/Spectral packs once there is spare money.
10. **The Hermit doubles money (max +$20)** — best used at $15+; Temperance
    pays total joker sell value (max $50) — best with a full board.
11. **Vouchers** are permanent but slow: buy Overstock, Grabber, Clearance Sale,
    Paint Brush, Antimatter, Seed Money and Telescope/Observatory when money
    is $10 above the reserve; skip Blank, Magic Trick, Hieroglyph.

## Planet doctrine

12. Planets scale linearly while blind requirements scale super-exponentially:
    strong early, weak late. Level the ONE hand you spam; levels on hands you
    don't play are wasted money. From Celestial packs take the spam-hand planet,
    otherwise the planet of a hand you play ≥25% of the time, otherwise skip.

## Skipping and bosses

13. **Skips**: play Small Blinds for the money and the shop visit; skip only
    for premium tags (Negative, Polychrome, Rare, Holographic, Voucher, Charm,
    Buffoon, Ethereal, Investment, Top-up) and only when the board clearly
    beats the next blind (strategist: estimated hand score x hands ≥ 2.5x the
    boss requirement, never at ante 1).
14. **Boss rerolls** (Director's Cut / Retcon) are for run-killers: The Needle,
    The Wall, The Ox, The Eye, The Arm, The Flint, Crimson Heart, Violet Vessel.
15. **Mr. Bones** is insurance, **Chicot** turns bosses off, **Luchador** is a
    one-shot boss cancel: hold them for the boss you fear.

## Consumable usage (strategist defaults)

- Planets: spam hand immediately; others only to unclog slots.
- Hermit at $15+, Temperance with $8+ of joker sell value, Fool/High Priestess/
  Emperor/Judgement/Wheel of Fortune on sight, Black Hole/The Soul always.
- Enhancement tarots and seals go on paired or spam-rank cards; Strength on
  pairs; suit converters on off-suit high cards; Hanged Man thins the lowest
  singletons early; Death copies the best card over the worst.
- Ankh/Hex only with a single premium joker; Ectoplasm when hand size ≥8;
  Immolate with ≥2 hands left; Wraith when broke.

## Discard doctrine

16. Discard when the best play is off pace (`best < remaining / hands_left`)
    and does not clear the blind outright; Green Joker halves that threshold.
    Keep pairs, 4-flushes and open straights; dump the lowest singletons first.
    Joker-aware: Castle's suit and Mail-In's rank go first, Hit the Road wants
    Jacks, Faceless wants three faces at once, Yorick wants volume.

## Anti-synergies the agent must respect

- Pareidolia + Ride the Bus (all cards are faces → RTB never scales)
- Green Joker + discard-hungry play (each discard = -1 mult)
- Obelisk + spam hand (resets whenever the most-played hand is played)
- Blueprint left of Vampire (the copy eats enhancements without growing)
- Castle wants discards / Green Joker hates them: prefer Green side

## PlannerAgent: measuring instead of guessing

`balatro_gym/env/planner.py` keeps the strategist's play policy but replaces
every board-shaping judgment with a measurement on cloned engine state:

- **Power** = the mean best-play score over K hands dealt from the run's own
  deck, scored with the real pipeline, on a board where scaling jokers have
  been projected a few rounds ahead (`GROWTH` table).
- **Shop**: each joker on offer is valued by the power ratio after adding it
  (or after replacing the joker whose removal costs least when the board is
  full); planets by the ratio after levelling their hand. Anything under +3%
  is not bought; rerolls continue while the projected board cannot beat the
  next boss and money allows.
- **Ordering**: adjacent swaps are kept when they raise measured power.
- **Build commitment**: the agent commits to one spam hand (Pair by default).
  Valuation only counts spam-hand plays, off-build planets are sold, and the
  play policy plays the spam hand whenever it clears the blind or keeps pace,
  falling back to the strongest hand otherwise.
- **Endgame rollouts**: when the best play cannot clear the blind and
  discards remain, "play now" and candidate discards are compared by short
  greedy rollouts on cloned state and the option with the most clears wins.
- **Free card cycling**: a pair (or high card / trips) is played together
  with the junk cards that cannot score, so they are redrawn — the same
  trick players use to see more cards per round. Steel/gold/sealed cards,
  paired ranks, flush draws and Baron/Shoot-the-Moon faces stay in hand;
  Half Joker caps the play at 3 cards, Square Joker pads to exactly 4; the
  padding is skipped under The Tooth (pays per card) and The Serpent.
- **Interest first, then rerolls**: the reserve ramps ($0 → $10 → $15 →
  $25) so the first scoring jokers are bought before interest is farmed,
  and from then on rerolls (the hunt for a multiplicative joker) come out
  of the surplus above the reserve only. Hunting below the reserve was
  measured at every floor from $4 to $10 and always lost: the interest a
  held reserve earns buys more rerolls over a run than the early ones.
- **Graded spending**: a purchase that raises measured power ≥12% may dip
  below the reserve; ≥30% may spend the bankroll when the ante's boss is
  out of reach; ≥60% always. Boss-reroll vouchers (Director's Cut, Retcon)
  are bought near the reserve, Telescope/Observatory on sight.

It costs ~30 s per full run on one core, which is fine for a benchmark floor
and for generating expert trajectories, but not for inner-loop RL.

### Measured results (White stake, Red Deck)

| variant | n | win rate | mean ante |
|---|---|---|---|
| strategist | 100 | 5% | 4.5 |
| planner, greedy play, no commitment | 40 | 0% | 4.6 |
| planner, adaptive commitment | 40 | 0-2.5% | 4.6-4.9 |
| planner, pair commitment (first version) | 100 | 4% | 4.7 |
| planner, pair commitment, $100 start (diagnostic) | 40 | 22% | 6.4 |
| planner, pair commitment, graded spending | 100 | 10% | 5.4 |
| + free card cycling, corrected scaler growth table | 60 | 17% | 6.2 |
| + xMult hunting rerolls, reserve ladder | 120 | 16% | 6.1 |
| same with a $15 interest reserve instead of $25 | 120 | 16% | 6.1 |
| same with buffoon packs valued over 3 sampled packs, Orbital skips | 120 | 16% | 6.3 |
| same, hunting floor $4 instead of $8 | 120 | 13% | 6.1 |
| same, purchases must gain >= 2% / 4% power per dollar | 120 | 15% / 10% | 6.1 / 5.9 |
| + 40 endgame rollouts (was 10), four play options | 120 | 20% | 6.2 |
| + at-risk trigger at 2x pace, 80 rollouts on boss blinds | 120 | 20% | 6.2 |
| + premium skip tags gated on the next blind only | 120 | 18% | 6.2 |
| + never reroll below the interest reserve (current default), seeds 0-119 | 120 | 26% | 6.5 |
| current default, fresh seeds 120-239 | 120 | 15% | 6.1 |
| **current default, all 240 seeds** | 240 | **20%** | 6.3 |
| same, economy jokers valued at 2x and Investment/Economy tags premium | 120 | 22% | 6.4 |
| same plus the full $25 reserve from ante 2 (with / without the above) | 120 | 21% / 13% | 6.4 / 6.2 |
| + joker synergies (Constellation uses every planet, Campfire sells junk, Hologram buys standard packs) | 240 | 18% | 6.2 |
| + cash skip tags (Handy, Garbage, Investment, Economy, Speed) valued at their payout | 240 | 20% | 6.2 |
| + Chaos the Clown and reroll-discount vouchers valued as reroll income | 240 | 18% | 6.2 |
| current default, $100 start (diagnostic) | 60 | 57% | 7.3 |
| shop-aware commitment (`joint=True`: builds re-measured with each offered joker added, 1.25x / 1.6x switching margin), seeds 0-119 | 120 | 17% | 6.4 |

Seed-to-seed variance is large: the same agent scores 26% on seeds 0-119
and 15% on seeds 120-239, so differences under ~8 points at 120 seeds are
noise. Every variant since the 40-rollout change sits in that band.

Income per round from a $4 start (12 runs): blind rewards $4.0, unused
hands $1.8, interest $1.6, jokers and tags $0.8 — about $8 against the $5
per round that the interest cap alone would pay. From $100 the same agent
spends $76 on jokers, $36 on celestial packs, $20 on vouchers and $178 on
rerolls per run; from $4 it spends $51, $11, $4 and $56.

Things that did not move the needle within noise (40 seeds each): an
interest-first reserve ($25 from ante 4), economy-joker valuation (expected
dollars as a power bonus), celestial-pack valuation by expected spam-planet
gain, relaxed skip-for-tag rules, stricter joker replacement thresholds.
Things that did: structural candidate generation (10x faster, same decisions),
paired sampling for valuations (removed noise-driven purchases), rollout
partial credit (stopped single-card plays at ante 1), pair commitment,
measuring "dead anyway" against the blind actually coming next (the earlier
boss-based test drained the bankroll every shop), a reserve that falls away
in antes 7-8, value-per-dollar shop ranking with buffoon packs valued by
sampling the pool, draw-chasing discard options in the endgame rollouts
(no more ante-1 losses), boss-aware requirement estimates with Director's
Cut rerolls of killer bosses, and no commitment under The Eye / The Mouth.

Where the current default dies: half of its losses never found a
multiplicative joker (the shop offers one in ~15% of two-slot views, so a
run sees about three, half of them conditional), and losers average pair
level 4.6 against 10.5 for winners. Money is the bottleneck, not judgment:
started with $100 instead of $4 the same agent wins 57% of runs, earning
$285 per run and spending $178 of it on rerolls (31 per run) against $160
earned and 11 rerolls from a $4 start. No spending policy closes that gap:
hunting below the interest reserve, holding the full reserve from ante 2,
per-dollar purchase thresholds and higher economy-joker valuations all
measure within noise of, or below, the default; adaptive hand commitment
picks Pair on nearly every seed.

The in-round play policy is not exhausted. Replaying every lost blind of a
120-seed run from its first hand (`scripts/replay_losses.py`: snapshot the
engine state, restore it into the env) showed that the
same agent with 40 rollouts instead of 10 clears 13 of 102 lost blinds on
every attempt, and a perfect-information greedy oracle clears 20 — so the
rollout budget was raised (10% → 20% win rate came from this and the shop
changes above). Widening the at-risk trigger further did not add to it.

A state-value model was also tried: 1,357 blind-select states from 77
runs (money, measured power against the ante's boss, pair level, xMult
count, ante) fitted by logistic regression to the run outcome. Power
margin (increasingly with ante), pair level and xMult count carry the
signal; money held at blind select has a slightly negative coefficient,
so such a model cannot supply the money-versus-power exchange rate the
shop would need — the value of money is in what it buys, which the state
does not show. Cash skip tags valued at their actual payout raise income
from $152 to $168 per run without changing the win rate.

Committing to another hand instead of Pair, same 60 seeds, same agent:
Flush 14%, Two Pair 11%, Three of a Kind 17%, High Card 17% (Pair 27% on
those seeds). The builds win different seeds — the union of all five is
26 of 60 — but that is what five nearly independent 15% draws would give,
so the winning build is decided by what the shops offer along the way,
not by anything readable from the seed; an adaptive commitment that
re-measures every shop picks Pair on nearly every seed.

Choosing the build together with the shop's offer (`joint=True`: every
candidate build is measured with each affordable joker on offer added to
the board, and the commitment switches when another build beats the
current one by 1.25x through ante 3 or 1.6x after) does not help either:
17% on seeds 0-119 against the default's 26%. The runs still level Pair
most often (45 of 120), then Two Pair (33) and High Card (16); switching
spreads planets over several hands and the switched-to build rarely gets
the second joker it needs.

## Headroom in the shop decisions (`scripts/replay_shops.py`)

Every shop of a lost planner run (24 seeds, 17 losses, 349 shop
snapshots from ante 2 on) was replayed from the engine snapshot in two
ways: the same planner again, and one forced first action (buy each
offered joker, reroll once, or leave at once) with the planner finishing
the run. About 16 CPU-hours.

| replay | wins |
|---|---|
| same planner from the snapshot (control) | 0 / 349 |
| forced: leave the shop at once | 58 / 349 (17%) |
| forced: reroll once | 39 / 328 (12%) |
| forced: buy an offered joker the planner had not bought | 11 / 107 (10%) |

The control never wins because the restored run and the planner are
deterministic enough to retrace the original losing path. Any deviation
re-rolls the rest of the run, and the win rate of those re-rolls matches
the planner's base rate for a run already at that ante (23/191 from ante
2 shops down to 0/36 from ante 8). Leaving at once does at least as well
as any purchase: the shop choice itself carries no measurable signal
against the variance of what the later shops and draws bring. 10 of the
17 lost runs had some winning alternative, but so would 10 of 17 fresh
runs from those points. This is the shop-side counterpart of the
in-round result above: a different shop policy cannot recover much, and
search over shop actions (the lookahead agent below) sees only this noise.

## LookaheadPlannerAgent: settling shop decisions by rollout

`balatro_gym/env/lookahead.py` keeps the planner for everything except the
contested shop decisions (buy a joker or a pack, reroll, leave). Those are
settled by simulation: each candidate is applied on a cloned run whose RNG
streams are re-seeded (the rollout cannot see what the real reroll or the
real next shop will hold), the rest of the ante is played out by a cheap
inner planner (2 sampled hands, greedy rounds, the same shop rules), and
the candidate with the best mean outcome over a few such futures wins.
A rollout scores 3 for a won run, 0.3 per blind cleared for a lost one,
and otherwise 1 plus 0.3 x log(projected power margin against the next
ante's boss) plus $0.004 per dollar carried. It costs about 8 minutes per
run on one core.

Measured on seeds 0-119, where the plain planner scores 29/120: 24/120,
mean ante 5.9, with more early deaths (it keeps money in antes 1-3 that
the greedy planner spends on its first jokers). A terminal value taken
from the fitted win-probability model instead (power margin, pair level,
xMult jokers, ante; no money term) was stopped at 1 win in 16 seeds: with
money worth nothing to it, it spends on marginal power at every shop.
A third variant applied the lookahead only in the shop before each boss
(antes 2-6) with a two-ante horizon and 4-6 sampled futures per candidate
(`horizon=2, boss_shops_only=True`): 6 wins in 57 seeds before it was
stopped. The rollouts share random numbers across candidates, but the
inner planner that plays them (2 sampled hands, greedy rounds) is weaker
and noisier than the real agent, so the comparison between candidates
is dominated by rollout variance and the greedy valuation, which has no
such variance, decides better. Search over shops would need rollouts
that are both cheap and low-variance, which this engine's per-round cost
does not allow at useful sample sizes.

## Learning from scratch on this hardware

`examples/train_maskable_ppo.py` (MaskablePPO, MultiInputPolicy, 1024-step
rollouts, batch 256, entropy 0.01) was run for 2,000,000 steps on 4 CPU
cores (about 5 hours each, no GPU):

| env | policy reward (50 episodes) | same-env baselines (20 seeds) |
|---|---|---|
| `Balatro-v0` (whole game) | 1.44 +/- 2.73 (about 1.4 blinds per run) | heuristic mean ante 3.75; strategist 10% / 4.3; planner 30% / 6.15 |
| `BalatroScriptedShop-v0` (policy plays hands, strategist runs the shop) | 4.48 +/- 4.05 (about 4.5 blinds per run) | heuristic 10% / 5.25; strategist 10% / 4.3 |

From scratch, two million steps are enough to learn to clear the first
blind or two of a run and no more; the whole-game policy is below the
greedy heuristic. Reaching the planner, let alone a consistent winner,
needs orders of magnitude more environment steps (the reward is one
blind at a time over runs of 150-500 steps with a 3,519-way action
space), a GPU, vectorised environments, and either reward shaping that
the env already offers (`reward_mode="shaped"`) or imitation of planner
trajectories to start from.

The next step that the diagnostics justify is not more hand tuning but
learning: use the planner to generate expert-labelled shop decisions and
train the shop/skip policy, or run a proper search (one-ante lookahead over
shop sequences) where the simulation budget allows.
