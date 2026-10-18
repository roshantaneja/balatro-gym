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
- **xMult hunting**: a board without a multiplicative joker cannot follow
  the requirement curve past ante 4, so from ante 3 the agent rerolls for
  one down to ~$10 instead of sitting on the interest cap, and the reserve
  itself ramps ($0 → $10 → $15 → $25) so the first scoring jokers are
  bought before interest is farmed.
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
| + at-risk trigger at 2x pace, 80 rollouts on boss blinds (current default) | 120 | 20% | 6.2 |

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

Where the current default dies: 56% of its losses never found a
multiplicative joker (the shop offers one in ~15% of two-slot views, so a
run sees about three, half of them conditional), and losers average pair
level 3.6 against 8.6 for winners. Economy knobs are exhausted: the
interest reserve cap, the hunting floor and per-dollar purchase thresholds
all land within noise or below, and adaptive hand commitment picks Pair on
nearly every seed.

The in-round play policy is not exhausted. Replaying every lost blind of a
120-seed run from its first hand (`scripts/replay_losses.py`: snapshot the
engine state, restore it into the env) showed that the
same agent with 40 rollouts instead of 10 clears 13 of 102 lost blinds on
every attempt, and a perfect-information greedy oracle clears 20 — so the
rollout budget was raised (10% → 20% win rate came from this and the shop
changes above). Widening the at-risk trigger further did not add to it.

The next step that the diagnostics justify is not more hand tuning but
learning: use the planner to generate expert-labelled shop decisions and
train the shop/skip policy, or run a proper search (one-ante lookahead over
shop sequences) where the simulation budget allows.
